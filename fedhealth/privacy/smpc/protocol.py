"""Secure computation of  SUM_k ( f_1[k] * f_2[k] * ... * O_j[k] )  over secret-shared vectors.

THE QUERY AS ARITHMETIC
    After PSI every hospital h has the same ordered list of m linked patients and can build, locally, a
    vector over it:   ind_h[k] = 1  iff linked patient k satisfies hospital h's predicate.
    A patient counts iff EVERY participating hospital says yes, so
            COUNT = SUM_k  ind_A[k] * ind_B[k] * ind_C[k]
    SUM of a value held by hospital o = SUM_k (ind_o[k] * value[k]) * prod(other indicators), and a
    HISTOGRAM is one such sum per bin with  ind_o[k] * [category[k] == bin]  .

    One hospital, the OWNER `o`, holds the "output vectors" O_1..O_t (t = 1 for count/sum, 2 for avg, one per
    bin for a histogram). Every other hospital contributes one 0/1 indicator vector (a "factor").

THE PROTOCOL (n computing hospitals, k = n-1 factors)
    1. SHARE     each hospital splits every one of its vectors into n additive shares and sends share j to
                 hospital j. Nobody ever sees another hospital's vector; any n-1 shares look random.
    2. MULTIPLY  P = f_1 * f_2 * ... * f_k, one Beaver multiplication per extra factor (k-1 layers of m triples)
    3. OUTPUT    out_j = SUM_k P[k] * O_j[k]: one more Beaver layer of t*m triples, then each hospital sums its
                 product shares locally. Each hospital ends with ONE additive share of every output cell.
    4. REVEAL    the hospitals send those shares to the coordinator, which adds them mod p. The coordinator
                 learns only the final aggregate (which DP then noises) - no indicator, no partial count.

    n = 1 needs no protocol: the lone hospital's aggregate is already the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from . import beaver
from .field import P, to_signed, vec_sum
from .sharing import reconstruct_scalar, share_vector


# ----------------------------------------------------------------------------- the public "program"
@dataclass(frozen=True)
class Program:
    """The SHAPE of a computation. It is public: it depends only on the number of hospitals, the number of
    output cells and the linked-cohort size - never on any patient's data."""
    n_parties: int
    t_outputs: int
    m: int

    @property
    def n_factors(self) -> int:
        return self.n_parties - 1

    @property
    def n_product_layers(self) -> int:
        return max(self.n_factors - 1, 0)

    @property
    def layer_sizes(self) -> list[int]:
        sizes = [self.m] * self.n_product_layers                     # f_1*f_2, (f_1*f_2)*f_3, ...
        if self.n_factors >= 1:
            sizes.append(self.t_outputs * self.m)                    # P * O_j for every output cell j
        return sizes

    @property
    def total_triples(self) -> int:
        return sum(self.layer_sizes)

    def offset(self, layer: int) -> int:
        return sum(self.layer_sizes[:layer])


def n_outputs(agg: str, n_categories: int = 0) -> int:
    """SUM and AVG both reveal [cohort count, total]: the count is needed for the k-threshold check."""
    return {"count": 1, "sum": 2, "avg": 2}.get(agg, n_categories)


def build_input_vectors(matches: Iterable[tuple[int, int | None, str | None]], m: int, *, is_owner: bool,
                        agg: str, categories: tuple[str, ...] | list[str] = ()) -> list[list[int]]:
    """Hospital-side: turn local predicate matches [(linked_index, value, category)] into the vectors that
    enter the protocol. Values may be negative; they are mapped into the field."""
    ind, val, cat = [0] * m, [0] * m, [None] * m
    for idx, value, category in matches:
        ind[idx], val[idx], cat[idx] = 1, (value or 0) % P, category
    if not is_owner or agg == "count":
        return [ind]
    if agg in ("sum", "avg"):
        return [ind, val]                                             # -> [cohort count, total]; val is 0 where no match
    return [[1 if cat[k] == c else 0 for k in range(m)] for c in categories]   # histogram: one vector per bin


def assemble_result(agg: str, outputs: list[int], categories: list[str] | None = None):
    """Coordinator-side: reconstructed output cells -> the query answer (exact; DP comes later)."""
    vals = [to_signed(o) for o in outputs]
    if agg == "count":
        return vals[0]
    if agg == "sum":
        return vals[1]                                                # vals = [count, total]
    if agg == "avg":
        count, total = vals
        return (total / count) if count else None
    return dict(zip(categories, vals, strict=True))


# ----------------------------------------------------------------------------- the dealer (offline phase)
def deal(program: Program, parties: list[str]) -> dict[str, tuple[list[int], list[int], list[int]]]:
    """DEALER (the coordinator): fresh Beaver triples for the whole program, shared among the parties."""
    per_party = beaver.deal_triples(program.total_triples, len(parties))
    return dict(zip(parties, per_party, strict=True))


# ----------------------------------------------------------------------------- one party's state machine
@dataclass
class MpcParty:
    name: str
    parties: list[str]                 # fixed order; position = index of this party's share
    owner: str                         # holds the output vectors
    program: Program
    own_vectors: list[list[int]] = field(default_factory=list)
    held: dict[str, list[list[int]]] = field(default_factory=dict)         # sender -> MY share of each of its vectors
    triples: tuple[list[int], list[int], list[int]] | None = None
    opened: dict[int, dict[str, tuple[list[int], list[int]]]] = field(default_factory=dict)  # layer -> sender -> (d, e)
    finished: set[int] = field(default_factory=set)
    running: list[int] | None = None   # my share of the running product P
    out_shares: list[int] | None = None

    # ---- bookkeeping ------------------------------------------------------------------------------
    @property
    def index(self) -> int:
        return self.parties.index(self.name)

    @property
    def factors(self) -> list[str]:
        return [p for p in self.parties if p != self.owner]

    def expected_vectors_from(self, sender: str) -> int:
        return self.program.t_outputs if sender == self.owner else 1

    # ---- step 1: share inputs -----------------------------------------------------------------------
    def split_inputs(self) -> dict[str, list[list[int]]]:
        """Split each own vector into n shares. Keeps my own share, returns {other party: its shares}."""
        n = len(self.parties)
        per_vector = [share_vector(v, n) for v in self.own_vectors]            # [vector][party]
        by_party = {p: [per_vector[v][j] for v in range(len(self.own_vectors))] for j, p in enumerate(self.parties)}
        self.held[self.name] = by_party[self.name]
        return {p: s for p, s in by_party.items() if p != self.name}

    def receive_input_shares(self, sender: str, vectors: list[list[int]]) -> None:
        if sender not in self.parties or sender == self.name or sender in self.held:
            raise ValueError(f"unexpected input shares from {sender!r}")
        if len(vectors) != self.expected_vectors_from(sender) or any(len(v) != self.program.m for v in vectors):
            raise ValueError("input shares have the wrong shape")
        self.held[sender] = vectors

    @property
    def inputs_complete(self) -> bool:
        return set(self.held) == set(self.parties)

    # ---- step 2/3: Beaver layers ----------------------------------------------------------------------
    def set_triples(self, x: list[int], y: list[int], z: list[int]) -> None:
        if not (len(x) == len(y) == len(z) == self.program.total_triples):
            raise ValueError("wrong number of triples")
        self.triples = (x, y, z)

    def _operands(self, layer: int) -> tuple[list[int], list[int]]:
        prog, factors = self.program, self.factors
        if layer < prog.n_product_layers:                              # P <- P * next factor
            a = self.held[factors[0]][0] if layer == 0 else self.running
            return a, self.held[factors[layer + 1]][0]
        p_share = self.held[factors[0]][0] if prog.n_factors == 1 else self.running
        out_vecs = self.held[self.owner]
        return p_share * prog.t_outputs, [v for vec in out_vecs for v in vec]   # P repeated t times  vs  O_1 || ... || O_t

    def _slice(self, layer: int) -> tuple[list[int], list[int], list[int]]:
        off, size = self.program.offset(layer), self.program.layer_sizes[layer]
        return tuple(v[off:off + size] for v in self.triples)           # type: ignore[return-value]

    def open_layer(self, layer: int) -> tuple[list[int], list[int]]:
        """Compute MY shares of d = a - x and e = b - y for this layer (to be sent to every other party)."""
        if not self.inputs_complete or self.triples is None:
            raise RuntimeError("inputs or triples missing")
        if layer >= len(self.program.layer_sizes):
            raise RuntimeError("no such layer")
        if layer > 0 and (layer - 1) not in self.finished:
            raise RuntimeError("previous layer not finished")
        if layer in self.finished or self.name in self.opened.get(layer, {}):
            raise RuntimeError("layer already opened (a triple may be used only once)")
        a, b = self._operands(layer)
        x, y, _ = self._slice(layer)
        d, e = beaver.mask(a, b, x, y)
        self.opened.setdefault(layer, {})[self.name] = (d, e)
        return d, e

    def receive_open(self, layer: int, sender: str, d: list[int], e: list[int]) -> None:
        if sender not in self.parties or sender == self.name:
            raise ValueError("unexpected sender")
        if layer >= len(self.program.layer_sizes):
            raise ValueError("no such layer")
        size = self.program.layer_sizes[layer]
        if len(d) != size or len(e) != size:
            raise ValueError("bad opening shape")
        if layer in self.finished:
            raise ValueError("layer already finished")
        slot = self.opened.setdefault(layer, {})
        if sender in slot:
            raise ValueError("duplicate opening")
        slot[sender] = (d, e)

    def finish_layer(self, layer: int) -> None:
        """Open d and e (sum everyone's shares) and compute MY share of the product."""
        slot = self.opened.get(layer, {})
        if layer in self.finished or set(slot) != set(self.parties):
            raise RuntimeError("layer not ready or already finished")
        d = beaver.open_values([slot[p][0] for p in self.parties])
        e = beaver.open_values([slot[p][1] for p in self.parties])
        x, y, z = self._slice(layer)
        prod = beaver.product_share(d, e, x, y, z, adds_constant_term=(self.index == 0))
        self.finished.add(layer)
        if layer < self.program.n_product_layers:
            self.running = prod
        else:                                                           # final layer: sum each output block locally
            m = self.program.m
            self.out_shares = [vec_sum(prod[j * m:(j + 1) * m]) for j in range(self.program.t_outputs)]
        self.opened[layer] = {}                                         # openings are no longer needed

    # ---- n = 1 -----------------------------------------------------------------------------------------
    def local_outputs(self) -> list[int]:
        """Single-hospital query: no sharing needed, the hospital's own vectors already sum to the answer."""
        assert self.program.n_parties == 1
        self.out_shares = [vec_sum(v) for v in self.own_vectors]
        return self.out_shares

    def output(self) -> list[int]:
        if self.out_shares is None:
            raise RuntimeError("computation not finished")
        return self.out_shares


# ----------------------------------------------------------------------------- the whole thing in one process
def run_local(vectors: dict[str, list[list[int]]], owner: str, m: int) -> list[int]:
    """Reference implementation: run the full protocol among in-process parties and return the REVEALED
    output cells (field elements). `vectors[name]` = that party's input vectors (owner: t vectors; others: 1).
    The hospital services execute exactly these steps, spread over HTTP."""
    parties = list(vectors)
    t = len(vectors[owner])
    prog = Program(len(parties), t, m)
    state = {p: MpcParty(p, parties, owner, prog, own_vectors=vectors[p]) for p in parties}
    if len(parties) == 1:
        return state[parties[0]].local_outputs()
    for p in parties:
        for other, shares in state[p].split_inputs().items():
            state[other].receive_input_shares(p, shares)
    for p, (x, y, z) in deal(prog, parties).items():
        state[p].set_triples(x, y, z)
    for layer in range(len(prog.layer_sizes)):
        opened = {p: state[p].open_layer(layer) for p in parties}
        for p in parties:
            for q, (d, e) in opened.items():
                if q != p:
                    state[p].receive_open(layer, q, d, e)
        for p in parties:
            state[p].finish_layer(layer)
    return [reconstruct_scalar([state[p].output()[j] for p in parties]) for j in range(t)]
