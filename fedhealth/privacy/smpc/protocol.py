"""Secure computation of  SUM_k ( f_1[k] * ... * O_j[k] ) + noise_j  over secret-shared vectors.

THE QUERY AS ARITHMETIC
    After PSI every hospital h has the same ordered list of m linked patients and can build, locally, a
    vector over it:   ind_h[k] = 1  iff linked patient k satisfies hospital h's predicate.
    A patient counts iff EVERY participating hospital says yes, so
            COUNT = SUM_k  ind_A[k] * ind_B[k] * ind_C[k]
    SUM of a value held by hospital o = SUM_k (ind_o[k] * value[k]) * prod(other indicators), and a
    HISTOGRAM is one such sum per bin with  ind_o[k] * [category[k] == bin]  .

    One hospital, the OWNER `o`, holds the "output vectors" O_1..O_t (t = 1 for count, 2 for sum/avg, one per
    bin for a histogram). Every other participating hospital contributes one 0/1 indicator vector (a "factor").

ROLES (assign_roles)  - three hospitals take part in every multi-hospital query
    inputs      the participating hospitals: they hold the data and secret-share it
    computers   P0, P1: hold 2-out-of-2 additive shares of everything and run the Beaver multiplications
    dealer      a third hospital that is NOT a computer: it generates the Beaver triples and never sees any value
                that was opened with them. For a 3-hospital query it is the third data holder; for a 2-hospital query
                it is the hospital that is not taking part (it then learns only the public shape: m and t).
    The coordinator holds no protocol secret at all: it neither deals triples nor sees an un-noised value.

THE PROTOCOL
    0. NOISE     every involved hospital (P0, P1, dealer) samples one DP noise share per output cell
                 (privacy/dp/distributed.py) - its own randomness, never revealed.
    1. SHARE     every input hospital splits each of its vectors into 2 additive shares, one for P0 and one for P1;
                 every hospital splits its noise vector the same way. Each share alone is uniformly random.
    2. DEAL      the dealer draws fresh triples (x, y, z = x*y) and gives P0 a 32-byte SEED from which P0 expands
                 its shares (x0, y0, z0), and P1 the explicit (x1, y1, z1 = x*y - z0).
    3. MULTIPLY  P = f_1 * f_2 * ...: one Beaver layer of m triples per extra factor, opened between P0 and P1 only.
    4. OUTPUT    out_j = SUM_k P[k]*O_j[k]: one more Beaver layer of t*m triples; each computer sums its product shares
                 and ADDS ITS SHARES OF EVERY HOSPITAL'S NOISE. The coordinator adds the two output shares and gets
                 exact + noise - and nothing else: each of the two shares is uniformly random on its own.

    n = 1 needs no protocol: the lone hospital computes its aggregate and adds the full noise itself.

TRUST (see SECURITY_THEOREMS.md): semi-honest parties; the coordinator may collude with any ONE hospital.
Two colluding hospitals can reconstruct the shared inputs (the dealer knows the triple masks, a computer sees the
openings) - the standard honest-majority assumption of three-party MPC.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from . import beaver
from .field import P, rand_vector, to_signed, vec_add, vec_mul, vec_sub, vec_sum
from .sharing import reconstruct_scalar, share_vector

SEED_BYTES = 32


# ----------------------------------------------------------------------------- the public "program"
@dataclass(frozen=True)
class Program:
    """The SHAPE of a computation. It is public: it depends only on the number of input hospitals, the number of
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


@dataclass(frozen=True)
class Roles:
    inputs: tuple[str, ...]          # participating hospitals, in ring order
    owner: str                       # holds the output vectors
    computers: tuple[str, str]       # P0, P1
    dealer: str                      # makes the triples; never a computer

    @property
    def involved(self) -> tuple[str, ...]:
        """Every hospital in the computation - also exactly the noise contributors."""
        return self.inputs + (() if self.dealer in self.inputs else (self.dealer,))

    @property
    def factors(self) -> list[str]:
        return [p for p in self.inputs if p != self.owner]


def assign_roles(inputs: Iterable[str], owner: str, federation: Iterable[str]) -> Roles:
    """Deterministic, public role assignment for a multi-hospital query (needs >= 3 hospitals in the federation)."""
    inputs, federation = tuple(inputs), tuple(federation)
    if len(set(inputs)) != len(inputs):
        raise ValueError("a hospital is listed more than once")
    if len(inputs) < 2:
        raise ValueError("a single hospital needs no protocol")
    if owner not in inputs:
        raise ValueError("the owner must be an input hospital")
    if len(inputs) == 2:
        spare = [h for h in federation if h not in inputs]
        if not spare:
            raise ValueError("a two-hospital query needs a third hospital as triple dealer")
        return Roles(inputs, owner, (inputs[0], inputs[1]), spare[0])
    return Roles(inputs, owner, (inputs[0], inputs[1]), inputs[2])


def n_outputs(agg: str, n_categories: int = 0) -> int:
    """SUM and AVG both reveal [cohort count, total]: the (noisy) count is needed for the k-threshold check."""
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
    """Revealed output cells -> the query answer (exact only on the insecure evaluation route)."""
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
def expand_seed(seed: bytes, count: int, label: bytes) -> list[int]:
    """PRG: `count` uniform field elements from a 32-byte seed (SHAKE-256, domain-separated by `label`).
    Rejection of the single 61-bit pattern that equals p keeps the output exactly uniform on Z_p."""
    out: list[int] = []
    block = 0
    while len(out) < count:
        need = count - len(out) + 8
        raw = hashlib.shake_256(b"fedhealth-triples|" + label + b"|" + block.to_bytes(4, "big") + b"|" + seed).digest(8 * need)
        vals = np.frombuffer(raw, dtype=">u8") & np.uint64(P)
        out.extend(int(v) for v in vals[vals != np.uint64(P)][: count - len(out)])
        block += 1
    return out


def expand_triples(seed: bytes, count: int) -> tuple[list[int], list[int], list[int]]:
    """P0's triple shares (x0, y0, z0), re-derived from the dealer's seed."""
    return expand_seed(seed, count, b"x"), expand_seed(seed, count, b"y"), expand_seed(seed, count, b"z")


def deal(count: int) -> tuple[bytes, tuple[list[int], list[int], list[int]]]:
    """DEALER: `count` fresh Beaver triples, shared 2-out-of-2. Returns (seed for P0, (x1, y1, z1) for P1).
    x = x0 + x1 and y = y0 + y1 are uniform; z1 = x*y - z0, so z0 + z1 = x*y."""
    seed = secrets.token_bytes(SEED_BYTES)
    x0, y0, z0 = expand_triples(seed, count)
    x1, y1 = rand_vector(count), rand_vector(count)
    z = vec_mul(vec_add(x0, x1), vec_add(y0, y1))
    return seed, (x1, y1, vec_sub(z, z0))


# ----------------------------------------------------------------------------- one hospital's state machine
@dataclass
class MpcParty:
    name: str
    roles: Roles | None                # None for a single-hospital query
    program: Program
    own_vectors: list[list[int]] = field(default_factory=list)
    noise: list[int] | None = None     # MY noise share per output cell (None = exact computation, evaluation only)
    held: dict[str, list[list[int]]] = field(default_factory=dict)         # sender -> MY share of each of its vectors
    noise_held: dict[str, list[int]] = field(default_factory=dict)         # contributor -> MY share of its noise
    triples: tuple[list[int], list[int], list[int]] | None = None
    opened: dict[int, dict[str, tuple[list[int], list[int]]]] = field(default_factory=dict)  # layer -> sender -> (d, e)
    finished: set[int] = field(default_factory=set)
    running: list[int] | None = None   # my share of the running product P
    out_shares: list[int] | None = None

    # ---- bookkeeping ------------------------------------------------------------------------------
    @property
    def is_computer(self) -> bool:
        return self.roles is not None and self.name in self.roles.computers

    @property
    def is_input(self) -> bool:
        return self.roles is None or self.name in self.roles.inputs

    @property
    def index(self) -> int:
        return self.roles.computers.index(self.name)

    @property
    def peer(self) -> str:
        return self.roles.computers[1 - self.index]

    def expected_vectors_from(self, sender: str) -> int:
        return self.program.t_outputs if sender == self.roles.owner else 1

    # ---- step 1: share inputs and noise -------------------------------------------------------------
    def _split(self, vectors: list[list[int]]) -> dict[str, list[list[int]]]:
        per_vector = [share_vector(v, 2) for v in vectors]                     # [vector][computer]
        return {c: [pv[j] for pv in per_vector] for j, c in enumerate(self.roles.computers)}

    def split_inputs(self) -> dict[str, list[list[int]]]:
        """2-out-of-2 shares of each own vector, one per computer. A computer keeps its own share; returns
        {other computer(s): shares}."""
        if not self.is_input:
            return {}
        by_computer = self._split(self.own_vectors)
        if self.is_computer:
            self.held[self.name] = by_computer.pop(self.name)
        return by_computer

    def split_noise(self) -> dict[str, list[int]]:
        if self.noise is None:
            return {}
        by_computer = {c: s[0] for c, s in self._split([self.noise]).items()}
        if self.is_computer:
            self.noise_held[self.name] = by_computer.pop(self.name)
        return by_computer

    def receive_input_shares(self, sender: str, vectors: list[list[int]]) -> None:
        if not self.is_computer:
            raise ValueError("only a computing hospital receives input shares")
        if sender not in self.roles.inputs or sender == self.name or sender in self.held:
            raise ValueError(f"unexpected input shares from {sender!r}")
        if len(vectors) != self.expected_vectors_from(sender) or any(len(v) != self.program.m for v in vectors):
            raise ValueError("input shares have the wrong shape")
        self.held[sender] = vectors

    def receive_noise_share(self, sender: str, share: list[int]) -> None:
        if not self.is_computer or self.noise is None:
            raise ValueError("this hospital expects no noise shares")
        if sender not in self.roles.involved or sender == self.name or sender in self.noise_held:
            raise ValueError(f"unexpected noise share from {sender!r}")
        if len(share) != self.program.t_outputs:
            raise ValueError("noise share has the wrong shape")
        self.noise_held[sender] = share

    @property
    def inputs_complete(self) -> bool:
        noise_ok = self.noise is None or set(self.noise_held) == set(self.roles.involved)
        return set(self.held) == set(self.roles.inputs) and noise_ok

    # ---- step 2: triples ------------------------------------------------------------------------------
    def set_triples(self, x: list[int], y: list[int], z: list[int]) -> None:
        if not self.is_computer:
            raise ValueError("only a computing hospital uses triples")
        if self.triples is not None:
            raise ValueError("triples already received")
        if not (len(x) == len(y) == len(z) == self.program.total_triples):
            raise ValueError("wrong number of triples")
        self.triples = (x, y, z)

    def set_triples_from_seed(self, seed: bytes) -> None:
        if len(seed) != SEED_BYTES:
            raise ValueError("bad seed")
        self.set_triples(*expand_triples(seed, self.program.total_triples))

    # ---- step 3/4: Beaver layers ----------------------------------------------------------------------
    def _operands(self, layer: int) -> tuple[list[int], list[int]]:
        prog, factors = self.program, self.roles.factors
        if layer < prog.n_product_layers:                              # P <- P * next factor
            a = self.held[factors[0]][0] if layer == 0 else self.running
            return a, self.held[factors[layer + 1]][0]
        p_share = self.held[factors[0]][0] if prog.n_factors == 1 else self.running
        out_vecs = self.held[self.roles.owner]
        return p_share * prog.t_outputs, [v for vec in out_vecs for v in vec]   # P repeated t times  vs  O_1 || ... || O_t

    def _slice(self, layer: int) -> tuple[list[int], list[int], list[int]]:
        off, size = self.program.offset(layer), self.program.layer_sizes[layer]
        return tuple(v[off:off + size] for v in self.triples)           # type: ignore[return-value]

    def open_layer(self, layer: int) -> tuple[list[int], list[int]]:
        """Compute MY shares of d = a - x and e = b - y for this layer (sent to the other computer only)."""
        if not self.is_computer:
            raise RuntimeError("only a computing hospital opens layers")
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
        if not self.is_computer or sender != self.peer:
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
        """Open d and e (add both computers' shares) and compute MY share of the product."""
        slot = self.opened.get(layer, {})
        if not self.is_computer or layer in self.finished or set(slot) != set(self.roles.computers):
            raise RuntimeError("layer not ready or already finished")
        d = beaver.open_values([slot[p][0] for p in self.roles.computers])
        e = beaver.open_values([slot[p][1] for p in self.roles.computers])
        x, y, z = self._slice(layer)
        prod = beaver.product_share(d, e, x, y, z, adds_constant_term=(self.index == 0))
        self.finished.add(layer)
        if layer < self.program.n_product_layers:
            self.running = prod
        else:                                                           # final layer: sum each output block locally
            m = self.program.m
            out = [vec_sum(prod[j * m:(j + 1) * m]) for j in range(self.program.t_outputs)]
            for share in self.noise_held.values():                     # + my shares of everyone's noise
                out = vec_add(out, share)
            self.out_shares = out
        self.opened[layer] = {}                                         # openings are no longer needed

    # ---- n = 1 -----------------------------------------------------------------------------------------
    def local_outputs(self) -> list[int]:
        """Single-hospital query: no sharing needed - the hospital's own sums plus its (full) noise share."""
        assert self.roles is None and self.program.n_parties == 1
        out = [vec_sum(v) for v in self.own_vectors]
        self.out_shares = vec_add(out, [n % P for n in self.noise]) if self.noise is not None else out
        return self.out_shares

    def output(self) -> list[int]:
        if self.out_shares is None:
            raise RuntimeError("computation not finished")
        return self.out_shares


# ----------------------------------------------------------------------------- the whole thing in one process
def run_local(vectors: dict[str, list[list[int]]], owner: str, m: int,
              noise: dict[str, list[int]] | None = None, federation: list[str] | None = None) -> list[int]:
    """Reference implementation: run the full protocol among in-process parties and return the REVEALED
    output cells (field elements). `vectors[name]` = that party's input vectors (owner: t vectors; others: 1).
    `noise[name]` = that hospital's noise share (for every involved hospital, or None for an exact run).
    The hospital services execute exactly these steps, spread over HTTP."""
    inputs = list(vectors)
    t = len(vectors[owner])
    prog = Program(len(inputs), t, m)
    if len(inputs) == 1:
        lone = MpcParty(owner, None, prog, own_vectors=vectors[owner], noise=(noise or {}).get(owner))
        return lone.local_outputs()
    roles = assign_roles(inputs, owner, federation or inputs + ["D"])
    state = {p: MpcParty(p, roles, prog, own_vectors=vectors.get(p, []),
                         noise=None if noise is None else [n % P for n in noise[p]]) for p in roles.involved}
    for p in roles.involved:
        for c, shares in state[p].split_inputs().items():
            state[c].receive_input_shares(p, shares)
        for c, share in state[p].split_noise().items():
            state[c].receive_noise_share(p, share)
    seed, explicit = deal(prog.total_triples)
    state[roles.computers[0]].set_triples_from_seed(seed)
    state[roles.computers[1]].set_triples(*explicit)
    for layer in range(len(prog.layer_sizes)):
        opened = {p: state[p].open_layer(layer) for p in roles.computers}
        for p in roles.computers:
            q = state[p].peer
            state[p].receive_open(layer, q, *opened[q])
        for p in roles.computers:
            state[p].finish_layer(layer)
    return [reconstruct_scalar([state[p].output()[j] for p in roles.computers]) for j in range(t)]
