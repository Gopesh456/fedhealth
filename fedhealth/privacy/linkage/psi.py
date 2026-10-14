"""Private set intersection by commutative encryption (Diffie-Hellman style, multi-party ring).

THE PROBLEM
    Hospitals A, B, C each hold a list of patient identifiers. They want to learn which patients they
    share - WITHOUT revealing the patients the others don't have, and without ever exchanging an
    identifier or a plain hash of one.

THE IDEA (commutative encryption)
    Every party i picks a secret exponent k_i. "Encrypting" x means  E_i(x) = x^(k_i) mod p.
    Because (x^a)^b = (x^b)^a, encrypting under all keys gives the SAME result in ANY order:
        E_C(E_B(E_A(x))) == E_A(E_C(E_B(x)))      (call this the *fully encrypted* value of x)
    So two parties hold the same patient  <=>  the fully encrypted values of that patient are equal.
    Nobody can compute a fully encrypted value alone (it needs every key), and without the keys the
    values look random (Decisional Diffie-Hellman assumption).

THE RING (n parties, ring order fixed, e.g. A -> B -> C -> A)
    round 1   each party hashes its ids into the group, encrypts with ITS key, sends the list on.
    round r   each party encrypts the list it just received with its key and passes it on.
    round n   after n hops every list has been encrypted by every party and arrives back at its
              OWNER, in the owner's original order - so the owner can map results back to its patients.
    publish   each owner sends only 128-bit TAGS (hashes) of its fully encrypted list to the others.
    finalize  every party intersects the tag sets. The intersection, sorted, is a *common linked index*:
              identical at every party, and patient p has index i at every hospital that holds p.

WHAT EACH PARTY LEARNS
    * its own patients that are in the intersection (this is the PSI output),
    * the size of every other party's list,
    * nothing about other parties' patients outside the intersection.
    The coordinator never sees any of this traffic.

WHAT THIS DOES NOT PROTECT (see THREAT_MODEL.md)
    * Honest-but-curious parties only. A malicious party could cheat (e.g. submit a made-up list to probe
      for a specific patient). Membership of the intersection is revealed to its members by design.
    * List sizes leak unless padded with dummies; the order of `ids` leaks to the last ring hop unless shuffled
      (the hospital service does both). Colluding parties can combine what they see.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .group import Group, get_group, pack_ints, unpack_ints

__all__ = ["PsiParty", "psi_intersection_local"]


@dataclass
class PsiParty:
    """One party's state for one linkage session. Used by the hospital service (over HTTP) and by
    `psi_intersection_local` (all parties in one process, for tests and benchmarks)."""
    name: str
    ring: list[str]                      # e.g. ["A", "B", "C"] (only the parties taking part)
    ids: list[str]                       # this party's identifiers (any order, unique)
    group: Group
    key: int | None = None               # secret exponent, destroyed by finalize()
    elements: list[int] = field(default_factory=list)       # H(id) for each id
    final_elements: list[int] | None = None                 # own list, encrypted by every party
    tags: list[bytes] | None = None
    mapping: dict[str, int] | None = None
    linked_digest: bytes | None = None

    def __post_init__(self) -> None:
        self.key = self.group.new_key()
        self.elements = [self.group.hash_to_group(i.encode()) for i in self.ids]

    # ---- ring bookkeeping ------------------------------------------------------------------------
    @property
    def index(self) -> int:
        return self.ring.index(self.name)

    @property
    def next_party(self) -> str:
        return self.ring[(self.index + 1) % len(self.ring)]

    @property
    def prev_party(self) -> str:
        return self.ring[(self.index - 1) % len(self.ring)]

    # ---- rounds ------------------------------------------------------------------------------------
    def encrypt_list(self, elements: list[int]) -> list[int]:
        """One ring hop: raise every element to this party's secret exponent."""
        return [self.group.encrypt(e, self.key) for e in elements]

    def round_one_message(self) -> list[int]:
        return self.encrypt_list(self.elements)

    def publish(self) -> list[bytes]:
        """Tags of the fully encrypted list, sorted. Sorting alone does NOT hide the order of this party's patients:
        the previous party in the ring computed the fully encrypted list in the order it was sent and can tag it
        itself. The hospital service therefore shuffles `ids` before creating the party (routes_psi.init)."""
        assert self.final_elements is not None, "own list has not come back round the ring yet"
        self.tags = [self.group.tag(e) for e in self.final_elements]
        return sorted(self.tags)

    def finalize(self, peer_tags: dict[str, set[bytes]]) -> int:
        """Intersect everyone's tags, build the common index, map own patients to it, destroy the key."""
        assert self.tags is not None
        common = set(self.tags)
        for other in self.ring:
            if other != self.name:
                common &= peer_tags[other]
        linked = sorted(common)                        # identical at every party
        position = {t: i for i, t in enumerate(linked)}
        self.mapping = {pid: position[t] for pid, t in zip(self.ids, self.tags) if t in position}
        # Agreement check for the coordinator: a hash of EVERY party's published tag set, in ring order. Every party holds
        # all of them, so the digest is identical everywhere, and it determines the linked index. (A hash of the linked
        # index alone would be the constant sha256(b"") whenever the intersection is empty, telling the coordinator so.)
        h = hashlib.sha256(b"fedhealth/psi-linked/v1")
        for name in self.ring:
            tags = sorted(self.tags) if name == self.name else sorted(peer_tags[name])
            h.update(len(tags).to_bytes(4, "big") + b"".join(tags))
        self.linked_digest = h.digest()[:8]
        # forward secrecy of the session: the key and all intermediate lists are no longer needed
        self.key, self.elements, self.final_elements = None, [], None
        return len(linked)


def psi_intersection_local(sets: dict[str, list[str]], group_name: str = "modp2048",
                           transcript: list | None = None) -> dict[str, dict[str, int]]:
    """Run the whole protocol in one process (no network). Returns {party: {id: linked_index}}.

    This is the reference implementation: the hospital HTTP routes execute exactly the same
    `PsiParty` steps, just spread over services. If `transcript` is a list, every message is appended as
    (sender, receiver, kind, payload) - i.e. exactly what each party gets to see."""
    group = get_group(group_name)
    ring = list(sets)
    n = len(ring)
    parties = {name: PsiParty(name, ring, ids, group) for name, ids in sets.items()}
    log = transcript if transcript is not None else []

    # in_flight[owner] = the owner's list as it travels round the ring
    in_flight = {name: p.round_one_message() for name, p in parties.items()}          # round 1
    for owner, elems in in_flight.items():
        log.append((owner, parties[owner].next_party, "ring-1", elems))
    for r in range(2, n + 1):                                                         # rounds 2..n
        nxt = {}
        for owner, elems in in_flight.items():
            holder = parties[ring[(ring.index(owner) + r - 1) % n]]
            nxt[owner] = holder.encrypt_list(elems)
            log.append((holder.name, holder.next_party, f"ring-{r}", nxt[owner]))
        in_flight = nxt
    for owner, elems in in_flight.items():
        parties[owner].final_elements = elems                                          # back at the owner

    published = {name: set(p.publish()) for name, p in parties.items()}
    for name, tags in published.items():
        for other in ring:
            if other != name:
                log.append((name, other, "tags", sorted(tags)))
    for p in parties.values():
        p.finalize({k: v for k, v in published.items() if k != p.name})
    return {name: p.mapping for name, p in parties.items()}
