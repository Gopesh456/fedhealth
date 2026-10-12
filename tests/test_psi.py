"""Phase 2: private set intersection (commutative encryption) + the HMAC baseline."""
from __future__ import annotations

import random

import numpy as np
import pytest

from fedhealth.common.schemas import QuerySpec, participating_hospitals
from fedhealth.privacy.linkage import hmac_baseline
from fedhealth.privacy.linkage.group import HAVE_GMPY2, get_group, pack_ints, unpack_ints
from fedhealth.privacy.linkage.psi import psi_intersection_local
from fedhealth.synth.generate import generate
from fedhealth.synth.ground_truth import GroundTruth

from .helpers import headline_spec, random_spec
from .leakscan import scan
from .system import build_system

FAST = "test768"      # 768-bit group: INSECURE, only used to keep tests quick


# ============================================================================ the group
def _is_probable_prime(n: int, rounds: int = 24) -> bool:
    if n < 4:
        return n in (2, 3)
    if n % 2 == 0:
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    rng = random.Random(1234)
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


@pytest.mark.parametrize("name,bits,prefix,suffix", [
    ("test768", 768, "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1", "FFFFFFFFFFFFFFFF"),
    ("modp1536", 1536, "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1", "CA237327FFFFFFFFFFFFFFFF"),
    ("modp2048", 2048, "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1", "15728E5A8AACAA68FFFFFFFFFFFFFFFF"),
])
def test_group_prime_is_the_published_safe_prime(name, bits, prefix, suffix):
    """The prime is computed from the RFC definition (no pasted constant). Verify it is what the RFC lists
    (known leading / trailing hex) AND that it really is a safe prime: p and (p-1)/2 both prime."""
    g = get_group(name)
    assert g.p.bit_length() == bits
    hexp = f"{g.p:X}"
    assert hexp.startswith(prefix) and hexp.endswith(suffix)
    assert _is_probable_prime(g.p) and _is_probable_prime(g.q)
    assert g.q == (g.p - 1) // 2


def test_hash_to_group_lands_in_the_prime_order_subgroup():
    g = get_group(FAST)
    for i in range(20):
        h = g.hash_to_group(f"SYN-{i}".encode())
        assert 1 < h < g.p
        assert pow(h, g.q, g.p) == 1                 # Euler's criterion: h is a quadratic residue
    assert g.hash_to_group(b"x") == g.hash_to_group(b"x") != g.hash_to_group(b"y")


def test_encryption_is_commutative_and_order_independent():
    g = get_group(FAST)
    x = g.hash_to_group(b"patient")
    a, b, c = g.new_key(), g.new_key(), g.new_key()
    abc = g.encrypt(g.encrypt(g.encrypt(x, a), b), c)
    assert abc == g.encrypt(g.encrypt(g.encrypt(x, c), a), b) == g.encrypt(g.encrypt(g.encrypt(x, b), c), a)
    assert abc != g.encrypt(g.encrypt(x, a), b)      # all keys are needed to reach the shared value


def test_pack_roundtrip():
    g = get_group(FAST)
    vals = [g.hash_to_group(str(i).encode()) for i in range(7)]
    assert unpack_ints(pack_ints(vals, g.nbytes), g.nbytes) == vals


# ============================================================================ the protocol (pure, no network)
def _random_sets(rng: random.Random, n_parties: int, universe: int = 60):
    names = "ABC"[:n_parties]
    pool = [f"SYN-{i:09d}" for i in range(universe)]
    return {n: rng.sample(pool, rng.randint(0, 40)) for n in names}


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("n_parties", [2, 3])
def test_psi_equals_plain_intersection(seed, n_parties):
    rng = random.Random(seed)
    sets = _random_sets(rng, n_parties)
    result = psi_intersection_local(sets, FAST)
    truth = set.intersection(*(set(v) for v in sets.values()))
    for party, mapping in result.items():
        assert set(mapping) == truth, party                           # each party learns exactly the intersection
    if truth:
        # the common index: identical position for the same patient at every party, and a dense 0..m-1 range
        for pid in truth:
            assert len({m[pid] for m in result.values()}) == 1
        assert sorted(next(iter(result.values())).values()) == list(range(len(truth)))


@pytest.mark.parametrize("sets", [
    {"A": [], "B": ["x", "y"], "C": ["x"]},                           # an empty party
    {"A": ["x"], "B": ["y"], "C": ["z"]},                             # disjoint
    {"A": ["x", "y"], "B": ["x", "y"], "C": ["y", "x"]},              # identical
])
def test_psi_edge_cases(sets):
    result = psi_intersection_local(sets, FAST)
    truth = set.intersection(*(set(v) for v in sets.values()))
    assert all(set(m) == truth for m in result.values())


def test_each_session_uses_fresh_keys():
    """No cross-session linkability: the same ids produce unrelated ciphertexts in two sessions."""
    t1, t2 = [], []
    sets = {"A": ["SYN-1", "SYN-2"], "B": ["SYN-2", "SYN-3"]}
    psi_intersection_local(sets, FAST, transcript=t1)
    psi_intersection_local(sets, FAST, transcript=t2)
    ring1 = {v for _, _, kind, payload in t1 if kind.startswith("ring") for v in payload}
    ring2 = {v for _, _, kind, payload in t2 if kind.startswith("ring") for v in payload}
    tags1 = {v for _, _, kind, payload in t1 if kind == "tags" for v in payload}
    tags2 = {v for _, _, kind, payload in t2 if kind == "tags" for v in payload}
    assert not ring1 & ring2 and not tags1 & tags2


def test_transcript_never_contains_plain_hashed_ids():
    g = get_group(FAST)
    sets = {"A": [f"SYN-{i}" for i in range(30)], "B": [f"SYN-{i}" for i in range(15, 45)]}
    transcript = []
    psi_intersection_local(sets, FAST, transcript=transcript)
    plain = {g.hash_to_group(i.encode()) for ids in sets.values() for i in ids}
    on_wire = {v for _, _, kind, payload in transcript if kind.startswith("ring") for v in payload}
    assert not plain & on_wire


# ============================================================================ PSI vs the HMAC baseline
def test_hmac_baseline_gives_the_same_intersection():
    sets = {"A": [f"SYN-{i}" for i in range(0, 40)], "B": [f"SYN-{i}" for i in range(20, 60)]}
    key = b"shared-secret-key"
    shared = hmac_baseline.intersect_tokens({p: hmac_baseline.pseudonymise(ids, key) for p, ids in sets.items()})
    psi = psi_intersection_local(sets, FAST)
    assert shared["A"] == set(psi["A"]) and shared["B"] == set(psi["B"])


def test_hmac_leaks_membership_of_non_shared_patients_but_psi_does_not():
    """Hospital A tries to find out which of B's patients it does NOT share. With HMAC (A holds the key)
    a dictionary over candidate IDs reveals B's whole membership. With PSI, A can only apply ITS OWN key,
    so nothing outside the intersection can be recognised in what it received."""
    universe = [f"SYN-{i:09d}" for i in range(200)]
    a_ids, b_ids = universe[:100], universe[60:160]                # overlap = universe[60:100]
    shared, b_only = set(a_ids) & set(b_ids), set(b_ids) - set(a_ids)
    assert len(b_only) == 60

    # --- HMAC: A knows K and B's published token list
    key = b"k" * 32
    b_tokens = set(hmac_baseline.pseudonymise(b_ids, key).values())
    guessed = {c for c, t in hmac_baseline.pseudonymise(universe, key).items() if t in b_tokens}
    assert guessed == set(b_ids)                                    # A now knows ALL of B's patients
    assert b_only <= guessed

    # --- PSI: everything A receives, plus A's own key, is all A has
    g = get_group(FAST)
    transcript = []
    psi_intersection_local({"A": a_ids, "B": b_ids}, FAST, transcript=transcript)
    received_by_a = {v for s, r, kind, p in transcript if r == "A" and kind.startswith("ring") for v in p}
    received_tags = {v for s, r, kind, p in transcript if r == "A" and kind == "tags" for v in p}
    key_a = g.new_key()      # even A's real key would be no better: the values it holds are under B's key too
    hits = {c for c in universe
            if g.encrypt(g.hash_to_group(c.encode()), key_a) in received_by_a
            or g.tag(g.encrypt(g.hash_to_group(c.encode()), key_a)) in received_tags}
    assert not hits & b_only


# ============================================================================ over HTTP (in-process services)
@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("small")
    generate(600, seed=5, out_dir=d)
    system = build_system(d, psi_group=FAST, psi_groups=(FAST,))
    return system, GroundTruth(d / "ground_truth"), d


def test_linkage_cohort_sizes_are_exact(small):
    system, _, _ = small
    orch = system.orchestrator
    abc = orch.link(["A", "B", "C"])
    ac = orch.link(["A", "C"])
    assert abc.m == 240 and not abc.cached           # 40% of 600 are at all three hospitals
    assert ac.m == 312                               # A∩C = ABC (40%) + AC (12%)
    assert orch.link(["A", "B", "C"]).cached          # second call is served from the cache


def test_force_relink_replaces_the_old_linkage(small):
    system, _, _ = small
    orch = system.orchestrator
    first = orch.link(["B", "C"])
    second = orch.link(["B", "C"], force=True)
    assert second.lid != first.lid and second.m == first.m == 270      # B∩C = BC (5%) + ABC (40%) of 600
    for h in "BC":
        assert first.lid not in system.apps[h].state.psi.linkages
        assert second.lid in system.apps[h].state.psi.linkages


def test_psi_linked_query_equals_plain_and_ground_truth(small):
    system, gt, _ = small
    orch = system.orchestrator
    rng = np.random.default_rng(99)
    multi = 0
    for spec in [headline_spec()] + [random_spec(rng) for _ in range(25)]:
        plain, linked, want = orch.plain_query(spec).result, orch.psi_plain_query(spec).result, gt.evaluate(spec)
        if isinstance(want, float):
            assert plain == pytest.approx(want) and linked == pytest.approx(want)
        else:
            assert plain == linked == want, spec.model_dump(exclude_none=True)
        multi += len(participating_hospitals(spec)) > 1
    assert multi >= 10, "most random queries should exercise the linkage path"


def test_wire_traffic_of_the_psi_protocol_is_clean(small):
    """The leak scanner finds no identifier, MRN, name, field name or plain hash in ANY message of a full
    linkage - not between hospitals and not between coordinator and hospitals."""
    system, gt, d = small
    system.rec.clear()
    info = system.orchestrator.link(["A", "B", "C"], force=True)
    assert info.m == 240
    ids = [p.national_id for p in gt.persons.values()]               # every identifier in the world
    assert system.rec.events, "recorder saw nothing - the test would be vacuous"
    assert scan(system.rec.events, group=get_group(FAST), plain_ids=ids) == []
    # and the coordinator-facing responses carry nothing but status words and (m, digest)
    coord_resp = [e.response for e in system.rec.events if e.src == "COORD"]
    assert all(len(r) < 200 for r in coord_resp), "a coordinator-facing response is suspiciously large"
    # peers exchanged the heavy data; the coordinator never received any data-plane message
    assert not [e for e in system.rec.events if e.src == "COORD" and e.path.endswith(("/inbox", "/tags"))]


def test_the_insecure_baseline_fails_the_same_scanner(small):
    system, _, _ = small
    system.rec.clear()
    system.orchestrator.plain_query(headline_spec())
    violations = scan(system.rec.events)
    assert any("SYN-" in v or "national_id" in v for v in violations)


def test_hospital_enforces_its_own_crypto_policy(small):
    """The coordinator cannot talk a hospital into a weak group: the hospital has its own allow-list."""
    system, _, d = small
    strict = build_system(d, psi_group="test768", psi_groups=("modp2048",))
    with pytest.raises(Exception, match="not accepted"):
        strict.orchestrator.link(["A", "B"])


def test_peer_messages_are_only_accepted_from_the_ring_neighbour(small):
    system, _, _ = small
    lid = "ab" * 8
    ring = ["A", "B", "C"]
    for h in ring:
        r = system.raw[h].post(f"/psi/{lid}/init", json={"ring": ring, "group": FAST})
        assert r.status_code == 200
    # C is A's predecessor in the ring A->B->C->A; B is NOT
    bad = system.raw["A"].post(f"/psi/{lid}/inbox", json={"sender": "B", "round": 1, "blob": ""})
    assert bad.status_code == 403
    ok = system.raw["A"].post(f"/psi/{lid}/inbox", json={"sender": "C", "round": 1, "blob": ""})
    assert ok.status_code == 200
    for h in ring:
        system.raw[h].delete(f"/psi/{lid}")


def test_malformed_linkage_ids_are_rejected(small):
    system, _, _ = small
    assert system.raw["A"].post("/psi/../etc/init", json={}).status_code in (404, 422)
    assert system.raw["A"].post("/psi/NOT-HEX!/init", json={"ring": ["A", "B"]}).status_code == 422


def test_psi_does_not_return_set_sizes_to_the_coordinator(small):
    system, _, _ = small
    system.rec.clear()
    system.orchestrator.link(["A", "B"], force=True)
    inits = [e for e in system.rec.events if e.src == "COORD" and e.path.endswith("/init")]
    assert inits and all(e.response == '{"status":"ready"}' for e in inits)


# ============================================================================ full size, real group (slow)
@pytest.mark.slow
def test_full_size_linkage_with_the_real_2048_bit_group(world):
    """5,000 patients, RFC 3526 2048-bit group: PSI cohort sizes equal the exact overlap of the generator."""
    system = build_system(world["dir"], psi_group="modp2048")
    abc = system.orchestrator.link(["A", "B", "C"])
    ac = system.orchestrator.link(["A", "C"])
    print(f"\n[PSI timing, in-process threads, gmpy2={HAVE_GMPY2}] "
          f"A,B,C: {abc.seconds:.1f}s (m={abc.m})   A,C: {ac.seconds:.1f}s (m={ac.m})")
    assert abc.m == 2000 and ac.m == 2600
    spec = headline_spec()
    gt = GroundTruth(world["gt_dir"])
    assert system.orchestrator.psi_plain_query(spec).result == gt.evaluate(spec)
