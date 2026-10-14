"""Phase 3b: the secret-shared aggregation (with distributed DP noise) running across the hospital services."""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from fedhealth.common.schemas import QuerySpec, participating_hospitals
from fedhealth.privacy.linkage.group import get_group
from fedhealth.privacy.smpc.field import decode_vector
from fedhealth.synth.generate import generate
from fedhealth.synth.ground_truth import GroundTruth

from .conftest import TEST_PAD_TO
from .helpers import headline_spec, random_spec
from .leakscan import scan
from .system import build_system

FAST = "test768"          # 768-bit PSI group (INSECURE) - speed only; the slow test below uses the real group
SMALL_BLOB_LIMIT = 2 ** 30


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("mpc_small")
    generate(600, seed=5, out_dir=d)
    return build_system(d, psi_group=FAST, psi_groups=(FAST,)), GroundTruth(d / "ground_truth"), d


def _same(got, want):
    if isinstance(want, float):
        return got == pytest.approx(want)
    return got == want


# ============================================================================ correctness
def test_headline_query_is_computed_correctly_under_mpc(small):
    system, gt, _ = small
    spec = headline_spec()
    res = system.orchestrator.mpc_query(spec)
    assert res.mode == "mpc-exact-no-dp" and res.hospitals == ["A", "B", "C"]
    assert res.result == system.orchestrator.plain_query(spec).result == gt.evaluate(spec)
    assert res.result > 0


def test_mpc_equals_ground_truth_on_random_queries(small):
    """All aggregates, every hospital subset (1, 2 and 3 parties), negative sums, empty cohorts."""
    system, gt, _ = small
    rng = np.random.default_rng(31337)
    seen_sizes, seen_aggs = set(), set()
    for _ in range(45):
        spec = random_spec(rng)
        got = system.orchestrator.mpc_query(spec).result
        assert _same(got, gt.evaluate(spec)), spec.model_dump(exclude_none=True)
        seen_sizes.add(len(participating_hospitals(spec)))
        seen_aggs.add(spec.agg)
    assert seen_sizes == {1, 2, 3} and seen_aggs == {"count", "sum", "avg", "histogram"}


@pytest.mark.parametrize("extra", [
    {"agg": "sum", "value_field": "lab_delta"},          # negative values (a falling HbA1c) survive the field
    {"agg": "avg", "value_field": "adherence_pct"},
    {"agg": "histogram", "group_by": "drug"},
    {"agg": "histogram", "group_by": "outcome"},
])
def test_value_and_histogram_queries(small, extra):
    system, gt, _ = small
    spec = headline_spec(b={}, c={}, a={}, **extra)
    assert _same(system.orchestrator.mpc_query(spec).result, gt.evaluate(spec))


def test_linkage_is_reused_across_queries(small):
    system, _, _ = small
    system.orchestrator.mpc_query(headline_spec())
    again = system.orchestrator.mpc_query(headline_spec(agg="histogram", group_by="outcome", b={}))
    assert again.timings["linkage_s"] == 0.0


# ============================================================================ what travels on the wire
def _mpc_blob_values(events) -> list[int]:
    out = []
    for ev in events:
        if "/mpc/" not in ev.path or not ev.request:
            continue
        doc = json.loads(ev.request)
        blobs = []
        for key in ("x", "y", "z", "d", "e", "noise"):
            if isinstance(doc.get(key), str):
                blobs.append(doc[key])
        blobs += doc.get("vectors", []) if isinstance(doc.get("vectors"), list) else []
        for b in blobs:
            out.extend(decode_vector(base64.b64decode(b)))
    return out


def test_no_patient_data_and_no_small_numbers_cross_any_boundary(small):
    system, _, d = small
    from fedhealth.coordinator.release import PrivacyParams, plan_noise
    system.orchestrator.link(["A", "B", "C"])           # make sure linkage already exists so only MPC traffic is recorded
    system.rec.clear()
    system.orchestrator.mpc_run(headline_spec(), plan_noise(headline_spec(), PrivacyParams(epsilon=1.0)))   # the private path
    ev = system.rec.events
    assert len(ev) > 25, "recorder saw too little - test would be vacuous"

    # 1. leak scanner: no identifiers / MRNs / field names / plain values anywhere
    assert scan(ev) == []

    # 2. every number inside every MPC message is a uniformly random field element: no 0/1 indicator, no
    #    count, no partial sum is visible. (P[random value < 2^30] ~ 2^-31 each.)
    values = _mpc_blob_values(ev)
    assert len(values) > 3000
    assert min(values) >= SMALL_BLOB_LIMIT, f"small value on the wire: {min(values)}"

    # 3. the coordinator-facing responses: acks, plus one share per output cell from each COMPUTING hospital
    out_events = [e for e in ev if e.path.endswith("/output")]
    assert sorted(e.dst for e in out_events) == ["A", "B"]                      # C is the dealer: no output share
    shares = [s for e in out_events for s in json.loads(e.response)["shares"]]
    assert all(s >= SMALL_BLOB_LIMIT for s in shares)            # each hospital's share is itself random-looking

    # 4. the coordinator is a conductor: it never touches the data plane and holds no triple material
    assert not [e for e in ev if e.src == "COORD" and e.path.endswith(("/inshare", "/inopen", "/intriples"))]
    coord_bodies = " ".join(e.request for e in ev if e.src == "COORD")
    assert '"seed"' not in coord_bodies and '"x"' not in coord_bodies
    # ... and no hospital talks to the coordinator at all (it only answers its calls)
    assert all(e.dst != "COORD" for e in ev)
    # 5. triples come from the dealer, openings stay between the two computers
    assert {(e.src, e.dst) for e in ev if e.path.endswith("/intriples")} == {("C", "A"), ("C", "B")}
    assert {(e.src, e.dst) for e in ev if e.path.endswith("/inopen")} == {("A", "B"), ("B", "A")}


def test_noisy_run_never_opens_the_exact_value(small):
    """The private pipeline: every involved hospital adds a noise share inside the MPC. The opened cells differ from
    the truth by exactly the sum of three independent shares, and nothing on the wire carries a small number."""
    from fedhealth.coordinator.release import PrivacyParams, plan_noise
    system, gt, _ = small
    spec = headline_spec(agg="histogram", group_by="outcome", b={})
    plan = plan_noise(spec, PrivacyParams(epsilon=0.5))
    assert plan.contributors == 3 and plan.honest == 2                   # tolerance 1: two honest shares suffice
    system.orchestrator.link(["A", "B", "C"])
    truth = list(gt.evaluate(spec).values())
    diffs = []
    for _ in range(12):
        system.rec.clear()
        run = system.orchestrator.mpc_run(spec, plan)
        diffs += [c - t for c, t in zip(run.cells, truth)]
        assert min(_mpc_blob_values(system.rec.events)) >= SMALL_BLOB_LIMIT
        prepared = [json.loads(e.request) for e in system.rec.events if e.path.endswith("/prepare")]
        assert len(prepared) == 3 and all(p["noise"]["mechanism"] == "laplace" for p in prepared)
    assert any(d != 0 for d in diffs)
    var = sum(d * d for d in diffs) / len(diffs)
    assert 0.2 * plan.total_variance(0) < var < 5 * plan.total_variance(0)


def test_a_two_hospital_query_uses_the_bystander_as_dealer(small):
    from fedhealth.coordinator.release import PrivacyParams, plan_noise
    system, gt, _ = small
    spec = headline_spec(c=None)                                          # A and B only
    assert participating_hospitals(spec) == ["A", "B"]
    system.orchestrator.link(["A", "B"])
    system.rec.clear()
    assert system.orchestrator.mpc_query(spec).result == gt.evaluate(spec)
    ev = system.rec.events
    prep = {e.dst: json.loads(e.request) for e in ev if e.path.endswith("/prepare")}
    assert set(prep) == {"A", "B", "C"} and prep["C"]["query"] is None and prep["C"]["dealer"] == "C"
    assert not [e for e in ev if e.dst == "C" and e.path.endswith(("/inshare", "/intriples", "/inopen"))]   # C receives nothing
    run = system.orchestrator.mpc_run(spec, plan_noise(spec, PrivacyParams(epsilon=1.0)))
    assert abs(run.cells[0] - gt.evaluate(spec)) < 60


def test_hospitals_refuse_weak_or_missing_noise(tmp_path_factory):
    """With the insecure baseline OFF (the real deployment), a hospital refuses an exact computation and any noise plan
    weaker than its own policy - a curious coordinator cannot switch the noise off."""
    from fedhealth.common.schemas import local_query
    from fedhealth.common.wire import MpcPrepare
    from fedhealth.coordinator.release import PrivacyParams, plan_noise
    d = tmp_path_factory.mktemp("strict_noise")
    generate(300, seed=9, out_dir=d)
    system = build_system(d, psi_group=FAST, psi_groups=(FAST,), allow_insecure=False)
    spec = headline_spec(a={}, b=None, c=None)                            # a single-hospital query: no linkage needed
    body = MpcPrepare(parties=["A"], owner="A", agg="count", t_outputs=1, m=0,
                      query=local_query(spec, "A")).model_dump(mode="json")
    a = system.raw["A"]
    assert a.post(f"/mpc/{'aa' * 8}/prepare", json=body).status_code == 400            # no noise -> refused
    ok = plan_noise(spec, PrivacyParams(epsilon=1.0)).to_wire()
    assert a.post(f"/mpc/{'ab' * 8}/prepare", json={**body, "noise": ok}).status_code == 200
    weak = {**ok, "cells": [{**ok["cells"][0], "s": ok["cells"][0]["s"] * 50}]}       # 50x less noise than the cap allows
    r = a.post(f"/mpc/{'ac' * 8}/prepare", json={**body, "noise": weak})
    assert r.status_code == 400 and "noise" in r.text
    lying = {**ok, "contributors": 3}                                                 # shape does not match the query
    assert a.post(f"/mpc/{'ad' * 8}/prepare", json={**body, "noise": lying}).status_code == 400
    out = system.orchestrator.mpc_run(spec, plan_noise(spec, PrivacyParams(epsilon=1.0)))
    assert out.noisy and len(out.cells) == 1


def test_hospitals_refuse_noise_that_needs_every_share_honest(small):
    """Collusion tolerance is the hospital's call: a plan sized for t = 0 (all three shares must stay secret, i.e. the
    coordinator plus one hospital could see past the noise) is refused; t = 2 (stronger) is accepted."""
    from fedhealth.coordinator.orchestrator import ProtocolError
    from fedhealth.coordinator.release import PrivacyParams, ReleasePolicy, plan_noise
    system, _, _ = small
    spec = headline_spec()
    with pytest.raises(ProtocolError, match="honest shares"):
        system.orchestrator.mpc_run(spec, plan_noise(spec, PrivacyParams(epsilon=1.0), ReleasePolicy(tolerance=0)))
    out = system.orchestrator.mpc_run(spec, plan_noise(spec, PrivacyParams(epsilon=1.0), ReleasePolicy(tolerance=2)))
    assert out.noisy and len(out.cells) == 1


def test_hospitals_detect_a_coordinator_that_tells_them_different_plans(small, monkeypatch):
    """Every hospital-to-hospital message carries a digest of the computation header its sender was told; the
    computers refuse a mismatch. Here the coordinator gives hospital A a different (still capped) noise plan, which
    would make A's share smaller than the plan the others checked - the computation is refused, nothing is opened."""
    from fedhealth.coordinator import orchestrator as orch
    from fedhealth.coordinator.orchestrator import ProtocolError
    from fedhealth.coordinator.release import PrivacyParams, plan_noise
    system, _, _ = small
    spec = headline_spec()
    other = plan_noise(spec, PrivacyParams(epsilon=2.0)).to_wire()
    honest_prepare = orch.MpcPrepare

    def equivocating(**kw):
        if kw.get("query") is not None and kw["query"].hospital == "A":
            kw = {**kw, "noise": other}
        return honest_prepare(**kw)

    monkeypatch.setattr(orch, "MpcPrepare", equivocating)
    with pytest.raises(ProtocolError, match="header mismatch"):
        system.orchestrator.mpc_run(spec, plan_noise(spec, PrivacyParams(epsilon=1.0)))
    monkeypatch.setattr(orch, "MpcPrepare", honest_prepare)
    assert system.orchestrator.mpc_run(spec, plan_noise(spec, PrivacyParams(epsilon=1.0))).noisy


def test_psi_lists_enter_the_ring_in_a_fresh_random_order(small, monkeypatch):
    """The last ring hop computes a hospital's fully encrypted list in the order it was sent and can tag it; in MRN
    order that would tell it the local record number of every shared patient. Each linkage reshuffles."""
    from fedhealth.hospital import routes_psi
    system, _, _ = small
    seen: dict[str, list[list[str]]] = {}
    real = routes_psi.PsiParty

    def capture(name, ring, ids, group):
        seen.setdefault(name, []).append(list(ids))
        return real(name, ring, ids, group)

    monkeypatch.setattr(routes_psi, "PsiParty", capture)
    system.orchestrator.link(["A", "C"], force=True)
    system.orchestrator.link(["A", "C"], force=True)
    for h in ("A", "C"):
        first, second = ([x for x in ids if not x.startswith("PAD-")] for ids in seen[h])
        assert sorted(first) == sorted(second) and first != second            # same patients, fresh order
        assert {len(ids) for ids in seen[h]} == {TEST_PAD_TO}                   # padded with fresh random dummies


def test_owner_rejects_an_aggregate_that_does_not_match_its_slice(small):
    """The owner holds the value field, so it knows the query is a SUM/AVG; told 'histogram' (to get a plan budgeted
    as one block) it refuses."""
    from fedhealth.common.schemas import local_query
    from fedhealth.common.wire import MpcPrepare
    from fedhealth.coordinator.release import PrivacyParams, plan_noise
    system, _, _ = small
    spec = QuerySpec.model_validate({"agg": "sum", "condition": "E11", "b": {}, "value_field": "lab_delta"})
    plan = plan_noise(spec, PrivacyParams(epsilon=1.0)).to_wire()
    body = MpcPrepare(parties=["B"], owner="B", agg="histogram", t_outputs=2, m=0, query=local_query(spec, "B"),
                      noise={**plan, "groups": [[0, 2]]}).model_dump(mode="json")
    r = system.raw["B"].post(f"/mpc/{'be' * 8}/prepare", json=body)
    assert r.status_code == 400 and "aggregate" in r.text


def test_owner_rejects_an_understated_sum_sensitivity(small):
    from fedhealth.common.schemas import local_query
    from fedhealth.common.wire import MpcPrepare
    from fedhealth.coordinator.release import PrivacyParams, plan_noise
    system, _, _ = small
    spec = headline_spec(agg="sum", value_field="age", a={}, b=None, c=None)
    good = plan_noise(spec, PrivacyParams(epsilon=1.0)).to_wire()
    low = {**good, "cells": [good["cells"][0], {**good["cells"][1], "sensitivity": 1, "t": good["cells"][1]["t"] // 120}]}
    body = MpcPrepare(parties=["A"], owner="A", agg="sum", t_outputs=2, m=0, query=local_query(spec, "A"),
                      noise=low).model_dump(mode="json")
    r = system.raw["A"].post(f"/mpc/{'be' * 8}/prepare", json=body)
    assert r.status_code == 400 and "sensitivity" in r.text


def test_the_wire_check_has_teeth(small, monkeypatch):
    """Mutation test of the test above: sabotage secret sharing so one 'share' is the plain vector. The
    computation still gives the right answer - but the wire check must now FAIL, otherwise it proves nothing."""
    from fedhealth.privacy.smpc import protocol
    system, gt, _ = small
    system.orchestrator.link(["A", "B", "C"])
    monkeypatch.setattr(protocol, "share_vector", lambda vec, n: [list(vec)] + [[0] * len(vec) for _ in range(n - 1)])
    system.rec.clear()
    res = system.orchestrator.mpc_query(headline_spec())
    assert res.result == gt.evaluate(headline_spec())                 # correctness is unaffected by the sabotage...
    assert min(_mpc_blob_values(system.rec.events)) < SMALL_BLOB_LIMIT   # ...but the leak is visible to the check


def test_each_hospital_receives_only_its_own_predicate(small):
    system, _, _ = small
    system.orchestrator.link(["A", "B", "C"])
    system.rec.clear()
    system.orchestrator.mpc_query(headline_spec())
    prep = {e.dst: e.request for e in system.rec.events if e.path.endswith("/prepare")}
    assert set(prep) == {"A", "B", "C"}
    assert "age_min" in prep["A"] and "metformin" not in prep["A"] and "improved" not in prep["A"]
    assert "metformin" in prep["C"] and "age_min" not in prep["C"] and "improved" not in prep["C"]
    assert "improved" in prep["B"] and "metformin" not in prep["B"] and "age_min" not in prep["B"]


def test_no_single_output_share_reveals_the_answer(small):
    """Neither computing hospital's output share equals the answer; only the SUM of both does (exact route here)."""
    system, gt, _ = small
    spec = headline_spec()
    system.orchestrator.link(["A", "B", "C"])
    system.rec.clear()
    res = system.orchestrator.mpc_query(spec)
    shares = {e.dst: json.loads(e.response)["shares"][0] for e in system.rec.events if e.path.endswith("/output")}
    assert set(shares) == {"A", "B"} and all(s != res.result for s in shares.values())
    assert sum(shares.values()) % ((1 << 61) - 1) == res.result == gt.evaluate(spec)


def test_hospital_state_is_cleaned_up_after_a_query(small):
    system, _, _ = small
    system.orchestrator.mpc_query(headline_spec())
    for h in "ABC":
        assert system.apps[h].state.mpc.sessions == {}
        assert system.apps[h].state.psi.sessions == {}          # PSI keys were destroyed at finalize


# ============================================================================ protocol guards
def test_protocol_rejects_misuse(small):
    system, _, _ = small
    a = system.raw["A"]
    sid = "cd" * 8
    assert a.post(f"/mpc/{sid}/open", json={"layer": 0}).status_code == 404            # unknown computation
    assert a.post("/mpc/not-hex/prepare", json={}).status_code == 422                   # malformed id
    spec = headline_spec()
    from fedhealth.common.schemas import local_query
    from fedhealth.common.wire import MpcPrepare
    info = system.orchestrator.link(["A", "B", "C"])
    good = MpcPrepare(parties=["A", "B", "C"], owner="B", computers=["A", "B"], dealer="C", agg="count", t_outputs=1,
                      m=info.size, lid=info.lid, query=local_query(spec, "A")).model_dump(mode="json")
    assert a.post(f"/mpc/{sid}/prepare", json={**good, "query": local_query(spec, "B").model_dump(mode="json")}).status_code == 400  # wrong hospital's query
    assert a.post(f"/mpc/{sid}/prepare", json={**good, "dealer": "A", "computers": ["B", "C"]}).status_code == 400  # roles are not negotiable
    assert a.post(f"/mpc/{sid}/prepare", json={**good, "m": info.size + 1}).status_code == 400          # not the public bound
    assert a.post(f"/mpc/{sid}/prepare", json={**good, "lid": "ee" * 8}).status_code == 409           # unknown linkage
    assert a.post(f"/mpc/{sid}/prepare", json=good).status_code == 200
    assert a.post(f"/mpc/{sid}/prepare", json=good).status_code == 409                               # id reuse
    assert a.post(f"/mpc/{sid}/open", json={"layer": 0}).status_code == 409                          # inputs / triples missing
    assert a.post(f"/mpc/{sid}/triples", json={"x": "", "y": "", "z": ""}).status_code in (404, 405)  # the coordinator cannot hand out triples
    assert a.post(f"/mpc/{sid}/deal").status_code == 409                                             # A is not the dealer
    digest = MpcPrepare.model_validate(good).header_digest(sid)
    assert a.post(f"/mpc/{sid}/intriples", json={"sender": "B", "session": digest, "seed": "AA=="}).status_code == 403  # only the dealer's triples
    assert a.post(f"/mpc/{sid}/inshare", json={"sender": "A", "session": digest, "vectors": []}).status_code == 403     # from itself
    assert a.post(f"/mpc/{sid}/inshare", json={"sender": "B", "session": "0" * 32, "vectors": []}).status_code == 409  # another header
    assert a.post(f"/mpc/{sid}/output").status_code == 409                                           # not finished
    a.delete(f"/mpc/{sid}")


def test_a_hospital_cannot_be_listed_twice_or_out_of_ring_order(small):
    """Listing A twice (parties A, B, A) once made A both the dealer and a computer: from the openings it could read B's
    vectors, and a plan sized for three noise shares got two. Parties must be distinct and in ring order, every
    hospital (the dealer too) checks the index length, and no hospital accepts a data-plane message from itself."""
    from fedhealth.common.config import load_topology
    from fedhealth.common.schemas import local_query
    from fedhealth.common.wire import MpcPrepare
    from fedhealth.privacy.smpc.protocol import assign_roles
    system, _, _ = small
    spec = headline_spec(c=None)                                                     # A and B; C deals
    info = system.orchestrator.link(["A", "B"])
    good = MpcPrepare(parties=["A", "B"], owner="B", computers=["A", "B"], dealer="C", agg="count", t_outputs=1,
                      m=info.size, lid=info.lid, query=local_query(spec, "A")).model_dump(mode="json")
    a, c = system.raw["A"], system.raw["C"]
    sid = "ef" * 8
    assert a.post(f"/mpc/{sid}/prepare", json={**good, "parties": ["A", "B", "A"], "dealer": "A"}).status_code == 422
    assert a.post(f"/mpc/{sid}/prepare", json={**good, "parties": ["B", "A"], "computers": ["B", "A"]}).status_code == 400
    assert c.post(f"/mpc/{sid}/prepare", json={**good, "query": None, "m": 10 ** 6}).status_code == 400   # the dealer checks m
    assert a.post(f"/psi/{'ef' * 8}/init", json={"ring": ["A", "A", "B"], "group": FAST}).status_code == 422
    with pytest.raises(ValueError):
        assign_roles(["A", "B", "A"], "B", load_topology().ring)
    assert c.post(f"/mpc/{sid}/prepare", json={**good, "query": None}).status_code == 200            # the honest request
    c.delete(f"/mpc/{sid}")


def test_owner_output_shape_must_match_the_query(small):
    system, _, _ = small
    from fedhealth.common.schemas import local_query
    from fedhealth.common.wire import MpcPrepare
    spec = headline_spec(agg="histogram", group_by="outcome", b={})
    info = system.orchestrator.link(["A", "B", "C"])
    body = MpcPrepare(parties=["A", "B", "C"], owner="B", computers=["A", "B"], dealer="C", agg="histogram",
                      t_outputs=2, m=info.size, lid=info.lid, query=local_query(spec, "B")).model_dump(mode="json")  # 3 outcomes, not 2
    assert system.raw["B"].post(f"/mpc/{'ab' * 8}/prepare", json=body).status_code == 400


# ============================================================================ full size, real PSI group (slow)
@pytest.mark.slow
def test_full_size_mpc_with_the_real_group(world, monkeypatch):
    monkeypatch.setenv("FEDHEALTH_LINK_PAD_TO", "4096")
    system = build_system(world["dir"], psi_group="modp2048")
    gt = GroundTruth(world["gt_dir"])
    spec = headline_spec()
    cold = system.orchestrator.mpc_query(spec)
    warm = system.orchestrator.mpc_query(spec)
    assert cold.result == warm.result == gt.evaluate(spec)
    hist = headline_spec(agg="histogram", group_by="drug", b={}, c={})
    assert system.orchestrator.mpc_query(hist).result == gt.evaluate(hist)
    t = warm.timings
    print(f"\n[MPC full size, in-process] cold total {cold.timings['total_s']:.1f}s (linkage {cold.timings['linkage_s']:.1f}s)  "
          f"warm count: " + ", ".join(f"{k}={v:.2f}s" for k, v in t.items()))
