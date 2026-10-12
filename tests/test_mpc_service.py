"""Phase 3b: the secret-shared aggregation running across the hospital services."""
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
        for key in ("x", "y", "z", "d", "e"):
            if isinstance(doc.get(key), str):
                blobs.append(doc[key])
        blobs += doc.get("vectors", []) if isinstance(doc.get("vectors"), list) else []
        for b in blobs:
            out.extend(decode_vector(base64.b64decode(b)))
    return out


def test_no_patient_data_and_no_small_numbers_cross_any_boundary(small):
    system, _, d = small
    system.orchestrator.link(["A", "B", "C"])           # make sure linkage already exists so only MPC traffic is recorded
    system.rec.clear()
    system.orchestrator.mpc_query(headline_spec())
    ev = system.rec.events
    assert len(ev) > 30, "recorder saw too little - test would be vacuous"

    # 1. leak scanner: no identifiers / MRNs / field names / plain values anywhere
    assert scan(ev) == []

    # 2. every number inside every MPC message is a uniformly random field element: no 0/1 indicator, no
    #    count, no partial sum is visible. (P[random value < 2^30] ~ 2^-31 each.)
    values = _mpc_blob_values(ev)
    assert len(values) > 5000
    assert min(values) >= SMALL_BLOB_LIMIT, f"small value on the wire: {min(values)}"

    # 3. the coordinator-facing responses: acks, plus one share per output cell at the very end
    out_events = [e for e in ev if e.path.endswith("/output")]
    assert len(out_events) == 3
    shares = [s for e in out_events for s in json.loads(e.response)["shares"]]
    assert all(s >= SMALL_BLOB_LIMIT for s in shares)            # each hospital's share is itself random-looking

    # 4. the coordinator is a conductor: it never touches the data plane
    assert not [e for e in ev if e.src == "COORD" and e.path.endswith(("/inshare", "/inopen"))]
    # ... and no hospital talks to the coordinator at all (it only answers its calls)
    assert all(e.dst != "COORD" for e in ev)


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


def test_the_exact_aggregate_exists_only_at_the_coordinator(small):
    """No single hospital's output share equals the true count; only the SUM of all three does."""
    system, gt, _ = small
    spec = headline_spec()
    system.orchestrator.link(["A", "B", "C"])
    system.rec.clear()
    res = system.orchestrator.mpc_query(spec)
    shares = {e.dst: json.loads(e.response)["shares"][0] for e in system.rec.events if e.path.endswith("/output")}
    assert all(s != res.result for s in shares.values())
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
    good = MpcPrepare(parties=["A", "B", "C"], owner="B", agg="count", t_outputs=1, m=info.m, lid=info.lid,
                      query=local_query(spec, "A")).model_dump(mode="json")
    assert a.post(f"/mpc/{sid}/prepare", json={**good, "query": local_query(spec, "B").model_dump(mode="json")}).status_code == 400  # wrong hospital's query
    assert a.post(f"/mpc/{sid}/prepare", json={**good, "m": info.m + 1}).status_code == 409          # inconsistent with linkage
    assert a.post(f"/mpc/{sid}/prepare", json={**good, "lid": "ee" * 8}).status_code == 409           # unknown linkage
    assert a.post(f"/mpc/{sid}/prepare", json=good).status_code == 200
    assert a.post(f"/mpc/{sid}/prepare", json=good).status_code == 409                               # id reuse
    assert a.post(f"/mpc/{sid}/open", json={"layer": 0}).status_code == 409                          # inputs / triples missing
    assert a.post(f"/mpc/{sid}/triples", json={"x": "", "y": "", "z": ""}).status_code == 400        # wrong triple count
    assert a.post(f"/mpc/{sid}/inshare", json={"sender": "A", "vectors": []}).status_code == 400     # from itself
    assert a.post(f"/mpc/{sid}/output").status_code == 409                                           # not finished
    a.delete(f"/mpc/{sid}")


def test_owner_output_shape_must_match_the_query(small):
    system, _, _ = small
    from fedhealth.common.schemas import local_query
    from fedhealth.common.wire import MpcPrepare
    spec = headline_spec(agg="histogram", group_by="outcome", b={})
    info = system.orchestrator.link(["A", "B", "C"])
    body = MpcPrepare(parties=["A", "B", "C"], owner="B", agg="histogram", t_outputs=2, m=info.m, lid=info.lid,
                      query=local_query(spec, "B")).model_dump(mode="json")        # 3 outcomes, not 2
    assert system.raw["B"].post(f"/mpc/{'ab' * 8}/prepare", json=body).status_code == 400


# ============================================================================ full size, real PSI group (slow)
@pytest.mark.slow
def test_full_size_mpc_with_the_real_group(world):
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
