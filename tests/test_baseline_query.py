"""Phase 1 exit criterion: the plain (insecure) baseline equals ground truth computed independently
from the generator's universe. Later phases compare PSI / MPC results against this same baseline."""
from __future__ import annotations

import numpy as np
import pytest

from fedhealth.common.schemas import QuerySpec

from .helpers import headline_spec, random_spec


def _baseline(client, spec: QuerySpec) -> dict:
    r = client.post("/insecure/query", json=spec.model_dump(mode="json", exclude_none=True))
    assert r.status_code == 200, r.text
    return r.json()


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a == pytest.approx(b)
    return a == b


def test_headline_query_matches_ground_truth(coordinator_client, ground_truth):
    spec = headline_spec()
    out = _baseline(coordinator_client, spec)
    assert out["mode"] == "plain-insecure" and out["hospitals"] == ["A", "B", "C"]
    assert out["result"] == ground_truth.evaluate(spec) == out["cohort_size"]
    assert out["result"] > 50                      # a meaningful (not vacuous) cohort at full size


@pytest.mark.parametrize("overrides", [
    {"b": None},                                                   # condition x treatment only   (A, C)
    {"a": None},                                                   # treatment x outcome          (B, C)
    {"a": None, "b": None},                                        # a single hospital           (C)
    {"c": {"drug": "empagliflozin"}},
    {"a": {"sex": "F"}, "b": {}},                                  # outcome-agnostic: "has any follow-up"
])
def test_count_variants(coordinator_client, ground_truth, overrides):
    d = headline_spec().model_dump(exclude_none=True)
    for k, v in overrides.items():
        d.pop(k, None)
        if v is not None:
            d[k] = v
    spec = QuerySpec.model_validate(d)
    assert _baseline(coordinator_client, spec)["result"] == ground_truth.evaluate(spec)


@pytest.mark.parametrize("extra", [
    {"agg": "sum", "value_field": "age"},
    {"agg": "sum", "value_field": "lab_delta"},
    {"agg": "avg", "value_field": "adherence_pct"},
    {"agg": "avg", "value_field": "age"},
    {"agg": "histogram", "group_by": "outcome"},
    {"agg": "histogram", "group_by": "drug"},
    {"agg": "histogram", "group_by": "age_band"},
    {"agg": "histogram", "group_by": "sex"},
])
def test_aggregates_match_ground_truth(coordinator_client, ground_truth, extra):
    spec = headline_spec(b={}, c={}, **extra)
    got = _baseline(coordinator_client, spec)["result"]
    want = ground_truth.evaluate(spec)
    assert _same(got, want), (got, want)
    if spec.agg == "histogram":
        assert sum(got.values()) > 0 and set(got) == set(want)      # zero bins are reported, shape is fixed


def test_differential_baseline_vs_ground_truth_on_random_queries(coordinator_client, ground_truth):
    """60 random valid queries across all aggregates and hospital subsets."""
    rng = np.random.default_rng(2024)
    kinds = set()
    for _ in range(60):
        spec = random_spec(rng)
        kinds.add(spec.agg)
        got, want = _baseline(coordinator_client, spec)["result"], ground_truth.evaluate(spec)
        assert _same(got, want), (spec.model_dump(exclude_none=True), got, want)
    assert kinds == {"count", "sum", "avg", "histogram"}


def test_row_level_requests_are_rejected_by_the_coordinator(coordinator_client):
    for bad in ({"agg": "select", "condition": "E11", "a": {}},
                {"agg": "count", "condition": "E11", "a": {}, "columns": ["full_name"]},
                {"agg": "count", "condition": "E11", "a": {"zip_code": "12345"}}):
        assert coordinator_client.post("/insecure/query", json=bad).status_code == 422


def test_baseline_really_is_insecure(hospital_clients):
    """Documents WHY the baseline exists: it ships raw national IDs. Phase 2's leak scanner
    asserts the opposite for the private pipeline."""
    r = hospital_clients["C"].post("/insecure/match", json={"hospital": "C", "condition": "E11", "predicate": {}})
    assert r.status_code == 200 and r.json()["rows"][0]["national_id"].startswith("SYN-")
