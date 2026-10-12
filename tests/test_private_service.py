"""Phase 4b: the full private pipeline (PSI -> MPC -> k-threshold -> DP -> ledger) across the services."""
from __future__ import annotations

import numpy as np
import pytest

from fedhealth.common.schemas import QuerySpec
from fedhealth.coordinator.orchestrator import ProtocolError
from fedhealth.coordinator.release import PrivacyParams
from fedhealth.coordinator.service import QueryService
from fedhealth.privacy.dp.ledger import EpsilonLedger, make_engine
from fedhealth.synth.generate import generate
from fedhealth.synth.ground_truth import GroundTruth

from .helpers import headline_spec, random_spec
from .system import build_system

FAST = "test768"          # insecure 768-bit PSI group, tests only


@pytest.fixture(scope="module")
def world_small(tmp_path_factory):
    d = tmp_path_factory.mktemp("dp_small")
    generate(600, seed=5, out_dir=d)
    return build_system(d, psi_group=FAST, psi_groups=(FAST,)), GroundTruth(d / "ground_truth")


@pytest.fixture()
def svc(world_small, tmp_path):
    system, gt = world_small
    ledger = EpsilonLedger(make_engine(str(tmp_path / "coordinator.db")))
    ledger.ensure_user("alice", 5.0, 1e-4)
    ledger.ensure_user("bob", 5.0, 1e-4)
    system.rec.clear()
    return QueryService(system.orchestrator, ledger), ledger, system, gt


def test_released_answer_is_a_noisy_version_of_the_truth_and_costs_epsilon(svc):
    service, ledger, _, gt = svc
    truth = gt.evaluate(headline_spec())
    out = service.execute("alice", headline_spec(), PrivacyParams(epsilon=5.0))      # alice's whole budget; noise scale 0.2
    assert out.status == "released" and out.hospitals == ["A", "B", "C"]
    assert abs(out.released.value - truth) <= 3
    assert out.epsilon_charged == 5.0 and out.epsilon_remaining == pytest.approx(0.0)
    assert ledger.balance("alice").spent_epsilon == 5.0


def test_budget_is_debited_and_exhaustion_blocks_without_touching_hospitals(svc):
    service, ledger, system, _ = svc
    p = PrivacyParams(epsilon=2.0)
    r1 = service.execute("alice", headline_spec(), p)
    r2 = service.execute("alice", headline_spec(), p)
    assert (r1.status, r2.status) == ("released", "released")
    assert r2.epsilon_remaining == pytest.approx(1.0)
    system.rec.clear()
    r3 = service.execute("alice", headline_spec(), p)                                 # needs 2.0, only 1.0 left
    assert r3.status == "refused" and r3.reason == "privacy budget exhausted" and r3.released is None
    assert r3.epsilon_remaining == pytest.approx(1.0)
    assert system.rec.events == [], "a refused query must not generate any hospital traffic"
    assert service.execute("bob", headline_spec(), p).status == "released"            # bob's budget is his own
    assert [e["status"] for e in ledger.entries("alice")] == ["committed", "committed"]


def test_small_cells_are_suppressed_but_still_charged(svc):
    service, ledger, _, gt = svc
    rare = headline_spec(condition="I50", a={}, b={"outcome": "improved"}, c={"drug": "ivabradine"})
    assert gt.evaluate(rare) < 10
    out = service.execute("alice", rare, PrivacyParams(epsilon=1.0))
    assert out.status == "suppressed" and out.released.value is None and out.released.suppressed is True
    assert out.epsilon_charged == 1.0 and ledger.balance("alice").spent_epsilon == 1.0


def test_suppression_matches_the_exact_threshold_on_random_count_queries(svc):
    service, ledger, _, gt = svc
    ledger.ensure_user("carol", 1000.0)
    rng = np.random.default_rng(77)
    seen = set()
    for _ in range(40):
        spec = random_spec(rng)
        if spec.agg != "count":
            continue
        out = service.execute("carol", spec, PrivacyParams(epsilon=1.0))
        assert (out.status == "suppressed") == (gt.evaluate(spec) < 10), spec.model_dump(exclude_none=True)
        seen.add(out.status)
    assert seen == {"released", "suppressed"}


def test_all_aggregates_are_released_in_natural_units(svc):
    service, ledger, _, gt = svc
    ledger.ensure_user("dora", 1000.0)
    p = PrivacyParams(epsilon=10.0)                                                   # tiny noise: compare with the truth
    wide = dict(a={}, b={}, c={})
    avg = service.execute("dora", headline_spec(agg="avg", value_field="adherence_pct", **wide), p).released
    assert avg.value == pytest.approx(gt.evaluate(headline_spec(agg="avg", value_field="adherence_pct", **wide)), abs=2.0)
    lab = service.execute("dora", headline_spec(agg="sum", value_field="lab_delta", **wide), p).released
    assert lab.value == pytest.approx(gt.evaluate(headline_spec(agg="sum", value_field="lab_delta", **wide)) / 100, abs=3.0)
    hist = service.execute("dora", headline_spec(agg="histogram", group_by="outcome", **wide), p).released
    truth = gt.evaluate(headline_spec(agg="histogram", group_by="outcome", **wide))
    assert set(hist.value) == set(truth)
    for k, v in truth.items():
        if hist.value[k] is not None:
            assert abs(hist.value[k] - v) <= 3


def test_a_hospital_failure_refunds_the_budget(svc, monkeypatch):
    service, ledger, system, _ = svc

    def boom(*a, **k):
        raise ProtocolError("hospital B went away")

    monkeypatch.setattr(system.orchestrator, "mpc_run", boom)
    out = service.execute("alice", headline_spec(), PrivacyParams(epsilon=3.0))
    assert out.status == "failed" and out.released is None
    assert ledger.balance("alice").spent_epsilon == 0.0
    assert [e["status"] for e in ledger.entries("alice")] == ["refunded"]


def test_repeating_a_query_gives_fresh_noise_each_time(svc):
    """Fresh noise per query: the answers differ (so a naive 'ask many times and average' needs many queries,
    and every one is charged by the ledger - see attacks/budget_exhaustion.py)."""
    service, ledger, _, _ = svc
    ledger.ensure_user("erin", 1000.0)
    answers = [service.execute("erin", headline_spec(), PrivacyParams(epsilon=0.3)).released.value for _ in range(8)]
    assert len(set(answers)) > 1
    assert ledger.balance("erin").spent_epsilon == pytest.approx(2.4)


def test_nothing_exact_leaves_the_coordinator_in_the_response(svc):
    """The outcome object (what an API layer would serialise) contains the noisy value and metadata only."""
    service, _, _, gt = svc
    out = service.execute("alice", headline_spec(), PrivacyParams(epsilon=0.1))
    blob = repr(out)
    assert "cells" not in blob and "total=" not in blob and "Exact" not in blob


def test_gaussian_mechanism_charges_delta(svc):
    service, ledger, _, _ = svc
    out = service.execute("alice", headline_spec(), PrivacyParams(epsilon=1.0, mechanism="gaussian"))
    assert out.status == "released" and out.delta_charged == 1e-6
    assert ledger.balance("alice").spent_delta == pytest.approx(1e-6)
