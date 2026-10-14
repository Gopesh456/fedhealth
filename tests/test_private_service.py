"""Phase 4b: the full private pipeline (PSI -> MPC -> k-threshold -> DP -> ledger) across the services."""
from __future__ import annotations

import httpx
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


WIDE = headline_spec(a={}, b={})          # true count 42 in this world: far above k, never suppressed by the noise


def test_released_answer_is_a_noisy_version_of_the_truth_and_costs_epsilon(svc):
    service, ledger, _, gt = svc
    truth = gt.evaluate(WIDE)
    out = service.execute("alice", WIDE, PrivacyParams(epsilon=2.0))                  # the per-query maximum; noise sd ~0.7
    assert out.status == "released" and out.hospitals == ["A", "B", "C"]
    assert abs(out.released.value - truth) <= 6
    assert out.epsilon_charged == 2.0 and out.epsilon_remaining == pytest.approx(3.0)
    assert ledger.balance("alice").spent_epsilon == 2.0


def test_a_plan_above_the_hospitals_limit_is_refused_before_any_traffic(svc):
    service, ledger, system, _ = svc
    out = service.execute("alice", WIDE, PrivacyParams(epsilon=5.0))                  # policy.yaml: at most 2 per query
    assert out.status == "refused" and "per-query" in out.reason and system.rec.events == []
    assert ledger.balance("alice").spent_epsilon == 0.0


def test_budget_is_debited_and_exhaustion_blocks_without_touching_hospitals(svc):
    service, ledger, system, _ = svc
    p = PrivacyParams(epsilon=2.0)
    r1 = service.execute("alice", WIDE, p)           # WIDE, not the headline query: its true count 11 sits at k = 10, so
    r2 = service.execute("alice", WIDE, p)           # the noise suppressed one of these three answers ~8% of the time
    assert (r1.status, r2.status) == ("released", "released")
    assert r2.epsilon_remaining == pytest.approx(1.0)
    system.rec.clear()
    r3 = service.execute("alice", headline_spec(), p)                                 # needs 2.0, only 1.0 left
    assert r3.status == "refused" and r3.reason == "privacy budget exhausted" and r3.released is None
    assert r3.epsilon_remaining == pytest.approx(1.0)
    assert system.rec.events == [], "a refused query must not generate any hospital traffic"
    assert service.execute("bob", WIDE, p).status == "released"                       # bob's budget is his own
    assert [e["status"] for e in ledger.entries("alice")] == ["committed", "committed"]


def test_small_cells_are_suppressed_but_still_charged(svc):
    service, ledger, _, gt = svc
    rare = headline_spec(condition="I50", a={}, b={"outcome": "improved"}, c={"drug": "ivabradine"})
    assert gt.evaluate(rare) < 10
    out = service.execute("alice", rare, PrivacyParams(epsilon=1.0))
    assert out.status == "suppressed" and out.released.value is None and out.released.suppressed is True
    assert out.epsilon_charged == 1.0 and ledger.balance("alice").spent_epsilon == 1.0


def test_suppression_is_decided_on_the_noisy_count(svc):
    """The k-threshold sees only the NOISY count: far from k it agrees with the exact threshold, near k it is fuzzy,
    and a released count is never below k."""
    service, ledger, _, gt = svc
    ledger.ensure_user("carol", 1000.0)
    rng = np.random.default_rng(77)
    seen = set()
    for _ in range(40):
        spec = random_spec(rng)
        if spec.agg != "count":
            continue
        out = service.execute("carol", spec, PrivacyParams(epsilon=1.0))               # total noise sd ~1.7
        truth = gt.evaluate(spec)
        if out.status == "released":
            assert out.released.value >= 10
        if abs(truth - 10) > 15:
            assert (out.status == "suppressed") == (truth < 10), spec.model_dump(exclude_none=True)
        seen.add(out.status)
    assert seen == {"released", "suppressed"}


def test_all_aggregates_are_released_in_natural_units(svc):
    service, ledger, _, gt = svc
    ledger.ensure_user("dora", 1000.0)
    p = PrivacyParams(epsilon=2.0)                                                    # the per-query maximum
    wide = dict(a={}, b={}, c={})
    avg = service.execute("dora", headline_spec(agg="avg", value_field="adherence_pct", **wide), p).released
    assert avg.value == pytest.approx(gt.evaluate(headline_spec(agg="avg", value_field="adherence_pct", **wide)), abs=8.0)
    lab = service.execute("dora", headline_spec(agg="sum", value_field="lab_delta", **wide), p).released
    truth_lab = gt.evaluate(headline_spec(agg="sum", value_field="lab_delta", **wide)) / 100
    assert lab.value == pytest.approx(truth_lab, abs=6 * lab.noise_scale)                # natural units, not scaled ints
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
    out = service.execute("alice", headline_spec(), PrivacyParams(epsilon=2.0))
    assert out.status == "failed" and out.released is None
    assert ledger.balance("alice").spent_epsilon == 0.0
    assert [e["status"] for e in ledger.entries("alice")] == ["refunded"]


def test_a_failure_after_the_hospitals_shared_their_data_is_charged(svc, monkeypatch):
    """Once shares have left a hospital, an abort is charged: if aborts depended on the data, a refunded retry would
    be a free channel outside the budget."""
    from fedhealth.coordinator.orchestrator import ComputationAborted
    service, ledger, system, _ = svc
    ledger.ensure_user("frank", 10.0)

    def late_boom(*a, **k):
        raise ComputationAborted("a computer went away during the multiplication")

    monkeypatch.setattr(system.orchestrator, "mpc_run", late_boom)
    out = service.execute("frank", headline_spec(), PrivacyParams(epsilon=2.0))
    assert out.status == "failed" and out.released is None and out.epsilon_charged == 2.0
    assert ledger.balance("frank").spent_epsilon == pytest.approx(2.0)
    assert [e["status"] for e in ledger.entries("frank")] == ["committed"]


def test_an_unreachable_hospital_during_cleanup_does_not_turn_a_charged_abort_into_a_refund(svc, monkeypatch):
    """B goes away after the shares were sent: its /open AND the cleanup DELETE fail. The abort must still be charged
    (the cleanup error used to replace ComputationAborted, and the query was refunded)."""
    service, ledger, system, _ = svc
    ledger.ensure_user("gina", 10.0)
    client = system.orchestrator.clients["B"]
    real_post, real_delete = client.post, client.delete
    gone = {"yes": False}

    def post(path, **kw):
        if path.endswith("/open"):
            gone["yes"] = True
        if gone["yes"]:
            raise httpx.ConnectError("B is unreachable")
        return real_post(path, **kw)

    def delete(path, **kw):
        if gone["yes"]:
            raise httpx.ConnectError("B is unreachable")
        return real_delete(path, **kw)

    monkeypatch.setattr(client, "post", post)
    monkeypatch.setattr(client, "delete", delete)
    out = service.execute("gina", headline_spec(), PrivacyParams(epsilon=1.0))
    assert gone["yes"] and out.status == "failed" and out.epsilon_charged == 1.0
    assert [e["status"] for e in ledger.entries("gina")] == ["committed"]


def test_a_hospital_that_lost_its_linkage_is_linked_again(svc):
    """Hospitals keep linkages in memory: after a restart the coordinator's cached linkage is gone there. The run is
    refused at prepare (nothing shared yet), so the coordinator links afresh and runs once more."""
    service, ledger, system, _ = svc
    ledger.ensure_user("hugo", 10.0)
    assert service.execute("hugo", headline_spec(), PrivacyParams(epsilon=0.5)).status in ("released", "suppressed")
    system.apps["B"].state.psi.linkages.clear()                                      # "B restarted"
    out = service.execute("hugo", headline_spec(), PrivacyParams(epsilon=0.5))
    assert out.status in ("released", "suppressed") and system.apps["B"].state.psi.linkages


def test_the_orchestrator_marks_aborts_after_sharing(svc, monkeypatch):
    from fedhealth.coordinator.orchestrator import ComputationAborted, ProtocolError
    from fedhealth.coordinator.release import plan_noise
    _, _, system, _ = svc
    spec = headline_spec()
    plan = plan_noise(spec, PrivacyParams(epsilon=1.0))
    real_post = system.orchestrator.clients["B"].post

    def fail_at(step):
        def post(path, **kw):
            if path.endswith(step):
                raise ProtocolError(f"B failed at {step}")
            return real_post(path, **kw)
        return post

    monkeypatch.setattr(system.orchestrator.clients["B"], "post", fail_at("/prepare"))
    with pytest.raises(ProtocolError) as before:
        system.orchestrator.mpc_run(spec, plan)
    assert not isinstance(before.value, ComputationAborted)                # nothing shared yet: refundable
    monkeypatch.setattr(system.orchestrator.clients["B"], "post", fail_at("/output"))
    with pytest.raises(ComputationAborted):
        system.orchestrator.mpc_run(spec, plan)


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


def test_gaussian_mechanism_is_accounted_in_zcdp(svc):
    service, ledger, _, _ = svc
    out = service.execute("alice", WIDE, PrivacyParams(epsilon=1.0, mechanism="gaussian"))
    assert out.status == "released" and out.delta_charged == 1e-6 and 0 < out.rho_charged < 0.5
    b = ledger.balance("alice")
    assert b.accountant == "zcdp" and b.spent_delta == pytest.approx(1e-4)          # converted at alice's budget delta
    assert b.spent_epsilon < 1.0                                                     # tighter than the epsilon asked for
