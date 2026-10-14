"""Public response deadlines (fedhealth/hospital/deadlines.py): the coordinator cannot time what a hospital computed."""
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict

import pytest

from fedhealth.common.schemas import QuerySpec
from fedhealth.coordinator.release import PrivacyParams, plan_noise
from fedhealth.coordinator.service import QueryService
from fedhealth.hospital.deadlines import ResponseDeadlines, answer_no_earlier_than, load_response_deadlines
from fedhealth.privacy.dp.distributed import NoisePolicy, make_plan
from fedhealth.privacy.federated.features import DIM
from fedhealth.synth.generate import generate

from .helpers import headline_spec
from .system import build_system

FAST = "test768"          # insecure 768-bit PSI group, tests only
SLACK = 0.25              # scheduling jitter allowed on top of a deadline in these wall-clock tests


def test_an_answer_waits_for_its_deadline_and_an_overrun_shows_only_its_doubling():
    overruns = Counter()
    for work, answered_at in ((0.0, 0.08), (0.1, 0.16), (0.2, 0.32)):
        t0 = time.monotonic()
        with answer_no_earlier_than(0.08, overruns, "step"):
            time.sleep(work)
        assert answered_at <= time.monotonic() - t0 < answered_at + SLACK / 2
    assert overruns["step"] == 2                                   # counted at the hospital, never reported
    t0 = time.monotonic()
    with pytest.raises(RuntimeError), answer_no_earlier_than(0.08):
        raise RuntimeError("refused half-way")                     # an error answer is held just the same
    assert time.monotonic() - t0 >= 0.08


def test_the_prepare_deadline_depends_on_the_public_plan_only(monkeypatch):
    d = ResponseDeadlines()
    count1 = plan_noise(headline_spec(), PrivacyParams(epsilon=1.0))
    count01 = plan_noise(headline_spec(), PrivacyParams(epsilon=0.1))
    total = plan_noise(headline_spec(agg="sum", value_field="lab_delta"), PrivacyParams(epsilon=1.0))
    gauss = plan_noise(headline_spec(), PrivacyParams(epsilon=1.0, mechanism="gaussian"))
    assert d.prepare(None, 1) == pytest.approx(d.mpc_prepare + d.noise_per_cell)
    assert d.prepare(None, 1) < d.prepare(count1, 1) < d.prepare(count01, 1) < d.prepare(total, 2)
    assert d.prepare(gauss, 1) == d.prepare(None, 1)              # the discrete Gaussian sampler's work does not grow
    assert ResponseDeadlines().scaled(0).prepare(total, 2) == 0
    monkeypatch.setenv("FEDHEALTH_DEADLINE_SCALE", "1")
    assert load_response_deadlines() == ResponseDeadlines()       # configs/policy.yaml and the defaults agree
    monkeypatch.setenv("FEDHEALTH_DEADLINE_SCALE", "2")
    assert load_response_deadlines().mpc_prepare == 2 * ResponseDeadlines().mpc_prepare


def test_noise_too_large_to_draw_in_bounded_time_is_refused_on_both_sides():
    pol = NoisePolicy(tolerance=1, max_epsilon_per_query=2.0, max_delta_per_query=1e-4)
    make_plan("laplace", 0.004, 0.0, [(1, 1), (1, 500)], 3, 1).check(pol, "sum", 2)          # scale 250,000: fine
    with pytest.raises(ValueError, match="scale"):
        make_plan("laplace", 0.001, 0.0, [(1, 1), (1, 500)], 3, 1).check(pol, "sum", 2)      # 1,000,000 > 500,000
    make_plan("gaussian", 0.001, 1e-6, [(1, 1), (1, 500)], 3, 1)                              # Gaussian: no cap needed
    spec = headline_spec(agg="sum", value_field="lab_delta")
    out = QueryService(None, None, noise_policy=pol).execute("u", spec, PrivacyParams(epsilon=0.001))
    assert out.status == "refused" and "scale" in out.reason and out.epsilon_charged == 0


@pytest.fixture(scope="module")
def timed(tmp_path_factory):
    d = tmp_path_factory.mktemp("deadline_world")
    generate(600, seed=5, out_dir=d)
    deadlines = ResponseDeadlines(psi_init=0.3, psi_finalize=0.2, mpc_prepare=0.5, noise_per_cell=0.0,
                                  noise_per_scale=0.0, fl_train_per_epoch=0.3)
    system = build_system(d, psi_group=FAST, psi_groups=(FAST,), deadlines=deadlines)
    seen: dict[str, list[float]] = defaultdict(list)

    class Timed:                                                   # what the coordinator can measure: each round trip
        def __init__(self, inner):
            self.inner = inner

        def post(self, path, **kw):
            t0 = time.monotonic()
            resp = self.inner.post(path, **kw)
            seen[path.rsplit("/", 1)[-1]].append(time.monotonic() - t0)
            return resp

        def delete(self, path, **kw):
            return self.inner.delete(path, **kw)

        def get(self, path, **kw):
            return self.inner.get(path, **kw)

    orch = system.orchestrator
    orch.clients = {h: Timed(c) for h, c in orch.clients.items()}
    return system, deadlines, seen


def test_the_coordinator_sees_the_deadline_not_how_many_patients_matched(timed):
    system, deadlines, seen = timed
    narrow = headline_spec()                                                         # a handful of patients
    broad = QuerySpec.model_validate({"agg": "count", "condition": "E11", "a": {}, "b": {}, "c": {}})   # all of them
    for spec in (narrow, broad, narrow, broad):
        system.orchestrator.mpc_run(spec, plan_noise(spec, PrivacyParams(epsilon=1.0)))
    assert len(seen["init"]) == 3 and len(seen["finalize"]) == 3
    assert all(deadlines.psi_init <= t < deadlines.psi_init + SLACK for t in seen["init"])
    assert all(deadlines.psi_finalize <= t < deadlines.psi_finalize + SLACK for t in seen["finalize"])
    assert len(seen["prepare"]) == 12
    assert all(deadlines.mpc_prepare <= t < deadlines.mpc_prepare + SLACK for t in seen["prepare"])
    assert not any(system.apps[h].state.deadline_overruns for h in "ABC")


def test_federated_training_answers_at_its_deadline(timed):
    system, deadlines, _ = timed
    body = {"weights": [0.0] * DIM, "parties": ["A", "B", "C"], "round_no": 1, "epochs": 2, "lr": 0.1, "batch": 32,
            "l2": 0.0, "clip": 1.0, "noise_multiplier": 0.0, "seed": 7}
    t0 = time.monotonic()
    assert system.raw["A"].post(f"/fl/{'f1' * 8}/train", json=body).status_code == 200
    assert deadlines.fl_train(2) <= time.monotonic() - t0 < deadlines.fl_train(2) + SLACK
    system.raw["A"].delete(f"/fl/{'f1' * 8}")


def test_nothing_else_runs_at_a_hospital_during_a_guarded_window(timed):
    """Timing another request (here /health) while a guarded call works would reveal how long the work took: every
    other request waits for the window to close, and a second guarded call queues behind the first."""
    system, deadlines, _ = timed
    a = system.raw["A"]
    body = {"weights": [0.0] * DIM, "parties": ["A", "B", "C"], "round_no": 1, "epochs": 1, "lr": 0.1, "batch": 32,
            "l2": 0.0, "clip": 1.0, "noise_multiplier": 0.0, "seed": 7}
    window = deadlines.fl_train(1)
    done: dict[str, float] = {}

    def train(sid):
        assert a.post(f"/fl/{sid}/train", json=body).status_code == 200
        done[sid] = time.monotonic()

    t0 = time.monotonic()
    first = threading.Thread(target=train, args=("e1" * 8,))
    first.start()
    time.sleep(window / 4)
    second = threading.Thread(target=train, args=("e2" * 8,))
    second.start()
    time.sleep(window / 8)
    assert a.get("/health").status_code == 200
    health_done = time.monotonic() - t0
    first.join()
    second.join()
    assert health_done >= window - 0.02                       # the probe waited for the first window to close
    assert done["e2" * 8] - t0 >= 2 * window - 0.02            # the second guarded call ran after it, alone
    for sid in ("e1" * 8, "e2" * 8):
        a.delete(f"/fl/{sid}")
