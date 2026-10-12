"""Phase 4a: noise mechanisms, the epsilon ledger, the k-threshold and the release stage (no services)."""
from __future__ import annotations

import math
import random
import threading
from fractions import Fraction

import pytest
from pydantic import ValidationError

from fedhealth.common.schemas import QuerySpec
from fedhealth.coordinator.release import Exact, PrivacyParams, ReleasePolicy, release
from fedhealth.privacy.dp import mechanisms
from fedhealth.privacy.dp.ledger import BudgetExhausted, EpsilonLedger, make_engine
from fedhealth.privacy.dp.mechanisms import (GaussianMechanism, LaplaceMechanism, gaussian_delta, gaussian_sigma)
from fedhealth.privacy.suppression import is_suppressed, suppress_cells

from .helpers import headline_spec


def seeded(seed: int):
    return random.Random(seed).randrange


def pmf(k: int, eps: float, sens: int) -> float:
    a = math.exp(-eps / sens)
    return (1 - a) / (1 + a) * a ** abs(k)


# ============================================================================ discrete Laplace
@pytest.mark.parametrize("eps,sens", [(0.1, 1), (0.5, 1), (1.0, 1), (2.0, 1), (5.0, 1), (0.5, 5), (1.0, 120)])
def test_discrete_laplace_matches_the_theoretical_distribution(eps, sens):
    n = 20_000
    mech = LaplaceMechanism(eps, sens, randbelow=seeded(int(eps * 100) + sens))
    xs = [mech.noise() for _ in range(n)]
    a = math.exp(-eps / sens)
    var_theory = 2 * a / (1 - a) ** 2
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    assert abs(mean) < 5 * math.sqrt(var_theory / n)                       # unbiased
    assert var == pytest.approx(var_theory, rel=0.08)                      # variance of the two-sided geometric
    assert sum(abs(x) for x in xs) / n == pytest.approx(mech.expected_abs_error, rel=0.08)
    assert xs.count(0) / n == pytest.approx(pmf(0, eps, sens), abs=0.012)  # mass at zero
    # symmetric: as many positive as negative draws (binomial test, 5 sigma)
    pos, neg = sum(x > 0 for x in xs), sum(x < 0 for x in xs)
    assert abs(pos - neg) < 5 * math.sqrt(pos + neg)


def test_discrete_laplace_chi_square_goodness_of_fit():
    eps, sens, n = 1.0, 1, 30_000
    mech = LaplaceMechanism(eps, sens, randbelow=seeded(7))
    xs = [mech.noise() for _ in range(n)]
    bins = list(range(-4, 5))
    observed = [xs.count(k) for k in bins]
    probs = [pmf(k, eps, sens) for k in bins]
    observed.append(n - sum(observed))
    probs.append(1 - sum(probs))
    chi2 = sum((o - n * p) ** 2 / (n * p) for o, p in zip(observed, probs))
    assert chi2 < 29.6                      # chi-square critical value, df = 9, p = 0.0005


def test_the_exact_bernoulli_primitives():
    rb = seeded(3)
    n = 40_000
    for gamma in (Fraction(1, 2), Fraction(5, 2)):                          # gamma <= 1 and gamma > 1 (recursion) paths
        hits = sum(mechanisms._bernoulli_exp_neg(gamma, rb) for _ in range(n))
        assert hits / n == pytest.approx(math.exp(-float(gamma)), abs=0.01)
    assert sum(mechanisms._bernoulli(Fraction(1, 4), rb) for _ in range(n)) / n == pytest.approx(0.25, abs=0.01)


def test_epsilon_is_never_rounded_up():
    for eps in (0.1, 0.3, 0.7, 1.0, 0.123, 0.9999):
        m = LaplaceMechanism(eps, 3)
        assert Fraction(m._s, m._t) <= Fraction(str(eps)) / 3              # actual epsilon/Delta <= requested
        assert Fraction(m._s, m._t) > Fraction(str(eps)) / 3 * Fraction(99, 100)   # ...and not wastefully smaller


def test_default_randomness_comes_from_the_os_and_is_not_repeatable():
    a = [LaplaceMechanism(0.5, 1).noise() for _ in range(200)]
    b = [LaplaceMechanism(0.5, 1).noise() for _ in range(200)]
    assert a != b
    s1 = [LaplaceMechanism(0.5, 1, randbelow=seeded(1)).noise() for _ in range(50)]
    s2 = [LaplaceMechanism(0.5, 1, randbelow=seeded(1)).noise() for _ in range(50)]
    assert s1 == s2                                                         # ...but injectable for reproducible evaluation


@pytest.mark.parametrize("bad", [dict(epsilon=0, sensitivity=1), dict(epsilon=-1, sensitivity=1),
                                 dict(epsilon=1, sensitivity=0), dict(epsilon=1, sensitivity=1.5)])
def test_invalid_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        LaplaceMechanism(**bad)


# ============================================================================ Gaussian
@pytest.mark.parametrize("eps,delta", [(0.1, 1e-6), (0.5, 1e-6), (1.0, 1e-5), (2.0, 1e-6), (5.0, 1e-6)])
def test_analytic_gaussian_is_calibrated_and_tight(eps, delta):
    s = gaussian_sigma(eps, delta, 1)
    assert gaussian_delta(s, eps, 1) <= delta * (1 + 1e-6)                  # satisfies the guarantee ...
    assert gaussian_delta(s * 0.98, eps, 1) > delta                         # ... and 2% less noise would not
    if eps < 1:
        assert s <= math.sqrt(2 * math.log(1.25 / delta)) / eps             # never worse than the textbook bound


def test_gaussian_sigma_is_monotone_and_scales_with_sensitivity():
    sig = [gaussian_sigma(e, 1e-6, 1) for e in (0.25, 0.5, 1, 2, 4)]
    assert sig == sorted(sig, reverse=True)
    assert gaussian_sigma(1, 1e-6, 10) == pytest.approx(10 * gaussian_sigma(1, 1e-6, 1), rel=1e-3)


def test_gaussian_mechanism_empirical_std():
    m = GaussianMechanism(1.0, 1e-5, 1, rng=random.Random(5))
    xs = [m.noise() for _ in range(20_000)]
    std = (sum(x * x for x in xs) / len(xs)) ** 0.5
    assert std == pytest.approx(m.sigma, rel=0.05)
    with pytest.raises(ValueError):
        GaussianMechanism(1.0, 0.0, 1)                                      # delta must be > 0


# ============================================================================ epsilon ledger
@pytest.fixture()
def ledger(tmp_path):
    led = EpsilonLedger(make_engine(str(tmp_path / "coordinator.db")))
    led.ensure_user("alice", 1.0, 1e-5)
    return led


def test_reserve_commit_refund_accounting(ledger):
    ledger.reserve("alice", 0.4, "q1")
    assert ledger.balance("alice").remaining_epsilon == pytest.approx(0.6)
    ledger.commit("q1")
    ledger.reserve("alice", 0.5, "q2")
    ledger.refund("q2")                                                     # failed before release: budget returns
    assert ledger.balance("alice").remaining_epsilon == pytest.approx(0.6)
    assert [e["status"] for e in ledger.entries("alice")] == ["committed", "refunded"]


def test_budget_is_enforced_exactly_ten_times_a_tenth(ledger):
    for i in range(10):
        ledger.reserve("alice", 0.1, f"q{i}")                               # 10 x 0.1 == 1.0 exactly (integer micro-epsilons)
    assert ledger.balance("alice").remaining_epsilon == 0
    with pytest.raises(BudgetExhausted):
        ledger.reserve("alice", 0.000001, "one-too-many")
    assert ledger.balance("alice").spent_epsilon == 1.0


def test_exhausted_request_creates_no_entry_and_changes_nothing(ledger):
    ledger.reserve("alice", 0.9, "q1")
    with pytest.raises(BudgetExhausted) as exc:
        ledger.reserve("alice", 0.2, "q2")
    assert exc.value.remaining == pytest.approx(0.1)
    assert len(ledger.entries("alice")) == 1 and ledger.balance("alice").spent_epsilon == pytest.approx(0.9)


def test_lifecycle_misuse_is_rejected(ledger):
    ledger.reserve("alice", 0.1, "q1")
    ledger.commit("q1")
    with pytest.raises(KeyError):
        ledger.refund("q1")                                                 # a committed spend can never be undone
    with pytest.raises(KeyError):
        ledger.commit("q1")
    with pytest.raises(KeyError):
        ledger.commit("never-reserved")
    ledger.reserve("alice", 0.1, "q2")
    ledger.refund("q2")
    with pytest.raises(KeyError):
        ledger.refund("q2")                                                 # no double refund (would mint budget)
    with pytest.raises(ValueError):
        ledger.reserve("alice", 0, "q3")
    with pytest.raises(KeyError):
        ledger.reserve("nobody", 0.1, "q4")


def test_ensure_user_never_tops_up(ledger):
    ledger.reserve("alice", 0.5, "q1")
    ledger.ensure_user("alice", 100.0, 1.0)                                 # e.g. a re-login must not reset the budget
    assert ledger.balance("alice").total_epsilon == 1.0 and ledger.balance("alice").spent_epsilon == 0.5


def test_delta_budget_is_enforced_too(ledger):
    ledger.reserve("alice", 0.1, "q1", delta=6e-6)
    with pytest.raises(BudgetExhausted):
        ledger.reserve("alice", 0.1, "q2", delta=6e-6)                      # 12e-6 > 10e-6 although epsilon would fit


def test_users_have_independent_budgets(ledger):
    ledger.ensure_user("bob", 1.0)
    ledger.reserve("alice", 1.0, "qa")
    ledger.reserve("bob", 1.0, "qb")                                        # alice being broke does not affect bob
    with pytest.raises(BudgetExhausted):
        ledger.reserve("alice", 0.1, "qa2")


def test_concurrent_requests_can_never_overspend(ledger):
    """40 threads race for a budget that fits exactly 10 of them."""
    results, barrier = [], threading.Barrier(40)

    def worker(i: int) -> None:
        barrier.wait()
        try:
            ledger.reserve("alice", 0.1, f"race-{i}")
            results.append(True)
        except BudgetExhausted:
            results.append(False)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(results) == 10 and len(results) == 40
    assert ledger.balance("alice").spent_epsilon == pytest.approx(1.0)


def test_ledger_survives_a_restart(tmp_path):
    path = str(tmp_path / "c.db")
    led = EpsilonLedger(make_engine(path))
    led.ensure_user("alice", 2.0)
    led.reserve("alice", 0.75, "q1")
    led.commit("q1")
    again = EpsilonLedger(make_engine(path))
    assert again.balance("alice").spent_epsilon == 0.75


# ============================================================================ k-threshold
def test_k_threshold_boundary():
    assert is_suppressed(9) and not is_suppressed(10) and is_suppressed(0)
    assert not is_suppressed(5, k=5) and is_suppressed(4, k=5)
    assert suppress_cells({"a": 3, "b": 10, "c": 0, "d": 11}) == {"a": True, "b": False, "c": True, "d": False}


# ============================================================================ release stage
COUNT = headline_spec()
HIST = headline_spec(agg="histogram", group_by="outcome", b={})
SUM_AGE = headline_spec(agg="sum", value_field="age")
AVG_LAB = headline_spec(agg="avg", value_field="lab_delta")
P1 = PrivacyParams(epsilon=1.0)


def test_count_is_suppressed_below_k_and_released_at_k():
    rb = seeded(1)
    small = release(COUNT, Exact("count", count=9), P1, randbelow=rb)
    assert small.value is None and small.suppressed is True and small.fully_suppressed
    ok = release(COUNT, Exact("count", count=10), P1, randbelow=rb)
    assert isinstance(ok.value, int) and ok.value >= 0 and ok.suppressed is False


def test_released_object_never_carries_the_exact_value():
    r = release(COUNT, Exact("count", count=57), P1, randbelow=seeded(2))
    assert set(r.__dataclass_fields__) == {"agg", "value", "suppressed", "epsilon", "delta", "mechanism",
                                           "noise_scale", "expected_abs_error"}


def test_histogram_cells_are_suppressed_individually_and_cost_epsilon_once():
    r = release(HIST, Exact("histogram", cells={"improved": 3, "no_change": 50, "worsened": 0}), P1, randbelow=seeded(4))
    assert r.suppressed == {"improved": True, "no_change": False, "worsened": True}
    assert r.value["improved"] is None and r.value["worsened"] is None and r.value["no_change"] >= 0
    assert r.epsilon == 1.0 and r.noise_scale == 1.0 and not r.fully_suppressed          # one epsilon, Delta = 1 per cell
    none = release(HIST, Exact("histogram", cells={"a": 1, "b": 2}), P1, randbelow=seeded(4))
    assert none.fully_suppressed


def test_sum_uses_the_clamp_bound_as_sensitivity():
    r = release(SUM_AGE, Exact("sum", count=500, total=27_000), P1, randbelow=seeded(5))
    assert r.noise_scale == pytest.approx(120.0)                         # Delta = 120 (age bound), epsilon = 1
    lab = release(headline_spec(agg="sum", value_field="lab_delta"), Exact("sum", count=500, total=-30_000), P1, randbelow=seeded(5))
    assert lab.noise_scale == pytest.approx(4.0)                         # clamp 4.0 HbA1c points, back in natural units
    assert lab.expected_abs_error == pytest.approx(4.0, rel=0.01)


def test_sum_k_check_looks_at_the_patient_count_not_the_sum():
    tiny_cohort_big_sum = release(SUM_AGE, Exact("sum", count=4, total=250), P1, randbelow=seeded(6))
    assert tiny_cohort_big_sum.suppressed is True and tiny_cohort_big_sum.value is None


def test_avg_splits_epsilon_and_stays_within_bounds():
    r = release(AVG_LAB, Exact("avg", count=400, total=-60_000), PrivacyParams(epsilon=2.0), randbelow=seeded(7))
    assert r.noise_scale == pytest.approx(400 / 1.0 / 100)               # total gets epsilon/2 = 1.0: b = 400/1 -> 4.0 natural
    assert -4.0 <= r.value <= 4.0 and r.epsilon == 2.0
    # even absurd exact inputs are clamped back into the declared range (post-processing)
    wild = release(AVG_LAB, Exact("avg", count=400, total=10 ** 9), P1, randbelow=seeded(8))
    assert wild.value == 4.0
    assert release(AVG_LAB, Exact("avg", count=3, total=0), P1, randbelow=seeded(8)).suppressed is True


def test_counts_are_never_negative_even_with_huge_noise():
    rb = seeded(9)
    vals = [release(COUNT, Exact("count", count=10), PrivacyParams(epsilon=0.01), randbelow=rb).value for _ in range(300)]
    assert all(v >= 0 for v in vals) and len(set(vals)) > 50             # clamped at 0 but clearly noisy


def test_k_basis_noisy_is_post_processing_of_the_released_value():
    pol = ReleasePolicy(k=10, k_basis="noisy")
    rb = seeded(10)
    outcomes = {release(COUNT, Exact("count", count=10), PrivacyParams(epsilon=0.5), pol, rb).suppressed for _ in range(200)}
    assert outcomes == {True, False}                                      # at the boundary the decision itself is noisy ...
    exact_pol = ReleasePolicy(k=10, k_basis="exact")
    assert {release(COUNT, Exact("count", count=10), PrivacyParams(epsilon=0.5), exact_pol, rb).suppressed for _ in range(200)} == {False}
    assert release(COUNT, Exact("count", count=5000), P1, pol, rb).suppressed is False


def test_gaussian_release_reports_delta():
    r = release(COUNT, Exact("count", count=300), PrivacyParams(epsilon=1.0, mechanism="gaussian"), randbelow=seeded(11))
    assert r.mechanism == "gaussian" and r.delta == 1e-6 and r.noise_scale > 1.0
    assert release(COUNT, Exact("count", count=300), P1, randbelow=seeded(11)).delta == 0.0


@pytest.mark.parametrize("bad", [dict(epsilon=0), dict(epsilon=-1), dict(epsilon=11), dict(epsilon=1, mechanism="uniform"),
                                 dict(epsilon=1, extra=1), dict(epsilon=1, delta=0.5)])
def test_privacy_params_validation(bad):
    with pytest.raises(ValidationError):
        PrivacyParams(**bad)
