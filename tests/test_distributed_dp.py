"""Distributed DP noise (generated in shares by the hospitals, inside the MPC) and the zCDP accountant."""
from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest

from fedhealth.coordinator.release import PrivacyParams, ReleasePolicy, plan_noise
from fedhealth.privacy.dp.accounting import dp_to_zcdp, pure_dp_rho, queries_within_budget, zcdp_to_dp
from fedhealth.privacy.dp.distributed import NoisePolicy, gaussian_rho, make_plan
from fedhealth.privacy.dp.mechanisms import sample_discrete_gaussian, sample_geometric, sample_polya

from .helpers import headline_spec


def seeded(seed: int):
    return random.Random(seed).randrange


def chi2(xs: list[int], pmf, lo: int, hi: int) -> float:
    n = len(xs)
    obs = [sum(1 for x in xs if x == k) for k in range(lo, hi + 1)]
    probs = [pmf(k) for k in range(lo, hi + 1)]
    obs.append(n - sum(obs))
    probs.append(1 - sum(probs))
    return sum((o - n * p) ** 2 / (n * p) for o, p in zip(obs, probs) if n * p > 0)


# ============================================================================ exact samplers
def test_geometric_sampler_matches_its_pmf():
    rb, s, t = seeded(1), 1, 2                                       # ratio a = exp(-1/2)
    a = math.exp(-s / t)
    xs = [sample_geometric(s, t, rb) for _ in range(20_000)]
    assert chi2(xs, lambda k: (1 - a) * a ** k, 0, 10) < 31.3       # df = 11, p = 0.001


@pytest.mark.parametrize("r", [Fraction(1, 2), Fraction(1, 3)])
def test_polya_sampler_matches_the_negative_binomial(r):
    rb, s, t = seeded(2), 1, 1
    a = math.exp(-s / t)
    rr = float(r)
    pmf = lambda k: math.exp(math.lgamma(k + rr) - math.lgamma(rr) - math.lgamma(k + 1)) * (1 - a) ** rr * a ** k  # noqa: E731
    xs = [sample_polya(r, s, t, rb) for _ in range(30_000)]
    assert chi2(xs, pmf, 0, 6) < 24.3                                # df = 7, p = 0.001


def test_h_polya_shares_sum_to_exactly_the_discrete_laplace():
    """Infinite divisibility: two Polya(1/2)-differences add up to one discrete Laplace - not approximately."""
    rb = seeded(3)
    plan = make_plan("laplace", 1.0, 0.0, [(1, 1)], contributors=2, tolerance=1)
    assert plan.honest == 1                                          # r = 1: each share alone is a full Laplace
    half = make_plan("laplace", 1.0, 0.0, [(1, 1)], contributors=2, tolerance=0)
    assert half.honest == 2
    xs = [half.sample_share(rb)[0] + half.sample_share(rb)[0] for _ in range(30_000)]
    a = math.exp(-1.0)
    assert chi2(xs, lambda k: (1 - a) / (1 + a) * a ** abs(k), -5, 5) < 32.9   # df = 11, p = 0.0005


def test_discrete_gaussian_sampler_has_the_right_variance_and_shape():
    rb = seeded(4)
    sigma2 = Fraction(9, 1)
    xs = [sample_discrete_gaussian(sigma2, rb) for _ in range(20_000)]
    var = sum(x * x for x in xs) / len(xs)
    assert abs(sum(xs) / len(xs)) < 0.1 and var == pytest.approx(9.0, rel=0.05)
    norm = sum(math.exp(-k * k / 18) for k in range(-60, 61))
    assert chi2(xs, lambda k: math.exp(-k * k / 18) / norm, -6, 6) < 34.5


# ============================================================================ accounting
def test_zcdp_conversion_is_tighter_than_the_textbook_bound_and_invertible():
    for rho in (0.001, 0.05, 0.5, 2.0):
        eps = zcdp_to_dp(rho, 1e-6)
        assert rho < eps <= rho + 2 * math.sqrt(rho * math.log(1e6))
        assert dp_to_zcdp(eps, 1e-6) == pytest.approx(rho, rel=1e-6)
    assert pure_dp_rho(1.0) == 0.5


def test_queries_within_budget_shows_where_zcdp_wins():
    small = queries_within_budget(pure_dp_rho(0.1), 0.1, 0.0, 10.0, 1e-6)
    big = queries_within_budget(pure_dp_rho(1.0), 1.0, 0.0, 10.0, 1e-6)
    assert small["zcdp"] > 3 * small["basic"] and big["basic"] > big["zcdp"]


def test_distributed_gaussian_rho_matches_the_central_one_up_to_tau():
    share = Fraction(50)
    central = 1 / (2 * 100)                                          # Delta = 1, two honest shares -> sigma^2 = 100
    assert gaussian_rho(1, share, 2, 1) == pytest.approx(central, rel=1e-9)
    assert gaussian_rho(1, Fraction(1), 3, 1) > 1 / (2 * 3)          # tiny shares: the tau term is charged


# ============================================================================ plans
@pytest.mark.parametrize("mech", ["laplace", "gaussian"])
def test_tolerating_more_colluders_costs_variance_not_privacy(mech):
    spec, pp = headline_spec(), PrivacyParams(epsilon=1.0, mechanism=mech)
    plans = [plan_noise(spec, pp, ReleasePolicy(tolerance=t)) for t in (0, 1, 2)]
    assert [p.honest for p in plans] == [3, 2, 1]
    var = [p.total_variance(0) for p in plans]
    assert var[0] < var[1] < var[2] and var[2] == pytest.approx(3 * var[0], rel=0.02)   # x c/h
    for p in plans:
        if mech == "laplace":
            assert p.rho == 0.5
        assert zcdp_to_dp(p.rho, 1e-6) <= 1.0 + 1e-6 or mech == "laplace"


def test_single_hospital_query_has_one_contributor_with_the_full_noise():
    plan = plan_noise(headline_spec(a={}, b=None, c=None), PrivacyParams(epsilon=1.0))
    assert plan.contributors == plan.honest == 1


def test_hospital_side_check_refuses_weak_plans():
    pol = NoisePolicy(tolerance=1, max_epsilon_per_query=2.0, max_delta_per_query=1e-4)
    make_plan("laplace", 2.0, 0.0, [(1, 1)], 3, 1).check(pol, "count", 1)
    make_plan("laplace", 2.0, 0.0, [(1, 1), (1, 500)], 3, 1).check(pol, "sum", 2)            # SUM: 1 + 1 = 2
    with pytest.raises(ValueError):
        make_plan("laplace", 4.0, 0.0, [(1, 1)], 3, 1).check(pol, "count", 1)
    with pytest.raises(ValueError):
        make_plan("laplace", 2.5, 0.0, [(1, 1), (1, 500)], 3, 1).check(pol, "sum", 2)        # rho 1.56 < 2, but epsilon 2.5 > 2
    make_plan("gaussian", 2.0, 1e-4, [(1, 1)], 3, 1).check(pol, "count", 1)
    with pytest.raises(ValueError):
        make_plan("gaussian", 2.0, 1e-3, [(1, 1)], 3, 1).check(pol, "count", 1)                # (2, 1e-3) is weaker than (2, 1e-4)
    with pytest.raises(ValueError):
        make_plan("laplace", 1.0, 0.0, [(1, 1)], 3, 0).check(pol, "count", 1)                  # needs all 3 shares honest: refused
    make_plan("laplace", 1.0, 0.0, [(1, 1)], 3, 2).check(pol, "count", 1)                      # any 1 share suffices: stronger, fine
    cautious = NoisePolicy(tolerance=2, max_epsilon_per_query=2.0, max_delta_per_query=1e-4)
    with pytest.raises(ValueError):
        make_plan("laplace", 1.0, 0.0, [(1, 1)], 3, 1).check(cautious, "count", 1)             # it assumes only itself honest
    with pytest.raises(ValueError):
        make_plan("none", 1.0, 0.0, [(1, 1)], 3, 1).check(pol, "count", 1)


def test_hospitals_refuse_a_plan_that_leaves_a_cell_outside_the_budget():
    """A SUM plan whose groups cover only the count cell would let the total through with almost no noise while the
    recomputed epsilon (summed over groups) stays under the cap. The groups are fixed by the aggregate, so it fails."""
    from dataclasses import replace
    from fedhealth.privacy.dp.distributed import CellNoise
    pol = NoisePolicy(tolerance=1, max_epsilon_per_query=2.0, max_delta_per_query=1e-4)
    honest = make_plan("laplace", 1.0, 0.0, [(1, 1), (1, 500)], 3, 1)
    honest.check(pol, "sum", 2)
    forged = replace(honest, cells=(honest.cells[0], CellNoise(500, s=10 ** 9, t=1)), groups=((0, 1),))
    assert forged.recomputed_epsilon() <= 2.0                     # the old check alone would have passed it
    with pytest.raises(ValueError, match="groups"):
        forged.check(pol, "sum", 2)
    with pytest.raises(ValueError, match="groups"):
        honest.check(pol, "histogram", 2)                         # the same plan presented as a 2-bin histogram
    with pytest.raises(ValueError, match="groups"):
        honest.check(pol, "count", 1)


@pytest.mark.parametrize("mech", ["laplace", "gaussian"])
def test_vectorised_simulation_matches_the_exact_samplers(mech):
    """The evaluation draws millions of releases with numpy; the services use the exact samplers. Same law."""
    plan = make_plan(mech, 0.5, 1e-6, [(1, 1)], 3, 1)
    rb = seeded(5)
    exact = [sum(plan.sample_share(rb)[0] for _ in range(3)) for _ in range(6000)]
    sim = plan.simulate_total(60_000, np.random.default_rng(5))[:, 0]
    assert np.var(exact) == pytest.approx(plan.total_variance(0), rel=0.08)
    assert np.var(sim) == pytest.approx(plan.total_variance(0), rel=0.03)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert abs(np.quantile(exact, q) - np.quantile(sim, q)) <= max(1.0, 0.1 * plan.std(0))
