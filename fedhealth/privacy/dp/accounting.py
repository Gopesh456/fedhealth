"""Privacy accounting with zero-concentrated DP (zCDP), plus the conversions to and from (epsilon, delta).

WHY zCDP
    Basic composition adds epsilons: k queries at epsilon each cost k * epsilon. That is tight for a handful of large
    queries but very loose for many small ones. zCDP (Bun & Steinke, TCC 2016) measures privacy loss as one number
    rho that ADDS UP exactly under composition, and turns into an (epsilon, delta) guarantee only at the end:
        k queries at rho each  ->  k * rho  ->  epsilon ~ k*rho + 2*sqrt(k*rho*log(1/delta)),   i.e. O(sqrt(k)).
    Mechanisms:
        * the (discrete) Gaussian with l2-sensitivity Delta and variance sigma^2 is  Delta^2 / (2 sigma^2) - zCDP
          (CKS20, Theorem 4);
        * any pure epsilon-DP mechanism (our discrete Laplace) is  epsilon^2 / 2 - zCDP  (Bun & Steinke, Prop. 1.4).

CONVERSION  (Canonne, Kamath & Steinke 2020, Corollary 13 - tighter than the textbook rho + 2 sqrt(rho log 1/delta))
    rho-zCDP  =>  (epsilon, delta)-DP  with
        epsilon(rho, delta) = inf_{alpha > 1}  alpha*rho + [ log(1/delta) + (alpha-1) log(1 - 1/alpha) - log(alpha) ] / (alpha - 1)

The ledger (ledger.py) keeps BOTH this zCDP account and the basic-composition account and admits a query when
either one stays within the requester's (epsilon, delta) budget: both are valid bounds on the same transcript, so
the guarantee is the tighter of the two.
"""
from __future__ import annotations

import math


def zcdp_to_dp(rho: float, delta: float) -> float:
    """Smallest epsilon such that rho-zCDP implies (epsilon, delta)-DP (CKS20 Cor. 13, minimised over alpha)."""
    if rho < 0 or not 0 < delta < 1:
        raise ValueError("need rho >= 0 and 0 < delta < 1")
    if rho == 0:
        return 0.0
    log_inv_delta = math.log(1 / delta)

    def eps_at(log_am1: float) -> float:                        # parametrised by log(alpha - 1) for a smooth search
        am1 = math.exp(log_am1)
        alpha = am1 + 1
        return alpha * rho + (log_inv_delta + am1 * math.log1p(-1 / alpha) - math.log(alpha)) / am1

    # the objective is unimodal in log(alpha - 1): golden-section search on a wide bracket
    lo, hi = -20.0, 25.0
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = eps_at(a), eps_at(b)
    for _ in range(200):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = eps_at(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = eps_at(b)
    simple = rho + 2 * math.sqrt(rho * log_inv_delta)          # the classic bound, as a safety net
    return max(0.0, min(fa, fb, simple))


def dp_to_zcdp(epsilon: float, delta: float) -> float:
    """Largest rho whose conversion is still (epsilon, delta)-DP: the zCDP budget equivalent to an (epsilon, delta)
    budget. Used to size a requester's account and to calibrate a Gaussian query that was asked for in (epsilon, delta)."""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("need epsilon > 0 and 0 < delta < 1")
    lo, hi = 0.0, epsilon                                       # zcdp_to_dp(rho) >= rho, so rho <= epsilon
    for _ in range(200):
        mid = (lo + hi) / 2
        if zcdp_to_dp(mid, delta) <= epsilon:
            lo = mid
        else:
            hi = mid
    return lo


def pure_dp_rho(epsilon: float) -> float:
    """epsilon-DP  =>  epsilon^2/2 - zCDP."""
    return epsilon * epsilon / 2


def basic_composition(epsilons: list[float], deltas: list[float]) -> tuple[float, float]:
    return sum(epsilons), sum(deltas)


def queries_within_budget(per_query_rho: float, per_query_eps: float, per_query_delta: float,
                          total_eps: float, total_delta: float) -> dict[str, int]:
    """How many identical queries fit in an (epsilon, delta) budget under each accountant (for the evaluation).
    Mirrors the ledger: basic composition counts pure-DP queries only (a delta > 0 query goes through zCDP)."""
    basic = int(total_eps / per_query_eps + 1e-9) if per_query_delta == 0 else 0
    zcdp = int(dp_to_zcdp(total_eps, total_delta) / per_query_rho + 1e-9) if total_delta > 0 else 0
    return {"basic": basic, "zcdp": zcdp, "ledger": max(basic, zcdp)}
