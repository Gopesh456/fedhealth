"""Differential-privacy noise mechanisms for INTEGER-valued queries (counts, sums of scaled integers).

DIFFERENTIAL PRIVACY IN ONE PARAGRAPH
    A randomised release M is (epsilon, delta)-DP if changing ONE patient (adding or removing them) changes the
    probability of any output by at most a factor e^epsilon (plus a tiny additive delta). The LAPLACE mechanism
    achieves (epsilon, 0)-DP by adding noise with density proportional to exp(-epsilon * |x| / Delta), where the
    SENSITIVITY Delta is the most one patient can change the true answer (1 for a count; the clamp bound for a sum).

WHY A *DISCRETE* LAPLACE
    Our answers are integers, so we add integer noise  P(X = k)  proportional to  exp(-epsilon * |k| / Delta)
    (the "two-sided geometric" distribution). It gives the same guarantee, and unlike `random.random()`-based
    continuous Laplace sampling it can be sampled EXACTLY with integer arithmetic - no floating point, hence none
    of the floating-point attacks on naive Laplace implementations (Mironov, 2012). We use the sampler of
    Canonne, Kamath & Steinke, "The Discrete Gaussian for Differential Privacy" (2020), Algorithms 1-2:
    everything reduces to Bernoulli(p) draws with RATIONAL p, and  Bernoulli(n/d) = [randbelow(d) < n].

GAUSSIAN
    The DISCRETE Gaussian N_Z(0, sigma^2), P(X = x) proportional to exp(-x^2 / (2 sigma^2)) over the integers, sampled
    EXACTLY with the rejection sampler of CKS20 (Algorithm 3: a discrete-Laplace proposal accepted with a rational
    Bernoulli(exp(-gamma))). It satisfies Delta^2 / (2 sigma^2)-zCDP (CKS20, Theorem 4), which is what the zCDP ledger
    charges. `gaussian_sigma` / `gaussian_delta` (the analytic Gaussian of Balle & Wang, 2018) are kept only as a
    reference point for the evaluation.

DISTRIBUTED NOISE (used by the MPC, see privacy/dp/distributed.py)
    Every hospital adds one noise SHARE inside the secure computation, so nobody - not even the coordinator - ever
    sees an un-noised aggregate. The shares are built so that the noise contributed by any h honest hospitals is
    already enough on its own:
      * Laplace: a share is  Polya(1/h) - Polya(1/h). Polya (negative binomial with a real "number of successes")
        is infinitely divisible: h shares sum to Geometric - Geometric, i.e. EXACTLY the discrete Laplace.
        Sampled exactly by rejection from a geometric proposal (`sample_polya`).
      * Gaussian: a share is N_Z(0, sigma^2 / h). Sums of discrete Gaussians are not exactly discrete Gaussian; the
        privacy loss of the sum is bounded by Kairouz, Liu & Steinke (ICML 2021), see distributed.gaussian_rho.
"""
from __future__ import annotations

import math
import secrets
from fractions import Fraction
from typing import Callable

RandBelow = Callable[[int], int]      # randbelow(n) -> uniform integer in [0, n)


# ----------------------------------------------------------------------------- exact Bernoulli / discrete Laplace
def _bernoulli(p: Fraction, randbelow: RandBelow) -> bool:
    """Bernoulli(p) for rational p in [0, 1], exactly."""
    return randbelow(p.denominator) < p.numerator


def _bernoulli_exp_neg(gamma: Fraction, randbelow: RandBelow) -> bool:
    """Bernoulli(exp(-gamma)) for rational gamma >= 0, exactly (CKS20, Algorithm 1)."""
    while gamma > 1:                                   # exp(-g) = exp(-1)^floor(g) * exp(-(g - floor g))
        if not _bernoulli_exp_neg(Fraction(1), randbelow):
            return False
        gamma -= 1
    k = 1                                              # 0 <= gamma <= 1:  alternating-series trick
    while _bernoulli(gamma / k, randbelow):
        k += 1
    return k % 2 == 1


def sample_discrete_laplace(s: int, t: int, randbelow: RandBelow = secrets.randbelow) -> int:
    """Exact sample of P(X = x) ~ exp(-|x| * s / t)  (CKS20, Algorithm 2). s, t positive integers."""
    if s <= 0 or t <= 0:
        raise ValueError("s and t must be positive")
    while True:
        u = randbelow(t)
        if not _bernoulli_exp_neg(Fraction(u, t), randbelow):
            continue
        v = 0
        while _bernoulli_exp_neg(Fraction(1), randbelow):
            v += 1
        y = (u + t * v) // s
        negative = randbelow(2) == 1
        if negative and y == 0:                        # avoid double-counting zero
            continue
        return -y if negative else y


def sample_geometric(s: int, t: int, randbelow: RandBelow = secrets.randbelow) -> int:
    """Exact sample of P(Y = y) proportional to exp(-y * s / t), y = 0, 1, 2, ...  (the one-sided half of CKS20, Alg. 2)."""
    if s <= 0 or t <= 0:
        raise ValueError("s and t must be positive")
    while True:
        u = randbelow(t)
        if not _bernoulli_exp_neg(Fraction(u, t), randbelow):
            continue
        v = 0
        while _bernoulli_exp_neg(Fraction(1), randbelow):
            v += 1
        return (u + t * v) // s


def sample_polya(r: Fraction, s: int, t: int, randbelow: RandBelow = secrets.randbelow) -> int:
    """Exact sample of the Polya / negative-binomial law NB(r, a), a = exp(-s/t), 0 < r <= 1:
        P(K = k) = C(k + r - 1, k) (1 - a)^r a^k.
    Rejection from the geometric proposal (1 - a) a^k: the ratio target/proposal is proportional to
        prod_{i=1..k} (i - 1 + r) / i   <= 1,
    a product of RATIONAL Bernoulli probabilities, tested one factor at a time (early exit on the first failure).
    Acceptance rate (1 - a)^(1 - r): r = 1 is the geometric itself."""
    r = Fraction(r)
    if not 0 < r <= 1:
        raise ValueError("r must be in (0, 1]")
    num, den = r.numerator, r.denominator
    while True:
        k = sample_geometric(s, t, randbelow)
        if r == 1:
            return k
        i = 1                       # factor i is (i - 1 + r)/i = (den*(i-1) + num) / (den*i): integer-only Bernoulli
        while i <= k and randbelow(den * i) < den * (i - 1) + num:
            i += 1
        if i > k:
            return k


def sample_discrete_gaussian(sigma2: Fraction, randbelow: RandBelow = secrets.randbelow) -> int:
    """Exact sample of N_Z(0, sigma2) for a RATIONAL variance sigma2 > 0 (CKS20, Algorithm 3)."""
    sigma2 = Fraction(sigma2)
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    t = math.isqrt(sigma2.numerator // sigma2.denominator) + 1          # floor(sigma) + 1
    while True:
        y = sample_discrete_laplace(1, t, randbelow)                     # proposal  P(y) ~ exp(-|y| / t)
        gamma = (abs(y) - sigma2 / t) ** 2 / (2 * sigma2)
        if _bernoulli_exp_neg(gamma, randbelow):
            return y


def _conservative_epsilon(epsilon: float, max_denominator: int = 10_000) -> Fraction:
    """Nearest rational to epsilon, never ABOVE it (rounding epsilon up would silently weaken the guarantee)."""
    exact = Fraction(str(epsilon))                     # the decimal the caller meant: 0.7 -> 7/10, not the binary float
    frac = exact.limit_denominator(max_denominator)
    if frac > exact:
        frac -= Fraction(1, max_denominator * 10)
    if frac <= 0:
        raise ValueError("epsilon too small to represent")
    return frac


# ----------------------------------------------------------------------------- the mechanisms
class LaplaceMechanism:
    """epsilon-DP for an integer query with sensitivity `sensitivity` (a positive integer)."""
    name = "laplace"

    def __init__(self, epsilon: float, sensitivity: int, randbelow: RandBelow = secrets.randbelow):
        if epsilon <= 0 or sensitivity <= 0 or int(sensitivity) != sensitivity:
            raise ValueError("epsilon > 0 and an integer sensitivity >= 1 are required")
        self.epsilon, self.sensitivity, self.randbelow = float(epsilon), int(sensitivity), randbelow
        eps = _conservative_epsilon(epsilon)
        # P(X=k) ~ exp(-|k| * epsilon / Delta) = exp(-|k| * s / t)  with  s/t = eps / Delta
        self._s, self._t = eps.numerator, eps.denominator * self.sensitivity

    def noise(self) -> int:
        return sample_discrete_laplace(self._s, self._t, self.randbelow)

    def release(self, value: int) -> int:
        return int(value) + self.noise()

    @property
    def scale(self) -> float:
        return self.sensitivity / self.epsilon

    @property
    def expected_abs_error(self) -> float:
        a = math.exp(-self.epsilon / self.sensitivity)           # E|X| of the two-sided geometric
        return 2 * a / ((1 - a) * (1 + a))

    @property
    def delta(self) -> float:
        return 0.0


def gaussian_delta(sigma: float, epsilon: float, sensitivity: float) -> float:
    """Exact delta of the Gaussian mechanism N(0, sigma^2) at a given epsilon (Balle & Wang 2018, Thm 8)."""
    def phi(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2))
    a = sensitivity / (2 * sigma)
    b = epsilon * sigma / sensitivity
    return phi(a - b) - math.exp(epsilon) * phi(-a - b)


def gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """Smallest sigma (by bisection) with gaussian_delta(sigma) <= delta."""
    lo, hi = 1e-9 * sensitivity, 1.0 * sensitivity
    while gaussian_delta(hi, epsilon, sensitivity) > delta:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if gaussian_delta(mid, epsilon, sensitivity) > delta:
            lo = mid
        else:
            hi = mid
    return hi
