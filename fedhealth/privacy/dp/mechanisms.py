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
    (epsilon, delta)-DP with N(0, sigma^2), sigma calibrated by the ANALYTIC Gaussian mechanism (Balle & Wang,
    2018), valid for every epsilon (the textbook sigma = Delta*sqrt(2 ln(1.25/delta))/epsilon only works for
    epsilon < 1). The sample is a floating-point Gaussian rounded to an integer: rounding is post-processing, but
    the floating-point caveat remains - documented in THREAT_MODEL.md.
"""
from __future__ import annotations

import math
import random
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


class GaussianMechanism:
    """(epsilon, delta)-DP via the analytic Gaussian mechanism; output rounded to an integer."""
    name = "gaussian"

    def __init__(self, epsilon: float, delta: float, sensitivity: int, rng: random.Random | None = None):
        if epsilon <= 0 or not 0 < delta < 1 or sensitivity <= 0:
            raise ValueError("need epsilon > 0, 0 < delta < 1, sensitivity > 0")
        self.epsilon, self.delta, self.sensitivity = float(epsilon), float(delta), int(sensitivity)
        self.sigma = gaussian_sigma(self.epsilon, self.delta, self.sensitivity)
        self._rng = rng or random.SystemRandom()

    def noise(self) -> int:
        return int(round(self._rng.gauss(0.0, self.sigma)))

    def release(self, value: int) -> int:
        return int(value) + self.noise()

    @property
    def scale(self) -> float:
        return self.sigma

    @property
    def expected_abs_error(self) -> float:
        return self.sigma * math.sqrt(2 / math.pi)


class NoNoise:
    """The identity 'mechanism': NO privacy. Exists only so the attack demos can show what the k-threshold alone
    (suppression without noise) fails to stop. It cannot be selected through the API (PrivacyParams does not
    accept it); only an explicit ReleasePolicy(dp_enabled=False) in evaluation code turns it on."""
    name = "none"
    scale = 0.0
    expected_abs_error = 0.0
    delta = 0.0

    def release(self, value: int) -> int:
        return int(value)


def make_mechanism(name: str, epsilon: float, sensitivity: int, delta: float = 0.0,
                   randbelow: RandBelow = secrets.randbelow):
    if name == "none":
        return NoNoise()
    if name == "laplace":
        return LaplaceMechanism(epsilon, sensitivity, randbelow)
    if name == "gaussian":
        return GaussianMechanism(epsilon, delta, sensitivity)
    raise ValueError(f"unknown mechanism {name!r}")
