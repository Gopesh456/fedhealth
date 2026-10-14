"""The release stage: from the requester's privacy request to the number they are allowed to see.

Since the noise moved INSIDE the secure computation, the coordinator never holds an exact aggregate. Its role here is
(1) to PLAN the noise before the computation and (2) to post-process the already-noisy cells afterwards:

    plan_noise   (requester's epsilon / mechanism + the query's sensitivity  ->  per-cell noise parameters, rho)
        --> MPC  (every hospital adds one noise share as a secret-shared input; only  exact + noise  is opened)
        --> finalize:
              k-threshold   on the NOISY count (post-processing: covered by the DP guarantee, no side channel)
              post-process  clamp counts >= 0, divide for AVG, clamp a mean to its bounds (free under DP)
        --> Released     (the only thing that ever leaves the coordinator)

SENSITIVITY  (how much can ONE patient change the exact answer? - patients are added/removed)
    COUNT          1
    HISTOGRAM      1        every patient is in exactly ONE bin, so adding a patient changes ONE cell by 1:
                            all bins together cost the budget ONCE (l1 = l2 sensitivity = 1)
    SUM            B        B = max |value| after clamping to the declared bounds (value_bounds); a noisy COUNT is
                            released alongside (budget split in two) because the k-threshold needs it
    AVG            noisy SUM / noisy COUNT, each with half the budget (post-processing the ratio is free)
The clamping bounds are enforced hospital-side (predicate_runner._scaler), so B really bounds every contribution.

`release()` runs plan -> noise -> finalize in one process, drawing every hospital's share with the same exact
samplers the hospitals use. The attack demos and tests use it to replay many releases of one exact answer.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fedhealth.common.config import load_fields, load_topology
from fedhealth.common.schemas import QuerySpec, categories, participating_hospitals, value_bounds
from fedhealth.privacy.dp.distributed import NoisePlan, make_plan
from fedhealth.privacy.suppression import DEFAULT_K, is_suppressed

DEFAULT_DELTA = 1e-6
DEFAULT_TOLERANCE = 1


class PrivacyParams(BaseModel):
    """What the requester asks for: how much privacy budget to spend on this query, and with which noise."""
    model_config = ConfigDict(extra="forbid")
    epsilon: float = Field(gt=0, le=10)
    mechanism: Literal["laplace", "gaussian"] = "laplace"
    delta: float = Field(default=0.0, ge=0, lt=0.01)        # only used by the Gaussian mechanism

    @model_validator(mode="after")
    def _delta_default(self) -> "PrivacyParams":
        if self.mechanism == "gaussian" and self.delta == 0:
            self.delta = DEFAULT_DELTA
        if self.mechanism == "laplace":
            self.delta = 0.0
        return self


@dataclass(frozen=True)
class ReleasePolicy:
    k: int = DEFAULT_K
    tolerance: int = DEFAULT_TOLERANCE  # hospitals that may collude with the coordinator and still not see past the noise
    dp_enabled: bool = True             # False = k-threshold ONLY, no noise. Attack demos / evaluation; never set by the API.


@dataclass
class Exact:
    """An exact answer. Only the EVALUATION harness has one (it can ask the insecure exact route); the private
    pipeline never does."""
    agg: str
    count: int | None = None                 # cohort size (count / sum / avg)
    total: int | None = None                 # scaled-integer total (sum / avg)
    cells: dict[str, int] | None = None      # histogram bins

    def as_cells(self) -> list[int]:
        if self.agg == "count":
            return [self.count]
        if self.agg in ("sum", "avg"):
            return [self.count, self.total]
        return list(self.cells.values())


@dataclass
class Released:
    agg: str
    value: float | int | dict[str, int | None] | None
    suppressed: bool | dict[str, bool]
    epsilon: float
    delta: float
    rho: float
    mechanism: str
    noise_scale: float                       # std of the total noise on the released number: public by design
    expected_abs_error: float | None         # of the released number, in the released unit

    @property
    def fully_suppressed(self) -> bool:
        if isinstance(self.suppressed, dict):
            return all(self.suppressed.values())
        return bool(self.suppressed)


def exact_from_cells(spec: QuerySpec, cells: list[int], categories: list[str]) -> Exact:
    if spec.agg == "count":
        return Exact("count", count=cells[0])
    if spec.agg in ("sum", "avg"):
        return Exact(spec.agg, count=cells[0], total=cells[1])
    return Exact("histogram", cells=dict(zip(categories, cells, strict=True)))


def noise_contributors(spec: QuerySpec) -> int:
    """How many hospitals add a noise share: the lone data holder for a one-hospital query (it already knows its own
    answer), otherwise the two computing hospitals plus the triple dealer (see smpc/protocol.py: assign_roles)."""
    return 1 if len(participating_hospitals(spec)) == 1 else min(3, len(load_topology().ring))


def plan_noise(spec: QuerySpec, privacy: PrivacyParams, policy: ReleasePolicy = ReleasePolicy(),
               contributors: int | None = None) -> NoisePlan:
    c = noise_contributors(spec) if contributors is None else contributors
    if spec.agg == "count":
        groups = [(1, 1)]
    elif spec.agg == "histogram":
        groups = [(len(categories(spec.group_by, spec.condition)), 1)]
    else:
        lo, hi = value_bounds(spec.value_field, spec.condition)
        groups = [(1, 1), (1, max(abs(lo), abs(hi)))]           # [count, total]
    mech = privacy.mechanism if policy.dp_enabled else "none"
    return make_plan(mech, privacy.epsilon, privacy.delta, groups, c, policy.tolerance)


def finalize(spec: QuerySpec, noisy: list[int], plan: NoisePlan, categories: list[str],
             policy: ReleasePolicy = ReleasePolicy()) -> Released:
    """k-threshold + post-processing of NOISY cells. Nothing here touches an exact value."""
    name = plan.mechanism if policy.dp_enabled else "none"
    eps, delta, rho = (plan.epsilon, plan.delta, plan.rho) if policy.dp_enabled else (0.0, 0.0, 0.0)

    if spec.agg == "count":
        v = max(0, int(noisy[0]))
        hidden = is_suppressed(v, policy.k)
        return Released("count", None if hidden else v, hidden, eps, delta, rho, name, plan.std(0), plan.expected_abs_error(0))

    if spec.agg == "histogram":
        values: dict[str, int | None] = {}
        hidden_bins: dict[str, bool] = {}
        for b, c in zip(categories, noisy, strict=True):
            v = max(0, int(c))
            hidden_bins[b] = is_suppressed(v, policy.k)
            values[b] = None if hidden_bins[b] else v
        return Released("histogram", values, hidden_bins, eps, delta, rho, name, plan.std(0), plan.expected_abs_error(0))

    scale = load_fields().value_fields[spec.value_field].scale
    noisy_count, noisy_total = max(0, int(noisy[0])), int(noisy[1])
    hidden = is_suppressed(noisy_count, policy.k)
    if spec.agg == "sum":
        return Released("sum", None if hidden else noisy_total / scale, hidden, eps, delta, rho, name,
                        plan.std(1) / scale, plan.expected_abs_error(1) / scale)
    hidden = hidden or noisy_count < 1                           # cannot divide by a non-positive noisy count
    if hidden:
        return Released("avg", None, True, eps, delta, rho, name, plan.std(1) / scale, None)
    lo, hi = value_bounds(spec.value_field, spec.condition)
    avg = noisy_total / noisy_count / scale                     # post-processing: a mean must lie within the bounds
    return Released("avg", min(max(avg, lo / scale), hi / scale), False, eps, delta, rho, name, plan.std(1) / scale, None)


def _categories(spec: QuerySpec, exact: Exact) -> list[str]:
    return list(exact.cells) if exact.cells is not None else []


def release(spec: QuerySpec, exact: Exact, privacy: PrivacyParams, policy: ReleasePolicy = ReleasePolicy(),
            randbelow=secrets.randbelow, contributors: int | None = None) -> Released:
    """One full release of a KNOWN exact answer, in one process (attack demos, tests): every contributor's noise
    share is drawn with the hospitals' own exact sampler and added, exactly as the MPC would."""
    plan = plan_noise(spec, privacy, policy, contributors)
    noisy = list(exact.as_cells())
    for _ in range(plan.contributors):
        noisy = [x + n for x, n in zip(noisy, plan.sample_share(randbelow), strict=True)]
    return finalize(spec, noisy, plan, _categories(spec, exact), policy)


def simulate_releases(spec: QuerySpec, exact: Exact, privacy: PrivacyParams, policy: ReleasePolicy, reps: int,
                      rng: np.random.Generator, contributors: int | None = None) -> list[Released]:
    """`reps` independent releases using the vectorised simulation of the total noise (evaluation only)."""
    plan = plan_noise(spec, privacy, policy, contributors)
    base = np.array(exact.as_cells(), dtype=np.int64)
    noise = plan.simulate_total(reps, rng)
    cats = _categories(spec, exact)
    return [finalize(spec, list(base + row), plan, cats, policy) for row in noise]
