"""The release stage: from the EXACT secret-shared result to what a requester is allowed to see.

    exact aggregate (only the coordinator can add the shares)
        --> k-threshold   (is the cohort big enough to talk about at all?)
        --> DP noise      (discrete Laplace / Gaussian, calibrated to the query's sensitivity)
        --> post-process  (round, clamp counts >= 0; free under DP)
        --> Released      (the only thing that ever leaves the coordinator)

SENSITIVITY  (how much can ONE patient change the exact answer? - patients are added/removed)
    COUNT          1
    HISTOGRAM      1        every patient is in exactly ONE bin, so adding a patient changes ONE cell by 1
                            (parallel composition): all bins together cost epsilon ONCE, not epsilon x bins
    SUM            B        B = max |value| after clamping to the declared bounds (value_bounds)
    AVG            via noisy SUM / noisy COUNT, each with epsilon/2 (post-processing the ratio is free)
The clamping bounds are enforced hospital-side (predicate_runner._scaled), so B really bounds every contribution.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fedhealth.common.config import load_fields
from fedhealth.common.schemas import QuerySpec, value_bounds
from fedhealth.privacy.dp.mechanisms import make_mechanism
from fedhealth.privacy.suppression import DEFAULT_K, is_suppressed

DEFAULT_DELTA = 1e-6


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
    k_basis: Literal["exact", "noisy"] = "exact"
    dp_enabled: bool = True          # False = k-threshold ONLY, no noise. Attack demos / evaluation; never set by the API.


@dataclass
class Exact:
    """What the MPC reveals to the coordinator. NEVER returned to a requester."""
    agg: str
    count: int | None = None                 # cohort size (count / sum / avg)
    total: int | None = None                 # scaled-integer total (sum / avg)
    cells: dict[str, int] | None = None      # histogram bins


@dataclass
class Released:
    agg: str
    value: float | int | dict[str, int | None] | None
    suppressed: bool | dict[str, bool]
    epsilon: float
    delta: float
    mechanism: str
    noise_scale: float                       # Laplace b = Delta/epsilon (or Gaussian sigma): public by design
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


def release(spec: QuerySpec, exact: Exact, privacy: PrivacyParams, policy: ReleasePolicy = ReleasePolicy(),
            randbelow=secrets.randbelow) -> Released:
    eps, delta, name = privacy.epsilon, privacy.delta, privacy.mechanism
    noisy_basis = policy.k_basis == "noisy"

    def mech(e: float, d: float, sensitivity: int):
        return make_mechanism(name if policy.dp_enabled else "none", e, sensitivity, d, randbelow)

    # ------------------------------------------------------------------ COUNT
    if spec.agg == "count":
        mc = mech(eps, delta, 1)
        noisy = max(0, mc.release(exact.count))
        hidden = is_suppressed(noisy if noisy_basis else exact.count, policy.k)
        return Released("count", None if hidden else noisy, hidden, eps, delta, name, mc.scale, mc.expected_abs_error)

    # ------------------------------------------------------------------ HISTOGRAM (parallel composition)
    if spec.agg == "histogram":
        mc = mech(eps, delta, 1)
        values: dict[str, int | None] = {}
        hidden: dict[str, bool] = {}
        for bin_name, c in exact.cells.items():
            noisy = max(0, mc.release(c))
            hidden[bin_name] = is_suppressed(noisy if noisy_basis else c, policy.k)
            values[bin_name] = None if hidden[bin_name] else noisy
        return Released("histogram", values, hidden, eps, delta, name, mc.scale, mc.expected_abs_error)

    # ------------------------------------------------------------------ SUM / AVG
    fields = load_fields()
    scale = fields.value_fields[spec.value_field].scale
    lo, hi = value_bounds(spec.value_field, spec.condition)
    bound = max(abs(lo), abs(hi))
    split = spec.agg == "avg" or noisy_basis                    # a noisy count is needed -> budget is split in two
    e_each, d_each = (eps / 2, delta / 2) if split else (eps, delta)

    m_total = mech(e_each, d_each, bound)
    noisy_total = m_total.release(exact.total)
    if split:
        m_count = mech(e_each, d_each, 1)
        noisy_count = max(0, m_count.release(exact.count))
    hidden = is_suppressed(noisy_count if noisy_basis else exact.count, policy.k)

    if spec.agg == "sum":
        value = None if hidden else noisy_total / scale
        return Released("sum", value, hidden, eps, delta, name, m_total.scale / scale, m_total.expected_abs_error / scale)

    hidden = hidden or noisy_count < 1                           # cannot divide by a non-positive noisy count
    if hidden:
        return Released("avg", None, True, eps, delta, name, m_total.scale / scale, None)
    avg = noisy_total / noisy_count / scale
    lo_n, hi_n = lo / scale, hi / scale                          # post-processing: a mean must lie within the bounds
    return Released("avg", min(max(avg, lo_n), hi_n), False, eps, delta, name, m_total.scale / scale, None)
