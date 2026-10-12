"""Query language and wire schemas.

The query language is deliberately tiny. A query is an *aggregate* over patients, described by:
  agg        count | sum | avg | histogram          (nothing else exists - no SELECT, no columns, no LIMIT)
  condition  an ICD-10 code from the closed vocabulary (shared with every participating hospital)
  a / b / c  optional per-hospital predicate slices (A: demographics, B: outcome, C: treatment)
  value_field / group_by   which numeric / categorical attribute SUM-AVG / HISTOGRAM use

Every model uses `extra="forbid"`, so any attempt to smuggle in a row-level request
(e.g. {"columns": ["name"]} or {"select": "*"}) fails validation. This is layer (i) of the
aggregate-only enforcement; hospitals re-validate their own slice (layer ii).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import vocab
from .config import load_fields

HOSPITAL_IDS = ("A", "B", "C")
HospitalId = Literal["A", "B", "C"]
Agg = Literal["count", "sum", "avg", "histogram"]
ValueField = Literal["age", "lab_delta", "adherence_pct"]
GroupBy = Literal["sex", "age_band", "outcome", "drug"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- per-hospital predicates
class APredicate(_Strict):
    """Hospital A (medical history): demographic filters. 'Has the condition' is implicit."""
    age_min: int | None = Field(default=None, ge=0, le=120)
    age_max: int | None = Field(default=None, ge=0, le=120)
    sex: Literal["F", "M"] | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "APredicate":
        if self.age_min is not None and self.age_max is not None and self.age_min > self.age_max:
            raise ValueError("age_min must be <= age_max")
        return self


class BPredicate(_Strict):
    """Hospital B (diagnostic reports): follow-up outcome for the condition."""
    outcome: Literal["improved", "no_change", "worsened"] | None = None


class CPredicate(_Strict):
    """Hospital C (prescriptions): drug prescribed *for the condition*."""
    drug: str | None = None

    @model_validator(mode="after")
    def _known_drug(self) -> "CPredicate":
        if self.drug is not None and self.drug not in vocab.ALL_DRUGS:
            raise ValueError(f"unknown drug {self.drug!r}")
        return self


PREDICATE_MODELS: dict[str, type[_Strict]] = {"A": APredicate, "B": BPredicate, "C": CPredicate}


def parse_predicate(hospital: str, raw: dict[str, Any]) -> _Strict:
    """Validate a predicate slice against the whitelist for `hospital` (raises ValidationError)."""
    return PREDICATE_MODELS[hospital].model_validate(raw)


# --------------------------------------------------------------------------- the query
class QuerySpec(_Strict):
    agg: Agg
    condition: str
    a: APredicate | None = None
    b: BPredicate | None = None
    c: CPredicate | None = None
    value_field: ValueField | None = None   # required for sum / avg
    group_by: GroupBy | None = None         # required for histogram

    @model_validator(mode="after")
    def _consistent(self) -> "QuerySpec":
        if self.condition not in vocab.CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}")
        if self.agg in ("sum", "avg"):
            if self.value_field is None or self.group_by is not None:
                raise ValueError(f"{self.agg} requires value_field (and no group_by)")
        elif self.agg == "histogram":
            if self.group_by is None or self.value_field is not None:
                raise ValueError("histogram requires group_by (and no value_field)")
        else:  # count
            if self.value_field is not None or self.group_by is not None:
                raise ValueError("count takes neither value_field nor group_by")
        if self.c is not None and self.c.drug is not None:
            if self.c.drug not in vocab.drug_names(self.condition):
                raise ValueError(f"{self.c.drug!r} is not in the formulary for {self.condition}")
        if not participating_hospitals(self):
            raise ValueError("query must involve at least one hospital (give a, b or c)")
        return self


def participating_hospitals(spec: QuerySpec) -> list[str]:
    """Hospitals that must take part: those with a predicate, plus the owner of value_field/group_by."""
    fields = load_fields()
    involved = set()
    if spec.a is not None:
        involved.add("A")
    if spec.b is not None:
        involved.add("B")
    if spec.c is not None:
        involved.add("C")
    if spec.value_field:
        involved.add(fields.value_fields[spec.value_field].owner)
    if spec.group_by:
        involved.add(fields.group_by[spec.group_by].owner)
    return [h for h in HOSPITAL_IDS if h in involved]   # ring order


def predicate_slice(spec: QuerySpec, hospital: str) -> dict[str, Any]:
    pred = {"A": spec.a, "B": spec.b, "C": spec.c}[hospital]
    return pred.model_dump(exclude_none=True) if pred is not None else {}


# --------------------------------------------------------------------------- what one hospital receives
class LocalQuery(_Strict):
    """The slice of a query sent to ONE hospital: its own predicate + the shared condition code.
    A hospital never learns the other hospitals' predicates."""
    hospital: HospitalId
    condition: str
    predicate: dict[str, Any] = Field(default_factory=dict)
    value_field: ValueField | None = None
    group_by: GroupBy | None = None

    @model_validator(mode="after")
    def _valid_for_this_hospital(self) -> "LocalQuery":
        if self.condition not in vocab.CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}")
        parse_predicate(self.hospital, self.predicate)   # raises on unknown keys / bad values
        fields = load_fields()
        if self.value_field and fields.value_fields[self.value_field].owner != self.hospital:
            raise ValueError(f"{self.value_field} is not held by hospital {self.hospital}")
        if self.group_by and fields.group_by[self.group_by].owner != self.hospital:
            raise ValueError(f"{self.group_by} is not held by hospital {self.hospital}")
        if self.value_field and self.group_by:
            raise ValueError("value_field and group_by are mutually exclusive")
        return self

    def parsed_predicate(self) -> _Strict:
        return parse_predicate(self.hospital, self.predicate)


def local_query(spec: QuerySpec, hospital: str) -> LocalQuery:
    fields = load_fields()
    vf = spec.value_field if spec.value_field and fields.value_fields[spec.value_field].owner == hospital else None
    gb = spec.group_by if spec.group_by and fields.group_by[spec.group_by].owner == hospital else None
    return LocalQuery(hospital=hospital, condition=spec.condition,
                      predicate=predicate_slice(spec, hospital), value_field=vf, group_by=gb)


# --------------------------------------------------------------------------- insecure baseline wire format
class LocalMatchRow(BaseModel):
    national_id: str            # <- the whole point of the baseline: raw identifiers cross the wire
    value: int | None = None    # scaled integer (SUM/AVG)
    category: str | None = None  # histogram bin


class LocalMatchResponse(BaseModel):
    hospital: HospitalId
    rows: list[LocalMatchRow]


class PlainQueryResult(BaseModel):
    """Result of an INSECURE path (exact, un-noised, cohort size included) - evaluation/testing only."""
    mode: Literal["plain-insecure", "psi-linked-insecure-aggregation", "mpc-exact-no-dp"] = "plain-insecure"
    agg: Agg
    hospitals: list[str]
    result: int | float | dict[str, int] | None
    cohort_size: int | None = None          # None when the aggregate does not reveal it (e.g. SUM)
    timings: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- helpers shared by hospitals/coordinator/tests
def categories(group_by: str, condition: str) -> list[str]:
    """Closed list of bins for a histogram (zero bins are reported too, so output shape never leaks)."""
    if group_by == "sex":
        return list(vocab.SEXES)
    if group_by == "age_band":
        return [b[0] for b in vocab.AGE_BANDS]
    if group_by == "outcome":
        return list(vocab.OUTCOMES)
    if group_by == "drug":
        return list(vocab.drug_names(condition))
    raise ValueError(group_by)


def value_bounds(value_field: str, condition: str) -> tuple[int, int]:
    """Clamping bounds (inclusive, in *scaled integer* units). max(|lo|,|hi|) is the SUM sensitivity."""
    spec = load_fields().value_fields[value_field]
    if value_field == "lab_delta":
        b = int(round(vocab.CONDITIONS[condition].lab.clamp * spec.scale))
        return -b, b
    return int(spec.min * spec.scale), int(spec.max * spec.scale)
