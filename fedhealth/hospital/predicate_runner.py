"""Turns a validated LocalQuery into a per-patient result, using the hospital's OWN database only.

This is the one place SQL is built. Everything the caller controls arrives as a *bound parameter*
of an ORM expression (never string-interpolated), and the shape of the query is fixed per hospital:
  - the condition is always required,
  - only whitelisted predicate fields (see fedhealth.common.schemas) can add WHERE clauses.

The output is one row per matching patient: (patient_id, national_id, scaled value, category).
Phase 1 baseline ships these rows to the coordinator (insecure!); from Phase 3 on they are turned into
0/1 indicator vectors and secret-shared - they never leave the hospital in the clear.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from fedhealth.common import vocab
from fedhealth.common.config import load_fields
from fedhealth.common.schemas import LocalQuery, value_bounds
from fedhealth.hospital import models as m


@dataclass(frozen=True)
class MatchRow:
    patient_id: str
    national_id: str
    value: int | None
    category: str | None


def _scaled(q: LocalQuery, raw: float) -> int:
    """Scale to an integer (secret sharing works over integers) and clamp to the declared bounds."""
    lo, hi = value_bounds(q.value_field, q.condition)
    scale = load_fields().value_fields[q.value_field].scale
    return max(lo, min(hi, int(round(raw * scale))))


def run_local_query(session: Session, q: LocalQuery) -> list[MatchRow]:
    return {"A": _run_a, "B": _run_b, "C": _run_c}[q.hospital](session, q)


# ----------------------------------------------------------------------------- Hospital A
def _run_a(session: Session, q: LocalQuery) -> list[MatchRow]:
    pred = q.parsed_predicate()
    stmt = select(m.APatient.patient_id, m.APatient.national_id, m.APatient.birth_year, m.APatient.sex).where(
        exists().where(m.ACondition.patient_id == m.APatient.patient_id, m.ACondition.icd10_code == q.condition)
    )
    if pred.sex is not None:
        stmt = stmt.where(m.APatient.sex == pred.sex)
    # age = AS_OF.year - birth_year  =>  age >= min  <=>  birth_year <= AS_OF.year - min
    if pred.age_min is not None:
        stmt = stmt.where(m.APatient.birth_year <= vocab.AS_OF.year - pred.age_min)
    if pred.age_max is not None:
        stmt = stmt.where(m.APatient.birth_year >= vocab.AS_OF.year - pred.age_max)

    rows = []
    for pid, nid, birth_year, sex in session.execute(stmt.order_by(m.APatient.patient_id)):
        age = vocab.age_of(birth_year)
        value = _scaled(q, age) if q.value_field == "age" else None
        category = {"sex": sex, "age_band": vocab.age_band_of(age)}.get(q.group_by) if q.group_by else None
        rows.append(MatchRow(pid, nid, value, category))
    return rows


# ----------------------------------------------------------------------------- Hospital B
def _run_b(session: Session, q: LocalQuery) -> list[MatchRow]:
    pred = q.parsed_predicate()
    stmt = (select(m.BPatient.patient_id, m.BPatient.national_id, m.BFollowupOutcome.outcome, m.BFollowupOutcome.delta)
            .join(m.BFollowupOutcome, m.BFollowupOutcome.patient_id == m.BPatient.patient_id)
            .where(m.BFollowupOutcome.condition_icd10 == q.condition))
    if pred.outcome is not None:
        stmt = stmt.where(m.BFollowupOutcome.outcome == pred.outcome)

    rows, seen = [], set()
    for pid, nid, outcome, delta in session.execute(stmt.order_by(m.BPatient.patient_id, m.BFollowupOutcome.outcome_id)):
        if pid in seen:          # one follow-up per (patient, condition) by construction; stay safe regardless
            continue
        seen.add(pid)
        value = _scaled(q, delta) if q.value_field == "lab_delta" else None
        category = outcome if q.group_by == "outcome" else None
        rows.append(MatchRow(pid, nid, value, category))
    return rows


# ----------------------------------------------------------------------------- Hospital C
def _run_c(session: Session, q: LocalQuery) -> list[MatchRow]:
    pred = q.parsed_predicate()
    stmt = (select(m.CPatient.patient_id, m.CPatient.national_id, m.CPrescription.drug_name, m.CPrescription.adherence_pct)
            .join(m.CPrescription, m.CPrescription.patient_id == m.CPatient.patient_id)
            .where(m.CPrescription.reason_icd10 == q.condition))
    if pred.drug is not None:
        stmt = stmt.where(m.CPrescription.drug_name == pred.drug)

    rows, seen = [], set()
    # A patient may have two prescriptions for one condition. Keep the EARLIEST matching one so every
    # patient lands in exactly one histogram bin (this also keeps the DP sensitivity of a histogram at 1).
    for pid, nid, drug, adherence in session.execute(
            stmt.order_by(m.CPatient.patient_id, m.CPrescription.start_date, m.CPrescription.rx_id)):
        if pid in seen:
            continue
        seen.add(pid)
        value = _scaled(q, adherence) if q.value_field == "adherence_pct" else None
        category = drug if q.group_by == "drug" else None
        rows.append(MatchRow(pid, nid, value, category))
    return rows
