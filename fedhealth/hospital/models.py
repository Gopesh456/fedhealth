"""SQLAlchemy models for the three hospitals. Each hospital has its OWN database file and its
OWN declarative base, so nothing here can accidentally join across hospitals.

SensitiveStr columns (marked 🔒 in the plan) are AES-GCM encrypted at rest from Phase 5.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import Boolean, Date, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from fedhealth.privacy.encryption.column_types import SensitiveStr


# ============================================================================ Hospital A: medical history
class ABase(DeclarativeBase):
    pass


class APatient(ABase):
    __tablename__ = "patients"
    patient_id: Mapped[str] = mapped_column(String(16), primary_key=True)        # local MRN, e.g. A-000123
    national_id: Mapped[str] = mapped_column(SensitiveStr(context="A.patients.national_id"), nullable=False)    # 🔒 linkage key
    full_name: Mapped[str] = mapped_column(SensitiveStr(context="A.patients.full_name"), nullable=False)     # 🔒
    date_of_birth: Mapped[str] = mapped_column(SensitiveStr(context="A.patients.date_of_birth"), nullable=False)  # 🔒 ISO date
    birth_year: Mapped[int] = mapped_column(Integer, nullable=False)              # plaintext: needed for age filters
    sex: Mapped[str] = mapped_column(String(1), nullable=False)
    zip_code: Mapped[str] = mapped_column(SensitiveStr(context="A.patients.zip_code"), nullable=False)        # 🔒
    registered_on: Mapped[date] = mapped_column(Date, nullable=False)


class ACondition(ABase):
    __tablename__ = "conditions"
    condition_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    icd10_code: Mapped[str] = mapped_column(String(8), nullable=False)
    condition_name: Mapped[str] = mapped_column(String(64), nullable=False)
    is_chronic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    diagnosed_on: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(12), nullable=False)               # active | resolved
    severity: Mapped[str] = mapped_column(String(10), nullable=False)             # mild | moderate | severe
    notes: Mapped[str | None] = mapped_column(SensitiveStr(context="A.conditions.notes"), nullable=True)   # 🔒 free text
    __table_args__ = (Index("ix_a_cond_code", "icd10_code", "patient_id"),)


# ============================================================================ Hospital B: diagnostic reports
class BBase(DeclarativeBase):
    pass


class BPatient(BBase):
    __tablename__ = "patients"
    patient_id: Mapped[str] = mapped_column(String(16), primary_key=True)         # B-000456
    national_id: Mapped[str] = mapped_column(SensitiveStr(context="B.patients.national_id"), nullable=False)    # 🔒
    registered_on: Mapped[date] = mapped_column(Date, nullable=False)


class BDiagnosticReport(BBase):
    __tablename__ = "diagnostic_reports"
    report_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    report_type: Mapped[str] = mapped_column(String(8), nullable=False)           # lab | imaging
    test_code: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ref_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ref_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    abnormal_flag: Mapped[str | None] = mapped_column(String(2), nullable=True)   # N | H | L
    imaging_finding: Mapped[str | None] = mapped_column(SensitiveStr(context="B.diagnostic_reports.imaging_finding"), nullable=True)  # 🔒
    performed_on: Mapped[date] = mapped_column(Date, nullable=False)
    ordering_reason_icd10: Mapped[str] = mapped_column(String(8), nullable=False)


class BFollowupOutcome(BBase):
    __tablename__ = "followup_outcomes"
    outcome_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    condition_icd10: Mapped[str] = mapped_column(String(8), nullable=False)
    outcome: Mapped[str] = mapped_column(String(10), nullable=False)              # improved | no_change | worsened
    baseline_value: Mapped[float] = mapped_column(Float, nullable=False)
    followup_value: Mapped[float] = mapped_column(Float, nullable=False)
    delta: Mapped[float] = mapped_column(Float, nullable=False)                   # followup - baseline
    assessed_on: Mapped[date] = mapped_column(Date, nullable=False)
    __table_args__ = (Index("ix_b_out_cond", "condition_icd10", "outcome"),)


# ============================================================================ Hospital C: prescriptions
class CBase(DeclarativeBase):
    pass


class CPatient(CBase):
    __tablename__ = "patients"
    patient_id: Mapped[str] = mapped_column(String(16), primary_key=True)         # C-000789
    national_id: Mapped[str] = mapped_column(SensitiveStr(context="C.patients.national_id"), nullable=False)    # 🔒
    registered_on: Mapped[date] = mapped_column(Date, nullable=False)


class CPrescription(CBase):
    __tablename__ = "prescriptions"
    rx_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.patient_id"), nullable=False)
    drug_name: Mapped[str] = mapped_column(String(40), nullable=False)
    atc_code: Mapped[str] = mapped_column(String(10), nullable=False)
    drug_class: Mapped[str] = mapped_column(String(40), nullable=False)
    dose_mg: Mapped[float] = mapped_column(Float, nullable=False)
    frequency: Mapped[str] = mapped_column(String(6), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)            # NULL = ongoing
    reason_icd10: Mapped[str] = mapped_column(String(8), nullable=False)          # prescribing reason
    adherence_pct: Mapped[float] = mapped_column(Float, nullable=False)           # pharmacy-refill adherence
    prescriber_id: Mapped[str] = mapped_column(String(10), nullable=False)
    notes: Mapped[str | None] = mapped_column(SensitiveStr(context="C.prescriptions.notes"), nullable=True)   # 🔒
    __table_args__ = (Index("ix_c_rx_reason", "reason_icd10", "drug_name"),)


# ============================================================================ federated-learning registry (all hospitals)
class TrainingEpisodeColumns:
    """IDENTICAL schema at every hospital: FedAvg needs horizontally partitioned data (see synth/registry.py).
    Rows are clinical features + the response label; they hold no identifier, name or date."""
    episode_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    split: Mapped[str] = mapped_column(String(5), nullable=False)                  # train | test
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    male: Mapped[int] = mapped_column(Integer, nullable=False)
    extra_comorbidities: Mapped[int] = mapped_column(Integer, nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    condition: Mapped[str] = mapped_column(String(8), nullable=False)
    drugs: Mapped[str] = mapped_column(String(120), nullable=False)                # '|'-joined drug names
    adherence_pct: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_lab_z: Mapped[float] = mapped_column(Float, nullable=False)
    responded: Mapped[int] = mapped_column(Integer, nullable=False)                # 1 = outcome improved


class ATrainingEpisode(TrainingEpisodeColumns, ABase):
    __tablename__ = "training_episodes"


class BTrainingEpisode(TrainingEpisodeColumns, BBase):
    __tablename__ = "training_episodes"


class CTrainingEpisode(TrainingEpisodeColumns, CBase):
    __tablename__ = "training_episodes"


BASES = {"A": ABase, "B": BBase, "C": CBase}
TRAINING_EPISODE = {"A": ATrainingEpisode, "B": BTrainingEpisode, "C": CTrainingEpisode}
