"""Shared medical vocabulary (SYNTHETIC - invented parameters, not clinical guidance).

This module is the single source of truth for:
  * which conditions / drugs / labs exist,
  * the generative parameters the synthetic-data generator uses,
  * the closed set of values a query is allowed to mention (the "whitelist").

Everything here is plain frozen dataclasses so it is easy to read and explain.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Reference date for the whole synthetic world. "Age" everywhere in this project is
#   age = AS_OF.year - birth_year
# (age-by-calendar-year), so age filters can be answered from a plain integer column.
AS_OF = date(2025, 12, 31)

OUTCOMES = ("improved", "no_change", "worsened")
SEXES = ("F", "M")
SEVERITIES = ("mild", "moderate", "severe")

# (label, min_age, max_age) - used by the `age_band` histogram grouping at Hospital A.
AGE_BANDS = (
    ("18-29", 18, 29),
    ("30-44", 30, 44),
    ("45-59", 45, 59),
    ("60-74", 60, 74),
    ("75+", 75, 200),
)


def age_of(birth_year: int) -> int:
    return AS_OF.year - birth_year


def age_band_of(age: int) -> str:
    for label, lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return label
    return AGE_BANDS[0][0] if age < AGE_BANDS[0][1] else AGE_BANDS[-1][0]


@dataclass(frozen=True)
class Drug:
    name: str
    atc: str
    drug_class: str
    dose_mg: float
    frequency: str
    weight: float   # relative prescribing share within its condition (small weight => rare => small cells)
    effect: float   # log-odds contribution to P(outcome == improved) in the generator


@dataclass(frozen=True)
class Lab:
    code: str
    name: str
    unit: str
    baseline_mean: float
    baseline_sd: float
    floor: float        # physiological floor for the baseline draw
    better: int         # -1: lower value is better, +1: higher is better
    threshold: float    # minimum clinically meaningful change (|delta| >= threshold => improved/worsened)
    ref_low: float
    ref_high: float
    clamp: float        # |delta| bound. Used as the SUM sensitivity bound, and respected by the generator.


@dataclass(frozen=True)
class Condition:
    code: str
    name: str
    # Prevalence model: logit p = intercept + age_slope*(age-55)/10 + male_effect*[male] + shared frailty
    intercept: float
    age_slope: float
    male_effect: float
    base_improve_logit: float
    lab: Lab
    drugs: tuple[Drug, ...]
    imaging: tuple[str, tuple[str, ...]] | None = None   # (modality, possible findings)


CONDITIONS: dict[str, Condition] = {c.code: c for c in (
    Condition(
        "E11", "Type 2 diabetes mellitus", -1.27, 0.35, 0.20, 0.2,
        Lab("HBA1C", "HbA1c", "%", 8.4, 1.1, 6.5, -1, 0.5, 4.0, 5.6, 4.0),
        (
            Drug("metformin", "A10BA02", "biguanide", 1000, "BID", 0.45, 0.0),
            Drug("empagliflozin", "A10BK03", "SGLT2 inhibitor", 10, "QD", 0.20, 0.5),
            Drug("sitagliptin", "A10BH01", "DPP-4 inhibitor", 100, "QD", 0.15, -0.1),
            Drug("glipizide", "A10BB07", "sulfonylurea", 5, "BID", 0.12, -0.2),
            Drug("insulin glargine", "A10AE04", "insulin", 20, "QD", 0.08, 0.3),
        ),
        None,
    ),
    Condition(
        "I10", "Essential hypertension", -0.71, 0.55, 0.20, 0.2,
        Lab("SBP", "Systolic blood pressure", "mmHg", 152, 12, 130, -1, 8, 90, 120, 50),
        (
            Drug("lisinopril", "C09AA03", "ACE inhibitor", 20, "QD", 0.34, 0.2),
            Drug("amlodipine", "C08CA01", "calcium channel blocker", 5, "QD", 0.30, 0.1),
            Drug("losartan", "C09CA01", "ARB", 50, "QD", 0.22, 0.2),
            Drug("hydrochlorothiazide", "C03AA03", "thiazide", 25, "QD", 0.11, -0.2),
            Drug("bisoprolol", "C07AB07", "beta blocker", 5, "QD", 0.03, 0.0),
        ),
        ("ECG", ("normal sinus rhythm", "left ventricular hypertrophy", "left atrial enlargement")),
    ),
    Condition(
        "E78", "Hyperlipidaemia", -1.27, 0.35, 0.10, 0.3,
        Lab("LDL", "LDL cholesterol", "mg/dL", 165, 25, 110, -1, 25, 0, 100, 100),
        (
            Drug("atorvastatin", "C10AA05", "statin", 40, "QD", 0.50, 0.3),
            Drug("rosuvastatin", "C10AA07", "statin", 20, "QD", 0.32, 0.4),
            Drug("ezetimibe", "C10AX09", "cholesterol absorption inhibitor", 10, "QD", 0.14, -0.1),
            Drug("evolocumab", "C10AX13", "PCSK9 inhibitor", 140, "Q2W", 0.04, 0.8),
        ),
        None,
    ),
    Condition(
        "J45", "Asthma", -2.00, -0.05, 0.0, 0.2,
        Lab("FEV1", "FEV1 % predicted", "%", 68, 10, 35, +1, 5, 80, 120, 40),
        (
            Drug("salbutamol", "R03AC02", "SABA", 0.1, "PRN", 0.36, -0.2),
            Drug("budesonide-formoterol", "R03AK07", "ICS/LABA", 0.2, "BID", 0.34, 0.5),
            Drug("montelukast", "R03DC03", "leukotriene antagonist", 10, "QD", 0.26, 0.0),
            Drug("tiotropium", "R03BB04", "LAMA", 0.0025, "QD", 0.04, 0.2),
        ),
        ("Chest X-ray", ("no acute findings", "hyperinflation", "bronchial wall thickening")),
    ),
    Condition(
        "J44", "Chronic obstructive pulmonary disease", -2.59, 0.70, 0.30, 0.0,
        Lab("FEV1", "FEV1 % predicted", "%", 52, 12, 25, +1, 4, 80, 120, 30),
        (
            Drug("tiotropium", "R03BB04", "LAMA", 0.0025, "QD", 0.40, 0.3),
            Drug("salbutamol", "R03AC02", "SABA", 0.1, "PRN", 0.30, -0.2),
            Drug("fluticasone-salmeterol", "R03AK06", "ICS/LABA", 0.25, "BID", 0.26, 0.4),
            Drug("roflumilast", "R03DX07", "PDE-4 inhibitor", 0.5, "QD", 0.04, 0.1),
        ),
        ("Chest X-ray", ("hyperinflation", "emphysematous changes", "infiltrate", "no acute findings")),
    ),
    Condition(
        "N18", "Chronic kidney disease", -2.31, 0.65, 0.0, -0.1,
        Lab("EGFR", "eGFR", "mL/min/1.73m2", 42, 10, 15, +1, 3, 90, 120, 25),
        (
            Drug("losartan", "C09CA01", "ARB", 50, "QD", 0.38, 0.2),
            Drug("dapagliflozin", "A10BK01", "SGLT2 inhibitor", 10, "QD", 0.34, 0.5),
            Drug("sevelamer", "V03AE02", "phosphate binder", 800, "TID", 0.24, -0.1),
            Drug("epoetin alfa", "B03XA01", "ESA", 4000, "QW", 0.04, -0.3),
        ),
        ("Renal ultrasound", ("normal echogenicity", "increased echogenicity", "cortical thinning")),
    ),
    Condition(
        "I50", "Heart failure", -2.94, 0.80, 0.30, -0.2,
        Lab("NTPROBNP", "NT-proBNP", "pg/mL", 2400, 800, 900, -1, 300, 0, 125, 2000),
        (
            Drug("carvedilol", "C07AG02", "beta blocker", 12.5, "BID", 0.36, 0.2),
            Drug("furosemide", "C03CA01", "loop diuretic", 40, "QD", 0.34, -0.2),
            Drug("spironolactone", "C03DA01", "MRA", 25, "QD", 0.16, 0.2),
            Drug("sacubitril-valsartan", "C09DX04", "ARNI", 97, "BID", 0.11, 0.6),
            Drug("ivabradine", "C01EB17", "If inhibitor", 5, "BID", 0.03, 0.2),
        ),
        ("Echocardiogram", ("EF preserved", "EF mildly reduced", "EF severely reduced", "LV hypertrophy")),
    ),
    Condition(
        "F32", "Major depressive disorder", -1.73, -0.05, -0.40, 0.1,
        Lab("PHQ9", "PHQ-9 score", "points", 15, 3.5, 10, -1, 3, 0, 4, 15),
        (
            Drug("sertraline", "N06AB06", "SSRI", 100, "QD", 0.34, 0.2),
            Drug("escitalopram", "N06AB10", "SSRI", 10, "QD", 0.30, 0.2),
            Drug("bupropion", "N06AX12", "NDRI", 150, "QD", 0.16, 0.1),
            Drug("mirtazapine", "N06AX11", "NaSSA", 30, "QD", 0.16, 0.0),
            Drug("vortioxetine", "N06AX26", "multimodal", 10, "QD", 0.04, 0.3),
        ),
        None,
    ),
)}

CONDITION_CODES = tuple(CONDITIONS)
ALL_DRUGS = tuple(sorted({d.name for c in CONDITIONS.values() for d in c.drugs}))


def drug_names(condition: str) -> tuple[str, ...]:
    return tuple(d.name for d in CONDITIONS[condition].drugs)


# Coefficients of the generator's response model (also written to ground_truth/generator_params.json
# so the federated-learning stretch goal can check it recovers the true signal).
RESPONSE_MODEL = {
    "adherence_per_pp": 0.035,       # per percentage point of adherence above 70
    "age_per_decade": -0.15,         # per decade above 60
    "per_extra_comorbidity": -0.25,
    "severity": {"mild": 0.4, "moderate": 0.0, "severe": -0.5},
    "worsened_given_not_improved": 0.30,
}

# Ground-truth overlap of the synthetic population across hospitals (fractions of all people).
MEMBERSHIP_FRACTIONS = {
    "ABC": 0.40, "AC": 0.12, "AB": 0.08, "BC": 0.05,
    "A": 0.15, "B": 0.10, "C": 0.10,
}
