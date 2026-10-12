"""Feature encoding of a treatment episode for the response model.

The transform uses only PUBLIC constants (domain scales and the closed vocabularies), never statistics computed
from any hospital's data - so every hospital encodes its rows identically without exchanging anything.

    x = [ 1 (bias), age_z, male, extra_comorbidities, severity_mild, severity_severe, adherence_z, baseline_lab_z,
          condition one-hot (8), drug multi-hot (33) ]
"""
from __future__ import annotations

import numpy as np

from fedhealth.common import vocab

NUMERIC = ["bias", "age_z", "male", "extra_comorbidities", "severity_mild", "severity_severe", "adherence_z", "baseline_lab_z"]
CONDITION_COLS = [f"cond_{c}" for c in vocab.CONDITION_CODES]
DRUG_COLS = [f"drug_{d}" for d in vocab.ALL_DRUGS]
FEATURE_NAMES = NUMERIC + CONDITION_COLS + DRUG_COLS
DIM = len(FEATURE_NAMES)

_COND_IDX = {c: NUMERIC.__len__() + i for i, c in enumerate(vocab.CONDITION_CODES)}
_DRUG_IDX = {d: len(NUMERIC) + len(CONDITION_COLS) + i for i, d in enumerate(vocab.ALL_DRUGS)}


def featurize(age: int, male: int, extra_comorbidities: int, severity: str, adherence_pct: float, baseline_lab_z: float,
              condition: str, drugs: list[str]) -> np.ndarray:
    x = np.zeros(DIM)
    x[0] = 1.0
    x[1] = (age - 60) / 15.0
    x[2] = float(male)
    x[3] = float(extra_comorbidities)
    x[4] = float(severity == "mild")
    x[5] = float(severity == "severe")
    x[6] = (adherence_pct - 70.0) / 20.0
    x[7] = baseline_lab_z
    x[_COND_IDX[condition]] = 1.0
    for d in drugs:
        x[_DRUG_IDX[d]] = 1.0
    return x


def row_to_vector(r) -> np.ndarray:
    """Encode one registry row (a dict, an ORM row, or a csv.DictReader row with the same field names)."""
    get = (lambda k: r[k]) if isinstance(r, dict) else (lambda k: getattr(r, k))
    return featurize(int(get("age")), int(get("male")), int(get("extra_comorbidities")), str(get("severity")),
                     float(get("adherence_pct")), float(get("baseline_lab_z")), str(get("condition")),
                     str(get("drugs")).split("|"))


def true_weights() -> np.ndarray:
    """The generator's TRUE logit coefficients expressed in this feature space (used only to check what FL recovers).
    logit = base[cond] + sum(drug effects) + 0.035*(adh-70) - 0.15*(age-60)/10 - 0.25*extra_comorb + severity term.
    The generator's per-condition base logit is folded into the condition one-hot; the baseline lab has no effect (0)."""
    m = vocab.RESPONSE_MODEL
    w = np.zeros(DIM)
    w[1] = m["age_per_decade"] * 15.0 / 10.0                          # per age_z unit (15 years)
    w[3] = m["per_extra_comorbidity"]
    sev = m["severity"]
    w[4], w[5] = sev["mild"] - sev["moderate"], sev["severe"] - sev["moderate"]
    w[6] = m["adherence_per_pp"] * 20.0                                # per adherence_z unit (20 pp)
    for c, cond in vocab.CONDITIONS.items():
        w[_COND_IDX[c]] = cond.base_improve_logit + sev["moderate"]    # moderate severity is the reference level
    seen: dict[str, float] = {}
    for cond in vocab.CONDITIONS.values():                             # a drug's effect is the same wherever it is used
        for d in cond.drugs:
            seen.setdefault(d.name, d.effect)
    for name, eff in seen.items():
        w[_DRUG_IDX[name]] = eff
    return w
