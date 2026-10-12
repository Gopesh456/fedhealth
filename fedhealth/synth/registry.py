"""Per-hospital TRAINING registries for the federated-learning stretch goal.

WHY A SEPARATE TABLE
    FedAvg needs HORIZONTALLY partitioned data: every client holds complete rows with the same columns. The A / B / C
    slices are VERTICAL (history | reports | prescriptions of the same patients), so no hospital can train the
    response model alone. Each hospital therefore also keeps a `training_episodes` table: patients it treated and
    followed up end to end locally (features + whether they responded), identical schema at all three.

NON-IID ON PURPOSE
    Real hospitals do not see the same patient mix. Each registry is a biased sample of the same underlying clinical
    process (covariate shift): hospital A skews younger and more severe, B older, C favours cardiovascular drugs. The
    CONDITIONAL model P(response | x) is the same everywhere, so a single global model is the right goal - but a model
    trained on one hospital alone generalises worse. That is exactly the situation federated learning is for.
"""
from __future__ import annotations

import numpy as np

from fedhealth.common import vocab
from fedhealth.synth.universe import build_universe

SEED_OFFSET = {"A": 101, "B": 202, "C": 303}
CARDIO = {"C07", "C08", "C09", "C10", "C03"}               # ATC prefixes C07/C08/C09/C10/C03: heart & statin drugs


def _accept_probability(hospital: str, age: int, severity: str, drugs: list[str]) -> float:
    if hospital == "A":                                     # tertiary centre: younger, sicker
        return (0.95 if age < 55 else 0.55) * (1.0 if severity != "mild" else 0.7)
    if hospital == "B":                                     # older population
        return 0.95 if age >= 60 else 0.5
    atc = {d.name: d.atc for c in vocab.CONDITIONS.values() for d in c.drugs}
    cardio = any(atc[d][:3] in CARDIO for d in drugs)       # C: pharmacy-led cardiovascular programme
    return 0.95 if cardio else 0.45


def build_registry(hospital: str, n_people: int, seed: int) -> list[dict]:
    """Rows for one hospital's training_episodes table (75% 'train' / 25% 'test', deterministic per seed)."""
    people = build_universe(n_people, seed + SEED_OFFSET[hospital])
    rng = np.random.default_rng(seed + SEED_OFFSET[hospital] + 7)
    rows = []
    for p in people:
        for e in p.episodes:
            if not e.treated:
                continue
            drugs = [r.drug.name for r in e.prescriptions]
            if rng.random() > _accept_probability(hospital, p.age, e.severity, drugs):
                continue
            lab = e.condition.lab
            rows.append({
                "split": "train" if rng.random() < 0.75 else "test",
                "age": p.age, "male": int(p.sex == "M"), "extra_comorbidities": len(p.episodes) - 1,
                "severity": e.severity, "condition": e.condition.code, "drugs": "|".join(drugs),
                "adherence_pct": round(float(np.mean([r.adherence for r in e.prescriptions])), 1),
                "baseline_lab_z": round((e.baseline_value - lab.baseline_mean) / lab.baseline_sd, 3),
                "responded": int(e.outcome == "improved"),
            })
    return rows


