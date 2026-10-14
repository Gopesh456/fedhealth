"""Data generator CLI: the built-in synthetic world, or a SYNTHEA population mapped onto the three hospitals.

    python -m fedhealth.synth.generate --patients 5000 --seed 42 [--out DIR]                    built-in generator
    python -m fedhealth.synth.generate --source synthea --synthea-csv DIR [--synthea-csv DIR2]   Synthea (synth/synthea.py)

Writes (SYNTHETIC DATA ONLY):
    <out>/hospital_a.db  hospital_b.db  hospital_c.db     one SQLite file per hospital
    <out>/ground_truth/universe.csv episodes.csv generator_params.json
The ground_truth/ folder exists only for tests, evaluation and the dashboard's demo mode.
No service ever opens it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import Session

from fedhealth.common import vocab
from fedhealth.common.config import repo_root, data_dir, load_topology
from fedhealth.hospital import models as m
from fedhealth.privacy.encryption.aesgcm import NullVault, Vault, load_or_create_key
from fedhealth.privacy.encryption.column_types import attach_vault
from fedhealth.synth.registry import build_registry
from fedhealth.synth.universe import Person, build_universe

IMAGING_CODES = {"ECG": "ECG", "Chest X-ray": "CXR", "Renal ultrasound": "RUS", "Echocardiogram": "ECHO"}


def _flag(value: float, lo: float, hi: float) -> str:
    return "H" if value > hi else "L" if value < lo else "N"


def _registered(rng: np.random.Generator, earliest: date) -> date:
    return max(date(2015, 1, 1), earliest - timedelta(days=int(rng.integers(0, 181))))


def _write_db(path: Path, base, tables: dict, vault) -> None:
    """Create a fresh SQLite file and bulk-insert {Model: [row dicts]}. Sensitive columns are encrypted
    on the way in by the SensitiveStr column type, using this hospital's vault."""
    if path.exists():
        path.unlink()
    engine = attach_vault(create_engine(f"sqlite:///{path}"), vault)
    base.metadata.create_all(engine)
    with Session(engine) as s:
        for model, rows in tables.items():
            if rows:
                s.execute(insert(model), rows)
        s.commit()
    engine.dispose()


def generate(n_patients: int = 5000, seed: int = 42, out_dir: str | Path | None = None,
             encrypt: bool = True, key_dir: str | Path | None = None, registry_people: int | None = None,
             source: str = "synthetic", synthea_csv: list[str | Path] | None = None) -> dict:
    """encrypt=True (default): sensitive columns are AES-GCM encrypted with a per-hospital key that is created
    in `key_dir` (default: ./secrets or $FEDHEALTH_SECRETS_DIR) - NOT next to the databases.
    encrypt=False writes plaintext; it exists only as the control for the encryption-at-rest tests.
    registry_people: size of each hospital's federated-learning training registry (default 30% of n_patients; 0 = none).
    source="synthea": people come from the Synthea CSV export(s) in `synthea_csv` (default: the shipped extract in
    ./synthea, see scripts/synthea_extract.py). n_patients is then a cap (None = everyone); the federated-learning
    registries stay synthetic - they are horizontal data with their own generator."""
    out = Path(out_dir) if out_dir else data_dir()
    out.mkdir(parents=True, exist_ok=True)
    gt_dir = out / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    if source == "synthea":
        from fedhealth.synth.synthea import load_synthea
        people = load_synthea(synthea_csv or [repo_root() / "synthea"], seed, limit=n_patients)
        n_patients = len(people)
    elif source == "synthetic":
        people = build_universe(n_patients, seed)
    else:
        raise ValueError(f"unknown source {source!r}")
    rng = np.random.default_rng(seed + 1000)          # slice-level randomness, independent of the universe

    # Hospital-local MRNs: shuffled so the MRN order carries no information about the person.
    members = {h: [p for p in people if h in p.membership] for h in "ABC"}
    mrn: dict[str, dict[int, str]] = {}
    for h, plist in members.items():
        order = rng.permutation(len(plist))
        mrn[h] = {plist[i].person_id: f"{h}-{rank + 1:06d}" for rank, i in enumerate(order)}

    files = {h: load_topology().hospitals[h].db_file for h in "ABC"}     # filenames come from topology.yaml
    kd = Path(key_dir) if key_dir else None
    vaults = {h: (Vault(load_or_create_key(f"hospital_{h.lower()}", kd)) if encrypt else NullVault()) for h in "ABC"}
    n_reg = round(0.3 * n_patients) if registry_people is None else registry_people
    registries = {h: (build_registry(h, n_reg, seed) if n_reg else []) for h in "ABC"}      # horizontal data for FedAvg
    slices = {"A": _slice_a(people, mrn["A"], rng), "B": _slice_b(people, mrn["B"], rng), "C": _slice_c(people, mrn["C"], rng)}
    for h in "ABC":
        slices[h][m.TRAINING_EPISODE[h]] = registries[h]
    _write_db(out / files["A"], m.ABase, slices["A"], vaults["A"])
    _write_db(out / files["B"], m.BBase, slices["B"], vaults["B"])
    _write_db(out / files["C"], m.CBase, slices["C"], vaults["C"])
    _write_registry_csv(gt_dir, registries)
    _write_ground_truth(gt_dir, people, seed, n_patients, members, source)

    return {
        "source": source, "patients": n_patients, "seed": seed, "out": str(out),
        "hospital_sizes": {h: len(v) for h, v in members.items()},
        "membership": dict(Counter(p.membership for p in people)),
        "registry_rows": {h: len(r) for h, r in registries.items()},
    }


# --------------------------------------------------------------------------- hospital slices
def _slice_a(people: list[Person], mrn: dict[int, str], rng: np.random.Generator) -> dict:
    patients, conditions = [], []
    for p in people:
        if "A" not in p.membership:
            continue
        pid = mrn[p.person_id]
        patients.append(dict(
            patient_id=pid, national_id=p.national_id, full_name=p.full_name, date_of_birth=p.dob.isoformat(),
            birth_year=p.birth_year, sex=p.sex, zip_code=p.zip_code,
            registered_on=_registered(rng, min(e.diagnosed_on for e in p.episodes)),
        ))
        for e in p.episodes:
            conditions.append(dict(
                patient_id=pid, icd10_code=e.condition.code, condition_name=e.condition.name, is_chronic=True,
                diagnosed_on=e.diagnosed_on, status=e.status, severity=e.severity, notes=e.note,
            ))
    return {m.APatient: patients, m.ACondition: conditions}


def _slice_b(people: list[Person], mrn: dict[int, str], rng: np.random.Generator) -> dict:
    patients, reports, outcomes = [], [], []
    for p in people:
        if "B" not in p.membership:
            continue
        pid = mrn[p.person_id]
        first_date = min(e.baseline_lab_date or e.diagnosed_on for e in p.episodes)
        patients.append(dict(patient_id=pid, national_id=p.national_id, registered_on=_registered(rng, first_date)))
        for e in p.episodes:
            lab = e.condition.lab
            if e.baseline_value is not None:
                reports.append(dict(                                   # baseline lab
                    patient_id=pid, report_type="lab", test_code=lab.code, value=e.baseline_value, unit=lab.unit,
                    ref_low=lab.ref_low, ref_high=lab.ref_high,
                    abnormal_flag=_flag(e.baseline_value, lab.ref_low, lab.ref_high),
                    imaging_finding=None, performed_on=e.baseline_lab_date, ordering_reason_icd10=e.condition.code,
                ))
            if e.treated and e.outcome:                                # a follow-up exists (always, for the generator)
                reports.append(dict(                                   # follow-up lab
                    patient_id=pid, report_type="lab", test_code=lab.code, value=e.followup_value, unit=lab.unit,
                    ref_low=lab.ref_low, ref_high=lab.ref_high,
                    abnormal_flag=_flag(e.followup_value, lab.ref_low, lab.ref_high),
                    imaging_finding=None, performed_on=e.assessed_on, ordering_reason_icd10=e.condition.code,
                ))
                outcomes.append(dict(
                    patient_id=pid, condition_icd10=e.condition.code, outcome=e.outcome,
                    baseline_value=e.baseline_value, followup_value=e.followup_value, delta=e.delta,
                    assessed_on=e.assessed_on,
                ))
            if e.imaging:
                modality, finding, when = e.imaging
                reports.append(dict(
                    patient_id=pid, report_type="imaging", test_code=IMAGING_CODES[modality], value=None, unit=None,
                    ref_low=None, ref_high=None, abnormal_flag=None, imaging_finding=finding,
                    performed_on=when, ordering_reason_icd10=e.condition.code,
                ))
    return {m.BPatient: patients, m.BDiagnosticReport: reports, m.BFollowupOutcome: outcomes}


def _slice_c(people: list[Person], mrn: dict[int, str], rng: np.random.Generator) -> dict:
    patients, rx_rows = [], []
    for p in people:
        if "C" not in p.membership:
            continue
        pid = mrn[p.person_id]
        first_rx = min(rx.start for e in p.episodes for rx in e.prescriptions)   # everyone has >= 1 treated condition
        patients.append(dict(patient_id=pid, national_id=p.national_id, registered_on=_registered(rng, first_rx)))
        for e in p.episodes:
            for rx in e.prescriptions:
                rx_rows.append(dict(
                    patient_id=pid, drug_name=rx.drug.name, atc_code=rx.drug.atc, drug_class=rx.drug.drug_class,
                    dose_mg=rx.drug.dose_mg, frequency=rx.drug.frequency, start_date=rx.start, end_date=rx.end,
                    reason_icd10=e.condition.code, adherence_pct=rx.adherence, prescriber_id=rx.prescriber,
                    notes=rx.note,
                ))
    return {m.CPatient: patients, m.CPrescription: rx_rows}


# --------------------------------------------------------------------------- ground truth (eval/test only)
def _write_ground_truth(gt_dir: Path, people: list[Person], seed: int, n: int, members: dict,
                        source: str = "synthetic") -> None:
    with open(gt_dir / "universe.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["person_id", "national_id", "full_name", "sex", "birth_year", "zip_code", "membership"])
        for p in people:
            w.writerow([p.person_id, p.national_id, p.full_name, p.sex, p.birth_year, p.zip_code, p.membership])
    with open(gt_dir / "episodes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["person_id", "condition", "severity", "diagnosed_on", "drugs", "adherences",
                    "outcome", "lab_delta", "n_conditions"])
        for p in people:
            for e in p.episodes:
                w.writerow([
                    p.person_id, e.condition.code, e.severity, e.diagnosed_on.isoformat(),
                    "|".join(r.drug.name for r in e.prescriptions),
                    "|".join(repr(r.adherence) for r in e.prescriptions),
                    e.outcome or "", repr(e.delta) if e.delta is not None and e.treated else "", len(p.episodes),
                ])
    params = {
        "source": source, "seed": seed, "n_patients": n, "as_of": vocab.AS_OF.isoformat(),
        "membership_fractions": vocab.MEMBERSHIP_FRACTIONS,
        "membership_counts": dict(Counter(p.membership for p in people)),
        "hospital_sizes": {h: len(v) for h, v in members.items()},
        "response_model": vocab.RESPONSE_MODEL if source == "synthetic" else None,
        "conditions": None if source != "synthetic" else {
            code: {
                "name": c.name, "intercept": c.intercept, "age_slope": c.age_slope, "male_effect": c.male_effect,
                "base_improve_logit": c.base_improve_logit,
                "drugs": {d.name: {"weight": d.weight, "effect": d.effect} for d in c.drugs},
            } for code, c in vocab.CONDITIONS.items()
        },
        "episodes_per_condition": dict(Counter(e.condition.code for p in people for e in p.episodes)),
        "outcome_counts": dict(Counter(e.outcome for p in people for e in p.episodes if e.outcome)),
    }
    (gt_dir / "generator_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


REGISTRY_COLUMNS = ["split", "age", "male", "extra_comorbidities", "severity", "condition", "drugs", "adherence_pct", "baseline_lab_z", "responded"]


def _write_registry_csv(gt_dir: Path, registries: dict) -> None:
    """Plain copy of the training registries for OFFLINE evaluation only (never read by a service)."""
    for h, rows in registries.items():
        with open(gt_dir / f"registry_{h}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(REGISTRY_COLUMNS)
            for r in rows:
                w.writerow([r[c] for c in REGISTRY_COLUMNS])


def db_content_digest(path: str | Path) -> str:
    """Hash of a SQLite file's *logical* content (schema + rows) - used to test determinism."""
    con = sqlite3.connect(str(path))
    try:
        h = hashlib.sha256()
        for line in con.iterdump():
            h.update(line.encode("utf-8"))
        return h.hexdigest()
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the hospital databases from Synthea (default) or the "
                                             "parametric synthetic generator.")
    ap.add_argument("--source", choices=("synthea", "synthetic"), default="synthea")
    ap.add_argument("--synthea-csv", action="append", default=[],
                    help="a Synthea CSV export directory, raw or extracted (repeatable; default: ./synthea)")
    ap.add_argument("--patients", type=int, default=None,
                    help="synthetic: population size (default 5000); synthea: optional cap (default: everyone)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output directory (default: ./data or $FEDHEALTH_DATA_DIR)")
    ap.add_argument("--no-encrypt", action="store_true", help="write PLAINTEXT databases (control / evaluation only)")
    args = ap.parse_args()
    n = args.patients if args.patients is not None or args.source == "synthea" else 5000
    summary = generate(n, args.seed, args.out, encrypt=not args.no_encrypt, source=args.source,
                       synthea_csv=args.synthea_csv or None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
