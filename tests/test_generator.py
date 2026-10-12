"""Synthetic-data generator: structure, internal consistency and determinism."""
from __future__ import annotations

import csv
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import date

import pytest

from fedhealth.common import vocab
from fedhealth.synth.generate import db_content_digest, generate


def _rows(path, sql):
    con = sqlite3.connect(str(path))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ population structure
def test_overlap_structure_matches_spec(plain_world):
    n = 5000
    counts = plain_world["summary"]["membership"]
    for key, frac in vocab.MEMBERSHIP_FRACTIONS.items():
        assert abs(counts[key] - frac * n) <= 1, (key, counts[key])
    sizes = plain_world["summary"]["hospital_sizes"]
    assert sizes == {"A": 3750, "B": 3150, "C": 3350}


def test_there_are_overlapping_and_unique_patients(plain_world):
    ids = {h: {r[0] for r in _rows(plain_world["dir"] / f"hospital_{h.lower()}.db", "select national_id from patients")}
           for h in "ABC"}
    assert len(ids["A"] & ids["B"] & ids["C"]) == 2000
    assert len(ids["A"] - ids["B"] - ids["C"]) == 750          # unique to A
    assert len(ids["B"] - ids["A"] - ids["C"]) == 500
    assert len(ids["C"] - ids["A"] - ids["B"]) == 500


def test_identifiers(plain_world):
    d = plain_world["dir"]
    nid = {h: [r[0] for r in _rows(d / f"hospital_{h.lower()}.db", "select national_id from patients")] for h in "ABC"}
    for h in "ABC":
        assert len(nid[h]) == len(set(nid[h])), "national_id must be unique within a hospital"
        assert all(re.fullmatch(r"SYN-\d{9}", x) for x in nid[h]), "ids must be obviously synthetic"
    mrns = {h: [r[0] for r in _rows(d / f"hospital_{h.lower()}.db", "select patient_id from patients")] for h in "ABC"}
    for h in "ABC":
        assert all(re.fullmatch(rf"{h}-\d{{6}}", x) for x in mrns[h])
    # each hospital has its own local ids: same person, different MRN, same linkage key
    assert mrns["A"][0].startswith("A-") and mrns["B"][0].startswith("B-")


def test_no_orphan_rows(plain_world):
    d = plain_world["dir"]
    assert _rows(d / "hospital_a.db", "select count(*) from conditions c left join patients p using(patient_id) where p.patient_id is null")[0][0] == 0
    assert _rows(d / "hospital_b.db", "select count(*) from diagnostic_reports c left join patients p using(patient_id) where p.patient_id is null")[0][0] == 0
    assert _rows(d / "hospital_b.db", "select count(*) from followup_outcomes c left join patients p using(patient_id) where p.patient_id is null")[0][0] == 0
    assert _rows(d / "hospital_c.db", "select count(*) from prescriptions c left join patients p using(patient_id) where p.patient_id is null")[0][0] == 0


# ------------------------------------------------------------------ cross-hospital consistency
@pytest.fixture(scope="module")
def story(plain_world):
    """Per (national_id, condition): everything the three hospitals say about it, joined on the linkage key."""
    d = plain_world["dir"]
    dx = {(n, c): date.fromisoformat(s) for n, c, s in _rows(
        d / "hospital_a.db",
        "select p.national_id, c.icd10_code, c.diagnosed_on from conditions c join patients p using(patient_id)")}
    rx = defaultdict(list)
    for n, reason, drug, start, end in _rows(
            d / "hospital_c.db",
            "select p.national_id, r.reason_icd10, r.drug_name, r.start_date, r.end_date "
            "from prescriptions r join patients p using(patient_id)"):
        rx[(n, reason)].append((drug, date.fromisoformat(start), date.fromisoformat(end) if end else None))
    out = {}
    for n, c, o, delta, base, fol, assessed in _rows(
            d / "hospital_b.db",
            "select p.national_id, f.condition_icd10, f.outcome, f.delta, f.baseline_value, f.followup_value, f.assessed_on "
            "from followup_outcomes f join patients p using(patient_id)"):
        out[(n, c)] = (o, delta, base, fol, date.fromisoformat(assessed))
    return dx, rx, out


def test_dates_are_ordered_across_hospitals(story):
    dx, rx, out = story
    checked = 0
    for key, d0 in dx.items():
        assert d0 <= vocab.AS_OF
        for drug, start, end in rx.get(key, []):
            assert d0 < start <= vocab.AS_OF                      # diagnosis strictly before first prescription
            assert end is None or start <= end <= vocab.AS_OF
            checked += 1
        if key in out and key in rx:
            assessed = out[key][4]
            assert min(s for _, s, _ in rx[key]) < assessed <= vocab.AS_OF
    assert checked > 1000, "test would be vacuous without overlapping A/C patients"


def test_prescriptions_and_outcomes_refer_to_real_conditions(plain_world):
    """Every prescription reason / outcome condition is a condition the person truly has, and the drug
    is in that condition's formulary. Uses the ground-truth export (covers people missing from Hospital A)."""
    d = plain_world["dir"]
    persons = {r["national_id"]: r["person_id"] for r in _csv(plain_world["gt_dir"] / "universe.csv")}
    truth = {(r["person_id"], r["condition"]): r for r in _csv(plain_world["gt_dir"] / "episodes.csv")}
    for n, reason, drug in _rows(d / "hospital_c.db",
                                 "select p.national_id, r.reason_icd10, r.drug_name from prescriptions r join patients p using(patient_id)"):
        ep = truth[(persons[n], reason)]
        assert drug in ep["drugs"].split("|")
        assert drug in vocab.drug_names(reason)
    for n, cond in _rows(d / "hospital_b.db",
                         "select p.national_id, f.condition_icd10 from followup_outcomes f join patients p using(patient_id)"):
        assert truth[(persons[n], cond)]["drugs"], "an outcome must follow a treatment"
    for cond, in _rows(d / "hospital_b.db", "select ordering_reason_icd10 from diagnostic_reports"):
        assert cond in vocab.CONDITIONS


def test_outcome_is_consistent_with_lab_delta(story):
    _, _, out = story
    assert len(out) > 4000
    for (n, cond), (outcome, delta, base, fol, _) in out.items():
        lab = vocab.CONDITIONS[cond].lab
        assert delta == pytest.approx(fol - base, abs=0.011)
        signed = lab.better * delta
        if outcome == "improved":
            assert signed >= lab.threshold - 1e-9
        elif outcome == "worsened":
            assert signed <= -lab.threshold + 1e-9
        else:
            assert abs(delta) < lab.threshold
        assert abs(delta) <= lab.clamp + 1e-9                 # the SUM clamp never actually truncates real data
        assert fol > 0 and base > 0


def test_patients_are_adults_at_diagnosis(plain_world):
    d = plain_world["dir"]
    for dob, dxd in _rows(d / "hospital_a.db",
                          "select p.date_of_birth, min(c.diagnosed_on) from patients p join conditions c using(patient_id) group by p.patient_id"):
        born, dx = date.fromisoformat(dob), date.fromisoformat(dxd)
        assert (dx - born).days >= 18 * 365


def test_data_has_signal_and_small_cells(ground_truth):
    """The outcome model must give a plausible response rate, and rare drugs must create small cells
    (otherwise the k-threshold phase would have nothing to suppress)."""
    outs = Counter(e.outcome for e in ground_truth.episodes if e.outcome)
    rate = outs["improved"] / sum(outs.values())
    assert 0.4 < rate < 0.65
    from fedhealth.common.schemas import QuerySpec
    small = [d for d in vocab.drug_names("I50")
             if ground_truth.evaluate(QuerySpec.model_validate(
                 {"agg": "count", "condition": "I50", "a": {}, "b": {"outcome": "improved"}, "c": {"drug": d}})) < 10]
    assert small, "expected at least one rare (<10) cell"


# ------------------------------------------------------------------ determinism & hygiene
def test_generation_is_deterministic(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    generate(300, seed=7, out_dir=a, encrypt=False)
    generate(300, seed=7, out_dir=b, encrypt=False)
    generate(300, seed=8, out_dir=c, encrypt=False)
    for f in ("hospital_a.db", "hospital_b.db", "hospital_c.db"):
        assert db_content_digest(a / f) == db_content_digest(b / f)
        assert db_content_digest(a / f) != db_content_digest(c / f)
    for f in ("universe.csv", "episodes.csv"):
        assert (a / "ground_truth" / f).read_bytes() == (b / "ground_truth" / f).read_bytes()


def test_generator_does_not_use_the_network():
    import ast
    import pathlib
    for p in pathlib.Path(__file__).parents[1].joinpath("fedhealth", "synth").glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not mods & {"socket", "requests", "httpx", "urllib", "http"}, p.name
