"""Synthea mapper: a hand-written four-file export with known answers, and the shipped extract."""
from __future__ import annotations

import csv
import gzip
from datetime import date

from fedhealth.synth.generate import db_content_digest, generate
from fedhealth.synth.synthea import load_synthea

PATIENTS = ["Id", "BIRTHDATE", "DEATHDATE", "SSN", "FIRST", "LAST", "GENDER", "ZIP"]
CONDITIONS = ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"]
MEDICATIONS = ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE"]
OBSERVATIONS = ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "VALUE"]


def _write(path, name, header, rows, packed=False):
    opener = (lambda: gzip.open(path / f"{name}.gz", "wt", newline="", encoding="utf-8")) if packed \
        else (lambda: open(path / name, "w", newline="", encoding="utf-8"))
    with opener() as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _export(path, packed=False):
    _write(path, "patients.csv", PATIENTS, [
        ["p1", "1960-04-02", "", "999-12-3456", "Ana123", "Silva45", "F", "2134"],
        ["p2", "2015-01-01", "", "999-00-0001", "Kid1", "Young2", "M", "02139"],      # under 18: dropped
        ["p3", "1950-01-01", "2020-05-05", "999-00-0002", "Old1", "Gone2", "M", ""],   # died before AS_OF: dropped
        ["p4", "1970-01-01", "", "999-00-0003", "No1", "Study2", "M", "02139"],        # no study condition: dropped
    ], packed)
    _write(path, "conditions.csv", CONDITIONS, [
        ["2018-03-01", "", "p1", "e1", "44054006", "Diabetes mellitus type 2"],
        ["2019-06-01", "", "p1", "e2", "368581000119106", "Neuropathy due to type 2 diabetes"],   # one complication
        ["2030-01-01", "", "p1", "e9", "59621000", "Hypertension (after AS_OF: ignored)"],
        ["2018-01-01", "", "p2", "e3", "44054006", "Diabetes mellitus type 2"],
        ["2018-01-01", "", "p4", "e4", "10509002", "Acute bronchitis"],
    ], packed)
    _write(path, "medications.csv", MEDICATIONS, [
        ["2018-04-01", "2018-10-01", "p1", "e5", "860975", "24 HR Metformin hydrochloride 500 MG", "44054006"],
        ["2018-10-01", "", "p1", "e6", "860975", "24 HR Metformin hydrochloride 500 MG", "44054006"],   # refill: merged
        ["2018-04-01", "", "p1", "e5", "314076", "lisinopril 10 MG", "59621000"],   # reason is not a held condition
    ], packed)
    _write(path, "observations.csv", OBSERVATIONS, [
        ["2018-03-15", "p1", "e1", "4548-4", "Hemoglobin A1c", "8.6"],     # baseline (last before first fill)
        ["2018-09-01", "p1", "e5", "4548-4", "Hemoglobin A1c", "7.9"],
        ["2019-09-01", "p1", "e6", "4548-4", "Hemoglobin A1c", "7.4"],     # follow-up (last >= 90 days later)
        ["2031-01-01", "p1", "e7", "4548-4", "Hemoglobin A1c", "5.0"],     # after AS_OF: ignored
    ], packed)


def test_mapping_of_a_known_export(tmp_path):
    _export(tmp_path)
    people = load_synthea([tmp_path])
    assert [p.national_id for p in people] == ["SYN-999123456"]         # minors, the dead, non-study people dropped
    p = people[0]
    assert (p.full_name, p.sex, p.birth_year, p.zip_code) == ("Ana Silva", "F", 1960, "02134")
    assert [e.condition.code for e in p.episodes] == ["E11"]             # the post-AS_OF hypertension is ignored
    ep = p.episodes[0]
    assert ep.diagnosed_on == date(2018, 3, 1) and ep.status == "active"
    assert ep.severity == "moderate"                                     # one coded complication
    assert [(rx.drug.name, rx.start, rx.end, rx.adherence) for rx in ep.prescriptions] == \
        [("metformin", date(2018, 4, 1), None, 100.0)]                    # two fills merged, lisinopril not counted
    assert (ep.baseline_value, ep.followup_value, ep.delta, ep.outcome) == (8.6, 7.4, -1.2, "improved")
    assert ep.assessed_on == date(2019, 9, 1)


def test_gzipped_export_reads_the_same(tmp_path):
    plain, packed = tmp_path / "plain", tmp_path / "packed"
    plain.mkdir()
    packed.mkdir()
    _export(plain)
    _export(packed, packed=True)
    assert load_synthea([plain]) == load_synthea([packed])


def test_shipped_extract_builds_deterministic_worlds(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    sa = generate(400, seed=3, out_dir=a, encrypt=False, source="synthea")
    generate(400, seed=3, out_dir=b, encrypt=False, source="synthea")
    assert sa["source"] == "synthea" and sa["patients"] == 400
    for f in ("hospital_a.db", "hospital_b.db", "hospital_c.db"):
        assert db_content_digest(a / f) == db_content_digest(b / f)
