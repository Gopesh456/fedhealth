"""Shrink one or more raw Synthea CSV exports to the rows FedHealth reads, as small gzipped CSVs.

A Synthea run of ~20,000 people writes several GB (claims, encounters, every observation). The mapper
(fedhealth/synth/synthea.py) needs four files and a fraction of their rows, so this keeps exactly those:

    patients.csv      every patient, only Id, BIRTHDATE, DEATHDATE, SSN, FIRST, LAST, GENDER, ZIP
    conditions.csv    rows whose SNOMED code is one of the eight mapped conditions
    medications.csv   rows of patients with a mapped condition whose reason is a mapped condition (or anaemia)
    observations.csv  rows of those patients whose LOINC code is one of the outcome labs

The extract is what the evaluation and `python -m fedhealth.synth.generate --source synthea` read by default
(./synthea/, small enough to ship with the code). Reading the raw export gives the same people - the filters only drop rows the mapper ignores.

How the shipped extract was made (Synthea master d9d07a6, Massachusetts, Java 21):
    java -jar synthea.jar -p 6000  -s 42   -cs 42   Massachusetts      (reference time 2026-10-12)
    java -jar synthea.jar -p 13000 -s 4343 -cs 4343 Massachusetts      (--exporter.csv.export=true)
    python scripts/synthea_extract.py out/csv out2/csv --out synthea
"""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

from fedhealth.synth.synthea import ANAEMIA, LAB_LOINC, SNOMED_TO_ICD10

COLUMNS = {
    "patients.csv": ["Id", "BIRTHDATE", "DEATHDATE", "SSN", "FIRST", "LAST", "GENDER", "ZIP"],
    "conditions.csv": ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"],
    "medications.csv": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE"],
    "observations.csv": ["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE", "UNITS"],
}


def _rows(dirs: list[Path], name: str):
    for d in dirs:
        with open(d / name, newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)


def extract(dirs: list[Path], out: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    loinc = {code for codes in LAB_LOINC.values() for code in codes}
    reasons = set(SNOMED_TO_ICD10) | {ANAEMIA}
    with_condition: set[str] = set()
    keep = {
        "patients.csv": lambda r: True,
        "conditions.csv": lambda r: r["CODE"] in SNOMED_TO_ICD10 and not with_condition.add(r["PATIENT"]),
        "medications.csv": lambda r: r["PATIENT"] in with_condition and r["REASONCODE"] in reasons,
        "observations.csv": lambda r: r["PATIENT"] in with_condition and r["CODE"] in loinc,
    }
    counts = {}
    for name, cols in COLUMNS.items():                          # conditions before medications/observations
        n = 0
        with gzip.open(out / f"{name}.gz", "wt", newline="", encoding="utf-8", compresslevel=9) as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in _rows(dirs, name):
                if keep[name](r):
                    w.writerow(r)
                    n += 1
        counts[name] = n
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("exports", nargs="+", help="Synthea CSV export directories (output/csv)")
    ap.add_argument("--out", default="synthea")
    args = ap.parse_args()
    for name, n in extract([Path(d) for d in args.exports], Path(args.out)).items():
        print(f"{name:18} {n:>9,} rows")


if __name__ == "__main__":
    main()
