"""Ground-truth answers computed straight from the generator's CSV export.

This is an INDEPENDENT implementation of the query semantics: it never touches the hospital databases
or services, so agreement between it and the (plain / PSI / MPC) pipelines is a real cross-check.
Used by tests, the evaluation scripts and the dashboard's "ground truth (demo mode only)" toggle.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from fedhealth.common import vocab
from fedhealth.common.schemas import APredicate, BPredicate, CPredicate, QuerySpec, categories, participating_hospitals, value_bounds
from fedhealth.common.config import load_fields


@dataclass(frozen=True)
class _Person:
    national_id: str
    sex: str
    birth_year: int
    membership: str


@dataclass(frozen=True)
class _Episode:
    person_id: int
    condition: str
    drugs: tuple[str, ...]
    adherences: tuple[float, ...]
    outcome: str          # "" if untreated
    lab_delta: float | None


class GroundTruth:
    def __init__(self, gt_dir: str | Path):
        gt_dir = Path(gt_dir)
        self.persons: dict[int, _Person] = {}
        with open(gt_dir / "universe.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                self.persons[int(r["person_id"])] = _Person(r["national_id"], r["sex"], int(r["birth_year"]), r["membership"])
        self.episodes: list[_Episode] = []
        with open(gt_dir / "episodes.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                self.episodes.append(_Episode(
                    int(r["person_id"]), r["condition"],
                    tuple(x for x in r["drugs"].split("|") if x),
                    tuple(float(x) for x in r["adherences"].split("|") if x),
                    r["outcome"], float(r["lab_delta"]) if r["lab_delta"] else None,
                ))

    # ------------------------------------------------------------------ matching
    def matches(self, spec: QuerySpec) -> list[tuple[str, int | None, str | None]]:
        """[(national_id, scaled_value, category)] for every patient the query counts."""
        hospitals = set(participating_hospitals(spec))
        a = spec.a or APredicate()
        b = spec.b or BPredicate()
        c = spec.c or CPredicate()
        fields = load_fields()
        out = []
        for e in self.episodes:
            if e.condition != spec.condition:
                continue
            p = self.persons[e.person_id]
            age = vocab.age_of(p.birth_year)
            if "A" in hospitals:
                if "A" not in p.membership:
                    continue
                if a.age_min is not None and age < a.age_min:
                    continue
                if a.age_max is not None and age > a.age_max:
                    continue
                if a.sex is not None and p.sex != a.sex:
                    continue
            if "B" in hospitals:
                if "B" not in p.membership or not e.outcome:
                    continue
                if b.outcome is not None and e.outcome != b.outcome:
                    continue
            matching_idx = list(range(len(e.drugs)))
            if "C" in hospitals:
                if "C" not in p.membership or not e.drugs:
                    continue
                if c.drug is not None:
                    matching_idx = [i for i, d in enumerate(e.drugs) if d == c.drug]
                    if not matching_idx:
                        continue
            primary = matching_idx[0] if matching_idx else None      # drugs are stored in start-date order

            value = category = None
            if spec.value_field:
                lo, hi = value_bounds(spec.value_field, spec.condition)
                scale = fields.value_fields[spec.value_field].scale
                raw = {"age": age, "lab_delta": e.lab_delta,
                       "adherence_pct": e.adherences[primary] if primary is not None else None}[spec.value_field]
                value = max(lo, min(hi, int(round(raw * scale))))
            if spec.group_by:
                category = {"sex": p.sex, "age_band": vocab.age_band_of(age), "outcome": e.outcome or None,
                            "drug": e.drugs[primary] if primary is not None else None}[spec.group_by]
            out.append((p.national_id, value, category))
        return out

    def evaluate(self, spec: QuerySpec):
        rows = self.matches(spec)
        if spec.agg == "count":
            return len(rows)
        if spec.agg == "sum":
            return sum(v for _, v, _ in rows)
        if spec.agg == "avg":
            return (sum(v for _, v, _ in rows) / len(rows)) if rows else None
        hist = {cat: 0 for cat in categories(spec.group_by, spec.condition)}
        for _, _, cat in rows:
            hist[cat] += 1
        return hist
