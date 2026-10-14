"""Load a SYNTHEA population into the three hospital databases.

    java -jar synthea-with-dependencies.jar -s 42 -p 20000 --exporter.csv.export=true Massachusetts
    python -m fedhealth.synth.generate --source synthea --synthea-csv output/csv [--synthea-csv more/csv]

Synthea (Walonoski et al., JAMIA 2018) simulates whole life histories - encounters, SNOMED-CT conditions, LOINC
observations, RxNorm prescriptions - from published clinical guidelines and population statistics. Using it removes
the objection that the evaluation is tuned to our own generator. This module maps a Synthea CSV export onto the
SAME intermediate `Person` / `Episode` objects as the built-in generator (synth/universe.py), so the hospital
slices, the encryption at rest and the ground-truth files are produced by exactly the same code.

THE MAPPING (documented, deterministic)
    people       alive adults (>= 18) at vocab.AS_OF with at least one of the eight study conditions.
                 national_id = "SYN-" + SSN digits; Synthea's digit suffixes are stripped from names.
    conditions   SNOMED-CT -> ICD-10 group (SNOMED_TO_ICD10). Diagnosis date = earliest onset <= AS_OF.
    drugs        RxNorm description keyword -> vocabulary drug (RX_KEYWORDS), counted only when Synthea's
                 REASONCODE is a SNOMED code of that condition (epoetin: prescribed for the anaemia of CKD).
                 Repeated fills of one drug are merged into one prescription (first start .. last stop).
    adherence    Synthea does not simulate adherence. `adherence_pct` is the share of days between the first start
                 and the last stop (or AS_OF) covered by a fill - a refill-continuity proxy that is mostly 100.
    labs         LOINC per condition (LAB_LOINC). FEV1 % predicted is not simulated by Synthea; FEV1/FVC (%) stands
                 in for it. Baseline = the last value up to the first prescription (or, when Synthea's history starts
                 later than the prescription, the first value after diagnosis); follow-up = the last value at least
                 90 days after the baseline and 30 days after the first prescription.
    outcome      delta = follow-up - baseline (clamped to the vocabulary bound): improved / worsened if it moves by
                 >= the lab's clinical threshold in the good / bad direction, else no_change. A treated episode
                 without both lab values has no outcome (it is invisible to Hospital B queries).
    severity     from the baseline lab relative to the reference range (or the CKD stage).
    membership   which hospitals hold a person: the same seeded draw as the built-in generator
                 (vocab.MEMBERSHIP_FRACTIONS), then B is dropped for people without any lab and C for people without
                 any mapped prescription (a hospital cannot hold records that do not exist).
Events after vocab.AS_OF are ignored.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from fedhealth.common import vocab
from fedhealth.common.vocab import AS_OF, CONDITIONS
from fedhealth.synth.universe import Episode, Person, Rx

SNOMED_TO_ICD10: dict[str, str] = {
    # E11 type 2 diabetes (and its coded complications)
    "44054006": "E11", "127013003": "E11", "90781000119102": "E11", "157141000119108": "E11",
    "368581000119106": "E11", "1551000119108": "E11", "1501000119109": "E11", "97331000119101": "E11",
    "60951000119105": "E11",
    "59621000": "I10",                                                   # essential hypertension
    "55822004": "E78",                                                   # hyperlipidaemia
    "195967001": "J45", "233678006": "J45",                              # asthma, childhood asthma
    "87433001": "J44", "185086009": "J44",                               # emphysema, chronic obstructive bronchitis
    "431855005": "N18", "431856006": "N18", "433144002": "N18", "431857002": "N18", "46177005": "N18",   # CKD 1-4, ESRD
    "88805009": "I50", "84114007": "I50",                                # (chronic congestive) heart failure
    "370143000": "F32", "36923009": "F32",                               # major depressive disorder / single episode
}
CKD_STAGE = {"431855005": 1, "431856006": 2, "433144002": 3, "431857002": 4, "46177005": 5}
E11_COMPLICATIONS = set(k for k, v in SNOMED_TO_ICD10.items() if v == "E11") - {"44054006"}

# (lower-case substring of the RxNorm description, vocabulary drug name, conditions it may be counted for)
RX_KEYWORDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("metformin", "metformin", ("E11",)),
    ("insulin isophane", "insulin NPH/regular 70/30", ("E11",)),
    ("liraglutide", "liraglutide", ("E11",)),
    ("hydrochlorothiazide", "hydrochlorothiazide", ("I10",)),
    ("amlodipine", "amlodipine", ("I10",)),
    ("lisinopril", "lisinopril", ("I10", "I50")),
    ("losartan", "losartan", ("I10", "I50", "N18")),
    ("simvastatin", "simvastatin", ("E78",)),
    ("atorvastatin", "atorvastatin", ("E78",)),
    ("salmeterol", "fluticasone-salmeterol", ("J44", "J45")),
    ("albuterol", "salbutamol", ("J44", "J45")),
    ("fluticasone", "fluticasone", ("J45",)),
    ("budesonide", "budesonide", ("J45",)),
    ("beclomethasone", "beclomethasone", ("J45",)),
    ("mometasone", "mometasone", ("J45",)),
    ("furosemide", "furosemide", ("I50",)),
    ("carvedilol", "carvedilol", ("I50",)),
    ("sacubitril", "sacubitril-valsartan", ("I50",)),
    ("fluoxetine", "fluoxetine", ("F32",)),
    ("sertraline", "sertraline", ("F32",)),
    ("escitalopram", "escitalopram", ("F32",)),
    ("bupropion", "bupropion", ("F32",)),
    ("epoetin", "epoetin alfa", ("N18",)),
)
ANAEMIA = "271737000"            # Synthea prescribes epoetin "for anaemia" - in its CKD module, the anaemia of CKD

LAB_LOINC: dict[str, tuple[str, ...]] = {
    "E11": ("4548-4",), "I10": ("8480-6",), "E78": ("18262-6",), "J45": ("19926-5",), "J44": ("19926-5",),
    "N18": ("33914-3",), "I50": ("33762-6", "71425-3"), "F32": ("44261-6",),
}

_DIGITS = re.compile(r"\d+")


def _d(s: str) -> date | None:
    return datetime.fromisoformat(s[:10]).date() if s else None


@dataclass
class _Fill:
    start: date
    stop: date | None
    encounter: str


def _read(paths: list[Path], name: str):
    """Rows of `name` from every export directory; reads the gzipped extract (scripts/synthea_extract.py) too."""
    for p in paths:
        plain, packed = p / name, p / f"{name}.gz"
        opener = (lambda: gzip.open(packed, "rt", newline="", encoding="utf-8")) if not plain.exists() and packed.exists() \
            else (lambda: open(plain, newline="", encoding="utf-8"))
        with opener() as fh:
            yield from csv.DictReader(fh)


def load_synthea(csv_dirs: list[str | Path], seed: int = 42, limit: int | None = None) -> list[Person]:
    """Map one or more Synthea CSV exports to the generator's Person objects (deterministic for a given input)."""
    dirs = [Path(d) for d in csv_dirs]
    adult_cutoff = date(AS_OF.year - 18, AS_OF.month, AS_OF.day)

    patients, seen_ssn = {}, set()
    for r in _read(dirs, "patients.csv"):
        dob, death = _d(r["BIRTHDATE"]), _d(r["DEATHDATE"])
        ssn = r["SSN"].replace("-", "")
        if dob > adult_cutoff or (death is not None and death <= AS_OF) or ssn in seen_ssn:
            continue
        seen_ssn.add(ssn)
        patients[r["Id"]] = r

    cond_rows: dict[str, dict[str, list[tuple[date, date | None, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in _read(dirs, "conditions.csv"):
        icd = SNOMED_TO_ICD10.get(r["CODE"])
        start = _d(r["START"])
        if icd and r["PATIENT"] in patients and start <= AS_OF:
            cond_rows[r["PATIENT"]][icd].append((start, _d(r["STOP"]), r["CODE"]))

    fills: dict[tuple[str, str, str], list[_Fill]] = defaultdict(list)       # (patient, condition, drug) -> fills
    for r in _read(dirs, "medications.csv"):
        pid = r["PATIENT"]
        if pid not in cond_rows:
            continue
        start = _d(r["START"])
        if start > AS_OF:
            continue
        desc, reason = r["DESCRIPTION"].lower(), r["REASONCODE"]
        reason_icd = SNOMED_TO_ICD10.get(reason)
        for key, drug, conds in RX_KEYWORDS:
            if key not in desc:
                continue
            for c in conds:
                if c not in cond_rows[pid]:
                    continue
                if reason_icd == c or (drug == "epoetin alfa" and reason == ANAEMIA):
                    fills[(pid, c, drug)].append(_Fill(start, _d(r["STOP"]), r["ENCOUNTER"]))
            break

    wanted_loinc = {code: cond for cond, codes in LAB_LOINC.items() for code in codes}
    labs: dict[tuple[str, str], list[tuple[date, float]]] = defaultdict(list)  # (patient, condition) -> [(date, value)]
    for r in _read(dirs, "observations.csv"):
        cond = wanted_loinc.get(r["CODE"])
        if cond is None or r["PATIENT"] not in cond_rows:
            continue
        when = _d(r["DATE"])
        if when > AS_OF:
            continue
        try:
            value = float(r["VALUE"])
        except ValueError:
            continue
        for c in (cond,) if cond not in ("J45", "J44") else ("J45", "J44"):
            if c in cond_rows[r["PATIENT"]]:
                labs[(r["PATIENT"], c)].append((when, value))

    ids = sorted(cond_rows)                                       # deterministic order
    if limit is not None:
        ids = ids[:limit]
    rng = np.random.default_rng(seed)
    membership = _membership(rng, len(ids))
    people: list[Person] = []
    for i, pid in enumerate(ids):
        r = patients[pid]
        dob = _d(r["BIRTHDATE"])
        p = Person(person_id=i, national_id=f"SYN-{r['SSN'].replace('-', '')}",
                   full_name=f"{_DIGITS.sub('', r['FIRST'])} {_DIGITS.sub('', r['LAST'])}",
                   sex="F" if r["GENDER"] == "F" else "M", birth_year=dob.year, dob=dob,
                   zip_code=(r["ZIP"] or "00000").zfill(5)[:5], membership="")
        for code in sorted(cond_rows[pid]):
            p.episodes.append(_episode(code, cond_rows[pid][code], fills, labs, pid))
        has_rx = any(e.prescriptions for e in p.episodes)
        has_lab = any(e.baseline_value is not None for e in p.episodes)
        m = "".join(h for h in membership[i] if (h != "B" or has_lab) and (h != "C" or has_rx))
        p.membership = m or "A"
        people.append(p)
    return people


def _membership(rng: np.random.Generator, n: int) -> list[str]:
    labels: list[str] = []
    items = list(vocab.MEMBERSHIP_FRACTIONS.items())
    for key, frac in items[:-1]:
        labels += [key] * int(round(frac * n))
    labels += [items[-1][0]] * (n - len(labels))
    perm = rng.permutation(len(labels))
    return [labels[j] for j in perm]


def _episode(code: str, rows, fills, labs, pid: str) -> Episode:
    cond = CONDITIONS[code]
    dx = min(r[0] for r in rows)
    active = any(r[1] is None or r[1] > AS_OF for r in rows)
    rxs: list[Rx] = []
    for drug in cond.drugs:
        fs = fills.get((pid, code, drug.name))
        if not fs:
            continue
        start = min(f.start for f in fs)
        open_ended = any(f.stop is None or f.stop > AS_OF for f in fs)
        end = None if open_ended else max(f.stop for f in fs)
        span_end = end or AS_OF
        covered = _covered_days(fs, span_end)
        span = max((span_end - start).days, 1)
        adherence = round(min(100.0, max(10.0, 100.0 * covered / span)), 1)
        prescriber = "DR-" + str(int(hashlib.sha256(fs[0].encounter.encode()).hexdigest(), 16) % 300 + 1).zfill(4)
        rxs.append(Rx(drug, start, end, adherence, prescriber, None))
    rxs.sort(key=lambda x: (x.start, x.drug.name))
    ep = Episode(cond, "moderate", dx, "active" if active else "resolved", None, prescriptions=rxs)

    lab = cond.lab
    series = sorted(labs.get((pid, code), []))
    window = [(d, v) for d, v in series if d >= dx - timedelta(days=30)]       # measurements from diagnosis on
    if rxs:
        first_rx = rxs[0].start
        pre = [x for x in window if x[0] <= first_rx]
        base = pre[-1] if pre else (window[0] if window else None)
        later = [x for x in window if base is not None and x[0] >= base[0] + timedelta(days=90)
                 and x[0] >= first_rx + timedelta(days=30)]
        follow = later[-1] if later else None
        if base is not None:
            ep.baseline_lab_date, ep.baseline_value = base[0], round(base[1], 2)
        if base is not None and follow is not None:
            delta = max(-lab.clamp, min(lab.clamp, follow[1] - base[1]))
            ep.followup_value = round(base[1] + delta, 2)
            ep.delta = round(ep.followup_value - ep.baseline_value, 2)
            ep.assessed_on = follow[0]
            if lab.better * ep.delta >= lab.threshold:
                ep.outcome = "improved"
            elif -lab.better * ep.delta >= lab.threshold:
                ep.outcome = "worsened"
            else:
                ep.outcome = "no_change"
    elif window:
        ep.baseline_lab_date, ep.baseline_value = window[0][0], round(window[0][1], 2)
    ep.severity = _severity(code, rows, ep.baseline_value, lab)
    return ep


def _covered_days(fs: list[_Fill], span_end: date) -> int:
    """Days in [first start, span_end] covered by at least one fill interval."""
    intervals = sorted((f.start, min(f.stop or span_end, span_end)) for f in fs)
    covered, cur_s, cur_e = 0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += (cur_e - cur_s).days
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += (cur_e - cur_s).days
    return covered


def _severity(code: str, rows, baseline: float | None, lab) -> str:
    if code == "N18":
        stage = max(CKD_STAGE.get(r[2], 1) for r in rows)
        return "mild" if stage <= 2 else "moderate" if stage == 3 else "severe"
    if code == "E11":
        n = len({r[2] for r in rows} & E11_COMPLICATIONS)
        return "mild" if n == 0 else "moderate" if n == 1 else "severe"
    if baseline is None:
        return "moderate"
    # distance outside the reference range, in units of the clinical threshold
    gap = (baseline - lab.ref_high) if lab.better < 0 else (lab.ref_low - baseline)
    return "mild" if gap <= lab.threshold else "moderate" if gap <= 4 * lab.threshold else "severe"
