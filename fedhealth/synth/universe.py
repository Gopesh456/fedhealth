"""Latent synthetic population ("universe").

We first build ONE internally consistent world of people - who they are, which conditions they
really have, how they were treated and how they responded - and only then project slices of it
into the three hospitals. Because every hospital slice is derived from the same ground truth,
records agree across hospitals (a drug is always prescribed *for* a condition the person really
has, an outcome always follows a prescription, dates are ordered, ...).

Everything is invented; nothing is read from or resembles a real dataset beyond textbook-level
plausibility (e.g. T2 diabetes is treated with metformin).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from fedhealth.common import vocab
from fedhealth.common.vocab import AS_OF, CONDITIONS, Condition, Drug

FIRST_F = ("Mary Linda Susan Karen Nancy Lisa Betty Helen Sandra Donna Carol Ruth Sharon Michelle Laura Sarah Kimberly "
           "Deborah Jessica Shirley Cynthia Angela Melissa Brenda Amy Anna Rebecca Virginia Kathleen Pamela Martha "
           "Debra Amanda Stephanie Carolyn Christine Marie Janet Catherine Frances Ann").split()
FIRST_M = ("James Robert John Michael David William Richard Joseph Thomas Charles Christopher Daniel Matthew Anthony "
           "Mark Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Timothy Ronald Edward Jason Jeffrey "
           "Ryan Jacob Gary Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon").split()
LAST = ("Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Gonzalez Wilson "
        "Anderson Thomas Taylor Moore Jackson Martin Lee Perez Thompson White Harris Sanchez Clark Ramirez Lewis "
        "Robinson Walker Young Allen King Wright Scott Torres Nguyen Hill Flores Green Adams Nelson Baker Hall "
        "Rivera Campbell Mitchell Carter Roberts Gomez Phillips Evans Turner Diaz Parker Cruz Edwards Collins "
        "Reyes Stewart Morris Morales Murphy Cook Rogers Gutierrez Ortiz Morgan Cooper Peterson Bailey Reed Kelly").split()

NOTE_TEMPLATES = (
    "Counselled on lifestyle changes.", "Patient reports good tolerance.", "Referred for specialist review.",
    "Dose adjusted at last visit.", "Missed last follow-up appointment.", "Family history noted.",
    "Patient prefers morning dosing.", "Home monitoring advised.",
)


@dataclass
class Rx:
    drug: Drug
    start: date
    end: date | None
    adherence: float
    prescriber: str
    note: str | None


@dataclass
class Episode:
    """One person x one condition: diagnosis -> (treatment -> follow-up outcome)."""
    condition: Condition
    severity: str
    diagnosed_on: date
    status: str
    note: str | None
    prescriptions: list[Rx] = field(default_factory=list)     # sorted by start date (earliest = "primary")
    outcome: str | None = None
    baseline_value: float | None = None
    followup_value: float | None = None
    delta: float | None = None
    assessed_on: date | None = None
    baseline_lab_date: date | None = None
    imaging: tuple[str, str, date] | None = None              # (modality, finding, date)

    @property
    def treated(self) -> bool:
        return bool(self.prescriptions)


@dataclass
class Person:
    person_id: int
    national_id: str
    full_name: str
    sex: str
    birth_year: int
    dob: date
    zip_code: str
    membership: str                       # which hospitals hold this person: subset of "ABC"
    episodes: list[Episode] = field(default_factory=list)

    @property
    def age(self) -> int:
        return vocab.age_of(self.birth_year)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _make_zips(rng: np.random.Generator, n_prefixes: int = 20, per_prefix: int = 5) -> tuple[list[str], np.ndarray]:
    """~100 synthetic ZIP codes sharing 3-digit prefixes, with a Zipf-like popularity (a few big zips,
    many small ones). The skew matters for the re-identification attack: rare zips => unique people."""
    prefixes = rng.choice(np.arange(100, 999), size=n_prefixes, replace=False)
    zips = [f"{int(p)}{s:02d}" for p in prefixes for s in rng.choice(100, size=per_prefix, replace=False)]
    w = 1.0 / np.arange(1, len(zips) + 1) ** 0.8
    rng.shuffle(w)
    return zips, w / w.sum()


def _assign_membership(rng: np.random.Generator, n: int) -> list[str]:
    """Exact, seed-shuffled assignment of people to hospital subsets (see vocab.MEMBERSHIP_FRACTIONS)."""
    labels: list[str] = []
    items = list(vocab.MEMBERSHIP_FRACTIONS.items())
    for key, frac in items[:-1]:
        labels += [key] * int(round(frac * n))
    labels += [items[-1][0]] * (n - len(labels))
    perm = rng.permutation(len(labels))
    return [labels[i] for i in perm]


def build_universe(n_patients: int = 5000, seed: int = 42) -> list[Person]:
    rng = np.random.default_rng(seed)
    n = n_patients

    ages = np.clip(rng.normal(55, 17, n), 22, 92).round().astype(int)
    sexes = np.where(rng.random(n) < 0.52, "F", "M")
    zips, zip_w = _make_zips(rng)
    zip_idx = rng.choice(len(zips), size=n, p=zip_w)
    national = rng.choice(900_000_000, size=n, replace=False) + 100_000_000     # unique 9-digit numbers
    membership = _assign_membership(rng, n)
    frailty = rng.normal(0, 0.6, n)

    people: list[Person] = []
    for i in range(n):
        age, sex = int(ages[i]), str(sexes[i])
        first = rng.choice(FIRST_F if sex == "F" else FIRST_M)
        birth_year = AS_OF.year - age
        dob = date(birth_year, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        p = Person(
            person_id=i, national_id=f"SYN-{int(national[i])}", full_name=f"{first} {rng.choice(LAST)}",
            sex=sex, birth_year=birth_year, dob=dob, zip_code=zips[int(zip_idx[i])], membership=membership[i],
        )
        _assign_conditions(rng, p, float(frailty[i]))
        people.append(p)
    return people


def _assign_conditions(rng: np.random.Generator, p: Person, frailty: float) -> None:
    age, male = p.age, 1.0 if p.sex == "M" else 0.0
    probs = {code: _sigmoid(c.intercept + c.age_slope * (age - 55) / 10 + c.male_effect * male + frailty)
             for code, c in CONDITIONS.items()}
    chosen = [code for code, pr in probs.items() if rng.random() < pr]
    if not chosen:                                   # every person in the system has >= 1 chronic condition
        codes = list(probs)
        w = np.array([probs[c] for c in codes])
        chosen = [str(rng.choice(codes, p=w / w.sum()))]
    n_comorb = len(chosen) - 1

    for k, code in enumerate(chosen):
        cond = CONDITIONS[code]
        sev_score = rng.normal(0, 1) + 0.3 * (age - 55) / 10 + 0.5 * frailty
        severity = "severe" if sev_score > 0.9 else "moderate" if sev_score > -0.3 else "mild"

        # Diagnosis date: after adulthood, early enough that the whole treatment story fits before AS_OF.
        earliest = max(date(2016, 1, 1), date(p.dob.year + 18, p.dob.month, min(p.dob.day, 28)))
        latest = AS_OF - timedelta(days=420)
        dx = earliest + timedelta(days=int(rng.integers(0, (latest - earliest).days + 1)))
        ep = Episode(cond, severity, dx, "active" if rng.random() < 0.9 else "resolved",
                     str(rng.choice(NOTE_TEMPLATES)) if rng.random() < 0.3 else None)

        treated = (k == 0) or rng.random() < 0.9      # first condition always treated -> everyone has an outcome
        if treated:
            _treat(rng, p, ep, n_comorb)
        _add_lab_and_imaging(rng, ep, severity)
        p.episodes.append(ep)


def _treat(rng: np.random.Generator, p: Person, ep: Episode, n_comorb: int) -> None:
    cond = ep.condition
    drugs = list(cond.drugs)
    w = np.array([d.weight for d in drugs])
    first = int(rng.choice(len(drugs), p=w / w.sum()))
    picked = [drugs[first]]
    if rng.random() < 0.25 and len(drugs) > 1:                 # combination / switch
        rest = [d for j, d in enumerate(drugs) if j != first]
        rw = np.array([d.weight for d in rest])
        picked.append(rest[int(rng.choice(len(rest), p=rw / rw.sum()))])

    base_adh = float(rng.beta(5, 2) * 100)                     # person-condition adherence, mean ~71%
    rxs: list[Rx] = []
    for d in picked:
        start = ep.diagnosed_on + timedelta(days=int(rng.integers(1, 91 if d is picked[0] else 121)))
        if rng.random() < 0.7:
            end = None                                         # ongoing
        else:
            end = min(start + timedelta(days=int(rng.integers(180, 901))), AS_OF)
        adh = round(float(np.clip(base_adh + rng.normal(0, 5), 10, 100)), 1)
        rxs.append(Rx(d, start, end, adh, f"DR-{int(rng.integers(1, 300)):04d}",
                      str(rng.choice(NOTE_TEMPLATES)) if rng.random() < 0.3 else None))
    rxs.sort(key=lambda r: r.start)
    ep.prescriptions = rxs

    # --- response model (coefficients are exported to generator_params.json) ---------------------------
    m = vocab.RESPONSE_MODEL
    mean_adh = float(np.mean([r.adherence for r in rxs]))
    logit = (cond.base_improve_logit + sum(d.effect for d in picked)
             + m["adherence_per_pp"] * (mean_adh - 70)
             + m["age_per_decade"] * (p.age - 60) / 10
             + m["per_extra_comorbidity"] * n_comorb
             + m["severity"][ep.severity])
    if rng.random() < _sigmoid(logit):
        outcome = "improved"
    else:
        outcome = "worsened" if rng.random() < m["worsened_given_not_improved"] else "no_change"
    ep.outcome = outcome

    # --- lab values consistent with the categorical outcome --------------------------------------------
    lab = cond.lab
    baseline = round(max(float(rng.normal(lab.baseline_mean, lab.baseline_sd)), lab.floor), 2)
    thr = lab.threshold
    if outcome == "improved":
        mag = thr + float(rng.exponential(thr * 1.2))
        direction = lab.better
    elif outcome == "worsened":
        mag = thr + float(rng.exponential(thr * 0.8))
        direction = -lab.better
    else:
        mag = float(rng.uniform(0, 0.9 * thr))
        direction = 1 if rng.random() < 0.5 else -1
    mag = min(mag, lab.clamp)
    if direction < 0:
        # A decrease can't exceed 40% of the baseline (keeps values physical). 0.4*floor >= threshold
        # for every lab in vocab.py, so improved/worsened still clear their clinical threshold.
        mag = min(mag, 0.4 * baseline)
        if outcome != "no_change":
            mag = max(mag, thr)
    followup = round(baseline + direction * mag, 2)
    ep.baseline_value, ep.followup_value = baseline, followup
    ep.delta = round(followup - baseline, 2)
    # Follow-up happens >= 90 days after the first prescription and >= 30 days after the last one started.
    assessed = max(rxs[0].start + timedelta(days=int(rng.integers(90, 366))), rxs[-1].start + timedelta(days=30))
    ep.assessed_on = min(assessed, AS_OF)


def _add_lab_and_imaging(rng: np.random.Generator, ep: Episode, severity: str) -> None:
    """Dates for the baseline lab and (sometimes) an imaging study."""
    if ep.treated:
        gap = (ep.prescriptions[0].start - ep.diagnosed_on).days            # >= 1
        ep.baseline_lab_date = ep.diagnosed_on + timedelta(days=int(rng.integers(0, gap + 1)))
    else:
        ep.baseline_lab_date = min(ep.diagnosed_on + timedelta(days=int(rng.integers(0, 31))), AS_OF)
        lab = ep.condition.lab
        ep.baseline_value = round(max(float(rng.normal(lab.baseline_mean, lab.baseline_sd)), lab.floor), 2)
    if ep.condition.imaging and rng.random() < 0.6:
        modality, findings = ep.condition.imaging
        # findings are listed roughly from benign to serious; severity nudges the draw towards the end
        u = rng.random() * 0.7 + {"mild": 0.0, "moderate": 0.15, "severe": 0.3}[severity]
        idx = min(int(u * len(findings)), len(findings) - 1)
        when = min(ep.diagnosed_on + timedelta(days=int(rng.integers(0, 61))), AS_OF)
        ep.imaging = (modality, findings[idx], when)
