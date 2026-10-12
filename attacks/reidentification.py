"""ATTACK 2 - Re-identification (linkage) of a "de-identified" release.

THE ATTACK
    A hospital decides to share its patient table "anonymously" by deleting the direct identifiers (name,
    national ID). What remains - birth year, sex, ZIP and the diagnoses - looks harmless. But the same three
    quasi-identifiers (QIs) exist in other datasets people can get: a voter roll, a loyalty-card list, a
    social-media scrape. Joining the "anonymous" table to such an AUXILIARY dataset on (birth year, sex, ZIP)
    attaches names to diagnoses (Sweeney, 2000: 87% of Americans are unique on ZIP + birth date + sex).

    Here the released table is Hospital A's patient list; the auxiliary dataset is a "voter roll" that covers a random
    70% of everyone in the synthetic world (names + the same three QIs). A record is RE-IDENTIFIED when exactly one
    person in the auxiliary data shares its QIs; the claim is CORRECT when that person really is the record's owner.

WHAT WE MEASURE
    * the share of records that are unique on each QI combination (the k-anonymity of the release),
    * the re-identification rate for every QI subset,
    * how much generalising the QIs (10-year age bands, 3-digit ZIP) and enforcing k-anonymity reduces it - and
      what it costs in released records,
    * and WHY THIS SYSTEM IS DIFFERENT: it never exports rows. The only way to reach the same information through
      the query interface is differencing (attack 1), which differential privacy stops.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from common import common_args, ensure_world, save_json, world_dir
from fedhealth.common import vocab

QI_ALL = ("birth_year", "sex", "zip")


def load(gt_dir):
    persons = {}
    with open(gt_dir / "universe.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            persons[int(r["person_id"])] = {"name": r["full_name"], "birth_year": int(r["birth_year"]), "sex": r["sex"],
                                            "zip": r["zip_code"], "membership": r["membership"]}
    diagnoses = defaultdict(set)
    with open(gt_dir / "episodes.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            diagnoses[int(r["person_id"])].add(r["condition"])
    return persons, diagnoses


def qi_key(p: dict, qis: tuple[str, ...], generalise: bool) -> tuple:
    out = []
    for q in qis:
        v = p[q]
        if generalise and q == "birth_year":
            v = vocab.age_band_of(vocab.age_of(v))               # 10/15-year age band instead of the exact year
        elif generalise and q == "zip":
            v = v[:3]                                            # 3-digit ZIP prefix
        out.append(v)
    return tuple(out)


def attack(released_ids, persons, aux_ids, qis, generalise, k_anon: int = 1) -> dict:
    """Re-identify the released records against the auxiliary dataset."""
    released_groups = Counter(qi_key(persons[i], qis, generalise) for i in released_ids)
    kept = [i for i in released_ids if released_groups[qi_key(persons[i], qis, generalise)] >= k_anon]   # k-anonymity: drop small groups
    aux_groups: dict[tuple, list[int]] = defaultdict(list)
    for i in aux_ids:
        aux_groups[qi_key(persons[i], qis, generalise)].append(i)
    claimed = correct = 0
    for i in kept:
        cands = aux_groups.get(qi_key(persons[i], qis, generalise), [])
        if len(cands) == 1:                                       # exactly one match: the attacker names the record
            claimed += 1
            correct += cands[0] == i
    n = len(released_ids)
    kept_groups = Counter(qi_key(persons[i], qis, generalise) for i in kept)       # the table that is actually RELEASED
    unique = sum(1 for c in kept_groups.values() if c == 1)
    return {"released_records": len(kept), "retained_fraction": len(kept) / n, "claimed": claimed, "correct": correct,
            "reid_rate": correct / n, "precision": (correct / claimed) if claimed else None,
            "unique_in_release": (unique / len(kept)) if kept else 0.0,
            "min_group_size": min(kept_groups.values()) if kept_groups else 0}


def run(gt_dir, coverage: float = 0.7, seed: int = 5) -> dict:
    persons, diagnoses = load(gt_dir)
    released = [i for i, p in persons.items() if "A" in p["membership"]]          # Hospital A's "anonymised" table
    rng = np.random.default_rng(seed)
    aux = [i for i in persons if rng.random() < coverage]                         # the attacker's voter roll
    out = {"attack": "re-identification", "released_records": len(released), "aux_records": len(aux),
           "aux_coverage": coverage, "by_qi": [], "defences": []}
    for r in range(1, 4):
        for qis in combinations(QI_ALL, r):
            out["by_qi"].append({"qi": "+".join(qis), **attack(released, persons, aux, qis, generalise=False)})
    full = QI_ALL
    out["defences"] = [
        {"label": "exact birth year + sex + ZIP5 (names removed only)", **attack(released, persons, aux, full, False)},
        {"label": "age band + sex + ZIP3 (generalised)", **attack(released, persons, aux, full, True)},
        {"label": "generalised + 5-anonymity", **attack(released, persons, aux, full, True, k_anon=5)},
        {"label": "generalised + 10-anonymity", **attack(released, persons, aux, full, True, k_anon=10)},
        {"label": "exact + 10-anonymity", **attack(released, persons, aux, full, False, k_anon=10)},
    ]
    return out


def main() -> None:
    ap = common_args("Re-identification of a naively de-identified release.")
    ap.add_argument("--coverage", type=float, default=0.7, help="fraction of the world present in the auxiliary data")
    args = ap.parse_args()
    n = 1000 if args.fast else args.patients
    ensure_world(n, args.seed)
    out = run(world_dir(n, args.seed) / "ground_truth", args.coverage)
    path = save_json("attack_reidentification", out)
    print(f"Released table: {out['released_records']} Hospital-A patients, names removed.  Auxiliary data: {out['aux_records']} people.")
    print(f"  {'quasi-identifiers':<26}{'unique in release':>18}{'re-identified (correct)':>26}")
    for r in out["by_qi"]:
        print(f"  {r['qi']:<26}{r['unique_in_release']:>17.1%}{r['reid_rate']:>20.1%}   (precision {r['precision'] or 0:.0%})")
    print("  defences:")
    for r in out["defences"]:
        print(f"    {r['label']:<52} re-id {r['reid_rate']:>6.1%}   records kept {r['retained_fraction']:>6.1%}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
