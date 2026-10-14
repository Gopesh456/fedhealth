"""The evaluation world at a glance: where the patients come from and what the hospitals hold.

Reads the world's ground truth (people, episodes, generator parameters) and writes results/data.json, which
make_report.py turns into the "Data" section of RESULTS.md. Nothing here touches the hospital services.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attacks"))
from common import common_args, ensure_world, save_json  # noqa: E402

from fedhealth.common import vocab  # noqa: E402


def run(world) -> dict:
    gt = world / "ground_truth"
    params = json.loads((gt / "generator_params.json").read_text(encoding="utf-8"))
    per: dict[str, Counter] = defaultdict(Counter)
    drugs: dict[str, Counter] = defaultdict(Counter)
    with open(gt / "episodes.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            c = per[r["condition"]]
            c["episodes"] += 1
            c["treated"] += bool(r["drugs"])
            c["with_outcome"] += bool(r["outcome"])
            c["improved"] += r["outcome"] == "improved"
            for d in filter(None, r["drugs"].split("|")):
                drugs[r["condition"]][d] += 1
    conditions = {code: {"name": vocab.CONDITIONS[code].name, **per[code],
                         "top_drugs": [d for d, _ in drugs[code].most_common(3)]}
                  for code in vocab.CONDITION_CODES if code in per}
    with open(gt / "universe.csv", newline="", encoding="utf-8") as fh:
        people = list(csv.DictReader(fh))
    return {"source": params["source"], "people": len(people), "hospital_sizes": params["hospital_sizes"],
            "membership_counts": params["membership_counts"], "as_of": params["as_of"],
            "sex": dict(Counter(p["sex"] for p in people)),
            "median_birth_year": sorted(int(p["birth_year"]) for p in people)[len(people) // 2],
            "conditions": conditions}


def main() -> None:
    ap = common_args("Summary of the evaluation world.")
    args = ap.parse_args()
    out = run(ensure_world(1000 if args.fast else args.patients, args.seed, args.source))
    path = save_json("data", out)
    print(f"{out['source']}: {out['people']:,} people; hospitals {out['hospital_sizes']}")
    for code, c in out["conditions"].items():
        print(f"  {code} {c['name'][:28]:<28} episodes {c['episodes']:>5}  treated {c['treated']:>5}  with outcome {c['with_outcome']:>5}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
