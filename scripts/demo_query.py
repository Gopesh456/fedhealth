"""Run the headline query against a RUNNING system and compare the paths.

    python scripts/run_local.py             # terminal 1  (dev mode: insecure baselines enabled)
    python scripts/demo_query.py            # terminal 2

"How many patients aged 40-65 with type-2 diabetes (E11), prescribed metformin, showed a positive response?"
Prints the plain (INSECURE) baseline, the PSI-linked path, the exact secret-shared (MPC) result and the independent
ground truth - they must all agree. For the PRIVATE, noisy, authenticated answer use the strict system
(`python scripts/run_local.py --secure`) with the dashboard or the curl example in the README.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedhealth.common.config import data_dir, load_topology  # noqa: E402
from fedhealth.common.schemas import QuerySpec  # noqa: E402
from fedhealth.synth.ground_truth import GroundTruth  # noqa: E402

SPEC = QuerySpec.model_validate({
    "agg": "count", "condition": "E11",
    "a": {"age_min": 40, "age_max": 65}, "b": {"outcome": "improved"}, "c": {"drug": "metformin"},
})


def main() -> None:
    topo = load_topology()
    base = f"{topo.scheme}://{topo.coordinator_host}:{topo.coordinator_port}"
    body = SPEC.model_dump(mode="json", exclude_none=True)
    print("Query:", body)
    answers = {}
    for label, path in (("plain, INSECURE baseline (raw IDs cross the wire)", "/insecure/query"),
                        ("PSI linkage + insecure aggregation", "/insecure/psi-query"),
                        ("PSI + secret-shared MPC (exact, no DP)", "/insecure/mpc-query")):
        r = httpx.post(f"{base}{path}", json=body, timeout=600)
        if r.status_code == 404:
            sys.exit("Insecure routes are not mounted (strict mode). Start the dev system: python scripts/run_local.py")
        r.raise_for_status()
        out = r.json()
        answers[label] = out["result"]
        t = out.get("timings", {})
        print(f"[{label}]\n    count = {out['result']}   ({t.get('total_s', 0):.2f} s" + (f", linkage {t['linkage_s']:.2f} s" if "linkage_s" in t else "") + ")")

    gt_dir = data_dir() / "ground_truth"
    if gt_dir.exists():
        truth = GroundTruth(gt_dir).evaluate(SPEC)
        print(f"[ground truth from the generator's universe]\n    count = {truth}")
        print("ALL PATHS AGREE" if all(v == truth for v in answers.values()) else "MISMATCH!")


if __name__ == "__main__":
    main()
