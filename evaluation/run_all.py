"""Run every attack and evaluation, then render the charts and RESULTS.md.

    python evaluation/run_all.py            # full size (5,000 Synthea patients, real 2048-bit PSI group)
    python evaluation/run_all.py --fast     # 1,000 patients, INSECURE 768-bit group: smoke test only
    python evaluation/run_all.py --source synthetic     # the parametric generator instead of Synthea

Results land in evaluation/results/*.json, charts in evaluation/charts/*.png, the report in evaluation/RESULTS.md.
Worlds are cached in evaluation/.world (gitignored); delete it if you regenerate your secrets.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("data: the evaluation world", ["evaluation/data_summary.py"]),
    ("attack: differencing", ["attacks/differencing.py"]),
    ("attack: re-identification", ["attacks/reidentification.py"]),
    ("attack: budget exhaustion", ["attacks/budget_exhaustion.py"]),
    ("attack: HMAC dictionary", ["attacks/hmac_dictionary.py"]),
    ("evaluation: accuracy vs epsilon", ["evaluation/accuracy_vs_epsilon.py"]),
    ("evaluation: federated learning", ["evaluation/federated.py"]),
]


def main() -> None:
    fast = "--fast" in sys.argv
    extra = ["--fast"] if fast else []
    if "--source" in sys.argv:
        extra += ["--source", sys.argv[sys.argv.index("--source") + 1]]
    t0 = time.time()
    steps = list(STEPS)
    steps.append(("evaluation: performance", ["evaluation/performance.py"] + (["--sizes", "500", "1000", "--reps", "5"] if fast else [])))
    for name, cmd in steps:
        print(f"\n=== {name} ===", flush=True)
        args = [sys.executable, str(ROOT / cmd[0]), *(extra if "performance" not in cmd[0] else extra[1:] if fast else extra), *cmd[1:]]
        subprocess.run(args, cwd=str(ROOT), check=True)
    print("\n=== charts + report ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "make_charts.py")], cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "make_report.py")], cwd=str(ROOT), check=True)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. See evaluation/RESULTS.md")


if __name__ == "__main__":
    main()
