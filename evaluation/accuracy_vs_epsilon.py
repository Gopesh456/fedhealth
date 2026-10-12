"""EVALUATION 1 - Accuracy vs privacy: how wrong are the released answers as epsilon varies?

METHOD
    * Three COUNT queries of very different cohort sizes (small ~20, medium ~150, large ~800) plus an AVG and a
      HISTOGRAM. For each, the exact aggregate comes from the REAL private pipeline (PSI + MPC) and is checked
      against the independent ground truth.
    * For every epsilon in {0.1, 0.5, 1, 2, 5} and each mechanism (Laplace; Gaussian with delta = 1e-6) the real
      `release()` stage is applied 2,000 times with fresh randomness (repeating a query re-draws only the noise).
    * We report mean absolute error (MAE), median, 95th percentile, and error relative to the true value.

EXPECTATION
    The noise does not depend on the true count, so the ABSOLUTE error is the same for every cohort and falls as epsilon
    grows (MAE ~ 1/epsilon for small epsilon; faster for large epsilon, where integer noise is mostly exactly zero),
    while the RELATIVE error is large for small cohorts and small for large ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attacks"))
from common import ExactOracle, common_args, oracle_from_args, save_json, seeded_randbelow  # noqa: E402

from fedhealth.common import vocab  # noqa: E402
from fedhealth.common.schemas import QuerySpec  # noqa: E402
from fedhealth.coordinator.release import PrivacyParams, ReleasePolicy, release  # noqa: E402
from fedhealth.privacy.dp.mechanisms import LaplaceMechanism  # noqa: E402

EPSILONS = (0.1, 0.5, 1.0, 2.0, 5.0)
REPS = 2000


def count_spec(cond: str, drug: str | None, outcome: str | None, wide: bool = False) -> QuerySpec:
    return QuerySpec.model_validate({"agg": "count", "condition": cond, "a": {},
                                     "b": {"outcome": outcome} if outcome else {}, "c": {"drug": drug} if drug else {}})


def pick_queries(oracle: ExactOracle) -> dict[str, QuerySpec]:
    """Choose one real query per target cohort size from the data (closest true count to 20 / 150 / 800)."""
    targets = {"small": 20, "medium": 150, "large": 800}
    best: dict[str, tuple[int, QuerySpec]] = {}
    candidates: list[QuerySpec] = []
    for c in vocab.CONDITION_CODES:
        candidates.append(count_spec(c, None, None))
        for d in vocab.drug_names(c):
            for o in (None, *vocab.OUTCOMES):
                candidates.append(count_spec(c, d, o))
    for spec in candidates:
        n = oracle.gt.evaluate(spec)                         # cheap scan; the chosen ones are re-run through the pipeline
        for label, goal in targets.items():
            if n >= 10 and (label not in best or abs(n - goal) < abs(best[label][0] - goal)):
                best[label] = (n, spec)
    return {label: spec for label, (n, spec) in best.items()}


def stats(errors: np.ndarray, true_value: float) -> dict:
    a = np.abs(errors)
    return {"mae": float(a.mean()), "median_abs_error": float(np.median(a)), "p95_abs_error": float(np.percentile(a, 95)),
            "relative_mae": float(a.mean() / abs(true_value)) if true_value else None, "bias": float(errors.mean())}


def run(oracle: ExactOracle, reps: int = REPS, seed: int = 21) -> dict:
    rb = seeded_randbelow(seed)
    policy = ReleasePolicy()
    rows, queries = [], {}
    for label, spec in pick_queries(oracle).items():
        ex = oracle.exact(spec)                              # real pipeline, asserted equal to ground truth
        queries[label] = {"true_count": ex.count, "query": spec.model_dump(mode="json", exclude_none=True)}
        for mech in ("laplace", "gaussian"):
            for eps in EPSILONS:
                pp = PrivacyParams(epsilon=eps, mechanism=mech)
                vals = np.array([release(spec, ex, pp, policy, rb).value for _ in range(reps)], dtype=float)
                rows.append({"kind": "count", "cohort": label, "true_value": ex.count, "mechanism": mech, "epsilon": eps,
                             "theory_mae": LaplaceMechanism(eps, 1).expected_abs_error if mech == "laplace" else None,
                             **stats(vals - ex.count, ex.count)})

    # AVG of adherence over a wide cohort (epsilon split between sum and count) - error in natural units (percentage points)
    avg_spec = QuerySpec.model_validate({"agg": "avg", "condition": "I10", "a": {}, "b": {}, "c": {}, "value_field": "adherence_pct"})
    ex = oracle.exact(avg_spec)
    truth = ex.total / ex.count
    for eps in EPSILONS:
        vals = np.array([release(avg_spec, ex, PrivacyParams(epsilon=eps), policy, rb).value for _ in range(reps)], dtype=float)
        rows.append({"kind": "avg", "cohort": f"avg adherence, n={ex.count}", "true_value": truth, "mechanism": "laplace", "epsilon": eps,
                     "theory_mae": None, **stats(vals - truth, truth)})

    # HISTOGRAM: per-bin error equals a single count's error (all bins together cost epsilon ONCE)
    hist_spec = QuerySpec.model_validate({"agg": "histogram", "condition": "I10", "a": {}, "b": {}, "c": {}, "group_by": "outcome"})
    ex = oracle.exact(hist_spec)
    for eps in EPSILONS:
        errs = []
        for _ in range(reps):
            out = release(hist_spec, ex, PrivacyParams(epsilon=eps), policy, rb).value
            errs += [out[k] - v for k, v in ex.cells.items()]
        rows.append({"kind": "histogram", "cohort": f"3 bins, n={sum(ex.cells.values())}", "true_value": float(np.mean(list(ex.cells.values()))),
                     "mechanism": "laplace", "epsilon": eps, "theory_mae": LaplaceMechanism(eps, 1).expected_abs_error,
                     **stats(np.array(errs, dtype=float), float(np.mean(list(ex.cells.values()))))})
    return {"evaluation": "accuracy vs epsilon", "repetitions": reps, "queries": queries, "rows": rows,
            "pipeline_queries_checked": oracle.pipeline_runs}


def main() -> None:
    ap = common_args("Accuracy vs epsilon.")
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    out = run(oracle_from_args(args), args.reps)
    path = save_json("accuracy_vs_epsilon", out)
    print(f"queries: " + ", ".join(f"{k}: true count {v['true_count']}" for k, v in out["queries"].items()))
    print(f"  {'cohort':<26}{'mech':<10}" + "".join(f"{'eps=' + format(e, 'g'):>12}" for e in EPSILONS) + "   (mean absolute error)")
    seen = set()
    for r in out["rows"]:
        key = (r["cohort"], r["mechanism"])
        if key in seen:
            continue
        seen.add(key)
        line = "".join(f"{x['mae']:>12.2f}" for x in out["rows"] if (x["cohort"], x["mechanism"]) == key)
        print(f"  {r['cohort']:<26}{r['mechanism']:<10}{line}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
