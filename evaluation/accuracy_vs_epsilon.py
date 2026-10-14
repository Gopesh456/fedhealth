"""EVALUATION 1 - Accuracy vs privacy: how wrong are the released answers as epsilon varies, and what does it cost
to stop trusting the coordinator?

METHOD
    * Three COUNT queries of very different cohort sizes (small ~20, medium ~150, large ~800) plus an AVG and a
      HISTOGRAM. For each, the exact aggregate comes from the REAL private pipeline (PSI + MPC, evaluation-only exact
      route) and is checked against the independent ground truth.
    * The noise is the hospitals' distributed noise: c = 3 shares (two computing hospitals + the triple dealer), each
      sized so that any h = 2 of them already give epsilon-DP (collusion tolerance t = 1). For every epsilon and
      mechanism (Laplace; Gaussian with delta = 1e-6) the TOTAL noise is drawn 20,000 times with the vectorised
      simulation (the same law as the hospitals' exact samplers; tests/test_distributed_dp.py checks this).
    * Reported: mean absolute error of the noisy count (MAE), median, 95th percentile, relative error, and how often
      the k-threshold - applied to the NOISY count - suppresses the answer.
    * TOLERANCE: the same medium query with t = 0 (all three shares needed - this is exactly the old trusted-
      coordinator noise), t = 1 (default) and t = 2.
    * ACCOUNTING: how many identical queries a researcher's budget (epsilon 10, delta 1e-6) admits under basic
      composition, under zCDP, and under the ledger (the better of the two; Gaussian queries are zCDP-only).

EXPECTATION
    The noise does not depend on the true count, so the ABSOLUTE error is the same for every cohort and falls as epsilon
    grows; the RELATIVE error is large for small cohorts. Tolerating t colluders multiplies the noise variance by
    c / (c - t). zCDP admits far more small-epsilon queries than basic composition, basic wins for a few large ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attacks"))
from common import ExactOracle, common_args, oracle_from_args, save_json  # noqa: E402

from fedhealth.common import vocab  # noqa: E402
from fedhealth.common.schemas import QuerySpec  # noqa: E402
from fedhealth.coordinator.release import PrivacyParams, ReleasePolicy, plan_noise, simulate_releases  # noqa: E402
from fedhealth.privacy.dp.accounting import queries_within_budget  # noqa: E402

EPSILONS = (0.1, 0.25, 0.5, 1.0, 2.0)          # 2 = the hospitals' default per-query cap (configs/policy.yaml)
TOLERANCES = (0, 1, 2)
REPS = 20_000
K = 10
BUDGET = (10.0, 1e-6)                            # the researcher role's budget
ACCOUNTING_EPS = tuple(float(x) for x in np.round(np.geomspace(0.02, 2.0, 21), 4))


def count_spec(cond: str, drug: str | None, outcome: str | None) -> QuerySpec:
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
            if n >= K and (label not in best or abs(n - goal) < abs(best[label][0] - goal)):
                best[label] = (n, spec)
    return {label: spec for label, (n, spec) in best.items()}


def stats(errors: np.ndarray, true_value: float) -> dict:
    a = np.abs(errors)
    return {"mae": float(a.mean()), "median_abs_error": float(np.median(a)), "p95_abs_error": float(np.percentile(a, 95)),
            "relative_mae": float(a.mean() / abs(true_value)) if true_value else None, "bias": float(errors.mean())}


def count_row(spec: QuerySpec, true_count: int, pp: PrivacyParams, policy: ReleasePolicy, reps: int,
              rng: np.random.Generator) -> dict:
    plan = plan_noise(spec, pp, policy)
    noise = plan.simulate_total(reps, rng)[:, 0]
    noisy = np.maximum(true_count + noise, 0)
    return {"mechanism": pp.mechanism, "epsilon": pp.epsilon, "tolerance": policy.tolerance,
            "noise_contributors": plan.contributors, "honest_shares": plan.honest, "rho": plan.rho,
            "theory_mae": plan.expected_abs_error(0), "noise_std": plan.std(0),
            "suppressed_rate": float(np.mean(noisy < K)), **stats(noise.astype(float), true_count)}


def run(oracle: ExactOracle, reps: int = REPS, seed: int = 21) -> dict:
    rng = np.random.default_rng(seed)
    policy = ReleasePolicy(k=K)
    rows, queries = [], {}
    picked = pick_queries(oracle)
    for label, spec in picked.items():
        ex = oracle.exact(spec)                              # real pipeline, asserted equal to ground truth
        queries[label] = {"true_count": ex.count, "query": spec.model_dump(mode="json", exclude_none=True)}
        for mech in ("laplace", "gaussian"):
            for eps in EPSILONS:
                rows.append({"kind": "count", "cohort": label, "true_value": ex.count,
                             **count_row(spec, ex.count, PrivacyParams(epsilon=eps, mechanism=mech), policy, reps, rng)})

    # the price of not trusting the coordinator: tolerance t = 0 is the old central noise (the coordinator added it)
    medium = picked["medium"]
    m_true = oracle.exact(medium).count
    for mech in ("laplace", "gaussian"):
        for t in TOLERANCES:
            for eps in EPSILONS:
                rows.append({"kind": "tolerance", "cohort": "medium", "true_value": m_true,
                             **count_row(medium, m_true, PrivacyParams(epsilon=eps, mechanism=mech),
                                         ReleasePolicy(k=K, tolerance=t), reps, rng)})

    # AVG of adherence over a wide cohort (epsilon split between the noisy count and the noisy total) - natural units
    avg_spec = QuerySpec.model_validate({"agg": "avg", "condition": "I10", "a": {}, "b": {}, "c": {}, "value_field": "adherence_pct"})
    ex = oracle.exact(avg_spec)
    truth = ex.total / ex.count
    for eps in EPSILONS:
        out = simulate_releases(avg_spec, ex, PrivacyParams(epsilon=eps), policy, reps, rng)
        vals = np.array([r.value for r in out if r.value is not None], dtype=float)
        rows.append({"kind": "avg", "cohort": f"avg adherence, n={ex.count}", "true_value": truth, "mechanism": "laplace",
                     "epsilon": eps, "tolerance": 1, "theory_mae": None, "suppressed_rate": 1 - len(vals) / reps,
                     **stats(vals - truth, truth)})

    # HISTOGRAM: per-bin error equals a single count's error (all bins together cost epsilon ONCE)
    hist_spec = QuerySpec.model_validate({"agg": "histogram", "condition": "I10", "a": {}, "b": {}, "c": {}, "group_by": "outcome"})
    ex = oracle.exact(hist_spec)
    bins = np.array(list(ex.cells.values()))
    for eps in EPSILONS:
        plan = plan_noise(hist_spec, PrivacyParams(epsilon=eps), policy)
        noise = plan.simulate_total(reps, rng)
        rows.append({"kind": "histogram", "cohort": f"{len(bins)} bins, n={int(bins.sum())}", "true_value": float(bins.mean()),
                     "mechanism": "laplace", "epsilon": eps, "tolerance": 1, "theory_mae": plan.expected_abs_error(0),
                     "suppressed_rate": float(np.mean(np.maximum(bins + noise, 0) < K)),
                     **stats(noise.ravel().astype(float), float(bins.mean()))})

    # how many identical queries one budget admits, per accountant
    accounting = []
    for eps in ACCOUNTING_EPS:
        lap = plan_noise(medium, PrivacyParams(epsilon=eps), policy)
        gau = plan_noise(medium, PrivacyParams(epsilon=eps, mechanism="gaussian"), policy)
        lq = queries_within_budget(lap.rho, eps, 0.0, *BUDGET)
        gq = queries_within_budget(gau.rho, eps, gau.delta, *BUDGET)
        accounting.append({"epsilon": eps, "laplace_rho": lap.rho, "gaussian_rho": gau.rho,
                           "laplace_basic": lq["basic"], "laplace_zcdp": lq["zcdp"], "laplace_ledger": lq["ledger"],
                           "gaussian_basic": gq["basic"], "gaussian_zcdp": gq["zcdp"], "gaussian_ledger": gq["ledger"]})
    return {"evaluation": "accuracy vs epsilon", "repetitions": reps, "k": K, "queries": queries, "rows": rows,
            "budget": {"epsilon": BUDGET[0], "delta": BUDGET[1]}, "accounting": accounting,
            "pipeline_queries_checked": oracle.pipeline_runs}


def main() -> None:
    ap = common_args("Accuracy vs epsilon.")
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    out = run(oracle_from_args(args), args.reps)
    path = save_json("accuracy_vs_epsilon", out)
    print("queries: " + ", ".join(f"{k}: true count {v['true_count']}" for k, v in out["queries"].items()))
    print(f"  {'cohort':<26}{'mech':<10}{'t':>3}" + "".join(f"{'eps=' + format(e, 'g'):>12}" for e in EPSILONS) + "   (mean absolute error)")
    seen = set()
    for r in out["rows"]:
        key = (r["kind"], r["cohort"], r["mechanism"], r.get("tolerance"))
        if key in seen:
            continue
        seen.add(key)
        line = "".join(f"{x['mae']:>12.2f}" for x in out["rows"] if (x["kind"], x["cohort"], x["mechanism"], x.get("tolerance")) == key)
        print(f"  {r['kind'] + ' ' + r['cohort']:<26}{r['mechanism']:<10}{r.get('tolerance', ''):>3}{line}")
    print(f"  queries within ({BUDGET[0]:g}, {BUDGET[1]:g}): " + ", ".join(
        f"eps {a['epsilon']:g}: basic {a['laplace_basic']} / zCDP {a['laplace_zcdp']}" for a in out["accounting"][::5]))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
