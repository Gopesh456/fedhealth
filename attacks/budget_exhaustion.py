"""ATTACK 3 - Averaging out the noise by repeating a query ("budget exhaustion").

THE ATTACK
    Differential privacy hides one patient behind noise of scale ~1/epsilon. But noise is random: ask the SAME
    question N times and average the answers, and the error shrinks like 1/sqrt(N). Without a limit, any amount of
    privacy per query can be bought back by asking more queries.

THE DEFENCE
    The epsilon ledger. Each answer costs epsilon; by composition N answers cost N*epsilon. Once a requester's budget
    (researcher: 10) is spent the system REFUSES - before touching any hospital.

WHAT WE MEASURE
    * attacker WITHOUT a budget limit: mean |error| of the running average after N queries (simulated from the
      real release stage, 100 independent attackers per point), and the total epsilon they would have burned,
    * attacker WITH the ledger, driven through the REAL pipeline (QueryService): how many queries are answered,
      where the first refusal happens, and the accuracy they are stuck with afterwards.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from common import ExactOracle, common_args, oracle_from_args, save_json, seeded_randbelow
from fedhealth.common.schemas import QuerySpec
from fedhealth.coordinator.release import PrivacyParams, ReleasePolicy, release
from fedhealth.coordinator.service import QueryService
from fedhealth.privacy.dp.ledger import EpsilonLedger, make_engine
from fedhealth.privacy.dp.mechanisms import LaplaceMechanism

SPEC = QuerySpec.model_validate({"agg": "count", "condition": "E11", "a": {"age_min": 40, "age_max": 65},
                                 "b": {"outcome": "improved"}, "c": {"drug": "metformin"}})
CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def theory_mean_abs_error(eps: float, n: int) -> float:
    """E|mean of n discrete-Laplace draws| ~ sqrt(2/pi) * sd / sqrt(n)  (CLT; good for n >= ~5)."""
    a = math.exp(-eps)
    sd = math.sqrt(2 * a) / (1 - a)
    return math.sqrt(2 / math.pi) * sd / math.sqrt(n)


def run(oracle: ExactOracle, eps: float = 0.5, budget: float = 10.0, attackers: int = 100, max_n: int = 1000, seed: int = 3) -> dict:
    exact = oracle.exact(SPEC)
    truth = exact.count

    # --- 1. unlimited attacker (no ledger): noise averages away -------------------------------------------------
    rb = seeded_randbelow(seed)
    policy, privacy = ReleasePolicy(), PrivacyParams(epsilon=eps)
    sums = np.zeros((attackers, max_n))
    for a in range(attackers):
        draws = np.array([release(SPEC, exact, privacy, policy, rb).value for _ in range(max_n)], dtype=float)
        sums[a] = np.cumsum(draws) / np.arange(1, max_n + 1)
    unlimited = [{"queries": n, "mean_abs_error": float(np.mean(np.abs(sums[:, n - 1] - truth))),
                  "theory": theory_mean_abs_error(eps, n), "epsilon_spent": n * eps} for n in CHECKPOINTS if n <= max_n]

    # --- 2. the same attacker against the ledger, through the REAL pipeline -------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(str(Path(tmp) / "coordinator.db"))
        ledger = EpsilonLedger(engine)
        ledger.ensure_user("attacker", budget)
        service = QueryService(oracle.orchestrator, ledger, ReleasePolicy())
        answers, first_refusal, statuses = [], None, []
        for i in range(1, int(budget / eps) + 11):                      # ten more attempts than the budget allows
            out = service.execute("attacker", SPEC, privacy)
            statuses.append(out.status)
            if out.status == "released":
                answers.append(out.released.value)
            elif first_refusal is None:
                first_refusal = i
        spent = ledger.balance("attacker").spent_epsilon
        engine.dispose()                                                # release the file (Windows cannot delete an open DB)
    estimate = float(np.mean(answers))
    limited = {"budget": budget, "epsilon_per_query": eps, "attempted": len(statuses), "answered": len(answers),
               "refused": statuses.count("refused"), "first_refusal_at_query": first_refusal, "epsilon_spent": spent,
               "final_estimate": estimate, "final_abs_error": abs(estimate - truth),
               "expected_abs_error_at_that_point": theory_mean_abs_error(eps, len(answers))}
    return {"attack": "budget exhaustion", "query": "count(E11, age 40-65, metformin, improved)", "true_value": truth,
            "epsilon_per_query": eps, "attackers_simulated": attackers, "unlimited": unlimited, "with_ledger": limited,
            "single_query_expected_abs_error": LaplaceMechanism(eps, 1).expected_abs_error,     # exact, not the CLT approximation
            "note": "theory = CLT approximation of E|mean|, accurate for N >= ~50; N=1 is exactly single_query_expected_abs_error"}


def main() -> None:
    ap = common_args("Repeated-query averaging vs the epsilon ledger.")
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--attackers", type=int, default=100)
    args = ap.parse_args()
    out = run(oracle_from_args(args), args.epsilon, attackers=args.attackers)
    path = save_json("attack_budget_exhaustion", out)
    print(f"Target: {out['query']} (true value {out['true_value']}), epsilon {out['epsilon_per_query']} per query")
    print(f"  {'queries N':>10}{'epsilon burned':>16}{'mean |error| (no ledger)':>28}{'CLT approx':>12}")
    for r in out["unlimited"]:
        print(f"  {r['queries']:>10}{r['epsilon_spent']:>16.1f}{r['mean_abs_error']:>28.3f}{r['theory']:>9.3f}")
    l = out["with_ledger"]
    print(f"  WITH the ledger (budget {l['budget']}): {l['answered']} of {l['attempted']} attempts answered, first refusal at query "
          f"{l['first_refusal_at_query']}, epsilon spent {l['epsilon_spent']}; final estimate error {l['final_abs_error']:.2f} "
          f"(expected ~{l['expected_abs_error_at_that_point']:.2f}) - stuck there for good.")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
