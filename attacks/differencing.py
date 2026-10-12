"""ATTACK 1 - Differencing.

THE ATTACK
    The attacker wants to know whether ONE target patient responded to treatment. They know the target's age,
    condition X and drug Y (e.g. from a leaked referral letter), and that the target is the only patient of that
    exact age in that (condition, drug) group. They never ask about the target directly - a cohort of 1 would be
    suppressed. Instead they ask two LARGE, perfectly innocent queries that differ by exactly one patient:

        Q1 = count( condition X, drug Y, outcome improved, age >= t   )
        Q2 = count( condition X, drug Y, outcome improved, age >= t+1 )

        Q1 - Q2  =  number of improved patients aged exactly t  =  1 if the target improved, else 0.

    Both counts are far above k = 10, so the k-threshold releases them. The attacker guesses
    "improved" iff  Q1 - Q2 >= 1.

WHAT WE MEASURE (accuracy of that guess over many targets x many repetitions; 50% = blind guessing)
    no protection         exact counts                       -> 100%
    k-threshold only      suppression, no noise              -> 100%   (the k-threshold does NOT stop differencing)
    full DP, epsilon e    discrete Laplace noise on each query -> approaches 50% as e shrinks
    and the DIRECT attempt (age == t only) to show what the k-threshold DOES stop.

THEORY (to check the measurement against)
    Each answer carries independent discrete-Laplace noise; the difference N = X1 - X2 has
    P(N = 0) = (1-a)(1+a^2)/(1+a)^3  with a = e^-epsilon, and the attacker's accuracy is exactly 0.5 + P(N=0)/2.
    Differential privacy additionally bounds ANY attacker: after two epsilon-DP answers, accuracy <= e^(2e)/(1+e^(2e)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from common import (ExactOracle, common_args, oracle_from_args, save_json, seeded_randbelow, wilson)
from fedhealth.common.schemas import QuerySpec
from fedhealth.coordinator.release import PrivacyParams, ReleasePolicy, release

EPSILONS = (0.1, 0.5, 1.0, 2.0, 5.0)


@dataclass(frozen=True)
class Target:
    person_id: int
    age: int
    condition: str
    drug: str
    improved: bool


def count_spec(cond: str, drug: str, age_min: int, age_max: int | None = None, outcome: str = "improved") -> QuerySpec:
    a = {"age_min": age_min} if age_max is None else {"age_min": age_min, "age_max": age_max}
    return QuerySpec.model_validate({"agg": "count", "condition": cond, "a": a, "b": {"outcome": outcome}, "c": {"drug": drug}})


def find_targets(oracle: ExactOracle, n_targets: int, k: int = 10, seed: int = 0) -> list[Target]:
    """Patients the attacker can single out by age inside a (condition, drug) group that is large enough for both
    differencing queries to clear the k-threshold. Balanced between responders and non-responders."""
    gt = oracle.gt
    groups: dict[tuple[str, str], list[tuple[int, int, bool]]] = {}
    for e in gt.episodes:
        p = gt.persons[e.person_id]
        if p.membership != "ABC" or not e.outcome:
            continue
        age = 2025 - p.birth_year
        for drug in set(e.drugs):
            groups.setdefault((e.condition, drug), []).append((e.person_id, age, e.outcome == "improved"))
    found: list[Target] = []
    for (cond, drug), members in groups.items():
        by_age: dict[int, list] = {}
        for m in members:
            by_age.setdefault(m[1], []).append(m)
        for age, ms in by_age.items():
            if len(ms) != 1:
                continue                                       # the target must be the ONLY patient of that exact age
            improved_older = sum(1 for _, a, imp in members if a >= age + 1 and imp)
            if improved_older >= k + 2:                        # both Q1 and Q2 stay above the k-threshold
                found.append(Target(ms[0][0], age, cond, drug, ms[0][2]))
    rng = np.random.default_rng(seed)
    yes = [t for t in found if t.improved]
    no = [t for t in found if not t.improved]
    rng.shuffle(yes)
    rng.shuffle(no)
    half = n_targets // 2
    return yes[:half] + no[:half]


def analytic_accuracy(eps: float) -> float:
    a = math.exp(-eps)
    return 0.5 + (1 - a) * (1 + a * a) / (2 * (1 + a) ** 3)


def dp_bound(eps: float) -> float:
    return math.exp(2 * eps) / (1 + math.exp(2 * eps))


def attack(oracle: ExactOracle, targets: list[Target], policy: ReleasePolicy, privacy: PrivacyParams | None,
           trials: int, seed: int) -> dict:
    rb = seeded_randbelow(seed)
    coin = np.random.default_rng(seed + 1)
    privacy = privacy or PrivacyParams(epsilon=1.0)
    correct = unreleased = 0
    for i in range(trials):
        t = targets[i % len(targets)]
        s1, s2 = count_spec(t.condition, t.drug, t.age), count_spec(t.condition, t.drug, t.age + 1)
        r1 = release(s1, oracle.exact(s1), privacy, policy, rb)
        r2 = release(s2, oracle.exact(s2), privacy, policy, rb)
        if r1.value is None or r2.value is None:               # suppressed: the attacker learns nothing and guesses
            unreleased += 1
            guess = bool(coin.integers(0, 2))
        else:
            guess = (r1.value - r2.value) >= 1
        correct += guess == t.improved
    lo, hi = wilson(correct, trials)
    return {"accuracy": correct / trials, "ci95": [lo, hi], "trials": trials, "unreleased": unreleased}


def direct_attempt(oracle: ExactOracle, targets: list[Target], policy: ReleasePolicy, privacy: PrivacyParams) -> dict:
    """Ask about the target directly (age == t): a cohort of 1 - what does the k-threshold do?"""
    rb = seeded_randbelow(7)
    suppressed = 0
    for t in targets:
        s = count_spec(t.condition, t.drug, t.age, t.age)
        suppressed += release(s, oracle.exact(s), privacy, policy, rb).fully_suppressed
    return {"queries": len(targets), "suppressed": suppressed}


def run(oracle: ExactOracle, n_targets: int = 40, trials: int = 3000, seed: int = 11) -> dict:
    targets = find_targets(oracle, n_targets, seed=seed)
    assert len(targets) >= 10, "world too small to find isolatable targets"
    # sanity: through the REAL pipeline, the exact difference reveals the target's outcome in every case
    for t in targets:
        d = oracle.exact(count_spec(t.condition, t.drug, t.age)).count - oracle.exact(count_spec(t.condition, t.drug, t.age + 1)).count
        assert d == int(t.improved), "attack premise broken"

    rows = []
    none = attack(oracle, targets, ReleasePolicy(k=0, dp_enabled=False), None, trials, seed)
    rows.append({"mode": "no protection (exact answers)", "epsilon": None, **none})
    konly = attack(oracle, targets, ReleasePolicy(k=10, dp_enabled=False), None, trials, seed)
    rows.append({"mode": "k-threshold only (k=10, no noise)", "epsilon": None, **konly})
    for eps in EPSILONS:
        r = attack(oracle, targets, ReleasePolicy(k=10), PrivacyParams(epsilon=eps), trials, seed + int(eps * 100))
        rows.append({"mode": "k-threshold + differential privacy", "epsilon": eps, **r,
                     "analytic_accuracy": analytic_accuracy(eps), "dp_bound": dp_bound(eps)})
    out = {
        "attack": "differencing", "n_targets": len(targets), "trials_per_setting": trials,
        "responders": sum(t.improved for t in targets), "pipeline_queries_checked": oracle.pipeline_runs,
        "target_ages": [t.age for t in targets][:5],
        "direct_attempt": {"k_only": direct_attempt(oracle, targets, ReleasePolicy(k=10, dp_enabled=False), PrivacyParams(epsilon=1.0)),
                           "no_protection": direct_attempt(oracle, targets, ReleasePolicy(k=0, dp_enabled=False), PrivacyParams(epsilon=1.0))},
        "rows": rows,
    }
    return out


def main() -> None:
    ap = common_args("Differencing attack: two queries that differ by one patient.")
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--trials", type=int, default=3000)
    args = ap.parse_args()
    out = run(oracle_from_args(args), args.targets, args.trials)
    path = save_json("attack_differencing", out)
    print(f"Differencing attack: {out['n_targets']} isolatable targets ({out['responders']} responders), "
          f"{out['trials_per_setting']} trials per setting.   (50% = blind guess)")
    d = out["direct_attempt"]
    print(f"  direct question about the target (cohort of 1): k-threshold suppressed {d['k_only']['suppressed']}/{d['k_only']['queries']}, "
          f"exact answers leaked {d['no_protection']['queries'] - d['no_protection']['suppressed']}/{d['no_protection']['queries']} without it")
    print(f"  {'setting':<42}{'eps':>5}  {'attacker accuracy':>18}   {'analytic':>8}  {'DP bound':>8}")
    for r in out["rows"]:
        eps = "-" if r["epsilon"] is None else f"{r['epsilon']:g}"
        extra = f"   {r['analytic_accuracy']:8.3f}  {r['dp_bound']:8.3f}" if r["epsilon"] else ""
        print(f"  {r['mode']:<42}{eps:>5}  {r['accuracy']:>10.1%} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]{extra}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
