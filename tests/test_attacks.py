"""Phase 6: the attack demos as regression tests. If a defence regresses, the attack starts succeeding and these fail.

Small world (1,000 patients, insecure 768-bit PSI group for speed); the full-size numbers are in evaluation/RESULTS.md.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attacks"))
import budget_exhaustion  # noqa: E402
import common  # noqa: E402
import differencing  # noqa: E402
import hmac_dictionary  # noqa: E402
import reidentification  # noqa: E402

from fedhealth.synth.generate import generate  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("attack_world")
    generate(1000, seed=42, out_dir=d)
    return d


@pytest.fixture(scope="module")
def oracle(world):
    return common.ExactOracle(world, psi_group="test768")


# ============================================================================ attack 1: differencing
@pytest.fixture(scope="module")
def diff(oracle):
    return differencing.run(oracle, n_targets=20, trials=1500, seed=11)


def test_differencing_succeeds_without_dp_and_with_the_k_threshold_alone(diff):
    none, konly = diff["rows"][0], diff["rows"][1]
    assert none["accuracy"] == 1.0 and konly["accuracy"] == 1.0, "exact answers must leak the target's outcome"
    assert konly["unreleased"] == 0, "both differencing queries clear k=10, so nothing is suppressed - that is the point"
    assert diff["responders"] == 10 and diff["n_targets"] == 20          # balanced targets: 50% is the honest baseline


def test_the_k_threshold_stops_only_the_direct_question(diff):
    d = diff["direct_attempt"]
    assert d["k_only"]["suppressed"] == d["k_only"]["queries"] == 20
    assert d["no_protection"]["suppressed"] == 0


def test_differential_privacy_defeats_differencing_and_matches_theory(diff):
    dp = [r for r in diff["rows"] if r["epsilon"]]
    assert [r["epsilon"] for r in dp] == [0.1, 0.5, 1.0, 2.0, 5.0]
    for r in dp:
        sigma = math.sqrt(r["analytic_accuracy"] * (1 - r["analytic_accuracy"]) / r["trials"])
        assert abs(r["accuracy"] - r["analytic_accuracy"]) < 4.5 * sigma, r          # measured == exact formula
        assert r["accuracy"] <= r["dp_bound"] + 4 * sigma                           # and below the DP ceiling for ANY attacker
    acc = [r["accuracy"] for r in dp]
    assert acc == sorted(acc)                                                       # more epsilon -> more attacker success
    assert acc[0] < 0.56 and acc[2] < 0.70 and acc[-1] > 0.95                       # eps 0.1 ~ guessing; eps 5 ~ no protection


def test_the_attack_premise_is_verified_through_the_real_pipeline(oracle, diff):
    assert oracle.pipeline_runs >= 2 * 20, "every distinct query ran end to end (PSI + MPC), checked against ground truth"


# ============================================================================ attack 2: re-identification
@pytest.fixture(scope="module")
def reid(world):
    return reidentification.run(world / "ground_truth", coverage=0.7, seed=5)


def test_removing_names_is_not_anonymisation(reid):
    top = next(r for r in reid["defences"] if r["label"].startswith("exact birth year"))
    assert top["unique_in_release"] > 0.6 and top["reid_rate"] > 0.4 and top["precision"] > 0.85
    one = {r["qi"]: r for r in reid["by_qi"]}
    assert one["sex"]["reid_rate"] < 0.01 and one["birth_year+sex+zip"]["reid_rate"] > 10 * one["birth_year+sex"]["reid_rate"]


def test_generalisation_and_k_anonymity_work_at_a_cost(reid):
    d = {r["label"]: r for r in reid["defences"]}
    exact, gen = d["exact birth year + sex + ZIP5 (names removed only)"], d["age band + sex + ZIP3 (generalised)"]
    k5, k10 = d["generalised + 5-anonymity"], d["generalised + 10-anonymity"]
    assert gen["reid_rate"] < exact["reid_rate"] / 5
    assert k5["reid_rate"] <= gen["reid_rate"] and k10["reid_rate"] <= 0.01
    assert k10["retained_fraction"] < k5["retained_fraction"] < gen["retained_fraction"] == 1.0   # privacy costs records
    assert k10["min_group_size"] >= 10 and k5["min_group_size"] >= 5


# ============================================================================ attack 3: budget exhaustion
@pytest.fixture(scope="module")
def budget(oracle):
    return budget_exhaustion.run(oracle, eps=0.5, budget=10.0, attackers=20, max_n=300, seed=3)


def test_the_ledger_stops_the_averaging_attack(budget):
    l = budget["with_ledger"]
    assert l["answered"] == 20 and l["first_refusal_at_query"] == 21 and l["refused"] == 10
    assert l["epsilon_spent"] == 10.0 and l["attempted"] == 30


def test_without_a_ledger_the_noise_averages_away(budget):
    u = {r["queries"]: r["mean_abs_error"] for r in budget["unlimited"]}
    assert u[300 if 300 in u else 200] < u[1] / 3                                    # volume buys back accuracy
    assert u[20] > u[200]                                                            # ... and the ledger's cut-off (N=20) is where it stops
    assert budget["with_ledger"]["final_abs_error"] < 3 * budget["single_query_expected_abs_error"] + 1


# ============================================================================ bonus: HMAC dictionary
def test_hmac_pseudonyms_fall_to_a_dictionary_but_psi_does_not(world):
    out = hmac_dictionary.run(world / "ground_truth", decoys=20_000)
    assert out["recovered_from_b"] == out["b_patients"] and out["false_positives"] == 0
    assert out["recovered_b_only"] == out["b_only_patients"] > 0                       # patients A never treated
    assert out["psi_non_shared_patients_recovered"] == 0 and out["psi_candidates_tested"] > 0
