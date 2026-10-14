"""EVALUATION 3 (stretch) - Federated learning: does FedAvg match centralised training, and what does privacy cost?

SETUP
    Each hospital holds a `training_episodes` registry (identical schema, deliberately non-IID: A skews younger and sicker,
    B older, C cardiovascular). The task: predict whether a treated patient responded ("improved"). Metric: AUC on the
    pooled held-out test rows of all three hospitals (the test rows are read OFFLINE from the CSV export; no service ever
    returns them).

METHODS COMPARED
    centralised      train on the pooled rows (what you could do ONLY if the data could be shared) - the target
    local only       each hospital alone, tested on everyone's test rows
    FedAvg           through the real hospital services: only weight updates leave a hospital
    FedAvg + SecAgg  the same, but the coordinator receives only the SUM of updates (additive secret sharing)
    FedAvg + DP      clipped updates + Gaussian noise (client-level DP; zCDP epsilon), swept over the noise multiplier
    true model       the generator's own coefficients: the best any learner could do
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attacks"))
from common import common_args, ensure_world, save_json  # noqa: E402

from fedhealth.inproc import build_system  # noqa: E402
from fedhealth.privacy.federated.features import DIM, FEATURE_NAMES, row_to_vector, true_weights  # noqa: E402
from fedhealth.privacy.federated.fedavg import FedAvgConfig, LocalClient, run_fedavg  # noqa: E402
from fedhealth.privacy.federated.logreg import auc, predict_proba, train_local  # noqa: E402

NOISE_MULTIPLIERS = (0.3, 0.5, 1.0, 2.0, 5.0)
BASE = dict(rounds=20, local_epochs=3, lr=0.2)


def load(world: Path):
    data = {}
    for h in "ABC":
        rows = list(csv.DictReader(open(world / "ground_truth" / f"registry_{h}.csv", newline="", encoding="utf-8")))
        enc = lambda rs: (np.array([row_to_vector(r) for r in rs]), np.array([int(r["responded"]) for r in rs], dtype=float))
        tr, te = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "test"]
        data[h] = {"train": enc(tr), "test": enc(te), "rows": rows}
    return data


def run(world: Path, seeds: int = 8) -> dict:
    data = load(world)
    Xte = np.vstack([data[h]["test"][0] for h in "ABC"])
    yte = np.concatenate([data[h]["test"][1] for h in "ABC"])
    Xtr = np.vstack([data[h]["train"][0] for h in "ABC"])
    ytr = np.concatenate([data[h]["train"][1] for h in "ABC"])
    score = lambda w: auc(yte, predict_proba(w, Xte))
    ev = lambda w: {"auc": score(w)}
    rng = np.random.default_rng(0)

    wc = train_local(np.zeros(DIM), Xtr, ytr, epochs=60, lr=0.2, batch=64, l2=1e-3, rng=rng)
    local = {h: score(train_local(np.zeros(DIM), *data[h]["train"], epochs=60, lr=0.2, batch=64, l2=1e-3, rng=rng)) for h in "ABC"}
    tw = true_weights()

    system = build_system(world, psi_group="modp2048")
    plain = system.orchestrator.federated_train(FedAvgConfig(**BASE), evaluate=ev)            # REAL services, plain updates
    secure = system.orchestrator.federated_train(FedAvgConfig(**BASE, secure_agg=True), evaluate=ev)

    dp_rows = []
    clients = [LocalClient(*data[h]["train"], name=h) for h in "ABC"]
    for z in NOISE_MULTIPLIERS:
        aucs, eps = [], None
        for s in range(seeds):
            r = run_fedavg(clients, DIM, FedAvgConfig(**BASE, clip=1.0, noise_multiplier=z, seed=s), ev)
            aucs.append(r.history[-1]["auc"])
            eps = r.epsilon
        dp_rows.append({"noise_multiplier": z, "epsilon": eps, "auc_mean": float(np.mean(aucs)), "auc_min": float(np.min(aucs)),
                        "auc_max": float(np.max(aucs)), "seeds": seeds})

    # WHERE FEDERATION HELPS: when each hospital's own data is SMALL. 120 training rows per hospital, 8 random subsets.
    small = {"rows_per_hospital": 120, "local_only": [], "fedavg": [], "centralised": []}
    for s in range(seeds):
        r = np.random.default_rng(100 + s)
        sub = {}
        for h in "ABC":
            X, y = data[h]["train"]
            idx = r.choice(len(y), size=120, replace=False)
            sub[h] = (X[idx], y[idx])
        small["local_only"].append(float(np.mean([score(train_local(np.zeros(DIM), *sub[h], epochs=60, lr=0.2, batch=32, l2=1e-2, rng=r))
                                                    for h in "ABC"])))
        fed = run_fedavg([LocalClient(*sub[h], name=h) for h in "ABC"], DIM, FedAvgConfig(**BASE, l2=1e-2, batch=32, seed=s))
        small["fedavg"].append(score(fed.weights))
        small["centralised"].append(score(train_local(np.zeros(DIM), np.vstack([sub[h][0] for h in "ABC"]),
                                                      np.concatenate([sub[h][1] for h in "ABC"]), epochs=60, lr=0.2, batch=32, l2=1e-2, rng=r)))

    drug_idx = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("drug_")]
    idx = {n: FEATURE_NAMES.index(n) for n in ("age_z", "adherence_z", "extra_comorbidities", "severity_severe")}
    return {
        "evaluation": "federated learning", "rows_per_hospital": {h: {"train": len(data[h]["train"][1]), "test": len(data[h]["test"][1])} for h in "ABC"},
        "mean_age": {h: float(np.mean([int(r["age"]) for r in data[h]["rows"]])) for h in "ABC"},
        "response_rate": {h: float(np.mean([int(r["responded"]) for r in data[h]["rows"]])) for h in "ABC"},
        "config": BASE, "auc": {"centralised": score(wc), "fedavg": plain.history[-1]["auc"], "fedavg_secure": secure.history[-1]["auc"],
                                "local_only": local, "true_model": score(tw)},
        "secure_vs_plain_max_weight_diff": float(np.max(np.abs(plain.weights - secure.weights))),
        "convergence": [h["auc"] for h in plain.history],
        "coefficients": {n: {"learned": float(plain.weights[i]), "true": float(tw[i])} for n, i in idx.items()},
        "drug_coefficient_correlation": float(np.corrcoef(plain.weights[drug_idx], tw[drug_idx])[0, 1]),
        "small_data": small, "dp": dp_rows, "dp_note": "client-level (whole-hospital) DP via clipped updates + Gaussian noise; zCDP accounting; delta = 1e-6",
    }


def main() -> None:
    ap = common_args("Federated learning evaluation.")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    n = 1000 if args.fast else args.patients
    out = run(ensure_world(n, args.seed, args.source), args.seeds)
    path = save_json("federated", out)
    a = out["auc"]
    print(f"rows per hospital (train/test): {out['rows_per_hospital']}   mean age: { {h: round(v, 1) for h, v in out['mean_age'].items()} }")
    print(f"AUC  centralised {a['centralised']:.4f} | FedAvg {a['fedavg']:.4f} | FedAvg+SecAgg {a['fedavg_secure']:.4f} | "
          f"local {', '.join(f'{h}={v:.3f}' for h, v in a['local_only'].items())} | true model {a['true_model']:.4f}")
    sm = out["small_data"]
    print(f"  small local data ({sm['rows_per_hospital']} rows/hospital, {len(sm['fedavg'])} subsets): local-only {np.mean(sm['local_only']):.3f} | "
          f"FedAvg {np.mean(sm['fedavg']):.3f} | centralised {np.mean(sm['centralised']):.3f}")
    for r in out["dp"]:
        print(f"  DP z={r['noise_multiplier']:<4g} epsilon={r['epsilon']:>7.1f}   AUC {r['auc_mean']:.3f} [{r['auc_min']:.3f}, {r['auc_max']:.3f}]")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
