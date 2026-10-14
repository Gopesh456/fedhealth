"""Phase 8 (stretch): federated logistic regression with FedAvg - library, services, wire contents."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from fedhealth.common import vocab
from fedhealth.coordinator.orchestrator import ProtocolError
from fedhealth.privacy.encryption.service_auth import ServiceAuth
from fedhealth.privacy.federated.features import DIM, FEATURE_NAMES, featurize, row_to_vector, true_weights
from fedhealth.privacy.federated.fedavg import (FedAvgConfig, LocalClient, clip_update, dequantize, quantize, run_fedavg,
                                                secure_sum, zcdp_epsilon)
from fedhealth.privacy.federated.logreg import auc, gradient, log_loss, predict_proba, sigmoid, train_local
from fedhealth.synth.generate import generate

from .system import build_system


# ============================================================================ logistic regression
def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 6)), (rng.random(40) < 0.5).astype(float)
    w = rng.normal(size=6)
    g = gradient(w, X, y, l2=0.1)
    num = np.array([(log_loss(w + 1e-6 * e, X, y, 0.1) - log_loss(w - 1e-6 * e, X, y, 0.1)) / 2e-6 for e in np.eye(6)])
    assert np.allclose(g, num, atol=1e-6)


def test_training_reduces_loss_and_auc_is_correct():
    rng = np.random.default_rng(1)
    X = np.hstack([np.ones((300, 1)), rng.normal(size=(300, 3))])
    y = (sigmoid(X @ np.array([0.2, 1.5, -1.0, 0.0])) > rng.random(300)).astype(float)
    w = train_local(np.zeros(4), X, y, epochs=40, lr=0.3, rng=rng)
    assert log_loss(w, X, y) < log_loss(np.zeros(4), X, y)
    s = predict_proba(w, X)
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])             # O(n^2) definition of AUC
    assert auc(y, s) == pytest.approx(brute)
    assert auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5          # ties


# ============================================================================ features
def test_feature_encoding_is_public_and_consistent():
    assert DIM == len(FEATURE_NAMES) == 8 + len(vocab.CONDITION_CODES) + len(vocab.ALL_DRUGS)
    x = featurize(70, 1, 2, "severe", 90.0, 0.5, "E11", ["metformin", "empagliflozin"])
    assert x[0] == 1 and x[FEATURE_NAMES.index("cond_E11")] == 1 and x[FEATURE_NAMES.index("drug_metformin")] == 1
    assert x[FEATURE_NAMES.index("drug_empagliflozin")] == 1 and x[FEATURE_NAMES.index("severity_severe")] == 1
    row = {"age": 70, "male": 1, "extra_comorbidities": 2, "severity": "severe", "adherence_pct": 90.0,
           "baseline_lab_z": 0.5, "condition": "E11", "drugs": "metformin|empagliflozin"}
    assert np.array_equal(row_to_vector(row), x)
    assert true_weights().shape == (DIM,)


# ============================================================================ data and fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("fl_world")
    generate(600, seed=5, out_dir=d, registry_people=1500)
    data = {}
    for h in "ABC":
        rows = list(csv.DictReader(open(d / "ground_truth" / f"registry_{h}.csv", newline="", encoding="utf-8")))
        enc = lambda rs: (np.array([row_to_vector(r) for r in rs]), np.array([int(r["responded"]) for r in rs], dtype=float))
        tr, te = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "test"]
        data[h] = (*enc(tr), *enc(te), rows)
    Xte = np.vstack([data[h][2] for h in "ABC"])
    yte = np.concatenate([data[h][3] for h in "ABC"])
    return {"dir": d, "data": data, "Xte": Xte, "yte": yte}


def evaluate(world):
    return lambda w: {"auc": auc(world["yte"], predict_proba(w, world["Xte"]))}


def clients(world):
    return [LocalClient(world["data"][h][0], world["data"][h][1], h) for h in "ABC"]


CFG = FedAvgConfig(rounds=20, local_epochs=3, lr=0.2)


def test_registries_are_non_iid_but_follow_one_clinical_process(world):
    ages = {h: np.mean([int(r["age"]) for r in world["data"][h][4]]) for h in "ABC"}
    assert ages["B"] - ages["A"] > 4                                              # B skews older than A: covariate shift
    rates = [np.mean([int(r["responded"]) for r in world["data"][h][4]]) for h in "ABC"]
    assert all(0.4 < r < 0.62 for r in rates)
    assert all(len(world["data"][h][0]) > 800 for h in "ABC")


# ============================================================================ the library
def test_fedavg_matches_centralised_training(world):
    """Phase 8 exit criterion: FedAvg AUC within ~0.02 of centralised training on the pooled data."""
    Xtr = np.vstack([world["data"][h][0] for h in "ABC"])
    ytr = np.concatenate([world["data"][h][1] for h in "ABC"])
    wc = train_local(np.zeros(DIM), Xtr, ytr, epochs=60, lr=0.2, batch=64, l2=1e-3, rng=np.random.default_rng(0))
    central = auc(world["yte"], predict_proba(wc, world["Xte"]))
    fed = run_fedavg(clients(world), DIM, CFG, evaluate(world)).history[-1]["auc"]
    assert central > 0.70 and fed > central - 0.02
    ceiling = auc(world["yte"], predict_proba(true_weights(), world["Xte"]))        # the generating model itself
    assert fed <= ceiling + 0.01


def test_federated_model_recovers_the_true_clinical_signal(world):
    w = run_fedavg(clients(world), DIM, CFG).weights
    tw = true_weights()
    idx = {n: FEATURE_NAMES.index(n) for n in ("adherence_z", "age_z", "extra_comorbidities", "severity_severe")}
    assert w[idx["adherence_z"]] == pytest.approx(tw[idx["adherence_z"]], abs=0.2)             # 0.70 per 20 pp of adherence
    assert w[idx["age_z"]] < 0 and w[idx["extra_comorbidities"]] < 0 and w[idx["severity_severe"]] < 0   # right directions
    drugs = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("drug_")]
    assert np.corrcoef(w[drugs], tw[drugs])[0, 1] > 0.4


def test_secure_aggregation_gives_the_same_model_without_revealing_updates(world):
    rng = np.random.default_rng(3)
    ups = [rng.normal(size=DIM) for _ in range(3)]
    assert np.allclose(secure_sum(ups), np.sum(ups, axis=0), atol=1e-6)                        # only the SUM is reconstructed
    assert np.allclose(dequantize(quantize(ups[0])), ups[0], atol=1e-6)
    plain = run_fedavg(clients(world), DIM, CFG).weights
    secure = run_fedavg(clients(world), DIM, FedAvgConfig(rounds=20, local_epochs=3, lr=0.2, secure_agg=True)).weights
    assert np.allclose(plain, secure, atol=1e-4)


def test_clipping_bounds_each_hospitals_influence():
    d = np.ones(50) * 3
    assert np.linalg.norm(clip_update(d, 1.0)) == pytest.approx(1.0)
    assert np.array_equal(clip_update(d * 0.01, 1.0), d * 0.01)                                # small updates untouched
    assert np.array_equal(clip_update(d, None), d)


def test_dp_noise_degrades_accuracy_gracefully_and_is_accounted(world):
    ev = evaluate(world)
    base = run_fedavg(clients(world), DIM, CFG, ev).history[-1]["auc"]
    res = {}
    for z in (0.3, 1.0, 5.0):
        r = run_fedavg(clients(world), DIM, FedAvgConfig(rounds=20, local_epochs=3, lr=0.2, clip=1.0, noise_multiplier=z), ev)
        res[z] = (r.history[-1]["auc"], r.epsilon)
    assert res[0.3][0] > base - 0.03                                                            # light noise: barely hurts
    assert res[0.3][0] > res[1.0][0] > res[5.0][0] > 0.5                                        # monotone degradation, still above chance
    assert res[0.3][1] > res[1.0][1] > res[5.0][1] > 0                                          # more noise -> smaller epsilon
    with pytest.raises(ValueError):
        FedAvgConfig(noise_multiplier=1.0)                                                      # noise without a clip has no sensitivity


def test_zcdp_accounting_formula():
    rho = 20 / (2 * 1.0 ** 2)
    assert zcdp_epsilon(1.0, 20, 1e-6) == pytest.approx(rho + 2 * math.sqrt(rho * math.log(1e6)))
    assert zcdp_epsilon(2.0, 20) < zcdp_epsilon(1.0, 20) < zcdp_epsilon(1.0, 40)
    assert zcdp_epsilon(0.0, 20) == math.inf


# ============================================================================ across the services
@pytest.fixture(scope="module")
def system(world):
    return build_system(world["dir"], psi_group="test768", psi_groups=("test768",))


@pytest.mark.parametrize("secure", [False, True])
def test_services_reproduce_the_in_process_run_exactly(world, system, secure):
    cfg = FedAvgConfig(rounds=8, local_epochs=3, lr=0.2, secure_agg=secure)
    via_http = system.orchestrator.federated_train(cfg).weights
    in_proc = run_fedavg(clients(world), DIM, cfg).weights
    assert np.allclose(via_http, in_proc, atol=1e-5 if secure else 1e-9)


def test_services_with_dp_report_epsilon_and_still_learn(world, system):
    cfg = FedAvgConfig(rounds=10, local_epochs=3, lr=0.2, clip=1.0, noise_multiplier=0.5, secure_agg=True)
    out = system.orchestrator.federated_train(cfg, evaluate=evaluate(world))
    assert out.epsilon == pytest.approx(zcdp_epsilon(0.5, 10)) and out.history[-1]["auc"] > 0.62


def test_only_weights_or_random_shares_ever_leave_a_hospital(world, system):
    """Wire contents of a full secure round trip: no feature, label, row count or plain per-hospital update."""
    system.rec.clear()
    system.orchestrator.federated_train(FedAvgConfig(rounds=2, local_epochs=1, lr=0.2, secure_agg=True))
    ev = [e for e in system.rec.events if "/fl/" in e.path]
    assert len(ev) > 20
    forbidden = ("age", "male", "responded", "drugs", "adherence", "baseline_lab", "split", "episode", "rows", "n_train", "count")
    allowed = {"weights", "parties", "round_no", "epochs", "lr", "batch", "l2", "clip", "noise_multiplier", "seed",
               "sender", "blob", "share", "status"}
    shares = []
    for e in ev:
        for text in (e.request, e.response):
            if not text:
                continue
            doc = json.loads(text)
            assert set(doc) <= allowed, (e.path, set(doc) - allowed)
            assert not any(f in json.dumps(doc) for f in forbidden if f not in ("count",)), e.path
            shares += doc.get("share", [])
    assert not [e for e in ev if e.path.endswith("/update")], "secure mode must never request a plain update"
    train_replies = {e.response for e in ev if e.path.endswith("/train")}
    assert train_replies == {'{"status":"trained"}'}                                            # not even the row count
    assert shares and min(shares) > 2 ** 20                                                      # each sum-share is random-looking
    # the exchanged blobs are uniform field elements too
    from fedhealth.privacy.smpc.field import decode_vector
    import base64
    vals = [v for e in ev if e.path.endswith("/inshare") for v in decode_vector(base64.b64decode(json.loads(e.request)["blob"]))]
    assert vals and min(vals) > 2 ** 20


def test_in_plain_mode_the_coordinator_sees_each_update_but_nothing_else(world, system):
    system.rec.clear()
    system.orchestrator.federated_train(FedAvgConfig(rounds=1, local_epochs=1, lr=0.2, secure_agg=False))
    upd = [e for e in system.rec.events if e.path.endswith("/update")]
    assert len(upd) == 3 and all(set(json.loads(e.response)) == {"update"} for e in upd)
    assert not [e for e in system.rec.events if e.path.endswith("/inshare")]


def test_the_coordinator_cannot_regenerate_the_dp_noise_or_take_both_releases(world, system):
    """The coordinator chooses the training seed; if the noise came from it, it could regenerate and subtract it.
    Two hospitals trained with the same seed and round therefore hold DIFFERENT noisy updates each time; and a round
    released as shares cannot also be fetched in the clear."""
    a = system.raw["A"]
    body = {"weights": [0.0] * DIM, "parties": ["A", "B", "C"], "round_no": 1, "epochs": 1, "lr": 0.1, "batch": 32,
            "l2": 0.0, "clip": 1.0, "noise_multiplier": 1.0, "seed": 7}
    ups = []
    for sid in ("a1" * 8, "a2" * 8):
        assert a.post(f"/fl/{sid}/train", json=body).status_code == 200
        ups.append(np.array(a.post(f"/fl/{sid}/update").json()["update"]))
        a.delete(f"/fl/{sid}")
    assert not np.allclose(ups[0], ups[1])                                        # same seed, fresh noise
    sid = "a3" * 8
    assert a.post(f"/fl/{sid}/train", json=body).status_code == 200
    assert a.post(f"/fl/{sid}/update").status_code == 200
    assert a.post(f"/fl/{sid}/share").status_code == 409                          # already released in the clear
    a.delete(f"/fl/{sid}")


def test_a_hospital_failing_mid_round_aborts_training_and_leaves_no_state_behind(world, system, monkeypatch):
    real_post = system.orchestrator.clients["B"].post

    def flaky(path, **kw):
        if path.endswith("/train"):
            raise ProtocolError("hospital B went away")                       # B dies while the others have already trained
        return real_post(path, **kw)
    monkeypatch.setattr(system.orchestrator.clients["B"], "post", flaky)
    with pytest.raises(ProtocolError, match="went away"):
        system.orchestrator.federated_train(FedAvgConfig(rounds=3, local_epochs=1, lr=0.2, secure_agg=True))
    for h in "ABC":
        assert system.apps[h].state.fl.sessions == {}, f"hospital {h} kept the failed round's update in memory"


# ============================================================================ strict mode
def test_fl_endpoints_obey_the_service_identity_rules(world, pki):
    sec = build_system(world["dir"], psi_group="test768", psi_groups=("test768",), allow_insecure=False, secure=True, tls_dir=pki)
    a = sec.raw["A"]
    as_b = ServiceAuth(sec.ids["hospital-b"], "hospital-a")
    as_coord = ServiceAuth(sec.ids["coordinator"], "hospital-a")
    sid = "ef" * 8
    body = {"weights": [0.0] * DIM, "parties": ["A", "B", "C"], "round_no": 1, "epochs": 1, "lr": 0.1, "batch": 32, "l2": 0.0}
    assert a.post(f"/fl/{sid}/train", json=body).status_code == 401                              # no credentials
    assert a.post(f"/fl/{sid}/train", json=body, auth=as_b).status_code == 403                   # a peer may not start training
    assert a.post(f"/fl/{sid}/update", auth=as_b).status_code == 403                             # nor fetch another hospital's update
    assert a.post(f"/fl/{sid}/inshare", json={"sender": "C", "blob": ""}, auth=as_b).status_code == 403   # B claiming to be C
    assert a.post(f"/fl/{sid}/train", json={**body, "weights": [0.0]}, auth=as_coord).status_code == 400   # wrong dimension
    assert a.post(f"/fl/{sid}/train", json={**body, "noise_multiplier": 1.0}, auth=as_coord).status_code == 400  # noise without clip
    assert a.post(f"/fl/{sid}/train", json={**body, "lr": 50}, auth=as_coord).status_code == 422             # validated hyper-parameters
    for few in (["A"], ["A", "B"], ["B", "A", "C"]):                                            # a round of one would
        assert a.post(f"/fl/{sid}/train", json={**body, "parties": few}, auth=as_coord).status_code == 400  # "aggregate" A alone
    assert a.post(f"/fl/{sid}/train", json=body, auth=as_coord).status_code == 200
    assert a.post(f"/fl/{sid}/update", auth=as_coord).status_code == 403                         # no plain updates in strict mode
    a.delete(f"/fl/{sid}", auth=as_coord)
    out = sec.orchestrator.federated_train(FedAvgConfig(rounds=2, local_epochs=1, lr=0.2, secure_agg=True))
    assert out.weights.shape == (DIM,)
