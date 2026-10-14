"""Phase 5c: the private API end to end, in STRICT mode (service identities on every hop, JWT + RBAC + ledger + audit)."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fedhealth.common.schemas import local_query
from fedhealth.coordinator.app import CoordinatorSecrets
from fedhealth.privacy.linkage.group import get_group
from fedhealth.synth.generate import generate
from fedhealth.synth.ground_truth import GroundTruth

from .helpers import headline_spec
from .leakscan import scan
from .system import build_system

FAST = "test768"
Q = headline_spec(a={}, b={}).model_dump(mode="json", exclude_none=True)     # true count 42: far above k = 10


def make_secure(tmp_path_factory, pki, name):
    d = tmp_path_factory.mktemp(name)
    generate(600, seed=5, out_dir=d)
    secrets = CoordinatorSecrets(os.urandom(32), os.urandom(32))
    system = build_system(d, psi_group=FAST, psi_groups=(FAST,), allow_insecure=False, secure=True, tls_dir=pki,
                          secrets=secrets, coordinator_db=d / "coordinator.db")
    ctx = system.secure
    for user, role in (("alice", "researcher"), ("rob", "researcher"), ("erin", "researcher"), ("carol", "clinician"),
                       ("audrey", "auditor")):
        ctx.users.create(user, role, f"pw-{user}")
    return SimpleNamespace(system=system, ctx=ctx, gt=GroundTruth(d / "ground_truth"), db=d / "coordinator.db",
                           client=system.client, pki=pki)


@pytest.fixture(scope="module")
def sec(tmp_path_factory, pki):
    return make_secure(tmp_path_factory, pki, "secure_world")


def login(sec, user) -> dict:
    r = sec.client.post("/auth/login", json={"username": user, "password": f"pw-{user}"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def ask(sec, user, query=Q, epsilon=1.0, **privacy):
    return sec.client.post("/v1/query", headers=login(sec, user), json={"query": query, "privacy": {"epsilon": epsilon, **privacy}})


# ============================================================================ authentication
def test_login_and_token_use(sec):
    r = sec.client.post("/auth/login", json={"username": "alice", "password": "pw-alice"})
    assert r.status_code == 200 and r.json()["role"] == "researcher" and r.json()["token_type"] == "bearer"
    me = sec.client.get("/v1/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).json()
    assert me["username"] == "alice" and me["aggs"] == ["count", "histogram"] and me["max_epsilon_per_query"] == 1.0


def test_unauthenticated_and_forged_requests_are_rejected_and_audited(sec):
    before = sec.ctx.audit.head()[0]
    assert sec.client.post("/v1/query", json={"query": Q, "privacy": {"epsilon": 1}}).status_code == 401
    assert sec.client.get("/v1/budget", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    expired = sec.ctx.tokens.issue("alice", "researcher", now=time.time() - 7200)
    assert sec.client.get("/v1/budget", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    ghost = sec.ctx.tokens.issue("alice", "superuser")                        # a validly signed token for a role that doesn't exist
    assert sec.client.get("/v1/budget", headers={"Authorization": f"Bearer {ghost}"}).status_code == 401
    entries = sec.ctx.audit.read(limit=sec.ctx.audit.head()[0] - before)
    assert len(entries) == 4 and all(e["action"] == "auth" and e["outcome"] == "denied" for e in entries)


def test_bad_logins_are_audited_and_throttled(sec):
    for _ in range(5):
        assert sec.client.post("/auth/login", json={"username": "victim", "password": "guess"}).status_code == 401
    assert sec.client.post("/auth/login", json={"username": "victim", "password": "pw-victim"}).status_code == 429
    reasons = {e["reason"] for e in sec.ctx.audit.read(actor="victim", limit=20)}
    assert {"bad credentials", "too many failed attempts"} <= reasons
    assert sec.client.post("/auth/login", json={"username": "alice", "password": "pw-alice", "extra": 1}).status_code == 422


# ============================================================================ the query path
def test_a_researcher_gets_a_noisy_answer_and_it_is_budgeted_and_audited(sec):
    truth = sec.gt.evaluate(headline_spec(a={}, b={}))
    r = ask(sec, "alice", epsilon=1.0)
    body = r.json()
    assert r.status_code == 200 and body["status"] == "released" and body["hospitals"] == ["A", "B", "C"]
    assert abs(body["result"] - truth) <= 15                                   # Laplace(scale 1): 15 is >14 sigma away
    assert body["epsilon_spent"] == 1.0 and body["epsilon_remaining"] == pytest.approx(9.0) and body["mechanism"] == "laplace"
    assert set(body) >= {"noise_scale", "expected_abs_error", "audit_seq"}
    entry = next(e for e in sec.ctx.audit.read(actor="alice", limit=5) if e["seq"] == body["audit_seq"])
    assert entry["action"] == "query" and entry["outcome"] == "released" and entry["epsilon_charged"] == 1.0
    assert entry["hospitals"] == ["A", "B", "C"] and entry["query"]["query"]["agg"] == "count"
    assert "result" not in str(entry)                                          # the released value is never logged


def test_role_and_cap_violations_are_denied_before_any_budget_or_hospital_is_touched(sec):
    sec.system.rec.clear()
    spent_before = sec.ctx.ledger.balance("alice").spent_epsilon
    cases = [
        (headline_spec(agg="avg", value_field="age").model_dump(mode="json", exclude_none=True), 1.0, {}, "avg"),     # not allowed aggregate
        (Q, 1.5, {}, "per-query cap"),                                                                                  # epsilon above cap
        (Q, 0.5, {"mechanism": "gaussian"}, "gaussian"),                                                                # mechanism not allowed
    ]
    for query, eps, extra, needle in cases:
        r = ask(sec, "alice", query, eps, **extra)
        assert r.status_code == 403 and needle in r.json()["detail"]["detail"], r.text
    assert sec.system.rec.events == [] and sec.ctx.ledger.balance("alice").spent_epsilon == spent_before
    last = [e for e in sec.ctx.audit.read(actor="alice", limit=12) if e["action"] == "query"][:3]    # (each ask() also logs in)
    assert [e["outcome"] for e in last] == ["denied"] * 3 and last[0]["reason"] and last[0]["query"] is not None
    assert ask(sec, "audrey").status_code == 403                               # auditors never query


def test_a_clinician_may_use_sum_avg_and_gaussian(sec):
    avg = headline_spec(agg="avg", value_field="adherence_pct", a={}, b={}, c={}).model_dump(mode="json", exclude_none=True)
    r = ask(sec, "carol", avg, 2.0)
    assert r.status_code == 200 and r.json()["status"] == "released"
    # 76 patients at epsilon 2: the noisy mean's sd is ~2.8 points, so +-5 failed ~8% of runs by chance; +-20 fails ~1e-5
    assert abs(r.json()["result"] - sec.gt.evaluate(headline_spec(agg="avg", value_field="adherence_pct", a={}, b={}, c={}))) < 20
    g = ask(sec, "carol", Q, 1.0, mechanism="gaussian")
    assert g.status_code == 200 and g.json()["mechanism"] == "gaussian" and g.json()["delta_spent"] == 1e-6


@pytest.mark.parametrize("bad_query", [
    {"agg": "select", "condition": "E11", "a": {}},
    {"agg": "count", "condition": "E11", "a": {}, "columns": ["full_name"]},
    {"agg": "count", "condition": "E11", "a": {"zip_code": "12345"}},
    {"agg": "histogram", "condition": "E11", "a": {}, "group_by": "national_id"},
])
def test_row_level_and_malformed_requests_are_refused_and_audited(sec, bad_query):
    sec.system.rec.clear()
    r = sec.client.post("/v1/query", headers=login(sec, "alice"), json={"query": bad_query, "privacy": {"epsilon": 1}})
    assert r.status_code == 422 and r.json()["detail"]["detail"] == "not a valid aggregate query"
    assert sec.system.rec.events == []
    assert sec.ctx.audit.read(actor="alice", limit=1)[0]["outcome"] == "refused"
    for broken in ({"query": Q}, {"query": Q, "privacy": {"epsilon": 1, "evil": 1}}, {"query": Q, "privacy": {"epsilon": 99}}, "str"):
        assert sec.client.post("/v1/query", headers=login(sec, "alice"), json=broken).status_code == 422


def test_budget_exhaustion_through_the_api(sec):
    headers = login(sec, "rob")
    statuses = []
    for _ in range(10):
        statuses.append(sec.client.post("/v1/query", headers=headers, json={"query": Q, "privacy": {"epsilon": 1}}).json()["status"])
    assert statuses == ["released"] * 10
    sec.system.rec.clear()
    r = sec.client.post("/v1/query", headers=headers, json={"query": Q, "privacy": {"epsilon": 1}})
    assert r.status_code == 200 and r.json()["status"] == "refused" and r.json()["result"] is None
    assert r.json()["epsilon_remaining"] == 0 and sec.system.rec.events == []
    assert sec.client.get("/v1/budget", headers=headers).json()["remaining_epsilon"] == 0
    assert sec.ctx.audit.read(actor="rob", limit=1)[0]["outcome"] == "refused"


def test_parallel_requests_cannot_overspend_a_budget(sec):
    sec.system.orchestrator.link(["A", "B", "C"])                              # share one linkage
    headers = login(sec, "erin")
    results = []
    barrier = threading.Barrier(14)

    def worker():
        c = TestClient(sec.system.coordinator)
        barrier.wait()
        results.append(c.post("/v1/query", headers=headers, json={"query": Q, "privacy": {"epsilon": 1}}).json()["status"])
    ts = [threading.Thread(target=worker) for _ in range(14)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count("released") == 10 and results.count("refused") == 4
    assert sec.ctx.ledger.balance("erin").remaining_epsilon == 0


def test_the_traffic_of_a_private_query_is_clean(sec):
    sec.system.orchestrator.link(["A", "B", "C"])
    sec.system.rec.clear()
    assert ask(sec, "carol", epsilon=0.5).json()["status"] == "released"
    ids = [p.national_id for p in sec.gt.persons.values()]
    assert len(sec.system.rec.events) > 20                                     # the whole MPC ran and was recorded
    assert scan(sec.system.rec.events, group=get_group(FAST), plain_ids=ids) == []


def test_an_answer_is_withheld_if_the_audit_entry_cannot_be_written(sec, monkeypatch):
    real = sec.ctx.audit.append

    def failing(**kw):
        if kw.get("action") == "query":
            raise RuntimeError("disk full")
        return real(**kw)
    monkeypatch.setattr(sec.ctx.audit, "append", failing)
    c = TestClient(sec.system.coordinator, raise_server_exceptions=False)
    r = c.post("/v1/query", headers=login(sec, "carol"), json={"query": Q, "privacy": {"epsilon": 0.5}})
    assert r.status_code == 500 and "result" not in r.text


# ============================================================================ audit visibility and integrity
def test_audit_scopes_and_verification(sec):
    mine = sec.client.get("/v1/audit?limit=200", headers=login(sec, "alice")).json()
    assert mine["scope"] == "own" and {e["actor"] for e in mine["entries"]} == {"alice"}
    everyone = sec.client.get("/v1/audit?limit=500", headers=login(sec, "audrey")).json()
    assert everyone["scope"] == "all" and {"alice", "rob", "carol", "audrey", "anonymous"} <= {e["actor"] for e in everyone["entries"]}
    v = sec.client.get("/v1/audit/verify", headers=login(sec, "audrey")).json()
    assert v["ok"] and v["entries"] > 20 and v["first_bad_seq"] is None
    assert sec.client.get("/v1/audit/verify", headers=login(sec, "alice")).status_code == 403
    assert sec.client.get("/v1/budgets", headers=login(sec, "alice")).status_code == 403
    accounts = {a["user_id"]: a for a in sec.client.get("/v1/budgets", headers=login(sec, "audrey")).json()["accounts"]}
    assert accounts["rob"]["remaining_epsilon"] == 0 and accounts["alice"]["total_epsilon"] == 10.0
    assert sec.client.get("/v1/audit?limit=9999", headers=login(sec, "audrey")).status_code == 422


def test_tampering_with_the_database_is_reported_by_the_api(tmp_path_factory, pki):
    s = make_secure(tmp_path_factory, pki, "tamper_world")
    assert ask(s, "alice").json()["status"] == "released"
    assert s.client.get("/v1/audit/verify", headers=login(s, "audrey")).json()["ok"]
    con = sqlite3.connect(str(s.db), isolation_level=None)                      # an insider with file access
    con.execute("DROP TRIGGER audit_no_update")
    con.execute("UPDATE audit_log SET micro_eps = 0 WHERE action = 'query'")    # 'that query cost nothing'
    con.close()
    v = s.client.get("/v1/audit/verify", headers=login(s, "audrey")).json()
    assert v["ok"] is False and v["first_bad_seq"] is not None and "mac mismatch" in v["reason"]


# ============================================================================ the hospitals' own gatekeeping
def test_hospitals_only_obey_the_coordinator_and_the_named_sender(sec):
    sys = sec.system
    a = sys.raw["A"]
    assert a.get("/health").status_code == 200                                               # health is open
    init = {"ring": ["A", "B"], "group": FAST}
    assert a.post(f"/psi/{'aa' * 8}/init", json=init).status_code == 401                    # no credentials
    assert a.post(f"/psi/{'aa' * 8}/init", json=init, headers={"Authorization": "Bearer junk"}).status_code == 401
    from fedhealth.privacy.encryption.service_auth import ServiceAuth
    as_b = ServiceAuth(sys.ids["hospital-b"], "hospital-a")
    as_coord = ServiceAuth(sys.ids["coordinator"], "hospital-a")
    assert a.post(f"/psi/{'aa' * 8}/init", json=init, auth=as_b).status_code == 403          # a peer may not issue control commands
    assert a.post(f"/mpc/{'bb' * 8}/output", auth=as_b).status_code == 403                  # ... nor fetch another hospital's output share
    assert a.post(f"/mpc/{'bb' * 8}/prepare", json={}, auth=as_b).status_code == 403
    assert a.post(f"/psi/{'aa' * 8}/inbox", json={"sender": "C", "round": 1, "blob": ""}, auth=as_b).status_code == 403   # B claims to be C
    assert a.post(f"/mpc/{'bb' * 8}/inshare", json={"sender": "C", "session": "0" * 32, "vectors": []},
                  auth=as_b).status_code == 403
    assert a.post(f"/psi/{'aa' * 8}/inbox", json={"sender": "B", "round": 1, "blob": ""}, auth=as_coord).status_code == 403  # coordinator may not inject peer data
    ok = a.post(f"/psi/{'aa' * 8}/init", json=init, auth=as_coord)
    assert ok.status_code == 200
    a.delete(f"/psi/{'aa' * 8}", auth=as_coord)


def test_strict_deployment_has_no_insecure_surface(sec):
    assert sec.client.post("/insecure/query", json=Q).status_code == 404
    assert sec.client.post("/insecure/mpc-query", json=Q).status_code == 404
    assert sec.system.raw["A"].post("/insecure/match", json={}).status_code == 404
    assert sec.client.get("/health").json()["private_api"] is True
