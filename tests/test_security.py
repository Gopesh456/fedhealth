"""Phase 5b: PKI, service identity, JWT, RBAC, passwords, login throttle, and the audit hash chain."""
from __future__ import annotations

import base64
import json
import sqlite3
import threading
import time

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from sqlalchemy import text

from fedhealth.common.sqlite import make_sqlite_engine
from fedhealth.coordinator.users import LoginThrottle, UserStore
from fedhealth.privacy.access_control import jwt_auth
from fedhealth.privacy.access_control.jwt_auth import (AuthError, TokenAuthority, hash_password, verify_password)
from fedhealth.privacy.access_control.rbac import Forbidden, authorize_query, load_policy
from fedhealth.privacy.audit import GENESIS, AuditLog
from fedhealth.privacy.encryption import tls
from fedhealth.privacy.encryption.service_auth import (ServiceAuthError, ServiceIdentity, ServiceVerifier)


# ============================================================================ PKI
def test_every_certificate_is_issued_by_the_ca_and_fit_for_purpose(pki):
    ca = tls.load_cert(pki / "ca.pem")
    bc = ca.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert bc.ca and bc.path_length == 0
    for svc in tls.SERVICES:
        cert = tls.load_cert(pki / f"{svc}.crt")
        cert.verify_directly_issued_by(ca)
        assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == svc
        assert not cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        assert ExtendedKeyUsageOID.SERVER_AUTH in eku and ExtendedKeyUsageOID.CLIENT_AUTH in eku
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert svc in san.get_values_for_type(x509.DNSName) and "localhost" in san.get_values_for_type(x509.DNSName)
        assert any(str(ip) == "127.0.0.1" for ip in san.get_values_for_type(x509.IPAddress))
        assert isinstance(cert.public_key(), ec.EllipticCurvePublicKey)


def test_a_certificate_from_a_foreign_ca_is_rejected(pki, other_pki, tmp_path):
    import shutil
    mixed = tmp_path / "mixed"
    shutil.copytree(pki, mixed)
    shutil.copy(other_pki / "hospital-b.crt", mixed / "hospital-b.crt")        # an impostor's certificate
    with pytest.raises(Exception):
        ServiceVerifier("hospital-a", mixed)                                    # chain check fails at start-up


# ============================================================================ service tokens
@pytest.fixture(scope="module")
def ids(pki):
    return {s: ServiceIdentity.load(pki, s) for s in tls.SERVICES}


def test_service_token_roundtrip_and_audience_binding(pki, ids):
    v_a, v_c = ServiceVerifier("hospital-a", pki), ServiceVerifier("hospital-c", pki)
    token = ids["coordinator"].token_for("hospital-a")
    assert v_a.verify(token) == "coordinator"
    with pytest.raises(ServiceAuthError):
        v_c.verify(token)                                                       # minted for A, useless against C


def test_service_tokens_cannot_be_forged_replayed_late_or_tampered(pki, ids, other_pki):
    v = ServiceVerifier("hospital-a", pki)
    with pytest.raises(ServiceAuthError):
        v.verify(ids["coordinator"].token_for("hospital-a", ttl=-5))            # expired
    forged = ServiceIdentity.load(other_pki, "coordinator").token_for("hospital-a")   # right name, wrong key/CA
    with pytest.raises(ServiceAuthError):
        v.verify(forged)
    hospital_b_pretending = jwt.encode({"iss": "coordinator", "aud": "hospital-a", "iat": int(time.time()),
                                        "exp": int(time.time()) + 60},
                                       ids["hospital-b"]._key, algorithm="ES256")      # B signs as the coordinator
    with pytest.raises(ServiceAuthError):
        v.verify(hospital_b_pretending)
    none_alg = jwt.encode({"iss": "coordinator", "aud": "hospital-a", "iat": 1, "exp": 2 ** 31}, None, algorithm="none")
    with pytest.raises(ServiceAuthError):
        v.verify(none_alg)
    good = ids["coordinator"].token_for("hospital-a")
    head, body, sig = good.split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=="))
    payload["aud"] = "hospital-b"
    tampered = ".".join([head, base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("="), sig])
    with pytest.raises(ServiceAuthError):
        ServiceVerifier("hospital-b", pki).verify(tampered)
    with pytest.raises(ServiceAuthError):
        v.verify(jwt.encode({"iss": "mallory", "aud": "hospital-a", "exp": 2 ** 31, "iat": 1}, ids["hospital-b"]._key, algorithm="ES256"))


# ============================================================================ user JWTs
SECRET = b"s" * 32


def test_jwt_roundtrip_and_claims():
    ta = TokenAuthority(SECRET, ttl_seconds=60)
    c = ta.verify(ta.issue("alice", "researcher"))
    assert (c.sub, c.role) == ("alice", "researcher") and c.exp > time.time()


def test_jwt_rejections():
    ta = TokenAuthority(SECRET, ttl_seconds=60)
    with pytest.raises(AuthError):
        ta.verify(ta.issue("alice", "researcher", now=time.time() - 3600))                        # expired
    with pytest.raises(AuthError):
        TokenAuthority(b"x" * 32).verify(ta.issue("alice", "researcher"))                         # other secret
    good = ta.issue("alice", "researcher")
    head, body, sig = good.split(".")
    forged_body = base64.urlsafe_b64encode(json.dumps({**json.loads(base64.urlsafe_b64decode(body + "==")), "role": "auditor"}).encode()).decode().rstrip("=")
    with pytest.raises(AuthError):
        ta.verify(f"{head}.{forged_body}.{sig}")                                                  # privilege escalation attempt
    with pytest.raises(AuthError):
        ta.verify(jwt.encode({"sub": "a", "role": "auditor", "iss": jwt_auth.ISSUER, "iat": 1, "exp": 2 ** 31}, None, algorithm="none"))
    with pytest.raises(AuthError):                                                                # alg confusion: other HMAC alg
        ta.verify(jwt.encode({"sub": "a", "role": "auditor", "iss": jwt_auth.ISSUER, "iat": 1, "exp": 2 ** 31}, SECRET, algorithm="HS512"))
    with pytest.raises(AuthError):                                                                # wrong issuer
        ta.verify(jwt.encode({"sub": "a", "role": "auditor", "iss": "evil", "iat": 1, "exp": 2 ** 31}, SECRET, algorithm="HS256"))
    with pytest.raises(AuthError):                                                                # missing role claim
        ta.verify(jwt.encode({"sub": "a", "iss": jwt_auth.ISSUER, "iat": 1, "exp": 2 ** 31}, SECRET, algorithm="HS256"))
    with pytest.raises(AuthError):
        ta.verify("not-a-jwt")
    with pytest.raises(ValueError):
        TokenAuthority(b"short")


# ============================================================================ RBAC
POLICY = load_policy()


@pytest.mark.parametrize("role,agg,mech,eps,allowed", [
    ("researcher", "count", "laplace", 1.0, True),
    ("researcher", "histogram", "laplace", 0.5, True),
    ("researcher", "sum", "laplace", 0.5, False),            # not an allowed aggregate
    ("researcher", "avg", "laplace", 0.5, False),
    ("researcher", "count", "gaussian", 0.5, False),         # mechanism not allowed
    ("researcher", "count", "laplace", 1.5, False),          # above the per-query cap
    ("clinician", "avg", "laplace", 2.0, True),
    ("clinician", "sum", "gaussian", 1.0, True),
    ("clinician", "count", "laplace", 2.5, False),
    ("auditor", "count", "laplace", 0.1, False),             # auditors never query
    ("ghost", "count", "laplace", 0.1, False),               # unknown role
])
def test_rbac_matrix(role, agg, mech, eps, allowed):
    if allowed:
        assert authorize_query(POLICY, role, agg, mech, eps).name == role
    else:
        with pytest.raises(Forbidden):
            authorize_query(POLICY, role, agg, mech, eps)


def test_policy_loads_the_documented_budgets():
    r, c, a = POLICY.roles["researcher"], POLICY.roles["clinician"], POLICY.roles["auditor"]
    assert (r.total_epsilon, r.max_epsilon_per_query) == (10.0, 1.0)
    assert (c.total_epsilon, c.max_epsilon_per_query) == (20.0, 2.0)
    assert not a.can_query and a.audit_scope == "all" and a.verify_audit and r.audit_scope == "own"
    assert POLICY.k_threshold == 10


# ============================================================================ passwords and login throttle
def test_passwords_are_salted_scrypt_hashes():
    h1, h2 = hash_password("correct horse"), hash_password("correct horse")
    assert h1 != h2 and h1.startswith("scrypt$") and "correct horse" not in h1
    assert verify_password("correct horse", h1) and not verify_password("wrong", h1)
    assert not verify_password("x", "garbage") and not verify_password("x", "scrypt$1$2$3$4$5")


def test_unknown_user_costs_one_hash_verification_like_a_wrong_password(tmp_path, monkeypatch):
    store = UserStore(make_sqlite_engine(str(tmp_path / "u.db")))
    store.create("alice", "researcher", "pw-alice")
    calls = []
    real = jwt_auth.verify_password
    import fedhealth.coordinator.users as users_mod
    monkeypatch.setattr(users_mod, "verify_password", lambda p, h: (calls.append(1), real(p, h))[1])
    assert store.authenticate("alice", "wrong") is None and store.authenticate("nobody", "x") is None
    assert len(calls) == 2                                                  # no timing oracle for "does this user exist"
    assert store.authenticate("alice", "pw-alice").role == "researcher"


def test_login_throttle():
    now = [0.0]
    t = LoginThrottle(max_failures=3, window=60, clock=lambda: now[0])
    for _ in range(3):
        assert not t.blocked("alice")
        t.record_failure("alice")
    assert t.blocked("alice") and not t.blocked("bob")
    now[0] = 61
    assert not t.blocked("alice")                                           # failures age out
    for _ in range(3):
        t.record_failure("alice")
    t.reset("alice")
    assert not t.blocked("alice")


# ============================================================================ audit trail
KEY = b"a" * 32


@pytest.fixture()
def audit(tmp_path):
    engine = make_sqlite_engine(str(tmp_path / "audit.db"))
    log = AuditLog(engine, KEY)
    for i in range(6):
        log.append(actor="alice", role="researcher", action="query", outcome="released",
                   query={"query": {"agg": "count", "n": i}}, hospitals=["A", "B"], epsilon=0.5, reason=None)
    return log, tmp_path / "audit.db"


def _raw(path):
    con = sqlite3.connect(str(path))
    return con


def test_a_clean_chain_verifies_and_commits_to_its_history(audit):
    log, _ = audit
    res = log.verify_chain()
    assert res.ok and res.entries == 6 and res.head_seq == 6
    rows = log.read(limit=10)
    assert [r["seq"] for r in rows] == [6, 5, 4, 3, 2, 1] and rows[0]["epsilon_charged"] == 0.5
    assert log.head() == (6, res.head_mac) and log.verify_chain(expected_head=log.head()).ok


def test_the_database_refuses_updates_and_deletes(audit):
    log, path = audit
    con = _raw(path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            con.execute("UPDATE audit_log SET actor='mallory' WHERE seq=3")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            con.execute("DELETE FROM audit_log WHERE seq=3")
    finally:
        con.rollback()                          # release the implicit write transaction the failed statement opened
        con.close()
    assert log.verify_chain().ok and log.verify_chain().entries == 6


def _bypass_triggers(path):
    con = _raw(path)
    con.execute("DROP TRIGGER audit_no_update")
    con.execute("DROP TRIGGER audit_no_delete")
    con.commit()
    return con


def test_editing_an_entry_is_detected_even_with_the_triggers_removed(audit):
    log, path = audit
    con = _bypass_triggers(path)
    con.execute("UPDATE audit_log SET micro_eps = 0 WHERE seq = 4")              # 'I never spent that epsilon'
    con.commit()
    res = log.verify_chain()
    assert not res.ok and res.first_bad_seq == 4 and "mac mismatch" in res.reason


def test_deleting_an_entry_is_detected(audit):
    log, path = audit
    con = _bypass_triggers(path)
    con.execute("DELETE FROM audit_log WHERE seq = 3")
    con.commit()
    res = log.verify_chain()
    assert not res.ok and res.first_bad_seq == 4 and "gap" in res.reason


def test_reordering_entries_is_detected(audit):
    log, path = audit
    con = _bypass_triggers(path)
    con.execute("UPDATE audit_log SET seq = 99 WHERE seq = 2")
    con.execute("UPDATE audit_log SET seq = 2 WHERE seq = 3")
    con.execute("UPDATE audit_log SET seq = 3 WHERE seq = 99")
    con.commit()
    assert not log.verify_chain().ok


def test_forging_an_entry_without_the_key_is_detected(audit):
    """An attacker who can write the file but does not know the audit key cannot extend or rewrite the chain."""
    log, path = audit
    con = _bypass_triggers(path)
    last_mac = con.execute("SELECT mac FROM audit_log WHERE seq = 6").fetchone()[0]
    import hashlib, hmac
    forged_mac = hmac.new(b"guessed-key" * 3, last_mac.encode() + b"|forged", hashlib.sha256).hexdigest()
    con.execute("INSERT INTO audit_log VALUES (7, '2026-01-01T00:00:00.000+00:00', 'alice', 'researcher', 'query', NULL, NULL, 'A', 0, 0, 'released', NULL, ?, ?)",
                (last_mac, forged_mac))
    con.commit()
    res = log.verify_chain()
    assert not res.ok and res.first_bad_seq == 7


def test_truncating_the_tail_needs_an_external_anchor_to_be_noticed(audit):
    """Honest limitation, demonstrated: dropping the NEWEST entries leaves a valid shorter chain. A head
    anchor stored elsewhere catches it."""
    log, path = audit
    anchor = log.head()
    con = _bypass_triggers(path)
    con.execute("DELETE FROM audit_log WHERE seq > 4")
    con.commit()
    assert log.verify_chain().ok                                              # chain alone cannot tell
    res = log.verify_chain(expected_head=anchor)
    assert not res.ok and "truncated" in res.reason


def test_verification_with_the_wrong_key_fails(audit):
    log, _ = audit
    assert not AuditLog(log.engine, b"b" * 32).verify_chain().ok


def test_concurrent_appends_never_fork_the_chain(tmp_path):
    log = AuditLog(make_sqlite_engine(str(tmp_path / "c.db")), KEY)
    barrier = threading.Barrier(25)

    def worker(i):
        barrier.wait()
        log.append(actor=f"u{i}", role="researcher", action="query", outcome="released", epsilon=0.1)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    res = log.verify_chain()
    assert res.ok and res.entries == 25 and sorted(r["seq"] for r in log.read(limit=100)) == list(range(1, 26))


def test_audit_input_validation_and_key_length(tmp_path):
    engine = make_sqlite_engine(str(tmp_path / "v.db"))
    with pytest.raises(ValueError):
        AuditLog(engine, b"short")
    log = AuditLog(engine, KEY)
    with pytest.raises(ValueError):
        log.append(actor="a", role="r", action="query", outcome="exploded")
    assert log.head() == (0, GENESIS)
