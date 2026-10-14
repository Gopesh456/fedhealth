"""The STRICT deployment over real sockets: 4 uvicorn processes with TLS, hospitals demanding client certificates,
service tokens between services, the private API in front. Slow (process start-up + real 2048-bit PSI)."""
from __future__ import annotations

import re
import ssl
import sys
from pathlib import Path

import httpx
import pytest

from fedhealth.common.config import load_topology
from fedhealth.common.http import client_ssl_context
from fedhealth.coordinator.bootstrap import bootstrap_secrets
from fedhealth.privacy.encryption.aesgcm import secrets_dir
from fedhealth.privacy.encryption.tls import TlsPaths
from fedhealth.synth.generate import generate
from fedhealth.synth.ground_truth import GroundTruth

from .helpers import headline_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_local  # noqa: E402

pytestmark = pytest.mark.slow


def _password(user: str) -> str:
    text = (secrets_dir() / "DEMO_CREDENTIALS.txt").read_text()
    return re.search(rf"^{user}\s+role=\w+\s+password=(\S+)", text, re.M).group(1)


@pytest.fixture(scope="module")
def strict_system(tmp_path_factory, pki, other_pki):
    mp = pytest.MonkeyPatch()
    data = tmp_path_factory.mktemp("tls_data")
    generate(800, seed=11, out_dir=data)
    bootstrap_secrets()
    mp.setenv("FEDHEALTH_DATA_DIR", str(data))
    mp.setenv("FEDHEALTH_PORT_OFFSET", "600")
    mp.setenv("FEDHEALTH_TLS_DIR", str(pki))
    procs = run_local.start_services(secure=True, quiet=True)
    try:
        yield load_topology(), GroundTruth(data / "ground_truth"), pki, other_pki
    finally:
        run_local.stop_services(procs)
        mp.undo()


def test_hospital_rejects_everything_that_is_not_mutual_tls(strict_system):
    topo, _, pki, other_pki = strict_system
    port = topo.hospitals["A"].port
    ca = pki / "ca.pem"
    with pytest.raises(httpx.HTTPError):                                              # plaintext on a TLS port
        httpx.get(f"http://127.0.0.1:{port}/health", timeout=5)
    no_cert = ssl.create_default_context(cafile=str(ca))
    with pytest.raises(httpx.HTTPError):                                              # no client certificate
        httpx.get(f"https://127.0.0.1:{port}/health", verify=no_cert, timeout=5)
    foreign = ssl.create_default_context(cafile=str(ca))
    foreign.load_cert_chain(str(other_pki / "coordinator.crt"), str(other_pki / "coordinator.key"))
    with pytest.raises(httpx.HTTPError):                                              # certificate from a different CA
        httpx.get(f"https://127.0.0.1:{port}/health", verify=foreign, timeout=5)
    # a client that trusts the wrong CA also refuses the hospital (server authentication works both ways)
    wrong_trust = ssl.create_default_context(cafile=str(other_pki / "ca.pem"))
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"https://127.0.0.1:{port}/health", verify=wrong_trust, timeout=5)


def test_a_valid_certificate_alone_is_not_enough_to_issue_commands(strict_system):
    topo, _, pki, _ = strict_system
    port = topo.hospitals["A"].port
    ctx = client_ssl_context(TlsPaths(pki, "hospital-b"))                             # a legitimate peer's certificate
    assert httpx.get(f"https://127.0.0.1:{port}/health", verify=ctx, timeout=5).status_code == 200
    r = httpx.post(f"https://127.0.0.1:{port}/psi/{'aa' * 8}/init", json={"ring": ["A", "B"]}, verify=ctx, timeout=5)
    assert r.status_code == 401                                                       # ...but no service token
    assert httpx.post(f"https://127.0.0.1:{port}/insecure/match", json={}, verify=ctx, timeout=5).status_code == 404


def test_full_private_query_over_mutual_tls(strict_system):
    topo, gt, pki, _ = strict_system
    base = f"https://127.0.0.1:{topo.coordinator_port}"
    requester = ssl.create_default_context(cafile=str(pki / "ca.pem"))               # requesters need no client cert
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"http://127.0.0.1:{topo.coordinator_port}/health", timeout=5)     # plaintext refused
    with httpx.Client(base_url=base, verify=requester, timeout=300) as c:
        assert c.get("/health").json()["private_api"] is True
        tok = c.post("/auth/login", json={"username": "clinician1", "password": _password("clinician1")}).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}
        spec = headline_spec(a={}, b={})                                              # far above k: never suppressed
        r = c.post("/v1/query", headers=h, json={"query": spec.model_dump(mode="json", exclude_none=True), "privacy": {"epsilon": 2.0}})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "released" and abs(body["result"] - gt.evaluate(spec)) <= 12
        assert body["epsilon_remaining"] == pytest.approx(18.0)
        assert c.post("/insecure/query", json={}).status_code == 404                  # no insecure surface in strict mode
        aud = c.post("/auth/login", json={"username": "auditor1", "password": _password("auditor1")}).json()["access_token"]
        v = c.get("/v1/audit/verify", headers={"Authorization": f"Bearer {aud}"}).json()
        assert v["ok"] and v["entries"] >= 3
        entries = c.get("/v1/audit", headers={"Authorization": f"Bearer {aud}"}).json()["entries"]
        assert any(e["actor"] == "clinician1" and e["action"] == "query" and e["outcome"] == "released" for e in entries)
