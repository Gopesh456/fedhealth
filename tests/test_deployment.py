"""Static checks of the Docker deployment files.

Docker is NOT installed on the machine this project was built on, so docker-compose.yml was never executed here.
These tests parse it and assert the security properties it is designed to have; the same topology is exercised for
real by tests/test_e2e_tls.py (python scripts/run_local.py --secure).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def mounts(svc: dict) -> list[str]:
    return [v if isinstance(v, str) else v.get("source", "") for v in svc.get("volumes", [])]


def host_paths(svc: dict) -> set[str]:
    return {m.split(":")[0] for m in mounts(svc)}


def test_all_services_are_defined(compose):
    assert {"hospital-a", "hospital-b", "hospital-c", "coordinator", "dashboard", "setup"} <= set(compose["services"])
    assert all(isinstance(v, str) for s in compose["services"].values() for v in s.get("volumes", [])), "volume lists must be flat"


@pytest.mark.parametrize("h", "abc")
def test_a_hospital_mounts_only_its_own_secrets_and_database(compose, h):
    svc = compose["services"][f"hospital-{h}"]
    paths = host_paths(svc)
    assert f"./data/hospital_{h}.db" in paths
    keys = {p for p in paths if p.endswith(".key")}
    assert keys == {f"./secrets/hospital_{h}.key", f"./secrets/tls/hospital-{h}.key"}, keys      # own DEK + own TLS key, nothing else
    assert not any(p.endswith("hospital_%s.db" % o) for p in paths for o in "abc" if o != h)
    assert not any(p.endswith(("jwt_secret.key", "audit_key.key", "users.json")) for p in paths)
    assert "./secrets/tls/ca.pem" in paths and all(".key" not in p for p in paths if p.startswith("./secrets/tls/") and f"hospital-{h}" not in p)


def test_the_coordinator_has_no_access_to_hospital_data_or_keys(compose):
    svc = compose["services"]["coordinator"]
    paths = host_paths(svc)
    assert not any("hospital_" in p and p.endswith((".db", ".key")) for p in paths), "coordinator must not mount any hospital DB or data key"
    assert not any(re.search(r"hospital-[abc]\.key", p) for p in paths)
    assert "coordinator-data" in paths                                            # its OWN database lives in a named volume
    assert {"./secrets/jwt_secret.key", "./secrets/audit_key.key", "./secrets/tls/coordinator.key"} <= paths
    assert not any(p.rstrip("/") in ("./data", "./secrets", "./") for p in paths), "never mount whole directories"


def test_the_dashboard_only_gets_the_ca_certificate(compose):
    assert host_paths(compose["services"]["dashboard"]) == {"./secrets/tls/ca.pem"}


def test_only_the_coordinator_and_dashboard_are_reachable_from_the_host(compose):
    for name, svc in compose["services"].items():
        if name in ("coordinator", "dashboard"):
            assert svc.get("ports"), name
        else:
            assert not svc.get("ports"), f"{name} must not publish ports"
    assert all("expose" in compose["services"][f"hospital-{h}"] for h in "abc")


def test_every_runtime_service_is_strict_and_has_no_insecure_routes(compose):
    for name in ("hospital-a", "hospital-b", "hospital-c", "coordinator"):
        env = compose["services"][name]["environment"]
        assert str(env["FEDHEALTH_STRICT"]) == "1" and env["FEDHEALTH_SCHEME"] == "https"
        assert "ALLOW_INSECURE_BASELINE" not in env and "ALLOW_PLAINTEXT_DB" not in env
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert not re.search(r"ALLOW_INSECURE_BASELINE\s*:\s*['\"]?(1|true)", text) and "ALLOW_PLAINTEXT_DB" not in text


def test_hospitals_demand_client_certificates_and_the_coordinator_does_not(compose):
    for h in "abc":
        cmd = compose["services"][f"hospital-{h}"]["command"]
        assert "--ssl-cert-reqs 2" in cmd and f"/certs/hospital-{h}.key" in cmd and "--ssl-ca-certs /certs/ca.pem" in cmd
    assert "--ssl-cert-reqs 0" in compose["services"]["coordinator"]["command"]      # requesters use JWTs, not client certificates


def test_the_demo_ground_truth_is_not_mounted_anywhere_by_default(compose):
    assert not any("ground_truth" in p for s in compose["services"].values() for p in host_paths(s))


def test_dockerfile_runs_as_non_root_and_copies_no_secrets_or_data():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^USER app", text, re.M) and "useradd" in text
    copies = " ".join(re.findall(r"^COPY (.+)$", text, re.M))
    assert "secrets" not in copies and "data" not in copies.replace("dashboard", "")
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "secrets/" in ignore and "data/" in ignore and ".env" in ignore


def test_gitignore_keeps_secrets_and_data_out_of_the_repository():
    ig = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^secrets/", ig, re.M) and re.search(r"^data/", ig, re.M) and re.search(r"^\.env", ig, re.M)
