"""Shared fixtures. All four services run IN-PROCESS: a FastAPI TestClient is an httpx.Client, so the
coordinator talks to the hospitals through exactly the same client interface it uses over real HTTP."""
from __future__ import annotations

import os
import tempfile

# Tests must never touch the repo's real ./secrets directory: all keys live in a throw-away folder. This runs at
# import time, before any fixture or subprocess, so the generator, the services and spawned uvicorn processes
# (which inherit os.environ) all agree on it.
os.environ.setdefault("FEDHEALTH_SECRETS_DIR", tempfile.mkdtemp(prefix="fedhealth-test-secrets-"))
# The test worlds have at most ~800 patients per hospital: pad PSI lists and the linked index to a public 1,024
# instead of the deployment's 4,608 (the full-size tests raise it again).
os.environ.setdefault("FEDHEALTH_LINK_PAD_TO", "1024")
os.environ.setdefault("FEDHEALTH_DEADLINE_SCALE", "0")       # no response deadlines (tests/test_deadlines.py sets its own)
TEST_PAD_TO = int(os.environ["FEDHEALTH_LINK_PAD_TO"])

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fedhealth.coordinator.app import create_coordinator_app  # noqa: E402
from fedhealth.hospital.app import create_hospital_app  # noqa: E402
from fedhealth.synth.generate import generate  # noqa: E402
from fedhealth.synth.ground_truth import GroundTruth  # noqa: E402


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """The full-size synthetic world (5,000 people, seed 42), ENCRYPTED at rest, in a temp dir (~7 s, built once)."""
    out = tmp_path_factory.mktemp("world")
    summary = generate(5000, seed=42, out_dir=out)
    return {"dir": out, "summary": summary, "gt_dir": out / "ground_truth"}


@pytest.fixture(scope="session")
def plain_world(tmp_path_factory):
    """The SAME world (same seed) written in PLAINTEXT. Lets structural tests read the databases with raw SQL;
    tests/test_encryption.py proves it is logically identical to the encrypted one."""
    out = tmp_path_factory.mktemp("plain_world")
    summary = generate(5000, seed=42, out_dir=out, encrypt=False)
    return {"dir": out, "summary": summary, "gt_dir": out / "ground_truth"}


@pytest.fixture(scope="session")
def pki(tmp_path_factory):
    """A throw-away CA + one certificate per service (ECDSA P-256, fast)."""
    from fedhealth.privacy.encryption import tls
    d = tmp_path_factory.mktemp("pki")
    tls.generate_all(d)
    return d


@pytest.fixture(scope="session")
def other_pki(tmp_path_factory):
    """A second, unrelated CA: certificates from here must never be trusted."""
    from fedhealth.privacy.encryption import tls
    d = tmp_path_factory.mktemp("other_pki")
    tls.generate_all(d)
    return d


@pytest.fixture(scope="session")
def ground_truth(world) -> GroundTruth:
    return GroundTruth(world["gt_dir"])


@pytest.fixture(scope="session")
def hospital_apps(world):
    return {h: create_hospital_app(h, db_path=world["dir"] / f"hospital_{h.lower()}.db", allow_insecure=True)
            for h in "ABC"}


@pytest.fixture(scope="session")
def hospital_clients(hospital_apps):
    return {h: TestClient(app) for h, app in hospital_apps.items()}


@pytest.fixture(scope="session")
def coordinator_app(hospital_clients):
    return create_coordinator_app(hospital_clients=hospital_clients, allow_insecure=True)


@pytest.fixture(scope="session")
def coordinator_client(coordinator_app):
    return TestClient(coordinator_app)
