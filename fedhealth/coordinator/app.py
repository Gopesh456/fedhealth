"""Coordinator service.

Two families of endpoints:
  * the PRIVATE API (`/auth/login`, `/v1/*`) - mounted when the coordinator has its secrets (JWT key, audit key);
    authenticated, role-checked, budgeted and audited (see secure_api.py);
  * the INSECURE baselines (`/insecure/*`) - mounted only with ALLOW_INSECURE_BASELINE=1, for tests and the evaluation.

Run:  uvicorn fedhealth.coordinator.app:create_app_from_env --factory --port 8000
      (FEDHEALTH_STRICT=1: TLS + hospital service identity + private API + keys required)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException

from fedhealth.common.config import load_topology
from fedhealth.common.http import build_hospital_clients
from fedhealth.common.schemas import PlainQueryResult, QuerySpec
from fedhealth.common.sqlite import make_sqlite_engine
from fedhealth.coordinator.orchestrator import Orchestrator, ProtocolError
from fedhealth.coordinator.release import ReleasePolicy
from fedhealth.coordinator.secure_api import SecureContext, mount_secure_api
from fedhealth.coordinator.service import QueryService
from fedhealth.coordinator.users import LoginThrottle, UserStore
from fedhealth.privacy.access_control.jwt_auth import TokenAuthority
from fedhealth.privacy.access_control.rbac import Policy, load_policy
from fedhealth.privacy.audit import AuditLog
from fedhealth.privacy.dp.ledger import EpsilonLedger
from fedhealth.privacy.encryption.aesgcm import load_key, secrets_dir
from fedhealth.privacy.encryption.service_auth import ServiceIdentity
from fedhealth.privacy.encryption.tls import COORDINATOR, TlsPaths
from fedhealth.privacy.encryption.tls import tls_dir as default_tls_dir


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class CoordinatorSecrets:
    jwt_secret: bytes
    audit_key: bytes


def load_coordinator_secrets() -> CoordinatorSecrets:
    """Loads (never creates) the coordinator's keys - the service refuses to start without them."""
    return CoordinatorSecrets(load_key("jwt_secret"), load_key("audit_key"))


def create_coordinator_app(hospital_clients: dict[str, httpx.Client] | None = None,
                           allow_insecure: bool | None = None,
                           psi_group: str | None = None, *,
                           secrets: CoordinatorSecrets | None = None,
                           db_path: str | Path | None = None,
                           policy: Policy | None = None,
                           tls_dir: str | Path | None = None,
                           users_file: str | Path | None = None) -> FastAPI:
    topo = load_topology()
    tls_paths = identity = None
    if tls_dir is not None:                                   # strict mode: mTLS + signed service tokens towards hospitals
        tls_paths, identity = TlsPaths(Path(tls_dir), COORDINATOR), ServiceIdentity.load(Path(tls_dir), COORDINATOR)
    clients = hospital_clients if hospital_clients is not None else build_hospital_clients(topo, tls=tls_paths, identity=identity)
    if allow_insecure is None:
        allow_insecure = _env_flag("ALLOW_INSECURE_BASELINE")
    orch = Orchestrator(clients, psi_group or os.environ.get("FEDHEALTH_PSI_GROUP", "modp2048"))

    app = FastAPI(title="Federated query coordinator", docs_url=None, redoc_url=None)
    app.state.clients = clients
    app.state.orchestrator = orch
    app.state.allow_insecure = allow_insecure
    app.state.secure = None

    @app.get("/health")
    def health() -> dict:
        return {"service": "coordinator", "status": "ok", "insecure_baseline": allow_insecure,
                "private_api": app.state.secure is not None}

    if secrets is not None:
        path = Path(db_path) if db_path else topo.coordinator_db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = make_sqlite_engine(str(path))
        pol = policy or load_policy()
        ledger = EpsilonLedger(engine)
        users = UserStore(engine)
        if users_file is not None:
            users.seed_from_file(Path(users_file))
        ctx = SecureContext(
            users=users, tokens=TokenAuthority(secrets.jwt_secret, pol.jwt_ttl_seconds), policy=pol, ledger=ledger,
            audit=AuditLog(engine, secrets.audit_key),
            service=QueryService(orch, ledger, ReleasePolicy(k=pol.k_threshold, tolerance=pol.noise_tolerance)),
            throttle=LoginThrottle())
        mount_secure_api(app, ctx)
        app.state.secure = ctx

    if allow_insecure:
        @app.post("/insecure/query", response_model=PlainQueryResult, tags=["INSECURE baseline"])
        def insecure_query(spec: QuerySpec) -> PlainQueryResult:
            try:
                return orch.plain_query(spec)
            except (httpx.HTTPError, ProtocolError) as exc:
                raise HTTPException(502, f"hospital error: {exc}") from exc

        @app.post("/insecure/psi-query", response_model=PlainQueryResult, tags=["INSECURE baseline"])
        def insecure_psi_query(spec: QuerySpec, force_link: bool = False) -> PlainQueryResult:
            try:
                return orch.psi_plain_query(spec, force_link=force_link)
            except (httpx.HTTPError, ProtocolError) as exc:
                raise HTTPException(502, f"hospital error: {exc}") from exc

        @app.post("/insecure/mpc-query", response_model=PlainQueryResult, tags=["exact, no DP"])
        def mpc_query(spec: QuerySpec, force_link: bool = False) -> PlainQueryResult:
            """PSI + secret sharing, but the EXACT answer is released (no DP / k / auth): evaluation only."""
            try:
                return orch.mpc_query(spec, force_link=force_link)
            except (httpx.HTTPError, ProtocolError) as exc:
                raise HTTPException(502, f"hospital error: {exc}") from exc

    return app


def create_app_from_env() -> FastAPI:
    """uvicorn --factory entry point.
    FEDHEALTH_STRICT=1  -> TLS identity from FEDHEALTH_TLS_DIR, private API with keys from FEDHEALTH_SECRETS_DIR,
                           demo users from <secrets>/users.json."""
    if _env_flag("FEDHEALTH_STRICT"):
        return create_coordinator_app(secrets=load_coordinator_secrets(), tls_dir=default_tls_dir(),
                                      users_file=secrets_dir() / "users.json")
    return create_coordinator_app()
