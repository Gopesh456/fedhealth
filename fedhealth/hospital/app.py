"""Hospital node: an independent FastAPI service with its own SQLite database.

Design rules (see plan):
  * A hospital exposes NO endpoint that returns patient rows. Only protocol endpoints exist
    (PSI / MPC in later phases) plus, if explicitly enabled, the insecure baseline.
  * The service opens its database read-only (PRAGMA query_only) - it only ever answers questions.

Run one node:   HOSPITAL_ID=A uvicorn fedhealth.hospital.app:create_app_from_env --factory --port 8001
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from fastapi import Depends, FastAPI
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

import httpx

from fedhealth.common.config import load_topology
from fedhealth.common.http import build_hospital_clients
from fedhealth.hospital import routes_fl, routes_insecure, routes_mpc, routes_psi
from fedhealth.hospital.deadlines import QuietGate, ResponseDeadlines, load_response_deadlines, quiet_reader
from fedhealth.privacy.encryption.aesgcm import NullVault, Vault, load_key
from fedhealth.privacy.encryption.column_types import attach_vault
from fedhealth.privacy.encryption.service_auth import ServiceIdentity, ServiceVerifier
from fedhealth.privacy.encryption.tls import TlsPaths, hospital_service_id
from fedhealth.privacy.encryption.tls import tls_dir as default_tls_dir

DEFAULT_PSI_GROUPS = ("modp2048", "modp1536")     # what this hospital is willing to run PSI with


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def create_hospital_app(hospital_id: str, db_path: str | Path | None = None,
                        allow_insecure: bool | None = None,
                        peers: dict[str, httpx.Client] | None = None,
                        psi_groups: tuple[str, ...] = DEFAULT_PSI_GROUPS,
                        vault: Vault | NullVault | None = None,
                        tls_dir: str | Path | None = None,
                        deadlines: ResponseDeadlines | None = None) -> FastAPI:
    """`peers` maps hospital id -> client used for hospital-to-hospital (data plane) calls. If omitted,
    clients are built lazily from the topology. Tests create the apps first and assign
    `app.state.peers = {...}` afterwards (the clients wrap the apps, hence the lazy hook).

    `vault` holds this hospital's AES-GCM data key. If omitted it is loaded from the secrets directory; if the key
    file is missing the service REFUSES TO START (fail closed) - unless ALLOW_PLAINTEXT_DB=1 is set explicitly.

    `tls_dir` switches on STRICT mode: the hospital loads its certificate identity from that directory, authenticates
    every caller (control plane: coordinator only; data plane: the named sender) and builds mTLS peer clients.

    `deadlines` are the public response deadlines (hospital/deadlines.py); by default read from configs/policy.yaml."""
    topo = load_topology()
    if hospital_id not in topo.hospitals:
        raise ValueError(f"unknown hospital {hospital_id!r}")
    db_path = Path(db_path) if db_path else topo.hospital_db_path(hospital_id)
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found - run `python -m fedhealth.synth.generate` first")
    if allow_insecure is None:
        allow_insecure = _env_flag("ALLOW_INSECURE_BASELINE")

    if vault is None:
        vault = NullVault() if _env_flag("ALLOW_PLAINTEXT_DB") else Vault(load_key(f"hospital_{hospital_id.lower()}"))

    # check_same_thread=False: FastAPI runs sync endpoints in a thread pool.
    engine = attach_vault(create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}), vault)

    @event.listens_for(engine, "connect")
    def _read_only(dbapi_conn, _record):  # noqa: ANN001
        dbapi_conn.execute("PRAGMA query_only = ON")

    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def get_session():
        with session_factory() as session:
            yield session

    # No docs/openapi routes: the smaller the surface of a data holder, the better.
    # quiet_reader: no request runs while a deadline-guarded call might be doing data-dependent work (deadlines.py)
    app = FastAPI(title=f"Hospital {hospital_id} node", docs_url=None, redoc_url=None, openapi_url=None,
                  dependencies=[Depends(quiet_reader)])
    app.state.hospital_id = hospital_id
    app.state.engine = engine
    app.state.allow_insecure = allow_insecure
    app.state.session_factory = session_factory
    app.state.peers = peers
    app.state.psi = routes_psi.PsiState()          # in-memory linkage sessions + finished linkages
    app.state.mpc = routes_mpc.MpcState()          # in-memory secret-shared computations
    app.state.fl = routes_fl.FlState()             # in-memory federated-learning rounds
    app.state.deadlines = deadlines if deadlines is not None else load_response_deadlines()
    app.state.deadline_overruns = Counter()        # local monitoring only: never returned to anyone
    app.state.quiet_gate = QuietGate()

    identity = verifier = tls_paths = None         # strict mode: who am I, and who may call me?
    if tls_dir is not None:
        service_id = hospital_service_id(hospital_id)
        identity = ServiceIdentity.load(Path(tls_dir), service_id)
        verifier = ServiceVerifier(service_id, Path(tls_dir))
        tls_paths = TlsPaths(Path(tls_dir), service_id)
    app.state.service_verifier = verifier
    app.state.service_identity = identity

    def get_peer(peer_id: str) -> httpx.Client:
        if app.state.peers is None:                # lazily build from the topology on first use
            app.state.peers = build_hospital_clients(topo, tls=tls_paths, identity=identity)
        return app.state.peers[peer_id]
    app.state.get_peer = get_peer

    @app.get("/health")
    def health() -> dict:
        return {"hospital": hospital_id, "status": "ok", "insecure_baseline": allow_insecure}

    app.include_router(routes_psi.make_router(hospital_id, get_session, psi_groups))
    app.include_router(routes_mpc.make_router(hospital_id, get_session))
    app.include_router(routes_fl.make_router(hospital_id, get_session))
    if allow_insecure:
        app.include_router(routes_insecure.make_router(hospital_id, get_session))
    return app


def create_app_from_env() -> FastAPI:
    """uvicorn --factory entry point (HOSPITAL_ID, optional HOSPITAL_DB, ALLOW_INSECURE_BASELINE).
    FEDHEALTH_STRICT=1 enables strict mode using the certificates in FEDHEALTH_TLS_DIR."""
    tls = default_tls_dir() if _env_flag("FEDHEALTH_STRICT") else None
    return create_hospital_app(os.environ["HOSPITAL_ID"], os.environ.get("HOSPITAL_DB"), tls_dir=tls)
