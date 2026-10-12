"""Build the whole system IN-PROCESS, with every inter-service call recorded (test / attack / evaluation harness).

`build_system` wires 3 hospital apps + a coordinator. Coordinator->hospital calls AND hospital->hospital
(peer) calls go through `RecordingClient`s, so a test can later inspect *every message that crossed a
service boundary* (see leakscan.py).

secure=True builds the STRICT deployment: service identities (signed tokens) on every hop, callers verified by
the hospitals, and the coordinator's private API (JWT, RBAC, ledger, audit) mounted."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from fedhealth.coordinator.app import CoordinatorSecrets, create_coordinator_app
from fedhealth.hospital.app import create_hospital_app
from fedhealth.privacy.encryption.service_auth import ServiceAuth, ServiceIdentity
from fedhealth.privacy.encryption.tls import COORDINATOR, hospital_service_id


@dataclass
class Event:
    src: str            # "COORD" or hospital id
    dst: str
    method: str
    path: str
    request: str        # JSON text (or "")
    status: int
    response: str


@dataclass
class Recorder:
    events: list[Event] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, ev: Event) -> None:
        with self._lock:
            self.events.append(ev)

    def clear(self) -> None:
        with self._lock:
            self.events.clear()


class RecordingClient:
    """Duck-types the bits of httpx.Client the services use (get/post/delete)."""

    def __init__(self, inner: httpx.Client, src: str, dst: str, rec: Recorder, auth: httpx.Auth | None = None):
        self.inner, self.src, self.dst, self.rec, self.auth = inner, src, dst, rec, auth

    def _do(self, method: str, path: str, **kw) -> httpx.Response:
        if self.auth is not None:
            kw["auth"] = self.auth
        resp = getattr(self.inner, method)(path, **kw)
        self.rec.add(Event(self.src, self.dst, method.upper(), path,
                           json.dumps(kw["json"]) if "json" in kw else "", resp.status_code, resp.text))
        return resp

    def get(self, path, **kw):
        return self._do("get", path, **kw)

    def post(self, path, **kw):
        return self._do("post", path, **kw)

    def delete(self, path, **kw):
        return self._do("delete", path, **kw)


def build_system(data_dir: Path, psi_group: str = "modp2048", psi_groups: tuple[str, ...] | None = None,
                 allow_insecure: bool = True, *, secure: bool = False, tls_dir: Path | None = None,
                 secrets: CoordinatorSecrets | None = None, coordinator_db: Path | None = None,
                 policy=None) -> SimpleNamespace:
    kw = {"psi_groups": psi_groups} if psi_groups else {}
    if secure:
        kw["tls_dir"] = tls_dir
    apps = {h: create_hospital_app(h, db_path=Path(data_dir) / f"hospital_{h.lower()}.db",
                                   allow_insecure=allow_insecure, **kw) for h in "ABC"}
    raw = {h: TestClient(app) for h, app in apps.items()}
    rec = Recorder()
    ids = {}
    if secure:
        ids = {s: ServiceIdentity.load(tls_dir, s) for s in (COORDINATOR, *[hospital_service_id(h) for h in "ABC"])}

    def auth(src_service: str, dst_hospital: str):
        return ServiceAuth(ids[src_service], hospital_service_id(dst_hospital)) if secure else None

    for src in "ABC":                                           # hospital -> hospital (data plane)
        apps[src].state.peers = {dst: RecordingClient(raw[dst], src, dst, rec, auth(hospital_service_id(src), dst)) for dst in "ABC"}
    coord_clients = {h: RecordingClient(raw[h], "COORD", h, rec, auth(COORDINATOR, h)) for h in "ABC"}
    coord = create_coordinator_app(hospital_clients=coord_clients, allow_insecure=allow_insecure, psi_group=psi_group,
                                   secrets=secrets, db_path=coordinator_db, policy=policy)
    return SimpleNamespace(apps=apps, raw=raw, rec=rec, coordinator=coord, client=TestClient(coord),
                           orchestrator=coord.state.orchestrator, secure=coord.state.secure, ids=ids)
