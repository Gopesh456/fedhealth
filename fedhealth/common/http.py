"""Small helpers for service-to-service calls.

Hospitals and the coordinator talk through plain `httpx.Client` objects. Production code builds them from the
topology (base URL per hospital). In STRICT mode (a TLS directory is configured) every client also
  * verifies the server against our private CA (and the hospital's certificate names must match),
  * presents its own client certificate (mutual TLS),
  * stamps each request with a short-lived service token for the destination (see service_auth.py).
Tests pass FastAPI `TestClient`s instead (a TestClient *is* an httpx.Client), so all services run in-process.
"""
from __future__ import annotations

import ssl
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

import httpx

from fedhealth.common.config import Topology
from fedhealth.privacy.encryption.service_auth import ServiceAuth, ServiceIdentity
from fedhealth.privacy.encryption.tls import TlsPaths, hospital_service_id

T = TypeVar("T")


def client_ssl_context(tls: TlsPaths, present_client_cert: bool = True) -> ssl.SSLContext:
    """TLS 1.2+ client context that trusts ONLY our CA (not the system store)."""
    ctx = ssl.create_default_context(cafile=str(tls.ca))
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    if present_client_cert:
        ctx.load_cert_chain(str(tls.cert), str(tls.key))
    return ctx


def build_hospital_clients(topo: Topology, timeout: float = 300.0, tls: TlsPaths | None = None,
                           identity: ServiceIdentity | None = None) -> dict[str, httpx.Client]:
    clients = {}
    for h, entry in topo.hospitals.items():
        kwargs: dict = {"timeout": timeout, "base_url": entry.url("https" if tls else topo.scheme)}
        if tls is not None:
            kwargs["verify"] = client_ssl_context(tls)
        if identity is not None:
            kwargs["auth"] = ServiceAuth(identity, hospital_service_id(h))
        clients[h] = httpx.Client(**kwargs)
    return clients


def parallel(calls: dict[str, Callable[[], T]]) -> dict[str, T]:
    """Run one zero-argument callable per key concurrently (hospitals work in parallel) and return
    {key: result}. The first exception is re-raised."""
    if len(calls) <= 1:
        return {k: fn() for k, fn in calls.items()}
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = {k: pool.submit(fn) for k, fn in calls.items()}
        return {k: f.result() for k, f in futures.items()}
