"""Caller authorisation for hospital endpoints.

  control plane   (init, step, prepare, output, ...)   only the COORDINATOR may call
  data plane      (inbox, tags, inshare, inopen)       only the hospital named in the message's `sender` field

When no verifier is configured (plain local development, in-process tests of the earlier phases) nothing is
enforced. Strict mode - the TLS deployment - always configures one.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from fedhealth.privacy.encryption.service_auth import ServiceAuthError
from fedhealth.privacy.encryption.tls import COORDINATOR, hospital_service_id


def authenticated_caller(request: Request) -> str | None:
    verifier = getattr(request.app.state, "service_verifier", None)
    if verifier is None:
        return None
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(401, "service credentials required")
    try:
        return verifier.verify(header[7:])
    except ServiceAuthError as exc:
        raise HTTPException(401, "invalid service credentials") from exc


def control_only(request: Request) -> None:
    """FastAPI dependency: the caller must be the coordinator."""
    caller = authenticated_caller(request)
    if request.app.state.service_verifier is not None and caller != COORDINATOR:
        raise HTTPException(403, "control-plane endpoint: coordinator only")


def require_peer(request: Request, claimed_sender: str) -> None:
    """Data-plane check: the authenticated caller must BE the hospital it says it is (and not this hospital)."""
    caller = authenticated_caller(request)
    if request.app.state.service_verifier is not None and caller != hospital_service_id(claimed_sender):
        raise HTTPException(403, "sender does not match the authenticated caller")
