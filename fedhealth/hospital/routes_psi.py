"""Hospital-side endpoints of the PSI (private linkage) protocol.

Control plane (coordinator -> hospital):  init, step, publish, finalize, delete
Data plane    (hospital -> hospital):     inbox, tags      <- the coordinator never calls these

A hospital never calls a peer from inside a peer-facing handler: `inbox` and `tags` only STORE what they
receive. All outbound calls happen in coordinator-triggered handlers (`step`, `publish`), so the call
graph has depth 1 and cannot deadlock.

After `finalize` only the mapping  national_id -> linked index  survives (in this process's memory).
It never leaves the hospital: the hospital uses it later to turn its local predicate matches into
indicator vectors over the common index.
"""
from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from fedhealth.common.wire import LINKAGE_ID_PATTERN, PsiFinalizeResult, PsiInbox, PsiInit, PsiStep, PsiTags
from fedhealth.hospital import models as m
from fedhealth.hospital import security
from fedhealth.privacy.linkage.group import get_group, pack_ints, unpack_ints
from fedhealth.privacy.linkage.psi import PsiParty

TAG_BYTES = 16
PATIENT_MODELS = {"A": m.APatient, "B": m.BPatient, "C": m.CPatient}
LinkageId = Annotated[str, Path(pattern=LINKAGE_ID_PATTERN)]      # a fresh constraint per use (no shared mutable default)


@dataclass
class PsiSession:
    party: PsiParty
    inbox: dict[int, list[int]] = field(default_factory=dict)      # round -> list received from prev party
    peer_tags: dict[str, set[bytes]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Linkage:
    ring: list[str]
    mapping: dict[str, int]       # national_id -> index in the common linked index (stays local)
    m: int


@dataclass
class PsiState:
    sessions: dict[str, PsiSession] = field(default_factory=dict)
    linkages: dict[str, Linkage] = field(default_factory=dict)


def _b64(blob: bytes) -> str:
    return base64.b64encode(blob).decode("ascii")


def _session(request: Request, lid: str) -> PsiSession:
    s = request.app.state.psi.sessions.get(lid)
    if s is None:
        raise HTTPException(404, "unknown linkage session")
    return s


def _post_peer(request: Request, peer: str, path: str, body: dict) -> None:
    try:
        resp = request.app.state.get_peer(peer).post(path, json=body)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"peer {peer} unreachable or refused the message") from exc


def make_router(hospital_id: str, get_session, allowed_groups: tuple[str, ...]) -> APIRouter:
    control = APIRouter(tags=["PSI linkage"], dependencies=[Depends(security.control_only)])   # coordinator only
    data = APIRouter(tags=["PSI linkage"])                                                      # hospital -> hospital
    patient_model = PATIENT_MODELS[hospital_id]

    @control.post("/psi/{lid}/init")
    def init(body: PsiInit, request: Request, lid: LinkageId, session: Session = Depends(get_session)) -> dict:
        state: PsiState = request.app.state.psi
        if hospital_id not in body.ring:
            raise HTTPException(400, "this hospital is not part of the requested ring")
        if body.group not in allowed_groups:        # a hospital, not the coordinator, decides which crypto it accepts
            raise HTTPException(400, f"group {body.group!r} not accepted by this hospital")
        if lid in state.sessions or lid in state.linkages:
            raise HTTPException(409, "linkage id already used")
        ids = [r[0] for r in session.execute(select(patient_model.national_id).order_by(patient_model.patient_id))]
        state.sessions[lid] = PsiSession(PsiParty(hospital_id, list(body.ring), ids, get_group(body.group)))
        return {"status": "ready"}              # deliberately NOT returning |set| to the coordinator

    @control.post("/psi/{lid}/step")
    def step(body: PsiStep, request: Request, lid: LinkageId) -> dict:
        s = _session(request, lid)
        party, n = s.party, len(s.party.ring)
        if body.round > n:
            raise HTTPException(400, "round out of range")
        with s.lock:
            if body.round == 1:
                out = party.round_one_message()                       # encrypt OWN list
            else:
                received = s.inbox.pop(body.round - 1, None)
                if received is None:
                    raise HTTPException(409, "previous round's message has not arrived")
                out = party.encrypt_list(received)                    # encrypt the list we were handed
        blob = _b64(pack_ints(out, party.group.nbytes))
        _post_peer(request, party.next_party, f"/psi/{lid}/inbox",
                   PsiInbox(sender=hospital_id, round=body.round, blob=blob).model_dump())
        return {"status": "sent"}

    @data.post("/psi/{lid}/inbox")                                    # data plane, called by the previous party
    def inbox(body: PsiInbox, request: Request, lid: LinkageId) -> dict:
        security.require_peer(request, body.sender)                    # the caller must BE the sender it claims
        s = _session(request, lid)
        party, n = s.party, len(s.party.ring)
        if body.sender != party.prev_party:
            raise HTTPException(403, "messages are only accepted from the previous party in the ring")
        elements = unpack_ints(base64.b64decode(body.blob), party.group.nbytes)
        with s.lock:
            if body.round == n:                                        # our own list, now encrypted by everyone
                if len(elements) != len(party.ids):
                    raise HTTPException(400, "returned list has the wrong length")
                party.final_elements = elements
            else:
                s.inbox[body.round] = elements
        return {"status": "stored"}

    @control.post("/psi/{lid}/publish")
    def publish(request: Request, lid: LinkageId) -> dict:
        s = _session(request, lid)
        if s.party.final_elements is None:
            raise HTTPException(409, "own list has not completed the ring")
        blob = _b64(b"".join(s.party.publish()))
        for other in s.party.ring:
            if other != hospital_id:
                _post_peer(request, other, f"/psi/{lid}/tags", PsiTags(sender=hospital_id, blob=blob).model_dump())
        return {"status": "published"}

    @data.post("/psi/{lid}/tags")                                     # data plane, called by every other party
    def tags(body: PsiTags, request: Request, lid: LinkageId) -> dict:
        security.require_peer(request, body.sender)
        s = _session(request, lid)
        if body.sender == hospital_id or body.sender not in s.party.ring:
            raise HTTPException(403, "unexpected sender")
        raw = base64.b64decode(body.blob)
        if len(raw) % TAG_BYTES:
            raise HTTPException(400, "malformed tag blob")
        with s.lock:
            s.peer_tags[body.sender] = {raw[i:i + TAG_BYTES] for i in range(0, len(raw), TAG_BYTES)}
        return {"status": "stored"}

    @control.post("/psi/{lid}/finalize", response_model=PsiFinalizeResult)
    def finalize(request: Request, lid: LinkageId) -> PsiFinalizeResult:
        state: PsiState = request.app.state.psi
        s = _session(request, lid)
        party = s.party
        missing = [o for o in party.ring if o != hospital_id and o not in s.peer_tags]
        if missing or party.tags is None:
            raise HTTPException(409, f"waiting for tags from {missing}" if missing else "publish first")
        with s.lock:
            size = party.finalize(s.peer_tags)
            state.linkages[lid] = Linkage(list(party.ring), party.mapping, size)
            state.sessions.pop(lid, None)                              # key + intermediate lists are gone
        return PsiFinalizeResult(m=size, digest=party.linked_digest.hex())

    @control.delete("/psi/{lid}")
    def delete(request: Request, lid: LinkageId) -> dict:
        state: PsiState = request.app.state.psi
        state.sessions.pop(lid, None)
        state.linkages.pop(lid, None)
        return {"status": "deleted"}

    router = APIRouter()
    router.include_router(control)
    router.include_router(data)
    return router
