"""Hospital-side endpoints of the secret-shared aggregation protocol (see fedhealth/privacy/smpc/protocol.py).

Control plane (coordinator -> hospital):  prepare, share, triples, open, finish, output, delete
Data plane    (hospital -> hospital):     inshare, inopen          <- the coordinator never calls these

What leaves this hospital, and where:
    prepare   nothing. The hospital evaluates its predicate against its OWN database and builds 0/1
              indicator vectors over the linked patient index. They stay in memory.
    share     additive SHARES of those vectors go to the peers (uniformly random numbers).
    open      shares of d = a - x and e = b - y go to the peers (uniformly random numbers).
    output    one additive share per output cell goes to the coordinator.
No endpoint returns a patient, an ID or an unshared vector.
"""
from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy.orm import Session

from fedhealth.common.schemas import categories
from fedhealth.common.wire import (LINKAGE_ID_PATTERN, MpcInOpen, MpcInShare, MpcLayer, MpcOutput, MpcPrepare,
                                   MpcTriples)
from fedhealth.hospital import security
from fedhealth.hospital.predicate_runner import run_local_query
from fedhealth.privacy.smpc.field import decode_vector, encode_vector
from fedhealth.privacy.smpc.protocol import MpcParty, Program, build_input_vectors, n_outputs

SessionId = Annotated[str, Path(pattern=LINKAGE_ID_PATTERN)]      # a fresh constraint per use (no shared mutable default)


@dataclass
class MpcSession:
    party: MpcParty
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class MpcState:
    sessions: dict[str, MpcSession] = field(default_factory=dict)


def _b64(vec: list[int]) -> str:
    return base64.b64encode(encode_vector(vec)).decode("ascii")


def _unb64(blob: str) -> list[int]:
    try:
        return decode_vector(base64.b64decode(blob, validate=True))
    except ValueError as exc:
        raise HTTPException(400, f"malformed field-element blob: {exc}") from exc


def _session(request: Request, sid: str) -> MpcSession:
    s = request.app.state.mpc.sessions.get(sid)
    if s is None:
        raise HTTPException(404, "unknown computation")
    return s


def _post_peer(request: Request, peer: str, path: str, body: dict) -> None:
    try:
        request.app.state.get_peer(peer).post(path, json=body).raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"peer {peer} unreachable or refused the message") from exc


def make_router(hospital_id: str, get_session) -> APIRouter:
    control = APIRouter(tags=["MPC aggregation"], dependencies=[Depends(security.control_only)])   # coordinator only
    data = APIRouter(tags=["MPC aggregation"])                                                      # hospital -> hospital

    @control.post("/mpc/{sid}/prepare")
    def prepare(body: MpcPrepare, request: Request, sid: SessionId, session: Session = Depends(get_session)) -> dict:
        state: MpcState = request.app.state.mpc
        q = body.query
        if sid in state.sessions:
            raise HTTPException(409, "computation id already used")
        if q.hospital != hospital_id or hospital_id not in body.parties or body.owner not in body.parties:
            raise HTTPException(400, "this hospital is not (correctly) part of the computation")
        is_owner = body.owner == hospital_id
        if not is_owner and (q.value_field or q.group_by):
            raise HTTPException(400, "only the owner may hold the value / group-by field")
        cats = categories(q.group_by, q.condition) if (is_owner and q.group_by) else []
        if is_owner and body.t_outputs != n_outputs(body.agg, len(cats)):
            raise HTTPException(400, "output shape does not match the query")

        n = len(body.parties)
        rows = run_local_query(session, q)                         # <- the ONLY database access, local
        if n == 1:
            matches, m = [(i, r.value, r.category) for i, r in enumerate(rows)], len(rows)
        else:
            linkage = request.app.state.psi.linkages.get(body.lid or "")
            if linkage is None or set(linkage.ring) != set(body.parties) or linkage.m != body.m:
                raise HTTPException(409, "linkage missing or inconsistent with this computation")
            matches = [(linkage.mapping[r.national_id], r.value, r.category)
                       for r in rows if r.national_id in linkage.mapping]
            m = linkage.m
        vectors = build_input_vectors(matches, m, is_owner=is_owner, agg=body.agg, categories=cats)
        program = Program(n, body.t_outputs, m)
        party = MpcParty(hospital_id, list(body.parties), body.owner, program, own_vectors=vectors)
        if n == 1:
            party.local_outputs()
        state.sessions[sid] = MpcSession(party)
        return {"status": "ready"}                                  # nothing about the data goes back

    @control.post("/mpc/{sid}/share")
    def share(request: Request, sid: SessionId) -> dict:
        s = _session(request, sid)
        with s.lock:
            outgoing = s.party.split_inputs()
        for peer, vectors in outgoing.items():
            _post_peer(request, peer, f"/mpc/{sid}/inshare",
                       MpcInShare(sender=hospital_id, vectors=[_b64(v) for v in vectors]).model_dump())
        return {"status": "shared"}

    @data.post("/mpc/{sid}/inshare")                              # data plane
    def inshare(body: MpcInShare, request: Request, sid: SessionId) -> dict:
        security.require_peer(request, body.sender)                    # the caller must BE the sender it claims
        s = _session(request, sid)
        with s.lock:
            try:
                s.party.receive_input_shares(body.sender, [_unb64(b) for b in body.vectors])
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"status": "stored"}

    @control.post("/mpc/{sid}/triples")
    def triples(body: MpcTriples, request: Request, sid: SessionId) -> dict:
        s = _session(request, sid)
        with s.lock:
            try:
                s.party.set_triples(_unb64(body.x), _unb64(body.y), _unb64(body.z))
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"status": "stored"}

    @control.post("/mpc/{sid}/open")
    def open_(body: MpcLayer, request: Request, sid: SessionId) -> dict:
        s = _session(request, sid)
        with s.lock:
            try:
                d, e = s.party.open_layer(body.layer)
            except (RuntimeError, KeyError) as exc:
                raise HTTPException(409, str(exc)) from exc
        for peer in s.party.parties:
            if peer != hospital_id:
                _post_peer(request, peer, f"/mpc/{sid}/inopen",
                           MpcInOpen(sender=hospital_id, layer=body.layer, d=_b64(d), e=_b64(e)).model_dump())
        return {"status": "opened"}

    @data.post("/mpc/{sid}/inopen")                               # data plane
    def inopen(body: MpcInOpen, request: Request, sid: SessionId) -> dict:
        security.require_peer(request, body.sender)
        s = _session(request, sid)
        with s.lock:
            try:
                s.party.receive_open(body.layer, body.sender, _unb64(body.d), _unb64(body.e))
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"status": "stored"}

    @control.post("/mpc/{sid}/finish")
    def finish(body: MpcLayer, request: Request, sid: SessionId) -> dict:
        s = _session(request, sid)
        with s.lock:
            try:
                s.party.finish_layer(body.layer)
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
        return {"status": "finished"}

    @control.post("/mpc/{sid}/output", response_model=MpcOutput)
    def output(request: Request, sid: SessionId) -> MpcOutput:
        s = _session(request, sid)
        try:
            shares = s.party.output()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return MpcOutput(shares=shares)

    @control.delete("/mpc/{sid}")
    def delete(request: Request, sid: SessionId) -> dict:
        request.app.state.mpc.sessions.pop(sid, None)
        return {"status": "deleted"}

    router = APIRouter()
    router.include_router(control)
    router.include_router(data)
    return router
