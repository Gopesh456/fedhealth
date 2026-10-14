"""Hospital-side endpoints of the secret-shared aggregation protocol (see fedhealth/privacy/smpc/protocol.py).

Control plane (coordinator -> hospital):  prepare, share, deal, open, finish, output, delete
Data plane    (hospital -> hospital):     inshare, intriples, inopen     <- the coordinator never calls these

What leaves this hospital, and where:
    prepare   nothing, not even how long it took: the answer comes at a public deadline (hospital/deadlines.py).
              The hospital re-checks the public noise plan against ITS OWN policy, evaluates its predicate
              against its OWN database, builds 0/1 indicator vectors over the linked patient index and samples ITS
              noise share. All of it stays in memory.
    share     additive SHARES of those vectors and of the noise share go to the two computing hospitals.
    deal      (dealer only) Beaver-triple shares go to the two computing hospitals (a seed for P0, values for P1).
    open      (computers only) shares of d = a - x and e = b - y go to the other computer.
    output    (computers only) one additive share per output cell of  exact + noise  goes to the coordinator.
No endpoint returns a patient, an ID, an unshared vector or an un-noised number.

Binding: every hospital hashes the computation header it was told (MpcPrepare minus its own query slice) and every
data-plane message carries that digest; a mismatch is refused, so all involved hospitals provably ran ONE plan.
"""
from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy.orm import Session

from fedhealth.common.config import load_topology
from fedhealth.common.schemas import categories, value_bounds
from fedhealth.common.wire import (LINKAGE_ID_PATTERN, MpcInOpen, MpcInShare, MpcInTriples, MpcLayer, MpcOutput,
                                   MpcPrepare)
from fedhealth.hospital import security
from fedhealth.hospital.deadlines import guarded
from fedhealth.hospital.predicate_runner import run_local_query
from fedhealth.privacy.dp.distributed import NoisePlan, load_noise_policy
from fedhealth.privacy.smpc.field import P, decode_vector, encode_vector
from fedhealth.privacy.smpc.protocol import (SEED_BYTES, MpcParty, Program, Roles, assign_roles, build_input_vectors,
                                             deal, n_outputs)

SessionId = Annotated[str, Path(pattern=LINKAGE_ID_PATTERN)]      # a fresh constraint per use (no shared mutable default)


@dataclass
class MpcSession:
    party: MpcParty
    digest: str                         # MpcPrepare.header_digest: every peer message must carry the same one
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


def _same_header(s: MpcSession, body) -> None:
    """Refuse a peer message sent under a different computation header: the coordinator told the two hospitals
    different things (plan, shape, roles, linkage or cohort size)."""
    if body.session != s.digest:
        raise HTTPException(409, "computation header mismatch: the hospitals were not told the same computation")


def _post_peer(request: Request, peer: str, path: str, body: dict) -> None:
    try:
        request.app.state.get_peer(peer).post(path, json=body).raise_for_status()
    except httpx.HTTPStatusError as exc:                    # the peer's refusal reason is a fixed protocol message
        raise HTTPException(502, f"peer {peer} refused the message: {exc.response.text[:160]}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"peer {peer} unreachable") from exc


def make_router(hospital_id: str, get_session) -> APIRouter:
    control = APIRouter(tags=["MPC aggregation"], dependencies=[Depends(security.control_only)])   # coordinator only
    data = APIRouter(tags=["MPC aggregation"])                                                      # hospital -> hospital
    noise_policy = load_noise_policy()

    @control.post("/mpc/{sid}/prepare")
    def prepare(body: MpcPrepare, request: Request, sid: SessionId, session: Session = Depends(get_session)) -> dict:
        state: MpcState = request.app.state.mpc
        deadlines = request.app.state.deadlines
        q = body.query
        if sid in state.sessions:
            raise HTTPException(409, "computation id already used")
        n = len(body.parties)
        if body.owner not in body.parties:
            raise HTTPException(400, "the owner must be an input hospital")
        topo = load_topology()
        if body.parties != [h for h in topo.ring if h in body.parties]:      # distinct, and in the public ring order
            raise HTTPException(400, "parties must be distinct hospitals in ring order")
        if body.m != (topo.link_pad_to if n >= 2 else 0):                       # the dealer checks it too
            raise HTTPException(400, "the index length must be the public linkage bound")
        roles: Roles | None = None
        if n >= 2:                                                  # the role assignment is public: recompute, never trust
            roles = assign_roles(body.parties, body.owner, topo.ring)
            if list(roles.computers) != body.computers or roles.dealer != body.dealer:
                raise HTTPException(400, "role assignment does not match the protocol")
            if hospital_id not in roles.involved:
                raise HTTPException(400, "this hospital is not part of the computation")
        elif hospital_id != body.parties[0]:
            raise HTTPException(400, "this hospital is not part of the computation")
        is_input = hospital_id in body.parties
        if is_input != (q is not None) or (q is not None and q.hospital != hospital_id):
            raise HTTPException(400, "this hospital is not (correctly) part of the computation")
        is_owner = body.owner == hospital_id
        if q is not None and not is_owner and (q.value_field or q.group_by):
            raise HTTPException(400, "only the owner may hold the value / group-by field")
        if is_owner and q is not None:
            allowed = ("histogram",) if q.group_by else ("sum", "avg") if q.value_field else ("count",)
            if body.agg not in allowed:
                raise HTTPException(400, "the aggregate does not match the owner's slice of the query")
        cats = categories(q.group_by, q.condition) if (is_owner and q.group_by) else []
        if is_owner and body.t_outputs != n_outputs(body.agg, len(cats)):
            raise HTTPException(400, "output shape does not match the query")

        plan = None                                                 # ---- the noise plan: re-checked against MY policy
        if body.noise is None:
            if not request.app.state.allow_insecure:
                raise HTTPException(400, "noise is required: exact computations are disabled on this hospital")
        else:
            plan = NoisePlan.from_wire(body.noise)
            contributors = len(roles.involved) if roles else 1
            if plan.contributors != contributors or len(plan.cells) != body.t_outputs:
                raise HTTPException(400, "noise plan does not match the computation")
            if is_owner and body.agg in ("sum", "avg") and q.value_field:
                lo, hi = value_bounds(q.value_field, q.condition)
                if plan.cells[1].sensitivity < max(abs(lo), abs(hi)):
                    raise HTTPException(400, "noise plan understates the sensitivity of the value field")
            try:
                plan.check(noise_policy, body.agg, body.t_outputs)
            except ValueError as exc:
                raise HTTPException(400, f"noise plan refused: {exc}") from exc

        linkage = None
        if is_input and n >= 2:                                     # public metadata: checked before the window
            linkage = request.app.state.psi.linkages.get(body.lid or "")
            if linkage is None or set(linkage.ring) != set(body.parties) or linkage.size != body.m:
                raise HTTPException(409, "linkage missing or inconsistent with this computation")
        deadline = deadlines.prepare(plan, body.t_outputs)
        if deadline > deadlines.max_prepare:
            raise HTTPException(400, "noise plan refused: its public deadline is above this hospital's cap")

        # Everything below touches the data or draws the noise: it runs alone at this hospital and answers at a
        # deadline fixed by public things only (hospital/deadlines.py).
        with guarded(request, deadline, "mpc prepare"):
            vectors: list[list[int]] = []
            m = body.m
            if is_input:
                rows = run_local_query(session, q)                   # <- the ONLY database access, local
                if n == 1:
                    matches, m = [(i, r.value, r.category) for i, r in enumerate(rows)], len(rows)
                else:
                    matches = [(linkage.mapping[r.national_id], r.value, r.category)
                               for r in rows if r.national_id in linkage.mapping]
                vectors = build_input_vectors(matches, m, is_owner=is_owner, agg=body.agg, categories=cats)
                del rows, matches                                    # freed inside the window: the cost grows with them
                session.close()
            noise = [x % P for x in plan.sample_share()] if plan else None  # MY share, MY randomness
            party = MpcParty(hospital_id, roles, Program(n, body.t_outputs, m), own_vectors=vectors, noise=noise)
            if n == 1:
                party.local_outputs()
                party.own_vectors = vectors = []                      # their length is the true match count: free now
            state.sessions[sid] = MpcSession(party, body.header_digest(sid))
        return {"status": "ready"}                                  # nothing about the data goes back

    @control.post("/mpc/{sid}/share")
    def share(request: Request, sid: SessionId) -> dict:
        s = _session(request, sid)
        with s.lock:
            outgoing = s.party.split_inputs()
            noise_out = s.party.split_noise()
        for peer in sorted(set(outgoing) | set(noise_out)):
            _post_peer(request, peer, f"/mpc/{sid}/inshare",
                       MpcInShare(sender=hospital_id, session=s.digest, vectors=[_b64(v) for v in outgoing.get(peer, [])],
                                  noise=_b64(noise_out[peer]) if peer in noise_out else None).model_dump())
        return {"status": "shared"}

    @data.post("/mpc/{sid}/inshare")                              # data plane
    def inshare(body: MpcInShare, request: Request, sid: SessionId) -> dict:
        security.require_peer(request, body.sender)                    # the caller must BE the sender it claims
        s = _session(request, sid)
        _same_header(s, body)
        with s.lock:
            try:
                if s.party.roles is not None and body.sender in s.party.roles.inputs:
                    s.party.receive_input_shares(body.sender, [_unb64(b) for b in body.vectors])
                elif body.vectors:
                    raise ValueError(f"{body.sender!r} holds no input in this computation")
                if body.noise is not None:
                    s.party.receive_noise_share(body.sender, _unb64(body.noise))
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"status": "stored"}

    @control.post("/mpc/{sid}/deal")
    def deal_(request: Request, sid: SessionId) -> dict:
        """Dealer only: fresh triples for the whole program, straight to the two computers."""
        s = _session(request, sid)
        roles = s.party.roles
        if roles is None or roles.dealer != hospital_id:
            raise HTTPException(409, "this hospital is not the dealer of this computation")
        seed, (x1, y1, z1) = deal(s.party.program.total_triples)
        p0, p1 = roles.computers
        _post_peer(request, p0, f"/mpc/{sid}/intriples",
                   MpcInTriples(sender=hospital_id, session=s.digest, seed=base64.b64encode(seed).decode("ascii")).model_dump())
        _post_peer(request, p1, f"/mpc/{sid}/intriples",
                   MpcInTriples(sender=hospital_id, session=s.digest, x=_b64(x1), y=_b64(y1), z=_b64(z1)).model_dump())
        return {"status": "dealt"}

    @data.post("/mpc/{sid}/intriples")                            # data plane: dealer -> computer
    def intriples(body: MpcInTriples, request: Request, sid: SessionId) -> dict:
        security.require_peer(request, body.sender)
        s = _session(request, sid)
        roles = s.party.roles
        if roles is None or body.sender != roles.dealer:
            raise HTTPException(403, "triples are accepted only from this computation's dealer")
        _same_header(s, body)
        with s.lock:
            try:
                if s.party.index == 0:
                    if body.seed is None:
                        raise ValueError("P0 expects a seed")
                    seed = base64.b64decode(body.seed, validate=True)
                    if len(seed) != SEED_BYTES:
                        raise ValueError("bad seed")
                    s.party.set_triples_from_seed(seed)
                else:
                    if None in (body.x, body.y, body.z):
                        raise ValueError("P1 expects explicit triple shares")
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
        _post_peer(request, s.party.peer, f"/mpc/{sid}/inopen",
                   MpcInOpen(sender=hospital_id, session=s.digest, layer=body.layer, d=_b64(d), e=_b64(e)).model_dump())
        return {"status": "opened"}

    @data.post("/mpc/{sid}/inopen")                               # data plane
    def inopen(body: MpcInOpen, request: Request, sid: SessionId) -> dict:
        security.require_peer(request, body.sender)
        s = _session(request, sid)
        _same_header(s, body)
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
