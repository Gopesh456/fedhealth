"""Hospital-side federated learning (FedAvg) - see fedhealth/privacy/federated/fedavg.py.

Each round the coordinator sends the global weights; the hospital trains on ITS OWN `training_episodes` rows (split =
'train') and holds a clipped, optionally noised, weight UPDATE. Then one of two things leaves the hospital:

  plain mode    POST /fl/{sid}/update    the update itself            -> the coordinator sees each hospital's update
  secure mode   POST /fl/{sid}/share     the update is split into additive shares (fixed-point), one per hospital,
                                         sent hospital -> hospital; POST /fl/{sid}/sumshare returns only the SUM of the
                                         shares this hospital holds -> the coordinator learns only the SUM of updates

Nothing else is ever sent: no row, no feature, no label, no per-patient gradient, not even the number of rows.
"""
from __future__ import annotations

import base64
import threading
import zlib
from dataclasses import dataclass, field
from typing import Annotated

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from fedhealth.common.wire import LINKAGE_ID_PATTERN, FlInShare, FlSumShare, FlTrain, FlUpdate
from fedhealth.hospital import models as m
from fedhealth.hospital import security
from fedhealth.privacy.federated.features import DIM, row_to_vector
from fedhealth.privacy.federated.fedavg import FedAvgConfig, add_local_noise, clip_update, quantize
from fedhealth.privacy.federated.logreg import train_local
from fedhealth.privacy.smpc.field import decode_vector, encode_vector
from fedhealth.privacy.smpc.sharing import share_vector

SessionId = Annotated[str, Path(pattern=LINKAGE_ID_PATTERN)]


@dataclass
class FlSession:
    parties: list[str]
    update: np.ndarray                                              # clipped (+ noised) update; leaves only as above
    shares: dict[str, list[int]] = field(default_factory=dict)      # sender -> my share of its update
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class FlState:
    sessions: dict[str, FlSession] = field(default_factory=dict)


def _session(request: Request, sid: str) -> FlSession:
    s = request.app.state.fl.sessions.get(sid)
    if s is None:
        raise HTTPException(404, "unknown federated-learning round")
    return s


def make_router(hospital_id: str, get_session) -> APIRouter:
    control = APIRouter(tags=["federated learning"], dependencies=[Depends(security.control_only)])
    data = APIRouter(tags=["federated learning"])
    model = m.TRAINING_EPISODE[hospital_id]

    @control.post("/fl/{sid}/train")
    def train(body: FlTrain, request: Request, sid: SessionId, session: Session = Depends(get_session)) -> dict:
        state: FlState = request.app.state.fl
        if sid in state.sessions or hospital_id not in body.parties or len(body.weights) != DIM:
            raise HTTPException(400, "bad federated-learning round")
        if body.noise_multiplier > 0 and not body.clip:
            raise HTTPException(400, "noise needs a clipping bound (the clip is the sensitivity)")
        rows = session.execute(select(model).where(model.split == "train")).scalars().all()
        if not rows:
            raise HTTPException(409, "no training rows")
        X = np.array([row_to_vector(r) for r in rows])
        y = np.array([r.responded for r in rows], dtype=float)
        w0 = np.array(body.weights)
        cfg = FedAvgConfig(local_epochs=body.epochs, lr=body.lr, batch=body.batch, l2=body.l2, clip=body.clip,
                           noise_multiplier=body.noise_multiplier)
        rng = np.random.default_rng([body.seed, body.round_no, zlib.crc32(hospital_id.encode())])
        w = train_local(w0, X, y, epochs=cfg.local_epochs, lr=cfg.lr, batch=cfg.batch, l2=cfg.l2, rng=rng)
        update = add_local_noise(clip_update(w - w0, cfg.clip), cfg, len(body.parties), rng)
        state.sessions[sid] = FlSession(list(body.parties), update)
        return {"status": "trained"}                                # deliberately NOT returning the row count

    @control.post("/fl/{sid}/update", response_model=FlUpdate)
    def update(request: Request, sid: SessionId) -> FlUpdate:       # plain mode
        return FlUpdate(update=[float(x) for x in _session(request, sid).update])

    @control.post("/fl/{sid}/share")
    def share(request: Request, sid: SessionId) -> dict:             # secure mode, step 1
        s = _session(request, sid)
        shares = share_vector(quantize(s.update), len(s.parties))    # additive shares, one per party (index = ring order)
        with s.lock:
            s.shares[hospital_id] = shares[s.parties.index(hospital_id)]
        for peer, sh in zip(s.parties, shares):
            if peer == hospital_id:
                continue
            body = FlInShare(sender=hospital_id, blob=base64.b64encode(encode_vector(sh)).decode("ascii")).model_dump()
            try:
                request.app.state.get_peer(peer).post(f"/fl/{sid}/inshare", json=body).raise_for_status()
            except httpx.HTTPError as exc:
                raise HTTPException(502, f"peer {peer} unreachable or refused the share") from exc
        return {"status": "shared"}

    @data.post("/fl/{sid}/inshare")                                  # data plane: hospital -> hospital
    def inshare(body: FlInShare, request: Request, sid: SessionId) -> dict:
        security.require_peer(request, body.sender)
        s = _session(request, sid)
        try:
            vec = decode_vector(base64.b64decode(body.blob, validate=True))
        except ValueError as exc:
            raise HTTPException(400, "malformed share") from exc
        if body.sender == hospital_id or body.sender not in s.parties or len(vec) != DIM:
            raise HTTPException(400, "unexpected share")
        with s.lock:
            if body.sender in s.shares:
                raise HTTPException(409, "duplicate share")
            s.shares[body.sender] = vec
        return {"status": "stored"}

    @control.post("/fl/{sid}/sumshare", response_model=FlSumShare)  # secure mode, step 2
    def sumshare(request: Request, sid: SessionId) -> FlSumShare:
        s = _session(request, sid)
        if set(s.shares) != set(s.parties):
            raise HTTPException(409, "shares from every party have not all arrived")
        from fedhealth.privacy.smpc.field import P
        return FlSumShare(share=[sum(col) % P for col in zip(*[s.shares[p] for p in s.parties])])

    @control.delete("/fl/{sid}")
    def delete(request: Request, sid: SessionId) -> dict:
        request.app.state.fl.sessions.pop(sid, None)
        return {"status": "deleted"}

    router = APIRouter()
    router.include_router(control)
    router.include_router(data)
    return router
