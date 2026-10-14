"""Protocol driver: how the coordinator talks to the hospitals.

The coordinator holds NO database connection to any hospital - only HTTP clients. It never reads a
hospital file; everything it learns arrives through a hospital's own API.

It is a CONDUCTOR: it tells hospitals "run round r" and waits for all of them (a barrier) before the next
round. The data itself (encrypted PSI lists, later secret shares) flows hospital -> hospital.

Query paths:
  plain_query      Phase 1  INSECURE   raw national IDs cross the wire
  psi_plain_query  Phase 2  INSECURE   linkage is private (PSI) but per-patient predicate results are
                                       still sent to the coordinator by index. Isolates the cost of PSI.
  mpc_run          PSI + secret-shared aggregation with distributed DP noise added inside the computation:
                   the coordinator opens only noisy cells (the private pipeline). Without a noise plan it is the
                   exact evaluation route, which hospitals refuse unless the insecure baseline is enabled.
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Hashable

import httpx

from fedhealth.common.config import load_fields, load_topology
from fedhealth.common.http import parallel
from fedhealth.common.schemas import (LocalMatchResponse, PlainQueryResult, QuerySpec, categories, local_query,
                                      participating_hospitals)
import numpy as np

from fedhealth.common.wire import (FlSumShare, FlTrain, FlUpdate, LinkedMatchRequest, LinkedMatchResponse, MpcOutput,
                                   MpcPrepare, PsiFinalizeResult, PsiInit, PsiStep)
from fedhealth.privacy.federated.features import DIM
from fedhealth.privacy.federated.fedavg import FedAvgConfig, FedAvgResult, dequantize, zcdp_epsilon
from fedhealth.privacy.smpc.sharing import reconstruct_vector
from fedhealth.privacy.smpc.field import P, to_signed
from fedhealth.privacy.dp.distributed import NoisePlan
from fedhealth.privacy.smpc.protocol import Program, assemble_result, assign_roles, n_outputs
from fedhealth.privacy.smpc.sharing import reconstruct_scalar

log = logging.getLogger("fedhealth.coordinator.orchestrator")


class ProtocolError(RuntimeError):
    """A hospital refused or failed a protocol step."""


class ComputationAborted(ProtocolError):
    """A computation failed AFTER hospitals had started sending shares of their data. Such a failure is charged, not
    refunded: if aborts could depend on the data, a free retry would be a channel outside the privacy budget."""


@dataclass(frozen=True)
class MpcRun:
    """Opened output of one secret-shared computation (signed integers): NOISY unless run without a noise plan."""
    hospitals: list[str]
    categories: list[str]
    cells: list[int]            # count: [n] | sum, avg: [n, total] | histogram: one per bin
    timings: dict[str, float]
    noisy: bool = True


@dataclass(frozen=True)
class LinkageInfo:
    lid: str
    hospitals: tuple[str, ...]
    size: int                   # length of the linked index: the PUBLIC padding bound, not the cohort size
    seconds: float              # wall-clock time of the PSI protocol (0 for a cache hit)
    cached: bool = False


def _check(resp: httpx.Response, what: str) -> httpx.Response:
    if resp.status_code >= 400:
        raise ProtocolError(f"{what}: HTTP {resp.status_code} {resp.text[:200]}")
    return resp


def aggregate(spec: QuerySpec, per_hospital: dict[str, dict[Hashable, tuple[int | None, str | None]]]):
    """Join per-hospital {patient key: (value, category)} maps and compute the aggregate IN THE CLEAR.
    Shared by the insecure paths; the private path computes the same function on secret shares."""
    common = set.intersection(*(set(rows) for rows in per_hospital.values()))
    fields = load_fields()
    if spec.agg == "count":
        return len(common), len(common)
    if spec.agg in ("sum", "avg"):
        owner = fields.value_fields[spec.value_field].owner
        total = sum(per_hospital[owner][k][0] for k in common)
        return (total if spec.agg == "sum" else (total / len(common) if common else None)), len(common)
    owner = fields.group_by[spec.group_by].owner
    counts = Counter(per_hospital[owner][k][1] for k in common)
    return {c: counts.get(c, 0) for c in categories(spec.group_by, spec.condition)}, len(common)


class Orchestrator:
    def __init__(self, clients: dict[str, httpx.Client], psi_group: str = "modp2048"):
        self.clients = clients
        self.psi_group = psi_group
        self._linkages: dict[tuple[str, ...], LinkageInfo] = {}
        self._lock = threading.Lock()
        self._link_lock = threading.Lock()

    # ------------------------------------------------------------------ Phase 1: plain baseline
    def plain_query(self, spec: QuerySpec) -> PlainQueryResult:
        """INSECURE: every participating hospital returns the national IDs of its matching patients; the
        coordinator intersects them in the clear. Every identifier crosses the wire."""
        t0 = time.perf_counter()
        hospitals = participating_hospitals(spec)

        def fetch(h: str) -> LocalMatchResponse:
            body = local_query(spec, h).model_dump(mode="json", exclude_none=True)
            return LocalMatchResponse.model_validate(_check(self.clients[h].post("/insecure/match", json=body), f"{h} match").json())

        responses = parallel({h: (lambda h=h: fetch(h)) for h in hospitals})
        per_hospital = {h: {r.national_id: (r.value, r.category) for r in resp.rows} for h, resp in responses.items()}
        result, size = aggregate(spec, per_hospital)
        return PlainQueryResult(agg=spec.agg, hospitals=hospitals, result=result, cohort_size=size,
                                timings={"total_s": time.perf_counter() - t0})

    def _forget(self, hospitals, path: str) -> None:
        """Best-effort cleanup at every hospital. It must never replace the real outcome (a result, or the exception
        already in flight - e.g. ComputationAborted, which decides that the budget is charged), so an unreachable
        hospital is only logged: its in-memory session goes with it anyway."""
        def one(h: str) -> None:
            try:
                self.clients[h].delete(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("cleanup of %s at hospital %s failed: %s", path, h, exc)
        parallel({h: (lambda h=h: one(h)) for h in hospitals})

    # ------------------------------------------------------------------ Phase 2: PSI linkage
    def link(self, hospitals: list[str], force: bool = False) -> LinkageInfo:
        """Run (or reuse) the PSI linkage for this hospital subset. Linkage runs on the FULL patient lists, so
        it is independent of the query: nothing about the query leaks through list sizes, and the result can
        be cached and reused by every query over the same hospitals."""
        if len(hospitals) < 2:
            raise ValueError("linkage needs at least two hospitals")
        with self._link_lock:                       # concurrent queries share ONE linkage run instead of racing
            return self._link_locked(hospitals, force)

    def _link_locked(self, hospitals: list[str], force: bool) -> LinkageInfo:
        key = tuple(hospitals)
        with self._lock:
            old = self._linkages.get(key)
            if old is not None and not force:
                return LinkageInfo(old.lid, old.hospitals, old.size, 0.0, cached=True)
        t0 = time.perf_counter()
        lid = secrets.token_hex(8)
        n = len(hospitals)

        def on_all(path: str, body: dict | None = None, method: str = "post") -> dict[str, httpx.Response]:
            return parallel({h: (lambda h=h: _check(getattr(self.clients[h], method)(path, **({"json": body} if body is not None else {})), f"{h} {path}"))
                             for h in hospitals})
        try:
            on_all(f"/psi/{lid}/init", PsiInit(ring=hospitals, group=self.psi_group).model_dump())
            for r in range(1, n + 1):                                   # n ring rounds, barrier after each
                on_all(f"/psi/{lid}/step", PsiStep(round=r).model_dump())
            on_all(f"/psi/{lid}/publish", {})
            done = {h: PsiFinalizeResult.model_validate(r.json()) for h, r in on_all(f"/psi/{lid}/finalize", {}).items()}
        except Exception:
            self._forget(hospitals, f"/psi/{lid}")
            raise
        if len({(d.size, d.digest) for d in done.values()}) != 1:
            raise ProtocolError("hospitals disagree on the linked index - aborting")
        info = LinkageInfo(lid, key, next(iter(done.values())).size, time.perf_counter() - t0)
        with self._lock:
            if old is not None:                                          # forced re-link: retire the previous one
                self._forget(hospitals, f"/psi/{old.lid}")
            self._linkages[key] = info
        return info

    # ------------------------------------------------------------------ Phase 3: PSI + secret-shared aggregation
    def mpc_run(self, spec: QuerySpec, noise: NoisePlan | None = None, force_link: bool = False) -> "MpcRun":
        """PSI linkage + secret-shared computation with the DP noise added INSIDE it. Returns the opened output cells:
        exact + noise when `noise` is given (the private pipeline), exact only when it is None (the insecure
        evaluation route - hospitals refuse that unless they run with the insecure baseline enabled).
        What each party sees:
          input hospital  its own predicate slice; it sends shares (uniformly random numbers) of its vectors
          computer P0/P1  shares of everyone's vectors and noise, and the Beaver openings d, e (uniformly random)
          dealer          the triples it made; nothing that was computed with them
          coordinator     the public index length, the query SHAPE (parties, #output cells) and two random-looking
                          shares per output cell, whose sum is the NOISY aggregate"""
        t_start = time.perf_counter()
        hospitals = participating_hospitals(spec)
        n = len(hospitals)
        fields = load_fields()
        owner = (fields.value_fields[spec.value_field].owner if spec.value_field
                 else fields.group_by[spec.group_by].owner if spec.group_by else hospitals[-1])
        cats = categories(spec.group_by, spec.condition) if spec.group_by else []
        t_out = n_outputs(spec.agg, len(cats))
        timings: dict[str, float] = {}
        if noise is not None and (len(noise.cells) != t_out or noise.mechanism == "none"):
            raise ValueError("noise plan does not match the query")

        info = self.link(hospitals, force=force_link) if n >= 2 else None
        timings["linkage_s"] = info.seconds if info else 0.0
        roles = assign_roles(hospitals, owner, load_topology().ring) if n >= 2 else None
        involved = list(roles.involved) if roles else hospitals
        computers = list(roles.computers) if roles else hospitals
        program = Program(n, t_out, info.size if info else 0)
        sid = secrets.token_hex(8)

        def on(targets: list[str], path: str, body_for=lambda h: None) -> dict[str, httpx.Response]:
            def call(h: str):
                body = body_for(h)
                return _check(self.clients[h].post(path, json=body if body is not None else {}), f"{h} {path}")
            return parallel({h: (lambda h=h: call(h)) for h in targets})

        wire_noise = noise.to_wire() if noise is not None else None
        t0 = time.perf_counter()
        data_sent = stale = False
        try:
            on(involved, f"/mpc/{sid}/prepare", lambda h: MpcPrepare(
                parties=hospitals, owner=owner, computers=list(roles.computers) if roles else [],
                dealer=roles.dealer if roles else None, agg=spec.agg, t_outputs=t_out, m=program.m,
                lid=info.lid if info else None, query=local_query(spec, h) if h in hospitals else None,
                noise=wire_noise).model_dump(mode="json"))
            timings["prepare_s"] = time.perf_counter() - t0
            data_sent = True                                                      # from here on, a failure is charged
            if n >= 2:
                t0 = time.perf_counter()
                on(involved, f"/mpc/{sid}/share")                                 # hospitals -> computers
                timings["share_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                on([roles.dealer], f"/mpc/{sid}/deal")                            # dealer -> computers (offline phase)
                timings["triples_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                for layer in range(len(program.layer_sizes)):
                    on(computers, f"/mpc/{sid}/open", lambda h, layer=layer: {"layer": layer})     # P0 <-> P1
                    on(computers, f"/mpc/{sid}/finish", lambda h, layer=layer: {"layer": layer})
                timings["multiply_s"] = time.perf_counter() - t0
            outs = {h: MpcOutput.model_validate(r.json()).shares for h, r in on(computers, f"/mpc/{sid}/output").items()}
        except Exception as exc:
            if data_sent:
                raise ComputationAborted(f"aborted after the hospitals shared their data: {exc}") from exc
            if not (n >= 2 and not force_link and isinstance(exc, ProtocolError) and "linkage missing" in str(exc)):
                raise
            stale = True              # a hospital no longer holds the cached linkage (it restarted): nothing was shared
        finally:
            self._forget(involved, f"/mpc/{sid}")
        if stale:
            log.info("a hospital lost linkage %s: linking %s afresh", info.lid, hospitals)
            return self.mpc_run(spec, noise, force_link=True)

        cells = [to_signed(reconstruct_scalar([outs[h][j] for h in computers])) for j in range(t_out)]
        timings["total_s"] = time.perf_counter() - t_start
        return MpcRun(hospitals=hospitals, categories=cats, cells=cells, timings=timings, noisy=noise is not None)

    def mpc_query(self, spec: QuerySpec, force_link: bool = False) -> PlainQueryResult:
        """EXACT answer (no DP, no k-threshold, no auth): evaluation / testing only, hence on an /insecure route."""
        run = self.mpc_run(spec, None, force_link)
        result = assemble_result(spec.agg, [c % P for c in run.cells], run.categories)
        size = (run.cells[0] if spec.agg in ("count", "sum", "avg")
                else sum(result.values()))
        return PlainQueryResult(mode="mpc-exact-no-dp", agg=spec.agg, hospitals=run.hospitals, result=result,
                                cohort_size=size, timings=run.timings)

    # ------------------------------------------------------------------ Phase 8 (stretch): federated learning
    def federated_train(self, cfg: FedAvgConfig, hospitals: list[str] | None = None, evaluate=None, delta: float = 1e-6) -> FedAvgResult:
        """FedAvg over the hospital services. Per round: send the global weights; every hospital trains on its own rows and
        holds a clipped (optionally noised) update; then EITHER each update is returned (plain) OR the updates are
        additively secret-shared hospital-to-hospital and only their SUM reaches the coordinator (secure_agg).
        `evaluate(weights) -> dict` is an optional offline callback (the coordinator itself has no test data)."""
        parties = list(hospitals or ["A", "B", "C"])
        w = np.zeros(DIM)
        history: list[dict] = []

        def on_all(path: str, body: dict | None = None, method: str = "post") -> dict[str, httpx.Response]:
            def call(h: str):
                kw = {"json": body if body is not None else {}} if method == "post" else {}
                return _check(getattr(self.clients[h], method)(path, **kw), f"{h} {path}")
            return parallel({h: (lambda h=h: call(h)) for h in parties})

        for r in range(1, cfg.rounds + 1):
            sid = secrets.token_hex(8)
            body = FlTrain(weights=w.tolist(), parties=parties, round_no=r, epochs=cfg.local_epochs, lr=cfg.lr,
                           batch=cfg.batch, l2=cfg.l2, clip=cfg.clip, noise_multiplier=cfg.noise_multiplier, seed=cfg.seed).model_dump()
            try:
                on_all(f"/fl/{sid}/train", body)
                if cfg.secure_agg:
                    on_all(f"/fl/{sid}/share")                                      # hospital -> hospital shares
                    sums = {h: FlSumShare.model_validate(resp.json()).share for h, resp in on_all(f"/fl/{sid}/sumshare").items()}
                    total = dequantize(reconstruct_vector([sums[h] for h in parties]))
                else:
                    total = np.sum([FlUpdate.model_validate(resp.json()).update for resp in on_all(f"/fl/{sid}/update").values()], axis=0)
            finally:
                on_all(f"/fl/{sid}", method="delete")
            w = w + total / len(parties)
            history.append({"round": r, **(evaluate(w) if evaluate else {})})
        eps = zcdp_epsilon(cfg.noise_multiplier, cfg.rounds, delta) if cfg.noise_multiplier > 0 else None
        return FedAvgResult(w, history, eps)

    # ------------------------------------------------------------------ Phase 2: PSI + insecure aggregation
    def psi_plain_query(self, spec: QuerySpec, force_link: bool = False) -> PlainQueryResult:
        t0 = time.perf_counter()
        hospitals = participating_hospitals(spec)
        if len(hospitals) < 2:                    # nothing to link: a single hospital answers on its own
            return self.plain_query(spec)
        info = self.link(hospitals, force=force_link)
        t1 = time.perf_counter()

        def fetch(h: str) -> LinkedMatchResponse:
            body = LinkedMatchRequest(query=local_query(spec, h)).model_dump(mode="json", exclude_none=True)
            return LinkedMatchResponse.model_validate(
                _check(self.clients[h].post(f"/insecure/match_linked/{info.lid}", json=body), f"{h} match_linked").json())

        responses = parallel({h: (lambda h=h: fetch(h)) for h in hospitals})
        per_hospital = {h: {r.idx: (r.value, r.category) for r in resp.rows} for h, resp in responses.items()}
        result, size = aggregate(spec, per_hospital)
        return PlainQueryResult(mode="psi-linked-insecure-aggregation", agg=spec.agg, hospitals=hospitals,
                                result=result, cohort_size=size,
                                timings={"linkage_s": info.seconds, "aggregation_s": time.perf_counter() - t1,
                                         "total_s": time.perf_counter() - t0, "linkage_cached": float(info.cached)})
