"""Public response deadlines: the coordinator cannot time what a hospital computed.

WHY
    The coordinator times every call it makes. Without care, how long a hospital takes to answer reveals what it did:
      /mpc/prepare   runs the hospital's predicate (longer the more patients match) and draws its DP noise share (the
                     exact Polya sampler's work grows with the size of the value it returns, so a slow answer means a
                     large noise share - which the coordinator could partly subtract);
      /psi/init      loads the true patient list;   /psi/finalize   maps the true linked cohort (and DELETE /psi frees it);
      /fl/train      loops over the hospital's training rows.

WHAT
    Each of these handlers answers no earlier than a deadline that depends only on PUBLIC things: this hospital's own
    configuration and the public parameters of the request (the noise plan, the number of output cells, the number
    of epochs). The deadlines sit well above the work's worst case, so the time the coordinator observes IS the
    deadline. If the work ever runs past it (an extreme noise draw, a loaded machine) the answer waits for the next
    doubling (2D, 4D, ...), so an overrun shows only which doubling it fell in; the hospital counts overruns locally
    (`app.state.deadline_overruns`) and never reports them. The overrun probability is the residual leak:
    SECURITY_THEOREMS.md (L12) turns it into an additive delta.

    The other control calls (PSI step/publish, MPC share/deal/open/finish/output, FL share/sumshare) do the same work
    whatever the data: they process lists of the public padded length (or the fixed model size).

    A deadline alone does not stop a coordinator from timing OTHER requests while the work runs (a /health probe is
    slowed while the predicate holds the CPU), nor concurrent guarded calls from pushing each other past their
    deadlines. So each guarded call runs ALONE at its hospital, in a quiet window (`QuietGate`): it starts once no other
    request is in progress, and no request starts until its deadline has passed. Anything the coordinator times then
    waits for public deadlines, never for data-dependent work. Guarded calls never call another hospital, so the
    window cannot deadlock with the hospital-to-hospital traffic of other computations.

The deadlines come from the HOSPITAL's own configs/policy.yaml (`response_deadlines`), never from the coordinator.
FEDHEALTH_DEADLINE_SCALE multiplies them all (0 switches them off, as the test suite does for speed; tests that check
the deadlines set their own). The defaults were measured on a 4-core machine (evaluation/performance.py reports the
overrun count); a deployment on slower hardware raises them.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Iterator

from fastapi import Request

from fedhealth.privacy.dp.distributed import NoisePlan

log = logging.getLogger("fedhealth.hospital.deadlines")


@dataclass(frozen=True)
class ResponseDeadlines:
    """Seconds. A field of 0 switches that deadline off."""
    psi_init: float = 1.5
    psi_finalize: float = 0.25
    mpc_prepare: float = 0.25           # predicate + indicator vectors over the padded linked index
    noise_per_cell: float = 0.005       # + this per output cell: the constant part of drawing a share
    noise_per_scale: float = 1.0e-4     # + this per cell per unit of Laplace scale t/s (the Polya sampler is ~linear)
    fl_train_per_epoch: float = 0.5
    max_prepare: float = 60.0           # a cap, not a deadline: a plan whose prepare deadline is longer is refused

    def scaled(self, k: float) -> "ResponseDeadlines":
        """Every deadline multiplied by k (the cap stays)."""
        return replace(self, **{f.name: getattr(self, f.name) * k for f in fields(self) if f.name != "max_prepare"})

    @staticmethod
    def noise_scales(plan: NoisePlan | None) -> list[float]:
        """Per-cell scale that the sampler's work grows with: t/s for Laplace-type shares (capped by the hospital's
        distributed_noise.max_noise_scale, so the deadline is bounded). The discrete Gaussian sampler does a constant
        expected amount of work whatever its variance, so its cells count 0 here."""
        if plan is None or plan.mechanism != "laplace":
            return []
        return [c.t / c.s for c in plan.cells]

    def prepare(self, plan: NoisePlan | None, t_outputs: int) -> float:
        """The /mpc/prepare deadline: a function of the public plan and output shape only."""
        if self.mpc_prepare <= 0:
            return 0.0
        return self.mpc_prepare + self.noise_per_cell * t_outputs + self.noise_per_scale * sum(self.noise_scales(plan))

    def fl_train(self, epochs: int) -> float:
        return self.fl_train_per_epoch * max(epochs, 1)


def load_response_deadlines(path=None) -> ResponseDeadlines:
    import yaml
    from fedhealth.common.config import config_dir
    with open(path or (config_dir() / "policy.yaml"), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh).get("response_deadlines") or {}
    known = {f.name for f in fields(ResponseDeadlines)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown response_deadlines keys: {sorted(unknown)}")
    deadlines = ResponseDeadlines(**{k: float(v) for k, v in raw.items()})
    scale = float(os.environ.get("FEDHEALTH_DEADLINE_SCALE", "1"))
    return deadlines if scale == 1 else deadlines.scaled(scale)


@contextmanager
def answer_no_earlier_than(seconds: float, overruns: Counter | None = None, what: str = "") -> Iterator[None]:
    """Hold the answer (normal or error) until `seconds` after entering the block; on an overrun, until the next
    doubling of `seconds` that has not yet passed."""
    start = time.monotonic()
    try:
        yield
    finally:
        if seconds > 0:
            elapsed = time.monotonic() - start
            target = seconds
            while target < elapsed:
                target *= 2
            if target > seconds:
                if overruns is not None:
                    overruns[what] += 1
                log.warning("%s overran its public deadline (%.3fs): answering at %.3fs", what, seconds, target)
            time.sleep(target - elapsed)


class QuietGate:
    """A readers-writer lock that favours readers. Every ordinary request is a reader for as long as it runs; a
    guarded call is the writer. The writer starts only when no reader is active and holds the gate until its deadline,
    so nothing else runs at this hospital while data-dependent work might. Readers never wait for a writer that has
    not started, and a writer never calls another hospital, so hospital-to-hospital calls cannot deadlock on it."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False

    @contextmanager
    def reader(self) -> Iterator[None]:
        with self._cond:
            self._cond.wait_for(lambda: not self._writer)
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        with self._cond:
            self._cond.wait_for(lambda: not self._writer and self._readers == 0)
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


GUARDED_ROUTES = frozenset({"/psi/{lid}/init", "/psi/{lid}/finalize", "/psi/{lid}", "/mpc/{sid}/prepare",
                            "/fl/{sid}/train"})


def quiet_reader(request: Request) -> Iterator[None]:
    """App-wide FastAPI dependency: every request except the guarded ones waits for (and holds off) guarded windows."""
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None) in GUARDED_ROUTES:
        yield                                          # the handler takes the gate exclusively itself
        return
    with request.app.state.quiet_gate.reader():
        yield


@contextmanager
def guarded(request, seconds: float, what: str) -> Iterator[None]:
    """A guarded call's quiet window: alone at this hospital, answering no earlier than `seconds` after it started."""
    state = request.app.state
    with state.quiet_gate.exclusive(), answer_no_earlier_than(seconds, state.deadline_overruns, what):
        yield
