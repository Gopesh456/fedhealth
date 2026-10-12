"""Federated averaging (FedAvg) with optional update clipping, Gaussian noise and secure aggregation.

THE IDEA
    Train one model on data that never leaves its owner. Each ROUND:
      1. the server sends the current global weights w to every hospital;
      2. each hospital trains locally for a few epochs on ITS OWN rows, starting from w, and returns only the CHANGE
         delta_h = w_h - w   (a vector of ~50 floats);
      3. the server averages the deltas:  w <- w + mean(delta_h).
    Raw rows, labels and features never leave a hospital; only these weight vectors do.

TWO OPTIONAL ADD-ONS
    * CLIP + NOISE (differential privacy for the update). Each hospital clips its update to L2 norm <= C (bounding how
      much one hospital can move the model) and adds Gaussian noise N(0, (z*C)^2 / n) per coordinate; the n noises sum
      to std z*C on the SUM of updates. Accounting (zCDP): one round costs rho = 1/(2 z^2); R rounds cost R*rho;
      epsilon(delta) = rho_total + 2*sqrt(rho_total * ln(1/delta)).
      HONEST LIMIT: this protects the contribution of a whole HOSPITAL (client-level DP). With only 3 clients that is a
      weak notion; record-level protection (DP-SGD with per-example clipping) is NOT implemented.
    * SECURE AGGREGATION. Hospitals additively secret-share their (fixed-point) updates, so the server learns only the SUM of
      the updates, never one hospital's update - the same sharing primitive as the query protocol.

WHAT THIS DOES NOT PROTECT: a trained model can still memorise or leak (membership inference, model inversion); updates
are not verified (a malicious hospital can poison the model); plain FedAvg reveals each hospital's update to the server.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from fedhealth.privacy.smpc.field import P, to_signed
from fedhealth.privacy.smpc.sharing import reconstruct_vector, share_vector

FIXED_POINT = 1 << 24            # secure aggregation works on integers: round(update * 2^24)


@dataclass(frozen=True)
class FedAvgConfig:
    rounds: int = 20
    local_epochs: int = 2
    lr: float = 0.1
    batch: int = 64
    l2: float = 1e-3
    clip: float | None = None            # L2 bound on each hospital's update (None = no clipping)
    noise_multiplier: float = 0.0        # z: noise std on the SUM of updates is z * clip
    secure_agg: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.noise_multiplier > 0 and not self.clip:
            raise ValueError("noise needs a clipping bound: sensitivity = clip")


class FedClient(Protocol):
    def local_update(self, weights: np.ndarray, cfg: FedAvgConfig, n_parties: int, round_no: int) -> np.ndarray: ...


# ----------------------------------------------------------------------------- helpers shared with the networked version
def clip_update(delta: np.ndarray, clip: float | None) -> np.ndarray:
    if not clip:
        return delta
    norm = float(np.linalg.norm(delta))
    return delta * min(1.0, clip / norm) if norm > 0 else delta


def add_local_noise(delta: np.ndarray, cfg: FedAvgConfig, n_parties: int, rng: np.random.Generator) -> np.ndarray:
    """Distributed noise: n hospitals each add N(0, (z*C)^2/n), so their sum carries N(0, (z*C)^2)."""
    if cfg.noise_multiplier <= 0:
        return delta
    return delta + rng.normal(0.0, cfg.noise_multiplier * cfg.clip / math.sqrt(n_parties), size=delta.shape)


def quantize(v: np.ndarray) -> list[int]:
    return [int(round(x * FIXED_POINT)) % P for x in v]


def dequantize(values: list[int]) -> np.ndarray:
    return np.array([to_signed(x) / FIXED_POINT for x in values])


def secure_sum(updates: list[np.ndarray]) -> np.ndarray:
    """Secure aggregation, simulated in one process: every update is split into n additive shares, party j adds the
    shares addressed to it, and ONLY those n sums are combined - no single update is ever reconstructed."""
    n = len(updates)
    shared = [share_vector(quantize(u), n) for u in updates]                         # shared[i][j] = share of update i for party j
    party_sums = [[sum(col) % P for col in zip(*[shared[i][j] for i in range(n)])] for j in range(n)]
    return dequantize(reconstruct_vector(party_sums))


def zcdp_epsilon(noise_multiplier: float, rounds: int, delta: float = 1e-6) -> float:
    """(epsilon, delta) of `rounds` Gaussian-mechanism rounds with noise multiplier z, via zCDP composition."""
    if noise_multiplier <= 0:
        return math.inf
    rho = rounds / (2.0 * noise_multiplier ** 2)
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


# ----------------------------------------------------------------------------- in-process reference implementation
@dataclass
class LocalClient:
    """A hospital's side of FedAvg for the in-process reference run (the services run the same steps)."""
    X: np.ndarray
    y: np.ndarray
    name: str = ""

    def local_update(self, weights: np.ndarray, cfg: FedAvgConfig, n_parties: int, round_no: int) -> np.ndarray:
        from .logreg import train_local
        rng = np.random.default_rng([cfg.seed, round_no, zlib.crc32(self.name.encode())])       # stable across processes
        w = train_local(weights, self.X, self.y, epochs=cfg.local_epochs, lr=cfg.lr, batch=cfg.batch, l2=cfg.l2, rng=rng)
        return add_local_noise(clip_update(w - weights, cfg.clip), cfg, n_parties, rng)


@dataclass
class FedAvgResult:
    weights: np.ndarray
    history: list[dict] = field(default_factory=list)
    epsilon: float | None = None


def run_fedavg(clients: list[FedClient], dim: int, cfg: FedAvgConfig,
               evaluate: Callable[[np.ndarray], dict] | None = None, delta: float = 1e-6) -> FedAvgResult:
    w = np.zeros(dim)
    hist = []
    for r in range(1, cfg.rounds + 1):
        updates = [c.local_update(w, cfg, len(clients), r) for c in clients]
        total = secure_sum(updates) if cfg.secure_agg else np.sum(updates, axis=0)
        w = w + total / len(clients)
        hist.append({"round": r, **(evaluate(w) if evaluate else {})})
    eps = zcdp_epsilon(cfg.noise_multiplier, cfg.rounds, delta) if cfg.noise_multiplier > 0 else None
    return FedAvgResult(w, hist, eps)
