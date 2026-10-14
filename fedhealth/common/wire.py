"""Message formats of the privacy protocols (PSI now, MPC in Phase 3).

Two planes, kept deliberately separate:
  * CONTROL plane  coordinator -> hospital    "run round r", "prepare", "send me your output share"
  * DATA plane     hospital -> hospital       encrypted PSI lists, tags, secret shares
The coordinator never receives data-plane payloads; the models below make that visible in the types.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schemas import Agg, HospitalId, LocalQuery

LINKAGE_ID_PATTERN = r"^[0-9a-f]{8,64}$"
SESSION_DIGEST_PATTERN = r"^[0-9a-f]{32}$"


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _distinct(v: list) -> list:
    """A hospital may appear once: listed twice it could, e.g., be both dealer and computer of one computation."""
    if len(set(v)) != len(v):
        raise ValueError("a hospital is listed more than once")
    return v


# ------------------------------------------------------------------ PSI control plane
class PsiInit(_Wire):
    ring: list[HospitalId] = Field(min_length=2, max_length=3)
    group: str = "modp2048"

    _ring_distinct = field_validator("ring")(_distinct)


class PsiStep(_Wire):
    round: int = Field(ge=1, le=3)


class PsiFinalizeResult(_Wire):
    size: int         # length of the linked index = the PUBLIC padding bound, never the true cohort size m_S
    digest: str       # hash of the common index; lets the coordinator check all hospitals agree


# ------------------------------------------------------------------ PSI data plane (hospital -> hospital)
class PsiInbox(_Wire):
    sender: HospitalId
    round: int = Field(ge=1, le=3)
    blob: str         # base64 of concatenated fixed-width group elements


class PsiTags(_Wire):
    sender: HospitalId
    blob: str         # base64 of concatenated 16-byte tags


# ------------------------------------------------------------------ INSECURE linked baseline (dev/eval only)
# ------------------------------------------------------------------ MPC control plane (coordinator -> hospital)
class MpcNoiseCell(_Wire):
    sensitivity: int = Field(ge=1)
    s: int = Field(default=1, ge=1)          # laplace: total noise P(x) ~ exp(-|x| s/t)
    t: int = Field(default=1, ge=1)
    sigma2: str = "0"                        # gaussian: per-share variance, an exact fraction "p/q"


class MpcNoise(_Wire):
    """The public noise plan (privacy/dp/distributed.py). Every involved hospital re-checks it against its own cap,
    samples its OWN share from it and feeds that share into the computation."""
    mechanism: Literal["laplace", "gaussian"]
    contributors: int = Field(ge=1, le=3)
    honest: int = Field(ge=1, le=3)
    cells: list[MpcNoiseCell] = Field(min_length=1, max_length=64)
    groups: list[tuple[int, int]] = Field(min_length=1, max_length=4)
    epsilon: float = Field(ge=0)
    delta: float = Field(ge=0)
    rho: float = Field(ge=0)


class MpcPrepare(_Wire):
    """Tells a hospital which computation it takes part in. `query` is ITS OWN predicate slice only (None for a dealer
    that holds no input)."""
    parties: list[HospitalId] = Field(min_length=1, max_length=3)    # input hospitals, ring order
    owner: HospitalId                                                # holds the output vectors
    computers: list[HospitalId] = Field(default_factory=list, max_length=2)   # P0, P1 (empty for one hospital)
    dealer: HospitalId | None = None                                 # makes the triples
    agg: Agg
    t_outputs: int = Field(ge=1, le=64)                              # number of output cells (public query shape)
    m: int = Field(ge=0, le=1_000_000)                               # length of the padded linked index (0: one hospital)
    lid: str | None = Field(default=None, pattern=LINKAGE_ID_PATTERN)
    query: LocalQuery | None = None
    noise: MpcNoise | None = None                                    # None = exact (insecure evaluation route only)

    _parties_distinct = field_validator("parties", "computers")(_distinct)

    def header_digest(self, sid: str) -> str:
        """Hash of everything every involved hospital must have been told IDENTICALLY (all but its own query slice).
        It travels in every hospital-to-hospital message of the computation, and the computers refuse a mismatch, so
        the coordinator cannot give different hospitals different plans, shapes or roles."""
        header = self.model_dump(mode="json", exclude={"query"})
        blob = json.dumps({"sid": sid, **header}, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(b"fedhealth/mpc-header/v1|" + blob).hexdigest()[:32]


class MpcLayer(_Wire):
    layer: int = Field(ge=0, le=8)


class MpcOutput(_Wire):
    """The ONLY thing a computing hospital returns to the coordinator at the end: one additive share per output cell
    of  exact + noise. On its own each share is uniformly random (for >= 2 hospitals)."""
    shares: list[int]


# ------------------------------------------------------------------ MPC data plane (hospital -> hospital)
class MpcInShare(_Wire):
    sender: HospitalId
    session: str = Field(pattern=SESSION_DIGEST_PATTERN)   # MpcPrepare.header_digest: binds the message to one header
    vectors: list[str]        # one base64 blob per vector the sender owns (this recipient's shares of them)
    noise: str | None = None  # this recipient's share of the sender's noise vector


class MpcInTriples(_Wire):
    """Dealer -> computer. P0 gets a 32-byte seed it expands itself; P1 gets its shares explicitly."""
    sender: HospitalId
    session: str = Field(pattern=SESSION_DIGEST_PATTERN)   # MpcPrepare.header_digest: binds the message to one header
    seed: str | None = None   # base64
    x: str | None = None
    y: str | None = None
    z: str | None = None


class MpcInOpen(_Wire):
    sender: HospitalId
    session: str = Field(pattern=SESSION_DIGEST_PATTERN)   # MpcPrepare.header_digest: binds the message to one header
    layer: int = Field(ge=0, le=8)
    d: str                    # sender's share of  a - x
    e: str                    # sender's share of  b - y


# ------------------------------------------------------------------ federated learning (FedAvg)
class FlTrain(_Wire):
    """Control plane: the current global weights + hyper-parameters. The ONLY thing a hospital sends back is a
    clipped (optionally noised) weight UPDATE or an additive share of it - never a row, a label or a feature."""
    weights: list[float] = Field(min_length=1, max_length=256)
    parties: list[HospitalId] = Field(min_length=1, max_length=3)
    round_no: int = Field(ge=1, le=1000)
    epochs: int = Field(ge=1, le=20)
    lr: float = Field(gt=0, le=2)
    batch: int = Field(ge=8, le=512)
    l2: float = Field(ge=0, le=1)
    clip: float | None = Field(default=None, gt=0, le=100)
    noise_multiplier: float = Field(default=0.0, ge=0, le=100)
    seed: int = Field(default=0, ge=0)


class FlUpdate(_Wire):
    update: list[float]


class FlInShare(_Wire):
    sender: HospitalId
    blob: str                 # base64 of 8-byte field elements: this recipient's additive share of the sender's update


class FlSumShare(_Wire):
    share: list[int]          # the sum of all shares this hospital holds: uniformly random on its own


class LinkedMatchRequest(_Wire):
    query: LocalQuery


class LinkedRow(BaseModel):
    idx: int                  # position in the common linked index (pseudonymous, but per-patient!)
    value: int | None = None
    category: str | None = None


class LinkedMatchResponse(BaseModel):
    hospital: HospitalId
    rows: list[LinkedRow]
