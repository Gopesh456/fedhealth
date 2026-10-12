"""Message formats of the privacy protocols (PSI now, MPC in Phase 3).

Two planes, kept deliberately separate:
  * CONTROL plane  coordinator -> hospital    "run round r", "prepare", "send me your output share"
  * DATA plane     hospital -> hospital       encrypted PSI lists, tags, secret shares
The coordinator never receives data-plane payloads; the models below make that visible in the types.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .schemas import Agg, HospitalId, LocalQuery

LINKAGE_ID_PATTERN = r"^[0-9a-f]{8,64}$"


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ PSI control plane
class PsiInit(_Wire):
    ring: list[HospitalId] = Field(min_length=2, max_length=3)
    group: str = "modp2048"


class PsiStep(_Wire):
    round: int = Field(ge=1, le=3)


class PsiFinalizeResult(_Wire):
    m: int            # size of the linked cohort (the one number the coordinator learns)
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
class MpcPrepare(_Wire):
    """Tells a hospital which computation it takes part in. `query` is ITS OWN predicate slice only."""
    parties: list[HospitalId] = Field(min_length=1, max_length=3)    # computing hospitals; order = share index
    owner: HospitalId                                                # holds the output vectors
    agg: Agg
    t_outputs: int = Field(ge=1, le=64)                              # number of output cells (public query shape)
    m: int = Field(ge=0)                                             # linked-cohort size (0 when only one hospital)
    lid: str | None = Field(default=None, pattern=LINKAGE_ID_PATTERN)
    query: LocalQuery


class MpcTriples(_Wire):
    """This party's shares of the dealer's Beaver triples (base64 of 8-byte field elements)."""
    x: str
    y: str
    z: str


class MpcLayer(_Wire):
    layer: int = Field(ge=0, le=8)


class MpcOutput(_Wire):
    """The ONLY thing a hospital returns to the coordinator at the end: one additive share per output cell.
    On its own each share is uniformly random (for >= 2 hospitals)."""
    shares: list[int]


# ------------------------------------------------------------------ MPC data plane (hospital -> hospital)
class MpcInShare(_Wire):
    sender: HospitalId
    vectors: list[str]        # one base64 blob per vector the sender owns (this recipient's shares of them)


class MpcInOpen(_Wire):
    sender: HospitalId
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
