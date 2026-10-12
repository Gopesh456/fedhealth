"""The coordinator in its second role: OFFLINE DEALER of Beaver triples (see privacy/smpc/beaver.py).

The dealer generates random multiplication triples (x, y, z = x*y) and gives every computing hospital an
additive share of each. The triples are independent of any patient data - they are one-time pads for the
multiplications. The dealer must NEVER see the values the hospitals open to each other (d = a - x, e = b - y);
that is why those openings travel hospital-to-hospital and never through the coordinator.

TRUST ASSUMPTION (see THREAT_MODEL.md): the dealer does not collude with a computing hospital. A dealer that
knew x and also saw d would recover a = d + x.
"""
from __future__ import annotations

import base64

from fedhealth.common.wire import MpcTriples
from fedhealth.privacy.smpc.field import encode_vector
from fedhealth.privacy.smpc.protocol import Program, deal


def deal_triples(program: Program, parties: list[str]) -> dict[str, MpcTriples]:
    def enc(vec: list[int]) -> str:
        return base64.b64encode(encode_vector(vec)).decode("ascii")
    return {p: MpcTriples(x=enc(x), y=enc(y), z=enc(z)) for p, (x, y, z) in deal(program, parties).items()}
