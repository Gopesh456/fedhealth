"""Arithmetic in the finite field Z_p, p = 2^61 - 1 (a Mersenne prime).

All secret-shared values live here. "Mod p" arithmetic wraps around, which is exactly what makes a random
share a perfect one-time pad: if r is uniform in Z_p then (secret + r) mod p is uniform too, whatever the
secret is. Counts and sums are tiny compared to p (a cohort has thousands of patients), so they never wrap;
negative sums (a lab value that went down) are represented as p - |x| and recovered with `to_signed`.
"""
from __future__ import annotations

import secrets

P = (1 << 61) - 1
ELEMENT_BYTES = 8          # 61 bits fit in 8 bytes


def rand_element() -> int:
    """Uniform element of Z_p from the OS CSPRNG."""
    return secrets.randbelow(P)


def rand_vector(n: int) -> list[int]:
    return [secrets.randbelow(P) for _ in range(n)]


def to_field(x: int) -> int:
    return x % P


def to_signed(x: int) -> int:
    """Map a field element back to a (possibly negative) integer in (-p/2, p/2]."""
    return x - P if x > P // 2 else x


def vec_add(a: list[int], b: list[int]) -> list[int]:
    return [(x + y) % P for x, y in zip(a, b, strict=True)]


def vec_sub(a: list[int], b: list[int]) -> list[int]:
    return [(x - y) % P for x, y in zip(a, b, strict=True)]


def vec_mul(a: list[int], b: list[int]) -> list[int]:
    """Element-wise product (Hadamard). NOTE: on *shares* this is NOT the product of the secrets -
    that is what Beaver triples are for."""
    return [(x * y) % P for x, y in zip(a, b, strict=True)]


def vec_sum(a: list[int]) -> int:
    return sum(a) % P


# ---- wire encoding: fixed 8-byte big-endian elements -----------------------------------------------
def encode_vector(vec: list[int]) -> bytes:
    return b"".join(v.to_bytes(ELEMENT_BYTES, "big") for v in vec)


def decode_vector(blob: bytes) -> list[int]:
    if len(blob) % ELEMENT_BYTES:
        raise ValueError("blob length is not a multiple of 8")
    out = [int.from_bytes(blob[i:i + ELEMENT_BYTES], "big") for i in range(0, len(blob), ELEMENT_BYTES)]
    if any(v >= P for v in out):
        raise ValueError("value outside the field")
    return out
