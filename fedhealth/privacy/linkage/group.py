"""The mathematical group used for Diffie-Hellman-style commutative encryption.

WHAT WE NEED
    A group where "raise to a secret power" is easy but undoing it is infeasible without the secret
    (the Decisional Diffie-Hellman assumption), and where  (x^a)^b == (x^b)^a  (commutativity).

THE GROUP
    p = a *safe prime* (p = 2q + 1, q prime). We work in the subgroup of *quadratic residues* mod p,
    which has prime order q. Squaring any number mod p lands in that subgroup, so hash-to-group is
    simply  H(id)^2 mod p  - no rejection sampling needed.

WHERE p COMES FROM
    Rather than paste a 2048-bit constant, we COMPUTE the well-known RFC 3526 / RFC 2409 primes from
    their published definition, so there is nothing to mistype:
        p = 2^n - 2^(n-64) - 1 + 2^64 * ( floor(2^(n-130) * pi) + c )
    with a per-size constant c. (tests/test_psi.py re-verifies that each p is a safe prime.)

    modp2048  RFC 3526 group 14   default; ~112-bit security
    modp1536  RFC 3526 group 5
    test768   RFC 2409 group 1    768 bit - INSECURE, only to keep unit tests fast
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from functools import lru_cache

import os

try:  # optional accelerator: ~7x faster modular exponentiation (pip install gmpy2)
    if os.environ.get("FEDHEALTH_NO_GMPY2"):          # lets the evaluation measure the pure-Python cost
        raise ImportError
    import gmpy2

    def _powmod(base: int, exp: int, mod: int) -> int:
        return int(gmpy2.powmod(base, exp, mod))
    HAVE_GMPY2 = True
except ImportError:  # pragma: no cover - exercised implicitly when gmpy2 is absent
    def _powmod(base: int, exp: int, mod: int) -> int:
        return pow(base, exp, mod)
    HAVE_GMPY2 = False


def _pi_scaled(bits: int) -> int:
    """floor(pi * 2**bits) via Machin's formula  pi = 16*atan(1/5) - 4*atan(1/239)  in integer arithmetic."""
    guard = 64
    one = 1 << (bits + guard)

    def arctan_inv(x: int) -> int:           # atan(1/x) scaled by `one`
        total, term, n, sign = 0, one // x, 1, 1
        x2 = x * x
        while term:
            total += sign * (term // n)
            term //= x2
            n += 2
            sign = -sign
        return total

    return (16 * arctan_inv(5) - 4 * arctan_inv(239)) >> guard


# name -> (bits, c) from RFC 2409 / RFC 3526
_DEFINITIONS = {"test768": (768, 149686), "modp1536": (1536, 741804), "modp2048": (2048, 124476)}


@dataclass(frozen=True)
class Group:
    name: str
    p: int            # safe prime
    q: int            # (p - 1) / 2, the (prime) order of the quadratic-residue subgroup
    nbytes: int       # fixed width of an element on the wire

    # ---- hash an identifier into the subgroup ---------------------------------------------------
    def hash_to_group(self, data: bytes, domain: bytes = b"fedhealth/psi/v1") -> int:
        """Map bytes to a pseudo-random element of the QR subgroup.
        Expand SHA-256 in counter mode to |p|+128 bits (so the reduction mod p is statistically uniform),
        reduce mod p, then SQUARE: squares mod p are exactly the order-q subgroup."""
        need = self.nbytes + 16
        stream, counter = b"", 0
        while len(stream) < need:
            stream += hashlib.sha256(domain + counter.to_bytes(4, "big") + data).digest()
            counter += 1
        h = int.from_bytes(stream[:need], "big") % self.p
        return (h * h) % self.p

    # ---- the commutative cipher -----------------------------------------------------------------
    def new_key(self) -> int:
        """A fresh secret exponent. 256 bits is plenty for a 1536/2048-bit group (cf. RFC 3526 §8) and
        keeps every exponentiation ~8x cheaper than a full-width exponent."""
        while True:
            k = secrets.randbits(256)
            if k > 1:
                return k

    def encrypt(self, element: int, key: int) -> int:
        """E_k(x) = x^k mod p.   E_a(E_b(x)) = x^(ab) = E_b(E_a(x))  (commutes)."""
        return _powmod(element, key, self.p)

    def encode(self, element: int) -> bytes:
        return element.to_bytes(self.nbytes, "big")

    def tag(self, element: int) -> bytes:
        """128-bit fingerprint of a (fully encrypted) element. Used only for equality tests in the final
        comparison, so we ship 16 bytes instead of 256."""
        return hashlib.sha256(self.encode(element)).digest()[:16]


def _compute_prime(bits: int, c: int) -> int:
    return (1 << bits) - (1 << (bits - 64)) - 1 + (1 << 64) * ((_pi_scaled(bits - 130) ) + c)


@lru_cache(maxsize=None)
def get_group(name: str = "modp2048") -> Group:
    if name not in _DEFINITIONS:
        raise ValueError(f"unknown group {name!r}; choose from {sorted(_DEFINITIONS)}")
    bits, c = _DEFINITIONS[name]
    p = _compute_prime(bits, c)
    return Group(name=name, p=p, q=(p - 1) // 2, nbytes=bits // 8)


# ----------------------------------------------------------------------------- wire packing
def pack_ints(values: list[int], nbytes: int) -> bytes:
    """Concatenate fixed-width big-endian integers (one blob instead of thousands of JSON strings)."""
    return b"".join(v.to_bytes(nbytes, "big") for v in values)


def unpack_ints(blob: bytes, nbytes: int) -> list[int]:
    if len(blob) % nbytes:
        raise ValueError("blob length is not a multiple of the element width")
    return [int.from_bytes(blob[i:i + nbytes], "big") for i in range(0, len(blob), nbytes)]
