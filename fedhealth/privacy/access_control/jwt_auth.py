"""JSON Web Tokens for requesters, and password hashing.

TOKENS
    After a successful login the coordinator issues a JWT:  {sub: user, role, iss, iat, exp}  signed with
    HMAC-SHA256 under a secret only the coordinator knows. Every API call presents it as `Authorization: Bearer`.
    The coordinator is the ONLY party that issues and checks these tokens, so a shared-secret (HS256) signature is
    the right tool (the hospitals never see user tokens; they authenticate SERVICES, see service_auth.py).

    Defences worth naming:
      * the accepted algorithm is pinned to HS256 ("alg: none" and algorithm-confusion tokens are rejected),
      * exp / iat / iss / sub / role are REQUIRED claims, expiry is enforced, tokens are short-lived (30 min),
      * the secret must be >= 32 bytes.
    Not done: revocation lists / refresh tokens (a stolen token is valid until it expires).

PASSWORDS
    Stored as scrypt hashes (memory-hard, random 16-byte salt per user), compared in constant time.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from dataclasses import dataclass

import jwt

ALGORITHM = "HS256"
ISSUER = "fedhealth-coordinator"


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Claims:
    sub: str
    role: str
    exp: int


class TokenAuthority:
    def __init__(self, secret: bytes, ttl_seconds: int = 1800):
        if len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 bytes")
        self._secret, self.ttl = secret, ttl_seconds

    def issue(self, sub: str, role: str, now: float | None = None) -> str:
        t = int(now if now is not None else time.time())
        return jwt.encode({"sub": sub, "role": role, "iss": ISSUER, "iat": t, "exp": t + self.ttl},
                          self._secret, algorithm=ALGORITHM)

    def verify(self, token: str) -> Claims:
        try:
            c = jwt.decode(token, self._secret, algorithms=[ALGORITHM], issuer=ISSUER,
                           options={"require": ["exp", "iat", "iss", "sub", "role"]})
        except jwt.PyJWTError as exc:
            raise AuthError(str(exc)) from exc
        return Claims(c["sub"], c["role"], c["exp"])


# ----------------------------------------------------------------------------- passwords
_SCRYPT = dict(n=2 ** 14, r=8, p=1)


def hash_password(password: str, **params) -> str:
    p = {**_SCRYPT, **params}
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, **p)
    return "scrypt${n}${r}${p}${salt}${dk}".format(**p, salt=base64.b64encode(salt).decode(), dk=base64.b64encode(dk).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, dk = stored.split("$")
        expected = base64.b64decode(dk)
        got = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, expected)
