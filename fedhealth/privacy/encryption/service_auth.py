"""Who is calling? Service identity on top of mutual TLS.

WHY TLS ALONE IS NOT ENOUGH
    mTLS proves "this connection comes from SOME service with a certificate from our CA". Our ASGI server does
    not hand the client certificate to the application, and we need more than "some service": a hospital must
    accept CONTROL messages ("run round r", "send me your output share") only from the COORDINATOR, and a
    hospital-to-hospital message only from the hospital it claims to be from. Otherwise a curious hospital could
    ask its peers for their output shares and reassemble the exact, un-noised aggregate.

HOW
    Every request carries a short-lived JWT (ES256, 60 s) signed with the SENDER'S OWN certificate key:
        {iss: "hospital-b", aud: "hospital-a", iat, exp, jti}
    The receiver holds every service's certificate (public), checks each was issued by our CA, and verifies the
    signature with the key from the certificate named in `iss`. Nobody can forge a token for another service
    without that service's private key, and a token minted for hospital-a is useless against hospital-c (`aud`).

LIMITS: no replay cache (tokens are only visible inside the mTLS channel, and expire in 60 s); no revocation.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import httpx
import jwt
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from .tls import SERVICES, TlsPaths, load_cert

ALGORITHM = "ES256"
TTL_SECONDS = 60


class ServiceAuthError(Exception):
    pass


class ServiceIdentity:
    """This service's own identity: its id and its private key."""

    def __init__(self, service_id: str, private_key_pem: bytes):
        self.service_id = service_id
        self._key = serialization.load_pem_private_key(private_key_pem, password=None)

    @classmethod
    def load(cls, directory: Path, service_id: str) -> "ServiceIdentity":
        return cls(service_id, TlsPaths(directory, service_id).key.read_bytes())

    def token_for(self, audience: str, ttl: int = TTL_SECONDS) -> str:
        now = int(time.time())
        return jwt.encode({"iss": self.service_id, "aud": audience, "iat": now, "exp": now + ttl, "jti": uuid.uuid4().hex},
                          self._key, algorithm=ALGORITHM)


class ServiceVerifier:
    """Verifies tokens addressed to `own_id` using the certificates of the services we trust."""

    def __init__(self, own_id: str, directory: Path):
        self.own_id = own_id
        ca = load_cert(TlsPaths(directory, own_id).ca)
        self._public_keys = {}
        for svc in SERVICES:
            cert = load_cert(TlsPaths(directory, svc).cert)
            cert.verify_directly_issued_by(ca)                               # must be signed by OUR CA
            cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
            if cn != svc:
                raise ServiceAuthError(f"certificate for {svc} has common name {cn!r}")
            self._public_keys[svc] = cert.public_key()

    def verify(self, token: str) -> str:
        """Return the authenticated sender id, or raise ServiceAuthError."""
        try:
            claimed = jwt.decode(token, options={"verify_signature": False}).get("iss")
            key = self._public_keys.get(claimed)
            if key is None:
                raise ServiceAuthError("unknown issuer")
            claims = jwt.decode(token, key, algorithms=[ALGORITHM], audience=self.own_id,
                                options={"require": ["iss", "aud", "exp", "iat"]})
        except jwt.PyJWTError as exc:
            raise ServiceAuthError(f"invalid service token: {exc}") from exc
        return claims["iss"]


class ServiceAuth(httpx.Auth):
    """httpx auth hook: stamps every outgoing request with a fresh token for ONE destination service."""

    def __init__(self, identity: ServiceIdentity, audience: str):
        self.identity, self.audience = identity, audience

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {self.identity.token_for(self.audience)}"
        yield request
