"""Authenticated encryption of sensitive database columns: AES-256-GCM.

WHAT THE `cryptography` LIBRARY DOES FOR US
    `AESGCM(key).encrypt(nonce, plaintext, aad)` / `.decrypt(...)` are the AES-256 block cipher in Galois/Counter
    Mode, implemented by OpenSSL. We never implement the cipher ourselves; we implement HOW IT IS USED:

    * KEY      32 random bytes per hospital (the "data-encryption key", DEK), kept OUTSIDE the database file.
    * NONCE    12 fresh random bytes for EVERY value. Reusing a (key, nonce) pair would break GCM, so the nonce is
               random and stored next to the ciphertext. With 96-bit random nonces the birthday bound (~2^32
               values per key) is far above our ~10^5 values.
    * AAD      "associated data" = the column's name (e.g. "A.patients.national_id"). It is authenticated but not
               encrypted: a ciphertext copied into a DIFFERENT column fails to decrypt.
    * TAG      GCM appends a 128-bit authentication tag; any modification of the stored bytes raises InvalidTag, so
               tampering is detected rather than silently producing garbage.

    Stored form:  "v1:" + base64( nonce(12) || ciphertext || tag(16) )

WHAT THIS DOES AND DOES NOT GIVE YOU (see THREAT_MODEL.md)
    + A stolen disk, backup or copy of the .db file reveals nothing about the encrypted columns.
    - It does NOT protect against someone with access to the running service (the key is in memory), nor against
      the hospital's own administrators. Real deployments keep the DEK in a KMS/HSM; here it is a file in secrets/.
    - AAD binds the column but not the ROW: an attacker with write access could swap two ciphertexts of the same
      column between rows without detection.
    - Plaintext columns (birth_year, sex, diagnoses, ...) are needed for filtering and are not encrypted.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

NONCE_BYTES = 12
KEY_BYTES = 32
PREFIX = "v1:"


class DecryptionError(Exception):
    """Ciphertext was modified, belongs to another column, or was made with another key."""


class Vault:
    """Holds one hospital's data-encryption key and encrypts / decrypts column values."""

    def __init__(self, key: bytes):
        if len(key) != KEY_BYTES:
            raise ValueError("AES-256-GCM needs a 32-byte key")
        self._aead = AESGCM(key)

    def encrypt(self, plaintext: str, context: str) -> str:
        nonce = os.urandom(NONCE_BYTES)
        ct = self._aead.encrypt(nonce, plaintext.encode("utf-8"), context.encode("utf-8"))
        return PREFIX + base64.b64encode(nonce + ct).decode("ascii")

    def decrypt(self, token: str, context: str) -> str:
        if not token.startswith(PREFIX):
            raise DecryptionError("not an encrypted value (missing version prefix)")
        raw = base64.b64decode(token[len(PREFIX):])
        nonce, ct = raw[:NONCE_BYTES], raw[NONCE_BYTES:]
        try:
            return self._aead.decrypt(nonce, ct, context.encode("utf-8")).decode("utf-8")
        except InvalidTag as exc:
            raise DecryptionError("authentication failed: wrong key, wrong column, or the data was modified") from exc


class NullVault:
    """Plaintext pass-through. Exists ONLY to build the 'encryption off' control in tests / evaluation."""

    def encrypt(self, plaintext: str, context: str) -> str:
        return plaintext

    def decrypt(self, token: str, context: str) -> str:
        return token


# ------------------------------------------------------------------------------------------- key files
def secrets_dir() -> Path:
    from fedhealth.common.config import repo_root
    env = os.environ.get("FEDHEALTH_SECRETS_DIR")
    return Path(env) if env else repo_root() / "secrets"


def key_path(name: str, directory: Path | None = None) -> Path:
    return (directory or secrets_dir()) / f"{name}.key"


def load_or_create_key(name: str, directory: Path | None = None) -> bytes:
    """Create a random 32-byte key file on first use (generator / scripts/gen_keys.py)."""
    path = key_path(name, directory)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64encode(os.urandom(KEY_BYTES)))
        try:
            path.chmod(0o600)
        except OSError:                                    # best effort (Windows ignores POSIX modes)
            pass
    return load_key(name, directory)


def load_key(name: str, directory: Path | None = None) -> bytes:
    """Load an existing key; NEVER creates one (a service must fail closed if its key is missing)."""
    path = key_path(name, directory)
    if not path.exists():
        raise FileNotFoundError(f"key file {path} not found - run scripts/gen_keys.py")
    return base64.b64decode(path.read_bytes())
