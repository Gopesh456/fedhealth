"""SQLAlchemy column type for sensitive text columns: transparent AES-GCM encryption at rest.

    national_id: Mapped[str] = mapped_column(SensitiveStr(context="A.patients.national_id"))

Application code reads and writes plain strings; the database file only ever contains ciphertext.

WHERE DOES THE KEY COME FROM?
    Three hospitals run in ONE process in the tests, each with its own key, so the key cannot be a global.
    SQLAlchemy hands every type its engine's `dialect` object, and each engine has its own dialect instance,
    so `attach_vault(engine, vault)` parks the hospital's Vault on its engine's dialect and the type finds it
    there. No vault attached => RuntimeError: the column type FAILS CLOSED instead of storing plaintext.
"""
from __future__ import annotations

from sqlalchemy.engine import Engine
from sqlalchemy.types import Text, TypeDecorator

from .aesgcm import NullVault, Vault

_ATTR = "fedhealth_vault"


def attach_vault(engine: Engine, vault: Vault | NullVault) -> Engine:
    setattr(engine.dialect, _ATTR, vault)
    return engine


def _vault(dialect) -> Vault | NullVault:
    vault = getattr(dialect, _ATTR, None)
    if vault is None:
        raise RuntimeError("no data-encryption key attached to this database engine "
                           "(refusing to read or write sensitive columns in plaintext)")
    return vault


class SensitiveStr(TypeDecorator):
    impl = Text
    cache_ok = True

    def __init__(self, *, context: str):
        super().__init__()
        self.context = context          # authenticated as AES-GCM "associated data": binds ciphertext to its column

    def process_bind_param(self, value, dialect):          # Python -> database
        return None if value is None else _vault(dialect).encrypt(value, self.context)

    def process_result_value(self, value, dialect):        # database -> Python
        return None if value is None else _vault(dialect).decrypt(value, self.context)
