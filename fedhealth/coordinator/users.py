"""Requester accounts (in the coordinator's own database) and a small login throttle."""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from sqlalchemy import String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from fedhealth.privacy.access_control.jwt_auth import hash_password, verify_password


class UserBase(DeclarativeBase):
    pass


class UserRow(UserBase):
    __tablename__ = "users"
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    pw_hash: Mapped[str] = mapped_column(String(256), nullable=False)


class UserStore:
    def __init__(self, engine: Engine):
        self.engine = engine
        UserBase.metadata.create_all(engine)
        self._dummy_hash = hash_password("dummy-password-for-timing")

    def create(self, user_id: str, role: str, password: str | None = None, pw_hash: str | None = None) -> None:
        """Create (or replace the role/password of) a user. Pass a plaintext password OR an existing hash."""
        with Session(self.engine) as s, s.begin():
            row = s.get(UserRow, user_id)
            h = pw_hash or hash_password(password)
            if row is None:
                s.add(UserRow(user_id=user_id, role=role, pw_hash=h))
            else:
                row.role, row.pw_hash = role, h

    def seed_from_file(self, path: Path) -> int:
        """Load {user_id, role, pw_hash} records (written by scripts/gen_keys.py). Hashes only - no plaintext."""
        if not path.exists():
            return 0
        records = json.loads(path.read_text(encoding="utf-8"))
        for r in records:
            self.create(r["user_id"], r["role"], pw_hash=r["pw_hash"])
        return len(records)

    def authenticate(self, user_id: str, password: str) -> UserRow | None:
        """Return the user on success. Always does one scrypt verification, so an unknown user costs the same
        time as a wrong password (no user enumeration through timing)."""
        with Session(self.engine) as s:
            row = s.execute(select(UserRow).where(UserRow.user_id == user_id)).scalar_one_or_none()
        ok = verify_password(password, row.pw_hash if row else self._dummy_hash)
        return row if (row and ok) else None


class LoginThrottle:
    """At most `max_failures` failed logins per user within `window` seconds; further attempts are refused until
    the oldest failure ages out. In-memory (single coordinator process)."""

    def __init__(self, max_failures: int = 5, window: float = 60.0, clock=time.monotonic):
        self.max_failures, self.window, self.clock = max_failures, window, clock
        self._fails: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, user: str) -> deque[float]:
        q = self._fails[user]
        while q and self.clock() - q[0] > self.window:
            q.popleft()
        return q

    def blocked(self, user: str) -> bool:
        with self._lock:
            return len(self._trim(user)) >= self.max_failures

    def record_failure(self, user: str) -> None:
        with self._lock:
            self._trim(user).append(self.clock())

    def reset(self, user: str) -> None:
        with self._lock:
            self._fails.pop(user, None)
