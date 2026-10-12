"""Append-only, tamper-evident audit trail (a keyed hash chain).

WHAT IS LOGGED   every query (who, what, when, epsilon charged, outcome: released / suppressed / refused / failed /
                 denied), every login attempt, every audit read/verify. The released VALUE is never stored.

HOW IT IS TAMPER-EVIDENT
    Each entry carries   mac_i = HMAC-SHA256( audit_key,  mac_{i-1} || canonical(entry_i) )
    so every entry commits to the ENTIRE history before it (like a blockchain, but with a secret key):
      * edit any field of any entry      -> its mac no longer matches,
      * delete or insert an entry        -> sequence gap / the next entry's prev_mac no longer matches,
      * reorder entries                  -> prev_mac / sequence no longer matches,
      * recompute the chain after editing -> impossible without the audit key (it is NOT stored in the database).
    Two layers stop honest mistakes and casual attackers: SQLite triggers make UPDATE and DELETE on the table fail,
    and verify_chain() recomputes everything from the genesis value.

THE LIMIT (and the answer to it)
    Truncating the END of the log (deleting the newest entries) leaves a perfectly valid, shorter chain. Detecting
    that needs a copy of the head (seq, mac) kept somewhere the attacker cannot reach: head() returns it, and
    verify_chain(expected_head=...) checks it. Someone who steals the audit key can forge a whole new chain.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DDL, Float, Integer, String, Text, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

GENESIS = "0" * 64
OUTCOMES = ("released", "suppressed", "refused", "failed", "denied", "success")


class AuditBase(DeclarativeBase):
    pass


class AuditRow(AuditBase):
    __tablename__ = "audit_log"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[str] = mapped_column(String(32), nullable=False)
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    query_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    query_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hospitals: Mapped[str] = mapped_column(String(16), nullable=False)
    micro_eps: Mapped[int] = mapped_column(BigInteger, nullable=False)
    pico_delta: Mapped[int] = mapped_column(BigInteger, nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    prev_mac: Mapped[str] = mapped_column(String(64), nullable=False)
    mac: Mapped[str] = mapped_column(String(64), nullable=False)


# Append-only at the database level (an attacker with raw file access can drop these - verify_chain still catches edits).
for _name, _op in (("audit_no_update", "UPDATE"), ("audit_no_delete", "DELETE")):
    event.listen(AuditRow.__table__, "after_create",
                 DDL(f"CREATE TRIGGER IF NOT EXISTS {_name} BEFORE {_op} ON audit_log "
                     f"BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;"))


def canonical(row: dict) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _mac(key: bytes, prev_mac: str, body: dict) -> str:
    return hmac.new(key, prev_mac.encode("ascii") + b"|" + canonical(body), hashlib.sha256).hexdigest()


def _body(row: AuditRow) -> dict:
    return {"seq": row.seq, "ts": row.ts, "actor": row.actor, "role": row.role, "action": row.action,
            "query_json": row.query_json, "query_hash": row.query_hash, "hospitals": row.hospitals,
            "micro_eps": row.micro_eps, "pico_delta": row.pico_delta, "outcome": row.outcome, "reason": row.reason}


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    entries: int
    head_seq: int
    head_mac: str
    first_bad_seq: int | None = None
    reason: str | None = None


class AuditLog:
    def __init__(self, engine: Engine, key: bytes):
        if len(key) < 32:
            raise ValueError("audit key must be at least 32 bytes")
        self.engine, self._key = engine, key
        AuditBase.metadata.create_all(engine)

    def append(self, *, actor: str, role: str, action: str, outcome: str, query: dict | None = None,
               hospitals: list[str] | tuple[str, ...] = (), epsilon: float = 0.0, delta: float = 0.0,
               reason: str | None = None) -> int:
        """Append one entry; returns its sequence number. The read-last-then-insert runs inside a BEGIN IMMEDIATE
        transaction, so concurrent writers are serialised and the chain cannot fork."""
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        qjson = json.dumps(query, sort_keys=True, separators=(",", ":")) if query is not None else None
        with Session(self.engine) as s, s.begin():
            last = s.execute(select(AuditRow).order_by(AuditRow.seq.desc()).limit(1)).scalar_one_or_none()
            row = AuditRow(
                seq=(last.seq + 1) if last else 1, ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                actor=actor, role=role, action=action, query_json=qjson,
                query_hash=hashlib.sha256(qjson.encode()).hexdigest() if qjson else None,
                hospitals=",".join(hospitals), micro_eps=int(round(epsilon * 1_000_000)),
                pico_delta=int(round(delta * 1_000_000_000_000)), outcome=outcome, reason=reason,
                prev_mac=last.mac if last else GENESIS, mac="")
            row.mac = _mac(self._key, row.prev_mac, _body(row))
            s.add(row)
            return row.seq

    def head(self) -> tuple[int, str]:
        """(sequence number, mac) of the newest entry - store it somewhere the log's operator cannot rewrite."""
        with Session(self.engine) as s:
            last = s.execute(select(AuditRow).order_by(AuditRow.seq.desc()).limit(1)).scalar_one_or_none()
            return (last.seq, last.mac) if last else (0, GENESIS)

    def verify_chain(self, expected_head: tuple[int, str] | None = None) -> VerifyResult:
        prev, n, seq_expected = GENESIS, 0, 1
        with Session(self.engine) as s:
            for row in s.execute(select(AuditRow).order_by(AuditRow.seq)).scalars():
                n += 1
                if row.seq != seq_expected:
                    return VerifyResult(False, n, row.seq, row.mac, row.seq, f"sequence gap: expected {seq_expected}, found {row.seq} (entry deleted or inserted)")
                if row.prev_mac != prev:
                    return VerifyResult(False, n, row.seq, row.mac, row.seq, "prev_mac does not match the previous entry (entry removed, reordered or replaced)")
                if not hmac.compare_digest(row.mac, _mac(self._key, row.prev_mac, _body(row))):
                    return VerifyResult(False, n, row.seq, row.mac, row.seq, "mac mismatch (entry was modified or forged without the key)")
                prev, seq_expected = row.mac, row.seq + 1
        head = (n, prev)
        if expected_head is not None and expected_head != head:
            return VerifyResult(False, n, head[0], head[1], None,
                                f"head {head} differs from the anchored head {tuple(expected_head)} (log truncated or extended)")
        return VerifyResult(True, n, head[0], head[1])

    def read(self, *, actor: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        with Session(self.engine) as s:
            q = select(AuditRow).order_by(AuditRow.seq.desc()).limit(limit).offset(offset)
            if actor is not None:
                q = q.where(AuditRow.actor == actor)
            return [{"seq": r.seq, "ts": r.ts, "actor": r.actor, "role": r.role, "action": r.action,
                     "query": json.loads(r.query_json) if r.query_json else None, "hospitals": r.hospitals.split(",") if r.hospitals else [],
                     "epsilon_charged": r.micro_eps / 1_000_000, "delta_charged": r.pico_delta / 1_000_000_000_000,
                     "outcome": r.outcome, "reason": r.reason, "mac": r.mac[:16] + "…"}
                    for r in s.execute(q).scalars()]
