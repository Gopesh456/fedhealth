"""Per-requester privacy-budget ledger (the "epsilon ledger").

WHY A BUDGET
    Every DP answer leaks a little. Asking many questions - or the same question many times and averaging the
    noise away - leaks more: by sequential composition the total privacy loss is the SUM of the epsilons. A
    requester therefore gets a finite budget; once it is spent, the system refuses further queries.

HOW IT IS KEPT HONEST
    * Amounts are stored as INTEGERS (micro-epsilons, pico-deltas): ten queries of 0.1 sum to exactly 1.0, no
      floating-point drift can let an eleventh through.
    * reserve() is ONE conditional UPDATE  ("add eps to spent IF spent + eps <= total"). SQLite serialises
      writers, so two concurrent requests can never both pass the check: no race, no overspend.
    * Lifecycle  reserved -> committed  (query answered)  or  reserved -> refunded  (query failed before any
      answer was released). A suppressed answer still COMMITS: the refusal itself carries information.
    * Basic (not advanced) composition is used - simple to explain and conservative.
The ledger lives in the coordinator's OWN database, never in a hospital's.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import BigInteger, Float, Integer, String, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from fedhealth.common.sqlite import make_sqlite_engine

MICRO = 1_000_000            # epsilon unit
PICO = 1_000_000_000_000     # delta unit


def to_micro(epsilon: float) -> int:
    return int(round(epsilon * MICRO))


def to_pico(delta: float) -> int:
    return int(round(delta * PICO))


class BudgetExhausted(Exception):
    def __init__(self, requested: float, remaining: float):
        super().__init__(f"privacy budget exhausted: requested epsilon={requested}, remaining={remaining}")
        self.requested, self.remaining = requested, remaining


class LedgerBase(DeclarativeBase):
    pass


class LedgerRow(LedgerBase):
    __tablename__ = "epsilon_ledger"
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    total_micro_eps: Mapped[int] = mapped_column(BigInteger, nullable=False)
    spent_micro_eps: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)    # committed + reserved
    total_pico_delta: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    spent_pico_delta: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class LedgerEntry(LedgerBase):
    __tablename__ = "ledger_entries"
    entry_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    micro_eps: Mapped[int] = mapped_column(BigInteger, nullable=False)
    pico_delta: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(10), nullable=False)       # reserved | committed | refunded
    created_at: Mapped[float] = mapped_column(Float, nullable=False)


@dataclass(frozen=True)
class Balance:
    total_epsilon: float
    spent_epsilon: float
    total_delta: float
    spent_delta: float

    @property
    def remaining_epsilon(self) -> float:
        return round(self.total_epsilon - self.spent_epsilon, 6)

    @property
    def remaining_delta(self) -> float:
        return self.total_delta - self.spent_delta


def make_engine(path: str) -> Engine:
    """SQLite file engine safe for use from several threads (writers wait for the lock instead of failing)."""
    engine = make_sqlite_engine(path)
    LedgerBase.metadata.create_all(engine)
    return engine


class EpsilonLedger:
    def __init__(self, engine: Engine):
        self.engine = engine
        LedgerBase.metadata.create_all(engine)

    # ---- accounts -----------------------------------------------------------------------------------
    def ensure_user(self, user_id: str, total_epsilon: float, total_delta: float = 0.0) -> None:
        """Create the account with its budget if it does not exist yet. NEVER tops up an existing account."""
        with Session(self.engine) as s, s.begin():
            if s.get(LedgerRow, user_id) is None:
                s.add(LedgerRow(user_id=user_id, total_micro_eps=to_micro(total_epsilon), spent_micro_eps=0,
                                total_pico_delta=to_pico(total_delta), spent_pico_delta=0))

    def balance(self, user_id: str) -> Balance:
        with Session(self.engine) as s:
            row = s.get(LedgerRow, user_id)
            if row is None:
                raise KeyError(user_id)
            return Balance(row.total_micro_eps / MICRO, row.spent_micro_eps / MICRO,
                           row.total_pico_delta / PICO, row.spent_pico_delta / PICO)

    # ---- lifecycle ----------------------------------------------------------------------------------
    def reserve(self, user_id: str, epsilon: float, query_id: str, delta: float = 0.0) -> None:
        """Atomically set aside `epsilon` (and `delta`) for a query. Raises BudgetExhausted if it does not fit."""
        eps_u, delta_u = to_micro(epsilon), to_pico(delta)
        if eps_u <= 0:
            raise ValueError("epsilon must be positive")
        with Session(self.engine) as s, s.begin():
            took = s.execute(
                update(LedgerRow)
                .where(LedgerRow.user_id == user_id,
                       LedgerRow.spent_micro_eps + eps_u <= LedgerRow.total_micro_eps,
                       LedgerRow.spent_pico_delta + delta_u <= LedgerRow.total_pico_delta)
                .values(spent_micro_eps=LedgerRow.spent_micro_eps + eps_u,
                        spent_pico_delta=LedgerRow.spent_pico_delta + delta_u)
            ).rowcount
            if took != 1:
                row = s.get(LedgerRow, user_id)
                if row is None:
                    raise KeyError(user_id)
                raise BudgetExhausted(epsilon, (row.total_micro_eps - row.spent_micro_eps) / MICRO)
            s.add(LedgerEntry(query_id=query_id, user_id=user_id, micro_eps=eps_u, pico_delta=delta_u,
                              status="reserved", created_at=time.time()))

    def commit(self, query_id: str) -> None:
        """The query was answered (or deliberately suppressed): the spend is final."""
        with Session(self.engine) as s, s.begin():
            done = s.execute(update(LedgerEntry).where(LedgerEntry.query_id == query_id, LedgerEntry.status == "reserved")
                             .values(status="committed")).rowcount
            if done != 1:
                raise KeyError(f"no reserved entry for {query_id}")

    def refund(self, query_id: str) -> None:
        """The query failed BEFORE anything was released: give the budget back."""
        with Session(self.engine) as s, s.begin():
            entry = s.execute(select(LedgerEntry).where(LedgerEntry.query_id == query_id)).scalar_one_or_none()
            if entry is None or entry.status != "reserved":
                raise KeyError(f"no reserved entry for {query_id}")
            moved = s.execute(update(LedgerEntry).where(LedgerEntry.entry_id == entry.entry_id,
                                                        LedgerEntry.status == "reserved")
                              .values(status="refunded")).rowcount
            if moved == 1:
                s.execute(update(LedgerRow).where(LedgerRow.user_id == entry.user_id)
                          .values(spent_micro_eps=LedgerRow.spent_micro_eps - entry.micro_eps,
                                  spent_pico_delta=LedgerRow.spent_pico_delta - entry.pico_delta))

    def accounts(self) -> list[dict]:
        """Every account's balance (for the auditor's overview)."""
        with Session(self.engine) as s:
            return [{"user_id": r.user_id, "total_epsilon": r.total_micro_eps / MICRO, "spent_epsilon": r.spent_micro_eps / MICRO,
                     "remaining_epsilon": (r.total_micro_eps - r.spent_micro_eps) / MICRO}
                    for r in s.execute(select(LedgerRow).order_by(LedgerRow.user_id)).scalars()]

    def entries(self, user_id: str) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.execute(select(LedgerEntry).where(LedgerEntry.user_id == user_id).order_by(LedgerEntry.entry_id)).scalars()
            return [{"query_id": r.query_id, "epsilon": r.micro_eps / MICRO, "delta": r.pico_delta / PICO,
                     "status": r.status, "created_at": r.created_at} for r in rows]
