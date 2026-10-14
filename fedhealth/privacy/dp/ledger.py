"""Per-requester privacy-budget ledger.

WHY A BUDGET
    Every DP answer leaks a little. Asking many questions - or the same question many times and averaging the
    noise away - leaks more. A requester therefore gets a finite (epsilon, delta) budget; once it is spent, the
    system refuses further queries.

TWO ACCOUNTANTS, ONE BUDGET  (see accounting.py; the guarantee is Theorem 2 of SECURITY_THEOREMS.md)
    * zCDP:  every query is charged rho (Laplace: epsilon^2/2; Gaussian: Delta^2/(2 sigma^2) + the distributed-noise
             term). rho adds up EXACTLY; the budget (E, D) becomes the cap  rho_cap = the largest rho whose
             conversion to (epsilon, D)-DP gives epsilon <= E. This is what lets a researcher ask many small
             questions (O(sqrt(k)) growth instead of O(k)), and it is the ONLY route for Gaussian (delta > 0) queries.
    * basic: sum(epsilon) over PURE-DP (Laplace) queries - tighter for a handful of large queries, and the only
             option when the budget's delta is 0. It stops counting as soon as one delta > 0 query was admitted.
    A query is admitted if, after it, EITHER account is still within the budget. Each account on its own is a valid
    privacy filter (adaptively chosen queries; Rogers et al. 2016 for basic, Feldman-Zrnic 2021 / Whitehouse et al.
    2023 for zCDP). The admitted transcripts split into those that end inside the zCDP cap and those that end inside
    the basic budget, and the two filters agree on each part, so the deltas add: D (zCDP) + 0 (pure basic) = D.
    `spent_epsilon` reports the tighter of the two for display; the guarantee is the budget (E, D).

HOW IT IS KEPT HONEST
    * Amounts are stored as INTEGERS (micro-epsilons, pico-deltas, nano-rhos; charges rounded UP, the rho cap DOWN):
      ten queries of 0.1 sum to exactly 1.0, no floating-point drift can let an eleventh through.
    * reserve() is ONE conditional UPDATE ("add the charge IF either account still fits"). SQLite serialises
      writers, so two concurrent requests can never both pass the check: no race, no overspend.
    * Lifecycle  reserved -> committed  (query answered)  or  reserved -> refunded  (query failed before any
      answer was released). A suppressed answer still COMMITS: the suppression decision is part of the release.
The ledger lives in the coordinator's OWN database, never in a hospital's.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from sqlalchemy import BigInteger, Float, Integer, String, or_, and_, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from fedhealth.common.sqlite import make_sqlite_engine
from fedhealth.privacy.dp.accounting import dp_to_zcdp, pure_dp_rho, zcdp_to_dp

MICRO = 1_000_000            # epsilon unit
PICO = 1_000_000_000_000     # delta unit
NANO = 1_000_000_000         # rho unit


def to_micro(epsilon: float) -> int:
    return int(round(epsilon * MICRO))


def to_pico(delta: float) -> int:
    return int(round(delta * PICO))


def rho_charge(rho: float) -> int:
    return math.ceil(rho * NANO - 1e-6)               # rounded UP (the 1e-6 absorbs float noise in exact decimals)


def rho_cap(total_epsilon: float, total_delta: float) -> int:
    """The zCDP budget equivalent to (total_epsilon, total_delta); 0 when delta = 0 (zCDP cannot give pure DP)."""
    if total_epsilon <= 0 or total_delta <= 0:
        return 0
    return math.floor(dp_to_zcdp(total_epsilon, total_delta) * NANO)


def default_rho(epsilon: float, delta: float) -> float:
    """The zCDP cost of a query known only by its (epsilon, delta): pure DP -> eps^2/2, else the conversion."""
    return pure_dp_rho(epsilon) if delta == 0 else dp_to_zcdp(epsilon, delta)


class BudgetExhausted(Exception):
    def __init__(self, requested: float, remaining: float):
        super().__init__(f"privacy budget exhausted: requested epsilon={requested}, remaining={remaining}")
        self.requested, self.remaining = requested, remaining


class LedgerBase(DeclarativeBase):
    pass


class LedgerRow(LedgerBase):
    __tablename__ = "privacy_accounts"
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    total_micro_eps: Mapped[int] = mapped_column(BigInteger, nullable=False)
    spent_micro_eps: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)    # basic: committed + reserved
    total_pico_delta: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    spent_pico_delta: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    cap_nano_rho: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)        # zCDP equivalent of the budget
    spent_nano_rho: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class LedgerEntry(LedgerBase):
    __tablename__ = "privacy_entries"
    entry_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    micro_eps: Mapped[int] = mapped_column(BigInteger, nullable=False)
    pico_delta: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    nano_rho: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(10), nullable=False)       # reserved | committed | refunded
    created_at: Mapped[float] = mapped_column(Float, nullable=False)


@dataclass(frozen=True)
class Balance:
    total_epsilon: float
    total_delta: float
    basic_epsilon: float           # sum of epsilons (basic composition)
    basic_delta: float
    rho: float                     # sum of rhos (zCDP)
    rho_cap: float

    @property
    def zcdp_epsilon(self) -> float:
        """sum(rho) converted to epsilon at the budget's delta (inf if the budget has no delta)."""
        if self.rho == 0:
            return 0.0
        return zcdp_to_dp(self.rho, self.total_delta) if self.total_delta > 0 else math.inf

    @property
    def basic_valid(self) -> bool:
        """Basic composition is used for pure-DP transcripts only: once a delta > 0 query is in, zCDP decides."""
        return self.basic_delta == 0

    @property
    def accountant(self) -> str:
        return "basic" if self.basic_valid and self.basic_epsilon <= self.zcdp_epsilon else "zcdp"

    @property
    def spent_epsilon(self) -> float:
        """The epsilon of the guarantee given so far: the tighter of the two accountants."""
        return round(self.basic_epsilon if self.accountant == "basic" else self.zcdp_epsilon, 6)

    @property
    def spent_delta(self) -> float:
        return 0.0 if self.accountant == "basic" else self.total_delta

    @property
    def remaining_epsilon(self) -> float:
        return round(max(self.total_epsilon - self.spent_epsilon, 0.0), 6)

    @property
    def remaining_delta(self) -> float:
        return self.total_delta - self.spent_delta

    @property
    def remaining_rho(self) -> float:
        return max(self.rho_cap - self.rho, 0.0)


def _balance(row: "LedgerRow") -> Balance:
    return Balance(row.total_micro_eps / MICRO, row.total_pico_delta / PICO, row.spent_micro_eps / MICRO,
                   row.spent_pico_delta / PICO, row.spent_nano_rho / NANO, row.cap_nano_rho / NANO)


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
                                total_pico_delta=to_pico(total_delta), spent_pico_delta=0,
                                cap_nano_rho=rho_cap(total_epsilon, total_delta), spent_nano_rho=0))

    def balance(self, user_id: str) -> Balance:
        with Session(self.engine) as s:
            row = s.get(LedgerRow, user_id)
            if row is None:
                raise KeyError(user_id)
            return _balance(row)

    # ---- lifecycle ----------------------------------------------------------------------------------
    def reserve(self, user_id: str, epsilon: float, query_id: str, delta: float = 0.0, rho: float | None = None) -> None:
        """Atomically set aside a query's charge in BOTH accounts. Raises BudgetExhausted unless at least one of
        them still fits the budget afterwards (basic: only while every query is pure DP). `rho` defaults to the
        zCDP cost implied by (epsilon, delta)."""
        eps_u, delta_u = to_micro(epsilon), to_pico(delta)
        rho_u = rho_charge(default_rho(epsilon, delta) if rho is None else rho)
        if eps_u <= 0:
            raise ValueError("epsilon must be positive")
        with Session(self.engine) as s, s.begin():
            fits_basic = and_(LedgerRow.spent_micro_eps + eps_u <= LedgerRow.total_micro_eps,
                              LedgerRow.spent_pico_delta + delta_u == 0)                   # pure-DP transcripts only
            fits_zcdp = LedgerRow.spent_nano_rho + rho_u <= LedgerRow.cap_nano_rho
            took = s.execute(
                update(LedgerRow)
                .where(LedgerRow.user_id == user_id, or_(fits_basic, fits_zcdp))
                .values(spent_micro_eps=LedgerRow.spent_micro_eps + eps_u,
                        spent_pico_delta=LedgerRow.spent_pico_delta + delta_u,
                        spent_nano_rho=LedgerRow.spent_nano_rho + rho_u)
            ).rowcount
            if took != 1:
                row = s.get(LedgerRow, user_id)
                if row is None:
                    raise KeyError(user_id)
                raise BudgetExhausted(epsilon, _balance(row).remaining_epsilon)
            s.add(LedgerEntry(query_id=query_id, user_id=user_id, micro_eps=eps_u, pico_delta=delta_u, nano_rho=rho_u,
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
                                  spent_pico_delta=LedgerRow.spent_pico_delta - entry.pico_delta,
                                  spent_nano_rho=LedgerRow.spent_nano_rho - entry.nano_rho))

    def accounts(self) -> list[dict]:
        """Every account's balance (for the auditor's overview)."""
        with Session(self.engine) as s:
            out = []
            for r in s.execute(select(LedgerRow).order_by(LedgerRow.user_id)).scalars():
                b = _balance(r)
                out.append({"user_id": r.user_id, "total_epsilon": b.total_epsilon, "spent_epsilon": b.spent_epsilon,
                            "remaining_epsilon": b.remaining_epsilon, "rho_spent": b.rho, "rho_cap": b.rho_cap,
                            "accountant": b.accountant})
            return out

    def entries(self, user_id: str) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.execute(select(LedgerEntry).where(LedgerEntry.user_id == user_id).order_by(LedgerEntry.entry_id)).scalars()
            return [{"query_id": r.query_id, "epsilon": r.micro_eps / MICRO, "delta": r.pico_delta / PICO,
                     "rho": r.nano_rho / NANO, "status": r.status, "created_at": r.created_at} for r in rows]
