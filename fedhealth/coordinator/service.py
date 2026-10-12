"""The private query pipeline as one transaction:

    reserve epsilon  ->  PSI linkage + secret-shared computation  ->  k-threshold + DP noise  ->  commit epsilon

* The budget is reserved BEFORE any hospital is contacted, so an exhausted requester costs the hospitals nothing
  (and cannot learn anything from timing or traffic).
* If anything fails before a result exists, the reservation is REFUNDED. Once a number has been produced the
  spend is committed - even when the answer is suppressed, because the refusal is itself a (small) disclosure.
* Authentication, roles and the audit trail wrap this service in Phase 5.
"""
from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass, field
from typing import Literal

from fedhealth.common.schemas import QuerySpec, participating_hospitals
from fedhealth.coordinator.orchestrator import Orchestrator
from fedhealth.coordinator.release import PrivacyParams, Released, ReleasePolicy, exact_from_cells, release
from fedhealth.privacy.dp.ledger import BudgetExhausted, EpsilonLedger

Status = Literal["released", "suppressed", "refused", "failed"]


@dataclass
class Outcome:
    query_id: str
    status: Status
    hospitals: list[str]
    released: Released | None = None
    reason: str | None = None
    epsilon_charged: float = 0.0
    delta_charged: float = 0.0
    epsilon_remaining: float | None = None
    timings: dict[str, float] = field(default_factory=dict)


class QueryService:
    def __init__(self, orchestrator: Orchestrator, ledger: EpsilonLedger, policy: ReleasePolicy = ReleasePolicy(),
                 randbelow=secrets.randbelow):
        self.orchestrator, self.ledger, self.policy, self.randbelow = orchestrator, ledger, policy, randbelow

    def execute(self, user_id: str, spec: QuerySpec, privacy: PrivacyParams) -> Outcome:
        query_id = uuid.uuid4().hex
        hospitals = participating_hospitals(spec)

        try:                                                    # 1. budget first - nothing else happens without it
            self.ledger.reserve(user_id, privacy.epsilon, query_id, privacy.delta)
        except BudgetExhausted as exc:
            return Outcome(query_id, "refused", hospitals, reason="privacy budget exhausted",
                           epsilon_remaining=exc.remaining)

        try:                                                    # 2. private computation + 3. release stage
            run = self.orchestrator.mpc_run(spec)
            exact = exact_from_cells(spec, run.cells, run.categories)
            released = release(spec, exact, privacy, self.policy, self.randbelow)
        except Exception as exc:                                # nothing was released: give the budget back
            self.ledger.refund(query_id)
            return Outcome(query_id, "failed", hospitals, reason=f"computation failed ({type(exc).__name__})",
                           epsilon_remaining=self.ledger.balance(user_id).remaining_epsilon)

        self.ledger.commit(query_id)                            # 4. a suppressed answer still costs epsilon
        return Outcome(query_id, "suppressed" if released.fully_suppressed else "released", hospitals,
                       released=released, epsilon_charged=privacy.epsilon, delta_charged=privacy.delta,
                       epsilon_remaining=self.ledger.balance(user_id).remaining_epsilon, timings=run.timings)
