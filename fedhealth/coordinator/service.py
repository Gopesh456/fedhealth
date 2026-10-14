"""The private query pipeline as one transaction:

    plan noise  ->  reserve budget  ->  PSI + secret-shared computation WITH the noise added inside it
                ->  k-threshold on the noisy count + post-processing  ->  commit budget

* The coordinator never holds an exact aggregate: the hospitals add the noise shares before anything is opened.

* The budget is reserved BEFORE any hospital is contacted, so an exhausted requester costs the hospitals nothing
  (and cannot learn anything from timing or traffic).
* A plan the hospitals would refuse (configs/policy.yaml: distributed_noise) is refused here first, with a reason,
  instead of failing half-way through the protocol. The hospitals still re-check it themselves.
* If anything fails before any hospital has sent a share of its data, the reservation is REFUNDED. From the moment
  shares leave a hospital the spend is COMMITTED, even if the computation then aborts (orchestrator.ComputationAborted)
  or the answer is suppressed: a free retry after a data-dependent abort would be a channel outside the budget, and
  the suppression decision is part of the noisy release.
* Authentication, roles and the audit trail wrap this service in Phase 5.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

from fedhealth.common.schemas import QuerySpec, participating_hospitals
from fedhealth.coordinator.orchestrator import ComputationAborted, Orchestrator
from fedhealth.coordinator.release import PrivacyParams, Released, ReleasePolicy, finalize, plan_noise
from fedhealth.privacy.dp.distributed import NoisePolicy, load_noise_policy
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
    rho_charged: float = 0.0
    epsilon_remaining: float | None = None
    timings: dict[str, float] = field(default_factory=dict)


class QueryService:
    def __init__(self, orchestrator: Orchestrator, ledger: EpsilonLedger, policy: ReleasePolicy = ReleasePolicy(),
                 noise_policy: NoisePolicy | None = None):
        self.orchestrator, self.ledger, self.policy = orchestrator, ledger, policy
        self.noise_policy = noise_policy or load_noise_policy()

    def execute(self, user_id: str, spec: QuerySpec, privacy: PrivacyParams) -> Outcome:
        query_id = uuid.uuid4().hex
        hospitals = participating_hospitals(spec)

        plan = plan_noise(spec, privacy, self.policy)            # public: what each hospital's noise share looks like
        if self.policy.dp_enabled:
            try:
                plan.check(self.noise_policy, spec.agg, len(plan.cells))
            except ValueError as exc:                           # the reason describes the public plan only
                reason = ("noise scale above the hospitals' limit" if "scale" in str(exc)
                          else "above the hospitals' per-query privacy limit")
                return Outcome(query_id, "refused", hospitals, reason=reason)
        try:                                                    # 1. budget first - nothing else happens without it
            self.ledger.reserve(user_id, privacy.epsilon, query_id, privacy.delta, rho=plan.rho)
        except BudgetExhausted as exc:
            return Outcome(query_id, "refused", hospitals, reason="privacy budget exhausted",
                           epsilon_remaining=exc.remaining)

        try:                                                    # 2. private computation (noise inside)
            run = self.orchestrator.mpc_run(spec, plan)
        except ComputationAborted as exc:                       # shares had left the hospitals: charged
            return self._charged_failure(query_id, user_id, hospitals, privacy, plan.rho, exc)
        except Exception as exc:                                # nothing left any hospital: give the budget back
            self.ledger.refund(query_id)
            return Outcome(query_id, "failed", hospitals, reason=f"computation failed ({type(exc).__name__})",
                           epsilon_remaining=self.ledger.balance(user_id).remaining_epsilon)
        try:                                                    # 3. release (post-processing of the noisy cells)
            released = finalize(spec, run.cells, plan, run.categories, self.policy)
        except Exception as exc:
            return self._charged_failure(query_id, user_id, hospitals, privacy, plan.rho, exc)

        self.ledger.commit(query_id)                            # 4. a suppressed answer still costs budget
        return Outcome(query_id, "suppressed" if released.fully_suppressed else "released", hospitals,
                       released=released, epsilon_charged=privacy.epsilon, delta_charged=privacy.delta,
                       rho_charged=plan.rho, epsilon_remaining=self.ledger.balance(user_id).remaining_epsilon,
                       timings=run.timings)

    def _charged_failure(self, query_id: str, user_id: str, hospitals: list[str], privacy: PrivacyParams, rho: float,
                         exc: Exception) -> Outcome:
        self.ledger.commit(query_id)
        return Outcome(query_id, "failed", hospitals,
                       reason=f"computation failed after the hospitals shared their data ({type(exc).__name__}); "
                              "the budget is charged",
                       epsilon_charged=privacy.epsilon, delta_charged=privacy.delta, rho_charged=rho,
                       epsilon_remaining=self.ledger.balance(user_id).remaining_epsilon)
