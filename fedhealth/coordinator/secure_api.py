"""The requester-facing API (authenticated, role-checked, budgeted, audited).

    POST /auth/login      username + password  -> short-lived JWT
    POST /v1/query        {query, privacy}     -> a NOISY answer (or a refusal) + remaining budget
    GET  /v1/me           who am I, what may I do
    GET  /v1/budget       my privacy budget
    GET  /v1/audit        my audit entries (auditor: everyone's)
    GET  /v1/audit/verify auditor only: recompute the hash chain
    GET  /v1/budgets      auditor only: every account's balance

Order of events for a query (each step can end the request, and EVERY ending is audited):
    authenticate -> validate (aggregates only, closed vocabulary) -> authorise (role, mechanism, epsilon cap)
    -> private pipeline (reserve budget, PSI, MPC with the noise added inside it, k-threshold on the noisy count,
       commit) -> audit -> respond.
The audit entry is written BEFORE the response is returned; if it cannot be written, the answer is withheld.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fedhealth.common.schemas import QuerySpec, participating_hospitals
from fedhealth.coordinator.release import PrivacyParams
from fedhealth.coordinator.service import QueryService
from fedhealth.coordinator.users import LoginThrottle, UserStore
from fedhealth.privacy.access_control.jwt_auth import AuthError, Claims, TokenAuthority
from fedhealth.privacy.access_control.rbac import Forbidden, Policy, authorize_query
from fedhealth.privacy.audit import AuditLog
from fedhealth.privacy.dp.ledger import EpsilonLedger


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: QuerySpec
    privacy: PrivacyParams


class QueryResponse(BaseModel):
    query_id: str
    status: Literal["released", "suppressed", "refused", "failed"]
    reason: str | None = None
    result: float | int | dict[str, int | None] | None = None
    suppressed: bool | dict[str, bool] | None = None
    mechanism: str | None = None
    noise_scale: float | None = None
    expected_abs_error: float | None = None
    hospitals: list[str]
    epsilon_spent: float
    delta_spent: float
    rho_spent: float = 0.0
    epsilon_remaining: float | None
    audit_seq: int


@dataclass
class SecureContext:
    users: UserStore
    tokens: TokenAuthority
    policy: Policy
    ledger: EpsilonLedger
    audit: AuditLog
    service: QueryService
    throttle: LoginThrottle


def mount_secure_api(app: FastAPI, ctx: SecureContext) -> None:

    def authenticate(request: Request) -> Claims:
        header = request.headers.get("authorization", "")
        try:
            if not header.lower().startswith("bearer "):
                raise AuthError("missing bearer token")
            claims = ctx.tokens.verify(header[7:])
            ctx.policy.role(claims.role)                         # a token for a role that no longer exists is void
            return claims
        except (AuthError, Forbidden) as exc:
            ctx.audit.append(actor="anonymous", role="-", action="auth", outcome="denied", reason=str(exc)[:200])
            raise HTTPException(401, "invalid or missing credentials", headers={"WWW-Authenticate": "Bearer"}) from exc

    # ------------------------------------------------------------------ login
    @app.post("/auth/login")
    def login(req: LoginRequest) -> dict:
        if ctx.throttle.blocked(req.username):
            ctx.audit.append(actor=req.username, role="-", action="login", outcome="denied", reason="too many failed attempts")
            raise HTTPException(429, "too many failed attempts, try again later")
        user = ctx.users.authenticate(req.username, req.password)
        if user is None:
            ctx.throttle.record_failure(req.username)
            ctx.audit.append(actor=req.username, role="-", action="login", outcome="denied", reason="bad credentials")
            raise HTTPException(401, "invalid username or password")
        ctx.throttle.reset(req.username)
        ctx.audit.append(actor=user.user_id, role=user.role, action="login", outcome="success")
        return {"access_token": ctx.tokens.issue(user.user_id, user.role), "token_type": "bearer",
                "role": user.role, "expires_in": ctx.tokens.ttl}

    # ------------------------------------------------------------------ the query
    @app.post("/v1/query", response_model=QueryResponse)
    def query(body: dict = Body(...), claims: Claims = Depends(authenticate)) -> QueryResponse:
        try:                                                     # 1. aggregate-only, closed vocabulary
            req = QueryRequest.model_validate(body)
        except ValidationError as exc:
            seq = ctx.audit.append(actor=claims.sub, role=claims.role, action="query", outcome="refused",
                                   reason="rejected: not a valid aggregate query")
            raise HTTPException(422, {"detail": "not a valid aggregate query",
                                      "errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()],
                                      "audit_seq": seq}) from exc
        spec, privacy = req.query, req.privacy
        hospitals = participating_hospitals(spec)
        logged = {"query": spec.model_dump(mode="json", exclude_none=True), "privacy": privacy.model_dump(mode="json")}

        try:                                                     # 2. authorise (role, mechanism, epsilon cap)
            role = authorize_query(ctx.policy, claims.role, spec.agg, privacy.mechanism, privacy.epsilon)
        except Forbidden as exc:
            seq = ctx.audit.append(actor=claims.sub, role=claims.role, action="query", outcome="denied", query=logged,
                                   hospitals=hospitals, reason=exc.reason)
            raise HTTPException(403, {"detail": exc.reason, "audit_seq": seq}) from exc

        ctx.ledger.ensure_user(claims.sub, role.total_epsilon, role.total_delta)      # first query creates the account
        outcome = ctx.service.execute(claims.sub, spec, privacy)                      # 3. the private pipeline
        seq = ctx.audit.append(actor=claims.sub, role=claims.role, action="query", outcome=outcome.status, query=logged,
                               hospitals=outcome.hospitals, epsilon=outcome.epsilon_charged, delta=outcome.delta_charged,
                               reason=outcome.reason)                                 # 4. audited BEFORE the answer leaves
        rel = outcome.released
        return QueryResponse(
            query_id=outcome.query_id, status=outcome.status, reason=outcome.reason,
            result=rel.value if rel else None, suppressed=rel.suppressed if rel else None,
            mechanism=rel.mechanism if rel else None, noise_scale=rel.noise_scale if rel else None,
            expected_abs_error=rel.expected_abs_error if rel else None, hospitals=outcome.hospitals,
            epsilon_spent=outcome.epsilon_charged, delta_spent=outcome.delta_charged, rho_spent=outcome.rho_charged,
            epsilon_remaining=outcome.epsilon_remaining, audit_seq=seq)

    # ------------------------------------------------------------------ introspection
    @app.get("/v1/me")
    def me(claims: Claims = Depends(authenticate)) -> dict:
        rp = ctx.policy.role(claims.role)
        return {"username": claims.sub, "role": claims.role, "can_query": rp.can_query, "aggs": sorted(rp.aggs),
                "mechanisms": sorted(rp.mechanisms), "max_epsilon_per_query": rp.max_epsilon_per_query,
                "k_threshold": ctx.policy.k_threshold, "noise_tolerance": ctx.policy.noise_tolerance, "audit_scope": rp.audit_scope, "can_verify_audit": rp.verify_audit}

    @app.get("/v1/budget")
    def budget(claims: Claims = Depends(authenticate)) -> dict:
        rp = ctx.policy.role(claims.role)
        ctx.ledger.ensure_user(claims.sub, rp.total_epsilon, rp.total_delta)
        b = ctx.ledger.balance(claims.sub)
        return {"total_epsilon": b.total_epsilon, "total_delta": b.total_delta, "spent_epsilon": b.spent_epsilon,
                "remaining_epsilon": b.remaining_epsilon, "remaining_delta": b.remaining_delta,
                "accountant": b.accountant, "rho_spent": b.rho, "rho_cap": b.rho_cap,
                "basic_composition": {"epsilon": b.basic_epsilon, "delta": b.basic_delta}}

    @app.get("/v1/audit")
    def audit_read(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                   claims: Claims = Depends(authenticate)) -> dict:
        rp = ctx.policy.role(claims.role)
        entries = ctx.audit.read(actor=None if rp.audit_scope == "all" else claims.sub, limit=limit, offset=offset)
        return {"scope": rp.audit_scope, "entries": entries}

    @app.get("/v1/audit/verify")
    def audit_verify(claims: Claims = Depends(authenticate)) -> dict:
        if not ctx.policy.role(claims.role).verify_audit:
            ctx.audit.append(actor=claims.sub, role=claims.role, action="audit_verify", outcome="denied", reason="role not permitted")
            raise HTTPException(403, "only an auditor may verify the audit log")
        res = ctx.audit.verify_chain()
        ctx.audit.append(actor=claims.sub, role=claims.role, action="audit_verify", outcome="success" if res.ok else "failed",
                         reason=res.reason)
        return {"ok": res.ok, "entries": res.entries, "head": {"seq": res.head_seq, "mac": res.head_mac},
                "first_bad_seq": res.first_bad_seq, "reason": res.reason}

    @app.get("/v1/budgets")
    def budgets(claims: Claims = Depends(authenticate)) -> dict:
        if ctx.policy.role(claims.role).audit_scope != "all":
            raise HTTPException(403, "auditors only")
        return {"accounts": ctx.ledger.accounts()}
