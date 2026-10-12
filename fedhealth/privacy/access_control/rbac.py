"""Role-based access control: WHAT a role may ask and HOW MUCH privacy budget it may spend.

The rules live in configs/policy.yaml, not in code:
    researcher  count + histogram, Laplace only, epsilon <= 1 per query, 10 in total
    clinician   + sum + avg, Gaussian allowed, epsilon <= 2 per query, 20 in total
    auditor     no data queries at all; may read EVERY audit entry and verify the log's integrity

The check happens in the coordinator BEFORE any hospital is contacted or any budget is reserved, and a denial
is itself recorded in the audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


class Forbidden(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RolePolicy:
    name: str
    aggs: frozenset[str]
    mechanisms: frozenset[str]
    max_epsilon_per_query: float
    total_epsilon: float
    total_delta: float
    audit_scope: Literal["own", "all"]
    verify_audit: bool = False
    description: str = ""

    @property
    def can_query(self) -> bool:
        return bool(self.aggs)


@dataclass(frozen=True)
class Policy:
    roles: dict[str, RolePolicy]
    k_threshold: int
    k_basis: str
    default_delta: float
    jwt_ttl_seconds: int

    def role(self, name: str) -> RolePolicy:
        try:
            return self.roles[name]
        except KeyError:
            raise Forbidden(f"unknown role {name!r}") from None


def load_policy(path: str | Path | None = None) -> Policy:
    from fedhealth.common.config import config_dir
    with open(path or (config_dir() / "policy.yaml"), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    roles = {}
    for name, r in raw["roles"].items():
        roles[name] = RolePolicy(
            name=name, aggs=frozenset(r.get("aggs", [])), mechanisms=frozenset(r.get("mechanisms", [])),
            max_epsilon_per_query=float(r.get("max_epsilon_per_query", 0)), total_epsilon=float(r.get("total_epsilon", 0)),
            total_delta=float(r.get("total_delta", 0)), audit_scope=r.get("audit", "own"),
            verify_audit=bool(r.get("verify_audit", False)), description=r.get("description", ""))
    return Policy(roles, int(raw["k_threshold"]), raw.get("k_basis", "exact"), float(raw["default_delta"]),
                  int(raw.get("jwt_ttl_seconds", 1800)))


def authorize_query(policy: Policy, role: str, agg: str, mechanism: str, epsilon: float) -> RolePolicy:
    """Raise Forbidden unless this role may run this kind of query with this privacy setting."""
    rp = policy.role(role)
    if not rp.can_query:
        raise Forbidden(f"role {role!r} may not run data queries")
    if agg not in rp.aggs:
        raise Forbidden(f"role {role!r} may not run {agg!r} queries")
    if mechanism not in rp.mechanisms:
        raise Forbidden(f"role {role!r} may not use the {mechanism!r} mechanism")
    if epsilon > rp.max_epsilon_per_query + 1e-12:
        raise Forbidden(f"epsilon {epsilon} exceeds the per-query cap of {rp.max_epsilon_per_query} for role {role!r}")
    return rp
