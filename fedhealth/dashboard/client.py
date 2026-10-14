"""The dashboard's only way to the system: the coordinator's authenticated HTTPS API.

The dashboard never talks to a hospital and never sees a database. It holds a JWT in session state and calls
/auth/login, /v1/query, /v1/budget, /v1/audit ... exactly like any other requester. Taking an `httpx.Client`
(or a FastAPI TestClient, which is one) keeps the app testable without a network.
"""
from __future__ import annotations

import os
import ssl
from typing import Any

import httpx

from fedhealth.common import vocab


class ApiError(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail

    @property
    def message(self) -> str:
        d = self.detail
        if isinstance(d, dict):
            errors = d.get("errors") or []
            reason = errors[0].get("msg", "").removeprefix("Value error, ") if errors and isinstance(errors[0], dict) else ""
            return str(d.get("detail", d)) + (f" ({reason})" if reason else "")
        return str(d)


class CoordinatorClient:
    def __init__(self, http: httpx.Client):
        self.http = http
        self.token: str | None = None

    @classmethod
    def from_env(cls) -> "CoordinatorClient":
        """FEDHEALTH_COORDINATOR_URL (default https://127.0.0.1:8000), FEDHEALTH_CA (default secrets/tls/ca.pem)."""
        from fedhealth.privacy.encryption.tls import tls_dir
        base = os.environ.get("FEDHEALTH_COORDINATOR_URL", "https://127.0.0.1:8000")
        verify: Any = True
        if base.startswith("https"):
            verify = ssl.create_default_context(cafile=os.environ.get("FEDHEALTH_CA", str(tls_dir() / "ca.pem")))
        return cls(httpx.Client(base_url=base, verify=verify, timeout=300))

    # ------------------------------------------------------------------ plumbing
    def _call(self, method: str, path: str, **kw) -> dict:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            r = self.http.request(method, path, headers=headers, **kw)
        except httpx.HTTPError as exc:
            raise ApiError(0, f"cannot reach the coordinator: {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(r.status_code, detail)
        return r.json()

    # ------------------------------------------------------------------ API
    def login(self, username: str, password: str) -> dict:
        out = self._call("POST", "/auth/login", json={"username": username, "password": password})
        self.token = out["access_token"]
        return out

    def logout(self) -> None:
        self.token = None

    def me(self) -> dict:
        return self._call("GET", "/v1/me")

    def budget(self) -> dict:
        return self._call("GET", "/v1/budget")

    def query(self, query: dict, epsilon: float, mechanism: str = "laplace") -> dict:
        return self._call("POST", "/v1/query", json={"query": query, "privacy": {"epsilon": epsilon, "mechanism": mechanism}})

    def audit(self, limit: int = 200) -> dict:
        return self._call("GET", "/v1/audit", params={"limit": limit})

    def verify_audit(self) -> dict:
        return self._call("GET", "/v1/audit/verify")

    def budgets(self) -> dict:
        return self._call("GET", "/v1/budgets")


# ----------------------------------------------------------------------------- building a query from the dropdowns
NOT_USED = "(not used)"


def build_query(*, condition: str, treatment: str = NOT_USED, outcome: str = NOT_USED, use_age: bool = False,
                age_range: tuple[int, int] = (18, 90), sex: str = NOT_USED, agg: str = "count",
                value_field: str | None = None, group_by: str | None = None) -> dict:
    """Dropdown selections -> a QuerySpec dict. A hospital takes part only if one of its filters is used
    (or it owns the value / group-by field): A = demographics, B = outcome, C = treatment."""
    q: dict = {"agg": agg, "condition": condition}
    a: dict = {}
    if use_age:
        a.update(age_min=int(age_range[0]), age_max=int(age_range[1]))
    if sex != NOT_USED:
        a["sex"] = sex
    if a:
        q["a"] = a
    if outcome != NOT_USED:
        q["b"] = {"outcome": outcome}
    if treatment != NOT_USED:
        q["c"] = {"drug": treatment}
    if agg in ("sum", "avg"):
        q["value_field"] = value_field
    if agg == "histogram":
        q["group_by"] = group_by
    return q


def condition_label(code: str) -> str:
    return f"{vocab.CONDITIONS[code].name} ({code})"
