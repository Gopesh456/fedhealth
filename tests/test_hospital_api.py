"""Hospital node: attack surface, input validation, read-only database."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from fedhealth.hospital.app import create_hospital_app

PROTOCOL_PATHS = {                                   # protocol endpoints only: none returns a patient or an ID
    "/health",
    "/psi/{lid}/init", "/psi/{lid}/step", "/psi/{lid}/inbox", "/psi/{lid}/publish", "/psi/{lid}/tags",
    "/psi/{lid}/finalize", "/psi/{lid}",
    "/mpc/{sid}/prepare", "/mpc/{sid}/share", "/mpc/{sid}/inshare", "/mpc/{sid}/deal", "/mpc/{sid}/intriples",
    "/mpc/{sid}/open",
    "/mpc/{sid}/inopen", "/mpc/{sid}/finish", "/mpc/{sid}/output", "/mpc/{sid}",
    "/fl/{sid}/train", "/fl/{sid}/update", "/fl/{sid}/share", "/fl/{sid}/inshare", "/fl/{sid}/sumshare", "/fl/{sid}",
}
INSECURE_PATHS = {"/insecure/match", "/insecure/match_linked/{lid}"}
ALLOWED_PATHS = PROTOCOL_PATHS | INSECURE_PATHS


@pytest.mark.parametrize("h", "ABC")
def test_health(hospital_clients, h):
    r = hospital_clients[h].get("/health")
    assert r.status_code == 200 and r.json()["hospital"] == h


def all_paths(routes) -> set[str]:
    """Every path in a route table, descending into included routers (FastAPI wraps them lazily)."""
    found = set()
    for r in routes:
        if hasattr(r, "path"):
            found.add(r.path)
        elif hasattr(r, "original_router"):
            found |= all_paths(r.original_router.routes)
        else:
            raise AssertionError(f"unrecognised route object {type(r).__name__} - extend all_paths()")
    return found


@pytest.mark.parametrize("h", "ABC")
def test_only_whitelisted_routes_exist(hospital_apps, h):
    """No endpoint can return rows: the route table is closed. (Docs/openapi are disabled too.)"""
    paths = all_paths(hospital_apps[h].routes)
    assert paths == ALLOWED_PATHS, paths ^ ALLOWED_PATHS


def test_insecure_routes_are_opt_in(world):
    app = create_hospital_app("A", db_path=world["dir"] / "hospital_a.db", allow_insecure=False)
    assert all_paths(app.routes) == PROTOCOL_PATHS          # nothing that returns per-patient data is mounted
    c = TestClient(app)
    assert c.get("/health").json()["insecure_baseline"] is False
    assert c.post("/insecure/match", json={"hospital": "A", "condition": "E11"}).status_code == 404


def test_hospital_refuses_other_hospitals_query(hospital_clients):
    r = hospital_clients["A"].post("/insecure/match", json={"hospital": "B", "condition": "E11"})
    assert r.status_code == 400


@pytest.mark.parametrize("body", [
    {"hospital": "A", "condition": "E11", "predicate": {"national_id": "SYN-1"}},          # unknown field
    {"hospital": "A", "condition": "E11", "predicate": {"sex": "F; DROP TABLE patients"}},  # not in enum
    {"hospital": "A", "condition": "E11' OR '1'='1"},                                       # injection-looking code
    {"hospital": "A", "condition": "XXX"},                                                  # unknown condition
    {"hospital": "A", "condition": "E11", "predicate": {"age_min": 80, "age_max": 20}},     # inverted range
    {"hospital": "A", "condition": "E11", "select": "*"},                                   # row-level attempt
    {"hospital": "A", "condition": "E11", "columns": ["full_name"]},                        # row-level attempt
    {"hospital": "A", "condition": "E11", "value_field": "lab_delta"},                      # B's attribute asked of A
    {"hospital": "A", "condition": "E11", "value_field": "age", "group_by": "sex"},         # mutually exclusive
])
def test_hospital_rejects_non_whitelisted_requests(hospital_clients, body):
    assert hospital_clients["A"].post("/insecure/match", json=body).status_code == 422


def test_hospital_c_rejects_sql_looking_drug(hospital_clients):
    body = {"hospital": "C", "condition": "E11", "predicate": {"drug": "metformin'; DROP TABLE prescriptions;--"}}
    assert hospital_clients["C"].post("/insecure/match", json=body).status_code == 422
    # and the table is still there
    ok = {"hospital": "C", "condition": "E11", "predicate": {"drug": "metformin"}}
    assert hospital_clients["C"].post("/insecure/match", json=ok).status_code == 200


@pytest.mark.parametrize("h", "ABC")
def test_service_database_is_read_only(hospital_apps, h):
    """A hospital node only answers questions; even a bug could not modify patient data through it."""
    with hospital_apps[h].state.engine.connect() as conn:
        with pytest.raises(OperationalError):
            conn.execute(text("DELETE FROM patients"))


def test_each_hospital_only_sees_its_own_tables(hospital_apps):
    tables = {}
    for h, app in hospital_apps.items():
        with app.state.engine.connect() as conn:
            tables[h] = {r[0] for r in conn.execute(text("select name from sqlite_master where type='table'"))}
    assert "conditions" in tables["A"] and "conditions" not in tables["B"] | tables["C"]
    assert "followup_outcomes" in tables["B"] and "followup_outcomes" not in tables["A"] | tables["C"]
    assert "prescriptions" in tables["C"] and "prescriptions" not in tables["A"] | tables["B"]
