"""Phase 7: the Streamlit dashboard, driven headlessly with streamlit.testing (no browser, no network): the app's
client is replaced by one that talks to an in-process STRICT system (service identities + JWT + RBAC + ledger + audit)."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from fedhealth.coordinator.app import CoordinatorSecrets
from fedhealth.dashboard import client as client_mod
from fedhealth.dashboard.client import NOT_USED, ApiError, CoordinatorClient, build_query
from fedhealth.synth.generate import generate
from fedhealth.synth.ground_truth import GroundTruth

from .helpers import headline_spec
from .system import build_system

APP = str(Path(__file__).resolve().parents[1] / "fedhealth" / "dashboard" / "app.py")


# ============================================================================ pure query builder
def test_build_query_maps_dropdowns_to_hospital_predicates():
    q = build_query(condition="E11", treatment="metformin", outcome="improved", use_age=True, age_range=(40, 65))
    assert q == {"agg": "count", "condition": "E11", "a": {"age_min": 40, "age_max": 65}, "b": {"outcome": "improved"},
                 "c": {"drug": "metformin"}}
    assert build_query(condition="I10") == {"agg": "count", "condition": "I10"}              # nothing chosen: no hospital involved
    h = build_query(condition="E11", outcome="improved", agg="histogram", group_by="drug")
    assert h["group_by"] == "drug" and "c" not in h and h["b"] == {"outcome": "improved"}
    s = build_query(condition="E11", sex="F", agg="avg", value_field="age")
    assert s["a"] == {"sex": "F"} and s["value_field"] == "age"


def test_not_used_selections_do_not_make_a_hospital_participate():
    q = build_query(condition="E11", treatment=NOT_USED, outcome=NOT_USED, use_age=False, sex=NOT_USED)
    assert set(q) == {"agg", "condition"}


# ============================================================================ the client
def test_client_surfaces_api_errors(tmp_path_factory, pki):
    s = make_system(tmp_path_factory, pki)
    c = CoordinatorClient(s.system.client)
    with pytest.raises(ApiError) as exc:
        c.login("researcher1", "wrong")
    assert exc.value.status == 401 and "invalid" in exc.value.message
    with pytest.raises(ApiError) as exc:
        c.me()                                                                              # no token yet
    assert exc.value.status == 401


# ============================================================================ the app
def make_system(tmp_path_factory, pki):
    d = tmp_path_factory.mktemp("dash_world")
    generate(600, seed=5, out_dir=d)
    system = build_system(d, psi_group="test768", psi_groups=("test768",), allow_insecure=False, secure=True, tls_dir=pki,
                          secrets=CoordinatorSecrets(os.urandom(32), os.urandom(32)), coordinator_db=d / "coordinator.db")
    for user, role in (("researcher1", "researcher"), ("clinician1", "clinician"), ("auditor1", "auditor")):
        system.secure.users.create(user, role, f"pw-{user}")
    return SimpleNamespace(system=system, gt_dir=d / "ground_truth", gt=GroundTruth(d / "ground_truth"))


@pytest.fixture(scope="module")
def dash(tmp_path_factory, pki):
    return make_system(tmp_path_factory, pki)


@pytest.fixture()
def app(dash, monkeypatch):
    monkeypatch.setattr(client_mod, "default_client_factory", lambda: CoordinatorClient(dash.system.client), raising=False)
    monkeypatch.delenv("FEDHEALTH_DEMO_MODE", raising=False)
    return AppTest.from_file(APP, default_timeout=120)


def login(at: AppTest, user: str, password: str | None = None) -> AppTest:
    at.run()
    at.sidebar.text_input(key="login_user").input(user)
    at.sidebar.text_input(key="login_pw").input(password or f"pw-{user}")
    at.sidebar.button(key="login_btn").click().run()
    return at


def labels(at: AppTest) -> list[str]:
    return [m.label for m in at.metric] + [m.label for m in at.sidebar.metric]


def test_logged_out_users_see_only_the_login(app):
    app.run()
    assert not app.exception and app.sidebar.text_input(key="login_user") is not None
    assert not app.tabs and any("Log in" in i.value for i in app.info)


def test_a_bad_password_is_rejected(app):
    login(app, "researcher1", "nope")
    assert any("invalid" in e.value.lower() for e in app.sidebar.error) and not app.tabs


def test_a_count_without_any_filter_cannot_be_sent_and_says_why(app, dash):
    """The form opens with no filter, and a count with no filter involves no hospital: the button says so instead of
    sending a query the coordinator would reject."""
    login(app, "researcher1")
    assert app.button(key="run").disabled and any("at least one filter" in i.value for i in app.info)
    app.selectbox(key="outcome").select("improved").run()
    assert not app.button(key="run").disabled
    c = CoordinatorClient(dash.system.client)
    c.login("researcher1", "pw-researcher1")
    with pytest.raises(ApiError) as exc:
        c.query({"agg": "count", "condition": "E11"}, 0.5, "laplace")
    assert exc.value.status == 422 and "at least one hospital" in exc.value.message      # the reason, not only "invalid"


def test_a_researcher_builds_and_runs_a_query_and_sees_the_budget_move(app, dash):
    login(app, "researcher1")
    assert not app.exception
    assert "Privacy budget remaining (ε)" in labels(app) and app.sidebar.metric[0].value == "10.00"
    assert [o for o in app.selectbox(key="agg").options] == ["count", "histogram"]          # role-limited aggregates
    assert app.slider(key="epsilon").max == 1.0                                              # role-limited epsilon
    app.selectbox(key="condition").select("E11")
    app.selectbox(key="treatment").select("metformin")
    app.selectbox(key="outcome").select("improved")
    app.checkbox(key="use_age").check().run()                                              # enables the (disabled) age slider
    app.slider(key="age_range").set_value((18, 90))                                          # true count 24: far above k
    app.slider(key="epsilon").set_value(1.0)
    app.run()
    app.button(key="run").click().run()
    assert not app.exception
    assert "Noisy answer" in labels(app) and "ε spent" in labels(app)
    spent = next(m for m in app.metric if m.label == "ε spent")
    assert spent.value == "1"
    assert app.sidebar.metric[0].value == "9.00"                                             # budget moved
    assert "True value" not in labels(app), "ground truth must not be visible outside demo mode"
    truth = dash.gt.evaluate(headline_spec(a={"age_min": 18, "age_max": 90}))
    shown = int(next(m for m in app.metric if m.label == "Noisy answer").value.replace(",", ""))
    assert abs(shown - truth) <= 15


def test_a_histogram_is_charted_and_tabulated(app):
    login(app, "researcher1")
    app.selectbox(key="agg").select("histogram")
    app.run()
    app.selectbox(key="group_by").select("outcome")
    app.selectbox(key="condition").select("I10")
    app.run()                                                                                  # the treatment list depends on the condition
    app.selectbox(key="outcome").select(NOT_USED)
    app.checkbox(key="use_age").uncheck()
    app.selectbox(key="treatment").select("lisinopril")
    app.run()
    app.button(key="run").click().run()
    assert not app.exception
    assert len(app.dataframe) >= 1


def test_ground_truth_toggle_exists_only_in_demo_mode(dash, monkeypatch):
    monkeypatch.setattr(client_mod, "default_client_factory", lambda: CoordinatorClient(dash.system.client), raising=False)
    monkeypatch.setenv("FEDHEALTH_DEMO_MODE", "1")
    monkeypatch.setenv("FEDHEALTH_GT_DIR", str(dash.gt_dir))
    at = AppTest.from_file(APP, default_timeout=120)
    login(at, "clinician1")
    at.selectbox(key="condition").select("I10")
    at.selectbox(key="outcome").select(NOT_USED)
    at.selectbox(key="treatment").select(NOT_USED)
    at.selectbox(key="sex").select("F")
    at.slider(key="epsilon").set_value(2.0)
    at.run()
    at.button(key="run").click().run()
    assert not at.exception
    assert not [t for t in at.toggle if t.key == "show_truth" and t.value]                  # off by default
    at.toggle(key="show_truth").set_value(True).run()
    assert "True value" in labels(at) and "Error caused by privacy noise" in labels(at)
    truth = dash.gt.evaluate(__import__("fedhealth.common.schemas", fromlist=["QuerySpec"]).QuerySpec.model_validate(
        {"agg": "count", "condition": "I10", "a": {"sex": "F"}}))
    assert next(m for m in at.metric if m.label == "True value").value == f"{truth:,}"


def test_a_suppressed_answer_is_explained_and_still_charged(app):
    login(app, "researcher1")
    app.selectbox(key="condition").select("I50")
    app.run()                                                                                  # the treatment list depends on the condition
    app.selectbox(key="treatment").select("ivabradine")
    app.selectbox(key="outcome").select("improved")
    app.run()
    before = float(app.sidebar.metric[0].value)
    app.button(key="run").click().run()
    assert any("Suppressed" in w.value for w in app.warning)
    assert float(app.sidebar.metric[0].value) < before


def test_the_auditor_sees_everything_and_the_integrity_status(app, dash):
    login(app, "auditor1")
    assert not app.exception
    assert any("cannot run data queries" in i.value for i in app.sidebar.info)                 # auditors do not query
    assert any("Integrity verified" in s.value for s in app.success)
    audit = app.dataframe[0].value                                                             # the auditor sees EVERYONE's entries
    assert {"researcher1", "auditor1"} <= set(audit["user"])
    assert len(app.dataframe) == 2                                                             # audit table + privacy budgets


def test_the_integrity_banner_turns_red_after_tampering(tmp_path_factory, pki, monkeypatch):
    d = make_system(tmp_path_factory, pki)
    monkeypatch.setattr(client_mod, "default_client_factory", lambda: CoordinatorClient(d.system.client), raising=False)
    import sqlite3
    at = AppTest.from_file(APP, default_timeout=120)
    login(at, "researcher1")                                                                  # creates audit entries
    con = sqlite3.connect(str(d.system.secure.audit.engine.url.database), isolation_level=None)
    con.execute("DROP TRIGGER audit_no_update")
    con.execute("UPDATE audit_log SET actor = 'mallory' WHERE action = 'login'")
    con.close()
    at2 = AppTest.from_file(APP, default_timeout=120)
    login(at2, "auditor1")
    assert any("Tampering detected" in e.value for e in at2.error)
