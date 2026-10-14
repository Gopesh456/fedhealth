"""Streamlit dashboard - a requester's view of the system.

    streamlit run fedhealth/dashboard/app.py
    (needs the STRICT deployment: python scripts/run_local.py --secure  or  docker compose up)

What you can do here, depending on your role:
  * log in (researcher / clinician / auditor),
  * build an aggregate query from dropdowns (condition, treatment, outcome, age range) and pick epsilon,
  * see the NOISY answer, the noise scale, what it cost, and your remaining privacy budget,
  * [demo mode only] toggle the TRUE value next to it to see the error the privacy noise causes,
  * browse the audit log and (auditors) check its cryptographic integrity.

The dashboard never reads a hospital or the coordinator's database: everything goes through the coordinator's API.
Demo mode (FEDHEALTH_DEMO_MODE=1) additionally lets THIS process read the synthetic ground-truth CSVs
(FEDHEALTH_GT_DIR, default ./data/ground_truth) - something no real requester could ever do.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from fedhealth.common import vocab
from fedhealth.common.config import data_dir, load_fields
from fedhealth.common.schemas import QuerySpec
from fedhealth.dashboard import client as client_mod
from fedhealth.dashboard.client import NOT_USED, ApiError, CoordinatorClient, build_query, condition_label

import sys

# Demo mode: `streamlit run fedhealth/dashboard/app.py -- --demo` or FEDHEALTH_DEMO_MODE=1
DEMO_MODE = os.environ.get("FEDHEALTH_DEMO_MODE", "").lower() in ("1", "true", "yes") or "--demo" in sys.argv

st.set_page_config(page_title="Federated healthcare queries", page_icon="🔒", layout="wide")


# ----------------------------------------------------------------------------- session / client
def client() -> CoordinatorClient:
    if "client" not in st.session_state:
        factory = getattr(client_mod, "default_client_factory", None)     # tests inject an in-process client here
        st.session_state.client = factory() if factory else CoordinatorClient.from_env()
    return st.session_state.client


def sidebar_login() -> dict | None:
    c = client()
    st.sidebar.title("🔒 Federated queries")
    st.sidebar.caption("Synthetic data only. Hospitals A, B and C keep their records; you only ever see noisy aggregates.")
    if c.token is None:
        st.sidebar.subheader("Log in")
        user = st.sidebar.text_input("Username", key="login_user")
        pw = st.sidebar.text_input("Password", type="password", key="login_pw")
        if st.sidebar.button("Log in", key="login_btn", type="primary"):
            try:
                c.login(user, pw)
                st.session_state.pop("last_result", None)
                st.rerun()
            except ApiError as exc:
                st.sidebar.error(exc.message)
        st.sidebar.caption("Demo accounts: see secrets/DEMO_CREDENTIALS.txt")
        return None
    try:
        me = c.me()
    except ApiError as exc:                                              # token expired etc.
        c.logout()
        st.sidebar.error(exc.message)
        return None
    st.sidebar.success(f"Signed in as **{me['username']}** ({me['role']})")
    if me["can_query"]:
        b = c.budget()
        st.sidebar.metric("Privacy budget remaining (ε)", f"{b['remaining_epsilon']:.2f}", f"of {b['total_epsilon']:.0f}", delta_color="off")
        st.sidebar.progress(max(0.0, min(1.0, b["remaining_epsilon"] / b["total_epsilon"])) if b["total_epsilon"] else 0.0)
        st.sidebar.caption(f"Allowed: {', '.join(me['aggs'])} · max ε per query {me['max_epsilon_per_query']:g} · k-threshold {me['k_threshold']}")
    else:
        st.sidebar.info("Auditors cannot run data queries; use the Audit tab.")
    if st.sidebar.button("Log out", key="logout_btn"):
        c.logout()
        st.session_state.pop("last_result", None)
        st.rerun()
    return me


# ----------------------------------------------------------------------------- ground truth (demo mode only)
def ground_truth_value(query: dict):
    """The TRUE answer in the same units the coordinator releases (demo mode only)."""
    from fedhealth.synth.ground_truth import GroundTruth
    gt_dir = Path(os.environ.get("FEDHEALTH_GT_DIR", data_dir() / "ground_truth"))
    spec = QuerySpec.model_validate(query)
    value = GroundTruth(gt_dir).evaluate(spec)
    if spec.agg in ("sum", "avg") and value is not None:
        value = value / load_fields().value_fields[spec.value_field].scale          # natural units, like the release
    return value


# ----------------------------------------------------------------------------- query tab
def query_tab(me: dict) -> None:
    if not me["can_query"]:
        st.info("Your role cannot run data queries.")
        return
    c = client()
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Build an aggregate query")
        st.caption("Only counts, sums, averages and histograms exist - there is no way to ask for individual patients.")
        cond = st.selectbox("Condition", vocab.CONDITION_CODES, format_func=condition_label, key="condition")
        treatment = st.selectbox("Treatment (drug prescribed for the condition)", [NOT_USED, *vocab.drug_names(cond)], key="treatment")
        outcome = st.selectbox("Follow-up outcome", [NOT_USED, *vocab.OUTCOMES], key="outcome")
        use_age = st.checkbox("Filter by age", key="use_age")
        age = st.slider("Age range", 18, 95, (40, 65), key="age_range", disabled=not use_age)
        sex = st.selectbox("Sex", [NOT_USED, *vocab.SEXES], key="sex")
        order = ["count", "histogram", "sum", "avg"]                                  # most common first (count is the default)
        aggs = sorted(me["aggs"], key=order.index)
        agg = st.selectbox("Aggregate", aggs, key="agg")
        value_field = group_by = None
        if agg in ("sum", "avg"):
            value_field = st.selectbox("Value", list(load_fields().value_fields), key="value_field")
        if agg == "histogram":
            group_by = st.selectbox("Group by", list(load_fields().group_by), key="group_by")
        st.markdown("**Privacy**")
        eps_cap = float(me["max_epsilon_per_query"])
        epsilon = st.slider("ε for this query (smaller = more private, noisier)", 0.05, eps_cap, min(1.0, eps_cap), 0.05, key="epsilon")
        mechanism = st.selectbox("Noise", me["mechanisms"], key="mechanism")
        query = build_query(condition=cond, treatment=treatment, outcome=outcome, use_age=use_age, age_range=age, sex=sex,
                            agg=agg, value_field=value_field, group_by=group_by)
        with st.expander("Query as sent to the coordinator"):
            st.json({"query": query, "privacy": {"epsilon": epsilon, "mechanism": mechanism}})
        involves_a_hospital = any(k in query for k in ("a", "b", "c", "value_field", "group_by"))
        if not involves_a_hospital:
            st.info("Choose at least one filter (treatment, follow-up outcome, age or sex) to run a count: each filter "
                    "brings in the hospital that holds it.")
        if st.button("Run query", key="run", type="primary", disabled=not involves_a_hospital):
            try:
                st.session_state.last_result = {"query": query, "response": c.query(query, epsilon, mechanism), "error": None}
            except ApiError as exc:
                st.session_state.last_result = {"query": query, "response": None, "error": exc.message}
            st.rerun()
    with right:
        st.subheader("Result")
        last = st.session_state.get("last_result")
        if last is None:
            st.caption("Run a query to see the noisy answer.")
            return
        show_result(last)


def show_result(last: dict) -> None:
    if last["error"]:
        st.error(f"Rejected: {last['error']}")
        return
    r = last["response"]
    if r["status"] == "refused":
        st.warning(f"Refused: {r['reason']}. Remaining ε: {r['epsilon_remaining']}")
        return
    if r["status"] == "failed":
        st.error(f"The computation failed ({r['reason']}); no budget was spent.")
        return
    if r["status"] == "suppressed":
        st.warning("Suppressed: fewer than k patients stand behind this answer, so no number is released. "
                   f"The query still cost ε = {r['epsilon_spent']:g}.")
    result = r["result"]
    if isinstance(result, dict):
        df = pd.DataFrame({"released (noisy)": [v for v in result.values()]}, index=list(result))
        st.bar_chart(df.fillna(0))
        shown = {k: ("suppressed" if v is None else v) for k, v in result.items()}
        st.dataframe(pd.DataFrame({"released (noisy)": shown}), width="stretch")
    elif result is not None:
        st.metric("Noisy answer", f"{result:,.2f}" if isinstance(result, float) else f"{result:,}")
    cols = st.columns(3)
    cols[0].metric("ε spent", f"{r['epsilon_spent']:g}")
    cols[1].metric("ε remaining", f"{r['epsilon_remaining']:.2f}")
    if r.get("expected_abs_error") is not None:
        cols[2].metric("Typical noise (±)", f"{r['expected_abs_error']:.2f}")
    st.caption(f"hospitals: {', '.join(r['hospitals'])} · mechanism: {r['mechanism']} · noise scale {r['noise_scale']:.2f} · audit entry #{r['audit_seq']}")
    if DEMO_MODE:
        st.divider()
        if st.toggle("Show ground truth (demo mode only)", key="show_truth"):
            truth = ground_truth_value(last["query"])
            if isinstance(truth, dict):
                err = {k: (None if result is None or result.get(k) is None else result[k] - v) for k, v in truth.items()}
                st.dataframe(pd.DataFrame({"truth": truth, "released": result, "error": err}), width="stretch")
            else:
                st.metric("True value", f"{truth:,.2f}" if isinstance(truth, float) else f"{truth:,}")
                if result is not None and truth is not None:
                    st.metric("Error caused by privacy noise", f"{result - truth:+,.2f}")
            st.caption("⚠ A real requester can never see this. It exists only so you can inspect the accuracy/privacy trade-off.")


# ----------------------------------------------------------------------------- audit tab
def audit_tab(me: dict) -> None:
    c = client()
    st.subheader("Audit log")
    try:
        log = c.audit(300)
    except ApiError as exc:
        st.error(exc.message)
        return
    st.caption("Showing " + ("every user's entries." if log["scope"] == "all" else "your own entries.") +
               " Every query, login and denial is recorded; the released value never is.")
    if me["can_verify_audit"]:
        try:
            v = c.verify_audit()
            if v["ok"]:
                st.success(f"✅ Integrity verified: {v['entries']} entries, hash chain intact (head #{v['head']['seq']}, mac {v['head']['mac'][:16]}…)")
            else:
                st.error(f"❌ Tampering detected at entry #{v['first_bad_seq']}: {v['reason']}")
        except ApiError as exc:
            st.error(exc.message)
    else:
        st.info("Integrity verification is performed by auditors.")
    rows = [{"#": e["seq"], "time": e["ts"], "user": e["actor"], "role": e["role"], "action": e["action"], "outcome": e["outcome"],
             "ε": e["epsilon_charged"], "hospitals": ",".join(e["hospitals"]), "reason": e["reason"] or "",
             "query": json.dumps((e["query"] or {}).get("query", ""), sort_keys=True) if e["query"] else ""} for e in log["entries"]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if me["can_verify_audit"]:
        st.subheader("Privacy budgets")
        st.dataframe(pd.DataFrame(c.budgets()["accounts"]), width="stretch", hide_index=True)


# ----------------------------------------------------------------------------- main
me = sidebar_login()
if me is None:
    st.title("Privacy-preserving federated healthcare queries")
    st.write("Three hospitals hold different slices of overlapping patients - history, diagnostic reports, prescriptions. "
             "This system answers aggregate questions across all three **without any hospital or the coordinator seeing another's patient data**: "
             "private set intersection links patients, secret sharing computes the answer, differential privacy and a budget limit what the answer reveals.")
    st.info("Log in from the sidebar to run queries.")
else:
    tab_q, tab_a = st.tabs(["Run a query", "Audit log"])
    with tab_q:
        query_tab(me)
    with tab_a:
        audit_tab(me)
