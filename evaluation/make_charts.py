"""Render every chart from the saved JSON results (evaluation/results/*.json -> evaluation/charts/*.png).

Chart conventions (validated palette; see the dataviz method): fixed categorical slot order blue / orange / aqua /
yellow / magenta; ONE y-axis per chart; 2 px lines, >= 8 px markers; recessive hairline grid; text in ink tokens, never
in a series colour; a legend whenever there are >= 2 series PLUS direct labels (aqua / yellow / magenta are below 3:1
contrast on the light surface, so labels and the tables in RESULTS.md are the relief); marker shape and line style are
secondary encodings so colour is never the only channel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, CHARTS = ROOT / "evaluation" / "results", ROOT / "evaluation" / "charts"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
MARKERS = ["o", "s", "^", "D", "v"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 10, "lines.linewidth": 2.0, "lines.markersize": 7.5,
    })


def title(ax, text: str, sub: str | None = None) -> None:
    ax.set_title(text, loc="left", fontsize=13, fontweight="bold", color=INK, pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def legend(ax, **kw):
    leg = ax.legend(**kw)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


def label(ax, x, y, text, color=INK2, dx=6, dy=0, ha="left", **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color, fontsize=9.5, va="center", ha=ha, **kw)


def save(fig, name: str) -> Path:
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def load(name: str):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------------------------- attack 1
def chart_differencing(d) -> Path:
    sys.path.insert(0, str(ROOT / "attacks"))
    from differencing import analytic_accuracy              # noqa: E402 - the exact formula, for a smooth line
    dp = [r for r in d["rows"] if r["epsilon"]]
    eps = np.array([r["epsilon"] for r in dp])
    acc = np.array([r["accuracy"] for r in dp])
    lo = np.array([r["ci95"][0] for r in dp])
    hi = np.array([r["ci95"][1] for r in dp])
    r_total = dp[0]["noise_contributors"] / dp[0]["honest_shares"]
    fine = np.geomspace(eps.min(), eps.max(), 80)
    analytic = [analytic_accuracy(e, r_total) for e in fine]
    coalition = [analytic_accuracy(e, 1.0) for e in fine]
    bound = [np.exp(2 * e) / (1 + np.exp(2 * e)) for e in fine]
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.axhline(1.0, color=INK2, lw=1.4, ls=(0, (6, 3)))
    ax.axhline(0.5, color=MUTED, lw=1.2)
    ax.fill_between(eps, lo, hi, color=BLUE, alpha=0.16, lw=0)
    ax.plot(fine, bound, color=AQUA, ls=":", lw=2.2)
    ax.plot(fine, coalition, color=YELLOW, ls="-.", lw=2)
    ax.plot(fine, analytic, color=ORANGE, ls="--", lw=2)
    ax.plot(eps, acc, color=BLUE, marker=MARKERS[0], label="Measured requester (attacker accuracy, 95% CI)")
    ax.plot([], [], color=ORANGE, ls="--", label="Exact theory: requester (all 3 noise shares)")
    ax.plot([], [], color=YELLOW, ls="-.", label="Exact theory: coordinator + 1 hospital (2 honest shares)")
    ax.plot([], [], color=AQUA, ls=":", label="DP ceiling for ANY attacker (2 queries)")
    ax.set_xscale("log")
    ax.set_xticks(eps)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0.45, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("ε spent per query (each attack uses two queries)")
    ax.set_ylabel("Attacker's accuracy at guessing one patient's outcome")
    label(ax, eps.min(), 1.0, "No protection, and k-threshold alone: 100%", INK2, dx=0, dy=9)
    label(ax, eps.max(), 0.5, "Blind guess", MUTED, dx=0, dy=-9, ha="right")
    legend(ax, loc="center left", bbox_to_anchor=(0.0, 0.64))
    title(ax, "Differencing attack: differential privacy is what stops it",
          f"{d['n_targets']} isolatable targets · {d['trials_per_setting']:,} trials per setting · noise added inside the MPC")
    return save(fig, "attack_differencing")


# --------------------------------------------------------------------------------------------- attack 2
def chart_reid(d) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.75})
    rows = sorted(d["by_qi"], key=lambda r: r["reid_rate"])
    names = [r["qi"].replace("birth_year", "birth year").replace("+", " + ") for r in rows]
    vals = [r["reid_rate"] for r in rows]
    a1.barh(names, vals, color=BLUE, height=0.62)
    a1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a1.set_xlim(0, max(vals) * 1.25 + 0.02)
    for i, v in enumerate(vals):
        a1.text(v + 0.008, i, f"{v:.1%}", va="center", color=INK2, fontsize=9.5)
    a1.set_xlabel("Records re-identified (correctly)")
    a1.grid(axis="y", visible=False)
    title(a1, "Which quasi-identifiers?", f"{d['released_records']:,} Hospital-A patients, names removed")
    defs = d["defences"]
    labels = ["names removed only", "generalised\n(age band, ZIP3)", "generalised +\n5-anonymity", "generalised +\n10-anonymity", "exact +\n10-anonymity"]
    y = np.arange(len(defs))[::-1]
    h = 0.36
    a2.barh(y + h / 2, [r["reid_rate"] for r in defs], height=h, color=BLUE, label="Records re-identified")
    a2.barh(y - h / 2, [r["retained_fraction"] for r in defs], height=h, color=ORANGE, label="Records still released")
    for yy, r in zip(y, defs):
        a2.text(r["reid_rate"] + 0.01, yy + h / 2, f"{r['reid_rate']:.1%}", va="center", color=INK2, fontsize=9.5)
        a2.text(r["retained_fraction"] + 0.01, yy - h / 2, f"{r['retained_fraction']:.0%}", va="center", color=INK2, fontsize=9.5)
    a2.set_yticks(y, labels)
    a2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a2.set_xlim(0, 1.12)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("Share of the released table")
    legend(a2, loc="lower right")
    title(a2, "Defences: generalise and enforce k-anonymity", "birth year + sex + ZIP against a voter-roll covering 70% of the world")
    return save(fig, "attack_reidentification")


# --------------------------------------------------------------------------------------------- attack 3
def chart_budget(d) -> Path:
    u = d["unlimited"]
    n = np.array([r["queries"] for r in u])
    err = np.array([r["mean_abs_error"] for r in u])
    clt = np.array([r["theory"] for r in u])
    lim = d["with_ledger"]
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.plot(n, clt, color=ORANGE, ls="--", lw=2, label="CLT approximation (accurate for N ≥ 50)")
    ax.plot(n, err, color=BLUE, marker=MARKERS[0], label="Attacker with NO budget limit (average of N answers)")
    ax.axvline(lim["answered"], color=INK2, lw=1.4)
    ax.scatter([lim["answered"]], [lim["expected_abs_error_at_that_point"]], s=90, color=AQUA, zorder=5, marker="D", label="Attacker WITH the ledger: refused after this point")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Number of identical queries averaged (ε = %g each)" % d["epsilon_per_query"])
    ax.set_ylabel("Mean absolute error of the attacker's estimate")
    label(ax, lim["answered"], err.max() * 0.9, f"ledger refuses query {lim['first_refusal_at_query']}\n(budget {lim['budget']:g} = {lim['answered']} × ε)", INK, dx=8)
    label(ax, n.max(), err[-1], f"{err[-1]:.2f}  after ε = {u[-1]['epsilon_spent']:g} burned", INK2, dx=-4, dy=-14, ha="right")
    legend(ax, loc="lower left")
    title(ax, "Repeating a query averages the noise away - until the ledger says stop",
          f"true value {d['true_value']} · {d['attackers_simulated']} independent attackers per point")
    return save(fig, "attack_budget_exhaustion")


# --------------------------------------------------------------------------------------------- evaluation 1
def chart_accuracy(d) -> list[Path]:
    rows = d["rows"]
    eps = sorted({r["epsilon"] for r in rows})
    counts = [r for r in rows if r["kind"] == "count"]
    fig, ax = plt.subplots(figsize=(8.4, 5))
    for mech, color, marker, name in (("laplace", BLUE, MARKERS[0], "Laplace (ε-DP)"), ("gaussian", ORANGE, MARKERS[1], "Gaussian ((ε, δ=10⁻⁶)-DP)")):
        ys = [np.mean([r["mae"] for r in counts if r["mechanism"] == mech and r["epsilon"] == e]) for e in eps]
        ax.plot(eps, ys, color=color, marker=marker, label=name)
        for cohort in ("small", "medium", "large"):                       # the three cohorts overlap: error does not depend on cohort size
            ax.scatter(eps, [r["mae"] for e in eps for r in counts if r["mechanism"] == mech and r["epsilon"] == e and r["cohort"] == cohort],
                       s=16, color=color, alpha=0.35, zorder=2)
    th = [counts[0]["theory_mae"]] and [next(r["theory_mae"] for r in counts if r["mechanism"] == "laplace" and r["epsilon"] == e) for e in eps]
    ax.plot(eps, th, color=AQUA, ls=":", lw=2.2, label="Laplace theory: E|sum of the noise shares|")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(eps)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("ε per query")
    ax.set_ylabel("Mean absolute error (patients)")
    legend(ax, loc="upper right")
    title(ax, "Error falls as ε grows - and does not depend on how big the cohort is",
          f"COUNT queries, small / medium / large cohorts (faint dots) · 3 hospitals' noise shares, any 2 suffice · {d['repetitions']:,} releases per point")
    p1 = save(fig, "accuracy_vs_epsilon")

    fig, ax = plt.subplots(figsize=(8.4, 5))
    order = [("small", BLUE, MARKERS[0]), ("medium", ORANGE, MARKERS[1]), ("large", AQUA, MARKERS[2])]
    for cohort, color, marker in order:
        sel = [next(r for r in counts if r["mechanism"] == "laplace" and r["epsilon"] == e and r["cohort"] == cohort) for e in eps]
        ax.plot(eps, [r["relative_mae"] for r in sel], color=color, marker=marker, label=f"{cohort} cohort (true count {sel[0]['true_value']})")
        label(ax, eps[-1], sel[-1]["relative_mae"], f"{cohort}", INK2, dx=8)
    ax.axhline(0.10, color=MUTED, lw=1.2, ls=(0, (5, 3)))
    label(ax, eps[0], 0.10, "10% error", MUTED, dx=0, dy=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(eps)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    ax.set_xlabel("ε per query")
    ax.set_ylabel("Mean absolute error ÷ true count")
    legend(ax, loc="upper right")
    title(ax, "Relative error: small cohorts pay for privacy, big ones barely notice", "Laplace mechanism")
    return [p1, save(fig, "relative_error_vs_epsilon"), chart_tolerance(d), chart_accounting(d)]


def chart_tolerance(d) -> Path:
    rows = [r for r in d["rows"] if r["kind"] == "tolerance"]
    eps = sorted({r["epsilon"] for r in rows})
    fig, ax = plt.subplots(figsize=(8.4, 5))
    names = {0: "t = 0: all 3 shares needed (= the old trusted-coordinator noise)", 1: "t = 1: any 2 shares suffice (default)",
             2: "t = 2: any 1 share suffices"}
    for t, color, marker in ((0, BLUE, MARKERS[0]), (1, ORANGE, MARKERS[1]), (2, AQUA, MARKERS[2])):
        lap = [next(r for r in rows if r["mechanism"] == "laplace" and r["tolerance"] == t and r["epsilon"] == e) for e in eps]
        ax.plot(eps, [r["mae"] for r in lap], color=color, marker=marker, label=names[t])
        label(ax, eps[-1], lap[-1]["mae"], f"t = {t}", INK2, dx=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(eps)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("ε per query")
    ax.set_ylabel("Mean absolute error (patients)")
    legend(ax, loc="upper right")
    m = rows[0]
    title(ax, "Not trusting the coordinator costs ×√1.5 in noise",
          f"Laplace, medium cohort (n={m['true_value']}) · t = hospitals that may collude with the coordinator · variance × 3/(3−t)")
    return save(fig, "collusion_tolerance_cost")


def chart_accounting(d) -> Path:
    a = d["accounting"]
    eps = np.array([r["epsilon"] for r in a])
    fig, ax = plt.subplots(figsize=(8.4, 5))
    series = [("laplace_basic", "Laplace, basic composition (Σε)", BLUE, MARKERS[0], "-"),
              ("laplace_zcdp", "Laplace, zCDP (Σρ, converted at δ)", ORANGE, MARKERS[1], "-"),
              ("gaussian_zcdp", "Gaussian (distributed), zCDP", AQUA, MARKERS[2], "--")]
    for key, name, color, marker, ls in series:
        ys = np.array([max(r[key], 0.3) for r in a], dtype=float)          # 0 queries drawn at the floor
        ax.plot(eps, ys, color=color, marker=marker, ls=ls, label=name, markersize=6)
    ax.plot(eps, [r["laplace_ledger"] for r in a], color=INK2, lw=1.0, ls=":", label="Ledger: admits if either account fits")
    cross = next((r["epsilon"] for r in a if r["laplace_basic"] >= r["laplace_zcdp"]), None)
    if cross:
        ax.axvline(cross, color=MUTED, lw=1.0)
        label(ax, cross, 2.0, f"basic wins from ε ≈ {cross:g}", MUTED, dx=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    top = max(r["gaussian_zcdp"] for r in a) * 1.6
    ax.set_ylim(0.22, top)
    ax.set_yticks([0.3] + [10 ** k for k in range(int(np.log10(top)) + 1)])     # the floor tick reads "0"
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v < 1 else f"{v:g}"))
    ax.set_xlabel("ε per query")
    ax.set_ylabel("Identical queries the budget admits")
    legend(ax, loc="upper right")
    b = d["budget"]
    title(ax, "zCDP accounting: many more small queries from the same budget",
          f"researcher budget ε = {b['epsilon']:g}, δ = {b['delta']:g} · Gaussian queries are accounted in zCDP only")
    return save(fig, "budget_accounting")


# --------------------------------------------------------------------------------------------- evaluation 2
def chart_performance(d) -> list[Path]:
    sizes = d["sizes"]
    ns = sorted(int(n) for n in sizes)
    big = str(ns[-1])
    e = sizes[big]
    stages = [("plain (insecure)", "plain", "plain"), ("+ PSI linkage", "psi_warm", "psi_cold"),
              ("+ secret-shared aggregation", "smpc_warm", "smpc_cold"), ("FULL pipeline\n(mTLS + JWT + DP + audit)", "full_warm", "full_cold")]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(stages))[::-1]
    h = 0.36
    warm = [e[w]["median"] for _, w, _ in stages]
    cold = [e[c]["median"] for _, _, c in stages]
    ax.barh(y + h / 2, warm, height=h, color=BLUE, label="Warm: linkage cached")
    ax.barh(y - h / 2, cold, height=h, color=ORANGE, label="Cold: first query runs the PSI linkage")
    for yy, a, b in zip(y, warm, cold):
        ax.text(a * 1.08, yy + h / 2, f"{a * 1000:.0f} ms" if a < 1 else f"{a:.2f} s", va="center", color=INK2, fontsize=9.5)
        ax.text(b * 1.08, yy - h / 2, f"{b * 1000:.0f} ms" if b < 1 else f"{b:.1f} s", va="center", color=INK2, fontsize=9.5)
    ax.set_yticks(y, [s[0] for s in stages])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_xlim(0.008, max(cold) * 2.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median latency of one query (log scale)")
    legend(ax, loc="upper right")
    title(ax, "What each privacy layer costs", f"{big} patients · real processes over HTTP · plain query has no linkage, so warm = cold")
    p1 = save(fig, "performance_stages")

    fig, ax = plt.subplots(figsize=(8.8, 5))
    series = [("Plain (insecure)", "plain"), ("Cold linkage (PSI)", "psi_cold_linkage_s"), ("Secret-shared query, warm", "smpc_warm"),
              ("FULL pipeline, warm", "full_warm"), ("FULL pipeline, cold", "full_cold")]
    for (name, key), color, marker in zip(series, SERIES, MARKERS):
        ys = [sizes[str(n)][key]["median"] for n in ns]
        ax.plot(ns, ys, color=color, marker=marker, label=name)
        short = name.split(" (")[0].replace("Secret-shared query, warm", "SMPC warm").replace("FULL pipeline, ", "FULL ")
        label(ax, ns[-1], ys[-1], short, INK2, dx=9, dy={"FULL cold": 9, "Cold linkage": -9, "FULL warm": 9, "SMPC warm": -9}.get(short, 0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_xlim(ns[0] * 0.85, ns[-1] * 2.6)
    ax.set_xlabel("Patients (people in the world, before the split into hospitals)")
    ax.set_ylabel("Median latency")
    lows = [sizes[str(n)][k]["median"] for n in ns for _, k in series]
    ax.set_ylim(min(lows) * 0.7, max(lows) * 1.8)
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)       # below the plot: the end labels already name each line
    title(ax, "Scaling: PSI linkage is the cost, everything after it is cheap",
          "lists are padded to a public bound, so the work is set by that bound, not the patient count; a cached linkage is reused")
    return [p1, save(fig, "performance_scaling")]


# --------------------------------------------------------------------------------------------- evaluation 3 (stretch)
def chart_federated(d) -> list[Path]:
    a = d["auc"]
    sm = d["small_data"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
    rows = [("FedAvg + secure aggregation", a["fedavg_secure"]), ("FedAvg (updates only)", a["fedavg"]), ("Centralised (data pooled)", a["centralised"]),
            ("Local only: hospital A", a["local_only"]["A"]), ("Local only: hospital B", a["local_only"]["B"]), ("Local only: hospital C", a["local_only"]["C"])]
    y = np.arange(len(rows))[::-1]
    ax1.axvline(0.5, color=MUTED, lw=1.2)
    ax1.axvline(a["true_model"], color=INK2, lw=1.4, ls=(0, (6, 3)))
    ax1.scatter([v for _, v in rows], y, s=70, color=BLUE, zorder=3)
    for yy, (_, v) in zip(y, rows):
        ax1.text(v - 0.005, yy, f"{v:.3f}", color=INK2, fontsize=9.5, ha="right", va="center")
    ax1.set_yticks(y, [n for n, _ in rows])
    ax1.set_xlim(0.49, 0.76)
    ax1.set_ylim(-0.6, 5.95)                      # headroom for the reference-line labels
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("AUC on pooled held-out test rows")
    label(ax1, 0.5, y[0] + 0.62, "chance", MUTED, dx=3)
    label(ax1, a["true_model"], y[0] + 0.62, "true model", INK2, dx=-3, ha="right")
    title(ax1, "Plenty of local data: federation ≈ centralised", f"{sum(v['train'] for v in d['rows_per_hospital'].values()):,} training rows over 3 non-IID hospitals")
    labels = ["Local only\n(mean of 3)", "FedAvg", "Centralised\n(pooled)"]
    series = [sm["local_only"], sm["fedavg"], sm["centralised"]]
    yy = np.arange(3)[::-1]
    ax2.axvline(0.5, color=MUTED, lw=1.2)
    for y_i, vals in zip(yy, series):
        ax2.plot([min(vals), max(vals)], [y_i, y_i], color=BLUE, lw=3, alpha=0.35, solid_capstyle="round")
        ax2.scatter([np.mean(vals)], [y_i], s=80, color=BLUE, zorder=3)
        ax2.text(np.mean(vals), y_i - 0.28, f"{np.mean(vals):.3f}", color=INK2, fontsize=9.5, ha="center", va="top")
    ax2.set_yticks(yy, labels)
    ax2.set_ylim(-0.7, 2.5)
    ax2.set_xlim(0.62, 0.76)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("AUC (dot = mean, bar = min-max over %d subsets)" % len(sm["fedavg"]))
    title(ax2, "Little local data: federation wins", f"only {sm['rows_per_hospital']} training rows per hospital")
    p1 = save(fig, "federated_comparison")

    dp = d["dp"]
    fig, ax = plt.subplots(figsize=(8.4, 5))
    eps = np.array([r["epsilon"] for r in dp])
    mean = np.array([r["auc_mean"] for r in dp])
    ax.fill_between(eps, [r["auc_min"] for r in dp], [r["auc_max"] for r in dp], color=BLUE, alpha=0.16, lw=0)
    ax.axhline(a["fedavg"], color=INK2, lw=1.4, ls=(0, (6, 3)))
    ax.axhline(0.5, color=MUTED, lw=1.2)
    ax.plot(eps, mean, color=BLUE, marker="o", label="FedAvg + clipping + Gaussian noise (mean; band = min-max over seeds)")
    for r in dp:
        label(ax, r["epsilon"], r["auc_mean"], f"z = {r['noise_multiplier']:g}", INK2, dx=9, dy=-12)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0.4, 0.78)
    ax.set_xlabel("ε (client-level, δ = 10⁻⁶, 20 rounds) - smaller is more private →")
    ax.set_ylabel("AUC on pooled held-out test rows")
    ax.invert_xaxis()
    label(ax, eps.min(), a["fedavg"], "no noise", INK2, dx=0, dy=9, ha="right")
    label(ax, eps.max(), 0.5, "chance", MUTED, dx=0, dy=-9, ha="left")
    legend(ax, loc="lower left")
    title(ax, "Privacy has a price: meaningful ε costs most of the accuracy", "z = noise multiplier; only 3 clients, so client-level DP is expensive")
    return [p1, save(fig, "federated_dp")]


def main() -> None:
    style()
    made: list[Path] = []
    for name, fn in (("attack_differencing", chart_differencing), ("attack_reidentification", chart_reid),
                     ("attack_budget_exhaustion", chart_budget), ("accuracy_vs_epsilon", chart_accuracy), ("performance", chart_performance),
                     ("federated", chart_federated)):
        data = load(name)
        if data is None:
            print(f"skip {name}: no results yet")
            continue
        out = fn(data)
        made += out if isinstance(out, list) else [out]
    for p in made:
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
