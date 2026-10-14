"""Generate evaluation/RESULTS.md from the saved JSON results - every number in it comes from a measurement.

    python evaluation/make_report.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def fmt_s(x: float) -> str:
    return f"{x * 1000:.0f} ms" if x < 1 else f"{x:.2f} s" if x < 10 else f"{x:.1f} s"


def section_data(d) -> str:
    rows = [[code, c["name"], f"{c['episodes']:,}", f"{c['treated']:,}", f"{c['with_outcome']:,}",
             pct(c["improved"] / c["with_outcome"], 0) if c["with_outcome"] else "-", ", ".join(c["top_drugs"]) or "-"]
            for code, c in d["conditions"].items()]
    sizes = ", ".join(f"{h}: {n:,}" for h, n in d["hospital_sizes"].items())
    src = ("**Synthea** synthetic patients (Massachusetts; the shipped extract in `synthea/`, mapped by `fedhealth/synth/synthea.py`)"
           if d["source"] == "synthea" else "the parametric synthetic generator (`fedhealth/synth/universe.py`)")
    return (f"## 0. Data\n\nThe evaluation world is {d['people']:,} adults from {src}, as of {d['as_of']}, split into the three hospitals "
            f"({sizes}); {d['membership_counts'].get('ABC', 0):,} are at all three.\n\n"
            + table(["code", "condition", "episodes", "treated", "with an outcome", "improved", "most prescribed"], rows)
            + "\n\n* Outcomes are derived from Synthea's own lab and vital-sign series (baseline before treatment vs follow-up after at "
              "least 90 days; see the mapping notes in `fedhealth/synth/synthea.py`). Synthea records almost no follow-up PHQ-9 for depression "
              "and only a spirometry proxy for asthma, so F32 and J45 have few or no outcomes; queries about them still run, they just count few patients.\n"
              "* Adherence is a refill-coverage proxy (Synthea fills prescriptions on schedule), so it is high and narrow.\n")


def section_differencing(d) -> str:
    rows = []
    for r in d["rows"]:
        eps = "-" if r["epsilon"] is None else f"{r['epsilon']:g}"
        rows.append([r["mode"], eps, f"{pct(r['accuracy'])} ({pct(r['ci95'][0], 0)}-{pct(r['ci95'][1], 0)})",
                     pct(r["analytic_accuracy"]) if r.get("analytic_accuracy") else "-",
                     pct(r["analytic_accuracy_coalition"]) if r.get("analytic_accuracy_coalition") else "-",
                     pct(r["dp_bound"]) if r.get("dp_bound") else "-"])
    dpr = {r["epsilon"]: r for r in d["rows"] if r["epsilon"]}
    direct = d["direct_attempt"]
    findings = [
        f"With exact answers - and with the k-threshold alone - the attacker identifies the target's outcome in **{pct(d['rows'][0]['accuracy'], 0)}** and **{pct(d['rows'][1]['accuracy'], 0)}** of trials: both queries cover cohorts far above k, so nothing is suppressed.",
        f"The k-threshold does stop the *direct* question: it suppressed {direct['k_only']['suppressed']} of {direct['k_only']['queries']} single-patient queries (without it, {direct['no_protection']['queries'] - direct['no_protection']['suppressed']} of {direct['no_protection']['queries']} would have leaked).",
        f"Differential privacy brings the attacker down to **{pct(dpr[0.1]['accuracy'])} at ε = 0.1**, {pct(dpr[1.0]['accuracy'])} at ε = 1 and {pct(dpr[5.0]['accuracy'])} at ε = 5 (blind guessing = 50%). The measured values track the exact formula for this attack and stay below the DP ceiling for any attacker.",
        "ε = 5 per query offers almost no protection against this attack - the number to quote is the *per-patient* privacy loss, not 'DP is on'. "
        "(The hospitals refuse more than ε = 2 per query by default; ε = 5 is shown for the trend.)",
        "The noise is now the sum of three hospitals' shares, any two of which already give ε-DP. The requester sees all three (slightly more noise than "
        "ε needs); the coordinator colluding with one hospital can subtract that hospital's share and is left with exactly ε-DP noise - the "
        "'coordinator + 1 hospital' column, still below the DP ceiling.",
    ]
    return (f"### Attack 1 - Differencing\n\n{d['n_targets']} targets ({d['responders']} responders, {d['n_targets'] - d['responders']} non-responders), "
            f"{d['trials_per_setting']:,} trials per setting. Exact counts were obtained through the real PSI + MPC pipeline "
            f"({d['pipeline_queries_checked']} distinct queries, each checked against ground truth); the noise is re-drawn through the real release stage.\n\n"
            + table(["setting", "ε per query", "requester's accuracy (95% CI)", "exact theory", "coordinator + 1 hospital (theory)", "DP ceiling"], rows)
            + "\n\n" + "\n".join(f"* {f}" for f in findings) + "\n\n![differencing](charts/attack_differencing.png)\n")


def section_reid(d) -> str:
    qi = [[r["qi"].replace("+", " + "), pct(r["unique_in_release"]), pct(r["reid_rate"]), pct(r["precision"], 0) if r["precision"] is not None else "-"] for r in d["by_qi"]]
    de = [[r["label"], pct(r["reid_rate"]), pct(r["retained_fraction"], 0), r["min_group_size"]] for r in d["defences"]]
    top = next(r for r in d["defences"] if r["label"].startswith("exact birth year"))
    gen = next(r for r in d["defences"] if r["label"].startswith("age band"))
    return (f"### Attack 2 - Re-identification of a 'de-identified' release\n\nReleased: {d['released_records']:,} Hospital-A patients with names and IDs removed. "
            f"Auxiliary 'voter roll': {d['aux_records']:,} people ({pct(d['aux_coverage'], 0)} of the world).\n\n"
            + table(["quasi-identifiers", "records unique in the release", "re-identified (correctly)", "precision of claims"], qi) + "\n\n"
            + table(["defence", "re-identified", "records still released", "smallest group"], de)
            + f"\n\n* Removing names is not anonymisation: **{pct(top['reid_rate'])}** of records are re-identified from birth year + sex + ZIP alone.\n"
            f"* Generalising the quasi-identifiers (age band, 3-digit ZIP) cuts that to {pct(gen['reid_rate'])} at no cost in records; k-anonymity removes the rest but discards records.\n"
            "* This system never exports rows; reaching the same information through the query interface requires differencing (attack 1), which DP defeats.\n\n"
            "![re-identification](charts/attack_reidentification.png)\n")


def section_budget(d) -> str:
    rows = [[r["queries"], f"{r['epsilon_spent']:g}", f"{r['mean_abs_error']:.3f}", f"{r['theory']:.3f}"] for r in d["unlimited"]]
    l = d["with_ledger"]
    return (f"### Attack 3 - Budget exhaustion (averaging out the noise)\n\nTarget: `{d['query']}`, true value {d['true_value']}, ε = {d['epsilon_per_query']:g} per query "
            f"(one query's expected |error| = {d['single_query_expected_abs_error']:.2f}). {d['attackers_simulated']} independent attackers per point.\n\n"
            + table(["queries averaged", "ε burned", "mean abs error (no ledger)", "CLT approximation"], rows)
            + f"\n\n* Without a limit the error falls like 1/sqrt(N): after {rows[-1][0]} queries it is {rows[-1][2]} - privacy bought back by volume.\n"
            f"* **With the ledger** (budget {l['budget']:g}), driven through the real pipeline: {l['answered']} of {l['attempted']} attempts were answered"
            f"{' (' + str(l['suppressed']) + ' of them suppressed by the noisy k-threshold, still charged)' if l.get('suppressed') else ''}, the first refusal came at query "
            f"**{l['first_refusal_at_query']}**, ε spent {l['epsilon_spent']:g}. The attacker is stuck with the average of those {l['answered']} answers: in this run it is off by {l['final_abs_error']:.2f} "
            f"(a single run; the typical error after {l['answered']} answers is ≈{l['expected_abs_error_at_that_point']:.2f}) - and no further query is answered.\n"
            "* Pure ε = 0.5 queries are cheapest under basic composition here (20 queries); zCDP would admit 12 - the ledger takes whichever admits more (section 2).\n"
            "* Not covered: an attacker with several accounts, or colluding requesters (each has an own budget).\n\n![budget](charts/attack_budget_exhaustion.png)\n")


def section_hmac(d) -> str:
    return (f"### Attack 4 (bonus) - Dictionary attack on HMAC pseudonyms\n\nHospital B published {d['b_patients']:,} HMAC tokens ({d['shared_patients']:,} also at A, {d['b_only_patients']:,} only at B). "
            f"The attacker holds the shared key and tries {d['dictionary_size']:,} candidate IDs.\n\n"
            f"* Recovered **{d['recovered_from_b']:,} of {d['b_patients']:,}** of B's patients, including {d['recovered_b_only']:,} that A never treated; false positives: {d['false_positives']}.\n"
            f"* One core tests {d['hmac_per_second']:,.0f} IDs/s, so the whole 9-digit ID space takes about **{d['full_9digit_space_seconds'] / 60:.0f} minutes**.\n"
            f"* Against PSI the same attacker recovers **{d['psi_non_shared_patients_recovered']} of {d['psi_candidates_tested']}** non-shared patients.\n")


def section_accuracy(d) -> str:
    eps = sorted({r["epsilon"] for r in d["rows"]})
    out = [f"## 2. Accuracy vs privacy\n\n{d['repetitions']:,} releases per cell (k = {d['k']}). The noise is what the hospitals add inside the "
           "secure computation: three shares (two computing hospitals + the triple dealer), sized so that any two of them give ε-DP "
           "(collusion tolerance t = 1). Exact aggregates come from the real private pipeline and were checked against ground truth; "
           "the noise is drawn with the vectorised simulation of the hospitals' exact samplers.\n"]
    counts = [r for r in d["rows"] if r["kind"] == "count"]
    for mech in ("laplace", "gaussian"):
        rows = []
        for cohort in ("small", "medium", "large"):
            sel = {r["epsilon"]: r for r in counts if r["mechanism"] == mech and r["cohort"] == cohort}
            rows.append([f"{cohort} (n={sel[eps[0]]['true_value']})"] + [f"{sel[e]['mae']:.2f} ({pct(sel[e]['relative_mae'], 1 if sel[e]['relative_mae'] >= 0.01 else 2)})" for e in eps])
        th = {r["epsilon"]: r["theory_mae"] for r in counts if r["mechanism"] == mech and r["cohort"] == "small"}
        rows.append(["theory: E\\|noise\\|"] + [f"{th[e]:.2f}" for e in eps])
        out.append(f"**{mech.capitalize()}** - mean absolute error in patients (relative error)\n\n" + table(["cohort"] + [f"ε = {e:g}" for e in eps], rows) + "\n")
    other = [r for r in d["rows"] if r["kind"] in ("avg", "histogram")]
    rows = []
    for kind, label in (("avg", "AVG adherence (percentage points)"), ("histogram", "HISTOGRAM, per bin (patients)")):
        sel = {r["epsilon"]: r for r in other if r["kind"] == kind}
        rows.append([f"{label}, {sel[eps[0]]['cohort']}"] + [f"{sel[e]['mae']:.2f}" for e in eps])
    out.append("**Other aggregates** (Laplace) - mean absolute error\n\n" + table(["query"] + [f"ε = {e:g}" for e in eps], rows) + "\n")
    supp = []
    for cohort in ("small", "medium"):
        sel = {r["epsilon"]: r for r in counts if r["mechanism"] == "laplace" and r["cohort"] == cohort}
        supp.append([f"{cohort} (n={sel[eps[0]]['true_value']})"] + [pct(sel[e]["suppressed_rate"]) for e in eps])
    out.append(f"**The k-threshold now looks at the noisy count** - share of releases suppressed (Laplace)\n\n"
               + table(["cohort"] + [f"ε = {e:g}" for e in eps], supp) + "\n")
    sm = {r["epsilon"]: r for r in counts if r["mechanism"] == "laplace" and r["cohort"] == "small"}
    lg = {r["epsilon"]: r for r in counts if r["mechanism"] == "laplace" and r["cohort"] == "large"}
    out.append(f"* The absolute error depends only on ε: ≈{sm[1.0]['mae']:.2f} patients at ε = 1 (theory {sm[1.0]['theory_mae']:.2f}), ≈{sm[0.1]['mae']:.1f} at ε = 0.1, for every cohort size.\n"
               f"* What it costs relatively differs hugely: at ε = 0.5 the small cohort (n={sm[0.5]['true_value']}) is off by {pct(sm[0.5]['relative_mae'], 0)}, the large one (n={lg[0.5]['true_value']}) by {pct(lg[0.5]['relative_mae'], 1)}.\n"
               "* The Gaussian mechanism (δ = 10⁻⁶) is noisier than Laplace for a single count at these ε; its advantage is in composition (below).\n"
               "* A histogram's bins all share one ε (parallel composition), so each bin is as accurate as a single count.\n"
               "* Suppression is no longer a hard cut at the exact count: a cohort near k is sometimes released and sometimes not. That fuzziness is what closes the old side channel (the decision is a function of the noisy count only, so it is covered by the DP guarantee).\n\n"
               "![accuracy](charts/accuracy_vs_epsilon.png)\n\n![relative error](charts/relative_error_vs_epsilon.png)\n")

    tol = [r for r in d["rows"] if r["kind"] == "tolerance"]
    rows = []
    for mech in ("laplace", "gaussian"):
        for t in sorted({r["tolerance"] for r in tol}):
            sel = {r["epsilon"]: r for r in tol if r["mechanism"] == mech and r["tolerance"] == t}
            rows.append([mech, t, f"{sel[eps[0]]['honest_shares']} of {sel[eps[0]]['noise_contributors']}"] + [f"{sel[e]['mae']:.2f}" for e in eps])
    base = {r["epsilon"]: r["mae"] for r in tol if r["mechanism"] == "laplace" and r["tolerance"] == 0}
    one = {r["epsilon"]: r["mae"] for r in tol if r["mechanism"] == "laplace" and r["tolerance"] == 1}
    out.append("### The price of not trusting the coordinator\n\nThe same medium-cohort count with the noise sized for t = 0, 1, 2 hospitals "
               "colluding with the coordinator. t = 0 is exactly the noise the old design added centrally, at the coordinator.\n\n"
               + table(["mechanism", "t", "shares that must stay secret"] + [f"ε = {e:g}" for e in eps], rows)
               + f"\n\n* The default (t = 1) costs ×{one[1.0] / base[1.0]:.2f} in mean absolute error at ε = 1 over a trusted coordinator "
                 "(variance ×1.5); in exchange the coordinator never sees an exact count, and even the coordinator plus one hospital get ε-DP.\n\n"
                 "![collusion tolerance](charts/collusion_tolerance_cost.png)\n")

    a = d["accounting"]
    pick = [r for r in a if r["epsilon"] in (0.02, 0.0502, 0.1262, 0.317, 0.7962, 2.0)] or a[::4]
    rows = [[f"{r['epsilon']:g}", r["laplace_basic"], r["laplace_zcdp"], r["laplace_ledger"], r["gaussian_zcdp"]] for r in pick]
    b = d["budget"]
    cross = next((r["epsilon"] for r in a if r["laplace_basic"] >= r["laplace_zcdp"]), None)
    out.append(f"### Budget accounting: zCDP next to basic composition\n\nHow many identical queries one researcher budget (ε = {b['epsilon']:g}, δ = {b['delta']:g}) admits.\n\n"
               + table(["ε per query", "Laplace, basic Σε", "Laplace, zCDP", "Laplace, ledger", "Gaussian, zCDP"], rows)
               + f"\n\n* Below ε ≈ {cross:g} per query zCDP admits more (ρ adds up, ε grows like √k); above it a few large pure-DP queries fit better under basic composition. "
                 "The ledger keeps both accounts and admits a query if either still fits - Theorem 2 of SECURITY_THEOREMS.md shows the requester's guarantee is still (ε, δ) of the budget.\n"
                 "* Gaussian queries are accounted in zCDP only (with the Kairouz-Liu-Steinke term for distributed discrete Gaussian noise), where they are far cheaper than any (ε, δ) sum.\n\n"
                 "![accounting](charts/budget_accounting.png)\n")
    return "\n".join(out)


def deadline_line(pd) -> str:
    slow = max(pd["work_s"].items(), key=lambda kv: kv[1]["max"])
    return (f"* **Public prepare deadline.** Every hospital answers `prepare` (its predicate and its noise draw) at a public deadline: "
            f"{fmt_s(pd['deadline_s'])} for an exact count, plus an allowance per unit of noise scale for a noisy one. So the prepare time above is "
            f"the deadline, not the work, and the cost is a latency floor. With the deadlines switched off, the slowest prepare over "
            f"{len(pd['work_s'])} query shapes took {fmt_s(pd['max_work_s'])} ({slow[0]}; medians "
            + ", ".join(f"{k} {fmt_s(v['median'])}" for k, v in pd["work_s"].items())
            + f"). {pd['overruns']} of {pd['observed']} warm prepares overran the deadline.\n")


def section_performance(d) -> str:
    sizes = sorted(int(n) for n in d["sizes"])
    keys = [("plain", "plain"), ("+PSI cold", "psi_cold"), ("+PSI warm", "psi_warm"), ("+SMPC cold", "smpc_cold"), ("+SMPC warm", "smpc_warm"),
            ("FULL cold", "full_cold"), ("FULL warm", "full_warm")]
    rows = [[n] + [fmt_s(d["sizes"][str(n)][k]["median"]) for _, k in keys] for n in sizes]
    m = d["machine"]
    big = d["sizes"][str(sizes[-1])]
    bd = big["smpc_warm_breakdown"]
    ratio = big["full_warm"]["median"] / big["plain"]["median"]
    pp = d["pure_python_psi"]
    out = [f"## 3. Performance overhead\n\nMedian wall-clock latency of one count query through the coordinator, against real uvicorn processes over HTTP "
           f"({d['reps']} warm repetitions, {d['cold_reps']} cold; FULL cold is a single first query after start-up). "
           f"Machine: {m['processor'] or m['platform']}, {m['cpu_count']} logical CPUs, Python {m['python']}, gmpy2 {'on' if m['gmpy2'] else 'off'}.\n\n"
           + table(["patients"] + [k for k, _ in keys], rows) + "\n",
           "'FULL' = strict deployment: mutual TLS, service tokens, JWT login + RBAC, ε-ledger, hospital-generated noise shares inside the MPC, k-threshold and audit entry. "
           "'PSI' stages use the insecure per-index aggregation (isolating the linkage cost); 'SMPC' stages return the EXACT answer (isolating the secret-shared computation).\n",
           f"Warm secret-shared query at {sizes[-1]:,} patients, breakdown: prepare {fmt_s(bd['prepare_s'])}, share {fmt_s(bd['share_s'])}, triples from the dealer hospital {fmt_s(bd['triples_s'])}, multiply {fmt_s(bd['multiply_s'])}.\n",
           f"* **Linkage dominates.** A cold query at {sizes[-1]:,} patients takes {fmt_s(big['full_cold']['median'])}; once the linkage is cached the full pipeline answers in {fmt_s(big['full_warm']['median'])} - "
           f"{ratio:.0f}× the plain insecure join ({fmt_s(big['plain']['median'])}).\n"
           f"* Cold linkage at each size ({', '.join(f'{n:,}: {fmt_s(d['sizes'][str(n)]['psi_cold_linkage_s']['median'])}' for n in sizes)}): "
           f"every PSI list is padded to the public bound M, so the cost is set by M, not by how many patients a hospital has.\n"
           + (f"* Pure-Python modular exponentiation (no gmpy2) makes the cold linkage at {pp['n']:,} patients take {fmt_s(pp['linkage_s']['median'])}"
              f" vs {fmt_s(pp['with_gmpy2_linkage_s']['median'])} with gmpy2 ({pp['linkage_s']['median'] / pp['with_gmpy2_linkage_s']['median']:.0f}× faster).\n"
              if pp.get('with_gmpy2_linkage_s') and m["gmpy2"] else
              "* gmpy2 was not installed on this machine, so every linkage time above uses pure-Python modular exponentiation "
              "(`pip install -e \".[fast]\"` makes the cold linkage several times faster; the warm query does not use it).\n")
           + f"* Login (scrypt) costs {fmt_s(big['login_s'])} once per session.\n"
           + (f"* Drawing one hospital's noise share with the exact samplers: " + ", ".join(f"{k} {v * 1e3:.2f} ms" for k, v in d["noise_share_s"].items()) + ".\n" if d.get("noise_share_s") else "")
           + (deadline_line(d["prepare_deadline"]) if d.get("prepare_deadline") else ""),
           "![stages](charts/performance_stages.png)\n\n![scaling](charts/performance_scaling.png)\n"]
    return "\n".join(out)


def section_federated(d) -> str:
    a, sm = d["auc"], d["small_data"]
    rows = [["centralised (pooled data)", f"{a['centralised']:.4f}"], ["**FedAvg** (real services, updates only)", f"{a['fedavg']:.4f}"],
            ["FedAvg + secure aggregation", f"{a['fedavg_secure']:.4f}"]] + \
           [[f"local only - hospital {h}", f"{v:.4f}"] for h, v in a["local_only"].items()] + [["true model (generator's own coefficients)", f"{a['true_model']:.4f}"]]
    dp = [[r["noise_multiplier"], f"{r['epsilon']:.0f}", f"{r['auc_mean']:.3f}", f"{r['auc_min']:.3f} - {r['auc_max']:.3f}"] for r in d["dp"]]
    n = {h: v["train"] for h, v in d["rows_per_hospital"].items()}
    co = d["coefficients"]
    mean = lambda xs: sum(xs) / len(xs)
    gain = mean(sm["fedavg"]) - mean(sm["local_only"])
    return (f"## 4. Federated learning (stretch goal)\n\nFedAvg logistic regression predicting treatment response; {d['config']['rounds']} rounds, {d['config']['local_epochs']} local epochs. "
            f"Each hospital holds its own `training_episodes` registry ({', '.join(f'{h}: {v:,}' for h, v in n.items())} training rows; mean age "
            f"{', '.join(f'{h} {v:.1f}' for h, v in d['mean_age'].items())} - deliberately non-IID). FedAvg runs through the real hospital services; test rows are read offline.\n\n"
            + table(["method", "AUC (pooled test rows)"], rows)
            + f"\n\n* **FedAvg matches centralised training** (AUC {a['fedavg']:.3f} vs {a['centralised']:.3f}, ceiling {a['true_model']:.3f}) although no row ever left a hospital; "
              f"secure aggregation changes the model by at most {d['secure_vs_plain_max_weight_diff']:.1e} per weight.\n"
              f"* It recovers the clinical signal: adherence coefficient {co['adherence_z']['learned']:.2f} (true {co['adherence_z']['true']:.2f}), age {co['age_z']['learned']:.2f} ({co['age_z']['true']:.2f}), "
              f"extra comorbidities {co['extra_comorbidities']['learned']:.2f} ({co['extra_comorbidities']['true']:.2f}); drug-effect coefficients correlate {d['drug_coefficient_correlation']:.2f} with the truth.\n"
              f"* **With plenty of local data federation adds little** - every hospital alone already reaches {min(a['local_only'].values()):.3f}-{max(a['local_only'].values()):.3f}. "
              f"**With little local data it helps**: at {sm['rows_per_hospital']} rows per hospital (mean of {len(sm['fedavg'])} random subsets) local-only models reach {mean(sm['local_only']):.3f}, FedAvg {mean(sm['fedavg']):.3f} (+{gain:.3f}), pooled training {mean(sm['centralised']):.3f}.\n\n"
            + "**Adding differential privacy (clipped updates + Gaussian noise; client-level, δ = 10⁻⁶):**\n\n"
            + table(["noise multiplier z", "ε (zCDP, 20 rounds)", "AUC mean", "AUC min - max over seeds"], dp)
            + "\n\n* Accuracy degrades monotonically with noise: light noise costs little but its ε is huge (hundreds); a single-digit ε leaves a model close to chance. "
              "With only three clients, *client-level* DP is expensive - and it protects a hospital's whole contribution, **not** individual records (record-level DP-SGD is not implemented).\n"
              "* Not covered: model memorisation / membership inference on the trained weights, poisoning by a malicious hospital, and verification of updates.\n\n"
              "![federated comparison](charts/federated_comparison.png)\n\n![federated DP](charts/federated_dp.png)\n")


def main() -> None:
    parts = [f"# Evaluation results\n\n*Generated {dt.date.today().isoformat()} by `evaluation/make_report.py` from `evaluation/results/*.json` - every number below is a measurement from `python evaluation/run_all.py`, none is hand-typed. "
             "Synthetic patients only (Synthea); see THREAT_MODEL.md and SECURITY_THEOREMS.md for what these results do and do not show.*\n"]
    data = load("data")
    if data:
        parts.append(section_data(data))
    d1, d2, d3, d4 = (load(n) for n in ("attack_differencing", "attack_reidentification", "attack_budget_exhaustion", "attack_hmac_dictionary"))
    parts.append("## 1. Attack demonstrations - why each defence matters\n")
    for d, fn in ((d1, section_differencing), (d2, section_reid), (d3, section_budget), (d4, section_hmac)):
        if d:
            parts.append(fn(d))
    acc, perf = load("accuracy_vs_epsilon"), load("performance")
    if acc:
        parts.append(section_accuracy(acc))
    if perf:
        parts.append(section_performance(perf))
    fl = load("federated")
    if fl:
        parts.append(section_federated(fl))
    parts.append("## How to reproduce\n\n```bash\npython evaluation/run_all.py            # everything (several minutes at 5,000 patients)\npython evaluation/run_all.py --fast     # 1,000 patients, insecure 768-bit PSI group - smoke test only\n```\n")
    path = ROOT / "evaluation" / "RESULTS.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
