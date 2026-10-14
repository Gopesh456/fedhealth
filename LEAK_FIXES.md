# Which leaks can be fixed

This note answers "can any of the leaks in `SECURITY_THEOREMS.md` §3.2 be fixed?". Each leak was checked against
the code. A fix was designed for each one, and a second, independent pass then tried to break every fix. All numbers
come from read-only runs on scratch copies of the 5,000-person world, in-process on a 4-core machine.

**Short answer:** yes, all three headline leaks can be fixed.

* The cohort sizes the coordinator sees are cheap to hide.
* The budget cap needs a policy decision.
* Hiding pairwise PSI membership needs a new linkage protocol.

The review also found **seven holes that are not on the leak list**. Several of them break claims the theorems
already make, and most are small fixes. The holes come first in the recommended order (§6).

**Status (built since this note was written):** the holes in §4 and the size and timing fixes (§1, §2.1, §2.2, §4.3)
are now in the code, with tests, and `SECURITY_THEOREMS.md` / `THREAT_MODEL.md` describe the new leakage. Not built:
the hospital odometer (§3, needs a federation cap), the shuffled PSI (§2.3), and the FL noise floor and round limit
(§4.5). The timing fix uses public deadlines rather than pre-drawn noise: see §4.3.

| leak | fixable? | recommended fix | cost | size | status |
|---|---|---|---|---|---|
| L2: coordinator sees exact cohort sizes m_S | fully for the coordinator and network (for the dealer only with §2.3) | pad the linked index to a public size with empty rows | warm query 0.16 → about 0.24 s; no privacy budget, identical answers | S | **done** (M = 4,608) |
| L7: a hospital learns which of its patients the others treat | fully (per patient) | shuffled PSI with oblivious placement | ≈+0.05 s per query, same linkage time; the coordinator learns overlap counts instead | M | not built |
| L8: list sizes | fully, down to a public bound | pad every list with random dummy identifiers | cold linkage +20–30 % (≈11 → 13.6 s with gmpy2, 42 → 55 s without) | S | **done** |
| no federation-wide budget | yes (the cap itself is inherent) | zCDP odometer at every hospital | no noise; a hard lifetime cap on total use | M | not built (needs a cap) |
| new: plan cells not checked | fully | hospitals check that the plan covers every output cell | none | S | **done** |
| new: plans not bound across hospitals | fully | session-header hash in every hospital-to-hospital message | none | S | **done** |
| new: timing (match counts, noise size) | fully for match counts; noise size needs off-path sampling | fixed-latency release, deadlines, pre-drawn noise | a 0.26 s latency floor per query (warm query 0.47 s measured) | S–M | **done** with deadlines (residual: overruns) |
| new: PSI ring reveals MRN order | fully | shuffle the list before PSI | none | S (2 lines) | **done** |
| new: FL noise seeded by the coordinator | fully | draw FL noise from hospital-private randomness | none | S | **done** (noise floor not built) |
| new: failed queries are refunded | fully | charge once any hospital has shared | none | S | **done** |

S = under ~150 lines, M = a few hundred lines, L = a new protocol.

---

## 1. Cohort sizes m_S (L2): fully fixable for the coordinator and the network

**What leaks today.**

* `PsiParty.finalize` returns `len(linked)`, and `routes_psi.finalize` sends it to the coordinator as `PsiFinalizeResult.m`.
* `orchestrator.mpc_run` forwards m to every involved hospital, including the dealer of a two-hospital query.
* Vector lengths, triple counts and every message size are proportional to m.
* The coordinator re-links whenever it likes (`link(force=True)`, and its linkage cache is in memory, so every restart re-links). So L2 is really a **time series**: it shows patients who are shared between hospitals joining and leaving over time.

**Fix (recommended): pad the linked index to a public size M**, for example 4,096, with all-zero rows.

* `finalize` returns M and keeps m local.
* The MPC runs on length-M vectors.
* `build_input_vectors` already writes 0 for every position that has no match. So the dummy rows have indicator 0 at every hospital and output 0 at the owner, and they add exactly 0. This was checked with `smpc.protocol.run_local` for 2 and 3 hospitals and 1, 2 and 8 output cells: the outputs at M = 4,096 are identical to those at the true m.
* **Cost:** the MPC phase scales by M/m, about 1.6–2×. In-process this measured as a COUNT taking 17 → 33 ms and an 8-bin histogram 119 → 258 ms. A warm query goes from 0.16 s to about 0.24 s (the measured breakdown at 5,000 patients without the prepare deadline; with the deadline of §4.3 it measures 0.47 s). No privacy budget is spent and answers are bit-identical.
* **Must ship with:** list padding (§2.2), and public deadlines on PSI `init` and `finalize`. Both of those handlers touch the real list, so their timing still reveals its size.
* **Residual:** members of the linked set still know m_S. That is inherent to any PSI that reveals the intersection; only circuit-PSI hides it. Today a dealer outside the set can also compute m_S from the three-way tag sets (L7). That stops only once §2 is fixed.

**Not recommended: DP padding** (a noisy number of dummy rows). It is cheaper at runtime, but every linkage run spends ε. No single hospital sees all four linked sets, so its odometer caps have to be set at about ⅔ of the target. And the cap becomes a lifetime limit on re-links, coordinator restarts included.

## 2. Pairwise PSI membership (L7) and list sizes (L8)

**What leaks today.**

* `PsiParty.publish` and `routes_psi.publish` send every hospital's full tag set to every other hospital. So each hospital learns, for each of its own patients, which other hospitals hold that patient, plus every region size of the Venn diagram.
* Each round of the ring hands a hospital another hospital's whole encrypted list, which gives that list's size (L8).
* **New:** `routes_psi.init` loads ids `ORDER BY patient_id`, and MRNs are dense. The hospital just before the owner in the ring computes the owner's fully encrypted list *in MRN order*, can tag it, and so learns membership by MRN rank. Sorting at `publish` does not prevent this.
* **New:** every re-link reveals current pairwise membership again. So hospitals learn admission events: which of their patients newly became, or stopped being, patients elsewhere.

**Fix 2.1: shuffle each list before PSI.** Two lines in `routes_psi.init`. It closes the MRN-order leak today, and the other L7 fixes require it.

**Fix 2.2: pad every list to one public bound N with random dummy identifiers (L8).**

* Cost: cold linkage +20–30 % (≈11 → 13.6 s with gmpy2, 42 → 55 s without, at 5,000 patients; padded, it takes the same time at every size). Warm queries are unaffected because the linkage is cached.
* No DP accounting is needed.
* It also evens out PSI timing and message sizes.

**Fix 2.3 (recommended for L7): shuffled DH-PSI with oblivious placement.**

* Each pair of hospitals shares a secret permutation, and lists are shuffled and re-exponentiated, so no hospital can map a tag back to one of its own patients.
* The last ring hop sends the finished tags to the coordinator, which aligns them. Nothing goes back to the owner.
* Hospitals receive their input positions through re-randomised shares.
* **Closes:** every per-patient membership bit, including today's knowledge of which own patients are in the intersection.
* **Cost:** the same number of exponentiations, so the cold linkage is unchanged; one extra round per query, ≈+0.02–0.05 s. No noise or budget cost.
* **The independent pass found three conditions** without which it breaks:
  * every PRG mask must be separated by (session, input, vector index). Without the vector index, an owner with two or more vectors recovers its own intersection slots;
  * the last hop must never return to the owner;
  * list padding (Fix 2.2) is mandatory.
* **Residual:** hospitals learn only m_S and N. The **coordinator now learns the Venn region sizes** instead, which is a trade worth stating in L2.
* Size M, plus a new proof step in Theorem 1.

**Stronger, later:** an index over everyone held by at least two hospitals, with secret-shared membership bits. It hides m_S from hospitals too, at 1.8× longer vectors and one extra Beaver layer. Full circuit-PSI hides everything, including the Venn sizes from everyone. It needs OT or OPPRF and would take minutes per linkage in pure Python, so it is future work.

## 3. No federation-wide budget: fixable, but the cap is a policy choice

**What leaks today.**

* The only budget is the coordinator's per-requester ledger, and hospitals keep no state across queries.
* So Theorem 2(b) grows with every account, Sybil accounts get full budgets, and a coordinator that does not follow the protocol can call the MPC endpoints directly and average unlimited capped answers.

**Fix: a zCDP odometer at every hospital.**

* Each hospital persists Σρ for every noise plan it contributes a share to, and refuses once the federation cap is reached.
* That makes coordinator replay and Sybil accounts harmless *for privacy*. They can still use up the shared budget, which is an availability problem, not a privacy one.
* **The independent pass showed it only holds with these conditions:**
  * hospitals must first validate the plan's groups (§4.1) and bind the plan across hospitals (§4.2). Otherwise the coordinator can drive the charge to almost nothing;
  * the charge must come from the mechanism and the hospital's own recomputation, never from the ε or δ the plan claims. A Gaussian plan claiming ε = 10⁻⁶ got 5,000 of 5,000 queries past a basic-composition check;
  * the hospital buckets must be zCDP-only. The ledger's pure-ε route makes a ρ sum invalid;
  * ρ must be split between multi-hospital and single-hospital queries, because a single-hospital query is metered only by that hospital;
  * a query must be charged, not refunded, once any hospital has shared (§4.6).
* **Cost:** no extra noise, but a **hard lifetime cap on total use**:
  * at (ε 10, δ 10⁻⁷): about 270 counts at ε 0.1, 43 at ε 0.25 or 19 at ε 0.37;
  * at (ε 20, δ 10⁻⁶): about 957 counts at ε 0.1 or 9 at ε 1.
* Requester budgets should become slices of that cap. The roles' per-query limits (1.0 and 2.0) must drop below the hospitals' per-query limit, or most queries are refused after PSI has already run.
* **Inherent:** some finite cap is unavoidable for any system that answers many counts accurately (Dinur–Nissim reconstruction). Choosing it is a governance decision.
* Size M (≈500–700 lines with tests).

## 4. Holes the review found that are not on the leak list

1. **Hospitals do not check that the noise plan covers every output cell.**
   * `NoisePlan.check` sums ε only over `plan.groups`, which the coordinator supplies.
   * A SUM plan whose groups leave the total cell uncovered, at near-zero noise, passed at all three hospitals. The exact total (4,581) came back on 3 of 3 runs.
   * This contradicts assumption A7 and the THREAT_MODEL row "hospitals refuse a weaker plan".
   * **Fix:** require the groups to be the fixed partition for the aggregate. ≈15–25 lines.
2. **Plans are not bound across hospitals.**
   * The coordinator builds each hospital's request separately, and hospital-to-hospital messages carry no session digest.
   * Telling the non-owners "histogram, 2 cells" while the owner gets "sum" passed every check, and gave the coordinator plus the owner ε 4 against a cap of 2.
   * **Fix:** a hash of the session header (parties, owner, aggregate, cells, m, linkage, noise plan) in every hospital-to-hospital message, with the computing hospitals refusing a mismatch. ≈30 lines.
3. **Timing.**
   * **Match counts:** `prepare` time grows linearly with how many rows the hospital's predicate matches. Measured 33 ms at 0 matches vs 50 ms at 1,400 (≈11.5 µs per row, r = 0.975), so a requester can estimate counts outside DP.
   * **Noise size:** the Pólya sampler loops once per unit of the drawn value, so for SUM/AVG queries latency follows the size of the noise share (r ≈ 0.6). At very small ε a SUM can hit the 300 s client timeout, and the run is then refunded.
   * **Fix:**
     * release every answer at a fixed public latency;
     * put public deadlines on every hospital control endpoint;
     * move noise sampling off the request path (pre-drawn shares on a public ε grid);
     * set a minimum ε for SUM/AVG.
   * **Built:** public deadlines on the calls that touch data or draw noise (PSI init, finalize and delete, MPC
     prepare, FL train; every other call does work fixed by the padded size). The prepare deadline grows with the
     plan's public noise scale (100 µs per unit per cell, capped at 60 s), and hospitals refuse a Laplace scale above
     500,000 per cell, which is the "minimum ε" in another form. Each guarded call runs alone at its hospital: it
     waits for requests in progress to finish, and nothing else starts there until its deadline has passed, so the
     coordinator cannot time a health probe or another query's step against it either (the review showed both
     attacks working before this). An overrun is answered at the next doubling of the deadline, so it shows only
     that bucket; its probability β enters Theorem 2 as an extra (1 + e^ε)·β per call. β is about 10⁻¹¹ for the
     noise draw and assumed for the rest; it holds only if each hospital has its own machine and is not flooded.
     Pre-drawn noise and a separate worker process per hospital would remove the residual; they are not built.
     Measured at 5,000 patients: the slowest prepare over six query shapes took 0.11 s against the 0.26 s deadline,
     and none of 60 warm queries overran. (The first measurement found an AVG prepare at 0.18–0.26 s, because each row
     re-read the field configuration; it now reads it once per query and takes 0.04 s.)
4. **The PSI ring reveals MRN order** (§2): shuffle, 2 lines.
5. **Federated learning.**
   * Hospitals draw their FL noise from `default_rng([seed, round, hospital])`, and the coordinator picks the seed. So the coordinator can regenerate the noise and subtract it exactly. Verified against the real endpoint: update norm 3.95 → 0.61, inside the clip.
   * FL also accepts zero noise, unlimited rounds and plain updates in strict mode.
   * **Fix:**
     * hospital-private randomness (≈5 lines);
     * FL off by default in strict mode;
     * a noise floor;
     * rounds charged to the odometer.
6. **Failed queries are refunded** (`service.execute` refunds every exception). Under any misbehaving party, data-dependent aborts become a free channel. **Fix:** charge once any hospital has shared.
7. **Re-linking** reveals a time series of m_S to the coordinator and admission events to hospitals. This is fixed by §1 and §2.

## 5. What cannot be fixed

* Members of a linked set know its size, as long as the PSI reveals the intersection.
* The coordinator sees query metadata: which hospitals, the aggregate family, ε. It issues the queries.
* A hospital can lie about its own data within range. Even malicious-secure MPC allows input substitution.
* Some finite total budget is required.
* On the network, who talks to whom and when, unless the system sends constant cover traffic.

**Trust-model items.** These are fixable, but each is a new protocol, so they are future work:

* **malicious hospitals:** SPDZ-style MACs, or MP-SPDZ. Watch for selective-failure attacks through the dealer's input masks;
* **two colluding hospitals:** 3-of-3 sharing with OT-based multiplication. This is feasible in pure Python, but it needs tolerance t = 2, which roughly doubles the Laplace error relative to central noise;
* **fabricated PSI lists:** only certified enrolment tokens, or not revealing membership at all (§2.3), stop a hospital from probing for a specific person.

## 6. Corrections to SECURITY_THEOREMS.md

* **L10** says the dealer of a two-hospital query learns "not the condition". That is wrong: the plan's sensitivities and the owner name the value field and often the condition. For example, owner B with sensitivity 200,000 means lab_delta for heart failure.
* **L2** is a time series, not "at most four numbers".
* **L5 and L12** should list match-count and noise-size timing.
* **A7** ("hospitals refuse a weaker plan") is not true until §4.1 and §4.2 land.

**Recommended order:**

1. Fix the holes: §4.1, §4.2, §4.4, §4.5 and §4.6, plus the corrections above. They are small and repair claims already made.
2. Index and list padding with the timing fixes: §1, §2.2 and §4.3.
3. The hospital odometer, once a federation cap is chosen: §3.
4. The shuffled PSI: §2.3.
