# Security and privacy theorems

This document states what FedHealth guarantees, under which assumptions, and **exactly what leaks**. It has two
results:

* **Theorem 1 (secure computation with explicit leakage).** Against a semi-honest coordinator that may collude with
  any *one* hospital, a run of the private pipeline reveals nothing beyond an explicit leakage list (§3.2) and the
  *noisy* output. In particular the coordinator never sees an exact aggregate and never holds a protocol secret.
* **Theorem 2 (end-to-end differential privacy).** Everything a requester receives over the life of its account is
  (ε, δ)-differentially private for the budget (ε, δ) of that account, with the k-threshold, the suppression
  decision and all post-processing included. Against the coordinator, alone or with one hospital, the noisy outputs of
  all queries are differentially private as well (computationally, and apart from the leakage list).

The proofs are sketches at the level of a systems paper: every step names the standard result it rests on (§6).
Where the code enforces an assumption or a step, §5 points to the file and the test.

**What changed** (the design these theorems are about):

| | before | now |
|---|---|---|
| who adds the DP noise | the coordinator, after opening the exact aggregate | every hospital in the computation adds one secret-shared noise share **inside** the MPC; only the noisy cell is ever opened |
| who sees an exact count | the coordinator | nobody (except a hospital for its own single-hospital query, which it can compute from its own data anyway) |
| Beaver triples | dealt by the coordinator | dealt by a third hospital that is not a computing party; P0 receives a 32-byte seed, P1 the correction terms |
| k-threshold | on the exact count: "count < k" leaked outside DP | on the noisy count: post-processing, covered by Theorem 2 |
| budget accounting | basic composition Σε, Σδ | zCDP (Σρ, converted to (ε, δ) at the account's δ) next to pure basic composition; a query is admitted if either fits |
| trust | one trusted coordinator; no collusion at all | coordinator + any one hospital may collude; honest majority among hospitals |
| linked-cohort size | the coordinator, the dealer and the network saw the exact m_S | the coordinator and the network see only the public bound M (`linkage.pad_to`): the linked index is padded with empty rows. A hospital still learns the overlap sizes of every linkage it takes part in (L7) |
| PSI list sizes and order | every hospital saw the others' list sizes; lists entered the ring in record-number order | every list is padded to M with random dummy identifiers and shuffled |
| timing | a hospital answered as soon as it was done (time grew with its matches and its noise) | calls that touch patient data or draw noise answer at a public deadline |
| plan checks | the coordinator chose which cells the budget covered, and could tell each hospital a different plan | the budgeted groups are fixed by the aggregate, and every hospital-to-hospital message carries a digest of the one computation header |
| failed queries | refunded | charged once any hospital has shared its data |

---

## 1. Setting

**Parties.** Three hospitals 𝓗 = {A, B, C}, in the fixed public ring order of `configs/topology.yaml`; one
coordinator K; any number of requesters (researchers, clinicians). Requesters talk only to K's API.

**Data.** Hospital *i* holds a database D_i of patient records keyed by a national identifier. D = (D_A, D_B, D_C).
A *person* is a national identifier together with every record any hospital holds under it.

**Queries.** A query q fixes an aggregate (COUNT, SUM, AVG, HISTOGRAM), a condition code, a predicate per
hospital, and (SUM/AVG/HISTOGRAM) a value field or group-by field held by one hospital, the *owner* o(q).
S(q) ⊆ 𝓗 is the set of participating hospitals. The output has t(q) integer cells: 1 for COUNT, 2 for SUM/AVG
(cohort count and scaled total), one per bin for a HISTOGRAM (the bins come from the public vocabulary, never from
the data). With the linked cohort L_S(D) = ∩_{i∈S} ids(D_i) and m_S = |L_S(D)|, the exact answer is

    f_q(D)_j  =  Σ_{p ∈ L_S(D)}  Π_{i ∈ S, i ≠ o} φ_i(p) · O_j(p)        (φ_i ∈ {0,1}: hospital i's predicate; O_j: the owner's output vector)

**Roles** (`smpc/protocol.py: assign_roles`, recomputed by every hospital from the public topology). For |S| ≥ 2 the
computing parties P0, P1 are the first two members of S in ring order; the *dealer* is the third member of S, or,
when |S| = 2, the first hospital in ring order that is not in S. The *involved* set is I(q) = S ∪ {dealer}, and
c = |I(q)| = 3. For |S| = 1 there is no protocol: the lone hospital computes its own aggregate, and c = 1.

**Noise plan** (`privacy/dp/distributed.py`). The coordinator builds a public plan π from the requester's (ε, δ,
mechanism), the query's sensitivity (§4.1) and the collusion tolerance t (policy; default 1). With
h = c − min(t, c − 1) (h = 2 for c = 3, t = 1), every involved hospital i draws, with its own randomness, a share
Z_i ∈ ℤ^{t(q)}:

* Laplace: Z_{i,j} = X − X′ with X, X′ ~ NegBin(1/h, 1 − a_g) independent, a_g = exp(−ε_g/Δ_g), where ε_g is the
  part of ε given to the cell's group g and Δ_g its sensitivity (exact integer sampler: Pólya via geometric rejection);
* Gaussian: Z_{i,j} ~ N_ℤ(0, s_g²) with s_g² = σ_g²/h ≥ 1 (exact discrete-Gaussian sampler of [CKS20]).

The opened value is Y = f_q(D) + Σ_{i∈I(q)} Z_i (mod p = 2⁶¹ − 1, decoded to a signed integer).

**Public bound M** (`configs/topology.yaml: linkage.pad_to`, 4,608). Every hospital pads its PSI list to M with
random dummy identifiers (which match nothing) and shuffles it; the linked index then has length M, positions
m_S … M − 1 being empty rows (all-zero indicators, so they add exactly 0). A hospital with more than M patients refuses
to link.

**Protocol Π.** (1) PSI linkage over S by commutative encryption (`privacy/linkage/psi.py`) on the padded, shuffled
lists, cached per set S;
(2) each input hospital 2-out-of-2 additively secret-shares its vectors to P0 and P1, and every involved hospital
secret-shares its noise vector the same way; (3) the dealer draws fresh Beaver triples (x, y, z = xy), sends P0 a
32-byte seed from which P0 expands (x₀, y₀, z₀) with SHAKE-256, and sends P1 (x₁, y₁, z₁ = xy − z₀);
(4) P0 and P1 evaluate the product layers with Beaver multiplication [Bea91], opening the masked differences to each
other only; (5) each computer adds its shares of every noise vector to its output shares and sends them to K, which
adds them to obtain Y; (6) K applies the release stage `finalize` (k-threshold on the noisy count, clamping, AVG
division) and returns the result to the requester. Before step 1, K's ledger has reserved the query's charge.

## 2. Assumptions

| | assumption | enforced or modelled where |
|---|---|---|
| A1 | **Semi-honest, static corruption.** The adversary corrupts the coordinator and/or **at most one hospital**, chosen before the run; corrupted parties follow the protocol and try to learn from what they see. Requesters are arbitrary (malicious) but only use the API. | model |
| A2 | **Honest majority of hospitals.** At least two of the three hospitals are not corrupted. | model (Theorem 1 is false without it, §3.4) |
| A3 | **Authenticated private channels.** Every link is mutual TLS; peers check the named sender; control endpoints obey only the coordinator. | `privacy/encryption/tls.py`, `service_auth.py`; `tests/test_e2e_tls.py`, `test_secure_api.py` |
| A4 | **DDH** in the 2048-bit MODP group (RFC 3526 group 14) with 256-bit exponents, and hash-to-group modelled as a random oracle. | `privacy/linkage/group.py` |
| A5 | **SHAKE-256 is a pseudorandom generator** (triple-seed expansion, domain-separated per session). | `smpc/protocol.py: expand_seed` |
| A6 | **One person, one identifier.** A person has the same national identifier at every hospital and no person has two. | data model (Synthea SSN) |
| A7 | **Public, fixed topology and policy.** The ring order, the role assignment, the bound M and every hospital's noise policy and response deadlines (`configs/policy.yaml: distributed_noise`, `response_deadlines`) are public; hospitals recompute roles themselves and refuse a plan weaker than their policy: more than the per-query cap, noise whose guarantee needs more honest shares than the hospital's own collusion tolerance assumes, budgeted groups other than the fixed partition for the aggregate (so no output cell escapes the ε/ρ sum), or a Laplace scale above the cap. Every hospital-to-hospital message carries a digest of the computation header (everything in the request but the hospital's own query slice), and a mismatch is refused, so all involved hospitals ran one plan. | `hospital/routes_mpc.py: prepare, _same_header`, `NoisePlan.check`, `expected_groups`, `MpcPrepare.header_digest`; `tests/test_mpc_service.py`, `tests/test_distributed_dp.py` |
| A8 | **Timing and failures.** Hospitals answer every call that touches patient data or draws noise (PSI init, finalize and delete, MPC prepare, FL train) at a public deadline, and run it alone: no other request runs at that hospital until the deadline has passed. The work overruns its deadline with probability at most β per call (§3.2 L5), which assumes each hospital has a machine of its own that nothing outside its request handling loads; every other call does work fixed by M, t(q) and the model size. A computation that aborts after any hospital has shared is charged, not refunded. Modelled: infrastructure failures (a crash, a network error) are independent of the data. | `hospital/deadlines.py`, `coordinator/service.py`; `tests/test_deadlines.py`, `tests/test_private_service.py` |

## 3. Theorem 1 - secure computation with explicit leakage

### 3.1 Statement

Let F be the ideal functionality that, on hospitals' inputs D and a sequence of queries q₁, q₂, … with their public
plans π₁, π₂, …: samples every involved hospital's noise share Z_{k,i} from π_k; gives the coordinator
Y_k = f_{q_k}(D) + Σ_{i∈I(q_k)} Z_{k,i}; and gives every party P the leakage ℒ_P(D, q₁, q₂, …) listed in §3.2.

> **Theorem 1.** Under A1-A8, for every set C of corrupted parties with C ⊆ {K} ∪ 𝓗 and |C ∩ 𝓗| ≤ 1, there is a
> probabilistic polynomial-time simulator Sim_C such that for all D and all query sequences
>
>     View_C^Π(D; q₁, q₂, …)   ≈_c   Sim_C( ℒ_C(D; q₁, q₂, …),  D_C,  (Z_{k,i})_{i∈C},  (Y_k)_k if K ∈ C )
>
> where D_C are the corrupted hospitals' databases and ℒ_C is the union of the corrupted parties' leakage. The
> distinguishing advantage is bounded by a polynomial in the number of queries and patients times the DDH advantage
> in the PSI group plus the PRG advantage of SHAKE-256, plus β per deadline-guarded call if the view includes the
> response times (L5).

In words: whatever the coordinator and one hospital see together can be generated from the leakage list, that
hospital's own data and noise share, and the noisy outputs. They never see an exact aggregate, an identifier of a
patient outside their own lists, or any per-patient value of another hospital.

### 3.2 What leaks - the explicit list

Everything below is revealed **exactly** (it is *not* protected by differential privacy) unless marked "DP".

**To the coordinator K**

* **L1 - the query and the plan.** q, the requester's (ε, δ, mechanism) and the public noise plan π (it builds them).
* **L2 - the public bound M** as the size of every linked index, plus an 8-byte digest of each linked index whose
  only use is to check that all hospitals agree (a hash of tags under destroyed session keys, fresh at every
  re-link). **Not m_S**: the true cohort size stays in each hospital's mapping. Re-linking therefore no longer gives a
  time series of m_S either. A hospital refuses to link if its list is longer than M (an operator error that shows
  only that one list exceeds the public bound). A *hospital*, however, learns m_S for every S inside a linkage it
  took part in, from the tag sets (L7). The coordinator decides which linkages run, so a coalition {K, i} learns every
  m_S that hospital i can compute: after any three-hospital linkage, all of m_AB, m_AC, m_BC and m_ABC.
* **L3 - noisy outputs (DP).** Y_k for every query of every requester, *before* the k-threshold and post-processing.
* **L4 - refusals.** Which hospital refused a plan and why. Refusals depend only on the plan and the public policies.
* **L5 - timing and response sizes** of the rounds it conducts. Sizes depend only on M and t(q). The time of PSI
  init/finalize/delete, MPC prepare and FL train is a public deadline computed from the hospital's policy and the
  public plan (`hospital/deadlines.py`: for prepare, 0.25 s + 5 ms per cell + 100 µs per cell per unit of Laplace
  scale). Each of these calls runs alone at its hospital (`QuietGate`): it starts once no other request is in progress
  there, and no other request starts until its deadline has passed, so timing a concurrent request (a health probe,
  another query's step) also shows only public deadlines. Every other round does work fixed by M and t(q).
  **Residual:** when the work overruns its deadline the answer comes at the next doubling (2D, 4D, …), so an overrun
  shows that it happened and which doubling it fell in. For the noise draw, that probability β is about 10⁻¹¹ per
  call on the reference machine at the default allowance, by extrapolating the measured tail of the sampling time per
  unit of scale (median 3-4.5 µs, 99.99th percentile 36 µs at scale 1,000, about 4 µs more per factor e of
  probability). For the other guarded work (the predicate, loading the list, training) there is no tail estimate: the
  base deadlines sit far above the measured work (`evaluation/performance.py` reports the largest prepare and the
  overruns), but β there is assumed. β also assumes a machine per hospital that nothing outside the hospital's request
  handling loads: the quiet window holds back the coordinator's other requests, but not the decoding of request bodies
  as they arrive, so a flood of requests can still slow guarded work (a separate worker process on its own core, or
  rate limiting in front of the hospital, would stop that; neither is built). Slower hardware needs larger deadlines.
  To keep the deadline bounded, hospitals refuse a Laplace scale above 500,000 per cell
  (`distributed_noise.max_noise_scale`).

**To every participating hospital i ∈ S(q)**

* **L6 - its own slice of the query and the query's shape.** Its own predicate, the condition code, for the owner
  the value or group-by field; the participating set S, the owner, the roles, t(q), M, and the plan π (mechanism,
  ε, δ, ρ, c, h and the per-cell sensitivities - so the SUM bound B, which names the value field and, for a lab, the
  condition). m_S it already knows from the PSI (L7).
* **L7 - the PSI output, pairwise.** Every hospital receives every other hospital's full tag set, so it learns,
  **for each of its own patients, which of the other participating hospitals also hold them** (ids(D_i) ∩ ids(D_j)
  for each j - not only the full intersection), and the sizes of every region of the other hospitals' Venn diagram
  (e.g. A learns |ids(D_B) ∩ ids(D_C)| and how many of those are not A's patients). Lists enter the ring in a fresh
  random order, so the hospital that completes another's list learns nothing about that hospital's record order.
* **L8 - list sizes: only the public bound M.** Every list is padded to M with random dummy identifiers, which never
  match and are indistinguishable from real ones after hashing (A4).
* **L9 - its own noise share** Z_i (it drew it).

Nothing else: every MPC message a hospital receives is a uniformly random share, a masked opening, or a PRG-derived
triple share; hospitals never receive the output shares, so **a hospital does not learn Y**.

**To the dealer of a two-hospital query** (a hospital outside S): **L10** - that a query between the two others runs,
its aggregate type, t(q), the bound M and the plan π. The plan's per-cell sensitivities and the owner name the value
field and, for a lab, often the condition (owner B with sensitivity 200,000 is the NT-proBNP change, i.e. heart
failure). Not the predicates or anything about any patient. It does know m_S if it ever took part in a linkage
containing S, e.g. any three-hospital linkage (L7); the computation itself tells it only M.

**To a requester**

* **L11 - the released answer (DP)**: the post-processed noisy value(s), the suppressed flag(s), the status, the
  plan's public noise scale and expected error, its own budget balance, and the participating hospitals (a function
  of its own query).
* **L12 - side channels**: the response time (a function of the public plan, t(q), M and the deadlines, of whether
  *some* earlier query already linked the same hospital set, and of a deadline overrun with probability β per call,
  L5); the audit sequence number (how many audit events happened overall); and a "failed" status if a computation
  aborts (charged once data was shared; infrastructure failures assumed data-independent, A8).

**To a network observer** (outside TLS): **L13** - who talks to whom and when, and message sizes: hence M and t(q)
(no longer m_S or the list sizes).

**Coalitions allowed by A1.** The coalition {K, i} learns the union of K's and i's leakage and, by subtracting Z_i
from Y, the value f_q(D) + Σ_{j ∈ I(q), j ≠ i} Z_j, which still contains h = 2 honest noise shares - exactly the noise
of an ε-DP (or ρ-zCDP) release (§4.2). It learns nothing else: with i = P_b it lacks the other computer's shares and
the dealer's masks; with i = dealer it holds the masks but never sees an opening.

**What no longer leaks** (compared with the previous design): the exact aggregate f_q(D) (it was in the coordinator's
memory); the Beaver masks (the coordinator dealt them, so coordinator + one computer could unmask every opening and
read the other hospitals' indicator vectors); the exact-count k-threshold bit [f_q(D) < k]; m_S to the coordinator
and the network; list sizes; record order in the PSI ring; and, up to β, how long a hospital's predicate
and noise draw took.

### 3.3 Proof sketch

The simulator works query by query (every session uses fresh shares, fresh triples and, for linkage, fresh PSI keys
that are destroyed at `finalize`; sessions compose sequentially and, being semi-honest with independent randomness,
concurrently [Gol04]). Hybrids, from the real view to the simulated one:

1. **PSI.** A corrupted hospital i sees, in each ring round, another party's list encrypted under a set of keys that
   contains at least one honest key and not k_i, and finally its own list under all keys and every party's tag set.
   Under DDH with a random-oracle hash, a list of distinct hashed identifiers raised to an unknown key is
   indistinguishable from uniform group elements (the standard argument for DH-based PSI [HFH99, DT10], applied once
   per honest key). The simulator therefore outputs M random group elements per list (L8: real and dummy
   identifiers hash to the same distribution), and tag sets of size M with the right overlaps with i's own tags and
   with each other (L7). The coordinator sees no PSI traffic, only M and the digest (L2).
2. **Triples.** Replace P0's SHAKE-256 expansion by truly random (x₀, y₀, z₀) (PRG security, A5). Now P1's
   z₁ = (x₀ + x₁)(y₀ + y₁) − z₀ is uniform given P1's view, and P1's own x₁, y₁ are uniform by construction. The
   dealer's view is its own random tape.
3. **Openings.** Each opening is d = a − x (and e = b − y) for a triple used exactly once; with the dealer honest
   and x = x₀ + x₁, the computer's view of d is uniform and independent of a. If the corrupted hospital is the dealer,
   it sees no openings at all (they flow only between P0 and P1, A3).
4. **Input and noise shares.** Every share a corrupted computer receives is one half of a 2-out-of-2 additive
   sharing whose other half is held by the honest computer: uniform in 𝔽_p and independent of the shared value.
5. **Output.** The coordinator receives (o₀, o₁) with o₀ + o₁ = Y. If no computer is corrupted, the simulator
   draws o₀ uniformly and sets o₁ = Y − o₀; if P_b is corrupted, o_b is determined by P_b's (simulated) view and
   o_{1−b} = Y − o_b. Control-plane responses carry no data (status acknowledgements).
6. **Single-hospital queries.** The lone hospital sends Y itself; the simulator for K uses Y.
7. **Timing.** Replace every deadline-guarded response time by its deadline, a function of public inputs. The two
   views differ only on an overrun event, of probability at most β per guarded call; every other round's work is
   fixed by M and t(q).

Every hybrid step is perfect except steps 1 (DDH), 2 (PRG) and 7 (β per guarded call), which gives the stated
bound. ∎

### 3.4 Why the assumptions are needed (tightness)

* **Two colluding hospitals break input privacy.** P0 together with the dealer computes a = d + x for every opening;
  P0 and P1 together hold both shares of every input. This is the standard honest-majority bound for three-party
  secret-sharing MPC; with three hospitals it cannot be raised without a different protocol (e.g. OT- or HE-based
  triple generation between the two computers).
* **Semi-honest only.** A malicious hospital can submit a fabricated PSI list (to test whether a given person is a
  patient elsewhere) or a wrong share (to corrupt or bias a result); a malicious coordinator can replay queries without
  charging anyone (see Theorem 2b and §7). No MACs or zero-knowledge proofs are used.

## 4. Theorem 2 - end-to-end differential privacy

### 4.1 Neighbouring databases and sensitivity

D ≃ D′ if D′ is obtained from D by adding or removing **all records of one person** (one national identifier) at any
subset of the hospitals. By A6 a person contributes at most one element to every linked cohort L_S.

> **Lemma 1 (sensitivity).** For D ≃ D′: COUNT changes by at most 1; a HISTOGRAM changes at most one bin, by at most
> 1 (ℓ₁ = ℓ₂ = 1 over all bins); SUM changes the count cell by at most 1 and the total cell by at most
> B = max(|lo|, |hi|) of the value field's declared bounds.

*Proof.* The person contributes one term p to the sums in §1, which is 0/1 per output cell; histogram bins are
disjoint (one category per person); the owner clamps every value to [lo, hi] before it enters the computation
(`predicate_runner._scaled`) and checks that the plan's B is at least that bound (`routes_mpc.prepare`). ∎

### 4.2 One query

> **Lemma 2 (distributed noise).** Fix a query q with plan π, and any set H ⊆ I(q) of at least h contributors whose
> shares are unknown to an observer. Then f_q(D) + Σ_{i∈H} Z_i, as a function of D, is
> * **Laplace:** ε-DP and ρ = Σ_g ε_g²/2-zCDP (ρ = ε²/2 for COUNT/HISTOGRAM, ε²/4 for SUM/AVG);
> * **Gaussian:** ρ-zCDP with ρ = Σ_g [Δ_g² / (2 h s_g²) + τ_g d_g / 4],
>   τ_g = 10 Σ_{k=1}^{h−1} exp(−2π² s_g² k/(k+1)), d_g the number of cells in group g;
>
> and the same holds for the sum over **all** c shares (what the requester's answer is computed from).

*Proof.* Laplace: NegBin(r, 1−a) is infinitely divisible, so the sum of h independent NegBin(1/h, 1−a) is
NegBin(1, 1−a) = Geometric(1−a), and the difference of two is exactly the discrete Laplace with
P(x) ∝ a^{|x|} [Goryczka-Xiong]; with a = exp(−ε_g/Δ_g) this is ε_g-DP for sensitivity Δ_g (the geometric mechanism
[GRS12]); a group of disjoint cells with ℓ₁ sensitivity Δ_g costs ε_g once; the groups compose to Σ_g ε_g = ε
(the plan splits ε evenly), and ε-DP implies ε²/2-zCDP [BS16, Prop. 3.3] per group. Gaussian: the sum of h
independent N_ℤ(0, s²) has the stated zCDP bound by [KLS21, Thm. 4] (s² ≥ 1 keeps τ below 10⁻³; it is charged anyway);
groups compose additively in ρ [BS16]. More shares than h add independent noise: post-processing. ∎

Lemma 2 does not depend on which h shares are honest - this is what makes the guarantee hold against every
coalition that knows at most c − h = t shares.

> **Lemma 3 (release is post-processing).** The released answer and the released/suppressed decision are a
> deterministic function g(Y) of the noisy cells: suppress a count (a histogram bin, the SUM/AVG cohort) when its
> **noisy** value is below k; clamp counts at 0; for AVG divide the noisy total by the noisy count and clamp to the
> field's bounds. Hence they inherit Lemma 2 [DR14, Prop. 2.1].

This is the point where the old **k-threshold leak** closes: the decision used to be [f_q(D) < k], an exact bit
about the data outside any DP guarantee; now it is [Y < k].

### 4.3 The ledger: many adaptive queries

Each requester r has an account with budget (ε_r, δ_r) (`configs/policy.yaml`; researcher (10, 10⁻⁶), clinician
(20, 10⁻⁴)). Its zCDP cap is ρ_r = max{ρ : zcdp_to_dp(ρ, δ_r) ≤ ε_r} (Canonne-Kamath-Steinke conversion
[CKS20, Cor. 13], evaluated at a concrete order α, so any numerical error only makes the cap smaller; stored rounded
down) - e.g. ρ = 1.539 for the researcher budget. Every query k carries (ε_k, δ_k, ρ_k) from its plan. The ledger
admits query k iff, counting it,

    (zCDP)  Σ ρ ≤ ρ_r          or          (basic)  every admitted query so far is pure (δ = 0)  and  Σ ε ≤ ε_r.

Charges are reserved atomically before any hospital is contacted (one conditional SQL UPDATE), committed when an
answer - released or suppressed - exists or once any hospital has shared its data (even if the computation then
aborts), and refunded only if it failed before that.

> **Lemma 4 (dual filter).** For every adaptive strategy of r, the whole transcript of r's interaction (all answers,
> statuses and refusals) is (ε_r, δ_r)-DP; if δ_r = 0 it is (ε_r, 0)-DP.

*Proof.* Consider the two single-account filters: F_z admits iff the zCDP condition holds, F_b iff the basic
condition holds. F_z is a valid zCDP filter: the condition Σρ_k ≤ ρ_r bounds Σ α ρ_k ≤ α ρ_r for every Rényi order α
at once, so by the Rényi filter theorem [FZ21; WRRW23] the adaptive composition is ρ_r-zCDP, hence (ε_r, δ_r)-DP by
the choice of ρ_r. F_b is a valid filter for pure DP: each answer's privacy loss is bounded by ε_k pointwise, so the
loss of any transcript it admits is at most Σ ε_k ≤ ε_r [RRUV16]. The ledger F admits iff F_z or F_b admits. Split
the transcripts into T_z (the admitted ρ's sum to at most ρ_r) and T_b (the rest). On T_z every admission of F is
also one of F_z (the sums only grow) and every refusal of F is a refusal of F_z, so the two interactions assign every
transcript in T_z the same probability; likewise on T_b with F_b (the last admission there was by the basic rule,
so all earlier ones satisfy it too). The parameters ε_k, δ_k, ρ_k are part of the transcript, so T_z and T_b are
events of the transcript. Hence for every set S of transcripts

    P_F[S | D] = P_{F_z}[S ∩ T_z | D] + P_{F_b}[S ∩ T_b | D] ≤ e^{ε_r} P_F[S | D′] + δ_r + 0. ∎

Gaussian queries (δ_k > 0) are therefore accounted in zCDP only; once one is admitted, the basic rule no longer
applies to that account (`tests/test_dp.py::test_after_a_delta_query_basic_composition_no_longer_admits`).

### 4.4 Statement

> **Theorem 2.** Under A1-A8:
>
> **(a) Requester.** For every requester r with budget (ε_r, δ_r) and every adaptive strategy, the transcript of r's
> interaction with the system is (ε_r, δ_r)-differentially private with respect to ≃ - information-theoretically, with
> the k-threshold, the suppression flags, AVG division and every other post-processing included. (Pure Laplace
> accounts with δ_r = 0 get (ε_r, 0)-DP.) If the transcript includes response times, add (1 + e^{ε_r})·N_r·β to δ,
> where N_r is the number of deadline-guarded hospital calls r's queries caused.
>
> **(b) Coordinator, alone or with one hospital.** Let C = {K} or {K, i} with t ≥ 1 (the default). The coalition's
> view of all queries executed for all requesters is computationally indistinguishable (Theorem 1) from a randomised
> function of (ℒ_C, D_C, the coalition's noise shares, and the values f_q(D) + Σ_{j ∉ C} Z_{q,j} for every query q).
> The last component is (Σ_r ε_r, Σ_r δ_r)-DP with respect to the records held by the honest hospitals. Hence, for
> neighbouring D ≃ D′ that agree on the coalition's own data and induce the same leakage ℒ_C, the coalition's view is
> (Σ_r ε_r, Σ_r δ_r + negl)-computationally differentially private (SIM-CDP [MPRV09]); with response times in the view,
> add (1 + e^{Σ ε_r})·N·β for N guarded calls.
>
> **(c) A hospital alone** learns only ℒ_i (L6-L10): it never receives a noisy or exact output.

*Proof.* (a) Each answer is g(Y) with Y the sum of all c shares (Lemmas 1-3); the transcript also contains refusals,
which are a function of r's own parameters only, and failures, which are data-independent by A8. Lemma 4 composes the
answers. For the timing term: let V be the view with real response times and V′ the same view with every guarded
response time replaced by its deadline. V′ is a function of the transcript and public inputs, so it is
(ε, δ)-DP, and V = V′ except on an overrun, so their statistical distance is at most Nβ for every database; then
P[V ∈ T | D] ≤ P[V′ ∈ T | D] + Nβ ≤ e^ε P[V′ ∈ T | D′] + δ + Nβ ≤ e^ε P[V ∈ T | D′] + δ + (1 + e^ε) Nβ. (b) By Theorem 1 the view is simulated from the listed inputs. Every query's coalition-visible output
contains at least h honest shares, so by Lemma 2 it has the same guarantee as the requester's answer; the coalition
sees every requester's queries, each requester's queries are a filter-bounded interactive mechanism (Lemma 4 applies
to the coalition's view too, since it is the same sequence of (ε_k, δ_k, ρ_k)), and interleaved interactive mechanisms
compose concurrently with the same bounds as basic composition [VW21; VZ23]. The coalition's own records and the
leakage ℒ_C are excluded from the guarantee by construction. ∎

### 4.5 Reading the theorem

* **The k-threshold is now inside the guarantee** (Lemma 3). The price is a fuzzy boundary: a cohort just above k is
  sometimes suppressed and one just below is sometimes released (`evaluation/RESULTS.md`, suppression table).
* **Distrust costs variance, not privacy.** With t = 1 every involved hospital adds a share that is half of a full
  noise draw; the requester sees c/h = 1.5 × the variance a trusted coordinator would have added, and the coordinator
  plus one hospital still face a full ε-DP draw (`charts/collusion_tolerance_cost.png`).
* **What is *not* DP-protected** is exactly the leakage list: most importantly the pairwise PSI membership (L7) and
  m_S seen by the hospitals of a linked set - a hospital learns which of its own patients the others treat. The
  coordinator no longer sees m_S (L2), so for the coordinator alone the leakage no longer depends on who is linked;
  for a coalition {K, i} where i took part in a linkage containing S it still does (any three-hospital linkage
  qualifies), and Theorem 2(b) applies to neighbours that leave
  that leakage unchanged.
* **The coordinator's budget is the sum over all requesters.** No component enforces a federation-wide cap, so the
  guarantee of Theorem 2(b) weakens with every new account. Hospitals cap each *query* (ε ≤ 2 by default) but do
  not count queries.

### 4.6 Numbers

| quantity | default | where |
|---|---|---|
| collusion tolerance t, honest shares h, contributors c | 1, 2, 3 (multi-hospital); 0, 1, 1 (single hospital) | `policy.yaml: distributed_noise.tolerance` |
| hospital per-query cap | Laplace: pure ε ≤ 2 (summed over a query's groups); Gaussian: ρ ≤ 0.139, i.e. (2, 10⁻⁴)-DP | `policy.yaml: distributed_noise`, `NoisePlan.check` |
| researcher budget | ε 10, δ 10⁻⁶ → ρ cap 1.539 | `policy.yaml: roles` |
| clinician budget | ε 20, δ 10⁻⁴ → ρ cap 6.18 | `policy.yaml: roles` |
| COUNT, Laplace, ε per query | ρ = ε²/2; admitted: 307 queries at ε 0.1 (zCDP) vs 100 (basic); 10 at ε 1 (basic) vs 3 (zCDP) | `accounting.py`, RESULTS.md |
| Gaussian per-share variance floor | s² ≥ 1 (τ ≈ 5.2·10⁻⁴ at h = 2) | `distributed.py: MIN_SHARE_SIGMA2` |
| field | p = 2⁶¹ − 1 (a wrap-around would only corrupt utility: Y mod p is still a function of Y) | `smpc/field.py` |
| linked-index and list padding M | 4,608 (tests 1,024); at most this many patients per hospital | `topology.yaml: linkage.pad_to`, `FEDHEALTH_LINK_PAD_TO` |
| response deadlines | PSI init 1.5 s, finalize and delete 0.25 s; prepare 0.25 s + 5 ms/cell + 100 µs/cell per unit of Laplace scale (at most 60 s); FL 0.5 s/epoch; each runs alone at its hospital | `policy.yaml: response_deadlines` |
| Laplace scale cap per cell | 500,000 (e.g. a heart-failure lab SUM needs ε ≥ 0.8, or the Gaussian mechanism, whose sampler's time does not grow with σ) | `policy.yaml: distributed_noise.max_noise_scale` |

## 5. Where each step is enforced and tested

| claim | code | test |
|---|---|---|
| the coordinator only ever opens noisy cells; hospitals refuse a computation without a noise plan | `hospital/routes_mpc.py: prepare` (refuses `noise=None` unless the insecure evaluation baseline is on) | `tests/test_mpc_service.py` |
| hospitals recompute the roles and refuse a weaker plan (above the cap, or sized for fewer colluders than they tolerate), a wrong contributor count or an understated SUM bound | `routes_mpc.py: prepare`, `NoisePlan.check` | `tests/test_mpc_service.py::test_hospitals_refuse_weak_or_missing_noise`, `::test_hospitals_refuse_noise_that_needs_every_share_honest`, `::test_owner_rejects_an_understated_sum_sensitivity`, `tests/test_distributed_dp.py::test_hospital_side_check_refuses_weak_plans` |
| only the dealer may deliver triples; P0 gets a seed, P1 explicit shares; openings go P0 ↔ P1 only | `routes_mpc.py: deal, intriples, open` | `tests/test_smpc.py`, `tests/test_mpc_service.py` |
| no identifier and no small (un-shared) value in any MPC message, on the private path | recorder + scan | `tests/test_mpc_service.py`, `tests/test_secure_api.py::test_the_traffic_of_a_private_query_is_clean` |
| h Pólya shares sum to exactly the discrete Laplace; samplers match their laws | `mechanisms.py: sample_polya, sample_discrete_gaussian` | `tests/test_distributed_dp.py` (χ² tests) |
| tolerating t colluders multiplies the variance by c/h and leaves ρ unchanged | `distributed.py: make_plan` | `test_tolerating_more_colluders_costs_variance_not_privacy` |
| zCDP conversion and its inverse | `accounting.py` | `test_zcdp_conversion_is_tighter_than_the_textbook_bound_and_invertible` |
| dual-filter ledger, atomic reservation, Gaussian queries zCDP-only | `privacy/dp/ledger.py: reserve` | `tests/test_dp.py` (incl. a 40-thread race) |
| suppression uses the noisy count | `coordinator/release.py: finalize` | `tests/test_dp.py`, `tests/test_private_service.py::test_suppression_is_decided_on_the_noisy_count` |
| attacks: differencing (requester and coordinator + 1 hospital), averaging vs the ledger | `attacks/` | `tests/test_attacks.py`, `evaluation/RESULTS.md` |
| the budgeted groups are fixed by the aggregate, so no cell escapes the cap | `distributed.py: expected_groups, NoisePlan.check` | `tests/test_distributed_dp.py::test_hospitals_refuse_a_plan_that_leaves_a_cell_outside_the_budget` |
| all involved hospitals ran one computation header; the owner's aggregate matches its slice | `wire.py: MpcPrepare.header_digest`, `routes_mpc.py: _same_header, prepare` | `tests/test_mpc_service.py::test_hospitals_detect_a_coordinator_that_tells_them_different_plans`, `::test_owner_rejects_an_aggregate_that_does_not_match_its_slice` |
| lists padded to M and shuffled; the coordinator sees M, never m_S | `routes_psi.py: init, inbox, finalize` | `tests/test_psi.py::test_lists_are_padded_to_the_public_bound`, `::test_linkage_cohort_sizes_are_exact_but_stay_at_the_hospitals`, `tests/test_mpc_service.py::test_psi_lists_enter_the_ring_in_a_fresh_random_order` |
| data-touching calls answer at a public deadline and run alone at their hospital; too large a Laplace scale is refused | `hospital/deadlines.py` (`guarded`, `QuietGate`), `NoisePlan.check` | `tests/test_deadlines.py` (incl. `::test_nothing_else_runs_at_a_hospital_during_a_guarded_window`) |
| a failure after the hospitals shared is charged | `orchestrator.py: ComputationAborted`, `service.py: _charged_failure` | `tests/test_private_service.py::test_a_failure_after_the_hospitals_shared_their_data_is_charged` |
| FL noise comes from hospital-private randomness; one release per round; every round includes the whole federation; no plain updates in strict mode | `routes_fl.py` | `tests/test_federated.py::test_the_coordinator_cannot_regenerate_the_dp_noise_or_take_both_releases`, `::test_fl_endpoints_obey_the_service_identity_rules` |

## 6. References

* [Bea91] D. Beaver. Efficient multiparty protocols using circuit randomization. CRYPTO 1991.
* [BS16] M. Bun, T. Steinke. Concentrated differential privacy: simplifications, extensions, and lower bounds. TCC 2016.
* [CKS20] C. Canonne, G. Kamath, T. Steinke. The discrete Gaussian for differential privacy. NeurIPS 2020.
* [DR14] C. Dwork, A. Roth. The Algorithmic Foundations of Differential Privacy. 2014.
* [DT10] E. De Cristofaro, G. Tsudik. Practical private set intersection protocols with linear complexity. FC 2010.
* [FZ21] V. Feldman, T. Zrnic. Individual privacy accounting via a Rényi filter. NeurIPS 2021.
* [Gol04] O. Goldreich. Foundations of Cryptography, Vol. 2, ch. 7 (semi-honest simulation and composition). 2004.
* [Goryczka-Xiong] S. Goryczka, L. Xiong. A comprehensive comparison of multiparty secure additions with differential privacy. IEEE TDSC 2017.
* [GRS12] A. Ghosh, T. Roughgarden, M. Sundararajan. Universally utility-maximizing privacy mechanisms. SICOMP 2012.
* [HFH99] B. Huberman, M. Franklin, T. Hogg. Enhancing privacy and trust in electronic communities. ACM EC 1999.
* [KLS21] P. Kairouz, Z. Liu, T. Steinke. The distributed discrete Gaussian mechanism for federated learning with secure aggregation. ICML 2021.
* [MPRV09] I. Mironov, O. Pandey, O. Reingold, S. Vadhan. Computational differential privacy. CRYPTO 2009.
* [RRUV16] R. Rogers, A. Roth, J. Ullman, S. Vadhan. Privacy odometers and filters: pay-as-you-go composition. NeurIPS 2016.
* [VW21] S. Vadhan, T. Wang. Concurrent composition of differential privacy. TCC 2021.
* [VZ23] S. Vadhan, W. Zhang. Concurrent composition theorems for differential privacy. STOC 2023.
* [WRRW23] J. Whitehouse, A. Ramdas, R. Rogers, Z. S. Wu. Fully-adaptive composition in differential privacy. ICML 2023.

## 7. Not covered (see THREAT_MODEL.md §4 for the full list)

* Malicious behaviour of any party (A1), and any two hospitals colluding (A2).
* The exact leakage L1-L13, in particular pairwise PSI membership and m_S inside the linked set (L6, L7). A shuffled
  PSI with oblivious placement would close the per-patient part (LEAK_FIXES.md §2.3); it is not implemented.
* A federation-wide budget: hospitals cap each query but nobody caps the total the coordinator sees (Theorem 2b
  grows with the number of accounts). A hospital-side zCDP odometer would close this; it is not implemented.
* Deadline overruns beyond the β estimate: a slower or shared machine, a request flood (the quiet window holds other
  requests back but still decodes their bodies), and the guarded work other than the noise draw, whose β is assumed
  rather than measured. Also infrastructure failures that correlate with the data (A8), and Sybil requesters (several
  accounts are several budgets).
* Federated learning still accepts zero noise and unlimited rounds; FL rounds are not charged to any budget.
