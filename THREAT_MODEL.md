# Threat model

This is a **research prototype on synthetic data**. This document says what it is designed to withstand, what it
assumes, and — just as important — what it does **not** defend against. Where a claim is backed by a test, the test
is named.

## 1. Assets

| asset | where | sensitivity |
|---|---|---|
| per-patient records (history, reports, prescriptions) | each hospital's own SQLite file | highest |
| the linkage key (`national_id`) and demographics | encrypted columns in every hospital DB | highest |
| per-patient predicate results ("patient *k* took drug Y") | in memory at a hospital; secret-shared on the wire | high |
| exact aggregate counts / sums | reconstructed in the coordinator's memory | medium — releasing them exactly enables differencing |
| a requester's privacy budget | coordinator DB (ledger) | integrity-critical |
| the audit trail | coordinator DB | integrity-critical |
| keys: hospital data keys, JWT secret, audit key, TLS keys | `./secrets` | critical |

## 2. Actors and trust assumptions

| actor | assumed to be | consequence |
|---|---|---|
| **Hospital A / B / C** | **honest-but-curious** (semi-honest): follows the protocol, tries to learn from what it sees | no defence against a hospital that *deviates* (sends wrong shares, probes with crafted PSI lists) |
| **Coordinator** | honest-but-curious, **and it is trusted with more than the hospitals**: it sees the *exact* aggregate before noise, acts as the Beaver-triple dealer, holds the JWT and audit keys | a compromised coordinator defeats DP (it sees exact answers), the audit log and authentication |
| **Requester** (researcher / clinician) | **potentially malicious**: crafts differencing queries, repeats queries to average noise, tries role escalation | defended by schema validation, DP, the ε-ledger, k-threshold, RBAC |
| **Auditor** | honest; read-only; cannot query | — |
| **Network attacker** | can observe and modify traffic between services and clients | defended by (mutual) TLS |
| **Offline attacker** (stolen disk / backup) | can read database files, not the running service | defended by AES-GCM at rest |
| **Collusion** | **assumed not to happen** between any two of {hospital, hospital, coordinator, requester} | see §4 for what each pair could do |

## 3. What the design defends against

| threat | defence | evidence |
|---|---|---|
| A hospital reads another's patient list during linkage | PSI by commutative encryption; only the intersection is revealed | `tests/test_psi.py` (incl. the HMAC dictionary-attack comparison) |
| A hospital or the coordinator reads per-patient predicate results | additive secret sharing + Beaver multiplication; coordinator receives only one random-looking share per cell | `tests/test_mpc_service.py` (wire scan: no identifier and no value < 2³⁰ in any MPC message; mutation test proves the scan has teeth) |
| A hospital learns another hospital's *query predicate* | each hospital is sent only its own slice (+ the shared condition code) | `test_each_hospital_receives_only_its_own_predicate` |
| A requester extracts rows | the query language has no row-level construct; `extra="forbid"`; hospitals re-validate and expose no row endpoint | `tests/test_schemas.py`, `tests/test_hospital_api.py`, `tests/test_secure_api.py` |
| Differencing / membership inference on released numbers | discrete-Laplace / Gaussian noise calibrated to sensitivity | `tests/test_dp.py`; attack demos in Phase 6 |
| Averaging out the noise by repeating a query | per-user ε-ledger refuses queries once the budget is spent | `tests/test_dp.py` (40-thread race), `tests/test_secure_api.py` (14 parallel requests, budget 10) |
| Tiny cells identify individuals | k-threshold (k = 10) | `tests/test_dp.py`, `tests/test_private_service.py` |
| Unauthenticated / escalated access | JWT (pinned alg, required claims, expiry), RBAC from `policy.yaml`, login throttle | `tests/test_security.py`, `tests/test_secure_api.py` |
| Theft of database files | AES-256-GCM on sensitive columns, key outside the data directory | `tests/test_encryption.py` (raw DB bytes scanned; plaintext control) |
| Eavesdropping / tampering between services | TLS 1.2+, hospitals require client certificates from our CA | `tests/test_e2e_tls.py` |
| A peer issuing control commands or impersonating another service | signed 60-second service tokens; control endpoints coordinator-only; peer endpoints check the named `sender` | `tests/test_secure_api.py::test_hospitals_only_obey_the_coordinator_and_the_named_sender` |
| Editing / deleting / reordering / forging audit entries | HMAC hash chain + append-only triggers + `verify_chain` | `tests/test_security.py` (every tamper case) |
| Budget lost on failure, or spent twice | reserve → commit / refund lifecycle; integer micro-ε | `tests/test_dp.py`, `tests/test_private_service.py` |

## 3b. Attacks that were actually run (details and numbers: `evaluation/RESULTS.md`)

| attack | result | what it teaches |
|---|---|---|
| **Differencing** — two large queries differing by one patient | **100%** attacker accuracy with exact answers *and with the k-threshold alone*; with DP: 52.5% at ε = 0.1, 61.5% at ε = 1, 98.6% at ε = 5 — matching the exact theory and below the DP ceiling | the k-threshold stops only the *direct* small-cohort question; DP is what stops differencing; **a large ε per query is no protection** |
| **Re-identification** of a "de-identified" release | 39% of a names-removed Hospital-A table re-identified on birth year + sex + ZIP (87% precision); 0.1% after generalising age and ZIP | removing names is not anonymisation — which is why the system exports no rows at all |
| **Budget exhaustion** — repeat a query to average out the noise | error ∝ 1/√N without a limit; the ledger refuses at query 21 (budget 10, ε = 0.5) | per-requester composition accounting is essential — and does not stop colluding accounts |
| **HMAC dictionary** (bonus) | recovers all 3,150 of B's patients (incl. 500 never seen by A); 0 of 103 against PSI | the case for PSI over shared-key pseudonyms |

## 4. Known limitations (what this prototype does NOT protect)

**Cryptographic protocols**
* **Semi-honest only.** No zero-knowledge proofs or MACs: a hospital can feed PSI a fabricated list to test whether a specific patient is shared, or send a wrong output share and silently corrupt (or bias) a result.
* **Dealer trust.** The coordinator deals Beaver triples. A coordinator colluding with *one* hospital can reconstruct another hospital's indicator vectors (it knows the triple mask `x`; the hospital sees the opening `d = a − x`). Generating triples without a dealer (OT / HE) is the standard remedy and is not implemented.
* **Two colluding hospitals** know their own vectors and the output, hence learn about the third's vector through the answer — inherent to the function being computed, mitigated only by DP.
* **PSI output and sizes.** Members of an intersection learn that a patient is shared; every hospital learns the other hospitals' *list sizes*; the coordinator learns the linked-cohort size `m` and the query *shape* (participants, number of output cells). Padding is not implemented.
* **Short exponents** (256 bit) over a 2048-bit group — standard for MODP groups, but not "full-size exponent" DDH. The test group `test768` is deliberately insecure and used only by tests.

**Differential privacy**
* **Central DP with a trusted coordinator:** the coordinator sees the exact aggregate before noise is added. Distributed noise generation inside the MPC would remove this; documented as a future upgrade.
* **Basic composition only** (Σε) — conservative, not tight. No result caching, so the same question asked in another phrasing spends budget again (that is the intended defence, but it also means honest users burn budget on repeats).
* **k-threshold side channel:** the suppress/release decision uses the exact count and reveals "count < k" outside the DP guarantee; `k_basis="noisy"` removes this at the cost of a fuzzy boundary.
* **Gaussian mechanism** samples a floating-point normal and rounds; the discrete-Laplace sampler is exact, the Gaussian one is not.
* **The unit of privacy is a patient who appears at the participating hospitals.** DP protects the *released answers*; it says nothing about the PSI intersection that hospitals learn.
* Patients who are rare on *plaintext* quasi-identifiers (birth year, sex) are protected only through the aggregate interface — the interface never exports rows, but if the databases leak the plaintext columns are readable.

**Federated learning (stretch)**
* The noise (clipped updates + Gaussian) gives **client-level** DP: it hides a *whole hospital's* contribution, not individual patients. With three clients the ε needed for real protection costs most of the accuracy (AUC 0.727 → 0.53 at ε ≈ 5). Record-level DP-SGD is not implemented.
* Plain FedAvg shows the coordinator each hospital's update; secure aggregation hides it, but updates are not verified (a malicious hospital can poison the model) and trained weights can memorise training rows (membership inference / inversion are not evaluated).
* Training is an operator job. Handing weights to a *requester* would need its own DP budget and audit trail and is deliberately not exposed.
* The training registries are a separate, non-IID synthetic sample (see `synth/registry.py`), not the vertical A/B/C slices.

**System security**
* **Encryption at rest protects a stolen file, not a running service**: the key is in process memory; administrators of a hospital can read its data (they own it). AAD binds ciphertext to its *column*, not its *row*. No key rotation, no KMS/HSM.
* **PKI is a toy:** keys are unencrypted PEM files next to the CA key; no revocation, no rotation, certificates valid for a year.
* **Tokens:** JWTs cannot be revoked before expiry; no refresh flow. Service tokens have no replay cache (they are visible only inside mTLS and live 60 s).
* **Audit log:** truncating the *newest* entries is invisible without an externally stored head anchor; whoever holds the audit key can forge a whole chain; hospitals keep no audit of their own.
* **The dashboard's "ground truth (demo mode only)" toggle** reads the synthetic CSVs inside the dashboard process (`FEDHEALTH_DEMO_MODE=1`). It defeats the whole point of the system for whoever can run the dashboard that way; it is off by default, absent from the Compose file, and exists only to let you inspect the accuracy/privacy trade-off.
* **The Docker deployment was never executed** (Docker is not installed on the build machine): `docker-compose.yml` is validated statically — least-privilege mounts, no published hospital ports, no insecure routes (`tests/test_deployment.py`) — while the same topology is exercised by `scripts/run_local.py --secure` and `tests/test_e2e_tls.py`.
* **Availability / DoS** is out of scope. SQLite and in-memory protocol sessions are single-node; the login throttle is in-memory.
* **Side channels** (timing of the MPC rounds, message sizes, memory) are not considered beyond the login-timing equalisation.
* **Synthetic data only.** Realism of correlations, rare diseases and linkage identifiers is limited; no claim is made about real-world re-identification risk.

## 5. Questions a reviewer should ask (and the honest answers)

* *"Does the coordinator see patient data?"* — No row, ID or per-patient value; it sees the cohort size and, transiently, the exact aggregate.
* *"Can the coordinator forge answers?"* — Yes. There is no verifiability of the computation.
* *"Is this production-ready?"* — No. It demonstrates the protocols, their composition and their tests; see the limitations above before reusing any of it.
