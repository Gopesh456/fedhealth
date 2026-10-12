"""EVALUATION 2 - Performance overhead: what does each privacy layer cost in latency?

STAGES (every number is a wall-clock HTTP round trip to the coordinator, against REAL uvicorn processes)
    plain          raw national IDs joined in the clear                               (the insecure baseline)
    + PSI (cold)   private linkage on the full patient lists + insecure aggregation   (first query: runs the PSI protocol)
    + PSI (warm)   same, linkage already cached
    + SMPC (cold)  PSI + secret-shared aggregation, exact result, no DP/auth
    + SMPC (warm)  secret-shared aggregation on a cached linkage                       (breakdown: prepare/share/triples/multiply)
    FULL (cold)    strict deployment: mutual TLS + service tokens + JWT login + RBAC + ledger + k + DP + audit, first query
    FULL (warm)    the same, linkage cached
    and the cost of pure-Python modular exponentiation (no gmpy2) for the cold linkage.

The cold/warm split matters: linkage depends only on WHO is in each hospital, not on the query, so it is computed once
and reused by every later query over the same hospitals.
"""
from __future__ import annotations

import contextlib
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attacks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, common_args, ensure_world, save_json  # noqa: E402
import run_local  # noqa: E402

from fedhealth.common.config import load_topology  # noqa: E402
from fedhealth.common.http import client_ssl_context  # noqa: E402
from fedhealth.common.schemas import QuerySpec  # noqa: E402
from fedhealth.coordinator.bootstrap import bootstrap_secrets  # noqa: E402
from fedhealth.privacy.encryption import tls  # noqa: E402
from fedhealth.privacy.encryption.aesgcm import secrets_dir  # noqa: E402
from fedhealth.privacy.encryption.tls import COORDINATOR, TlsPaths  # noqa: E402
from fedhealth.privacy.linkage.group import HAVE_GMPY2  # noqa: E402

SIZES = (500, 1000, 2500, 5000)
# A wide 3-hospital cohort so every size releases a number (the work done does not depend on the predicate, only on
# who is in each hospital and the cohort size m).
SPEC = QuerySpec.model_validate({"agg": "count", "condition": "I10", "a": {}, "b": {}, "c": {}})
BODY = SPEC.model_dump(mode="json", exclude_none=True)


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def summarize(xs: list[float]) -> dict:
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], statistics.median(xs), xs[-1]]
    return {"median": statistics.median(xs), "p25": q[0], "p75": q[2], "min": xs[0], "max": xs[-1], "n": len(xs)}


def timed_post(client: httpx.Client, path: str, **kw) -> tuple[float, dict]:
    t0 = time.perf_counter()
    r = client.post(path, **kw)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return dt, r.json()


def measure_dev(world: Path, offset: int, reps: int, cold_reps: int, extra_env: dict | None = None, only_cold_psi: bool = False) -> dict:
    """Plain / PSI / SMPC stages against dev-mode services (insecure routes mounted)."""
    out: dict = {}
    with env(FEDHEALTH_DATA_DIR=world, FEDHEALTH_PORT_OFFSET=offset, **(extra_env or {})):
        procs = run_local.start_services(quiet=True, timeout=180)
        try:
            topo = load_topology()
            with httpx.Client(base_url=f"http://127.0.0.1:{topo.coordinator_port}", timeout=900) as c:
                if only_cold_psi:
                    out["psi_cold_linkage_s"] = [timed_post(c, "/insecure/psi-query?force_link=true", json=BODY)[1]["timings"]["linkage_s"]
                                                 for _ in range(cold_reps)]
                    return out
                out["plain"] = [timed_post(c, "/insecure/query", json=BODY)[0] for _ in range(reps)]
                cold = [timed_post(c, "/insecure/psi-query?force_link=true", json=BODY) for _ in range(cold_reps)]
                out["psi_cold"] = [t for t, _ in cold]
                out["psi_cold_linkage_s"] = [j["timings"]["linkage_s"] for _, j in cold]
                out["psi_warm"] = [timed_post(c, "/insecure/psi-query", json=BODY)[0] for _ in range(reps)]
                coldm = [timed_post(c, "/insecure/mpc-query?force_link=true", json=BODY) for _ in range(cold_reps)]
                out["smpc_cold"] = [t for t, _ in coldm]
                warm = [timed_post(c, "/insecure/mpc-query", json=BODY) for _ in range(reps)]
                out["smpc_warm"] = [t for t, _ in warm]
                out["smpc_warm_breakdown"] = {k: statistics.median([j["timings"][k] for _, j in warm])
                                              for k in ("prepare_s", "share_s", "triples_s", "multiply_s")}
        finally:
            run_local.stop_services(procs)
    return out


def measure_strict(world: Path, offset: int, reps: int, tls_dir: Path) -> dict:
    """The full pipeline through the authenticated API over mutual TLS (strict deployment)."""
    for f in world.glob("coordinator.db*"):
        f.unlink()                                                  # fresh ledger: the cold query is the first one
    creds = (secrets_dir() / "DEMO_CREDENTIALS.txt").read_text()
    password = re.search(r"^clinician1\s+role=\w+\s+password=(\S+)", creds, re.M).group(1)
    out: dict = {}
    with env(FEDHEALTH_DATA_DIR=world, FEDHEALTH_PORT_OFFSET=offset, FEDHEALTH_TLS_DIR=tls_dir):
        procs = run_local.start_services(secure=True, quiet=True, timeout=180)
        try:
            topo = load_topology()
            ctx = client_ssl_context(TlsPaths(tls_dir, COORDINATOR), present_client_cert=False)
            with httpx.Client(base_url=f"https://127.0.0.1:{topo.coordinator_port}", verify=ctx, timeout=900) as c:
                t0 = time.perf_counter()
                tok = c.post("/auth/login", json={"username": "clinician1", "password": password}).json()["access_token"]
                out["login_s"] = time.perf_counter() - t0
                h = {"Authorization": f"Bearer {tok}"}
                body = {"query": BODY, "privacy": {"epsilon": 0.5}}
                t, j = timed_post(c, "/v1/query", json=body, headers=h)
                assert j["status"] == "released", j
                out["full_cold"] = [t]
                out["full_warm"] = [timed_post(c, "/v1/query", json=body, headers=h)[0] for _ in range(reps)]
        finally:
            run_local.stop_services(procs)
    return out


def machine() -> dict:
    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "gmpy2": HAVE_GMPY2}


def run(sizes=SIZES, reps: int = 15, cold_reps: int = 3, seed: int = 42, pure_python_n: int = 1000) -> dict:
    bootstrap_secrets()
    tls_tmp = Path(tempfile.mkdtemp(prefix="fedhealth-eval-tls-"))
    tls.generate_all(tls_tmp)
    result: dict = {"evaluation": "performance", "machine": machine(), "reps": reps, "cold_reps": cold_reps, "sizes": {}}
    for n in sizes:
        world = ensure_world(n, seed)
        print(f"[N={n}] dev-mode stages ...", flush=True)
        dev = measure_dev(world, 700, reps, cold_reps)
        print(f"[N={n}] strict mode (mTLS + JWT + DP + audit) ...", flush=True)
        strict = measure_strict(world, 750, reps, tls_tmp)
        entry = {k: summarize(v) for k, v in {**dev, **strict}.items() if isinstance(v, list)}
        entry["smpc_warm_breakdown"] = dev["smpc_warm_breakdown"]
        entry["login_s"] = strict["login_s"]
        # patients per hospital and linked cohort for context
        entry["patients"] = n
        result["sizes"][str(n)] = entry
    print(f"[N={pure_python_n}] cold linkage WITHOUT gmpy2 (pure-Python pow) ...", flush=True)
    world = ensure_world(pure_python_n, seed)
    pure = measure_dev(world, 800, reps, 2, extra_env={"FEDHEALTH_NO_GMPY2": "1"}, only_cold_psi=True)
    accel = result["sizes"].get(str(pure_python_n), {}).get("psi_cold_linkage_s")
    result["pure_python_psi"] = {"n": pure_python_n, "linkage_s": summarize(pure["psi_cold_linkage_s"]),
                                 "with_gmpy2_linkage_s": accel}
    return result


def main() -> None:
    ap = common_args("Latency of plain vs PSI vs SMPC vs the full pipeline.")
    ap.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()
    out = run(tuple(args.sizes), args.reps, seed=args.seed, pure_python_n=1000 if 1000 in args.sizes else args.sizes[0])
    path = save_json("performance", out)
    print(f"\n{'N':>6}{'plain':>9}{'psi cold':>10}{'psi warm':>10}{'smpc cold':>11}{'smpc warm':>11}{'FULL cold':>11}{'FULL warm':>11}   (median seconds)")
    for n, e in out["sizes"].items():
        print(f"{n:>6}" + "".join(f"{e[k]['median']:>{w}.3f}" for k, w in (("plain", 9), ("psi_cold", 10), ("psi_warm", 10),
                                                                              ("smpc_cold", 11), ("smpc_warm", 11), ("full_cold", 11), ("full_warm", 11))))
    pp = out["pure_python_psi"]
    print(f"cold linkage at N={pp['n']}: pure Python {pp['linkage_s']['median']:.1f}s"
          + (f" vs gmpy2 {pp['with_gmpy2_linkage_s']['median']:.1f}s" if pp["with_gmpy2_linkage_s"] else ""))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
