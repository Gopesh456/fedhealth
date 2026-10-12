"""Start the whole system locally: 3 hospital nodes + coordinator, each its own uvicorn process.

    python scripts/run_local.py             # DEV: plain http, insecure baselines enabled (evaluation default)
    python scripts/run_local.py --secure    # STRICT: mutual TLS, service identities, private API, no insecure routes
                                            #   (run scripts/gen_keys.py and scripts/gen_certs.py first)

Ctrl-C stops everything. This is the same topology docker-compose.yml describes (Phase 7); it exists
so the project is runnable without Docker. `start_services` / `stop_services` are also imported by
the demo script and the end-to-end tests.
"""
from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedhealth.common.config import load_topology  # noqa: E402
from fedhealth.common.http import client_ssl_context  # noqa: E402
from fedhealth.privacy.encryption.tls import COORDINATOR, TlsPaths, hospital_service_id, tls_dir  # noqa: E402


def start_services(strict: bool = False, secure: bool = False, extra_env: dict[str, str] | None = None,
                   timeout: float = 90.0, quiet: bool = False) -> list[subprocess.Popen]:
    """Launch the 4 services and wait until all answer /health. Returns the process handles.
    strict: do not mount the insecure baseline routes.  secure: strict + mutual TLS + service identities."""
    strict = strict or secure
    topo = load_topology()
    env_base = {**os.environ, **(extra_env or {}), "ALLOW_INSECURE_BASELINE": "0" if strict else "1"}
    certs = tls_dir()
    if secure:
        env_base.update({"FEDHEALTH_STRICT": "1", "FEDHEALTH_SCHEME": "https", "FEDHEALTH_TLS_DIR": str(certs)})
    procs: list[subprocess.Popen] = []

    def launch(label: str, target: str, port: int, extra: dict[str, str], service_id: str | None, require_client_cert: bool) -> None:
        cmd = [sys.executable, "-m", "uvicorn", target, "--factory", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
        if secure:
            paths = TlsPaths(certs, service_id)
            cmd += ["--ssl-keyfile", str(paths.key), "--ssl-certfile", str(paths.cert), "--ssl-ca-certs", str(paths.ca),
                    "--ssl-cert-reqs", str(int(ssl.CERT_REQUIRED if require_client_cert else ssl.CERT_NONE))]
        procs.append(subprocess.Popen(cmd, env={**env_base, **extra}, cwd=str(ROOT)))
        if not quiet:
            print(f"  started {label:<14} {'https' if secure else 'http'}://127.0.0.1:{port}")

    for hid, h in topo.hospitals.items():            # hospitals demand a client certificate (mutual TLS)
        launch(f"hospital {hid}", "fedhealth.hospital.app:create_app_from_env", h.port, {"HOSPITAL_ID": hid},
               hospital_service_id(hid), require_client_cert=True)
    launch("coordinator", "fedhealth.coordinator.app:create_app_from_env", topo.coordinator_port, {}, COORDINATOR,
           require_client_cert=False)                # requesters authenticate with a JWT instead

    scheme = "https" if secure else "http"
    ctx = client_ssl_context(TlsPaths(certs, COORDINATOR)) if secure else True
    pending = {f"{scheme}://127.0.0.1:{h.port}/health" for h in topo.hospitals.values()}
    pending.add(f"{scheme}://127.0.0.1:{topo.coordinator_port}/health")
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        for u in list(pending):
            try:
                if httpx.get(u, timeout=1.0, verify=ctx).status_code == 200:
                    pending.discard(u)
            except httpx.HTTPError:
                pass
        if any(p.poll() is not None for p in procs):
            break
        time.sleep(0.3)
    if pending:
        stop_services(procs)
        raise RuntimeError(f"services did not become healthy: {sorted(pending)}")
    return procs


def stop_services(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="do not mount the insecure baseline routes")
    ap.add_argument("--secure", action="store_true", help="strict + mutual TLS + service identities + private API")
    args = ap.parse_args()
    print("Starting services (Ctrl-C to stop)...")
    procs = start_services(strict=args.strict, secure=args.secure)
    print("All services healthy." + ("  Requesters: https://127.0.0.1:8000 (trust secrets/tls/ca.pem)" if args.secure else ""))
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("A service exited; shutting down.")
    except KeyboardInterrupt:
        pass
    finally:
        stop_services(procs)


if __name__ == "__main__":
    main()
