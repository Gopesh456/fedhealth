"""One-click start for the website: the 3 hospitals + coordinator (strict mode: mutual TLS) and the dashboard.

    python scripts/start_website.py            # what start-website.bat runs
    python scripts/start_website.py --reset    # first forget every budget and audit entry (deletes data/coordinator.db)

Creates the secrets and the hospital databases if they are missing, starts every service, waits until they answer,
opens the dashboard in the browser and prints the demo logins. Ctrl-C (or closing the window) stops everything.
The dashboard runs in demo mode, so it can show the true value next to the noisy answer.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fedhealth.common.config import data_dir, load_topology  # noqa: E402
from fedhealth.privacy.encryption.tls import tls_dir  # noqa: E402
from run_local import start_services, stop_services  # noqa: E402

DASHBOARD_PORT = 8501
URL = f"http://127.0.0.1:{DASHBOARD_PORT}"


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=str(ROOT), check=True)


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def prepare_secrets_and_data() -> None:
    """First use only: keys, certificates and the encrypted hospital databases (never overwrites what exists)."""
    secrets = ROOT / "secrets"
    keys = [secrets / f"hospital_{h}.key" for h in "abc"]
    had_keys = all(k.exists() for k in keys)
    if not (had_keys and (secrets / "jwt_secret.key").exists() and (secrets / "users.json").exists()):
        print("First use: creating the keys and demo users ...")
        run("scripts/gen_keys.py")
    if not (tls_dir() / "ca.pem").exists():
        print("First use: creating the TLS certificates ...")
        run("scripts/gen_certs.py")
    dbs = [data_dir() / f"hospital_{h}.db" for h in "abc"]
    if not had_keys or not all(db.exists() for db in dbs):
        print("First use: building the three hospital databases from the Synthea extract (about a minute) ...")
        run("-m", "fedhealth.synth.generate")


def reset_coordinator() -> None:
    """Budgets, logins and the audit log live in the coordinator's own database; it is rebuilt on the next start."""
    db = load_topology().coordinator_db_path
    for path in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        if path.exists():
            path.unlink()
    print("Budgets and audit log reset.")


def start_dashboard() -> subprocess.Popen:
    env = {**os.environ, "FEDHEALTH_DEMO_MODE": "1"}
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "fedhealth" / "dashboard" / "app.py"),
           "--server.port", str(DASHBOARD_PORT), "--server.address", "127.0.0.1", "--server.headless", "true",
           "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal"]
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def wait_for_dashboard(proc: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("the dashboard exited during start-up (see the messages above)")
        try:
            if httpx.get(f"http://127.0.0.1:{DASHBOARD_PORT}/_stcore/health", timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("the dashboard did not answer within two minutes")


def print_logins() -> None:
    creds = ROOT / "secrets" / "DEMO_CREDENTIALS.txt"
    if creds.exists():
        print("Demo logins (from secrets/DEMO_CREDENTIALS.txt):")
        for line in creds.read_text(encoding="utf-8").splitlines():
            if "role=" in line:
                print("   " + line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Start the FedHealth services and the dashboard.")
    ap.add_argument("--reset", action="store_true", help="forget every budget and audit entry first")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args()

    loopback = "127.0.0.1,localhost,::1"            # a system or corporate proxy must not intercept local traffic
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = f"{os.environ[key]},{loopback}" if os.environ.get(key) else loopback

    topo = load_topology()
    ports = [h.port for h in topo.hospitals.values()] + [topo.coordinator_port, DASHBOARD_PORT]
    busy = [p for p in ports if port_in_use(p)]
    if busy:
        which = f"Port {busy[0]} is" if len(busy) == 1 else f"Ports {', '.join(map(str, busy))} are"
        print(f"{which} already in use. Is FedHealth already running in another window? Close that window first.")
        return 1

    prepare_secrets_and_data()
    if args.reset:
        reset_coordinator()

    print("Starting the three hospitals and the coordinator ...")
    try:
        procs = start_services(secure=True, timeout=180)
    except RuntimeError as e:
        print(f"Could not start the services: {e}")
        return 1
    try:
        print("Starting the dashboard ...")
        dashboard = start_dashboard()
        procs.append(dashboard)
        wait_for_dashboard(dashboard)
        print()
        print("=" * 72)
        print(f" FedHealth is running.  Website: {URL}")
        print("=" * 72)
        print_logins()
        print()
        print("The first query for a set of hospitals links their patients privately, which can take a minute;")
        print("later queries take under a second.")
        print("To stop everything, close this window or press Ctrl-C.")
        if not args.no_browser:
            webbrowser.open(URL)
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("A service stopped unexpectedly (see the messages above); shutting everything down.")
        return 1
    except RuntimeError as e:
        print(f"Could not start the dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        print("Stopping ...")
        return 0
    finally:
        stop_services(procs)


if __name__ == "__main__":
    sys.exit(main())
