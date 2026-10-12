"""Create a private CA and one certificate per service (coordinator, hospital-a/b/c) for mutual TLS.

    python scripts/gen_certs.py [--dns extra.name ...]

Files land in ./secrets/tls (or $FEDHEALTH_TLS_DIR). Local use and Docker Compose both work: every certificate
already names the service (e.g. `hospital-a`), `localhost` and 127.0.0.1; add --dns for any other host name.
Existing certificates are overwritten.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedhealth.privacy.encryption import tls  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dns", nargs="*", default=[], help="additional DNS names to put in every certificate")
    args = ap.parse_args()
    out = tls.generate_all(extra_dns=tuple(args.dns))
    print(f"CA + {len(tls.SERVICES)} service certificates written to {out}")
    for svc in tls.SERVICES:
        print(f"  {svc:<12} {svc}.crt / {svc}.key")


if __name__ == "__main__":
    main()
