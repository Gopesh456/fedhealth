"""Create all secrets (data keys, JWT secret, audit key, demo users).

    python scripts/gen_keys.py            # creates what is missing, never overwrites
    python scripts/gen_keys.py --force    # regenerate EVERYTHING (the encrypted databases become unreadable:
                                          #   re-run `python -m fedhealth.synth.generate` afterwards)

Passwords are written to secrets/DEMO_CREDENTIALS.txt, not printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedhealth.coordinator.bootstrap import bootstrap_secrets  # noqa: E402
from fedhealth.privacy.encryption.aesgcm import secrets_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate every secret")
    args = ap.parse_args()
    creds = bootstrap_secrets(force=args.force)
    d = secrets_dir()
    print(f"Secrets directory: {d}")
    print("  data keys, JWT secret and audit key: ready")
    print(f"  demo users: {'created - passwords are in ' + str(d / 'DEMO_CREDENTIALS.txt') if creds else 'already existed (unchanged)'}")
    if args.force:
        print("  NOTE: regenerate the databases now: python -m fedhealth.synth.generate --patients 5000 --seed 42")


if __name__ == "__main__":
    main()
