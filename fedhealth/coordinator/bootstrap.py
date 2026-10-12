"""One-time creation of every secret the system needs (used by scripts/gen_keys.py and by the tests).

Creates, inside the secrets directory (default ./secrets, gitignored - never commit it):
    hospital_a.key hospital_b.key hospital_c.key   AES-256-GCM data keys, one per hospital
    jwt_secret.key                                 signs requesters' access tokens
    audit_key.key                                  keys the audit log's hash chain
    users.json                                     demo users: {user_id, role, scrypt hash}   (hashes only)
    DEMO_CREDENTIALS.txt                           the demo users' plaintext passwords, for the human at the keyboard

The passwords are random; they are written to a file, never printed.
"""
from __future__ import annotations

import json
import secrets as pysecrets
from pathlib import Path

from fedhealth.privacy.access_control.jwt_auth import hash_password
from fedhealth.privacy.encryption.aesgcm import load_or_create_key, secrets_dir

DEMO_USERS = (("researcher1", "researcher"), ("clinician1", "clinician"), ("auditor1", "auditor"))
KEY_NAMES = ("hospital_a", "hospital_b", "hospital_c", "jwt_secret", "audit_key")


def bootstrap_secrets(directory: Path | None = None, force: bool = False) -> dict[str, str]:
    """Create missing secrets (all of them again with force=True). Returns {demo user: plaintext password} for
    users created in THIS call (empty if users.json already existed)."""
    d = directory or secrets_dir()
    d.mkdir(parents=True, exist_ok=True)
    if force:
        for name in KEY_NAMES:
            (d / f"{name}.key").unlink(missing_ok=True)
        (d / "users.json").unlink(missing_ok=True)
    for name in KEY_NAMES:
        load_or_create_key(name, d)
    users_file, creds_file = d / "users.json", d / "DEMO_CREDENTIALS.txt"
    if users_file.exists():
        return {}
    creds = {user: pysecrets.token_urlsafe(12) for user, _ in DEMO_USERS}
    users_file.write_text(json.dumps([{"user_id": u, "role": r, "pw_hash": hash_password(creds[u])} for u, r in DEMO_USERS], indent=2),
                          encoding="utf-8")
    creds_file.write_text("DEMO ONLY - synthetic system. Keep this file private (it is gitignored).\n\n"
                          + "\n".join(f"{u:<12} role={r:<11} password={creds[u]}" for u, r in DEMO_USERS) + "\n", encoding="utf-8")
    return creds
