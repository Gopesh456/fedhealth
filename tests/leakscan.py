"""The privacy-leak scanner.

Given every message that crossed a service boundary, report anything that must never travel:
  * national IDs, local MRNs, names, ZIP codes, dates of birth (by value, by pattern and by field name),
  * the *plain* group-hash of any identifier (a PSI party must only ever send encrypted values).
The plain insecure baseline is EXPECTED to fail this scan - that contrast is the point of the test.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Iterable

from fedhealth.privacy.linkage.group import Group, unpack_ints

FORBIDDEN_FIELD_NAMES = ("national_id", "full_name", "zip_code", "date_of_birth", "patient_id")
FORBIDDEN_PATTERNS = (re.compile(r"SYN-\d{3,}"), re.compile(r"\b[ABC]-\d{6}\b"))


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _blob_ints(text: str, nbytes: int) -> list[int]:
    """Decode every base64 'blob' field found in a JSON message into group elements."""
    out: list[int] = []
    try:
        doc = json.loads(text)
    except (ValueError, TypeError):
        return out
    stack = [doc]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k == "blob" and isinstance(v, str):
                    try:
                        out.extend(unpack_ints(base64.b64decode(v), nbytes))
                    except ValueError:
                        pass
                else:
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return out


def scan(events, *, group: Group | None = None, plain_ids: Iterable[str] = (), extra_forbidden: Iterable[str] = ()) -> list[str]:
    """Return a list of human-readable violations (empty list = clean)."""
    violations: list[str] = []
    plain_hashes = {group.hash_to_group(i.encode()) for i in plain_ids} if group else set()
    for ev in events:
        for direction, text in (("request", ev.request), ("response", ev.response)):
            where = f"{ev.src}->{ev.dst} {ev.method} {ev.path} [{direction}]"
            for token in _walk_json(text):
                if isinstance(token, str):
                    if token in FORBIDDEN_FIELD_NAMES:
                        violations.append(f"{where}: field name {token!r}")
                    for pat in FORBIDDEN_PATTERNS:
                        if pat.search(token):
                            violations.append(f"{where}: identifier pattern {pat.pattern!r} in {token[:40]!r}")
                    for needle in extra_forbidden:
                        if needle and needle in token:
                            violations.append(f"{where}: forbidden value {needle!r}")
            if group and plain_hashes:
                hit = plain_hashes.intersection(_blob_ints(text, group.nbytes))
                if hit:
                    violations.append(f"{where}: {len(hit)} PLAIN hashed identifier(s) on the wire")
    return violations


def _walk_json(text: str):
    if not text:
        return
    try:
        doc = json.loads(text)
    except ValueError:
        yield text
        return
    yield from _walk(doc)
