"""Baseline for comparison: keyed-HMAC pseudonymisation ("tokenisation").

HOW IT WORKS
    All hospitals share one secret key K. Each computes  token(id) = HMAC-SHA256(K, id)  for every
    patient and sends its whole token list to the others (or to a coordinator). Equal tokens <=> same patient.

WHY IT IS ONLY A BASELINE
    * Fast and simple (one hash per patient), and the tokens are not reversible *without K*.
    * BUT everyone who holds K can run a DICTIONARY ATTACK: identifiers have little entropy (a 9-digit
      national ID has 10^9 possibilities), so anyone with K can pseudonymise every candidate ID and look
      it up in the other hospital's list - learning whether ANY person is a patient there, not just the
      shared ones. PSI reveals only the intersection.
    * Every hospital must send its entire list, so each learns the full token list of the others.
    * If K leaks (or the coordinator is given it), all hospitals' lists are exposed.
    See tests/test_psi.py (membership leak) and attacks/hmac_dictionary.py.
"""
from __future__ import annotations

import hashlib
import hmac


def pseudonymise(ids: list[str], key: bytes) -> dict[str, str]:
    """{id: token}"""
    return {i: hmac.new(key, i.encode(), hashlib.sha256).hexdigest() for i in ids}


def intersect_tokens(lists: dict[str, dict[str, str]]) -> dict[str, set[str]]:
    """Given {party: {id: token}}, return {party: ids that are shared with EVERY other party}.
    (Note what this needs: every party's complete token list, in one place.)"""
    token_sets = [set(m.values()) for m in lists.values()]
    common = set.intersection(*token_sets)
    return {party: {i for i, t in m.items() if t in common} for party, m in lists.items()}
