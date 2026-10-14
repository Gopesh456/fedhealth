"""ATTACK 4 (bonus) - Dictionary attack on keyed-HMAC pseudonymisation: why we use PSI.

THE ATTACK
    Hospitals A and B link patients by exchanging HMAC-SHA256(K, national_id) tokens - the "simple" baseline. B
    publishes the tokens of ALL its patients. Hospital A holds the shared key K, so it can compute the token of ANY
    identifier it can guess, and look it up in B's list. National IDs have little entropy, so A can walk through a
    dictionary of candidate IDs and learn whether each is a patient of B - including people A has never treated.

WHAT WE MEASURE
    * recovery of B's patients from a dictionary that contains the real IDs among many decoys,
    * the measured HMAC throughput and the time to enumerate the WHOLE 9-digit ID space on this machine,
    * and the same attempt against PSI, where A can apply only its own key: nothing outside the intersection.
"""
from __future__ import annotations

import hashlib
import hmac
import time

import numpy as np

from common import common_args, ensure_world, save_json, world_dir
from fedhealth.privacy.linkage.group import get_group
from fedhealth.privacy.linkage.psi import psi_intersection_local
from fedhealth.privacy.linkage.hmac_baseline import pseudonymise
import csv


def run(gt_dir, decoys: int = 200_000, seed: int = 9) -> dict:
    rows = list(csv.DictReader(open(gt_dir / "universe.csv", newline="", encoding="utf-8")))
    a_ids = [r["national_id"] for r in rows if "A" in r["membership"]]
    b_ids = [r["national_id"] for r in rows if "B" in r["membership"]]
    b_only = {r["national_id"] for r in rows if r["membership"] in ("B",)}
    shared = set(a_ids) & set(b_ids)

    key = b"shared-hmac-key-known-to-both-hospitals"
    b_tokens = set(pseudonymise(b_ids, key).values())                      # what B publishes

    rng = np.random.default_rng(seed)
    decoy_ids = [f"SYN-{int(x)}" for x in rng.integers(100_000_000, 999_999_999, decoys)]
    dictionary = list(dict.fromkeys(decoy_ids + [r["national_id"] for r in rows]))   # real IDs hidden among decoys
    t0 = time.perf_counter()
    hits = {i for i in dictionary if hmac.new(key, i.encode(), hashlib.sha256).hexdigest() in b_tokens}
    dt = time.perf_counter() - t0
    rate = len(dictionary) / dt

    # PSI: A's whole view is the transcript; it can only use its own key, so guesses never match outside the intersection
    g = get_group("test768")
    small_a, small_b = a_ids[:300], b_ids[:300]
    transcript: list = []
    psi_intersection_local({"A": small_a, "B": small_b}, "test768", transcript=transcript)
    seen = {v for s, r, kind, p in transcript if r == "A" and kind.startswith("ring") for v in p}
    tags = {v for s, r, kind, p in transcript if r == "A" and kind == "tags" for v in p}
    k_a = g.new_key()
    psi_hits = {c for c in set(small_b) - set(small_a)
                if g.encrypt(g.hash_to_group(c.encode()), k_a) in seen or g.tag(g.encrypt(g.hash_to_group(c.encode()), k_a)) in tags}

    return {"attack": "hmac dictionary", "b_patients": len(b_ids), "b_only_patients": len(b_only), "shared_patients": len(shared),
            "dictionary_size": len(dictionary), "recovered_from_b": len(hits & set(b_ids)),
            "recovered_b_only": len(hits & b_only), "false_positives": len(hits - set(b_ids)),
            "hmac_per_second": rate, "full_9digit_space_seconds": 9e8 / rate,
            "psi_non_shared_patients_recovered": len(psi_hits), "psi_candidates_tested": len(set(small_b) - set(small_a))}


def main() -> None:
    ap = common_args("Dictionary attack on HMAC pseudonyms vs PSI.")
    args = ap.parse_args()
    n = 1000 if args.fast else args.patients
    out = run(ensure_world(n, args.seed, args.source) / "ground_truth")
    path = save_json("attack_hmac_dictionary", out)
    print(f"B published {out['b_patients']} HMAC tokens ({out['shared_patients']} also at A, {out['b_only_patients']} only at B).")
    print(f"  dictionary of {out['dictionary_size']:,} candidate IDs -> attacker recovers {out['recovered_from_b']}/{out['b_patients']} of B's patients "
          f"({out['recovered_b_only']} of them are patients A never treated); false positives: {out['false_positives']}")
    print(f"  one core tests {out['hmac_per_second']:,.0f} IDs/s: the WHOLE 9-digit ID space in ~{out['full_9digit_space_seconds'] / 60:.0f} minutes")
    print(f"  against PSI the same attacker recovers {out['psi_non_shared_patients_recovered']} of {out['psi_candidates_tested']} non-shared patients")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
