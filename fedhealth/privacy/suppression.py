"""Small-cell suppression (the "k-threshold").

THE RISK
    A count of 1 or 2 describes (almost) individuals: "exactly one patient aged 63 with condition X took drug Y".
    Statistical agencies have long refused to publish cells smaller than a threshold k (here k = 10 by default).

HOW WE APPLY IT
    * COUNT / SUM / AVG: the *cohort* (number of patients behind the answer) must be >= k, else the answer is
      withheld. For SUM and AVG this is the patient COUNT, not the value of the sum.
    * HISTOGRAM: each bin is checked on its own; small bins are replaced by "suppressed".
    * The check uses the EXACT count by default (k_basis="exact", as specified). Honest limitation: whether a
      result was suppressed is itself one bit of information about the exact count ("< k"), and it is NOT covered
      by the differential-privacy guarantee. k_basis="noisy" tests the noisy value instead: that is pure
      post-processing and stays inside the DP guarantee, at the cost of occasionally releasing/suppressing
      the wrong side of the boundary. The privacy budget is charged either way.

ON ITS OWN THE K-THRESHOLD DOES NOT STOP DIFFERENCING ATTACKS: two LARGE queries that differ by one patient are
both released. That is exactly why differential privacy is added (see attacks/differencing.py).
"""
from __future__ import annotations

DEFAULT_K = 10


def is_suppressed(count: int, k: int = DEFAULT_K) -> bool:
    """A cell is suppressed when fewer than k patients stand behind it (zero included)."""
    return count < k


def suppress_cells(cells: dict[str, int], k: int = DEFAULT_K) -> dict[str, bool]:
    """Per-cell decision for a histogram: {bin: suppressed?}."""
    return {name: is_suppressed(count, k) for name, count in cells.items()}
