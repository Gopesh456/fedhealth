"""Small-cell suppression (the "k-threshold").

THE RISK
    A count of 1 or 2 describes (almost) individuals: "exactly one patient aged 63 with condition X took drug Y".
    Statistical agencies have long refused to publish cells smaller than a threshold k (here k = 10 by default).

HOW WE APPLY IT
    * COUNT / SUM / AVG: the *cohort* (number of patients behind the answer) must be >= k, else the answer is
      withheld. For SUM and AVG this is the patient COUNT, not the value of the sum.
    * HISTOGRAM: each bin is checked on its own; small bins are replaced by "suppressed".
    * The check is applied to the NOISY count. The noise is added inside the secure computation, so no exact count
      exists anywhere outside the MPC: the suppress/release decision is post-processing of a differentially private
      value and is covered by the DP guarantee (earlier versions tested the exact count, which leaked "count < k"
      outside DP). The price is a fuzzy boundary: a true count of 10 is occasionally suppressed, a true 9 occasionally
      released. The privacy budget is charged either way.

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
