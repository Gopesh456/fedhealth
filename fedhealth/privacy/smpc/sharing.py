"""Additive secret sharing (n-out-of-n).

    secret s  ->  shares s_1 + s_2 + ... + s_n = s  (mod p),   s_1..s_{n-1} uniformly random

* Any n-1 shares are uniformly random and independent of s: they carry NO information (information-theoretic
  security - it does not rely on any hardness assumption).
* Shares can be ADDED locally: if each party adds its shares of a and b, the results are shares of a+b.
  That is why sums/counts need no interaction. Products do (see beaver.py).
"""
from __future__ import annotations

from .field import P, rand_vector, vec_sum


def share_vector(vec: list[int], n: int) -> list[list[int]]:
    """Split a vector into n additive share-vectors (one per party)."""
    if n < 1:
        raise ValueError("need at least one party")
    shares = [rand_vector(len(vec)) for _ in range(n - 1)]
    last = [(v - sum(col)) % P for v, col in zip(vec, zip(*shares))] if n > 1 else [v % P for v in vec]
    return shares + [last]


def reconstruct_vector(shares: list[list[int]]) -> list[int]:
    return [sum(col) % P for col in zip(*shares)]


def reconstruct_scalar(shares: list[int]) -> int:
    return sum(shares) % P


def secure_sum(partials: list[int]) -> int:
    """The textbook secure-sum protocol, simulated in one process (used by tests and as the explanation
    primitive for "each hospital computes a partial result locally and the total is computed without any
    party seeing another's partial"):

      1. party i splits its private partial v_i into n shares and sends share j to party j,
      2. party j adds up the n shares it received:  S_j = sum_i share_{i->j},
      3. an aggregator adds the S_j:  sum_j S_j = sum_i v_i.

    The aggregator sees only the S_j, each uniformly random; party j sees only n-1 random shares of
    every other partial. Nobody sees any other party's partial value."""
    n = len(partials)
    sent = [share_vector([v], n) for v in partials]                 # sent[i][j][0] = share of v_i for party j
    received_sums = [sum(sent[i][j][0] for i in range(n)) % P for j in range(n)]
    return reconstruct_scalar(received_sums)
