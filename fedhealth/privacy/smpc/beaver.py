"""Secure multiplication of secret-shared values with Beaver triples.

THE PROBLEM
    Parties hold additive shares [a] and [b]. They want shares of a*b without revealing a or b.
    Local addition works on shares; local multiplication does not (the cross terms a_i * b_j are missing).

BEAVER'S TRICK (offline/online split)
    OFFLINE   a dealer picks random x, y, computes z = x*y and hands every party a share of each: [x], [y], [z].
              A triple is single-use: it is a one-time pad for ONE multiplication.
    ONLINE    parties compute shares of d = a - x and e = b - y and OPEN them (everyone learns d and e).
              d and e are uniformly random, because x and y are, so they reveal nothing about a or b.
              Then
                  a*b = (x + d)(y + e) = z + d*y + e*x + d*e
              every term except d*e is linear in the dealt shares, so each party computes its share locally:
                  [a*b]_i = [z]_i + d*[y]_i + e*[x]_i            (+ d*e, added by exactly ONE party)

TRUST
    The dealer knows x, y, z but never sees d and e (parties open them to each other, not to the dealer),
    so it cannot recover a or b. If the dealer colludes with a computing party it can (documented in
    THREAT_MODEL.md). In this project the coordinator plays the dealer.
"""
from __future__ import annotations

from .field import P, rand_vector, vec_mul
from .sharing import share_vector


def deal_triples(count: int, n_parties: int) -> list[tuple[list[int], list[int], list[int]]]:
    """DEALER: returns, per party, its shares (x_i, y_i, z_i) of `count` fresh triples."""
    x, y = rand_vector(count), rand_vector(count)
    z = vec_mul(x, y)
    xs, ys, zs = share_vector(x, n_parties), share_vector(y, n_parties), share_vector(z, n_parties)
    return [(xs[i], ys[i], zs[i]) for i in range(n_parties)]


def mask(a_share: list[int], b_share: list[int], x_share: list[int], y_share: list[int]) -> tuple[list[int], list[int]]:
    """PARTY, step 1: my shares of d = a - x and e = b - y. These are broadcast to the other parties."""
    return ([(a - x) % P for a, x in zip(a_share, x_share, strict=True)],
            [(b - y) % P for b, y in zip(b_share, y_share, strict=True)])


def open_values(shares_from_all_parties: list[list[int]]) -> list[int]:
    """PARTY, step 2: add up every party's share of d (or e) to learn the (uniformly random) opened value."""
    return [sum(col) % P for col in zip(*shares_from_all_parties)]


def product_share(d: list[int], e: list[int], x_share: list[int], y_share: list[int], z_share: list[int],
                  adds_constant_term: bool) -> list[int]:
    """PARTY, step 3: my share of a*b = z + d*y + e*x (+ d*e at one designated party)."""
    out = [(z + dd * y + ee * x) % P for z, dd, y, ee, x in zip(z_share, d, y_share, e, x_share, strict=True)]
    if adds_constant_term:
        out = [(o + dd * ee) % P for o, dd, ee in zip(out, d, e, strict=True)]
    return out
