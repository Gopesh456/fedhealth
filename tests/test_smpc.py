"""Phase 3a: additive secret sharing, Beaver multiplication, the dealer and the noise inputs (no services)."""
from __future__ import annotations

import random

import pytest

from fedhealth.privacy.smpc import beaver
from fedhealth.privacy.smpc.field import P, decode_vector, encode_vector, rand_vector, to_signed
from fedhealth.privacy.smpc.protocol import (MpcParty, Program, Roles, assemble_result, assign_roles,
                                             build_input_vectors, deal, expand_seed, expand_triples, n_outputs,
                                             run_local)
from fedhealth.privacy.smpc.sharing import reconstruct_scalar, reconstruct_vector, secure_sum, share_vector


def _chi_square_uniform(values: list[int], buckets: int = 16) -> float:
    """Chi-square statistic of the top bits of field elements vs. uniform (df = buckets-1)."""
    counts = [0] * buckets
    for v in values:
        counts[min(v * buckets // P, buckets - 1)] += 1
    expected = len(values) / buckets
    return sum((c - expected) ** 2 / expected for c in counts)


CHI2_P001_DF15 = 37.7        # chi-square critical value, df=15, p=0.001 -> a fair test fails < 0.1% of the time


# ============================================================================ field & wire encoding
def test_field_basics():
    assert P == 2 ** 61 - 1 and to_signed(P - 5) == -5 and to_signed(7) == 7
    v = rand_vector(50)
    assert decode_vector(encode_vector(v)) == v
    with pytest.raises(ValueError):
        decode_vector(b"\xff" * 8)               # >= p is rejected


# ============================================================================ additive sharing
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_share_and_reconstruct(n):
    secret = [random.randrange(P) for _ in range(40)]
    shares = share_vector(secret, n)
    assert len(shares) == n and all(len(s) == 40 for s in shares)
    assert reconstruct_vector(shares) == secret


def test_shares_add_locally():
    a, b = [3, 10, P - 1], [4, 20, 5]
    sa, sb = share_vector(a, 3), share_vector(b, 3)
    summed = [[(x + y) % P for x, y in zip(u, v)] for u, v in zip(sa, sb)]
    assert reconstruct_vector(summed) == [(x + y) % P for x, y in zip(a, b)]


@pytest.mark.parametrize("secret_value", [0, 1])
def test_any_n_minus_1_shares_are_uniform_whatever_the_secret(secret_value):
    """Information-theoretic hiding, checked statistically: shares of an all-0 or all-1 vector are
    indistinguishable from uniform noise to anyone missing one share."""
    n = 3
    shares = share_vector([secret_value] * 20000, n)
    for held in (shares[0], shares[1], shares[2]):           # each single share
        assert _chi_square_uniform(held) < CHI2_P001_DF15
    both = [(a + b) % P for a, b in zip(shares[0], shares[1])]   # even two of three shares combined
    assert _chi_square_uniform(both) < CHI2_P001_DF15


def test_secure_sum_matches_the_plain_sum():
    for partials in ([5, 7, 11], [0, 0, 0], [1000, 2, 30, 4], [42]):
        assert secure_sum(partials) == sum(partials)


# ============================================================================ Beaver multiplication
@pytest.mark.parametrize("n", [2, 3, 4])
def test_beaver_multiplication_is_correct(n):
    m = 200
    a, b = rand_vector(m), rand_vector(m)
    sa, sb = share_vector(a, n), share_vector(b, n)
    triples = beaver.deal_triples(m, n)
    masked = [beaver.mask(sa[i], sb[i], triples[i][0], triples[i][1]) for i in range(n)]
    d = beaver.open_values([mk[0] for mk in masked])
    e = beaver.open_values([mk[1] for mk in masked])
    prod_shares = [beaver.product_share(d, e, *triples[i], adds_constant_term=(i == 0)) for i in range(n)]
    assert reconstruct_vector(prod_shares) == [(x * y) % P for x, y in zip(a, b)]


def test_the_constant_term_must_be_added_exactly_once():
    n, m = 3, 20
    a, b = rand_vector(m), rand_vector(m)
    sa, sb = share_vector(a, n), share_vector(b, n)
    tr = beaver.deal_triples(m, n)
    masked = [beaver.mask(sa[i], sb[i], tr[i][0], tr[i][1]) for i in range(n)]
    d, e = beaver.open_values([x[0] for x in masked]), beaver.open_values([x[1] for x in masked])
    never = reconstruct_vector([beaver.product_share(d, e, *tr[i], adds_constant_term=False) for i in range(n)])
    assert never != [(x * y) % P for x, y in zip(a, b)]


def test_opened_values_do_not_depend_on_the_secret():
    """d = a - x is uniform because x is: opening it leaks nothing, even for a constant secret."""
    m, n = 20000, 3
    zeros = share_vector([0] * m, n)
    ones = share_vector([1] * m, n)
    tr = beaver.deal_triples(m, n)
    for secret_shares in (zeros, ones):
        d = beaver.open_values([beaver.mask(secret_shares[i], secret_shares[i], tr[i][0], tr[i][1])[0] for i in range(n)])
        assert _chi_square_uniform(d) < CHI2_P001_DF15


# ============================================================================ the whole protocol
def _plain(vectors: dict[str, list[list[int]]], owner: str) -> list[int]:
    """What the answer would be if everything were in the clear."""
    m = len(vectors[owner][0])
    factors = [v[0] for p, v in vectors.items() if p != owner]
    prod = [1] * m
    for f in factors:
        prod = [(x * y) % P for x, y in zip(prod, f)]
    return [sum(p * o for p, o in zip(prod, out)) % P for out in vectors[owner]]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("t", [1, 2, 6])
def test_secure_dot_matches_plaintext(n, t):
    rng = random.Random(n * 10 + t)
    m = 120
    names = list("ABCD")[:n]
    owner = names[-1]
    vectors = {}
    for p in names:
        if p == owner:
            vectors[p] = [[rng.choice([0, 1, 5, P - 3]) for _ in range(m)] for _ in range(t)]
        else:
            vectors[p] = [[rng.randint(0, 1) for _ in range(m)]]
    assert run_local(vectors, owner, m) == _plain(vectors, owner)


def test_program_shapes():
    assert Program(1, 3, 100).layer_sizes == []
    assert Program(2, 3, 100).layer_sizes == [300]
    assert Program(3, 1, 100).layer_sizes == [100, 100]
    assert Program(3, 4, 100).layer_sizes == [100, 400] and Program(3, 4, 100).total_triples == 500


def test_input_vector_builders_and_result_assembly():
    matches = [(0, 5, "improved"), (2, -3, "worsened"), (3, 7, "improved")]
    cats = ["improved", "no_change", "worsened"]
    assert build_input_vectors(matches, 5, is_owner=False, agg="sum") == [[1, 0, 1, 1, 0]]
    assert build_input_vectors(matches, 5, is_owner=True, agg="count") == [[1, 0, 1, 1, 0]]
    sum_vecs = build_input_vectors(matches, 5, is_owner=True, agg="sum")
    assert len(sum_vecs) == 2 and sum_vecs[0] == [1, 0, 1, 1, 0]            # [count vector, value vector]
    assert [to_signed(x) for x in sum_vecs[1]] == [5, 0, -3, 7, 0]
    assert len(build_input_vectors(matches, 5, is_owner=True, agg="avg")) == 2
    assert n_outputs("sum") == n_outputs("avg") == 2 and n_outputs("count") == 1
    hist = build_input_vectors(matches, 5, is_owner=True, agg="histogram", categories=cats)
    assert hist == [[1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]] and n_outputs("histogram", 3) == 3
    assert assemble_result("histogram", [2, 0, 1], cats) == {"improved": 2, "no_change": 0, "worsened": 1}
    assert assemble_result("avg", [4, (-8) % P]) == -2.0 and assemble_result("avg", [0, 0]) is None


def test_end_to_end_semantics_count_sum_avg_histogram():
    """Three hospitals over a 6-patient linked cohort; compare every aggregate with plain arithmetic."""
    a_ind, b_rows = [1, 1, 0, 1, 1, 0], [(0, -40, "improved"), (1, 25, "no_change"), (3, 10, "improved"),
                                         (4, -5, "worsened"), (5, 99, "improved")]
    c_ind_rows = [(0, None, None), (1, None, None), (3, None, None), (4, None, None)]
    a_rows = [(i, None, None) for i, v in enumerate(a_ind) if v]
    cats = ["improved", "no_change", "worsened"]
    for agg, expect in [("count", 4), ("sum", -40 + 25 + 10 - 5), ("avg", (-40 + 25 + 10 - 5) / 4)]:
        vec = {"A": build_input_vectors(a_rows, 6, is_owner=False, agg=agg),
               "C": build_input_vectors(c_ind_rows, 6, is_owner=False, agg=agg),
               "B": build_input_vectors(b_rows, 6, is_owner=True, agg=agg)}
        got = assemble_result(agg, run_local(vec, "B", 6))
        assert got == pytest.approx(expect)
    vec = {"A": build_input_vectors(a_rows, 6, is_owner=False, agg="histogram"),
           "C": build_input_vectors(c_ind_rows, 6, is_owner=False, agg="histogram"),
           "B": build_input_vectors(b_rows, 6, is_owner=True, agg="histogram", categories=cats)}
    assert assemble_result("histogram", run_local(vec, "B", 6), cats) == {"improved": 2, "no_change": 1, "worsened": 1}


@pytest.mark.parametrize("n", [1, 2, 3])
def test_noise_shares_are_added_inside_the_computation(n):
    """Every involved hospital feeds one noise vector in; the opened cells are exactly  exact + sum of the noise."""
    rng = random.Random(n)
    m, t = 50, 3
    names = list("ABC")[:n]
    owner = names[-1]
    vectors = {p: ([[rng.randint(0, 1) for _ in range(m)] for _ in range(t)] if p == owner
                   else [[rng.randint(0, 1) for _ in range(m)]]) for p in names}
    involved = names if n != 2 else names + ["D"]
    noise = {p: [rng.randint(-50, 50) for _ in range(t)] for p in (involved if n > 1 else names)}
    got = [to_signed(x) for x in run_local(vectors, owner, m, noise=noise)]
    exact = [to_signed(x) for x in _plain(vectors, owner)]
    assert got == [e + sum(nv[j] for nv in noise.values()) for j, e in enumerate(exact)]


# ============================================================================ roles and the dealer
def test_role_assignment_keeps_the_dealer_out_of_the_computation():
    r3 = assign_roles(["A", "B", "C"], "B", ["A", "B", "C"])
    assert r3.computers == ("A", "B") and r3.dealer == "C" and r3.involved == ("A", "B", "C")
    r2 = assign_roles(["A", "C"], "C", ["A", "B", "C"])
    assert r2.computers == ("A", "C") and r2.dealer == "B" and r2.involved == ("A", "C", "B")   # the bystander deals
    for r in (r3, r2):
        assert r.dealer not in r.computers
    with pytest.raises(ValueError):
        assign_roles(["A", "B"], "B", ["A", "B"])                    # nobody left to deal
    with pytest.raises(ValueError):
        assign_roles(["A"], "A", ["A", "B", "C"])


def test_dealt_triples_are_correct_and_seed_compressed():
    seed, (x1, y1, z1) = deal(500)
    x0, y0, z0 = expand_triples(seed, 500)
    x, y = [(a + b) % P for a, b in zip(x0, x1)], [(a + b) % P for a, b in zip(y0, y1)]
    assert [(a + b) % P for a, b in zip(z0, z1)] == [(a * b) % P for a, b in zip(x, y)]
    assert len(seed) == 32 and expand_triples(seed, 500) == (x0, y0, z0)          # P0 re-derives the same shares
    assert expand_seed(seed, 10, b"x") != expand_seed(seed, 10, b"y")             # domain separation


def test_each_computers_triple_shares_look_uniform():
    """What a single computer holds about the triples (its own share) is uniform: z0 from the PRG, z1 = xy - z0."""
    seed, (x1, y1, z1) = deal(20000)
    x0, _, z0 = expand_triples(seed, 20000)
    for held in (x0, z0, x1, z1):
        assert _chi_square_uniform(held) < CHI2_P001_DF15


# ============================================================================ state-machine guards
def _two_parties(m=8, t=1):
    prog = Program(2, t, m)
    roles = Roles(("A", "B"), "B", ("A", "B"), "C")
    a = MpcParty("A", roles, prog, own_vectors=[[1] * m])
    b = MpcParty("B", roles, prog, own_vectors=[[1] * m for _ in range(t)])
    for src, dst in ((a, b), (b, a)):
        for other, shares in src.split_inputs().items():
            dst.receive_input_shares(src.name, shares)
    return prog, a, b


def test_cannot_open_before_triples_or_use_a_triple_twice():
    prog, a, b = _two_parties()
    with pytest.raises(RuntimeError):
        a.open_layer(0)                                              # no triples yet
    seed, explicit = deal(prog.total_triples)
    a.set_triples_from_seed(seed)
    b.set_triples(*explicit)
    with pytest.raises(ValueError):
        b.set_triples(*explicit)                                     # triples arrive once
    a.open_layer(0)
    with pytest.raises(RuntimeError):
        a.open_layer(0)                                              # one triple, one use
    with pytest.raises(RuntimeError):
        a.finish_layer(0)                                            # B's opening has not arrived


def test_malformed_input_shares_are_rejected():
    prog, a, b = _two_parties()
    roles = Roles(("A", "B"), "B", ("A", "B"), "C")
    fresh = MpcParty("A", roles, prog)
    with pytest.raises(ValueError):
        fresh.receive_input_shares("B", [[1, 2, 3]])                 # wrong length
    with pytest.raises(ValueError):
        fresh.receive_input_shares("Z", [[0] * 8])                   # not a party
    with pytest.raises(ValueError):
        fresh.receive_input_shares("C", [[0] * 8])                   # the dealer holds no input here
    with pytest.raises(ValueError):
        fresh.receive_input_shares("B", [[0] * 8, [0] * 8])          # owner must send exactly t vectors
    dealer = MpcParty("C", roles, prog)
    with pytest.raises(ValueError):
        dealer.receive_input_shares("A", [[0] * 8])                  # the dealer never receives shares
    with pytest.raises(RuntimeError):
        dealer.open_layer(0)                                         # ... and never takes part in an opening
    with pytest.raises(ValueError):
        a.receive_open(0, "C", [0] * 8, [0] * 8)                     # openings come from the other computer only
