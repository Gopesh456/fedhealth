"""Test helpers: a random-but-valid QuerySpec generator used for differential testing
(pipeline result must equal the ground-truth result for ANY valid query, not just hand-picked ones)."""
from __future__ import annotations

import numpy as np

from fedhealth.common import vocab
from fedhealth.common.schemas import QuerySpec


def random_spec(rng: np.random.Generator) -> QuerySpec:
    cond = str(rng.choice(vocab.CONDITION_CODES))
    d: dict = {"condition": cond}

    # which hospitals take part (at least one)
    take = {h: rng.random() < 0.7 for h in "abc"}
    if not any(take.values()):
        take[str(rng.choice(list("abc")))] = True

    if take["a"]:
        a: dict = {}
        if rng.random() < 0.6:
            lo = int(rng.integers(22, 70))
            a["age_min"], a["age_max"] = lo, lo + int(rng.integers(5, 40))
        if rng.random() < 0.3:
            a["sex"] = str(rng.choice(vocab.SEXES))
        d["a"] = a
    if take["b"]:
        d["b"] = {"outcome": str(rng.choice(vocab.OUTCOMES))} if rng.random() < 0.7 else {}
    if take["c"]:
        d["c"] = {"drug": str(rng.choice(vocab.drug_names(cond)))} if rng.random() < 0.7 else {}

    kind = str(rng.choice(["count", "count", "sum", "avg", "histogram"]))
    d["agg"] = kind
    if kind in ("sum", "avg"):
        d["value_field"] = str(rng.choice(["age", "lab_delta", "adherence_pct"]))
    elif kind == "histogram":
        d["group_by"] = str(rng.choice(["sex", "age_band", "outcome", "drug"]))
    return QuerySpec.model_validate(d)


def headline_spec(**overrides) -> QuerySpec:
    """'How many patients with condition X prescribed Y showed a positive response?'"""
    d = {"agg": "count", "condition": "E11", "a": {"age_min": 40, "age_max": 65},
         "b": {"outcome": "improved"}, "c": {"drug": "metformin"}}
    d.update(overrides)
    return QuerySpec.model_validate(d)
