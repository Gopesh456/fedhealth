"""The query language: only aggregates, only whitelisted fields, each hospital sees only its own slice."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fedhealth.common.schemas import (QuerySpec, categories, local_query, participating_hospitals, value_bounds)

from .helpers import headline_spec


def test_headline_query_involves_all_three_hospitals():
    assert participating_hospitals(headline_spec()) == ["A", "B", "C"]


def test_each_hospital_receives_only_its_own_predicate():
    spec = headline_spec()
    la, lb, lc = (local_query(spec, h) for h in "ABC")
    assert la.predicate == {"age_min": 40, "age_max": 65} and la.condition == "E11"
    assert lb.predicate == {"outcome": "improved"}
    assert lc.predicate == {"drug": "metformin"}
    # nothing about the other hospitals' filters leaks into a slice
    assert "drug" not in str(la.model_dump()) and "improved" not in str(lc.model_dump())


def test_value_field_owner_participates_even_without_a_predicate():
    spec = QuerySpec.model_validate({"agg": "avg", "condition": "E11", "a": {}, "value_field": "lab_delta"})
    assert participating_hospitals(spec) == ["A", "B"]
    assert local_query(spec, "B").value_field == "lab_delta" and local_query(spec, "A").value_field is None


@pytest.mark.parametrize("bad", [
    {"agg": "select", "condition": "E11", "a": {}},                         # not an aggregate
    {"agg": "count", "condition": "E11", "a": {}, "columns": ["name"]},     # row-level smuggling
    {"agg": "count", "condition": "E11", "a": {}, "limit": 10},
    {"agg": "count", "condition": "E11", "a": {}, "where": "age > 3"},      # free-form SQL
    {"agg": "count", "condition": "E11"},                                    # no hospital involved
    {"agg": "count", "condition": "ZZZ", "a": {}},                           # unknown condition
    {"agg": "count", "condition": "E11", "c": {"drug": "sacubitril-valsartan"}},   # not in E11's formulary
    {"agg": "count", "condition": "E11", "a": {}, "group_by": "sex"},        # count takes no group_by
    {"agg": "histogram", "condition": "E11", "a": {}},                       # histogram needs group_by
    {"agg": "sum", "condition": "E11", "a": {}},                             # sum needs value_field
    {"agg": "sum", "condition": "E11", "a": {}, "value_field": "full_name"}, # not a value field
    {"agg": "histogram", "condition": "E11", "a": {}, "group_by": "zip_code"},  # quasi-identifier not allowed
    {"agg": "count", "condition": "E11", "a": {"zip_code": "12345"}},        # quasi-identifier not allowed
])
def test_non_aggregate_or_non_whitelisted_queries_are_rejected(bad):
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(bad)


def test_categories_and_bounds():
    assert categories("outcome", "E11") == ["improved", "no_change", "worsened"]
    assert "metformin" in categories("drug", "E11") and "salbutamol" not in categories("drug", "E11")
    assert value_bounds("age", "E11") == (0, 120)
    assert value_bounds("adherence_pct", "I10") == (0, 100)
    assert value_bounds("lab_delta", "E11") == (-400, 400)     # HbA1c clamp 4.0 x scale 100
