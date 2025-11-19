"""
Tests for the scalar validation functions in multinull_jsd._validators.
"""
# noinspection PyProtectedMember
from multinull_jsd._validators import validate_bounded_value, validate_int_value
from hypothesis import given, strategies as st
from typing import TypeAlias, Optional, Any, cast

import numpy as np

import pytest

ScalarInt: TypeAlias = int | np.integer
ScalarFloat: TypeAlias = float | np.floating
Number: TypeAlias = ScalarInt | ScalarFloat


@pytest.mark.parametrize(
    argnames="val, min_v, max_v, expected_type",
    argvalues=[
        (3, None, None, int), (2.5, 0.0, 3.0, float),
        (np.int64(7), 0, 10, np.integer), (np.float64(2.5), 0.0, 3.0, np.floating)
    ]
)
def test_validate_bounded_value_accepts_reals_and_preserves_type(
    val: Number, min_v: Optional[float], max_v: Optional[float], expected_type: type
) -> None:
    """
    Test that validate_bounded_value accepts valid inputs and preserves type.
    """
    out: Number = validate_bounded_value(name="x", value=cast(Any, val), min_value=min_v, max_value=max_v)
    assert float(out) == float(val)
    assert isinstance(out, expected_type)


def test_validate_bounded_value_inclusive_bounds() -> None:
    """
    Test that validate_bounded_value accepts values equal to the bounds.
    """
    assert validate_bounded_value(name="x", value=5, min_value=5, max_value=10) == 5
    assert validate_bounded_value(name="x", value=10.0, min_value=5.0, max_value=10.0) == 10.0


def test_validate_bounded_value_inconsistent_bounds_raises() -> None:
    """
    Test that validate_bounded_value raises ValueError for inconsistent bounds.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_bounded_value(name="x", value=1, min_value=5, max_value=2)


@pytest.mark.parametrize(
    argnames="val, min_v, max_v", argvalues=[(-1, 0, None), (11, None, 10), (1.001, 2.0, 3.0), (9.999, 0.0, 9.0)]
)
def test_validate_bounded_value_out_of_bounds_raises(
    val: Number, min_v: Optional[float], max_v: Optional[float]
) -> None:
    """
    Test that validate_bounded_value raises ValueError for out-of-bounds values.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_bounded_value(name="x", value=cast(Any, val), min_value=min_v, max_value=max_v)


@pytest.mark.parametrize(argnames="bad_number", argvalues=[True, np.bool_(True), "3", object()])
def test_validate_bounded_value_rejects_non_real_or_bool(bad_number: Any) -> None:
    """
    Test that validate_bounded_value raises TypeError for non-numeric values.
    """
    with pytest.raises(expected_exception=TypeError):
        validate_bounded_value(name="x", value=bad_number, min_value=0, max_value=10)


@given(
    v=st.integers(min_value=-10**6, max_value=10**6),
    a=st.integers(min_value=0, max_value=10**6),
    b=st.integers(min_value=0, max_value=10**6),
)
def test_validate_bounded_value_property_inclusive_range(v: Number, a: Number, b: Number) -> None:
    """
    Property: for integer v within inclusive [min,max], function returns Python int v.
    """
    assert validate_bounded_value(name="x", value=cast(Any, v), min_value=v - a, max_value=v + b) == v


@pytest.mark.parametrize(argnames="good_int", argvalues=[0, 1, -5, 123456789, np.int64(3), np.int32(-2)])
def test_validate_int_value_accepts_int_like(good_int: ScalarInt) -> None:
    """
    Test that validate_int_value accepts valid integer-like inputs and returns Python int.
    """
    output: ScalarInt = validate_int_value(name="n", value=good_int, min_value=-10**9, max_value=10**9)
    assert isinstance(output, int)
    assert output == int(good_int)


@pytest.mark.parametrize(argnames="bad_int", argvalues=[True, np.bool_(False), 3.0, np.float64(4.0), "7", 1.5])
def test_validate_int_value_rejects_non_int_types(bad_int: Any) -> None:
    """
    Test that validate_int_value raises TypeError for non-integer inputs.
    """
    with pytest.raises(expected_exception=TypeError):
        validate_int_value(name="n", value=bad_int)


def test_validate_int_value_inclusive_bounds_and_errors() -> None:
    """
    Test that validate_int_value accepts values equal to the bounds and raises for out-of-bounds.
    """
    assert validate_int_value(name="n", value=5, min_value=5, max_value=10) == 5
    assert validate_int_value(name="n", value=10, min_value=5, max_value=10) == 10
    with pytest.raises(expected_exception=ValueError):
        validate_int_value(name="n", value=4, min_value=5, max_value=10)
    with pytest.raises(expected_exception=ValueError):
        validate_int_value(name="n", value=11, min_value=5, max_value=10)


def test_validate_int_value_inconsistent_bounds_raises() -> None:
    with pytest.raises(expected_exception=ValueError):
        validate_int_value(name="n", value=1, min_value=5, max_value=2)


@given(
    v=st.integers(min_value=-10**6, max_value=10**6),
    a=st.integers(min_value=0, max_value=10**6),
    b=st.integers(min_value=0, max_value=10**6),
)
def test_validate_int_value_property_inclusive_range(v: ScalarInt, a: ScalarInt, b: ScalarInt) -> None:
    """
    Property: for integer v within inclusive [min,max], function returns Python int v.
    """
    output: ScalarInt = validate_int_value(name="n", value=v, min_value=v - a, max_value=v + b)
    assert isinstance(output, int)
    assert output == v
