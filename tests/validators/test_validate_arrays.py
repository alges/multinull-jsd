"""
Unit tests for the validate_finite_array function in the multinull_jsd._validators module.
"""
# noinspection PyProtectedMember
from multinull_jsd._validators import validate_finite_array

from hypothesis.extra import numpy as hnp
from hypothesis import strategies as st
from hypothesis import given
from typing import TypeAlias

import numpy.typing as npt
import numpy as np

import pytest

IntDType: TypeAlias = np.int64
FloatDType: TypeAlias = np.float64

IntArray: TypeAlias = npt.NDArray[IntDType]
FloatArray: TypeAlias = npt.NDArray[FloatDType]


def test_validate_finite_array_accepts_numeric_real_python_lists() -> None:
    """
    Test that validate_finite_array accepts a Python list of real numbers (ints and floats).
    """
    original: list[int | float] = [1, 2, 3.0]
    output: IntArray | FloatArray = validate_finite_array(name="a", value=original)
    assert isinstance(output, np.ndarray)
    assert output.dtype.kind in ("i", "f")
    assert np.array_equal(a1=output, a2=np.asarray(original))


def test_validate_finite_array_accepts_numpy_arrays_int_and_float() -> None:
    """
    Test that validate_finite_array accepts NumPy arrays of int64 and float64 types.
    """
    array_int: IntArray = np.array(object=[1, 2, 3], dtype=np.int64)
    array_float: FloatArray = np.array(object=[1.0, 2.5, 3.0], dtype=np.float64)
    out_int: IntArray = validate_finite_array(name="array_int", value=array_int)
    out_float: FloatArray = validate_finite_array(name="array_float", value=array_float)

    assert isinstance(out_int, np.ndarray) and out_int.dtype == np.int64
    assert isinstance(out_float, np.ndarray) and out_float.dtype == np.float64
    assert np.array_equal(a1=out_int, a2=array_int)
    assert np.array_equal(a1=out_float, a2=array_float)


def test_validate_finite_array_accepts_zero_dimensional_numeric_scalars() -> None:
    """
    Test that validate_finite_array accepts zero-dimensional NumPy arrays (scalars) of int64 and float64 types.
    """
    out_int: IntArray = validate_finite_array(name="y", value=np.int64(7))
    out_float: FloatArray = validate_finite_array(name="x", value=np.float64(2.5))
    assert isinstance(out_int, np.ndarray) and out_int.shape == ()
    assert isinstance(out_float, np.ndarray) and out_float.shape == ()
    assert out_int.dtype == np.int64 and out_float.dtype == np.float64
    assert int(out_int) == 7 and float(out_float) == 2.5


def test_validate_finite_array_rejects_complex_array_and_scalar() -> None:
    """
    Test that validate_finite_array raises TypeError for complex-valued arrays and scalars.
    """
    with pytest.raises(expected_exception=TypeError):
        validate_finite_array(name="z", value=np.array(object=[1+0j, 2+0j]))
    with pytest.raises(expected_exception=TypeError):
        validate_finite_array(name="z", value=np.complex64(1+0j))


def test_validate_finite_array_rejects_boolean_array_and_scalar() -> None:
    """
    Test that validate_finite_array raises TypeError for boolean arrays and scalars.
    """
    with pytest.raises(expected_exception=TypeError):
        validate_finite_array(name="b", value=np.array(object=[True, False], dtype=np.bool_))
    with pytest.raises(expected_exception=TypeError):
        validate_finite_array(name="b", value=np.bool_(True))


def test_validate_finite_array_rejects_non_numeric_dtype() -> None:
    """
    Test that validate_finite_array raises TypeError for arrays with non-numeric dtypes (e.g., string, object).
    """
    with pytest.raises(expected_exception=TypeError):
        validate_finite_array(name="s", value=np.array(object=["a", "b"]))
    with pytest.raises(expected_exception=TypeError):
        validate_finite_array(name="o", value=np.array(object=[object(), object()], dtype=object))


def test_validate_finite_array_rejects_nan_and_inf() -> None:
    """
    Test that validate_finite_array raises ValueError for arrays containing NaN or infinite values.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_finite_array(name="x", value=np.array(object=[1.0, np.nan]))
    with pytest.raises(expected_exception=ValueError):
        validate_finite_array(name="y", value=np.array(object=[1.0, np.inf]))


@given(
    array=hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=5),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
)
def test_validate_finite_array_property_accepts_real_finite_float_arrays(array: FloatArray) -> None:
    """
    Property: for any finite real-valued float64 array, function returns identical array of type float64.
    """
    output: FloatArray = validate_finite_array(name="array", value=array)
    assert isinstance(output, np.ndarray)
    assert output.dtype == np.float64
    assert output.shape == array.shape
    assert np.array_equal(a1=output, a2=np.asarray(array))


@given(
    array=hnp.arrays(
        dtype=np.int64,
        shape=hnp.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=5),
        elements=st.integers(min_value=-10**6, max_value=10**6),
    )
)
def test_validate_finite_array_property_accepts_int_arrays(array: IntArray) -> None:
    """
    Property: for any int64 array, function returns identical array of type int64.
    """
    output: IntArray = validate_finite_array(name="array", value=array)
    assert isinstance(output, np.ndarray)
    assert output.dtype == np.int64
    assert output.shape == array.shape
    assert np.array_equal(a1=output, a2=np.asarray(array))
