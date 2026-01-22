"""
Tests for the probability and non-negativity validators in mn_squared._validators.
"""
# noinspection PyProtectedMember
from mn_squared._validators import (
    validate_probability_vector, validate_probability_batch, validate_non_negative_batch, FLOAT_TOL
)
from tests.conftest import p_vector, p_batch
from hypothesis import given
from typing import TypeAlias

import numpy.typing as npt
import numpy as np

import pytest

IntDType: TypeAlias = np.int64
FloatDType: TypeAlias = np.float64

IntArray: TypeAlias = npt.NDArray[IntDType]
FloatArray: TypeAlias = npt.NDArray[FloatDType]


def test_validate_probability_vector_ok_and_dtype() -> None:
    """
    Test that validate_probability_vector accepts a valid probability vector and returns float64 dtype.
    """
    p: FloatArray = np.array(object=[0.5, 0.25, 0.25], dtype=np.float32)  # non-float64 on purpose
    output: FloatArray = validate_probability_vector(name="p", value=p, n_categories=3)
    assert output.shape == (3,)
    assert output.dtype == np.float64
    assert np.allclose(a=output.sum(), b=1.0, atol=FLOAT_TOL, rtol=0.0)


def test_validate_probability_vector_accepts_integer_one_hot_and_casts_to_float64() -> None:
    """
    Test that an integer one-hot vector is accepted and cast to float64.
    """
    p_int: IntArray = np.array(object=[1, 0, 0], dtype=np.int64)
    output: FloatArray = validate_probability_vector(name="p", value=p_int, n_categories=3)
    assert output.dtype == np.float64
    assert np.array_equal(a1=output, a2=np.array(object=[1.0, 0.0, 0.0], dtype=np.float64))


def test_validate_probability_vector_rejects_wrong_length() -> None:
    """
    Test that validate_probability_vector rejects vectors with wrong number of categories.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_probability_vector(name="p", value=np.array(object=[0.6, 0.4]), n_categories=3)


def test_validate_probability_vector_rejects_negative_entries() -> None:
    """
    Test that validate_probability_vector rejects vectors with negative entries.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_probability_vector(name="p", value=np.array(object=[0.5, -0.1, 0.6]), n_categories=3)


def test_validate_probability_vector_rejects_bad_sum_outside_tolerance() -> None:
    """
    Test that validate_probability_vector rejects vectors whose sum is outside FLOAT_TOL.
    """
    eps: float = FLOAT_TOL * 1.1
    p_bad: FloatArray = np.array(object=[0.5, 0.3, 0.2 - eps], dtype=np.float64)
    with pytest.raises(expected_exception=ValueError):
        validate_probability_vector(name="p", value=p_bad, n_categories=3)


def test_validate_probability_vector_allows_sum_within_tolerance() -> None:
    """
    Test that validate_probability_vector accepts vectors whose sum is within FLOAT_TOL.
    """
    eps: float = FLOAT_TOL * 0.5
    p_ok: FloatArray = np.array(object=[0.5, 0.3, 0.2 - eps], dtype=np.float64)
    output: FloatArray = validate_probability_vector(name="p", value=p_ok, n_categories=3)
    assert output.shape == (3,)
    assert np.allclose(a=output.sum(), b=1.0, atol=FLOAT_TOL, rtol=0.0)


def test_validate_probability_vector_enforces_1d() -> None:
    """
    Test that validate_probability_vector rejects 2-D arrays (enforces 1-D). It may be of shape (k, ) or (1,k).
    """
    p_2d: FloatArray = np.array(object=[[0.2, 0.5, 0.3], [0.5, 0.3, 0.2]], dtype=np.float64)
    with pytest.raises(expected_exception=ValueError):
        validate_probability_vector(name="p", value=p_2d, n_categories=3)


def test_validate_probability_batch_accepts_1d_and_returns_row_shape_and_float64() -> None:
    """
    Test that validate_probability_batch accepts 1-D vector, returns shape (1,k) and float64 dtype.
    """
    p: FloatArray = np.array(object=[0.2, 0.3, 0.5], dtype=np.float64)
    output: FloatArray = validate_probability_batch(name="pb", value=p, n_categories=3)
    assert output.shape == (1, 3)
    assert output.dtype == np.float64
    assert np.allclose(a=output.sum(axis=1), b=np.array(object=[1.0]), atol=FLOAT_TOL, rtol=0.0)


def test_validate_probability_batch_accepts_2d_and_preserves_m_k() -> None:
    """
    Test that validate_probability_batch accepts 2-D arrays and preserves (m,k) shape.
    """
    m: FloatArray = np.array(object=[[0.5, 0.25, 0.25], [0.1, 0.2, 0.7]], dtype=np.float64)
    output: FloatArray = validate_probability_batch(name="pb", value=m, n_categories=3)
    assert output.shape == (2, 3)
    assert output.dtype == np.float64
    assert np.allclose(a=output.sum(axis=1), b=np.array(object=[1.0, 1.0], dtype=np.float64), atol=FLOAT_TOL, rtol=0.0)


def test_validate_probability_batch_rejects_when_rows_do_not_sum_to_one() -> None:
    """
    Test that validate_probability_batch rejects rows that do not sum to one within tolerance.
    """
    bad: FloatArray = np.array(object=[[0.6, 0.4, 0.1]], dtype=np.float64)  # sums to 1.1
    with pytest.raises(expected_exception=ValueError):
        validate_probability_batch(name="pb", value=bad, n_categories=3)


def test_validate_probability_batch_enforces_n_categories_when_given() -> None:
    """
    Test that validate_probability_batch enforces the number of categories when provided.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_probability_batch(name="pb", value=np.array(object=[0.5, 0.5, 0.0, 0.0]), n_categories=3)


def test_validate_probability_batch_without_n_categories_allows_any_k() -> None:
    """
    Test that validate_probability_batch allows any number of categories when n_categories=None.
    """
    p4: FloatArray = np.array(object=[0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    output: FloatArray = validate_probability_batch(name="pb", value=p4, n_categories=None)
    assert output.shape == (1, 4)
    assert np.allclose(a=output.sum(), b=1.0, atol=FLOAT_TOL, rtol=0.0)


def test_validate_probability_batch_rejects_negative_entries() -> None:
    """
    Test that validate_probability_batch rejects negative entries.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_probability_batch(name="pb", value=np.array(object=[[0.5, -1e-12, 0.5]]), n_categories=3)


def test_validate_non_negative_batch_1d_becomes_row_and_enforces_width() -> None:
    """
    Test that validate_non_negative_batch expands 1-D to (1,k) and enforces width when provided.
    """
    arr: FloatArray = validate_non_negative_batch(name="x", value=[1, 0, 2], n_categories=3)
    assert arr.shape == (1, 3)
    with pytest.raises(expected_exception=ValueError):
        validate_non_negative_batch(name="x", value=[1, 2, 3, 4], n_categories=3)


def test_validate_non_negative_batch_rejects_ndim_not_1_or_2() -> None:
    """
    Test that validate_non_negative_batch rejects arrays with ndim other than 1 or 2.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_non_negative_batch(name="x", value=np.zeros(shape=(2, 3, 4)), n_categories=None)


@given(vector=p_vector(k=5))
def test_validate_probability_vector_property_accepts_strategy_vectors(vector: FloatArray) -> None:
    """
    Property: validate_probability_vector accepts any well-formed probability vector from the shared strategy.
    """
    output: FloatArray = validate_probability_vector(name="p", value=vector, n_categories=5)
    assert output.shape == (5,)
    assert output.dtype == np.float64
    assert np.allclose(a=output.sum(), b=1.0, atol=FLOAT_TOL, rtol=0.0)
    assert np.all(a=output >= 0.0)


@given(batch=p_batch(m=4, k=3))
def test_validate_probability_batch_property_accepts_strategy_batches(batch: FloatArray) -> None:
    """
    Property: validate_probability_batch accepts any well-formed (m,k) batch from the shared strategy.
    """
    output: FloatArray = validate_probability_batch(name="pb", value=batch, n_categories=3)
    assert output.shape == (batch.shape[0], 3)
    assert output.dtype == np.float64
    prob_sums: FloatArray = output.sum(axis=1)
    assert np.allclose(a=prob_sums, b=np.ones_like(prob_sums), atol=FLOAT_TOL, rtol=0.0)
    assert np.all(a=output >= 0.0)
