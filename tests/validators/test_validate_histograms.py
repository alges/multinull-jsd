"""
Tests for the validate_histogram_batch function.
"""
# noinspection PyProtectedMember
from mn_squared._validators import validate_histogram_batch
from tests.conftest import histogram, histogram_batch
from hypothesis import given
from typing import TypeAlias

import numpy.typing as npt
import numpy as np

import pytest

IntDType: TypeAlias = np.int64

IntArray: TypeAlias = npt.NDArray[IntDType]


def test_validate_histogram_batch_accepts_int_and_integer_like_float() -> None:
    """
    Test that validate_histogram_batch accepts integer arrays and integer-like float arrays.
    """
    out_from_int: IntArray = validate_histogram_batch(name="h", value=[5, 3, 2], n_categories=3, histogram_size=10)
    assert out_from_int.dtype == np.int64
    assert out_from_int.shape == (1, 3)
    assert np.array_equal(a1=out_from_int, a2=np.array(object=[[5, 3, 2]], dtype=np.int64))

    # Floats that are exactly integers are accepted
    out_from_float: IntArray = validate_histogram_batch(
        name="h", value=np.array(object=[[5.0, 3.0, 2.0]], dtype=np.float64), n_categories=3, histogram_size=10
    )
    assert out_from_float.dtype == np.int64
    assert out_from_float.shape == (1, 3)
    assert np.array_equal(a1=out_from_float, a2=np.array(object=[[5, 3, 2]], dtype=np.int64))


def test_validate_histogram_batch_rejects_non_integer_like_floats() -> None:
    """
    Test that validate_histogram_batch rejects float arrays that are not integer-like.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_histogram_batch(name="h", value=[4.9, 3.1, 2.0], n_categories=3, histogram_size=10)


def test_validate_histogram_batch_rejects_wrong_sum_or_width() -> None:
    """
    Test that validate_histogram_batch rejects histograms with wrong total sum or wrong number of categories.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_histogram_batch(name="h", value=[5, 3, 1], n_categories=3, histogram_size=10)  # sums to 9
    with pytest.raises(expected_exception=ValueError):
        validate_histogram_batch(name="h", value=[5, 3, 2, 0], n_categories=3, histogram_size=10)  # wrong width


def test_validate_histogram_batch_rejects_negative_and_boolean() -> None:
    """
    Test that validate_histogram_batch rejects negative counts and boolean arrays.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_histogram_batch(name="h", value=[5, -1, 6], n_categories=3, histogram_size=10)
    with pytest.raises(expected_exception=TypeError):
        validate_histogram_batch(
            name="h", value=np.array(object=[True, False, True]), n_categories=3, histogram_size=10
        )


def test_validate_histogram_batch_rejects_invalid_n_categories_bound() -> None:
    """
    Test that validate_histogram_batch rejects n_categories < 1.
    """
    with pytest.raises(expected_exception=ValueError):
        validate_histogram_batch(name="h", value=[], n_categories=0, histogram_size=10)


def test_validate_histogram_batch_accepts_2d_batch_and_preserves_shape_and_dtype() -> None:
    """
    Test that validate_histogram_batch accepts 2-D arrays and preserves (m,k) with int64 dtype.
    """
    value: IntArray = np.array(object=[[5, 3, 2], [2, 5, 3]], dtype=np.int64)
    out: IntArray = validate_histogram_batch(name="h", value=value, n_categories=3, histogram_size=10)
    assert out.dtype == np.int64
    assert out.shape == (2, 3)
    assert np.array_equal(a1=out, a2=value)


@given(h=histogram(n=20, k=4))
def test_validate_histogram_batch_property_accepts_strategy_histogram(h: IntArray) -> None:
    """
    Property: validate_histogram_batch accepts any well-formed 1-D histogram from the shared strategy.
    """
    hist: IntArray = validate_histogram_batch(name="h", value=h, n_categories=4, histogram_size=20)
    assert hist.dtype == np.int64
    assert hist.shape == (1, 4)
    assert int(hist.sum()) == 20
    assert np.array_equal(a1=hist[0], a2=h.astype(np.int64))


@given(hb=histogram_batch(m=5, n=30, k=3))
def test_validate_histogram_batch_property_accepts_strategy_batch(hb: IntArray) -> None:
    """
    Property: validate_histogram_batch accepts any well-formed (m,k) histogram batch from the shared strategy.
    """
    hist_batch: IntArray = validate_histogram_batch(name="H", value=hb, n_categories=3, histogram_size=30)
    assert hist_batch.dtype == np.int64
    assert hist_batch.shape == (hb.shape[0], 3)
    assert np.all(a=hist_batch.sum(axis=1) == 30)
    assert np.array_equal(a1=hist_batch, a2=hb.astype(np.int64))
