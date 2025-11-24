"""
Unit tests for multinull_jsd._jsd_distance.jsd.
"""
# noinspection PyProtectedMember
from multinull_jsd._jsd_distance import jsd

from tests.conftest import p_vector
from hypothesis import given
from typing import TypeAlias

import numpy.typing as npt
import numpy as np

FloatDType: TypeAlias = np.float64
FloatArray: TypeAlias = npt.NDArray[FloatDType]


def test_jsd_self_is_zero_and_finite() -> None:
    p: FloatArray = np.full(shape=5, fill_value=0.2)
    d: FloatArray = jsd(p=p, q=p)
    assert np.isfinite(d).all()
    np.testing.assert_allclose(actual=d, desired=0.0, atol=1e-12)


@given(p=p_vector(k=5))
def test_jsd_zero_for_identical_distributions(p: FloatArray) -> None:
    """
    JSD(p, p) must be exactly zero (up to numerical tolerance).
    """
    d: FloatArray = jsd(p=p, q=p)
    assert d.shape == (1,)
    assert np.allclose(a=d, b=0.0, atol=1e-12)


@given(p=p_vector(k=5), q=p_vector(k=5))
def test_jsd_is_symmetric_and_bounded(p: FloatArray, q: FloatArray) -> None:
    """
    JSD(p, q) = JSD(q, p) and lies in [0, 1].
    """
    d1: FloatArray = jsd(p=p, q=q)
    d2: FloatArray = jsd(p=q, q=p)
    assert d1.shape == (1,)
    assert d2.shape == (1,)

    assert np.all(a=d1 >= 0.0) and np.all(a=d1 <= 1.0)
    assert np.all(a=d2 >= 0.0) and np.all(a=d2 <= 1.0)
    assert np.allclose(a=d1, b=d2, atol=1e-12)


def test_jsd_two_point_extreme_case() -> None:
    """
    For p=[1,0] and q=[0,1] the JSD distance must be 1.
    """
    p: FloatArray = np.array(object=[1.0, 0.0], dtype=np.float64)
    q: FloatArray = np.array(object=[0.0, 1.0], dtype=np.float64)

    d: FloatArray = jsd(p=p, q=q)
    assert d.shape == (1,)
    assert np.allclose(a=d[0], b=1.0, atol=1e-12)


def test_jsd_batch_broadcasting_shapes() -> None:
    """
    Check that the function handles batch versus vector and batch versus batch inputs correctly.
    """
    p_batch: FloatArray = np.array(object=[[0.5, 0.5], [0.9, 0.1]], dtype=np.float64)
    q_vec: FloatArray = np.array(object=[0.7, 0.3], dtype=np.float64)

    # (batch, vector) -> length = batch size
    d1: FloatArray = jsd(p=p_batch, q=q_vec)
    assert d1.shape == (2,)

    # (vector, batch) -> length = batch size
    d2: FloatArray = jsd(p=q_vec, q=p_batch)
    assert d2.shape == (2,)

    # (batch, batch) with matching lengths
    d3: FloatArray = jsd(
        p=np.array(object=[[0.5, 0.5], [0.7, 0.3]], dtype=np.float64),
        q=np.array(object=[[0.6, 0.4], [0.4, 0.6]], dtype=np.float64),
    )
    assert d3.shape == (2,)


def test_jsd_raises_on_incompatible_batch_lengths() -> None:
    """
    If both inputs are batches, they must have the same length or one of them length 1.
    """
    p_batch: FloatArray = np.array(object=[[0.5, 0.5], [0.9, 0.1]], dtype=np.float64)
    q_batch: FloatArray = np.array(object=[[0.6, 0.4], [0.4, 0.6], [0.3, 0.7]], dtype=np.float64)

    with np.testing.assert_raises(ValueError):
        _ = jsd(p=p_batch, q=q_batch)
