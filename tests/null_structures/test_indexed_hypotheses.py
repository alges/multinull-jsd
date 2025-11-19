"""
Unit tests for the IndexedHypotheses class.
"""
from multinull_jsd.null_structures import IndexedHypotheses, NullHypothesis
from tests.conftest import TestCDFBackend

from typing import TypeAlias
import numpy.typing as npt
import numpy as np
import pytest

IntDType: TypeAlias = np.int64
FloatDType: TypeAlias = np.float64

IntArray: TypeAlias = npt.NDArray[IntDType]
FloatArray: TypeAlias = npt.NDArray[FloatDType]


def test_indexed_hypotheses_init_rejects_non_backend(k_default: int) -> None:
    """
    The constructor must reject any cdf_backend that is not an instance of CDFBackend.
    """
    with pytest.raises(expected_exception=TypeError):
        IndexedHypotheses(cdf_backend=object(), prob_dim=k_default)  # type: ignore[arg-type]


def test_indexed_hypotheses_init_rejects_non_integer_or_bool_prob_dim(fake_backend: TestCDFBackend) -> None:
    """
    prob_dim must be an integer (bool/float rejected).
    """
    with pytest.raises(expected_exception=TypeError):
        IndexedHypotheses(cdf_backend=fake_backend, prob_dim=True)
    with pytest.raises(expected_exception=TypeError):
        IndexedHypotheses(cdf_backend=fake_backend, prob_dim=3.0)  # type: ignore[arg-type]


def test_indexed_hypotheses_init_rejects_non_positive_prob_dim(fake_backend: TestCDFBackend) -> None:
    """
    prob_dim must be >= 1.
    """
    with pytest.raises(expected_exception=ValueError):
        IndexedHypotheses(cdf_backend=fake_backend, prob_dim=0)
    with pytest.raises(expected_exception=ValueError):
        IndexedHypotheses(cdf_backend=fake_backend, prob_dim=-5)


@pytest.mark.xfail(reason="IndexedHypotheses.add_null not implemented yet.")
def test_add_null_validates_probability_vector_shape(fake_backend: TestCDFBackend, k_default: int) -> None:
    """
    add_null must validate the probability vector: 1-D of length k, non-negative, sums to one.
    """
    ih: IndexedHypotheses = IndexedHypotheses(cdf_backend=fake_backend, prob_dim=k_default)
    # Wrong shape (2-D)
    with pytest.raises(expected_exception=ValueError):
        ih.add_null(
            prob_vector=np.array(object=[[0.5, 0.5, 0.0], [0.8, 0.1, 0.1]], dtype=np.float64), target_alpha=0.05
        )
    # Wrong length (k+1)
    with pytest.raises(expected_exception=ValueError):
        ih.add_null(prob_vector=np.array(object=[0.5, 0.3, 0.1, 0.1], dtype=np.float64), target_alpha=0.05)
    # Negative entry
    with pytest.raises(expected_exception=ValueError):
        ih.add_null(prob_vector=np.array(object=[0.6, -0.1, 0.5], dtype=np.float64), target_alpha=0.05)
    # Sum not 1
    with pytest.raises(expected_exception=ValueError):
        ih.add_null(prob_vector=np.array(object=[0.6, 0.3, 0.3], dtype=np.float64), target_alpha=0.05)


@pytest.mark.xfail(reason="IndexedHypotheses.add_null not implemented yet.")
def test_add_null_validates_alpha_bounds(fake_backend: TestCDFBackend, k_default: int) -> None:
    """
    add_null must enforce 0 ≤ target_alpha ≤ 1 and type must be real (not bool).
    """
    ih: IndexedHypotheses = IndexedHypotheses(cdf_backend=fake_backend, prob_dim=k_default)
    p: FloatArray = np.array(object=[0.5, 0.3, 0.2], dtype=np.float64)
    with pytest.raises(expected_exception=TypeError):
        ih.add_null(prob_vector=p, target_alpha=True)
    with pytest.raises(expected_exception=ValueError):
        ih.add_null(prob_vector=p, target_alpha=-1e-9)
    with pytest.raises(expected_exception=ValueError):
        ih.add_null(prob_vector=p, target_alpha=1.0000001)


@pytest.mark.xfail(reason="IndexedHypotheses.add_null not implemented yet.")
def test_add_null_returns_consecutive_one_based_indices(fake_backend: TestCDFBackend, k_default: int) -> None:
    """
    add_null should assign indices 1,2,3,... in insertion order.
    """
    ih: IndexedHypotheses = IndexedHypotheses(cdf_backend=fake_backend, prob_dim=k_default)
    assert len(ih) == 0
    p1: FloatArray = np.array(object=[0.5, 0.3, 0.2], dtype=np.float64)
    p2: FloatArray = np.array(object=[0.4, 0.4, 0.2], dtype=np.float64)
    idx1: int = int(ih.add_null(prob_vector=p1, target_alpha=0.05))
    idx2: int = int(ih.add_null(prob_vector=p2, target_alpha=0.01))
    assert idx1 == 1 and idx2 == 2
    assert len(ih) == 2


@pytest.mark.xfail(reason="IndexedHypotheses.__getitem__ not implemented yet.")
def test_getitem_validates_indices_and_slices(fake_backend: TestCDFBackend, k_default: int) -> None:
    """
    __getitem__ must validate 1-based indices and proper slices.
    """
    ih: IndexedHypotheses = IndexedHypotheses(cdf_backend=fake_backend, prob_dim=k_default)
    p: FloatArray = np.array(object=[0.5, 0.3, 0.2], dtype=np.float64)
    ih.add_null(prob_vector=p, target_alpha=0.05)

    # Invalid index 0
    with pytest.raises(expected_exception=ValueError):
        _ = ih[0]

    # Invalid negative index
    with pytest.raises(expected_exception=ValueError):
        _ = ih[-1]

    # Invalid step in slice (0)
    with pytest.raises(expected_exception=ValueError):
        _ = ih[slice(1, 1, 0)]


@pytest.mark.xfail(reason="IndexedHypotheses.__getitem__ not implemented yet.")
def test_getitem_returns_objects_and_slices(fake_backend: TestCDFBackend, k_default: int) -> None:
    """
    __getitem__ should return a NullHypothesis for int indices, and a list[NullHypothesis] for slices/iterables.
    """
    # noinspection DuplicatedCode
    ih: IndexedHypotheses = IndexedHypotheses(cdf_backend=fake_backend, prob_dim=k_default)
    p1: FloatArray = np.array(object=[0.5, 0.3, 0.2], dtype=np.float64)
    p2: FloatArray = np.array(object=[0.4, 0.4, 0.2], dtype=np.float64)
    ih.add_null(prob_vector=p1, target_alpha=0.05)
    ih.add_null(prob_vector=p2, target_alpha=0.01)

    one: NullHypothesis = ih[1]
    many: list[NullHypothesis] = ih[1:3]
    assert isinstance(one, NullHypothesis)
    assert isinstance(many, list) and all(isinstance(x, NullHypothesis) for x in many)


@pytest.mark.xfail(reason="IndexedHypotheses.__delitem__ not implemented yet.")
def test_delitem_compacts_indices(fake_backend: TestCDFBackend, k_default: int) -> None:
    """
    __delitem__ should remove items and shift subsequent indices left to keep continuity (1,2,3,...).
    """
    ih = IndexedHypotheses(cdf_backend=fake_backend, prob_dim=k_default)
    p1 = np.array(object=[0.5, 0.3, 0.2], dtype=np.float64)
    p2 = np.array(object=[0.4, 0.4, 0.2], dtype=np.float64)
    p3 = np.array(object=[0.2, 0.3, 0.5], dtype=np.float64)
    ih.add_null(prob_vector=p1, target_alpha=0.05)  # -> 1
    ih.add_null(prob_vector=p2, target_alpha=0.05)  # -> 2
    ih.add_null(prob_vector=p3, target_alpha=0.05)  # -> 3
    assert isinstance(ih[3], NullHypothesis)
    assert len(ih) == 3

    del ih[2]
    assert len(ih) == 2
    # Now former index 3 should be at 2
    assert isinstance(ih[2], NullHypothesis)
    with pytest.raises(expected_exception=ValueError):
        _ = ih[3]


@pytest.mark.xfail(reason="IndexedHypotheses.__contains__ not implemented yet.")
def test_contains_accepts_null_hypothesis_and_raw_vector(fake_backend: TestCDFBackend, k_default: int) -> None:
    """
    __contains__ should work with either a NullHypothesis instance or a raw probability vector.
    """
    ih: IndexedHypotheses = IndexedHypotheses(cdf_backend=fake_backend, prob_dim=k_default)
    p: FloatArray = np.array(object=[0.5, 0.3, 0.2], dtype=np.float64)
    idx: int = int(ih.add_null(prob_vector=p, target_alpha=0.05))
    nh: NullHypothesis = ih[idx]
    assert nh in ih
    assert p in ih
    assert np.array(object=[0.4, 0.4, 0.2], dtype=np.float64) not in ih


@pytest.mark.xfail(reason="IndexedHypotheses.__iter__/__len__ not implemented yet.")
def test_iter_and_len(fake_backend: TestCDFBackend, k_default: int) -> None:
    """
    Iteration should yield the stored nulls in index order; len should match their count.
    """
    # noinspection DuplicatedCode
    ih: IndexedHypotheses = IndexedHypotheses(cdf_backend=fake_backend, prob_dim=k_default)
    p1: FloatArray = np.array(object=[0.5, 0.3, 0.2], dtype=np.float64)
    p2: FloatArray = np.array(object=[0.4, 0.4, 0.2], dtype=np.float64)
    ih.add_null(prob_vector=p1, target_alpha=0.05)
    ih.add_null(prob_vector=p2, target_alpha=0.05)
    xs: list[NullHypothesis] = list(iter(ih))
    assert len(xs) == len(ih) == 2
    assert all(isinstance(x, NullHypothesis) for x in xs)


@pytest.mark.xfail(reason="IndexedHypotheses.__repr__ not implemented yet.")
def test_repr_contains_size_and_prob_dim(fake_backend: TestCDFBackend, k_default: int) -> None:
    """
    __repr__ should contain the class name, number of stored nulls, and the probability dimension.
    """
    ih: IndexedHypotheses = IndexedHypotheses(cdf_backend=fake_backend, prob_dim=k_default)
    p: FloatArray = np.array(object=[0.5, 0.3, 0.2], dtype=np.float64)
    ih.add_null(prob_vector=p, target_alpha=0.05)
    repr_str: str = repr(ih)
    assert "IndexedHypotheses" in repr_str
    assert str(k_default) in repr_str and "1" in repr_str
