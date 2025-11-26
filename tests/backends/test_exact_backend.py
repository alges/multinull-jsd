from multinull_jsd.cdf_backends import ExactCDFBackend
from multinull_jsd.types import CDFCallable

from typing import Callable, TypeAlias

import numpy.typing as npt
import numpy as np

import pytest
import math

IntDType: TypeAlias = np.int64
FloatDType: TypeAlias = np.float64

IntArray: TypeAlias = npt.NDArray[IntDType]
FloatArray: TypeAlias = npt.NDArray[FloatDType]


def _slow_histograms(n: int, k: int) -> set[tuple[int, ...]]:
    """
    Simple reference implementation: all non-negative integer k-tuples summing to n.

    Parameters
    ----------
    n
        The sum to reach.
    k
        The number of elements in the tuples.

    Returns
    -------
    The set of all k-tuples summing to n.
    """
    result: list[tuple[int, ...]] = []

    def rec(remaining: int, idx: int, current: list[int]) -> None:
        """
        Recursively generates all possible ways to split a given integer `remaining` into `k` parts and stores the
        results as tuples in the `result` list.

        This function operates by distributing the remaining value among the `k` parts, ensuring all possible
        distributions are considered. The resulting distributions are stored in the `result` list, which must be
        predefined in the enclosing scope. The recursion continues until all parts are filled or the base case is
        reached.

        Parameters
        ----------
        remaining
            The remaining integer value to be distributed among the parts.
        idx
            The current index in the cycle of parts being filled.
        current
            The list representing the distribution of values among the parts being constructed.
        """
        if idx == k - 1:
            current.append(remaining)
            result.append(tuple(current))
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            rec(remaining=remaining - v, idx=idx + 1, current=current)
            current.pop()

    rec(remaining=n, idx=0, current=[])
    return set(result)


@pytest.fixture(scope="session")
def make_backend() -> Callable[[int], ExactCDFBackend | None]:
    """
    Factory fixture that builds an ExactCDFBackend.

    Notes
    -----
    If the class is not implemented yet and raises NotImplementedError on construction, we mark the contract tests as
    XFAIL. Once implemented, those tests will run.
    """
    def _factory(evidence_size: int) -> ExactCDFBackend | None:
        try:
            return ExactCDFBackend(evidence_size=evidence_size)
        except NotImplementedError:
            pytest.xfail(reason="ExactCDFBackend is not implemented yet.")
    return _factory


def test_exact_backend_rejects_non_integer_or_bool_evidence_size() -> None:
    """
    Test that the constructor rejects non-integer evidence_size values (including bool).
    """
    with pytest.raises(expected_exception=TypeError):
        ExactCDFBackend(evidence_size=True)
    with pytest.raises(expected_exception=TypeError):
        ExactCDFBackend(evidence_size=3.0)    # type: ignore[arg-type]


def test_exact_backend_rejects_non_positive_evidence_size() -> None:
    """
    Test that the constructor rejects non-positive evidence_size values.
    """
    with pytest.raises(expected_exception=ValueError):
        ExactCDFBackend(evidence_size=0)
    with pytest.raises(expected_exception=ValueError):
        ExactCDFBackend(evidence_size=-5)


def test_exact_backend_caches_cdf_per_probability_vector(prob_vec3_default: FloatArray) -> None:
    """
    Repeated calls to get_cdf with the same probability vector should return the same cached callable (cache keyed by
    probability vector values).
    """
    backend: ExactCDFBackend = ExactCDFBackend(evidence_size=8)
    cdf1: CDFCallable = backend.get_cdf(prob_vector=prob_vec3_default)
    cdf2: CDFCallable = backend.get_cdf(prob_vector=prob_vec3_default.copy())
    assert cdf1 is cdf2


def test_exact_backend_repr_contains_class_and_evidence_size() -> None:
    """
    __repr__ should include the class name and the evidence_size value.
    """
    backend: ExactCDFBackend = ExactCDFBackend(evidence_size=12)
    repr_str: str = repr(backend)
    assert "ExactCDFBackend" in repr_str and "12" in repr_str


@pytest.mark.parametrize(argnames="n, k", argvalues=[(3, 2), (4, 2), (4, 3)])
def test_enumerate_histograms_matches_reference(n: int, k: int) -> None:
    """
    _enumerate_histograms must enumerate all integer histograms of length k summing to n, with the right cardinality
    (n+k-1 choose k-1).
    """
    backend: ExactCDFBackend = ExactCDFBackend(evidence_size=n)
    hist: IntArray = backend._enumerate_histograms(k=k)

    # Shape and non-negativity
    assert hist.ndim == 2
    assert hist.shape[1] == k
    assert np.all(a=hist >= 0)
    assert np.all(a=hist.sum(axis=1) == n)

    # Cardinality
    assert hist.shape[0] == math.comb(n + k - 1, k - 1)

    # Exact set equality versus slow reference
    got: set[tuple[int, ...]] = {tuple(row.tolist()) for row in hist}
    expected: set[tuple[int, ...]] = _slow_histograms(n=n, k=k)
    assert got == expected


def test_enumerate_histograms_k_equals_one() -> None:
    """
    For k=1, the only histogram is [n].
    """
    n: int = 5
    backend: ExactCDFBackend = ExactCDFBackend(evidence_size=n)
    hist: IntArray = backend._enumerate_histograms(k=1)

    assert hist.shape == (1, 1)
    assert int(hist[0, 0]) == n


def test_exact_backend_histogram_enumeration_small_case() -> None:
    """
    For n=3, k=2, the exact backend's histogram enumeration should match the stars-and-bars combinations.
    """
    backend: ExactCDFBackend = ExactCDFBackend(evidence_size=3)
    hist: IntArray = backend._enumerate_histograms(k=2)

    expected: IntArray = np.array(object=[[0, 3], [1, 2], [2, 1], [3, 0]], dtype=IntDType)

    assert hist.shape == expected.shape
    # Enumeration order is deterministic in this implementation, so we can compare directly.
    assert np.array_equal(a1=hist, a2=expected)


# Pull in the shared backend contract tests (vectorization, clipping, monotonicity, etc.)
from tests.backends._contract import *  # noqa
