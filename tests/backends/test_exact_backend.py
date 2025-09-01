from multinull_jsd.cdf_backends import ExactCDFBackend
from tests.conftest import CDFCallable
from typing import Callable, TypeAlias

import numpy.typing as npt
import numpy as np

import pytest

IntDType: TypeAlias = np.int64
FloatDType: TypeAlias = np.float64

IntArray: TypeAlias = npt.NDArray[IntDType]
FloatArray: TypeAlias = npt.NDArray[FloatDType]


@pytest.fixture(scope="session")
def make_backend() -> Callable[[int], ExactCDFBackend]:
    """
    Factory fixture that builds an ExactCDFBackend.

    Notes
    -----
    If the class is not implemented yet and raises NotImplementedError on construction, we mark the contract tests as
    XFAIL. Once implemented, those tests will run.
    """
    def _factory(evidence_size: int) -> ExactCDFBackend:
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
        ExactCDFBackend(evidence_size=True)   # type: ignore[arg-type]
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


@pytest.mark.xfail(reason="Per-probability-vector caching not implemented yet.")
def test_exact_backend_caches_cdf_per_probability_vector(prob_vec3: FloatArray) -> None:
    """
    Repeated calls to get_cdf with the same probability vector should return the same cached callable (cache keyed by
    probability vector values).
    """
    backend: ExactCDFBackend = ExactCDFBackend(evidence_size=8)
    cdf1: CDFCallable = backend.get_cdf(prob_vector=prob_vec3)
    cdf2: CDFCallable = backend.get_cdf(prob_vector=prob_vec3.copy())
    assert cdf1 is cdf2


@pytest.mark.xfail(reason="__repr__ not implemented yet for ExactCDFBackend.")
def test_exact_backend_repr_contains_class_and_evidence_size() -> None:
    """
    __repr__ should include the class name and the evidence_size value.
    """
    backend: ExactCDFBackend = ExactCDFBackend(evidence_size=12)
    repr_str: str = repr(backend)
    assert "ExactCDFBackend" in repr_str and "12" in repr_str


# Pull in the shared backend contract tests (vectorisation, clipping, monotonicity, etc.)
from tests.backends._contract import *  # noqa: F401
