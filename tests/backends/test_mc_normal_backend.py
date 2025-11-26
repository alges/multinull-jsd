from multinull_jsd.cdf_backends import MultinomialMCCDFBackend, NormalMCCDFBackend
from multinull_jsd.types import CDFCallable

from typing import Callable, TypeAlias

import numpy.typing as npt
import numpy as np

import pytest

IntDType: TypeAlias = np.int64
FloatDType: TypeAlias = np.float64

IntArray: TypeAlias = npt.NDArray[IntDType]
FloatArray: TypeAlias = npt.NDArray[FloatDType]


@pytest.fixture(scope="session")
def make_backend() -> Callable[[int], NormalMCCDFBackend | None]:
    """
    Factory fixture that builds a NormalMCCDFBackend with stable MC params. If construction raises NotImplementedError
    (backend not yet implemented), we mark the contract tests as XFAIL. Once implemented, they will run.
    """
    def _factory(evidence_size: int) -> NormalMCCDFBackend | None:
        try:
            return NormalMCCDFBackend(evidence_size=evidence_size, mc_samples=5000, seed=1234)
        except NotImplementedError:
            pytest.xfail(reason="NormalMCCDFBackend is not implemented yet.")
    return _factory


def test_mc_normal_rejects_non_integer_or_bool_evidence_size() -> None:
    """
    evidence_size must be an integer (bool/float rejected).
    """
    with pytest.raises(expected_exception=TypeError):
        NormalMCCDFBackend(evidence_size=True, mc_samples=1000, seed=0)
    with pytest.raises(expected_exception=TypeError):
        NormalMCCDFBackend(evidence_size=3.0, mc_samples=1000, seed=0)  # type: ignore[arg-type]


def test_mc_normal_rejects_non_positive_evidence_size() -> None:
    """
    evidence_size must be >= 1.
    """
    with pytest.raises(expected_exception=ValueError):
        NormalMCCDFBackend(evidence_size=0, mc_samples=1000, seed=0)
    with pytest.raises(expected_exception=ValueError):
        NormalMCCDFBackend(evidence_size=-5, mc_samples=1000, seed=0)


def test_mc_normal_rejects_non_integer_or_bool_mc_samples() -> None:
    """
    mc_samples must be an integer (bool/float rejected).
    """
    with pytest.raises(expected_exception=TypeError):
        NormalMCCDFBackend(evidence_size=10, mc_samples=True, seed=0)
    with pytest.raises(expected_exception=TypeError):
        NormalMCCDFBackend(evidence_size=10, mc_samples=1.5, seed=0)  # type: ignore[arg-type]


def test_mc_normal_rejects_non_positive_mc_samples() -> None:
    """
    mc_samples must be >= 1.
    """
    with pytest.raises(expected_exception=ValueError):
        NormalMCCDFBackend(evidence_size=10, mc_samples=0, seed=0)
    with pytest.raises(expected_exception=ValueError):
        NormalMCCDFBackend(evidence_size=10, mc_samples=-100, seed=0)


def test_mc_normal_rejects_bad_seed_type_or_negative() -> None:
    """
    seed must be an integer >= 0.
    """
    with pytest.raises(expected_exception=TypeError):
        NormalMCCDFBackend(evidence_size=10, mc_samples=1000, seed=1.2)  # type: ignore[arg-type]
    with pytest.raises(expected_exception=TypeError):
        NormalMCCDFBackend(evidence_size=10, mc_samples=1000, seed=False)
    with pytest.raises(expected_exception=ValueError):
        NormalMCCDFBackend(evidence_size=10, mc_samples=1000, seed=-1)


def test_mc_normal_reproducibility_by_seed(prob_vec3_default: FloatArray) -> None:
    """
    With the same seed and mc_samples, the estimated CDF values must be identical
    for the same probability vector and tau grid.
    """
    backend_1: NormalMCCDFBackend = NormalMCCDFBackend(evidence_size=12, mc_samples=5_000, seed=42)
    backend_2: NormalMCCDFBackend = NormalMCCDFBackend(evidence_size=12, mc_samples=5_000, seed=42)
    # noinspection DuplicatedCode
    cdf_1: CDFCallable = backend_1.get_cdf(prob_vector=prob_vec3_default)
    cdf_2: CDFCallable = backend_2.get_cdf(prob_vector=prob_vec3_default)

    tau: FloatArray = np.array(object=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0], dtype=np.float64)
    output_1: FloatArray = cdf_1(tau=tau)
    output_2: FloatArray = cdf_2(tau=tau)
    assert isinstance(output_1, np.ndarray) and isinstance(output_2, np.ndarray)
    assert output_1.dtype == np.float64 and output_2.dtype == np.float64
    assert output_1.shape == tau.shape == output_2.shape
    # Exact equality expected for fixed RNG & ECDF implementation
    assert np.array_equal(a1=output_1, a2=output_2)


def test_mc_normal_repr_contains_params() -> None:
    """
    __repr__ should include the class name and key parameters (n, mc_samples, seed).
    """
    backend: NormalMCCDFBackend = NormalMCCDFBackend(evidence_size=15, mc_samples=2_000, seed=7)
    repr_str: str = repr(backend)
    assert "NormalMCCDFBackend" in repr_str
    assert "15" in repr_str and "2000" in repr_str and "7" in repr_str


def test_mc_normal_degenerate_prob_vector_point_mass_at_zero() -> None:
    """
    For a degenerate probability vector (one-hot), the normal backend should produce a point mass at JSd = 0.
    """
    backend: NormalMCCDFBackend = NormalMCCDFBackend(evidence_size=10, mc_samples=1_000, seed=0)
    p: FloatArray = np.array(object=[1.0, 0.0, 0.0], dtype=FloatDType)

    cdf: CDFCallable = backend.get_cdf(prob_vector=p)
    tau: FloatArray = np.array(object=[-0.1, 0.0, 0.5, 1.0], dtype=FloatDType)
    vals: FloatArray = cdf(tau=tau)

    # Below 0 -> 0; at/above 0 -> 1
    expected: FloatArray = np.array(object=[0.0, 1.0, 1.0, 1.0], dtype=FloatDType)
    assert np.allclose(a=vals, b=expected)

def test_mc_normal_roughly_matches_mc_multinomial() -> None:
    """
    For a moderate (n, k) problem, the normal-based CDF should roughly match the multinomial MC CDF. A loose tolerance
    is used because the CLT approximation is only asymptotically accurate.
    """
    n: int = 50
    p: FloatArray = np.array(object=[0.2, 0.3, 0.5], dtype=FloatDType)
    tau_grid: FloatArray = np.linspace(start=0.0, stop=1.0, num=11, dtype=FloatDType)

    normal_backend: NormalMCCDFBackend = NormalMCCDFBackend(evidence_size=n, mc_samples=10_000, seed=123)
    multinom_backend: MultinomialMCCDFBackend = MultinomialMCCDFBackend(evidence_size=n, mc_samples=20_000, seed=456)

    normal_cdf: CDFCallable = normal_backend.get_cdf(prob_vector=p)
    multinom_cdf: CDFCallable = multinom_backend.get_cdf(prob_vector=p)

    normal_vals: FloatArray = normal_cdf(tau=tau_grid)
    multinom_vals: FloatArray = multinom_cdf(tau=tau_grid)

    assert normal_vals.shape == multinom_vals.shape

    # Very loose bound to accommodate CLT + MC noise; we only care that curves are in the same ballpark.
    diff: FloatArray = np.abs(normal_vals - multinom_vals)
    assert np.all(diff < 3e-2)


# Pull in the shared backend contract tests (vectorization, clipping, monotonicity, basic get_cdf validation)
from tests.backends._contract import *  # noqa
