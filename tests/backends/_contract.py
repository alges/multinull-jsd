from multinull_jsd.cdf_backends import CDFBackend
from tests.conftest import p_vector, n_default, prob_vec3_default
from hypothesis import given
from typing import Callable, TypeAlias

import numpy.typing as npt
import numpy as np

import pytest

IntDType: TypeAlias = np.int64
FloatDType: TypeAlias = np.float64

IntArray: TypeAlias = npt.NDArray[IntDType]
FloatArray: TypeAlias = npt.NDArray[FloatDType]

# Public type for the backend factory fixture each implementation must provide:
#     @pytest.fixture
#     def make_backend():
#         return lambda evidence_size: ExactCDFBackend(evidence_size=evidence_size)
BackendFactory: TypeAlias = Callable[[int], CDFBackend]


def test_backend_contract_vectorised_monotone_and_bounded(
    make_backend: BackendFactory, n_default: int, prob_vec3_default: FloatArray
) -> None:
    """
    Contract: get_cdf returns a callable cdf such that:
      - it is vectorised (array-in → array-out, same shape),
      - it is clipped to [0,1],
      - it is monotone non-decreasing in tau (allowing tiny numerical jitter).
    """
    backend: CDFBackend = make_backend(n_default)
    cdf = backend.get_cdf(prob_vector=prob_vec3_default)

    # Vectorised behavior & clipping on a small array
    tau_small: FloatArray = np.array(object=[-0.2, 0.0, 0.25, 1.0, 1.3], dtype=np.float64)
    out_small: FloatArray = cdf(tau=tau_small)
    assert isinstance(out_small, np.ndarray)
    assert out_small.dtype == np.float64
    assert out_small.shape == tau_small.shape
    assert np.all(a=out_small >= 0.0) and np.all(a=out_small <= 1.0)

    # Monotonicity on a grid (tolerate tiny negative diffs due to numeric/MC noise)
    assert np.all(
        a=np.diff(cdf(tau=np.linspace(start=-0.5, stop=1.5, num=401, dtype=np.float64)).astype(np.float64)) >= -1e-10
    )


def test_backend_contract_scalar_and_array_return_types(
    make_backend: BackendFactory, n_default: int, prob_vec3_default: FloatArray
) -> None:
    """
    Contract: scalar input → Python float; array input → float64 ndarray with same shape.
    """
    backend: CDFBackend = make_backend(n_default)
    cdf = backend.get_cdf(prob_vector=prob_vec3_default)

    f_output = cdf(tau=0.33)
    assert isinstance(f_output, float)

    arr: FloatArray = np.array(object=[[0.0, 0.5, 1.0], [-1.0, 0.25, 2.0]], dtype=np.float64)
    arr_out: FloatArray = cdf(tau=arr)
    assert isinstance(arr_out, np.ndarray)
    assert np.all(a=arr_out >= 0.0) and np.all(a=arr_out <= 1.0)
    assert arr_out.dtype == np.float64
    assert arr_out.shape == arr.shape


def test_backend_contract_property_and_repr(make_backend: BackendFactory, n_default: int) -> None:
    """
    Contract: backend exposes evidence_size and has a non-empty __repr__ (implementation-defined contents).
    """
    backend: CDFBackend = make_backend(n_default)
    assert isinstance(backend.evidence_size, int)
    assert backend.evidence_size == n_default

    rep: str = repr(backend)
    assert isinstance(rep, str) and len(rep) > 0


def test_backend_contract_get_cdf_rejects_invalid_probability_vector(
    make_backend: BackendFactory, n_default: int
) -> None:
    """
    Contract: get_cdf must validate probability vectors and reject invalid inputs.
    """
    backend: CDFBackend = make_backend(n_default)

    # Wrong shape (2-D instead of 1-D)
    with pytest.raises(expected_exception=ValueError):
        backend.get_cdf(prob_vector=np.array(object=[[0.5, 0.5], [0.2, 0.8]], dtype=np.float64))

    # Negative entry
    with pytest.raises(expected_exception=ValueError):
        backend.get_cdf(prob_vector=np.array(object=[0.6, -0.1, 0.5], dtype=np.float64))

    # Sum not 1 (beyond tolerance)
    with pytest.raises(expected_exception=ValueError):
        backend.get_cdf(prob_vector=np.array(object=[0.6, 0.3, 0.3], dtype=np.float64))


@given(p=p_vector(k=4))
def test_backend_contract_property_accepts_various_prob_vectors(
    make_backend: BackendFactory, n_default: int, p: FloatArray
) -> None:
    """
    Property: for a variety of valid probability vectors, cdf returns values in [0,1] and preserves input shapes.
    """
    backend: CDFBackend = make_backend(n_default)
    cdf = backend.get_cdf(prob_vector=p)

    # Scalar tau
    tau = cdf(tau=0.5)
    assert isinstance(tau, float)
    assert 0.0 <= tau <= 1.0

    # Array tau
    x: FloatArray = np.array(object=[-1.0, 0.0, 0.4, 0.9, 2.0], dtype=np.float64)
    y: FloatArray = cdf(tau=x)
    assert np.all(a=y >= 0.0) and np.all(a=y <= 1.0)
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert y.dtype == np.float64
