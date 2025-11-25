from multinull_jsd.cdf_backends import CDFBackend
from multinull_jsd.types import FloatDType, FloatArray, CDFCallable

import numpy as np

import pytest


def test_build_cdf_from_samples_uniform_weights() -> None:
    """
    _build_cdf_from_samples with uniform weights should reproduce the standard ECDF.
    """
    distances: FloatArray = np.array(object=[0.0, 0.5, 1.0], dtype=FloatDType)
    cdf: CDFCallable = CDFBackend._build_cdf_from_samples(distances=distances, weights=None)

    # Scalar behaviour
    assert isinstance(cdf(tau=0.25), float)

    # Below minimum -> 0
    assert cdf(tau=-0.1) == 0.0
    # Between 0 and 0.5 -> only the first sample included
    assert np.isclose(a=cdf(tau=0.0), b=1.0 / 3.0)
    assert np.isclose(a=cdf(tau=0.25), b=1.0 / 3.0)
    # Between 0.5 and 1.0 -> first two samples included
    assert np.isclose(a=cdf(tau=0.5), b=2.0 / 3.0)
    assert np.isclose(a=cdf(tau=0.75), b=2.0 / 3.0)
    # At/above max -> 1
    assert np.isclose(a=cdf(tau=1.0), b=1.0)
    assert np.isclose(a=cdf(tau=1.5), b=1.0)

    # Vectorized monotonicity check
    grid: FloatArray = np.linspace(start=0.0, stop=1.0, num=101, dtype=FloatDType)
    vals: FloatArray = cdf(tau=grid)
    diffs: FloatArray = np.diff(a=vals.astype(dtype=FloatDType))
    # Allow tiny numerical jitter
    assert np.all(a=diffs >= -1e-12)


def test_build_cdf_from_samples_non_uniform_weights() -> None:
    """
    _build_cdf_from_samples with explicit weights must normalize them and use them in the ECDF.
    """
    distances: FloatArray = np.array([0.0, 1.0], dtype=FloatDType)
    weights: FloatArray = np.array([1.0, 3.0], dtype=FloatDType)
    # Normalized weights: [0.25, 0.75]
    cdf: CDFCallable = CDFBackend._build_cdf_from_samples(distances=distances, weights=weights)

    # Below min -> 0
    assert cdf(tau=-0.1) == 0.0
    # Between 0 and 1 -> only the first sample counted
    assert np.isclose(a=cdf(tau=0.0), b=0.25)
    assert np.isclose(a=cdf(tau=0.5), b=0.25)
    # At/above 1 -> full mass
    assert np.isclose(a=cdf(tau=1.0), b=1.0)
    assert np.isclose(a=cdf(tau=2.0), b=1.0)


def test_build_cdf_from_samples_rejects_invalid_inputs() -> None:
    """
    _build_cdf_from_samples must reject malformed distances / weights.
    """
    # distances must be 1-D and in [0,1]
    with pytest.raises(ValueError):
        CDFBackend._build_cdf_from_samples(distances=np.array(object=[[0.0, 0.5], [0.2, 0.8]], dtype=FloatDType))
    with pytest.raises(ValueError):
        CDFBackend._build_cdf_from_samples(distances=np.array(object=[-0.1, 0.2], dtype=FloatDType))
    with pytest.raises(ValueError):
        CDFBackend._build_cdf_from_samples(distances=np.array(object=[0.0, 1.2], dtype=FloatDType))
    with pytest.raises(ValueError):
        CDFBackend._build_cdf_from_samples(distances=np.array(object=[], dtype=FloatDType))

    # weights must match shape, be non-negative, and sum to positive finite value
    distances: FloatArray = np.array(object=[0.0, 0.5], dtype=FloatDType)

    with pytest.raises(ValueError):
        CDFBackend._build_cdf_from_samples(distances=distances, weights=np.array([1.0], dtype=FloatDType))
    with pytest.raises(ValueError):
        CDFBackend._build_cdf_from_samples(distances=distances, weights=np.array([1.0, -0.5], dtype=FloatDType))
    with pytest.raises(ValueError):
        CDFBackend._build_cdf_from_samples(distances=distances, weights=np.array([0.0, 0.0], dtype=FloatDType))
