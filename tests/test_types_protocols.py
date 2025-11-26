"""
Unit tests for type aliases and protocols in multinull_jsd.types.
"""
from multinull_jsd.types import IntDType, FloatDType, FloatArray, ScalarFloat, CDFCallable

import numpy as np


def test_type_aliases() -> None:
    """
    Test that the type aliases for IntDType and FloatDType are correctly defined.
    """
    assert IntDType is np.int64
    assert FloatDType is np.float64


def test_cdf_callable_protocol() -> None:
    """
    Test that a sample function conforms to the CDFCallable protocol.
    """
    class SampleCDF:
        def __call__(self, tau: ScalarFloat | FloatArray) -> ScalarFloat | FloatArray:
            array: FloatArray = np.clip(np.asarray(tau, dtype=FloatDType), 0.0, 1.0)
            if np.isscalar(tau):
                return float(array)
            return array
    assert isinstance(SampleCDF(), CDFCallable)


def test_cdf_callable_protocol_rejects_noncallable() -> None:
    """
    Test that a non-callable object does not conform to the CDFCallable protocol.
    """
    assert not isinstance(object(), CDFCallable)
