"""
Unit tests for the MultiNullJSDTest class constructor and methods.
"""
from multinull_jsd.core import MultiNullJSDTest
from multinull_jsd.cdf_backends import MC_CDF_BACKEND_FACTORY
from tests.conftest import FloatArray
from typing import TypeAlias

import numpy as np
import pytest

IntDType: TypeAlias = np.int64
FloatDType: TypeAlias = np.float64


def test_init_rejects_non_integer_or_bool_evidence_size(k_default: int) -> None:
    """
    evidence_size must be an integer (bool/float rejected).
    """
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(evidence_size=True, prob_dim=k_default)
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(evidence_size=10.0, prob_dim=k_default)  # type: ignore[arg-type]


def test_init_rejects_non_integer_or_bool_prob_dim(n_default: int) -> None:
    """
    prob_dim must be an integer (bool/float rejected).
    """
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=True)
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=3.5)  # type: ignore[arg-type]


def test_init_rejects_non_positive_evidence_size(k_default: int) -> None:
    """
    evidence_size must be >= 1.
    """
    with pytest.raises(expected_exception=ValueError):
        MultiNullJSDTest(evidence_size=0, prob_dim=k_default)
    with pytest.raises(expected_exception=ValueError):
        MultiNullJSDTest(evidence_size=-10, prob_dim=k_default)


def test_init_rejects_non_positive_prob_dim(n_default: int) -> None:
    """
    prob_dim must be >= 1.
    """
    with pytest.raises(expected_exception=ValueError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=0)
    with pytest.raises(expected_exception=ValueError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=-3)


def test_init_rejects_unknown_cdf_method(n_default: int, k_default: int) -> None:
    """
    Unknown cdf_method should raise ValueError listing available methods.
    """
    with pytest.raises(expected_exception=ValueError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method="not-a-method")


@pytest.mark.parametrize("mc_method", sorted(MC_CDF_BACKEND_FACTORY.keys()))
def test_init_mc_requires_integer_mc_samples(n_default: int, k_default: int, mc_method: str) -> None:
    """
    For MC backends, mc_samples must be an integer (bool/float/None rejected).
    """
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method, mc_samples=None, seed=0)
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method, mc_samples=True, seed=0)
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(
            evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method,
            mc_samples=1.5, seed=0  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("mc_method", sorted(MC_CDF_BACKEND_FACTORY.keys()))
def test_init_mc_requires_positive_mc_samples(n_default: int, k_default: int, mc_method: str) -> None:
    """
    For MC backends, mc_samples must be >= 1.
    """
    with pytest.raises(expected_exception=ValueError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method, mc_samples=0, seed=0)
    with pytest.raises(expected_exception=ValueError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method, mc_samples=-100, seed=0)


@pytest.mark.parametrize("mc_method", sorted(MC_CDF_BACKEND_FACTORY.keys()))
def test_init_mc_requires_integer_non_negative_seed(n_default: int, k_default: int, mc_method: str) -> None:
    """
    For MC backends, seed must be an integer >= 0.
    """
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method, mc_samples=100, seed=None)
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(
            evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method, mc_samples=100,
            seed=1.2  # type: ignore[arg-type]
        )
    with pytest.raises(expected_exception=TypeError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method, mc_samples=100, seed=False)
    with pytest.raises(expected_exception=ValueError):
        MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method, mc_samples=100, seed=-1)


def test_init_accepts_exact_backend(n_default: int, k_default: int) -> None:
    """
    When implemented, constructing with the exact backend should succeed.
    """
    _ = MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method="exact")


@pytest.mark.parametrize("mc_method", sorted(MC_CDF_BACKEND_FACTORY.keys()))
def test_init_accepts_mc_backends(n_default: int, k_default: int, mc_method: str) -> None:
    """
    When implemented, constructing with any MC backend should succeed given valid mc_samples and seed.
    """
    _ = MultiNullJSDTest(
        evidence_size=n_default, prob_dim=k_default, cdf_method=mc_method, mc_samples=5_000, seed=123
    )


def test_add_nulls_validates_shapes_and_alpha(n_default: int, k_default: int, prob_vec3_default: FloatArray) -> None:
    """
    add_nulls must validate the probability vector: 1-D of length k, non-negative, sums to one.
    """
    test: MultiNullJSDTest = MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method="exact")
    # Wrong shape (2-D but k mismatch)
    with pytest.raises(expected_exception=ValueError):
        test.add_nulls(prob_vector=np.array([[0.5, 0.5], [0.8, 0.2]], dtype=np.float64), target_alpha=0.05)
    # Wrong length
    with pytest.raises(expected_exception=ValueError):
        test.add_nulls(prob_vector=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64), target_alpha=0.05)
    # Alpha invalid
    with pytest.raises(expected_exception=ValueError):
        test.add_nulls(prob_vector=prob_vec3_default, target_alpha=1.2)


def test_remove_nulls_validates_indices(n_default: int, k_default: int) -> None:
    """
    remove_nulls must validate the null_index: integer, in range.
    """
    test: MultiNullJSDTest = MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method="exact")
    test.add_nulls(prob_vector=np.array([0.5, 0.3, 0.2], dtype=np.float64), target_alpha=0.05)
    with pytest.raises(expected_exception=ValueError):
        test.remove_nulls(null_index=0)
    with pytest.raises(expected_exception=TypeError):
        test.remove_nulls(null_index="1")  # type: ignore[arg-type]


def test_infer_p_values_validates_histograms(n_default: int, k_default: int) -> None:
    """
    infer_p_values must validate the histogram: 1-D of length k, non-negative integers, sums to n.
    """
    test: MultiNullJSDTest = MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method="exact")
    test.add_nulls(prob_vector=np.array([0.5, 0.3, 0.2], dtype=np.float64), target_alpha=0.05)
    with pytest.raises(expected_exception=ValueError):
        test.infer_p_values(hist_query=np.array([1, 2, 3, 4], dtype=np.int64))  # wrong k


def test_infer_decisions_validates_histograms(n_default: int, k_default: int) -> None:
    """
    infer_decisions must validate the histogram: 1-D of length k, non-negative integers, sums to n.
    """
    test: MultiNullJSDTest = MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method="exact")
    test.add_nulls(prob_vector=np.array([0.5, 0.3, 0.2], dtype=np.float64), target_alpha=0.05)
    with pytest.raises(expected_exception=ValueError):
        test.infer_decisions(hist_query=np.array([1, 2], dtype=np.int64))


def test_get_alpha_validates_indices(n_default: int, k_default: int) -> None:
    """
    get_alpha must validate the null_index: integer, in range.
    """
    test = MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method="exact")
    with pytest.raises(expected_exception=ValueError):
        _ = test.get_alpha(null_index=0)


def test_get_beta_validates_probability_vectors(n_default: int, k_default: int) -> None:
    """
    get_beta must validate the probability vector: 1-D of length k, non-negative, sums to one.
    """
    test: MultiNullJSDTest = MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method="exact")
    with pytest.raises(expected_exception=ValueError):
        _ = test.get_beta(prob_query=np.array([0.5, 0.6, -0.1], dtype=np.float64))  # invalid p

def test_repr_contains_key_params(n_default: int, k_default: int) -> None:
    test: MultiNullJSDTest = MultiNullJSDTest(evidence_size=n_default, prob_dim=k_default, cdf_method="exact")
    repr_str = repr(test)
    assert "MultiNullJSDTest" in repr_str
    assert str(n_default) in repr_str and str(k_default) in repr_str
