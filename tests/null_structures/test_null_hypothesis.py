"""
Unit tests for the NullHypothesis class.
"""
from mn_squared.null_structures import NullHypothesis
from tests.conftest import TestCDFBackend
from typing import TypeAlias, cast

import numpy.typing as npt
import numpy as np
import pytest

IntDType: TypeAlias = np.int64
FloatDType: TypeAlias = np.float64

IntArray: TypeAlias = npt.NDArray[IntDType]
FloatArray: TypeAlias = npt.NDArray[FloatDType]


def test_null_hypothesis_init_rejects_non_backend(prob_vec3_default: FloatArray) -> None:
    """
    The constructor must reject any cdf_backend that is not an instance of CDFBackend.
    """
    with pytest.raises(expected_exception=TypeError):
        NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=object())  # type: ignore[arg-type]


def test_null_hypothesis_init_rejects_bad_probability_vector(fake_backend: TestCDFBackend) -> None:
    """
    The constructor must validate the probability vector: 1-D, non-negative, sums to one.
    """
    # Wrong shape (2-D)
    with pytest.raises(expected_exception=ValueError):
        NullHypothesis(
            prob_vector=np.array(object=[[0.5, 0.5], [0.8, 0.2]], dtype=np.float64), cdf_backend=fake_backend
        )

    # Negative entry
    with pytest.raises(expected_exception=ValueError):
        NullHypothesis(prob_vector=np.array(object=[0.6, -0.1, 0.5], dtype=np.float64), cdf_backend=fake_backend)

    # Sum not 1 within tolerance
    with pytest.raises(expected_exception=ValueError):
        NullHypothesis(prob_vector=np.array(object=[0.6, 0.3, 0.3], dtype=np.float64), cdf_backend=fake_backend)


def test_set_target_alpha_rejects_non_real(prob_vec3_default: FloatArray, fake_backend: TestCDFBackend) -> None:
    """
    set_target_alpha must reject non-real types (including bool).
    """
    nh: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    with pytest.raises(expected_exception=TypeError):
        nh.set_target_alpha(target_alpha=True)
    with pytest.raises(expected_exception=TypeError):
        nh.set_target_alpha(target_alpha=complex(1, 0))  # type: ignore[arg-type]


def test_set_target_alpha_rejects_out_of_bounds(prob_vec3_default: FloatArray, fake_backend: TestCDFBackend) -> None:
    """
    set_target_alpha must enforce alpha ∈ [0,1].
    """
    nh: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    with pytest.raises(expected_exception=ValueError):
        nh.set_target_alpha(target_alpha=-1e-6)
    with pytest.raises(expected_exception=ValueError):
        nh.set_target_alpha(target_alpha=1.0000001)


def test_get_jsd_threshold_raises_if_alpha_not_set(
    prob_vec3_default: FloatArray, fake_backend: TestCDFBackend
) -> None:
    """
    get_jsd_threshold must raise RuntimeError if target alpha has not been set.
    """
    nh: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    with pytest.raises(expected_exception=RuntimeError):
        nh.get_jsd_threshold()


def test_get_jsd_threshold_success_path(prob_vec3_default: FloatArray, fake_backend: TestCDFBackend) -> None:
    """
    When implemented, get_jsd_threshold should return a finite scalar once alpha is set.
    """
    nh: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    nh.set_target_alpha(target_alpha=0.1)
    tau: float | np.floating = nh.get_jsd_threshold()
    assert isinstance(tau, float)
    assert 0.0 <= tau <= 1.0


def test_infer_p_value_rejects_wrong_histogram_shape(
    prob_vec3_default: FloatArray, fake_backend: TestCDFBackend
) -> None:
    """
    infer_p_value must validate the trailing dimension equals k.
    """
    nh: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    with pytest.raises(expected_exception=ValueError):
        nh.infer_p_value(query=np.array(object=[1, 2, 7, 0], dtype=np.int64))  # k=4 vs expected 3


def test_infer_p_value_rejects_non_integer_counts(prob_vec3_default: FloatArray, fake_backend: TestCDFBackend) -> None:
    """
    infer_p_value must reject non-integer histogram counts.
    """
    nh: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    with pytest.raises(expected_exception=ValueError):
        nh.infer_p_value(query=np.array(object=[5.0, 3.5, 1.5], dtype=np.float64))


def test_infer_p_value_rejects_wrong_total_count(prob_vec3_default: FloatArray, fake_backend: TestCDFBackend) -> None:
    """
    infer_p_value must enforce that each histogram sums to n (backend.evidence_size).
    """
    nh: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    with pytest.raises(expected_exception=ValueError):
        nh.infer_p_value(query=np.array(object=[1, 1, 1], dtype=np.int64))


def test_infer_p_value_valid_inputs(
    prob_vec3_default: FloatArray, fake_backend: TestCDFBackend, n_default: int
) -> None:
    """
    With a valid histogram or batch, infer_p_value should return a scalar/array of p-values.
    """
    nh: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    single: IntArray = np.array(object=[n_default, 0, 0], dtype=np.int64)
    batch: IntArray = np.array(
        object=[[n_default, 0, 0], [n_default - 1, 1, 0], [n_default - 2, 2, 0]], dtype=np.int64
    )
    out_single: float = float(nh.infer_p_value(query=single))
    out_batch: FloatArray = cast(FloatArray, nh.infer_p_value(query=batch))
    assert isinstance(out_single, float)
    assert isinstance(out_batch, np.ndarray) and out_batch.shape == (3,)


def test_null_hypothesis_equality_semantics(prob_vec3_default: FloatArray, fake_backend: TestCDFBackend) -> None:
    """
    Objects with identical p and the same backend should compare equal; otherwise unequal.
    """
    nh1: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    nh2: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default.copy(), cdf_backend=fake_backend)
    assert nh1 == nh2
    nh3: NullHypothesis = NullHypothesis(
        prob_vector=np.array(object=[0.4, 0.4, 0.2], dtype=np.float64), cdf_backend=fake_backend
    )
    assert nh1 != nh3


def test_null_hypothesis_repr_contains_class_and_dimensions(
    prob_vec3_default: FloatArray, fake_backend: TestCDFBackend
) -> None:
    """
    __repr__ should contain the class name and at least the probability dimension and evidence size.
    """
    nh: NullHypothesis = NullHypothesis(prob_vector=prob_vec3_default, cdf_backend=fake_backend)
    repr_str: str = repr(nh)
    assert "NullHypothesis" in repr_str
    assert str(fake_backend.evidence_size) in repr_str and str(prob_vec3_default.shape[-1]) in repr_str
