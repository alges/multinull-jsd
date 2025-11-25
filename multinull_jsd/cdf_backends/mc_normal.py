"""
Monte-Carlo CDF backend based on the Gaussian CLT approximation.
"""
from .base import CDFBackend

from multinull_jsd._jsd_distance import jsd
from multinull_jsd._validators import validate_int_value, validate_probability_vector
from multinull_jsd.types import FloatDType, FloatArray, CDFCallable

import numpy.typing as npt
import numpy as np


class NormalMCCDFBackend(CDFBackend):
    """
    Monte-Carlo estimator of the CDF based on the **Gaussian CLT approximation**:
    :math:`\\mathrm{Multinomial}(n,\\mathbf{p})
    \\approx\\mathcal{N}(n\\mathbf{p},n(\\mathrm{diag}(\\mathbf{p})-\\mathbf{p}\\mathbf{p}^\\mathsf{T}))`.

    Useful when :math:`n` is large and :math:`k` moderate.

    Parameters
    ----------
    evidence_size
        Number of samples :math:`n`.
    mc_samples
        Number of Monte-Carlo repetitions :math:`N`. Must be positive.
    seed
        Random-state seed for reproducibility.
    """
    def __init__(self, evidence_size: int, mc_samples: int, seed: int):
        super().__init__(evidence_size=evidence_size)

        self._mc_samples: int = validate_int_value(name="mc_samples", value=mc_samples, min_value=1)
        self._seed: int = validate_int_value(name="seed", value=seed, min_value=0)
        self._rng: np.random.Generator = np.random.default_rng(seed=self._seed)

    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        prob_vector = validate_probability_vector(
            name="prob_vector", value=prob_vector, n_categories=None
        ).astype(dtype=FloatDType, copy=False)

        cdf_key: tuple[float, ...] = self._prob_vector_to_key(prob_vector=prob_vector)
        if cdf_key in self._cdf_cache:
            return self._cdf_cache[cdf_key]

        # The probability vector is non-degenerate; i.e., there are multiple non-zero entries
        if np.count_nonzero(a=prob_vector) > 1:
            cov_matrix: FloatArray = (
                (np.diag(v=prob_vector) - np.outer(a=prob_vector, b=prob_vector)) / self.evidence_size
            )
            cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Ensure symmetry
            cov_matrix.flat[::prob_vector.shape[0] + 1] += 1e-12  # Add small epsilon to avoid a singular matrix

            pseudo_histograms: FloatArray = self._rng.multivariate_normal(
                mean=prob_vector, cov=cov_matrix, size=self._mc_samples
            ).astype(dtype=FloatDType, copy=False)
            pseudo_histograms = np.clip(a=pseudo_histograms, a_min=0.0, a_max=1.0)
            pseudo_h_sum: FloatArray = pseudo_histograms.sum(axis=1, keepdims=True)

            zero_pseudo_h: npt.NDArray[np.bool_] = np.isclose(a=pseudo_h_sum, b=0.0)
            if np.any(zero_pseudo_h):
                pseudo_histograms[zero_pseudo_h[:, 0], :] = prob_vector  # Replace zero-sum histograms with the mean
                pseudo_h_sum = pseudo_histograms.sum(axis=1, keepdims=True)

            distances: FloatArray = jsd(p=prob_vector, q=pseudo_histograms / pseudo_h_sum)

        # The probability vector is a degenerate case with only one non-zero entry
        else:
            distances = np.zeros(shape=(self._mc_samples,), dtype=FloatDType)

        cdf_callable: CDFCallable = self._build_cdf_from_samples(distances=distances, weights=None)
        self._cdf_cache[cdf_key] = cdf_callable
        return cdf_callable

    def __repr__(self) -> str:
        return (
            f"NormalMCCDFBackend(evidence_size={self.evidence_size}, mc_samples={self._mc_samples}, seed={self._seed})"
        )
