"""
Monte-Carlo CDF backend based on the Gaussian CLT approximation.
"""
from .base import CDFBackend

from multinull_jsd._validators import validate_int_value, validate_probability_vector
from multinull_jsd.types import IntDType, FloatDType, IntArray, FloatArray

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

    def obtain_histograms_and_probabilities(self, prob_vector: FloatArray) -> tuple[IntArray, FloatArray]:
        prob_vector = validate_probability_vector(
            name="prob_vector", value=prob_vector, n_categories=None
        ).astype(dtype=FloatDType, copy=False)

        # Uniform weights for each MC sample
        weights: FloatArray = np.full(shape=self._mc_samples, fill_value=1.0 / self._mc_samples, dtype=FloatDType)

        n: int = self._evidence_size
        k: int = prob_vector.shape[0]

        # Degenerate case: all mass in a single category → JSd is identically 0
        if np.count_nonzero(a=prob_vector) <= 1:
            histograms: IntArray = np.zeros(shape=(self._mc_samples, k), dtype=IntDType)
            histograms[:, int(np.argmax(a=prob_vector))] = n
            return histograms, weights

        # CLT covariance in categorical (per-sample) space
        cov_matrix: FloatArray = (np.diag(v=prob_vector) - np.outer(a=prob_vector, b=prob_vector)) / n
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0  # Enforce symmetry
        # Small diagonal jitter for numerical stability
        cov_matrix.flat[::k + 1] += 1e-12

        # Sample Z ~ N(0, Σ)
        pseudo_probabilities: FloatArray = self._rng.multivariate_normal(
            mean=prob_vector, cov=cov_matrix, size=self._mc_samples
        ).astype(dtype=FloatDType, copy=False)
        pseudo_probabilities = np.clip(a=pseudo_probabilities, a_min=0.0, a_max=None)
        pseudo_h_sum: FloatArray = pseudo_probabilities.sum(axis=1, keepdims=True)

        # Fix any pathological rows where everything went to zero
        zero_pseudo_h: npt.NDArray[np.bool_] = np.isclose(a=pseudo_h_sum, b=0.0)
        if np.any(zero_pseudo_h):
            pseudo_probabilities[zero_pseudo_h[:, 0], :] = prob_vector  # Replace zero-sum rows with the mean
            pseudo_h_sum = pseudo_probabilities.sum(axis=1, keepdims=True)

        # Normalize back to Δ_k
        pseudo_probabilities /= pseudo_h_sum

        # Convert each q^{(i)} to an integer histogram with sum exactly n:
        # 1) scale to sum n, 2) floor, 3) distribute remaining counts by largest fractional parts.
        scaled_pseudo_p: FloatArray = pseudo_probabilities * float(n)
        floor_probabilities: FloatArray = np.floor(scaled_pseudo_p)
        histograms = floor_probabilities.astype(dtype=IntDType, copy=False)
        fractional_part: FloatArray = scaled_pseudo_p - floor_probabilities

        # Remainders per sample: how many counts we still need to add (can be negative in extreme cases)
        current_sums: IntArray = histograms.sum(axis=1)
        remainders: IntArray = n - current_sums

        # r > 0: add 1 to r[i] largest fractional entries per row
        positive_mask = remainders > 0
        if np.any(a=positive_mask):
            positive_remainders: IntArray = remainders[positive_mask]  # shape (n_pos,)
            positive_remaining_frac: FloatArray = fractional_part[positive_mask]  # shape (n_pos, k)
            # Indices sorted by fractional part, descending
            r_frac_order_desc: IntArray = np.argsort(a=-positive_remaining_frac, axis=1)  # shape (n_pos, k)

            max_remainder: int = int(positive_remainders.max())
            if max_remainder > 0:
                max_remainder = min(max_remainder, k)
                # We only need the first `max_remainder` candidates per row
                cols_top: IntArray = r_frac_order_desc[:, :max_remainder]  # (n_pos, max_r)
                # Build a boolean mask with True for bins to increment. shape (n_pos, k)
                increment_mask: npt.NDArray[np.bool_] = np.zeros_like(a=positive_remaining_frac, dtype=bool)
                row_idx: IntArray = np.arange(increment_mask.shape[0])[:, None]  # (n_pos, 1)
                rank: IntArray = np.arange(max_remainder)[None, :]  # (1, max_r)
                # For each row j, mark columns cols_top[j, :r_pos[j]] as True
                increment_mask[row_idx, cols_top] = rank < positive_remainders[:, None]
                # Finally, add 1 to those entries in the original histograms
                histograms[positive_mask] += increment_mask.astype(dtype=histograms.dtype)

        # This branch should not occur; it's only a defensive practice in case of numerical issues
        # r < 0: subtract 1 from the -r[i] smaller fractional entries per row
        negative_mask = remainders < 0
        if np.any(negative_mask):
            negative_remainders: IntArray = -remainders[negative_mask]  # shape (n_neg,)
            negative_remaining_part: FloatArray = fractional_part[negative_mask].copy()  # (n_neg, k)
            overcounted_histograms: IntArray = histograms[negative_mask]  # (n_neg, k)
            # Bins with zero counts cannot be decremented: move them to the "end" by giving them +inf fractional part
            # before sorting.
            negative_remaining_part[overcounted_histograms <= 0] = np.inf
            # Indices sorted by fractional part, ascending
            r_frac_order_asc = np.argsort(a=negative_remaining_part, axis=1)  # (n_neg, k)

            max_remove: int = int(negative_remainders.max())
            if max_remove > 0:
                max_remove = min(max_remove, k)
                cols_top = r_frac_order_asc[:, :max_remove]  # (n_neg, max_remove)
                # Boolean mask for bins to decrement
                decrement_mask: npt.NDArray[np.bool_] = np.zeros_like(negative_remaining_part, dtype=bool)  # (n_neg,k)
                row_idx = np.arange(decrement_mask.shape[0])[:, None]  # (n_neg, 1)
                rank = np.arange(max_remove)[None, :]  # (1, max_remove)
                # For each row j, mark columns cols_top[j, :r_neg[j]] as True
                decrement_mask[row_idx, cols_top] = rank < negative_remainders[:, None]
                # Subtract 1 from those entries
                histograms[negative_mask] -= decrement_mask.astype(histograms.dtype)

        return histograms, weights

    def __repr__(self) -> str:
        return (
            f"NormalMCCDFBackend(evidence_size={self.evidence_size}, mc_samples={self._mc_samples}, seed={self._seed})"
        )
