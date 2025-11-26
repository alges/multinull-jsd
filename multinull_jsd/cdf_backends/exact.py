"""
Exact CDF backend
=================

Exhaustively enumerates all histograms in the non-normalized histogram space :math:`\\Delta'_{k,n}` to obtain the exact
distribution of JSd.

Complexity
----------
* For fixed :math:`k`: :math:`O(n^{k-1})` (stars-and-bars).
* For fixed :math:`n`: :math:`O(k^n)`.

Notes
-----
* Enumeration should be **cached per probability vector** so repeated calls avoid re-computation.
"""
from .base import CDFBackend

from multinull_jsd._validators import validate_probability_vector
from multinull_jsd.types import FloatArray, IntArray, FloatDType, IntDType

from typing import Optional

import numpy.typing as npt
import numpy as np

import itertools
import math


class ExactCDFBackend(CDFBackend):
    """
    Exhaustively enumerates all histograms in the non-normalized histogram space :math:`\\Delta'_{k,n}` to obtain the
    exact distribution of JSd.

    Complexity
    ----------
    :math:`O(n^{k-1})` for fixed :math:`k` (stars-and-bars enumeration) or :math:`O(k^n)` for fixed :math:`n`.

    Parameters
    ----------
    evidence_size
        Number of samples :math:`n`. See ``CDFBackend`` for details.

    Notes
    -----
    Enumeration is cached **per probability vector** so repeated calls with the same vector avoid re-computation.
    """
    def __init__(self, evidence_size: int):
        super().__init__(evidence_size=evidence_size)

        # Cache for histogram enumerations keyed by dimension (:math:`k`)
        self._histogram_cache: dict[int, IntArray] = {}
        # Cache for log-factorial values
        self._lf_cache: Optional[FloatArray] = None

    def _enumerate_histograms(self, k: int) -> IntArray:
        """
        Enumerates all possible histograms for a given dimension.

        This function generates all possible histograms (frequency distributions) for the provided dimension. The
        output is a numpy array where each element represents a unique histogram configuration associated with the
        given dimension.

        Parameters
        ----------
        k
            Dimension for which the histograms are enumerated.

        Returns
        -------
        IntArray
            A 2-D numpy array where each row corresponds to a unique histogram configuration for the specified
            dimension.
        """
        n: int = self.evidence_size

        if k == 1:
            return np.array(object=[[n]], dtype=IntDType)

        total_enumeration_positions: int = n + k - 1
        histograms: list[list[int]] = []

        # Iterates over all possible combinations of histogram positions: this follows a "stars-and-bars" enumeration
        for bar_positions in itertools.combinations(iterable=range(total_enumeration_positions), r=k - 1):
            histogram: list[int] = [0] * k
            histogram[0] = bar_positions[0]
            for bar_idx, bar_position in enumerate(bar_positions[1:], start=1):
                histogram[bar_idx] = bar_position - bar_positions[bar_idx - 1] - 1
            histogram[-1] = total_enumeration_positions - bar_positions[-1] - 1
            histograms.append(histogram)

        return np.array(object=histograms, dtype=IntDType)

    def obtain_histograms_and_probabilities(self, prob_vector: FloatArray) -> tuple[IntArray, FloatArray]:
        prob_vector = validate_probability_vector(
            name="prob_vector", value=prob_vector, n_categories=None
        ).astype(dtype=FloatDType, copy=False)

        k: int = prob_vector.shape[0]
        n: int = self.evidence_size

        if k in self._histogram_cache:
            histogram_array: IntArray = self._histogram_cache[k]
        else:
            histogram_array = self._enumerate_histograms(k=k)
            self._histogram_cache[k] = histogram_array

        # If the probability vector has exact zeros, we discard histograms with non-zero counts on those positions
        p_zero_mask: npt.NDArray[np.bool_] = np.equal(prob_vector, 0.0)
        if np.any(p_zero_mask):
            histogram_array = histogram_array[np.all(histogram_array[:, p_zero_mask] == 0, axis=1)]

        if self._lf_cache is None:
            self._lf_cache = np.fromiter(iter=(math.lgamma(h + 1) for h in range(n + 1)), dtype=FloatDType)
        lf_n: float = float(self._lf_cache[n])
        sum_lf_h: FloatArray = self._lf_cache[histogram_array].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sum_h_weighted_log_p: FloatArray = (
                histogram_array * np.where(prob_vector > 0, np.log(prob_vector), 0.0)
            ).sum(axis=1)
        histogram_weights: FloatArray = np.exp(lf_n + sum_h_weighted_log_p - sum_lf_h)

        total_prob: float = float(histogram_weights.sum())
        if total_prob <= 0.0 or not np.isfinite(total_prob):
            raise RuntimeError(
                "Failed to compute a valid multinomial probability mass function.",
            )

        return histogram_array, histogram_weights / total_prob

    def __repr__(self) -> str:
        return f"ExactCDFBackend(evidence_size={self.evidence_size})"
