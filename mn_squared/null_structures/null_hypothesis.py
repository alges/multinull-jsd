"""
Data holder representing **one** null hypothesis.

Responsibilities
----------------
* Store probability vector :math:`\\mathbf{p}` and target significance level.
* Offer fast vectorised p-value computation via a supplied CDF backend.
* Provide helpers for p-value calculation and threshold retrieval.

This module focuses on input validation, bookkeeping, and clean method signatures. Heavy numerical work belongs in CDF
backends.
"""
from mn_squared._jsd_distance import jsd
from mn_squared.cdf_backends import CDFBackend
from mn_squared._validators import validate_bounded_value, validate_probability_vector, validate_histogram_batch
from mn_squared.types import FloatArray, ScalarFloat, IntArray, CDFCallable, FloatDType
from typing import Any, Optional

import numpy.typing as npt
import numpy as np

BINARY_SEARCH_DEPTH: int = 2048


class NullHypothesis:
    """
    Lightweight data class that wraps a *single* null hypothesis.

    It stores

    * the reference probability vector :math:`\\mathbf{p}` (shape ``(k,)``)
    * per-hypothesis target significance level.
    * a callable CDF obtained from a CDF backend.

    Parameters
    ----------
    prob_vector
        Probability vector (1-D, non-negative, sums to one).
    cdf_backend
        Backend used to create the CDF callable. It also fixes the evidence size :math:`n`.

    Raises
    ------
    TypeError
        If inputs are not of the expected types.
    ValueError
        If *prob_vector* is not 1-D, contains negative values, or does not sum to one.
    """
    def __init__(self, prob_vector: npt.ArrayLike, cdf_backend: CDFBackend) -> None:
        if not isinstance(cdf_backend, CDFBackend):
            raise TypeError("cdf_backend must be an instance of CDFBackend.")

        self._p: FloatArray = validate_probability_vector(name="prob_vector", value=prob_vector, n_categories=None)
        self._backend: CDFBackend = cdf_backend
        self._cdf: CDFCallable = self._backend.get_cdf(prob_vector=self._p)
        self._alpha: Optional[ScalarFloat] = None
        self._cached_jsd_threshold: Optional[ScalarFloat] = None

    def set_target_alpha(self, target_alpha: ScalarFloat) -> None:
        """
        Store the user-specified target significance level (Type-I error budget) for this null.

        Parameters
        ----------
        target_alpha
            Desired significance level in :math:`[0,1]`.

        Raises
        ------
        TypeError
            If ``target_alpha`` is not a real number.
        ValueError
            If ``target_alpha`` is outside :math:`[0,1]`.
        """
        if isinstance(target_alpha, bool):
            raise TypeError("target_alpha must be a real number.")

        self._alpha = validate_bounded_value(
            name="target_alpha", value=float(target_alpha), min_value=0.0, max_value=1.0
        )
        self._cached_jsd_threshold = None

    def get_target_alpha(self) -> Optional[ScalarFloat]:
        """
        Retrieve the target alpha value.

        This method returns the target alpha value, which is an optional scalar floating-point number.

        Returns
        -------
        The target alpha value, or None if it has not been set yet.
        """
        return self._alpha

    def get_jsd_threshold(self) -> ScalarFloat:
        """
        Return the critical JSd value :math:`\\tau` such that :math:`\\mathrm{CDF}(\\tau^-) \\geq 1-\\alpha`.

        Raises
        ------
        RuntimeError
            If no target α has been set via ``set_target_alpha``.

        Returns
        -------
        ScalarFloat
            The smallest value satisfying the constraint (numerically approximated).
        """
        if self._alpha is None:
            raise RuntimeError("Target alpha must be set before retrieving the JSD threshold.")

        if self._cached_jsd_threshold is None:
            # Compute the target CDF value for the threshold to satisfy
            target_cdf: float = 1.0 - float(self._alpha)

            # Edge cases
            if target_cdf <= 0.0:
                return 0.0
            if target_cdf >= 1.0:
                return 1.0

            # Binary search for the threshold in the interval [0, 1]
            interval_low: float = 0.0
            interval_high: float = 1.0

            for _ in range(BINARY_SEARCH_DEPTH):
                interval_mid: float = (interval_low + interval_high) / 2.0
                if interval_mid == interval_low or interval_mid == interval_high:
                    break
                if float(self._cdf(np.nextafter(interval_mid, -np.inf))) >= target_cdf:
                    interval_high = interval_mid
                else:
                    interval_low = interval_mid

            self._cached_jsd_threshold = (interval_low + interval_high) / 2.0

        return self._cached_jsd_threshold

    def infer_p_value(self, query: IntArray) -> ScalarFloat | FloatArray:
        """
        Compute p-value(s) for a histogram or batch of histograms.

        Parameters
        ----------
        query
            1-D or 2-D array of shape ``(k,)`` or ``(m, k)``, where :math:`k` is the number of categories in the
            multinomial distribution. The trailing dimension must match the number of categories in the reference
            probability vector :math:`\\mathbf{p}`. If 2-D, the first dimension corresponds to the number of queries
            :math:`m` and the second dimension to the categories. Each query must sum to the evidence size :math:`n`.

        Raises
        ------
        ValueError
            If *query*’s trailing dimension differs from :math:`k` or if queries do not sum to the evidence size
            :math:`n`.

        Returns
        -------
        FloatArray
            Array of p-values, one for each query. If the input is 1-D, the output is a scalar; if 2-D, the output is
            a 1-D array of p-values corresponding to each query.
        """
        n: int = self._backend.evidence_size
        query_batch: IntArray = validate_histogram_batch(
            name="query", value=query, n_categories=self._p.shape[-1], histogram_size=n
        )
        distances: FloatArray = jsd(p=self._p, q=query_batch.astype(dtype=FloatDType, copy=False) / n)
        p_values: FloatArray = 1.0 - self._cdf(np.nextafter(distances, -np.inf))

        if query_batch.shape[0] == 1:
            return float(p_values[0])

        return p_values

    @property
    def probability_vector(self) -> FloatArray:
        """
        The reference probability vector :math:`\\mathbf{p}`.

        Returns
        -------
        FloatArray
            1-D array of shape ``(k,)`` containing the probability vector.
        """
        return self._p

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NullHypothesis):
            return False
        if self._backend is not other._backend:
            return False
        return bool(np.array_equal(a1=self._p, a2=other._p))

    def __repr__(self) -> str:
        return (
            f"NullHypothesis(k={int(self._p.shape[-1])}, n={self._backend.evidence_size}, "
            f"alpha={'unset' if self._alpha is None else f'{float(self._alpha):.6g}'})"
        )
