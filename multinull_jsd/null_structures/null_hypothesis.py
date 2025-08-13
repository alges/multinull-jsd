"""
Data holder representing **one** null hypothesis.

Responsibilities
----------------
* Store probability vector :math:`\\mathbf{p}` and target significance level.
* Offer fast vectorised p-value computation via a supplied CDF backend.
* Provide helpers for p-value calculation & threshold retrieval.

This module focuses on input validation, bookkeeping and clean method signatures. Heavy numerical work belongs in CDF
backends.
"""

from multinull_jsd.cdf_backends import CDFBackend
from multinull_jsd._validators import validate_bounded_value, validate_probability_batch, validate_histogram_batch
from multinull_jsd.types import FloatArray, ScalarFloat
from typing import Any, Optional

import numpy.typing as npt
import numpy as np

import numbers


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
        prob_vector = np.asarray(prob_vector)
        if prob_vector.ndim  == 0:
            raise ValueError("prob_vector must be a 1-D array-like object.")
        prob_vector = validate_probability_batch(
            name="prob_vector", value=prob_vector, n_categories=prob_vector.shape[-1]
        )
        if prob_vector.shape[0] != 1:
            raise ValueError("prob_vector must be a 1-D array-like object.")
        if not isinstance(cdf_backend, CDFBackend):
            raise TypeError("cdf_backend must be an instance of CDFBackend.")

        self._p: FloatArray = prob_vector[0]
        self._backend: CDFBackend = cdf_backend
        self._alpha: Optional[ScalarFloat] = None

        raise NotImplementedError

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
        if not isinstance(target_alpha, numbers.Real) or isinstance(target_alpha, bool):
            raise TypeError("target_alpha must be a real number.")

        self._alpha = validate_bounded_value(
            name="target_alpha", value=float(target_alpha), min_value=0.0, max_value=1.0
        )

        raise NotImplementedError

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
            The smallest value satisfying the constraint.
        """
        if self._alpha is None:
            raise RuntimeError("Target alpha must be set before retrieving the JSD threshold.")

        raise NotImplementedError

    def infer_p_value(self, query: FloatArray) -> ScalarFloat | FloatArray:
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
        validate_histogram_batch(
            name="query", value=query, n_categories=self._p.shape[-1], histogram_size=self._backend.evidence_size
        )
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
