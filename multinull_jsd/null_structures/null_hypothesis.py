"""
Data holder representing **one** null hypothesis.

Responsibilities
----------------
* Store probability vector :math:`\\mathbf{p}` and target significance level.
* Offer fast vectorised p-value computation via a supplied CDF backend.
* Provide the decision threshold needed by the JSd rule.

The heavy mathematical lifting lives in the CDF backend; this module focuses on input validation, bookkeeping and a
clean comparison API (``__eq__``).
"""
from multinull_jsd.cdf_backends import CDFBackend
from multinull_jsd.types import FloatArray

from typing import Any


class NullHypothesis:
    """
    Lightweight data class that wraps a *single* null hypothesis.

    It stores

    * the reference probability vector :math:`\\mathbf{p}` (shape ``(k,)``)
    * per-hypothesis target significance level.
    * a callable CDF obtained from a CDF backend.

    and exposes helpers for p-value calculation & threshold retrieval.

    Parameters
    ----------
    prob_vector
        Probability vector (1-D, non-negative, sums to one).
    cdf_backend
        Instance used to create the CDF callable.

    Raises
    ------
    ValueError
        If *prob_vector* is not 1-D, contains negative values, or does not sum to one.
    """
    def __init__(self, prob_vector: FloatArray, cdf_backend: CDFBackend) -> None:
        raise NotImplementedError

    def set_target_alpha(self, target_alpha: float) -> None:
        """
        Store the user-specified target significance level (Type-I error budget) for this null.

        Parameters
        ----------
        target_alpha
            Desired significance level in :math:`[0,1]`.

        Raises
        ------
        ValueError
            If ``target_alpha`` is outside :math:`[0,1]`.
        """
        raise NotImplementedError

    def get_jsd_threshold(self) -> float:
        """
        Return the critical JSd value :math:`\\tau` such that :math:`\\mathrm{CDF}(\\tau^-) \\geq 1-\\alpha`.

        Returns
        -------
        float
            The smallest value satisfying the constraint.

        Raises
        ------
        RuntimeError
            If no target α has been set via ``set_target_alpha``.
        """
        raise NotImplementedError

    def infer_p_value(self, query: FloatArray) -> FloatArray:
        """
        Compute p-value(s) for a histogram or batch of histograms.

        Parameters
        ----------
        query
            Histogram(s) with shape ``(...,k)``. Accepts only histograms of raw counts.

        Returns
        -------
        FloatArray
            Array of p-values broadcast to ``query.shape[:-1]``.

        Raises
        ------
        ValueError
            If *query*’s trailing dimension differs from :math:`k`.
        """
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
