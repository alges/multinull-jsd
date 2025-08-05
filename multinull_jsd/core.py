"""
High-level orchestrator for the Multi-Null JSd test.

Typical usage
-------------
>>> from multinull_jsd import MultiNullJSDTest
>>> test = MultiNullJSDTest(evidence_size=100, prob_dim=3, cdf_method="mc_multinomial", mc_samples=10_000, seed=0)
>>> test.add_nulls([0.5, 0.3, 0.2], target_alpha=0.05)  # Add a null hypothesis
>>> test.add_nulls([0.4, 0.4, 0.2], target_alpha=0.01)  # Add another null hypothesis
>>> h = [55, 22, 23]  # Observed histogram to test
>>> p_vals = test.infer_p_values(h)  # Array of p-values for each null hypothesis
>>> decisions = test.infer_decisions(h)  # Array of decisions (1 or 2 for each null hypothesis, -1 for the alternative)
"""
from multinull_jsd.null_structures import IndexedHypotheses
from multinull_jsd.types import CDFBackendName, FloatArray
from typing import Optional, Sequence, overload

import numpy.typing as npt


class MultiNullJSDTest:
    """
    Class that orchestrates the Multi-Null JSd test decision rule.

    Parameters
    ----------
    evidence_size
        Number of samples :math:`n` in each histogram.
    prob_dim
        Number of categories :math:`k`.
    cdf_method
        CDF computation backend to use. Available options are ``"exact"``, ``"mc_multinomial"``, and ``"mc_normal"``.
    mc_samples
        Monte-Carlo repetitions :math:`N` (only for MC backends). Ignored for the exact CDF backend.
    seed
        RNG seed for reproducibility of Monte-Carlo backends. Ignored for the exact CDF backend.

    Raises
    ------
    ValueError
        On invalid ``evidence_size``, ``prob_dim``, ``cdf_method``, ``mc_samples``, or ``seed``.
    """

    def __init__(
        self, evidence_size: int, prob_dim: int, cdf_method: CDFBackendName = "exact",
        mc_samples: Optional[int] = None, seed: Optional[int] = None
    ) -> None:
        raise NotImplementedError

    def add_nulls(self, prob_vector: npt.ArrayLike, target_alpha: float | Sequence[float]) -> None:
        """
        Add one or multiple null hypotheses.

        Rules
        -----
        * If *target_alpha* is a scalar, the same α is applied to all new nulls; if it is a 1-D sequence, it must match
          the number of probability vectors provided.
        * Probability vectors must sum to one. ``prob_vector``  must have shape ``(k,)`` or ``(m, k)``.

        Raises
        ------
        ValueError
            Shape mismatch, invalid probability vector, or invalid target significance level.
        """
        raise NotImplementedError

    def remove_nulls(self, null_index: int | Sequence[int]) -> None:
        """
        Remove one or multiple null hypotheses.

        Parameters
        ----------
        null_index
            Index or sequence of indices of null hypotheses to remove. Must be valid indices of the current nulls. The
            indexing is one-based, i.e., the first null hypothesis has index 1.
        """
        # TODO: Ensure unique null indexes are considered
        raise NotImplementedError

    def get_nulls(self) -> IndexedHypotheses:
        """
        Return the current null hypotheses.

        Returns
        -------
        IndexedHypotheses
            Container with the current null hypotheses, providing access by index.
        """
        raise NotImplementedError

    def infer_p_values(self, query: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute per-null p-values for a histogram or batch of histograms.

        Parameters
        ----------
        query
            Histogram or batch of histograms to test. Must be a 1-D array of shape ``(k,)`` or a 2-D array of shape
            ``(m,k)``, where ``m`` is the number of histograms and ``k`` is the number of categories. The histograms
            must be not normalized, i.e., they need to be raw counts of samples in each category and sum to the
            evidence size.

        Returns
        -------
        npt.ArrayLike
            Array of p-values for each null hypothesis. If the input is a single histogram, the output will have shape
            ``(L,)``, where ``L`` is the number of null hypotheses. Each entry corresponds to the p-value for the
            respective null hypothesis. If the input is a batch, the output will have shape ``(m,L)``.
        """
        raise NotImplementedError

    def infer_decisions(self, query: npt.ArrayLike) -> int | npt.NDArray[int]:
        """
        Apply the decision rule and return an *integer label array* with the same batch shape as *query*:

        * Decision outputs the index ``k`` when the null hypothesis of index ``k`` is accepted.
        * Decision outputs ``-1`` when the alternative hypothesis is chosen (i.e., all nulls are rejected).

        Parameters
        ----------
        query
            Histogram or batch of histograms to test. Must be a 1-D array of shape ``(k,)`` or a 2-D array of shape
            ``(m,k)``, where ``m`` is the number of histograms and ``k`` is the number of categories. The histograms
            must be not normalized, i.e., they need to be raw counts of samples in each category and sum to the
            evidence size.

        Returns
        -------
        int | npt.NDArray[int]
            Array of decisions with the same batch shape as *query*. Each entry corresponds to the decision for the
            respective histogram in the batch. If the input is a single histogram, the output will be a scalar integer.
            If the input is a batch, the output will be a 1-D array of integers.
        """
        raise NotImplementedError

    @overload
    def get_alpha(self, null_index: int) -> float: ...
    @overload
    def get_alpha(self, null_index: Sequence[int]) -> FloatArray: ...

    def get_alpha(self, null_index: int | Sequence[int]) -> float | FloatArray:
        """
        Return the actual significance level (Type-I error probability) for a null hypothesis or a list of hypotheses.

        Parameters
        ----------
        null_index
            Index or sequence of indices of null hypotheses. Must be valid indices of the current nulls. The indexing
            is one-based, i.e., the first null hypothesis has index 1.

        Returns
        -------
        float | Sequence[float]
            The actual significance level for the specified null hypothesis or a list of significance levels for each
            specified null hypothesis. If a single index is provided, a scalar float is returned; if a sequence of
            indices is provided, a 1-D array of floats is returned.
        """
        raise NotImplementedError

    def get_beta(self, query: npt.ArrayLike) -> float | FloatArray:
        """
        Get the maximum Type-II error probability (:math:`\\beta`) over all null hypotheses for a given histogram
        or batch of histograms. This is the probability of failing to reject null hypotheses when the alternative is
        true.

        Parameters
        ----------
        query
            Histogram or batch of histograms to test. Must be a 1-D array of shape ``(k,)`` or a 2-D array of shape
            ``(m,k)``, where ``m`` is the number of histograms and ``k`` is the number of categories.

        Returns
        -------
        float | FloatArray
            Estimated maximum Type-II error probability over all null hypotheses. If the input is a single histogram,
            a scalar float is returned; if the input is a batch, a 1-D array of floats is returned.
        """
        raise NotImplementedError

    def get_fwer(self) -> float:
        """
        Returns the actual Family-Wise Error Rate (FWER) of the Multi-Null JSd test, i.e., the probability of making at
        least one Type-I error when any of the null hypotheses is true.

        Returns
        -------
        float
            The actual FWER of the Multi-Null JSd test.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
