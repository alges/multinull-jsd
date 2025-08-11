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
from multinull_jsd.cdf_backends import CDF_BACKEND_FACTORY, MC_CDF_BACKENDS
from multinull_jsd._validators import (
    validate_int_value, validate_finite_array, validate_histogram_batch, validate_probability_batch,
    validate_null_indices
)
from multinull_jsd.types import FloatArray, IntArray, FloatDType
from typing import Optional, Sequence, overload

import numpy.typing as npt
import numpy as np


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
    TypeError
        If any of the parameters are of incorrect type.
    ValueError
        If any of the parameters are invalid, such as negative or non-integer values.
    """

    def __init__(
        self, evidence_size: int, prob_dim: int, cdf_method: str = "exact", mc_samples: Optional[int] = None,
        seed: Optional[int] = None
    ) -> None:

        # Parameter validation
        self._n: int = validate_int_value(name="evidence_size", value=evidence_size, min_value=1)
        self._k: int = validate_int_value(name="prob_dim", value=prob_dim, min_value=1)

        if cdf_method not in CDF_BACKEND_FACTORY:
            raise ValueError(
                f"Invalid CDF method '{cdf_method!r}'. Must be one of {", ".join(sorted(CDF_BACKEND_FACTORY.keys()))}."
            )

        if cdf_method in MC_CDF_BACKENDS:
            validate_int_value(name="mc_samples", value=mc_samples, min_value=1)
            validate_int_value(name="seed", value=seed, min_value=0)

        # Initialization of container for null hypotheses
        self._nulls: IndexedHypotheses = IndexedHypotheses(
            cdf_backend=CDF_BACKEND_FACTORY[cdf_method](self._n, mc_samples, seed)
        )

        raise NotImplementedError

    def add_nulls(self, prob_vector: npt.ArrayLike, target_alpha: float | Sequence[float]) -> None:
        """
        Add one or multiple null hypotheses.

        Parameters
        ----------
        prob_vector
            Probability vector(s) for the null hypothesis or hypotheses. Can be a 1-D array of shape ``(k,)`` or a 2-D
            array of shape ``(m, k)``, where ``m`` is the number of nulls and ``k`` is the number of categories.
        target_alpha
            Desired significance level(s) for the null hypothesis or hypotheses. Can be a scalar float or a 1-D array of
            floats of length ``m``. If a scalar is provided, the same significance level is applied to all new nulls.

        Raises
        ------
        ValueError
            Shape mismatch, invalid probability vector, or invalid target significance level.
        """
        # Validation of the probability vector(s)
        prob_array: FloatArray = validate_probability_batch(
            name="prob_vector", value=prob_vector, n_categories=self._k
        )

        # Validation of the target alpha(s)
        target_alpha_vec: FloatArray = validate_finite_array(
            name="target_alpha", value=np.atleast_1d(target_alpha)
        ).astype(dtype=FloatDType)
        if target_alpha_vec.ndim != 1:
            raise ValueError("Target alpha must be a scalar or a 1-D sequence.")
        if target_alpha_vec.size == 1:
            target_alpha_vec = np.broadcast_to(array=target_alpha_vec, shape=(prob_array.shape[0],))
        elif target_alpha_vec.shape[0] != prob_array.shape[0]:
            raise ValueError("Target alpha vector and probability vector must have the same length.")
        if np.any(a=target_alpha_vec < 0) or np.any(a=target_alpha_vec > 1):
            raise ValueError("Target alpha values must lie in [0, 1].")

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
        null_index_tuple: tuple[int, ...] = validate_null_indices(
            name="null_index", value=null_index, n_nulls=len(self._nulls)
        )
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

    def infer_p_values(self, hist_query: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute per-null p-values for a histogram or batch of histograms.

        Parameters
        ----------
        hist_query
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
        query_array: IntArray = validate_histogram_batch(
            name="hist_query", value=hist_query, n_categories=self._k, histogram_size=self._n
        )
        raise NotImplementedError

    def infer_decisions(self, hist_query: npt.ArrayLike) -> int | npt.NDArray[int]:
        """
        Apply the decision rule and return an *integer label array* with the same batch shape as *query*:

        * Decision outputs the index ``k`` when the null hypothesis of index ``k`` is selected as the least-rejected
          (accepted).
        * Decision outputs ``-1`` when the alternative hypothesis is chosen (i.e., all nulls are rejected).

        Parameters
        ----------
        hist_query
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
        query_array: IntArray = validate_histogram_batch(
            name="hist_query", value=hist_query, n_categories=self._k, histogram_size=self._n
        )
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
        null_indices: tuple[int, ...] = validate_null_indices(
            name="null_index", value=null_index, n_nulls=len(self._nulls)
        )
        raise NotImplementedError

    def get_beta(self, prob_query: npt.ArrayLike) -> float | FloatArray:
        """
        Get the maximum Type-II error probability (:math:`\\beta`) over all null hypotheses for a given probability
        vector

        Parameters
        ----------
        prob_query
            Probability vector or batch of probability vectors to test. Must be a 1-D array of shape ``(k,)`` or a 2-D
            array of shape ``(m,k)``, where ``m`` is the number of vectors and ``k`` is the number of categories.

        Returns
        -------
        float | FloatArray
            Estimated maximum Type-II error probability over all null hypotheses. If the input is a single histogram,
            a scalar float is returned; if the input is a batch, a 1-D array of floats is returned.
        """
        query_array: FloatArray = validate_probability_batch(name="prob_query", value=prob_query, n_categories=self._k)
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
