"""
High-level orchestrator for the Multi-Null Non-Parametric JSd test.

Typical usage
-------------
>>> from mn_squared import MNSquaredTest
>>> test = MNSquaredTest(evidence_size=100, prob_dim=3, cdf_method="mc_multinomial", mc_samples=10_000, seed=0)
>>> test.add_nulls([0.5, 0.3, 0.2], target_alpha=0.05)  # Add a null hypothesis
>>> test.add_nulls([0.4, 0.4, 0.2], target_alpha=0.01)  # Add another null hypothesis
>>> h = [55, 22, 23]  # Observed histogram to test
>>> p_vals = test.infer_p_values(h)  # Array of p-values for each null hypothesis
>>> decisions = test.infer_decisions(h)  # Array of decisions (1 or 2 for each null hypothesis, -1 for the alternative)
"""
from mn_squared.null_structures import IndexedHypotheses
from mn_squared.cdf_backends import NON_MC_CDF_BACKEND_FACTORY, MC_CDF_BACKEND_FACTORY, CDFBackend
from mn_squared._validators import (
    validate_int_value, validate_finite_array, validate_histogram_batch, validate_probability_batch,
    validate_null_indices
)
from mn_squared.types import FloatArray, IntArray, FloatDType, IntDType, ScalarFloat, ScalarInt
from typing import Optional, Sequence, overload

import numpy.typing as npt
import numpy as np


class MNSquaredTest:
    """
    Class that orchestrates the Multi-Null Non-Parametric JSd test decision rule.

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
        If any of the parameters are of an incorrect type.
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

        self._cdf_method: str = cdf_method
        self._backend: CDFBackend
        self._mc_samples: Optional[int] = mc_samples
        self._seed: Optional[int] = seed

        if cdf_method in NON_MC_CDF_BACKEND_FACTORY:
            self._backend = NON_MC_CDF_BACKEND_FACTORY[cdf_method](self._n)
        elif cdf_method in MC_CDF_BACKEND_FACTORY:
            self._backend = MC_CDF_BACKEND_FACTORY[cdf_method](
                self._n,
                validate_int_value(name="mc_samples", value=mc_samples, min_value=1),
                validate_int_value(name="seed", value=seed, min_value=0)
            )
        else:
            raise ValueError(
                f"Invalid CDF method '{cdf_method!r}'. Must be one of "
                f"{', '.join(sorted(NON_MC_CDF_BACKEND_FACTORY.keys() | MC_CDF_BACKEND_FACTORY.keys()))}."
            )

        # Initialization of container for null hypotheses
        self._nulls: IndexedHypotheses = IndexedHypotheses(cdf_backend=self._backend, prob_dim=self._k)

    def add_nulls(self, prob_vector: npt.ArrayLike, target_alpha: ScalarFloat | Sequence[ScalarFloat]) -> None:
        """
        Add one or multiple null hypotheses.

        Parameters
        ----------
        prob_vector
            Probability vector(s) for the null hypothesis or hypotheses. Can be a 1-D array of shape ``(k,)`` or a 2-D
            array of shape ``(m, k)``, where ``m`` is the number of nulls, and ``k`` is the number of categories.
        target_alpha
            Desired significance level(s) for the null hypothesis or hypotheses. Can be a scalar float or a 1-D array
            of floats of length ``m``. If a scalar is provided, the same significance level is applied to all new
            nulls.

        Raises
        ------
        ValueError
            Shape mismatch, invalid probability vector, or invalid target significance level.
        TypeError
            If target_alpha contains boolean values.
        """
        # Validation of the probability vector(s)
        prob_array: FloatArray = validate_probability_batch(
            name="prob_vector", value=prob_vector, n_categories=self._k
        )

        # Detect and reject boolean alphas explicitly
        if isinstance(target_alpha, bool):
            raise TypeError("target_alpha must be a real number or a sequence of real numbers, not bool.")
        if isinstance(target_alpha, Sequence) and not isinstance(target_alpha, (str, bytes)):
            if any(isinstance(a, bool) for a in target_alpha):
                raise TypeError("target_alpha sequence must not contain boolean values.")

        # Validation of the target alpha(s)
        target_alpha_vec: FloatArray = validate_finite_array(
            name="target_alpha", value=np.atleast_1d(np.asarray(a=target_alpha, dtype=float))
        ).astype(dtype=FloatDType)
        if target_alpha_vec.ndim != 1:
            raise ValueError("Target alpha must be a scalar or a 1-D sequence.")

        m_new_nulls: int = prob_array.shape[0]
        if target_alpha_vec.size == 1:
            target_alpha_vec = np.broadcast_to(array=target_alpha_vec, shape=(m_new_nulls,))
        elif target_alpha_vec.shape[0] != m_new_nulls:
            raise ValueError("Target alpha vector and probability vector must have the same length.")
        if np.any(a=target_alpha_vec < 0) or np.any(a=target_alpha_vec > 1):
            raise ValueError("Target alpha values must lie in [0, 1].")

        # Add new nulls on validation success
        for null_prob, null_alpha in zip(prob_array, target_alpha_vec, strict=True):
            self._nulls.add_null(prob_vector=null_prob, target_alpha=float(null_alpha))

    def remove_nulls(self, null_index: ScalarInt | Sequence[ScalarInt]) -> None:
        """
        Remove one or multiple null hypotheses.

        Parameters
        ----------
        null_index
            Index or sequence of indices of null hypotheses to remove. Must be valid indices of the current nulls. The
            indexing is one-based, i.e., the first null hypothesis has index 1.
        """
        # null_index validation is on IndexedHypotheses.__delitem__ method
        del self._nulls[null_index]

    def get_nulls(self) -> IndexedHypotheses:
        """
        Return the current null hypotheses.

        Returns
        -------
        IndexedHypotheses
            Container with the current null hypotheses, providing access by index.
        """
        return self._nulls

    def infer_p_values(self, hist_query: npt.ArrayLike) -> FloatArray:
        """
        Compute per-null p-values for a histogram or batch of histograms.

        Parameters
        ----------
        hist_query
            Histogram or batch of histograms to be tested. Must be a 1-D array of shape ``(k,)`` or a 2-D array of
            shape ``(m,k)``, where ``m`` is the number of histograms, and ``k`` is the number of categories. The
            histograms must be not normalized, i.e., they need to be raw counts of samples in each category and sum to
            the evidence size.

        Raises
        ------
        RuntimeError
            If no null hypotheses have been added yet.

        Returns
        -------
        FloatArray
            Array of p-values for each null hypothesis. If the input is a single histogram, the output will have a
            shape of ``(L,)``, where ``L`` is the number of null hypotheses. Each entry corresponds to the p-value for
            the respective null hypothesis. If the input is a batch, the output will have a shape of ``(m,L)``.
        """
        if len(self._nulls) == 0:
            raise RuntimeError("At least one null hypothesis must be added before calling infer_p_values")

        single_hist_queried: bool = np.asarray(a=hist_query).ndim == 1
        hist_query_array: IntArray = validate_histogram_batch(
            name="hist_query", value=hist_query, n_categories=self._k, histogram_size=self._n
        )

        p_values: FloatArray = np.empty(shape=(hist_query_array.shape[0], len(self._nulls)), dtype=FloatDType)

        for null_idx, null_hypothesis in enumerate(self._nulls):
            p_values[:, null_idx] = null_hypothesis.infer_p_value(query=hist_query_array)

        if single_hist_queried:
            return p_values[0]
        return p_values

    def infer_decisions(self, hist_query: npt.ArrayLike) -> ScalarInt | IntArray:
        """
        Apply the decision rule and return an *integer label array* with the same batch shape as *query*:

        * Decision outputs the index ``k`` when the null hypothesis of index ``k`` is selected as the least-rejected
          (accepted).
        * Decision outputs ``-1`` when the alternative hypothesis is chosen (i.e., all nulls are rejected).

        Parameters
        ----------
        hist_query
            Histogram or batch of histograms to be tested. Must be a 1-D array of shape ``(k,)`` or a 2-D array of
            shape ``(m,k)``, where ``m`` is the number of histograms, and ``k`` is the number of categories. The
            histograms must be not normalized, i.e., they need to be raw counts of samples in each category and sum to
            the evidence size.

        Raises
        ------
        RuntimeError
            If no null hypotheses have been added yet.

        Returns
        -------
        ScalarInt | IntArray
            Array of decisions with the same batch shape as *query*. Each entry corresponds to the decision for the
            respective histogram in the batch. If the input is a single histogram, the output will be a scalar integer.
            If the input is a batch, the output will be a 1-D array of integers.
        """
        if len(self._nulls) == 0:
            raise RuntimeError("At least one null hypothesis must be added before calling infer_decisions")

        single_hist_queried: bool = np.asarray(a=hist_query).ndim == 1
        hist_query_array: IntArray = validate_histogram_batch(
            name="hist_query", value=hist_query, n_categories=self._k, histogram_size=self._n
        )

        p_values: FloatArray = self.infer_p_values(hist_query=hist_query_array)
        target_alphas: FloatArray = np.array(object=[nh.get_target_alpha() for nh in self._nulls], dtype=FloatDType)
        if np.any(a=np.isnan(target_alphas)):
            raise RuntimeError(
                "Null hypotheses must have a valid target significance level before calling infer_decisions"
            )

        non_rejected_mask: npt.NDArray[np.bool_] = p_values > target_alphas[np.newaxis, :]

        m: int = hist_query_array.shape[0]
        decisions: IntArray = np.empty(shape=(m,), dtype=IntDType)
        for query_idx in range(m):
            non_rejected_indexes: IntArray = np.nonzero(a=non_rejected_mask[query_idx])[0]
            if non_rejected_indexes.size > 0:  # At least one non-rejected null
                non_rejected_p_values: FloatArray = p_values[query_idx, non_rejected_indexes]
                max_p_value: float = float(non_rejected_p_values.max())
                tied_null_indexes: IntArray = np.nonzero(a=np.equal(non_rejected_p_values, float(max_p_value)))[0]
                decisions[query_idx] = int(non_rejected_indexes[tied_null_indexes[0]]) + 1
            else:  # All nulls rejected
                decisions[query_idx] = -1

        if single_hist_queried:
            return decisions[0]
        return decisions

    @overload
    def get_alpha(self, null_index: ScalarInt) -> float: ...
    @overload
    def get_alpha(self, null_index: Sequence[ScalarInt]) -> FloatArray: ...

    def get_alpha(self, null_index: ScalarInt | Sequence[ScalarInt]) -> float | FloatArray:
        """
        Return the actual significance level (Type-I error probability) for a null hypothesis or a list of hypotheses.

        Parameters
        ----------
        null_index
            Index or sequence of indices of null hypotheses. Must be valid indices of the current nulls. The indexing
            is one-based, i.e., the first null hypothesis has index 1.

        Returns
        -------
        float | FloatArray
            The actual significance level for the specified null hypothesis or a list of significance levels for each
            specified null hypothesis. If a single index is provided, a scalar float is returned; if a sequence of
            indices is provided, a 1-D array of floats is returned.
        """
        n_nulls: int = len(self._nulls)
        if n_nulls == 0:
            raise ValueError("No null hypotheses are currently registered.")

        validate_null_indices(name="null_index", value=null_index, n_nulls=n_nulls, keep_duplicates=True)

        null_indexes_list: list[int] = np.atleast_1d(np.asarray(a=null_index, dtype=IntDType)).tolist()
        is_scalar_request: bool = isinstance(null_index, (int, np.integer))

        alphas: FloatArray = np.empty(shape=len(null_indexes_list), dtype=FloatDType)

        for idx_pos, null_idx_base_1 in enumerate(null_indexes_list):
            null_p: FloatArray = self._nulls[int(null_idx_base_1)].probability_vector.astype(
                dtype=FloatDType, copy=False,
            )
            # Probability mass of *each* null decision under H ~ Multinomial(n, p_ℓ)
            decision_probs: FloatArray = self._backend.decision_distribution_on_hypothesis(
                prob_vector=null_p,
                decision_fn=lambda h_batch: np.asarray(self.infer_decisions(hist_query=h_batch), dtype=IntDType),
                n_nulls=n_nulls
            )
            null_alpha: float = 1.0 - decision_probs.sum()
            # Clip for numerical stability
            alphas[idx_pos] = float(np.clip(a=null_alpha, a_min=0.0, a_max=1.0))

        if is_scalar_request:
            return float(alphas[0])

        return alphas

    def get_beta(self, prob_query: npt.ArrayLike) -> float | FloatArray:
        """
        Get the overall Type-II error probability (:math:`\\beta`) over all null hypotheses for a given probability
        vector

        Parameters
        ----------
        prob_query
            Probability vector or batch of probability vectors to test. Must be a 1-D array of shape ``(k,)`` or a 2-D
            array of shape ``(m,k)``, where ``m`` is the number of vectors, and ``k`` is the number of categories.

        Returns
        -------
        float | FloatArray
            Estimated overall Type-II error probability over all null hypotheses. If the input is a single histogram,
            a scalar float is returned; if the input is a batch, a 1-D array of floats is returned.
        """
        prob_batch: FloatArray = validate_probability_batch(
            name="prob_query", value=prob_query, n_categories=self._k
        ).astype(dtype=FloatDType, copy=False)

        n_alternatives: int = prob_batch.shape[0]
        n_nulls: int = len(self._nulls)

        if n_nulls == 0:
            # No nulls → decision is always -1, so β ≡ 0.
            if n_alternatives == 1:
                return 0.0
            return np.zeros(shape=(n_alternatives,), dtype=FloatDType)

        betas: FloatArray = np.empty(shape=(n_alternatives,), dtype=FloatDType)

        for alternative_idx in range(n_alternatives):
            decision_probs: FloatArray = self._backend.decision_distribution_on_hypothesis(
                prob_vector=prob_batch[alternative_idx, :],
                decision_fn=lambda h_batch: np.asarray(a=self.infer_decisions(hist_query=h_batch), dtype=IntDType),
                n_nulls=n_nulls,
            )
            betas[alternative_idx] = np.clip(a=decision_probs.sum(), a_min=0.0, a_max=1.0)

        if n_alternatives == 1:
            return float(betas[0])
        return betas

    def get_fwer(self) -> float:
        """
        Returns the actual Family-Wise Error Rate (FWER) of the Multi-Null JSd test, i.e., the probability of making at
        least one Type-I error when any of the null hypotheses is true.

        Returns
        -------
        float
            The actual FWER of the Multi-Null JSd test.
        """
        n_nulls: int = len(self._nulls)
        if n_nulls == 0:
            return 0.0  # No nulls → no Type-I errors are possible.
        return float(np.max(self.get_alpha(null_index=list(range(1, n_nulls + 1)))))

    def __repr__(self) -> str:
        representation: str = (
            f"MNSquaredTest(n={self._n}, k={self._k}, cdf_method={self._cdf_method!r}, n_nulls={len(self._nulls)}"
        )
        if self._mc_samples is not None and self._seed is not None:
            representation += f", mc_samples={self._mc_samples}, seed={self._seed}"
        representation += ")"

        return representation
