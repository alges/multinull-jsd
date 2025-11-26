"""
Monte-Carlo CDF backend that draws multinomial histograms.
"""
from .base import CDFBackend

from multinull_jsd._validators import validate_int_value, validate_probability_vector
from multinull_jsd.types import FloatDType, FloatArray, IntArray, IntDType

import numpy as np


class MultinomialMCCDFBackend(CDFBackend):
    """
    Monte-Carlo estimator that draws **multinomial** histograms exactly from the provided probability vector and builds
    an empirical CDF.

    Parameters
    ----------
    evidence_size
        Number of samples :math:`n`.
    mc_samples
        Number of Monte-Carlo repetitions :math:`N`. Must be positive.
    seed
        Random-state seed for reproducibility.

    Notes
    -----
    The estimator satisfies the Strong Law of Large Numbers; hence, it converges to the exact CDF as
    :math:`N\\rightarrow\\infty`.
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
        histogram_array = self._rng.multinomial(
            n=self._evidence_size, pvals=prob_vector, size=self._mc_samples
        ).astype(dtype=IntDType, copy=False)
        histogram_probabilities: FloatArray = np.full(
            shape=self._mc_samples, fill_value=1.0 / self._mc_samples, dtype=FloatDType
        )
        return histogram_array, histogram_probabilities

    def __repr__(self) -> str:
        return (
            f"MultinomialMCCDFBackend"
            f"(evidence_size={self.evidence_size}, mc_samples={self._mc_samples}, seed={self._seed})"
        )
