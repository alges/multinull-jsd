"""
Monte-Carlo CDF backend that draws multinomial histograms.
"""
from .base import CDFBackend

from multinull_jsd._jsd_distance import jsd
from multinull_jsd._validators import validate_int_value, validate_probability_vector
from multinull_jsd.types import FloatDType, FloatArray, CDFCallable

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

    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        prob_vector = validate_probability_vector(
            name="prob_vector", value=prob_vector, n_categories=None
        ).astype(dtype=FloatDType, copy=False)

        cdf_key: tuple[float, ...] = self._prob_vector_to_key(prob_vector=prob_vector)
        if cdf_key in self._cdf_cache:
            return self._cdf_cache[cdf_key]

        histogram_array = self._rng.multinomial(n=self._evidence_size, pvals=prob_vector, size=self._mc_samples)
        distances: FloatArray = jsd(
            p=prob_vector,
            q=histogram_array.astype(dtype=FloatDType, copy=False) / self._evidence_size
        )

        cdf_callable: CDFCallable = self._build_cdf_from_samples(distances=distances, weights=None)
        self._cdf_cache[cdf_key] = cdf_callable
        return cdf_callable

    def __repr__(self) -> str:
        return (
            f"MultinomialMCCDFBackend"
            f"(evidence_size={self.evidence_size}, mc_samples={self._mc_samples}, seed={self._seed})"
        )
