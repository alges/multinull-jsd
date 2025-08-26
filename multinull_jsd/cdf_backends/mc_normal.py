from .base import CDFBackend

from multinull_jsd._validators import validate_int_value, validate_probability_vector
from multinull_jsd.types import FloatArray, CDFCallable


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
        validate_int_value(name="mc_samples", value=mc_samples, min_value=1)
        validate_int_value(name="seed", value=seed, min_value=0)
        # TODO: Incorporate Monte-Carlo elements
        raise NotImplementedError

    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        validate_probability_vector(name="prob_vector", value=prob_vector, n_categories=None)
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
