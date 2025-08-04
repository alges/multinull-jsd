from .base import CDFBackend

from multinull_jsd.types import FloatArray, CDFCallable


class MultinomialMCCDFBackend(CDFBackend):
    def __init__(self, evidence_size: int, mc_samples: int, seed: int):
        super().__init__(evidence_size)
        # TODO: Incorporate Monte-Carlo elements

    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        pass
