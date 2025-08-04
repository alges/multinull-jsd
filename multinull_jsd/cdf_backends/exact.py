from .base import CDFBackend

from multinull_jsd.types import FloatArray, CDFCallable


class ExactCDFBackend(CDFBackend):

    def __init__(self, evidence_size: int):
        super().__init__(evidence_size)

    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        pass
