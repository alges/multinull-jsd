from multinull_jsd.types import FloatArray, CDFCallable
from abc import ABC, abstractmethod


class CDFBackend(ABC):

    def __init__(self, evidence_size: int) -> None:
        pass

    @abstractmethod
    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        pass
