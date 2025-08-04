from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np


class CDFBackend(ABC):

    def __init__(self, evidence_size: int) -> None:
        pass

    @abstractmethod
    def get_cdf(self, prob_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass
