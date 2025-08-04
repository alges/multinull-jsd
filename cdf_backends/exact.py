from .base import CDFBackend

import numpy.typing as npt
import numpy as np


class ExactCDFBackend(CDFBackend):

    def __init__(self, evidence_size: int):
        super().__init__(evidence_size)

    def get_cdf(self, prob_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass
