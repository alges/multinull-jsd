from .base import CDFBackend

import numpy.typing as npt
import numpy as np


class NormalMCCDFBackend(CDFBackend):
    def __init__(self, evidence_size: int, mc_samples: int, seed: int):
        super().__init__(evidence_size)
        # TODO: Incorporate Monte-Carlo elements

    def get_cdf(self, prob_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass
