from cdf_backends import CDFBackend

import numpy.typing as npt
import numpy as np


class IndexedHypotheses:

    def __init__(self, cdf_backend: CDFBackend) -> None:
        pass

    def add_null(self, prob_vector: npt.NDArray[np.float64], target_alpha: float) -> int:
        pass

    def __getitem__(self, index):
        pass

    def __delitem__(self, idx):
        pass

    def __contains__(self, idx):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass
