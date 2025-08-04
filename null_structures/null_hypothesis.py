from cdf_backends import CDFBackend

from typing import Any

import numpy.typing as npt
import numpy as np


class NullHypothesis:

    def __init__(self, prob_vector: npt.NDArray[np.float64], cdf_backend: CDFBackend) -> None:
        pass

    def set_target_alpha(self, target_alpha: float) -> None:
        pass

    def get_jsd_threshold(self) -> float:
        pass

    def infer_p_value(self, query: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    def __eq__(self, other: Any) -> bool:
        pass

    def __repr__(self) -> str:
        pass
