from multinull_jsd.cdf_backends import CDFBackend
from multinull_jsd.types import FloatArray

from typing import Any


class NullHypothesis:

    def __init__(self, prob_vector: FloatArray, cdf_backend: CDFBackend) -> None:
        pass

    def set_target_alpha(self, target_alpha: float) -> None:
        pass

    def get_jsd_threshold(self) -> float:
        pass

    def infer_p_value(self, query: FloatArray) -> FloatArray:
        pass

    def __eq__(self, other: Any) -> bool:
        pass

    def __repr__(self) -> str:
        pass
