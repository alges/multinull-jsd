from .types import CDFBackendName

from null_structures import IndexedHypotheses
from typing import Optional, Union, Sequence

import numpy.typing as npt


class MultiNullJSDTest:

    def __init__(
        self, evidence_size: int, prob_dim: int, cdf_method: CDFBackendName = "exact",
        mc_samples: Optional[int] = None, seed: Optional[int] = None
    ) -> None:
        pass

    def add_nulls(self, prob_vector: npt.ArrayLike, target_alpha: Union[float, Sequence[float]]) -> None:
        pass

    def remove_nulls(self, null_index: Union[int, Sequence[int]]) -> None:
        pass

    def get_nulls(self) -> IndexedHypotheses:
        pass

    def infer_p_values(self, query: npt.ArrayLike) -> npt.ArrayLike:
        pass

    def infer_decisions(self, query: npt.ArrayLike) -> npt.NDArray[int]:
        pass

    def get_alpha(self, null_index: Union[int, Sequence[int]]) -> float:
        pass

    def get_beta(self, query: npt.ArrayLike) -> float:
        pass

    def get_fwer(self) -> float:
        pass

    def __repr__(self) -> str:
        pass
