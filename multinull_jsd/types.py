from typing import Literal, Protocol, runtime_checkable, overload

import numpy.typing as npt
import numpy as np


CDFBackendName = Literal["exact", "mc_multinomial", "mc_normal"]
FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class CDFCallable(Protocol):
    @overload
    def __call__(self, tau: float) -> float: ...
    @overload
    def __call__(self, tau: npt.ArrayLike) -> FloatArray: ...

    def __call__(self, tau: float | npt.ArrayLike) -> float | npt.ArrayLike: ...
