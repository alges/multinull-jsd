from typing import Literal

import numpy.typing as npt
import numpy as np


CDFBackendName = Literal["exact", "mc_multinomial", "mc_normal"]
FloatArray = npt.NDArray[np.float64]
