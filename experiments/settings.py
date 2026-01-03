"""
Global settings, shared types, and default configuration.

This module centralizes lightweight, import-safe constants that are reused across the codebase, including:

- Common numpy typing aliases (`FloatArray`, `IntArray`).
- Conventional decision labels (e.g., `REJECT_DECISION`).
- Default parallelism knobs used by baseline routines (jobs, chunks, backend, tqdm).
- A global reproducibility seed and a default RNG (`rng_global`).
- Default Monte-Carlo and sampling sizes for experiments.

Notes
-----
- Because this module is imported broadly, keep it dependency-light and free of side effects beyond defining constants.
- `rng_global` is provided as a convenience for quick experiments; for strict
  reproducibility across functions, prefer passing explicit seeds/RNGs through APIs.
"""
from typing import Literal

import numpy.typing as npt
import numpy as np

REJECT_DECISION: int = -1
"""
Conventional numeric label for the "reject all null hypotheses" decision.

All utilities that work with decision labels assume that this constant represents the decision in favor of the
alternative hypothesis H1, i.e., "none of the null hypotheses H0^ℓ is accepted".
"""

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Global parallelism parameters (for baselines)
N_JOBS: int = 50
N_CHUNKS: int = 50
PARALLEL_BACKEND: Literal["serial", "threads", "processes"] = "processes"
SHOW_PROGRESS: bool = False

# Global reproducibility seed
GLOBAL_SEED: int = 1234
rng_global: np.random.Generator = np.random.default_rng(seed=GLOBAL_SEED)

# Monte Carlo parameters
M_MONTE_CARLO: int = 10_000
M_HIST_NULL: int = 10
M_HIST_ALT: int = 10
M_ALTERNATIVE_SAMPLES: int = 100_000
