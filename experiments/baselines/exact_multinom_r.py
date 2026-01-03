"""
Exact multinomial goodness-of-fit p-values via R's ExactMultinom (optional).

This module provides a thin rpy2-based wrapper around the R package `ExactMultinom`, exposing p-values for the three
statistics returned by `ExactMultinom::multinom.test`:

- "Prob": probability-mass ordering
- "Chisq": Pearson chi-square
- "LLR": log-likelihood ratio (G-test)

It also handles "structural zeros" in the null probability vector by either returning zero p-values for impossible
observations (h_i > 0 when p_i == 0), or removing zero-probability categories with zero observed counts.

Notes
-----
- This module is an optional dependency: it requires both R and `rpy2`.
- In rpy2, the R function `multinom.test` is accessed as `multinom_test`.
"""
from __future__ import annotations

from typing import Literal, Final, Any

import numpy.typing as npt
import numpy as np

from experiments.settings import FloatArray, IntArray

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
except ImportError as exc:  # pragma: no cover - optional dependency
    ro = None
    importr = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


ExactMultinomStat = Literal["Prob", "Chisq", "LLR"]
ExactMultinomMethod = Literal["exact", "asymptotic", "Monte-Carlo"]

# ExactMultinom R package convention:
#   index 0 -> "Prob"
#   index 1 -> "Chisq"
#   index 2 -> "LLR"
_STAT_ORDER: Final[tuple[str, str, str]] = ("Prob", "Chisq", "LLR")
_STAT_TO_INDEX: Final[dict[str, int]] = {"Prob": 0, "Chisq": 1, "LLR": 2}

_EXACT_MULTINOM_R_PKG: Any | None = None


def _get_exact_multinom_pkg() -> Any:
    """
    Lazily import and cache the R package handle.

    This avoids repeatedly calling `importr("ExactMultinom")` inside loops.
    """
    global _EXACT_MULTINOM_R_PKG
    if _EXACT_MULTINOM_R_PKG is None:
        assert importr is not None
        _EXACT_MULTINOM_R_PKG = importr(name="ExactMultinom")
    return _EXACT_MULTINOM_R_PKG


def _sanitize_structural_zeros(
    histogram: IntArray,
    p_null: FloatArray,
    zero_tol: float = 0.0,
) -> tuple[IntArray, FloatArray, bool]:
    """
    Handle structural zeros in the null probabilities.

    ExactMultinom requires strictly positive probabilities. When p_i == 0:

    - If h_i > 0, the observed outcome has probability 0 under H0, so p-values should be 0 for all reasonable GOF
      statistics (Prob/Chisq/LLR).
    - If h_i == 0, the category is deterministically zero and can be removed.

    Parameters
    ----------
    histogram:
        Histogram of observed counts (k,).
    p_null:
        Null hypothesis probability vector (k,).
    zero_tol:
        Zero tolerance: probabilities <= this value are treated as zeros.

    Returns
    -------
    A tuple of three elements: h_sanitized, p_sanitized, impossible_event If `impossible_event` is True, the caller
    should return p-values = 0.
    """
    h: IntArray = np.asarray(a=histogram, dtype=np.int64)
    p: FloatArray = np.asarray(a=p_null, dtype=np.float64)

    if zero_tol < 0.0:
        raise ValueError(f"`zero_tol` must be >= 0; got {zero_tol}.")

    # Treat tiny values as zeros if requested.
    zero_mask: npt.NDArray[np.bool_] = p <= float(zero_tol)

    if np.any(a=zero_mask):
        # If we observed positive counts where p=0 => impossible under H0.
        if np.any(a=h[zero_mask] > 0):
            return h, p, True

        # Drop zero-probability categories (all have h_i == 0 here).
        keep_mask: npt.NDArray[np.bool_] = ~zero_mask
        h2: IntArray = h[keep_mask]
        p2: FloatArray = p[keep_mask]

        if p2.size == 0 or float(np.sum(a=p2)) <= 0.0:
            raise ValueError("After removing zero-probability categories, `p_null` has no positive mass.")

        # Re-normalize for numerical stability (ExactMultinom allows non-normalized proportions, but normalization is
        # harmless and often helps downstream).
        p2 = p2 / float(np.sum(p2))
        return h2, p2, False

    return h, p, False


def exact_multinom_pvalues(
    histogram: IntArray,
    p_null: FloatArray,
    method: Literal["exact", "asymptotic", "Monte-Carlo"] = "exact",
    theta: float = 1e-4,
    timelimit: float = 10.0,
    n_mc: int = 10_000,
    zero_tol: float = 0.0,
) -> FloatArray:
    """
    One-sample exact multinomial goodness-of-fit p-value using R's ExactMultinom.

    This is a thin wrapper around the R function `ExactMultinom::multinom.test`, called via rpy2.

    Notes
    -----
    The ExactMultinom documentation states that each method computes p-values for all three test statistics
    simultaneously, and exposes them as pvals_ex, pvals_as, or pvals_mc, where:
      - index 0 -> Prob (probability mass ordering)
      - index 1 -> Chisq (Pearson chi-square)
      - index 2 -> LLR (log-likelihood ratio)  (see package manual)

    Structural zeros:
    - ExactMultinom requires strictly positive probabilities. If `p_null` has zeros:
        * if h_i > 0 where p_i == 0 => return [0, 0, 0]
        * else drop those categories and re-normalize p_raw over the positive support.

    Parameters
    ----------
    histogram:
        Observed counts (k,).
    p_null:
        Null probability vector p_ℓ (k,).
    method:
        Method to obtain p-values in ExactMultinom:
        - "exact"
        - "asymptotic"
        - "Monte-Carlo"
    theta:
        Truncation threshold θ for rare configurations (see ExactMultinom docs).
    timelimit:
        Time limit in seconds for the exact enumeration or Monte Carlo.
    n_mc:
        Number of Monte Carlo samples when `method == "Monte-Carlo"`.
    zero_tol:
        Zero tolerance: probabilities <= this value are treated as zeros.

    Returns
    -------
    FloatArray of shape (3,) in the order [Prob, Chisq, LLR]

    Raises
    ------
    RuntimeError
        If rpy2 or ExactMultinom are not available or if ExactMultinom fails.
    ValueError
        If inputs are invalid.
    """
    if _IMPORT_ERROR is not None or ro is None or importr is None:
        raise RuntimeError(
            "rpy2 is not available. Install it and ensure R is installed to use the ExactMultinom baseline."
        ) from _IMPORT_ERROR

    h_raw: IntArray = np.asarray(a=histogram, dtype=np.int64)
    p_raw: FloatArray = np.asarray(a=p_null, dtype=np.float64)

    if h_raw.ndim != 1 or p_raw.ndim != 1:
        raise ValueError("`histogram` and `p_null` must be one-dimensional.")
    if h_raw.shape[0] != p_raw.shape[0]:
        raise ValueError(f"Dimension mismatch: len(histogram)={h_raw.shape[0]}, len(p_null)={p_raw.shape[0]}.")
    if np.any(a=h_raw < 0):
        raise ValueError("`histogram` must be non-negative.")
    if np.any(a=p_raw < 0):
        raise ValueError("`p_null` must be non-negative.")

    h, p, impossible = _sanitize_structural_zeros(histogram=h_raw, p_null=p_raw, zero_tol=zero_tol)
    if impossible:
        return np.zeros(shape=(3,), dtype=np.float64)

    exact_multinom_r_pkg = _get_exact_multinom_pkg()

    x_r = ro.IntVector(list(map(int, h)))
    p_r = ro.FloatVector(list(map(float, p)))

    # Call R function: ExactMultinom::multinom.test(x, p_raw, ...)
    res = exact_multinom_r_pkg.multinom_test(
        x=x_r,
        p=p_r,
        stat="Prob",  # stat is required by the R signature, but p-values for all 3 stats are returned
        method=method,
        theta=theta,
        timelimit=timelimit,
        N=n_mc,
    )

    # Choose the appropriate p-value vector based on the method
    method_lower: str = method.lower()
    if method_lower.startswith("exact"):
        pvec_name: str = "pvals_ex"
    elif method_lower.startswith("asymp"):
        pvec_name = "pvals_as"
    elif method_lower.startswith("monte"):
        pvec_name = "pvals_mc"
    else:
        raise ValueError(f"Unsupported method {method!r}. Must be one of 'exact', 'asymptotic', or 'Monte-Carlo'.")

    # Extract the p-value vector for the chosen method
    pvec_r = res.rx2(pvec_name)
    pvec: FloatArray = np.asarray(a=pvec_r, dtype=np.float64)

    if pvec.shape[0] < 3:
        raise RuntimeError(f"ExactMultinom returned p-value vector of length {pvec.shape[0]} (expected 3).")
    if not np.all(a=np.isfinite(pvec[:3])):
        raise RuntimeError(f"ExactMultinom returned non-finite p-values: {pvec[:3]}.")

    return np.asarray(a=pvec[:3], dtype=np.float64)
