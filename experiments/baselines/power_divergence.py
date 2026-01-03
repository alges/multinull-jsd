"""
Power-divergence multinomial GOF baselines (asymptotic chi-square calibration).

Provides single-null p-values for:
- Pearson chi-square (lambda_="pearson")
- Log-likelihood ratio / G-test (lambda_="log-likelihood")

These rely on SciPy's `scipy.stats.power_divergence` and therefore use asymptotic chi-square calibration. Numerical
warnings (e.g., divide-by-zero when expected counts contain zeros) are suppressed locally via `np.errstate`.
"""
from __future__ import annotations

from scipy.stats import power_divergence

import numpy as np

from experiments.settings import FloatArray, IntArray


def _expected_counts_from_prob(histogram: IntArray, p_null: FloatArray) -> FloatArray:
    """
    Compute expected counts vector under a null probability vector p_null.

    Parameters
    ----------
    histogram:
        Observed histogram (counts) of shape (k,).
    p_null:
        Null probability vector of shape (k,).

    Returns
    -------
    Expected counts n * p_null, where n = sum(histogram).
    """
    h: IntArray = np.asarray(a=histogram, dtype=np.int64)
    p: FloatArray = np.asarray(a=p_null, dtype=np.float64)

    if h.ndim != 1 or p.ndim != 1:
        raise ValueError("Both `histogram` and `p_null` must be one-dimensional.")
    if h.shape[0] != p.shape[0]:
        raise ValueError(f"Dimension mismatch: len(histogram)={h.shape[0]}, len(p_null)={p.shape[0]}.")

    n: int = int(h.sum())
    if n <= 0:
        raise ValueError("Total count n must be positive for power-divergence tests.")

    return (float(n) * p).astype(dtype=np.float64, copy=False)


def chisquare_gof_pvalue(histogram: IntArray, p_null: FloatArray) -> float:
    """
    One-sample χ² goodness-of-fit p-value against a multinomial null.

    This uses SciPy's `power_divergence` with λ = 1 (Pearson's χ²) and the usual χ² asymptotic calibration.

    Parameters
    ----------
    histogram:
        Observed counts (k,).
    p_null:
        Null probability vector p_ℓ (k,).

    Returns
    -------
    p_value:
        Asymptotic χ² p-value for H₀^ℓ: "multinomial with probabilities p_null".
    """
    h: IntArray = np.asarray(a=histogram, dtype=np.int64)
    expected: FloatArray = _expected_counts_from_prob(histogram=h, p_null=p_null)

    # SciPy's power_divergence defaults: lambda_ = 'pearson' → χ² test.
    with np.errstate(divide="ignore", invalid="ignore"):
        stat, p_value = power_divergence(f_obs=h, f_exp=expected, lambda_="pearson")
    return float(p_value)


def gtest_gof_pvalue(histogram: IntArray, p_null: FloatArray) -> float:
    """
    One-sample G-test (log-likelihood ratio) p-value against a multinomial null.

    This uses SciPy's `power_divergence` with λ = 0 ("log-likelihood ratio").

    Parameters
    ----------
    histogram:
        Observed counts (k,).
    p_null:
        Null probability vector p_ℓ (k,).

    Returns
    -------
    p_value:
        Asymptotic G-test p-value for H₀^ℓ: "multinomial with probabilities p_null".
    """
    h: IntArray = np.asarray(a=histogram, dtype=np.int64)
    expected: FloatArray = _expected_counts_from_prob(histogram=h, p_null=p_null)

    # lambda_ = 0 corresponds to the G-test (likelihood ratio).
    with np.errstate(divide="ignore", invalid="ignore"):
        stat, p_value = power_divergence(f_obs=h, f_exp=expected, lambda_="log-likelihood")
    return float(p_value)
