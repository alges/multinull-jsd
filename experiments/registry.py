"""
Method registry for experiments.

Defines canonical method name strings and maps baseline method labels to their single-null p-value functions. Also
defines the derived method labels for the ExactMultinom multi-statistic backend and exposes a partially-applied
callable that returns `[Prob, Chisq, LLR]` p-values in one call.
"""
from typing import Callable

import functools

from baselines.power_divergence import chisquare_gof_pvalue, gtest_gof_pvalue
from baselines.exact_multinom_r import exact_multinom_pvalues
from baselines.kernel_mmd import mmd_gaussian_pvalue, mmd_laplacian_pvalue

from settings import M_MONTE_CARLO, FloatArray, IntArray


METHOD_JSD_PREFIX: str = "MNSquared"
METHOD_JSD: str = METHOD_JSD_PREFIX

BASELINE_SINGLE: dict[str, Callable[[IntArray, FloatArray], float]] = {
    "Chi2-Pearson+Holm": chisquare_gof_pvalue,
    "G-test-LLR+Holm": gtest_gof_pvalue,
    "MMD-Gaussian+Holm": functools.partial(mmd_gaussian_pvalue, reps=M_MONTE_CARLO),
    "MMD-Laplacian+Holm": functools.partial(mmd_laplacian_pvalue, reps=M_MONTE_CARLO),
}

EXACT_PREFIX: str = "ExactMultinom"
EXACT_STATS: list[str] = ["Prob", "Chisq", "LLR"]
EXACT_METHODS: list[str] = [f"{EXACT_PREFIX}-{s}+Holm" for s in EXACT_STATS]

exact_pvals_fn: Callable[[IntArray, FloatArray], FloatArray] = functools.partial(
    exact_multinom_pvalues,
    method="Monte-Carlo",
    n_mc=M_MONTE_CARLO,
)
