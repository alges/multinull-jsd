"""
MMD-based multinomial goodness-of-fit baselines using synthetic null samples.

The multinomial GOF problem is turned into a two-sample test:
1) Expand an observed histogram into a sample of categorical indices.
2) Draw a synthetic sample from the null probability vector.
3) Apply `hyppo.ksample.MMD` with a chosen kernel and permutation testing.

This is an approximate baseline: the null distribution is estimated via permutation (`reps`), and the one-sample GOF is
reduced to a two-sample test. A process-level RNG is provided to diversify random streams across workers.
"""
from __future__ import annotations

from hyppo.ksample import MMD  # hyppo.ksample.MMD.test returns (stat, pvalue)
from typing import Literal

import numpy.typing as npt
import numpy as np
import os

from experiments.settings import FloatArray, IntArray


_PROCESS_RNG: np.random.Generator | None = None


def _get_process_rng() -> np.random.Generator:
    """
    Get a process-level numpy random number generator with a random seed. This implementation is still not
    reproducible across processes, but it's good enough for our purposes.

    Returns
    -------
    rng:
        Numpy random number generator.
    """
    global _PROCESS_RNG
    if _PROCESS_RNG is None:
        # Different processes will have different PIDs, so this is a simple way
        # to diversify streams across workers.
        seed = (os.getpid() * 1_000_003) ^ int.from_bytes(os.urandom(8), "little")
        _PROCESS_RNG = np.random.default_rng(seed=seed)
    return _PROCESS_RNG

def _expand_histogram_to_samples(histogram: IntArray) -> FloatArray:
    """
    Expand a histogram into a vector of categorical samples {0, …, k-1}.

    Parameters
    ----------
    histogram:
        One-dimensional integer array of shape (k,) with counts.

    Returns
    -------
    samples:
        Two-dimensional float array of shape (n, 1) where n = sum(histogram), containing one sample per individual
        observation, encoded as its category index.
    """
    h: IntArray = np.asarray(a=histogram, dtype=np.int64)
    if h.ndim != 1:
        raise ValueError(f"`histogram` must be one-dimensional; got shape {h.shape}.")

    n: int = int(h.sum())
    if n <= 0:
        raise ValueError("Total count n must be positive to run an MMD test.")

    categories: npt.NDArray = np.arange(h.shape[0], dtype=np.int64)
    samples_flat: npt.NDArray = np.repeat(a=categories, repeats=h)
    # MMD expects shape (n, p); here p = 1.
    return samples_flat.reshape(-1, 1).astype(dtype=np.float64, copy=False)


def _sample_from_null_probabilities(
    p_null: FloatArray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """
    Draw a sample of size n_samples from a discrete distribution p_null.

    Parameters
    ----------
    p_null:
        Null probability vector of shape (k,).
    n_samples:
        Number of samples to draw.
    rng:
        Optional numpy RNG. If None, `np.random.default_rng()` is used.

    Returns
    -------
    samples:
        Two-dimensional float array of shape (n_samples, 1) with category indices.
    """
    p: FloatArray = np.asarray(a=p_null, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"`p_null` must be one-dimensional; got shape {p.shape}.")

    if n_samples <= 0:
        raise ValueError(f"`n_samples` must be >= 1; got {n_samples}.")

    rng_obj: np.random.Generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
    categories: npt.NDArray = np.arange(p.shape[0], dtype=np.int64)

    draws: npt.NDArray = rng_obj.choice(a=categories, size=int(n_samples), replace=True, p=p)
    return draws.reshape(-1, 1).astype(dtype=np.float64, copy=False)


def mmd_multinomial_pvalue(
    histogram: IntArray,
    p_null: FloatArray,
    compute_kernel: Literal[
        "gaussian",
        "rbf",
        "laplacian",
        "linear",
        "poly",
        "polynomial",
        "cosine",
        "sigmoid",
    ] = "gaussian",
    n_synth: int | None = None,
    reps: int = 1_000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    MMD-based p-value for testing a multinomial null using synthetic null samples.

    The idea is:

    1. Expand the observed histogram into n samples x_array ∈ {0, …, k-1}.
    2. Draw n_synth samples y_array from the null pmf p_null.
    3. Run a 2-sample MMD test (hyppo.ksample.MMD) between x_array and y_array.

    Parameters
    ----------
    histogram:
        Observed counts (k,).
    p_null:
        Null probability vector p_ℓ (k,).
    compute_kernel:
        Kernel name passed to `hyppo.ksample.MMD`. Valid options include "gaussian", "rbf", "laplacian", "linear",
        "poly", "polynomial", "cosine", "sigmoid". See hyppo documentation for details.
    n_synth:
        Number of synthetic null samples. If None, uses n_synth = sum(histogram).
    reps:
        Number of permutation replications used by hyppo to estimate the null distribution of the MMD statistic.
    rng:
        Optional numpy RNG used for synthetic sampling from p_null.

    Returns
    -------
    p_value:
        MMD-based 2-sample p-value for H₀: "multinomial with probabilities p_null".
    """
    x_array: FloatArray = _expand_histogram_to_samples(histogram=histogram)
    n_obs: int = x_array.shape[0]
    n_synth_eff: int = n_obs if n_synth is None else int(n_synth)

    y_array: FloatArray = _sample_from_null_probabilities(p_null=p_null, n_samples=n_synth_eff, rng=rng)

    if np.var(a=np.vstack(tup=[np.asarray(a=x_array), np.asarray(a=y_array)])) == 0.0:
        return 1.0

    mmd: MMD = MMD(compute_kernel=compute_kernel)
    _, p_value = mmd.test(x=x_array, y=y_array, reps=reps, workers=1)
    return float(p_value)


def mmd_gaussian_pvalue(histogram: IntArray, p_null: FloatArray, reps: int) -> float:
    """
    Dedicated wrapper for MMD with Gaussian (RBF) kernel.

    Parameters
    ----------
    histogram:
        Observed counts (k,).
    p_null:
        Null probability vector p_ℓ (k,).
    reps:
        Number of permutation replications used by hyppo to estimate the null distribution of the MMD
        statistic.

    Returns
    -------
    MMD-based 2-sample p-value for H₀: "multinomial with probabilities p_null".
    """
    rng: np.random.Generator = _get_process_rng()
    return float(
        mmd_multinomial_pvalue(
            histogram=histogram,
            p_null=p_null,
            compute_kernel="gaussian",
            reps=reps,
            rng=rng,
        )
    )


def mmd_laplacian_pvalue(histogram: IntArray, p_null: FloatArray, reps: int) -> float:
    """
    Dedicated wrapper for MMD with Laplacian kernel.

    Parameters
    ----------
    histogram:
        Observed counts (k,).
    p_null:
        Null probability vector p_ℓ (k,).
    reps:
        Number of permutation replications used by hyppo to estimate the null distribution of the MMD
        statistic.

    Returns
    -------
    MMD-based 2-sample p-value for H₀: "multinomial with probabilities p_null".
    """
    rng: np.random.Generator = _get_process_rng()
    return float(
        mmd_multinomial_pvalue(
            histogram=histogram,
            p_null=p_null,
            compute_kernel="laplacian",
            reps=reps,
            rng=rng,
        )
    )
