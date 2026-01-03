"""
Alternative (H1) construction utilities for multi-null experiments.

Alternatives are probability vectors q in the simplex designed to achieve specified target values of
mJSd(q) = min_ℓ JSd(q, p_ℓ) relative to a fixed set of null probability vectors (p_ℓ).

The main workflow:
- Sample a large pool of candidate q vectors (global Dirichlet + local mixtures).
- Compute each candidate's mJSd to the null set.
- Select, without replacement, candidates whose mJSd values best match a set of target levels.
"""
from typing import Any, Callable

import numpy.typing as npt
import pandas as pd
import numpy as np

from scenarios import sample_dirichlet_probabilities
from settings import FloatArray, IntArray
from metrics import jensen_shannon_distance, minimum_js_distance
from utils import make_generator


def build_alternatives_df_by_mjsd_targets(
    null_probabilities: FloatArray,
    mjsd_targets: FloatArray,
    num_candidate_samples: int,
    dirichlet_alpha: float | FloatArray,
    rng: int | np.random.Generator | None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a DataFrame with alternative probability vectors (q) that approximately achieve given mJSd targets.

    Parameters
    ----------
    null_probabilities:
        Two-dimensional array of shape (L, k) with the null base probabilities (p_ℓ)_{ℓ=1}^L.
    mjsd_targets:
        One-dimensional array of target mJSd values for the alternatives.
    num_candidate_samples:
        Number of candidate alternative probability vectors to sample when building alternatives for this scenario.
    dirichlet_alpha:
        Dirichlet alpha parameter for alternative probability sampling.
    rng:
        RNG seed or generator for sampling alternative probabilities.
    verbose:
        Print progress to stdout.

    Returns
    -------
    DataFrame with columns:
    - "alt_id": Unique alternative ID (1-based).
    - "mjsd_target": Target mJSd value.
    - "mjsd": Achieved minimum JS distance to nulls.
    - "mjsd_error": Difference between achieved and target mJSd.
    - "closest_null": Index (1-based) of the closest null hypothesis.
    - "q_vector": Alternative probability vector (array of shape (k,)).
    """
    rng_obj: np.random.Generator = make_generator(seed_or_rng=rng)
    null_probabilities = np.asarray(a=null_probabilities, dtype=np.float64)

    targets: FloatArray = np.asarray(a=mjsd_targets, dtype=np.float64).reshape(-1)
    if targets.size == 0:
        raise ValueError("`mjsd_targets` must be non-empty.")

    q_mat: FloatArray = sample_alternative_probabilities(
        null_probabilities=null_probabilities,
        mjsd_targets=targets,
        num_candidate_samples=int(num_candidate_samples),
        dirichlet_alpha=dirichlet_alpha,
        jsd_fn=jensen_shannon_distance,
        rng=rng_obj,
    )  # (T, k)

    rows: list[dict[str, Any]] = []
    for alt_id, (target, q) in enumerate(zip(targets, q_mat, strict=True), start=1):
        mjsd, argmin0 = minimum_js_distance(
            candidates=q,
            null_probabilities=null_probabilities,
            jsd_fn=jensen_shannon_distance,
        )
        mjsd_val: float = float(mjsd.item())
        err: float = float(mjsd_val - target)
        closest_null: int = int(argmin0.item() + 1)  # 1-based
        if verbose:
            print(
                f"[alts] alt_id={alt_id:02d} | target={float(target):.3f} | "
                f"achieved={mjsd_val:.6f} | err={err:+.6f} | closest_null={closest_null}"
        )
        rows.append(
            {
                "alt_id": int(alt_id),
                "mjsd_target": float(target),
                "mjsd": mjsd_val,
                "mjsd_error": err,
                "closest_null": closest_null,
                "q_vector": q,
            }
        )

    return pd.DataFrame(data=rows)


def sample_alternative_probabilities(
    null_probabilities: FloatArray,
    mjsd_targets: FloatArray,
    num_candidate_samples: int,
    dirichlet_alpha: float | FloatArray = 1.0,
    jsd_fn: Callable[[FloatArray, FloatArray], FloatArray] | None = None,
    local_factor_beta: tuple[float, float] = (0.5, 0.5),
    rng: int | np.random.Generator | None = None,
) -> FloatArray:
    """
    Sample a pool of M candidates q_i ~ Dirichlet(alpha), compute mJSd(q_i) for all, then greedily select distinct
    candidates whose mJSd values are closest to the given targets (without replacement).

    Parameters
    ----------
    null_probabilities:
        Two-dimensional numpy array of shape (num_nulls, k) containing the null probability vectors. The mJSd is
        computed with respect to these rows.
    mjsd_targets:
        One-dimensional array of shape (num_alternatives,) containing target mJSd values for each alternative.
        If provided, the function attempts to select alternatives whose mJSd values are as close as possible to these
        targets.
    num_candidate_samples:
        This parameter specifies the number of candidate samples to draw from the Dirichlet proposal distribution
        before selecting alternatives.
    dirichlet_alpha:
        Concentration parameter(s) for the Dirichlet proposal distribution. See `sample_dirichlet_probabilities` for
        accepted shapes and semantics.
    jsd_fn:
        Optional distance function to use instead of `jensen_shannon_distance`. The signature must be::
            jsd_fn(p: FloatArray, q: FloatArray) -> FloatArray
    local_factor_beta:
        Beta distribution parameters for the local factor distribution. The first parameter controls the shape of the
        distribution near 0, and the second parameter controls the shape near 1.
    rng:
        Random number generator or seed. See `make_generator` for accepted values.

    Returns
    -------
    Two-dimensional numpy array of shape (num_alternatives, k) containing the accepted alternative probability vectors.

    Raises
    ------
    ValueError
        If shapes or parameters are invalid.
    """
    nulls: FloatArray = np.asarray(a=null_probabilities, dtype=np.float64)
    if nulls.ndim != 2:
        raise ValueError(f"`null_probabilities` must be a 2D array (L,k); got shape {nulls.shape}.")

    num_nulls, k = nulls.shape
    if num_nulls <= 0:
        raise ValueError("Number of null hypotheses num_nulls must be >= 1.")
    if k <= 0:
        raise ValueError("Number of categories k must be >= 1.")

    targets: FloatArray = np.asarray(a=mjsd_targets, dtype=np.float64).reshape(-1)
    if targets.size == 0:
        raise ValueError("`mjsd_targets` must be non-empty.")

    rng_obj: np.random.Generator = make_generator(seed_or_rng=rng)
    jsd: Callable[[FloatArray, FloatArray], FloatArray] = jensen_shannon_distance if jsd_fn is None else jsd_fn

    targets = np.asarray(a=mjsd_targets, dtype=np.float64).reshape(-1)
    if targets.size == 0:
        raise ValueError("`mjsd_targets` must be non-empty.")

    # Global candidate pool
    num_candidates: int = int(num_candidate_samples)
    if num_candidates <= 0:
        raise ValueError(f"`num_candidate_samples` must be >= 1; got {num_candidates}.")

    # Global candidate pool
    q_global: FloatArray = sample_dirichlet_probabilities(
        alpha=dirichlet_alpha,
        num_categories=k,
        num_samples=num_candidates,
        rng=rng_obj,
    )  # (M,k)

    # Local candidate pool
    local_null_idx: IntArray = rng_obj.integers(low=0, high=num_nulls, size=num_candidates, dtype=np.int64)
    local_factors: FloatArray = rng_obj.beta(
        a=local_factor_beta[0],
        b=local_factor_beta[1],
        size=num_candidates
    ).astype(dtype=np.float64)
    local_factors = np.clip(a=local_factors, a_min=0.0, a_max=1.0)[:, None]
    local_randomness: FloatArray = sample_dirichlet_probabilities(
        alpha=dirichlet_alpha,
        num_categories=k,
        num_samples=num_candidates,
        rng=rng_obj
    )
    p_local: FloatArray = nulls[local_null_idx]  # shape (M,k)
    q_local: FloatArray = local_factors * p_local + (1.0 - local_factors) * local_randomness

    q_candidates: FloatArray = np.vstack(tup=(q_global, q_local))  # shape (2M,k)

    # Compute mJSd for all candidates
    mjsd_vals, _ = minimum_js_distance(candidates=q_candidates, null_probabilities=nulls, jsd_fn=jsd)
    mjsd_vals = np.asarray(a=mjsd_vals, dtype=np.float64)

    # Greedy “closest without replacement” selection
    available: npt.NDArray[np.bool_] = np.ones(shape=(q_candidates.shape[0],), dtype=np.bool_)
    chosen_idx: IntArray = np.empty(shape=(targets.size,), dtype=np.int64)

    for t_idx, target in enumerate(targets):
        diffs: FloatArray = np.abs(mjsd_vals - target)
        diffs = np.where(available, diffs, np.inf)
        idx: int = int(np.argmin(a=diffs))
        if not np.isfinite(diffs[idx]):
            raise RuntimeError("No available candidate left to match a target (unexpected).")
        chosen_idx[t_idx] = idx
        available[idx] = False

    return np.asarray(a=q_candidates[chosen_idx], dtype=np.float64)
