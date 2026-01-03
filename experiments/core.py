"""
Experiment orchestration for multi-null JSd testing and baselines.

This module runs scenario-based Monte Carlo experiments comparing:
- `MultiNullJSDTest` (the proposed method), and
- Holm-based baselines (single-stat and multi-stat backends).

For each scenario, it:
1) samples a set of null probability vectors,
2) constructs matched alternatives at specified mJSd target levels,
3) runs each method across a union of method-specific n-grids using shared simulated histograms for fair comparisons,
4) records decision counts, empirical metrics, and runtime summaries, and
5) writes scenario artifacts under `results/` (CSV + cached nulls/alts, and optionally compressed histogram arrays).
"""
from multinull_jsd import MultiNullJSDTest
from typing import Any, Callable
from utils import make_generator

import pandas as pd
import numpy as np
import time

from baselines.multiple_testing import multinull_decisions_holm_batch, multinull_decisions_holm_batch_multistat

from alternatives import build_alternatives_df_by_mjsd_targets
from scenarios import (
    ExperimentScenario,
    NullSamplingConfig,
    SamplingPlan,
    _plan_lookup,
    sample_multinomial_histograms_for_nulls,
    sample_multinomial_histograms_for_null,
    generate_null_probabilities
)
from settings import N_JOBS, N_CHUNKS, PARALLEL_BACKEND, SHOW_PROGRESS, rng_global, FloatArray, IntArray
from registry import METHOD_JSD, BASELINE_SINGLE, EXACT_METHODS, EXACT_PREFIX, EXACT_STATS, exact_pvals_fn
from metrics import (
    decision_count_rows,
    evaluate_multinull_method_on_nulls,
    evaluate_multinull_method_on_alts,
    _metrics_rows_from_decisions,
)
from io_utils import _slug, _maybe_save_histograms, RESULTS_DIR


def _sample_shared_histograms_for_n(
    null_p: FloatArray,
    q_mat: FloatArray,
    n: int,
    m_null_max: int,
    m_alt_max: int,
    rng_sampling: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    """
    Sample shared multinomial histograms for all nulls and all alternatives at a fixed n.

    This helper draws, with a single RNG stream, the maximum number of histograms required
    for each null probability vector in `null_p`, and each alternative probability vector
    in `q_mat`.

    It is useful when multiple methods will reuse the same simulated evidence to ensure fair
    comparisons.

    Parameters
    ----------
    null_p:
        Array of shape (L, k) with null base probabilities (p_ℓ).
    q_mat:
        Array of shape (T, k) with alternative probability vectors (q).
    n:
        Sample size (number of observations per histogram).
    m_null_max:
        Number of histograms to draw per null hypothesis.
    m_alt_max:
        Number of histograms to draw per alternative hypothesis.
    rng_sampling:
        Numpy Generator used for all draws to keep simulations synchronized.

    Returns
    -------
    tuple of IntArray
        A pair (h0_all, hq_all) where:
        - h0_all has shape (L, m_null_max, k) with multinomial counts under each null.
        - hq_all has shape (T, m_alt_max, k) with multinomial counts under each alternative.

    Raises
    ------
    ValueError
        If input shapes are incompatible or parameters are invalid (as enforced by the
        underlying sampling utilities).
    """
    h0_all: IntArray = sample_multinomial_histograms_for_nulls(
        base_probabilities=null_p,
        num_observations=int(n),
        num_histograms_per_null=int(m_null_max),
        rng=rng_sampling,
    )
    hq_all: IntArray = sample_multinomial_histograms_for_nulls(
        base_probabilities=q_mat,
        num_observations=int(n),
        num_histograms_per_null=int(m_alt_max),
        rng=rng_sampling,
    )
    return h0_all, hq_all


def make_multinull_jsd_test(
    null_probabilities: FloatArray,
    alpha_vector: FloatArray,
    evidence_size: int,
    cdf_method: str = "exact",
    mc_samples: int | None = None,
    seed: int | None = None,
) -> MultiNullJSDTest:
    """
    Construct and configure a `MultiNullJSDTest` instance for a given set of null models.

    Parameters
    ----------
    null_probabilities:
        Two-dimensional array of shape (L, k) with the null base probabilities
        (p_ℓ)_{ℓ=1}^L.
    alpha_vector:
        One-dimensional array of shape (L,) with per-null target significance
        levels \bar{α}_ℓ. This is the \bar{\boldsymbol{α}} in the theory.
    evidence_size:
        Histogram size n (number of samples). Must match the sum of entries in
        each histogram that will be provided to the test.
    cdf_method:
        CDF computation backend. Must be one of:
        - "exact"
        - "mc_multinomial"
        - "mc_normal"
    mc_samples:
        Number of Monte Carlo samples N for MC backends. Ignored if
        `cdf_method == "exact"`, required for MC methods.
    seed:
        RNG seed for Monte Carlo backends. Ignored if `cdf_method == "exact"`.

    Returns
    -------
    Configured `MultiNullJSDTest` instance, with all nulls added and target
    alphas set.
    """
    nulls_arr: FloatArray = np.asarray(a=null_probabilities, dtype=np.float64)
    alpha_vec: FloatArray = np.asarray(a=alpha_vector, dtype=np.float64)

    if nulls_arr.ndim != 2:
        raise ValueError(f"`null_probabilities` must have shape (L, k); got {nulls_arr.shape}.")
    n_nulls, k = nulls_arr.shape

    if alpha_vec.ndim != 1 or alpha_vec.shape[0] != n_nulls:
        raise ValueError(
            f"`alpha_vector` must have shape (L,), L={n_nulls}; got {alpha_vec.shape}."
        )

    test: MultiNullJSDTest = MultiNullJSDTest(
        evidence_size=evidence_size,
        prob_dim=k,
        cdf_method=cdf_method,
        mc_samples=mc_samples,
        seed=seed,
    )

    # We can add all nulls in one call: prob_vector is (L, k), target_alpha is (L,)
    test.add_nulls(prob_vector=nulls_arr, target_alpha=alpha_vec)

    return test


def get_decisions_for_histograms(test: MultiNullJSDTest, histograms: IntArray) -> IntArray:
    """
    Evaluate `MultiNullJSDTest` decisions on a batch of histograms.

    Parameters
    ----------
    test:
        A configured `MultiNullJSDTest` instance.

    histograms:
        Two-dimensional array of shape (m, k) with integer counts per
        histogram. Each row must sum to n (the evidence size).

    Returns
    -------
    One-dimensional array of shape (m,) with integer decisions in {1, ..., L, REJECT_DECISION}.
    """
    decisions_raw = test.infer_decisions(hist_query=histograms)
    decisions_arr: IntArray = np.asarray(a=decisions_raw, dtype=np.int64)
    if decisions_arr.ndim != 1:
        raise RuntimeError(f"Expected 1-D decisions; got shape {decisions_arr.shape}.")
    return decisions_arr


def run_consistency_point_multinull_jsd(
    null_probabilities: FloatArray,
    alpha_vector: FloatArray,
    alts_df: pd.DataFrame,
    n: int,
    m_null: int,
    m_alt: int,
    scenario_name: str,
    method_name: str,
    cdf_method: str,
    mc_samples: int | None,
    mc_seed: int | None,
    rng_sampling: int | np.random.Generator | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a strong-consistency evaluation point for MultiNullJSDTest.

    Parameters
    ----------
    null_probabilities:
        Two-dimensional array of shape (L, k) with the null base probabilities
        (p_ℓ)_{ℓ=1}^L.
    alpha_vector:
        One-dimensional array of shape (L,) with per-null target significance
        levels \bar{α}_ℓ. This is the \bar{\boldsymbol{α}} in the theory.
    alts_df:
        DataFrame with alternative metadata, as returned by
        `build_alternatives_df_by_mjsd_targets`.
    n:
        Sample size (number of observations per histogram).
    m_null:
        Number of histograms to sample per null hypothesis.
    m_alt:
        Number of histograms to sample per alternative hypothesis.
    scenario_name:
        Name of the scenario (for progress display).
    method_name:
        Name of the method (for progress display).
    cdf_method:
        CDF computation backend for `MultiNullJSDTest`. Must be one of:
        - "exact"
        - "mc_multinomial"
        - "mc_normal"
    mc_samples:
        Number of Monte Carlo samples N for MC backends. Ignored if
        `cdf_method == "exact"`, required for MC methods.
    mc_seed:
        RNG seed for Monte Carlo backends. Ignored if `cdf_method == "exact"`.
    rng_sampling:
        RNG seed or generator for sampling histograms.

    Returns
    -------
    Tuple with two DataFrames:
    - Null metrics DataFrame, as returned by `evaluate_multinull_method_on_nulls`.
    - Alternative metrics DataFrame, as returned by `evaluate_multinull_method_on_alts`.
    - Counts DataFrame with decision counts for each scenario point.
    """
    rng_obj: np.random.Generator = make_generator(rng_sampling)

    null_p: FloatArray = np.asarray(a=null_probabilities, dtype=np.float64)
    n_nulls, k = null_p.shape

    test: MultiNullJSDTest = make_multinull_jsd_test(
        null_probabilities=null_p,
        alpha_vector=np.asarray(alpha_vector, dtype=np.float64),
        evidence_size=int(n),
        cdf_method=cdf_method,
        mc_samples=mc_samples,
        seed=mc_seed,
    )

    # --- Under each null (sample in one call) ---
    h0_all: IntArray = sample_multinomial_histograms_for_nulls(
        base_probabilities=null_p,
        num_observations=int(n),
        num_histograms_per_null=int(m_null),
        rng=rng_obj,
    )  # shape (n_nulls, m_null, k)

    decisions_by_ell: dict[int, IntArray] = {}
    for ell0 in range(n_nulls):  # 0-based
        h: IntArray = h0_all[ell0]
        decisions_by_ell[ell0 + 1] = get_decisions_for_histograms(test=test, histograms=h)

    count_rows: list[dict[str, Any]] = []

    for ell, dec in decisions_by_ell.items():
        count_rows += decision_count_rows(
            scenario_name=scenario_name,
            method=method_name,
            n=n,
            true_kind="null",
            true_id=ell,
            decisions=dec,
            n_nulls=n_nulls,
        )

    null_metrics: pd.DataFrame = evaluate_multinull_method_on_nulls(
        decisions_by_ell=decisions_by_ell,
        n_nulls=n_nulls
    )

    # --- Under alternatives ---
    decisions_by_alt: dict[int, IntArray] = {}
    for _, row in alts_df.iterrows():
        alt_id: int = int(row.at["alt_id"])
        q: FloatArray = np.asarray(a=row.at["q_vector"], dtype=np.float64)

        hq: IntArray = sample_multinomial_histograms_for_null(
            base_probabilities=q,
            num_observations=int(n),
            num_histograms=int(m_alt),
            rng=rng_obj,
        )

        decisions_by_alt[alt_id] = get_decisions_for_histograms(test=test, histograms=hq)

    alts_idx: pd.DataFrame = alts_df.set_index(keys="alt_id", drop=False)
    for alt_id, dec in decisions_by_alt.items():
        extra: dict[str, Any] = {
            "mjsd_target": float(alts_idx.at[alt_id, "mjsd_target"]),
            "mjsd": float(alts_idx.at[alt_id, "mjsd"]),
            "mjsd_error": float(alts_idx.at[alt_id, "mjsd_error"]),
            "closest_null": int(alts_idx.at[alt_id, "closest_null"]),
        }
        count_rows += decision_count_rows(
            scenario_name=scenario_name,
            method=method_name,
            n=n,
            true_kind="alt",
            true_id=alt_id,
            decisions=dec,
            n_nulls=n_nulls,
            extra_meta=extra,
        )

    alt_metrics: pd.DataFrame = evaluate_multinull_method_on_alts(
        decisions_by_alt=decisions_by_alt,
        alts_df=alts_df
    )

    counts_df: pd.DataFrame = pd.DataFrame(data=count_rows)

    return null_metrics, alt_metrics, counts_df


def run_consistency_point_holm_baseline(
    method_name: str,
    scenario_name: str,
    single_null_pvalue_fn: Callable[[IntArray, FloatArray], float],
    null_probabilities: FloatArray,
    alpha_global: float,
    alts_df: pd.DataFrame,
    n: int,
    m_null: int,
    m_alt: int,
    rng_sampling: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a strong-consistency evaluation point for a Holm-based baseline method.

    Parameters
    ----------
    method_name:
        Human-readable name for the baseline method (for progress display).
    scenario_name:
        Human-readable name for the scenario (for progress display).
    single_null_pvalue_fn:
        Function that computes a p-value for a single null hypothesis,
        given a histogram and the null base probabilities.
    null_probabilities:
        Two-dimensional array of shape (L, k) with the null base probabilities
        (p_ℓ)_{ℓ=1}^L.
    alpha_global:
        Global significance level for FWER control.
    alts_df:
        DataFrame with alternative metadata, as returned by
        `build_alternatives_df_by_mjsd_targets`.
    n:
        Sample size (number of observations per histogram).
    m_null:
        Number of histograms to sample per null hypothesis.
    m_alt:
        Number of histograms to sample per alternative hypothesis.
    rng_sampling:
        RNG seed or generator for sampling histograms.

    Returns
    -------
    Tuple with three DataFrames:
    - Null metrics DataFrame, as returned by `evaluate_multinull_method_on_nulls`.
    - Alternative metrics DataFrame, as returned by `evaluate_multinull_method_on_alts`.
    - Counts DataFrame with decision counts for each scenario point.
    """
    null_p: FloatArray = np.asarray(a=null_probabilities, dtype=np.float64)
    n_nulls, k = null_p.shape

    # --- Under each null ---
    decisions_by_ell: dict[int, IntArray] = {}
    for ell in range(1, n_nulls + 1):
        h: IntArray = sample_multinomial_histograms_for_null(
            base_probabilities=null_p[ell - 1],
            num_observations=n,
            num_histograms=m_null,
            rng=rng_sampling
        )
        decisions: IntArray = multinull_decisions_holm_batch(
            histograms=h,
            null_probabilities=null_p,
            alpha_global=float(alpha_global),
            single_null_pvalue_fn=single_null_pvalue_fn,
            n_jobs=N_JOBS,
            n_chunks=N_CHUNKS,
            parallel_backend=PARALLEL_BACKEND,
            mp_start_method="spawn",
            show_progress=SHOW_PROGRESS,
            progress_desc=f"{method_name} | n={n} | H0 ell={ell}",
        )
        decisions_by_ell[ell] = np.asarray(a=decisions, dtype=np.int64)

    count_rows: list[dict[str, Any]] = []

    for ell, dec in decisions_by_ell.items():
        count_rows += decision_count_rows(
            scenario_name=scenario_name,
            method=method_name,
            n=n,
            true_kind="null",
            true_id=ell,
            decisions=dec,
            n_nulls=n_nulls,
        )

    null_metrics: pd.DataFrame = evaluate_multinull_method_on_nulls(
        decisions_by_ell=decisions_by_ell,
        n_nulls=n_nulls
    )

    # --- Under alternatives ---
    decisions_by_alt: dict[int, IntArray] = {}
    for _, row in alts_df.iterrows():
        alt_id: int = int(row.at["alt_id"])
        q: FloatArray = row.at["q_vector"]
        hq: IntArray = sample_multinomial_histograms_for_null(
            base_probabilities=q,
            num_observations=n,
            num_histograms=m_alt,
            rng=rng_sampling
        )

        decisions = multinull_decisions_holm_batch(
            histograms=hq,
            null_probabilities=null_p,
            alpha_global=float(alpha_global),
            single_null_pvalue_fn=single_null_pvalue_fn,
            n_jobs=N_JOBS,
            n_chunks=N_CHUNKS,
            parallel_backend=PARALLEL_BACKEND,
            mp_start_method="spawn",
            show_progress=SHOW_PROGRESS,
            progress_desc=f"{method_name} | n={n} | H1 alt_id={alt_id} (mJSd={row.at["mjsd"]:.3f})",
        )
        decisions_by_alt[alt_id] = np.asarray(a=decisions, dtype=np.int64)

    alts_idx: pd.DataFrame = alts_df.set_index(keys="alt_id", drop=False)
    for alt_id, dec in decisions_by_alt.items():
        extra: dict[str, Any] = {
            "mjsd_target": float(alts_idx.at[alt_id, "mjsd_target"]),
            "mjsd": float(alts_idx.at[alt_id, "mjsd"]),
            "mjsd_error": float(alts_idx.at[alt_id, "mjsd_error"]),
            "closest_null": int(alts_idx.at[alt_id, "closest_null"]),
        }
        count_rows += decision_count_rows(
            scenario_name=scenario_name,
            method=method_name,
            n=n,
            true_kind="alt",
            true_id=alt_id,
            decisions=dec,
            n_nulls=n_nulls,
            extra_meta=extra,
        )

    alt_metrics: pd.DataFrame = evaluate_multinull_method_on_alts(
        decisions_by_alt=decisions_by_alt,
        alts_df=alts_df
    )

    counts_df: pd.DataFrame = pd.DataFrame(data=count_rows)

    return null_metrics, alt_metrics, counts_df


def run_consistency_point_holm_baseline_multistat(
    method_prefix: str,
    scenario_name: str,
    stat_names: list[str],
    single_null_pvalues_fn: Callable[[IntArray, FloatArray], FloatArray],
    null_probabilities: FloatArray,
    alpha_global: float,
    alts_df: pd.DataFrame,
    n: int,
    m_null: int,
    m_alt: int,
    rng_sampling: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a strong-consistency evaluation point for a Holm-based **multi-statistic** baseline method,
    returning both metrics and per-run decision-count histograms.

    This is intended for single-null backends that return multiple p-values at once (e.g., ExactMultinom
    returning [Prob, Chisq, LLR]). For each statistic, we interpret it as defining its own Holm-based
    multinull selector, and compute null-consistency and power metrics.

    Parameters
    ----------
    method_prefix:
        Prefix for the baseline method name. Each statistic yields a method label:
        f"{method_prefix}-{stat}+Holm".
    scenario_name:
        Human-readable scenario name (stored in the counts dataframe metadata).
    stat_names:
        List of statistic names, of length s, matching the order of the p-values returned by
        `single_null_pvalues_fn`.
    single_null_pvalues_fn:
        Callable with signature:
            single_null_pvalues_fn(histogram: IntArray, p_null: FloatArray) -> FloatArray
        returning a 1-D array of shape (s,).
    null_probabilities:
        Array of shape (L, k) with null probabilities.
    alpha_global:
        Global FWER level for Holm.
    alts_df:
        Alternatives DataFrame with at least columns:
        - "alt_id" (unique)
        - "q_vector"
        And typically: "mjsd_target", "mjsd", "mjsd_error", "closest_null".
    n, m_null, m_alt:
        Histogram size and number of histograms under nulls / alternatives.
    rng_sampling:
        RNG for multinomial sampling.

    Returns
    -------
    null_metrics_all, alt_metrics_all, counts_df:
        - null_metrics_all: metrics under H0, concatenated across stats, with "method".
        - alt_metrics_all: metrics under H1, concatenated across stats, with "method".
        - counts_df: decision-count histogram rows for each (scenario, method, n, true_kind, true_id).
    """
    null_p: FloatArray = np.asarray(a=null_probabilities, dtype=np.float64)
    n_nulls, _k = null_p.shape

    # Validate alternatives indexing
    if not alts_df["alt_id"].is_unique:
        raise ValueError("alts_df['alt_id'] must be unique.")
    alts_idx: pd.DataFrame = alts_df.set_index(keys="alt_id", drop=False)

    # We'll collect decisions per-stat, per truth (null/alt), then evaluate once per stat.
    # decisions_by_ell_by_stat[stat][ell] = decisions (m_null,)
    decisions_by_ell_by_stat: dict[str, dict[int, IntArray]] = {s: {} for s in stat_names}
    decisions_by_alt_by_stat: dict[str, dict[int, IntArray]] = {s: {} for s in stat_names}

    count_rows: list[dict[str, Any]] = []

    # --- Under each null ---
    for ell in range(1, n_nulls + 1):
        h: IntArray = sample_multinomial_histograms_for_null(
            base_probabilities=null_p[ell - 1],
            num_observations=int(n),
            num_histograms=int(m_null),
            rng=rng_sampling,
        )

        # (m_null, s)
        decisions_mat: IntArray = multinull_decisions_holm_batch_multistat(
            histograms=h,
            null_probabilities=null_p,
            alpha_global=float(alpha_global),
            single_null_pvalues_fn=single_null_pvalues_fn,
            n_jobs=N_JOBS,
            n_chunks=N_CHUNKS,
            parallel_backend=PARALLEL_BACKEND,
            mp_start_method="spawn",
            show_progress=SHOW_PROGRESS,
            progress_desc=f"{method_prefix} | n={n} | H0 ell={ell}",
        )

        s_out: int = int(decisions_mat.shape[1])
        if s_out != len(stat_names):
            raise RuntimeError(f"Expected {len(stat_names)} stats, got {s_out}.")

        for j, stat in enumerate(stat_names):
            method_name: str = f"{method_prefix}-{stat}+Holm"
            dec: IntArray = np.asarray(decisions_mat[:, j], dtype=np.int64)

            decisions_by_ell_by_stat[stat][ell] = dec

            count_rows += decision_count_rows(
                scenario_name=scenario_name,
                method=method_name,
                n=n,
                true_kind="null",
                true_id=ell,
                decisions=dec,
                n_nulls=n_nulls,
            )

    # Build null-metrics once per stat (expects ALL ell present)
    rows_null_all: list[pd.DataFrame] = []
    for stat in stat_names:
        method_name = f"{method_prefix}-{stat}+Holm"
        df_null: pd.DataFrame = evaluate_multinull_method_on_nulls(
            decisions_by_ell=decisions_by_ell_by_stat[stat],
            n_nulls=n_nulls,
        )
        df_null["method"] = method_name
        rows_null_all.append(df_null)

    null_metrics_all: pd.DataFrame = (
        pd.concat(objs=rows_null_all, ignore_index=True) if rows_null_all else pd.DataFrame()
    )

    # --- Under alternatives ---
    for _, row in alts_df.iterrows():
        alt_id: int = int(row.at["alt_id"])
        q: FloatArray = np.asarray(a=row.at["q_vector"], dtype=np.float64)

        hq: IntArray = sample_multinomial_histograms_for_null(
            base_probabilities=q,
            num_observations=int(n),
            num_histograms=int(m_alt),
            rng=rng_sampling,
        )

        decisions_mat = multinull_decisions_holm_batch_multistat(
            histograms=hq,
            null_probabilities=null_p,
            alpha_global=float(alpha_global),
            single_null_pvalues_fn=single_null_pvalues_fn,
            n_jobs=N_JOBS,
            n_chunks=N_CHUNKS,
            parallel_backend=PARALLEL_BACKEND,
            mp_start_method="spawn",
            show_progress=SHOW_PROGRESS,
            progress_desc=f"{method_prefix} | n={n} | H1 alt_id={alt_id} (mJSd={float(row.at['mjsd']):.3f})",
        )

        s_out = int(decisions_mat.shape[1])
        if s_out != len(stat_names):
            raise RuntimeError(f"Expected {len(stat_names)} stats, got {s_out}.")

        extra: dict[str, Any] = {
            "mjsd_target": float(alts_idx.at[alt_id, "mjsd_target"]),
            "mjsd": float(alts_idx.at[alt_id, "mjsd"]),
            "mjsd_error": float(alts_idx.at[alt_id, "mjsd_error"]),
            "closest_null": int(alts_idx.at[alt_id, "closest_null"]),
        }

        for j, stat in enumerate(stat_names):
            method_name = f"{method_prefix}-{stat}+Holm"
            dec = np.asarray(decisions_mat[:, j], dtype=np.int64)

            decisions_by_alt_by_stat[stat][alt_id] = dec

            count_rows += decision_count_rows(
                scenario_name=scenario_name,
                method=method_name,
                n=n,
                true_kind="alt",
                true_id=alt_id,
                decisions=dec,
                n_nulls=n_nulls,
                extra_meta=extra,
            )

    # Build alt-metrics once per stat (expects ALL alt_id present for that stat)
    rows_alt_all: list[pd.DataFrame] = []
    for stat in stat_names:
        method_name = f"{method_prefix}-{stat}+Holm"
        df_alt: pd.DataFrame = evaluate_multinull_method_on_alts(
            decisions_by_alt=decisions_by_alt_by_stat[stat],
            alts_df=alts_df,
        )
        df_alt["method"] = method_name
        rows_alt_all.append(df_alt)

    alt_metrics_all: pd.DataFrame = (
        pd.concat(objs=rows_alt_all, ignore_index=True) if rows_alt_all else pd.DataFrame()
    )

    counts_df: pd.DataFrame = pd.DataFrame(data=count_rows)

    return null_metrics_all, alt_metrics_all, counts_df


def run_experiment_core_for_scenario(
    scenario: ExperimentScenario,
    include_baselines: bool = True,
    save_histograms: bool = False,
) -> pd.DataFrame:
    """
    Run all configured methods for a single `ExperimentScenario`.

    This is the main scenario-level driver. It samples null probabilities once, builds a fixed set of alternatives
    once, and then evaluates each active method across the union of n values implied by `scenario.method_plans`. For
    each n, histograms are sampled once (shared across methods) and then sliced per method according to its replication
    plan.

    Side effects
    -----------
    - Writes `{scenario_slug}_null_probabilities.npy` and `{scenario_slug}_alternatives.pkl` under `RESULTS_DIR`.
    - Writes a scenario-level CSV `{scenario_slug}.csv` containing decision-count rows merged with per-(true_id)
      empirical metrics and runtime metadata.
    - Optionally writes per-n histogram archives `{scenario_slug}_histograms_n{n}.npz` when `save_histograms=True`.

    Parameters
    ----------
    scenario:
        Scenario definition (null family, alpha vector, alternatives targets, and method plans).
    include_baselines:
        If True, include Holm-based baselines in addition to `MultiNullJSD`.
    save_histograms:
        If True, persist sampled histograms to disk (can be large).

    Returns
    -------
    pd.DataFrame
        Scenario-level results table (the same content written to CSV).
    """
    cfg: NullSamplingConfig = scenario.null_sampling_config
    n_nulls: int = int(cfg.num_nulls)
    k: int = int(cfg.num_categories)

    # Dedicated RNGs
    rng_nulls: np.random.Generator = make_generator(
        seed_or_rng=int(rng_global.integers(0, 2**31 - 1))
    )
    rng_alts: np.random.Generator = make_generator(
        seed_or_rng=int(rng_global.integers(0, 2**31 - 1))
    )
    rng_sampling: np.random.Generator = make_generator(
        seed_or_rng=int(rng_global.integers(0, 2**31 - 1))
    )

    scen_slug: str = _slug(s=scenario.name)

    # 1) Sample null probabilities once
    null_p: FloatArray = generate_null_probabilities(config=cfg, rng=rng_nulls)
    if null_p.shape != (n_nulls, k):
        raise RuntimeError(f"Expected null_probs shape {(n_nulls, k)}, got {null_p.shape}.")
    np.save(file=RESULTS_DIR / f"{scen_slug}_null_probabilities.npy", arr=null_p)

    # 2) Build the alternative hypotheses once
    alts_df: pd.DataFrame = build_alternatives_df_by_mjsd_targets(
        null_probabilities=null_p,
        mjsd_targets=scenario.mjsd_targets,
        num_candidate_samples=scenario.alt_num_candidate_samples,
        dirichlet_alpha=scenario.alt_dirichlet_alpha,
        rng=rng_alts,
        verbose=True,
    )
    alts_df.to_pickle(path=RESULTS_DIR / f"{scen_slug}_alternatives.pkl")

    # Prepare alt matrix in alt_id order (1..T)
    alts_df_sorted: pd.DataFrame = alts_df.sort_values(by="alt_id").reset_index(drop=True)
    alt_ids: IntArray = alts_df_sorted["alt_id"].to_numpy(dtype=int)
    q_mat: FloatArray = np.stack(
        arrays=alts_df_sorted["q_vector"].to_numpy(), axis=0
    ).astype(dtype=np.float64)

    alpha_vec: FloatArray = np.asarray(a=scenario.alpha_vector, dtype=np.float64)
    alpha_global: float = float(np.max(alpha_vec))

    # 3) Build method list (names are canonical)
    methods: list[str] = [METHOD_JSD]

    if include_baselines and not scenario.ignore_baselines:
        methods += list(BASELINE_SINGLE.keys())
        methods += EXACT_METHODS

    # 4) Determine per-method n_grids and union them
    if scenario.method_plans is None:
        raise RuntimeError("scenario.method_plans must be set in build_default_scenarios().")

    n_grid_union: list[int] = sorted(
        {n for m in methods for n in _plan_lookup(plans=scenario.method_plans, method_name=m).n_grid}
    )

    count_rows_all: list[dict[str, Any]] = []
    runtime_rows_all: list[dict[str, Any]] = []
    metrics_rows_all: list[pd.DataFrame] = []

    for n in n_grid_union:
        # Methods that actually run at this n
        active_methods: list[str] = [
            m for m in methods
            if int(n) in set(_plan_lookup(plans=scenario.method_plans, method_name=m).n_grid)
        ]

        # Handle MMD constraint here (skip method at this n)
        active_methods = [
            m for m in active_methods if not (("MMD-" in m) and (int(n) < 3))
        ]

        if not active_methods:
            continue

        # Compute max m needed across active methods for shared sampling
        m_null_max: int = max(
            _plan_lookup(plans=scenario.method_plans, method_name=m).m_for(n=int(n), true_kind="null")
            for m in active_methods
        )
        m_alt_max: int = max(
            _plan_lookup(plans=scenario.method_plans, method_name=m).m_for(n=int(n), true_kind="alt")
            for m in active_methods
        )

        # Sample shared histograms once
        h0_all, hq_all = _sample_shared_histograms_for_n(
            null_p=null_p,
            q_mat=q_mat,
            n=int(n),
            m_null_max=int(m_null_max),
            m_alt_max=int(m_alt_max),
            rng_sampling=rng_sampling,
        )
        _maybe_save_histograms(
            scenario_slug=scen_slug,
            n=int(n),
            h0_all=h0_all,
            hq_all=hq_all,
            alt_ids=alt_ids,
            save_histograms=save_histograms,
        )

        # Run each method using slices of the shared histograms
        for method in active_methods:
            plan: SamplingPlan = _plan_lookup(plans=scenario.method_plans, method_name=method)
            m_null: int = plan.m_for(n=int(n), true_kind="null")
            m_alt: int = plan.m_for(n=int(n), true_kind="alt")

            # Slice shared randomness
            h0: IntArray = h0_all[:, :int(m_null), :]  # (L, m_null, k)
            hq: IntArray = hq_all[:, :int(m_alt), :]  # (T, m_alt, k)

            # MultiNullJSD variants
            if method == METHOD_JSD:
                cdf_method: str = scenario.cdf_method
                mc_samples: int | None = scenario.mc_samples
                mc_seed: int | None = scenario.mc_seed

                t_total0: float = time.perf_counter()
                t0: float = time.perf_counter()
                test: MultiNullJSDTest = make_multinull_jsd_test(
                    null_probabilities=null_p,
                    alpha_vector=alpha_vec,
                    evidence_size=int(n),
                    cdf_method=cdf_method,
                    mc_samples=mc_samples,
                    seed=mc_seed,
                )
                t_make: float = time.perf_counter() - t0

                # decisions under nulls
                decisions_by_ell: dict[int, IntArray] = {}
                runtime_by_ell: dict[int, float] = {}
                for ell0 in range(n_nulls):
                    t1: float = time.perf_counter()
                    decisions_by_ell[ell0 + 1] = get_decisions_for_histograms(test=test, histograms=h0[ell0])
                    runtime_by_ell[ell0 + 1] = time.perf_counter() - t1

                # decisions under alts
                decisions_by_alt: dict[int, IntArray] = {}
                runtime_by_alt: dict[int, float] = {}
                for t, alt_id in enumerate(alt_ids, start=0):
                    alt_id_i: int = int(alt_id)  # noqa
                    t1 = time.perf_counter()
                    decisions_by_alt[alt_id_i] = get_decisions_for_histograms(test=test, histograms=hq[t])
                    runtime_by_alt[alt_id_i] = time.perf_counter() - t1

                # Backend quantities
                t0 = time.perf_counter()
                alpha_backend_by_ell: FloatArray = np.asarray(
                    test.get_alpha(null_index=list(range(1, n_nulls + 1))),
                    dtype=np.float64,
                )
                fwer_backend: float = float(test.get_fwer())
                beta_backend_by_alt: FloatArray = np.asarray(
                    test.get_beta(prob_query=q_mat),
                    dtype=np.float64,
                )
                t_backend: float = time.perf_counter() - t0

                t_total: float = time.perf_counter() - t_total0

                alts_idx: pd.DataFrame = alts_df_sorted.set_index(keys="alt_id", drop=False)

                for ell, dec in decisions_by_ell.items():
                    count_rows_all += decision_count_rows(
                        scenario_name=scenario.name,
                        method=method,
                        n=int(n),
                        true_kind="null",
                        true_id=int(ell),
                        decisions=dec,
                        n_nulls=n_nulls,
                        extra_meta={"scenario_slug": scen_slug, "alpha_global": alpha_global},
                    )

                for alt_id, dec in decisions_by_alt.items():
                    extra: dict[str, Any] = {
                        "scenario_slug": scen_slug,
                        "alpha_global": alpha_global,
                        "mjsd_target": float(alts_idx.at[int(alt_id), "mjsd_target"]),
                        "mjsd": float(alts_idx.at[int(alt_id), "mjsd"]),
                        "mjsd_error": float(alts_idx.at[int(alt_id), "mjsd_error"]),
                        "closest_null": int(alts_idx.at[int(alt_id), "closest_null"]),
                    }
                    count_rows_all += decision_count_rows(
                        scenario_name=scenario.name,
                        method=method,
                        n=int(n),
                        true_kind="alt",
                        true_id=int(alt_id),
                        decisions=dec,
                        n_nulls=n_nulls,
                        extra_meta=extra,
                    )

                # --- Metrics (null + alt) ---
                df_m_null: pd.DataFrame = _metrics_rows_from_decisions(
                    scenario=scenario,
                    scenario_slug=scen_slug,
                    method=method,
                    n=int(n),
                    true_kind="null",
                    decisions_by_id=decisions_by_ell,
                    null_p=null_p,
                    alts_df=alts_df_sorted,
                    alpha_backend_by_ell=alpha_backend_by_ell,
                    beta_backend_by_alt=None,
                    fwer_backend=fwer_backend,
                    cdf_method=cdf_method,
                    mc_samples=mc_samples,
                    mc_seed=mc_seed,
                )
                df_m_alt: pd.DataFrame = _metrics_rows_from_decisions(
                    scenario=scenario,
                    scenario_slug=scen_slug,
                    method=method,
                    n=int(n),
                    true_kind="alt",
                    decisions_by_id=decisions_by_alt,
                    null_p=null_p,
                    alts_df=alts_df_sorted,
                    alpha_backend_by_ell=None,
                    beta_backend_by_alt=beta_backend_by_alt,
                    fwer_backend=fwer_backend,
                    cdf_method=cdf_method,
                    mc_samples=mc_samples,
                    mc_seed=mc_seed,
                )
                metrics_rows_all += [df_m_null, df_m_alt]

                # shared runtime overhead
                shared: dict[str, Any] = {
                    "runtime_make_test_s": t_make,
                    "runtime_backend_s": t_backend,
                    "runtime_total_method_s": t_total,
                    "L": int(n_nulls),
                    "k": int(k),
                    "m_null": int(m_null),
                    "m_alt": int(m_alt),
                    "cdf_method": cdf_method,
                    "mc_samples": mc_samples if mc_samples is not None else np.nan,
                    "mc_seed": mc_seed if mc_seed is not None else np.nan,
                }

                for ell, rt in runtime_by_ell.items():
                    runtime_rows_all.append(
                        {
                            "scenario_name": scenario.name,
                            "scenario_slug": scen_slug,
                            "method": method,
                            "n": int(n),
                            "true_kind": "null",
                            "true_id": int(ell),
                            "runtime_infer_s": float(rt),
                            "runtime_per_hist_s": float(rt) / float(m_null),
                            **shared,
                        }
                    )

                for alt_id, rt in runtime_by_alt.items():
                    runtime_rows_all.append(
                        {
                            "scenario_name": scenario.name,
                            "scenario_slug": scen_slug,
                            "method": method,
                            "n": int(n),
                            "true_kind": "alt",
                            "true_id": int(alt_id),
                            "runtime_infer_s": float(rt),
                            "runtime_per_hist_s": float(rt) / float(m_alt),
                            **shared,
                        }
                    )

                continue

            # Single-stat Holm baselines
            if method in BASELINE_SINGLE:
                t_total0 = time.perf_counter()
                pval_fn: Callable[[IntArray, FloatArray], float] = BASELINE_SINGLE[method]

                decisions_by_ell = {}
                runtime_by_ell = {}
                for ell0 in range(n_nulls):
                    t1 = time.perf_counter()
                    decisions: IntArray = multinull_decisions_holm_batch(
                        histograms=h0[ell0],
                        null_probabilities=null_p,
                        alpha_global=alpha_global,
                        single_null_pvalue_fn=pval_fn,
                        n_jobs=N_JOBS,
                        n_chunks=N_CHUNKS,
                        parallel_backend=PARALLEL_BACKEND,
                        mp_start_method="spawn",
                        show_progress=SHOW_PROGRESS,
                        progress_desc=f"{scenario.name} | {method} | n={n} | H0 ell={ell0+1}",
                    )
                    runtime_by_ell[ell0 + 1] = time.perf_counter() - t1
                    decisions_by_ell[ell0 + 1] = np.asarray(a=decisions, dtype=np.int64)

                decisions_by_alt = {}
                runtime_by_alt = {}
                for t, alt_id in enumerate(alt_ids, start=0):
                    alt_id_i = int(alt_id)  # noqa
                    t1 = time.perf_counter()
                    decisions = multinull_decisions_holm_batch(
                        histograms=hq[t],
                        null_probabilities=null_p,
                        alpha_global=alpha_global,
                        single_null_pvalue_fn=pval_fn,
                        n_jobs=N_JOBS,
                        n_chunks=N_CHUNKS,
                        parallel_backend=PARALLEL_BACKEND,
                        mp_start_method="spawn",
                        show_progress=SHOW_PROGRESS,
                        progress_desc=f"{scenario.name} | {method} | n={n} | H1 alt_id={alt_id_i}",
                    )
                    runtime_by_alt[alt_id_i] = time.perf_counter() - t1
                    decisions_by_alt[alt_id_i] = np.asarray(a=decisions, dtype=np.int64)

                # Counts
                for ell, dec in decisions_by_ell.items():
                    count_rows_all += decision_count_rows(
                        scenario_name=scenario.name,
                        method=method,
                        n=int(n),
                        true_kind="null",
                        true_id=int(ell),
                        decisions=dec,
                        n_nulls=n_nulls,
                        extra_meta={"scenario_slug": scen_slug, "alpha_global": alpha_global},
                    )

                alts_idx = alts_df_sorted.set_index(keys="alt_id", drop=False)
                for alt_id, dec in decisions_by_alt.items():
                    extra = {
                        "scenario_slug": scen_slug,
                        "alpha_global": alpha_global,
                        "mjsd_target": float(alts_idx.at[int(alt_id), "mjsd_target"]),
                        "mjsd": float(alts_idx.at[int(alt_id), "mjsd"]),
                        "mjsd_error": float(alts_idx.at[int(alt_id), "mjsd_error"]),
                        "closest_null": int(alts_idx.at[int(alt_id), "closest_null"]),
                    }
                    count_rows_all += decision_count_rows(
                        scenario_name=scenario.name,
                        method=method,
                        n=int(n),
                        true_kind="alt",
                        true_id=int(alt_id),
                        decisions=dec,
                        n_nulls=n_nulls,
                        extra_meta=extra,
                    )

                # Metrics rows (no backend)
                df_m_null = _metrics_rows_from_decisions(
                    scenario=scenario,
                    scenario_slug=scen_slug,
                    method=method,
                    n=int(n),
                    true_kind="null",
                    decisions_by_id=decisions_by_ell,
                    null_p=null_p,
                    alts_df=alts_df_sorted,
                )
                df_m_alt = _metrics_rows_from_decisions(
                    scenario=scenario,
                    scenario_slug=scen_slug,
                    method=method,
                    n=int(n),
                    true_kind="alt",
                    decisions_by_id=decisions_by_alt,
                    null_p=null_p,
                    alts_df=alts_df_sorted,
                )
                metrics_rows_all += [df_m_null, df_m_alt]

                t_total_method: float = time.perf_counter() - t_total0

                shared = {
                    "runtime_total_method_s": float(t_total_method),
                    "L": int(n_nulls),
                    "k": int(k),
                    "m_null": int(m_null),
                    "m_alt": int(m_alt),
                }
                for ell, rt in runtime_by_ell.items():
                    runtime_rows_all.append(
                        {
                            "scenario_name": scenario.name,
                            "scenario_slug": scen_slug,
                            "method": method,
                            "n": int(n),
                            "true_kind": "null",
                            "true_id": int(ell),
                            "runtime_infer_s": float(rt),
                            "runtime_per_hist_s": float(rt) / float(m_null),
                            **shared,
                        }
                    )
                for alt_id, rt in runtime_by_alt.items():
                    runtime_rows_all.append(
                        {
                            "scenario_name": scenario.name,
                            "scenario_slug": scen_slug,
                            "method": method,
                            "n": int(n),
                            "true_kind": "alt",
                            "true_id": int(alt_id),
                            "runtime_infer_s": float(rt),
                            "runtime_per_hist_s": float(rt) / float(m_alt),
                            **shared,
                        }
                    )
                continue

            # ExactMultinom multistat (3 derived methods)
            if method.startswith(f"{EXACT_PREFIX}-"):
                t_total0 = time.perf_counter()
                # We run the 3 stats together ONCE, then split into EXACT_METHODS
                # So here, only do work when the "first" exact method is encountered:
                if method != EXACT_METHODS[0]:
                    continue

                decisions_by_ell_by_stat: dict[str, dict[int, IntArray]] = {s: {} for s in EXACT_STATS}
                decisions_by_alt_by_stat: dict[str, dict[int, IntArray]] = {s: {} for s in EXACT_STATS}

                # Null side
                runtime_by_ell = {}
                for ell0 in range(n_nulls):
                    t1 = time.perf_counter()
                    dec_mat: IntArray = multinull_decisions_holm_batch_multistat(
                        histograms=h0[ell0],
                        null_probabilities=null_p,
                        alpha_global=alpha_global,
                        single_null_pvalues_fn=exact_pvals_fn,
                        n_jobs=N_JOBS,
                        n_chunks=N_CHUNKS,
                        parallel_backend=PARALLEL_BACKEND,
                        mp_start_method="spawn",
                        show_progress=SHOW_PROGRESS,
                        progress_desc=f"{scenario.name} | {EXACT_PREFIX} | n={n} | H0 ell={ell0+1}",
                    )
                    runtime_by_ell[ell0 + 1] = time.perf_counter() - t1
                    for j, s in enumerate(EXACT_STATS):
                        decisions_by_ell_by_stat[s][ell0 + 1] = np.asarray(a=dec_mat[:, j], dtype=np.int64)

                # Alt side
                runtime_by_alt = {}
                for t, alt_id in enumerate(alt_ids, start=0):
                    alt_id_i = int(alt_id)  # noqa
                    t1 = time.perf_counter()
                    dec_mat = multinull_decisions_holm_batch_multistat(
                        histograms=hq[t],
                        null_probabilities=null_p,
                        alpha_global=alpha_global,
                        single_null_pvalues_fn=exact_pvals_fn,
                        n_jobs=N_JOBS,
                        n_chunks=N_CHUNKS,
                        parallel_backend=PARALLEL_BACKEND,
                        mp_start_method="spawn",
                        show_progress=SHOW_PROGRESS,
                        progress_desc=f"{scenario.name} | {EXACT_PREFIX} | n={n} | H1 alt_id={alt_id_i}",
                    )
                    runtime_by_alt[alt_id_i] = time.perf_counter() - t1
                    for j, s in enumerate(EXACT_STATS):
                        decisions_by_alt_by_stat[s][int(alt_id)] = np.asarray(  # noqa
                            a=dec_mat[:, j], dtype=np.int64
                        )

                alts_idx = alts_df_sorted.set_index(keys="alt_id", drop=False)
                t_total = time.perf_counter() - t_total0

                # Now emit counts and metrics per derived EXACT_METHODS
                for s in EXACT_STATS:
                    method_s: str = f"{EXACT_PREFIX}-{s}+Holm"

                    for ell, rt in runtime_by_ell.items():
                        runtime_rows_all.append(
                            {
                                "scenario_name": scenario.name,
                                "scenario_slug": scen_slug,
                                "method": method_s,
                                "n": int(n),
                                "true_kind": "null",
                                "true_id": int(ell),
                                "runtime_infer_s": float(rt),
                                "runtime_per_hist_s": float(rt) / float(m_null),
                                "runtime_total_method_s": float(t_total),  # same for all 3 derived
                                "L": int(n_nulls),
                                "k": int(k),
                                "m_null": int(m_null),
                                "m_alt": int(m_alt),
                            }
                        )

                    for alt_id, rt in runtime_by_alt.items():
                        runtime_rows_all.append(
                            {
                                "scenario_name": scenario.name,
                                "scenario_slug": scen_slug,
                                "method": method_s,
                                "n": int(n),
                                "true_kind": "alt",
                                "true_id": int(alt_id),
                                "runtime_infer_s": float(rt),
                                "runtime_per_hist_s": float(rt) / float(m_alt),
                                "runtime_total_method_s": float(t_total),
                                "L": int(n_nulls),
                                "k": int(k),
                                "m_null": int(m_null),
                                "m_alt": int(m_alt),
                            }
                        )

                    for ell, dec in decisions_by_ell_by_stat[s].items():
                        count_rows_all += decision_count_rows(
                            scenario_name=scenario.name,
                            method=method_s,
                            n=int(n),
                            true_kind="null",
                            true_id=int(ell),
                            decisions=dec,
                            n_nulls=n_nulls,
                            extra_meta={"scenario_slug": scen_slug, "alpha_global": alpha_global},
                        )

                    for alt_id, dec in decisions_by_alt_by_stat[s].items():
                        extra = {
                            "scenario_slug": scen_slug,
                            "alpha_global": alpha_global,
                            "mjsd_target": float(alts_idx.at[int(alt_id), "mjsd_target"]),
                            "mjsd": float(alts_idx.at[int(alt_id), "mjsd"]),
                            "mjsd_error": float(alts_idx.at[int(alt_id), "mjsd_error"]),
                            "closest_null": int(alts_idx.at[int(alt_id), "closest_null"]),
                        }
                        count_rows_all += decision_count_rows(
                            scenario_name=scenario.name,
                            method=method_s,
                            n=int(n),
                            true_kind="alt",
                            true_id=int(alt_id),
                            decisions=dec,
                            n_nulls=n_nulls,
                            extra_meta=extra,
                        )

                    df_m_null = _metrics_rows_from_decisions(
                        scenario=scenario,
                        scenario_slug=scen_slug,
                        method=method_s,
                        n=int(n),
                        true_kind="null",
                        decisions_by_id=decisions_by_ell_by_stat[s],
                        null_p=null_p,
                        alts_df=alts_df_sorted,
                    )
                    df_m_alt = _metrics_rows_from_decisions(
                        scenario=scenario,
                        scenario_slug=scen_slug,
                        method=method_s,
                        n=int(n),
                        true_kind="alt",
                        decisions_by_id=decisions_by_alt_by_stat[s],
                        null_p=null_p,
                        alts_df=alts_df_sorted,
                    )
                    metrics_rows_all += [df_m_null, df_m_alt]

                t_total = time.perf_counter() - t_total0
                for s in EXACT_STATS:
                    method_s = f"{EXACT_PREFIX}-{s}+Holm"
                    runtime_rows_all.append(
                        {
                            "scenario_name": scenario.name,
                            "scenario_slug": scen_slug,
                            "method": method_s,
                            "n": int(n),
                            "runtime_total_s": t_total,
                            "L": int(n_nulls),
                            "k": int(k),
                            "m_null": int(m_null),
                            "m_alt": int(m_alt),
                        }
                    )
                continue

            raise RuntimeError(f"Unhandled method {method!r}")

    # 5) Build counts DF and metrics DF
    df_counts: pd.DataFrame = pd.DataFrame(data=count_rows_all)

    df_metrics: pd.DataFrame = pd.concat(
        objs=metrics_rows_all, ignore_index=True
    ) if metrics_rows_all else pd.DataFrame()

    if runtime_rows_all and not df_metrics.empty:
        df_runtime: pd.DataFrame = pd.DataFrame(data=runtime_rows_all)
        df_metrics = df_metrics.merge(
            right=df_runtime,
            on=["scenario_name", "scenario_slug", "method", "n", "true_kind", "true_id"],
            how="left",
            suffixes=("", "__right"),
        )
        df_metrics = df_metrics.loc[:, ~df_metrics.columns.str.endswith("__right")]

    # 6) Compute FWER-hat from metrics null rows (per scenario/method/n)
    if not df_metrics.empty:
        null_only: pd.DataFrame = df_metrics[df_metrics["true_kind"] == "null"].copy()
        fwer_hat: pd.DataFrame = (
            null_only.groupby(by=["scenario_name", "scenario_slug", "method", "n"], as_index=False)
            .agg(fwer_hat_empirical=("alpha_hat_empirical", "max"))
        )
        df_metrics = df_metrics.merge(
            right=fwer_hat, on=["scenario_name", "scenario_slug", "method", "n"], how="left", suffixes=("", "__right")
        )
        df_metrics = df_metrics.loc[:, ~df_metrics.columns.str.endswith("__right")]
        df_metrics["fwer_target_method"] = df_metrics["alpha_global"]

    # 7) Merge metrics into counts (ONE CSV per scenario)
    join_keys: list[str] = ["scenario_name", "scenario_slug", "method", "n", "true_kind", "true_id"]
    df: pd.DataFrame = df_counts.merge(right=df_metrics, on=join_keys, how="left", suffixes=("", "__right"))
    df = df.loc[:, ~df.columns.str.endswith("__right")]

    df["n_jobs"] = int(N_JOBS)
    df["n_chunks"] = int(N_CHUNKS)
    df["parallel_backend"] = str(PARALLEL_BACKEND)
    df["show_progress"] = bool(SHOW_PROGRESS)
    df["mp_start_method"] = "spawn"  # you hard-code this in the calls

    baseline_mask: pd.Series = df["method"].ne(other=METHOD_JSD)
    for col in ["n_jobs", "n_chunks"]:
        df.loc[~baseline_mask, col] = pd.NA
    df.loc[~baseline_mask, "parallel_backend"] = ""
    df.loc[~baseline_mask, "show_progress"] = pd.NA
    df.loc[~baseline_mask, "mp_start_method"] = ""

    # Save scenario-level CSV
    df.to_csv(path_or_buf=RESULTS_DIR / f"{scen_slug}.csv", index=False)
    return df
