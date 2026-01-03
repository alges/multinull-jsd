"""
Metrics and information-theoretic utilities for experiment evaluation.

Includes:
- Decision-count expansion (`decision_count_rows`) for downstream aggregation.
- Empirical null/alt evaluation summaries (accuracy/type-I/misclass, power/type-II).
- Normal-approximation binomial confidence intervals.
- Jensen–Shannon divergence/distance utilities and minimum distance to a null set.
- Confusion matrix construction for simulated decision outputs.
"""
from typing import Any, Optional, Callable, Sequence

import pandas as pd
import numpy as np

from scenarios import ExperimentScenario
from settings import FloatArray, IntArray, REJECT_DECISION
from registry import METHOD_JSD_PREFIX


def decision_count_rows(
    scenario_name: str,
    method: str,
    n: int,
    true_kind: str,
    true_id: int,
    decisions: IntArray,
    n_nulls: int,
    extra_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Expand a vector of decisions into per-label count/proportion rows.

    Given decisions in {1, ..., L, REJECT_DECISION}, returns one output row per possible decision label (including
    reject-all), with both counts and proportions, plus scenario/method metadata for easy concatenation.

    Parameters
    ----------
    scenario_name:
        The name of the scenario being evaluated.
    method:
        The method or algorithm being applied in the decision process.
    n:
        The size or number of instances in the scenario.
    true_kind:
        Specifies the true condition, either "null" or "alt".
    true_id:
        The identifier associated with the true condition.
    decisions:
        An array of decision outcomes, where each value represents assigned
        decision type. Shape is (m,).
    n_nulls:
        The number of null decisions or hypotheses.
    extra_meta:
        Additional metadata to be included, optional.

    Returns
    -------
    A list of dictionaries, each containing information about scenario properties,
    decision counts, proportions, and additional metadata.
    """
    dec: IntArray = np.asarray(a=decisions, dtype=np.int64).reshape(-1)
    mapped: IntArray = np.where(dec == REJECT_DECISION, 0, dec)  # 0 == reject-all
    counts: IntArray = np.bincount(mapped, minlength=n_nulls + 1)
    m: int = int(dec.size)

    out: list[dict[str, Any]] = []
    meta: dict[str, Any] = {} if extra_meta is None else dict(extra_meta)
    for b in range(n_nulls + 1):
        label: int = REJECT_DECISION if b == 0 else b
        c: int = int(counts[b])
        out.append(
            {
                "scenario_name": scenario_name,
                "method": method,
                "n": int(n),
                "true_kind": true_kind,
                "true_id": int(true_id),
                "decision": int(label),
                "count": c,
                "prop": float(c / m) if m > 0 else 0.0,
                "m": m,
                **meta,
            }
        )
    return out


def evaluate_multinull_method_on_nulls(
    decisions_by_ell: dict[int, IntArray],
    n_nulls: int,
) -> pd.DataFrame:
    """
    decisions_by_ell[ell] are decisions for histograms sampled under H0^ell.
    Returns per-null accuracy, type1, and misclassification.

    Parameters
    ----------
    decisions_by_ell:
        Dictionary mapping null index (1-based) to an array of decisions.
    n_nulls:
        Total number of null hypotheses L.

    Returns
    -------
    DataFrame with columns:
    - "true_ell": Null index (1-based).
    - "p_correct": Estimated accuracy under H_0^{ell}.
    - "p_reject_all": Estimated type-I error under H_0^{ell}.
    - "p_misclass": Estimated misclassification rate under H_0^{ell}.
    """
    rows: list[dict[str, Any]] = []
    for ell in range(1, n_nulls + 1):
        dec: IntArray = np.asarray(a=decisions_by_ell[ell], dtype=np.int64)
        p_reject: float = estimate_type_i_error(decisions=dec, reject_all_label=REJECT_DECISION)
        p_correct: float = float(np.mean(dec == ell))
        p_misclass: float = float(1.0 - p_reject - p_correct)
        rows.append(
            {
                "true_ell": ell,
                "p_correct": p_correct,
                "p_reject_all": p_reject,
                "p_misclass": p_misclass,
            }
        )
    return pd.DataFrame(data=rows)


def evaluate_multinull_method_on_alts(
    decisions_by_alt: dict[int, IntArray],
    alts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    decisions_by_alt[alt_id] are decisions for histograms sampled under H1^{q_alt}.
    Returns per-alternative power and type-2 error.

    Parameters
    ----------
    decisions_by_alt:
        Dictionary mapping alternative ID to an array of decisions.
    alts_df:
        DataFrame with alternative metadata, as returned by
        `build_alternatives_df_by_mjsd_bins`.

    Returns
    -------
    DataFrame with columns:
    - "alt_id": Unique alternative ID (1-based).
    - "mjsd_target": Target mJSd value to nulls.
    - "mjsd": Minimum JS distance to nulls.
    - "mjsd_error": Difference of the actual and target mJSd values
    - "closest_null": Index (1-based) of the closest null hypothesis.
    - "power": Estimated power under H_1^{q_alt}.
    - "type2": Estimated type-2 error under H_1^{q_alt}.
    """
    rows: list[dict[str, Any]] = []

    if not alts_df["alt_id"].is_unique:
        raise ValueError("alts_df['alt_id'] must be unique to use .at lookups safely.")
    alts_idx: pd.DataFrame = alts_df.set_index(keys="alt_id", drop=False)

    for alt_id, dec in decisions_by_alt.items():
        dec = np.asarray(a=dec, dtype=np.int64)
        power: float = estimate_power(decisions=dec, reject_all_label=REJECT_DECISION)
        type2: float = float(1.0 - power)

        # Scalar lookups
        mjsd: float = float(alts_idx.at[alt_id, "mjsd"])
        closest_null: float = int(alts_idx.at[alt_id, "closest_null"])
        mjsd_target: float = float(alts_idx.at[alt_id, "mjsd_target"])
        mjsd_error: float = float(alts_idx.at[alt_id, "mjsd_error"])

        rows.append(
            {
                "alt_id": alt_id,
                "mjsd_target": mjsd_target,
                "mjsd": mjsd,
                "mjsd_error": mjsd_error,
                "closest_null": closest_null,
                "power": power,
                "type2": type2,
            }
        )

    return pd.DataFrame(data=rows)


def normal_approx_binomial_ci(
    success_fraction: float,
    num_trials: int,
    confidence_level: float = 0.99,
) -> tuple[float, float]:
    """
    Normal-approximation confidence interval for a binomial proportion.

    Uses p̂ ± z_{1-α/2} sqrt(p̂(1-p̂)/n), with simple clipping to [0, 1].

    Parameters
    ----------
    success_fraction:
        Empirical success proportion p̂ ∈ [0, 1].
    num_trials
        Number of Bernoulli trials n ≥ 1 used to compute p̂.
    confidence_level
        Desired confidence level in (0, 1). Supported keys are {0.90, 0.95, 0.99}.
        Defaults to 0.99.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval (clipped to [0, 1]).

    Raises
    ------
    KeyError
        If `confidence_level` is not one of the supported lookup keys.
    ValueError
        If `num_trials` < 1 or `success_fraction` is outside [0, 1].
    """
    if num_trials < 1:
        raise ValueError(f"`num_trials` must be >= 1; got {num_trials}.")
    if not (0.0 <= success_fraction <= 1.0):
        raise ValueError(f"`success_fraction` must be in [0, 1]; got {success_fraction}.")

    z_lookup: dict[float, float] = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z: float = float(z_lookup[confidence_level])
    p_hat: float = float(success_fraction)
    var_hat: float = p_hat * (1.0 - p_hat) / float(num_trials)
    std_hat: float = float(np.sqrt(var_hat))
    return max(0.0, p_hat - z * std_hat), min(1.0, p_hat + z * std_hat)


def _metrics_rows_from_decisions(
    scenario: ExperimentScenario,
    scenario_slug: str,
    method: str,
    n: int,
    true_kind: str,
    decisions_by_id: dict[int, IntArray],  # ell->decisions or alt_id->decisions
    null_p: FloatArray,
    alts_df: pd.DataFrame,
    alpha_backend_by_ell: Optional[FloatArray] = None,
    beta_backend_by_alt: Optional[FloatArray] = None,
    fwer_backend: float | None = None,
    cdf_method: str | None = None,
    mc_samples: int | None = None,
    mc_seed: int | None = None,
) -> pd.DataFrame:
    """
    Build per-true-ID metric rows (nulls or alternatives).

    For each true identifier (ℓ for nulls or alt_id for alternatives), aggregates empirical
    error/power metrics and optionally includes backend targets/estimates.

    Parameters
    ----------
    scenario::
        Scenario configuration containing alpha vector and backend settings.
    scenario_slug:
        Slug for the scenario (used in downstream joins/exports).
    method:
        Canonical method label (e.g., "MultiNullJSD", "Chi2-Pearson+Holm").
    n:
        Evidence size (histogram total count).
    true_kind:
        Either "null" or "alt", determining which metrics are computed.
    decisions_by_id:
        Mapping from true_id (ℓ for nulls; alt_id for alternatives) to a 1-D array
        of integer decisions in {1, ..., L, REJECT_DECISION}.
    null_p:
        Array of shape (L, k) with null base probabilities.
    alts_df:
        Alternatives metadata DataFrame; required when `true_kind == "alt"`.
    alpha_backend_by_ell:
        Optional backend per-null α targets/estimates of shape (L,). If provided,
        placed in the "alpha_backend" column for null rows.
    beta_backend_by_alt:
        Optional backend per-alt β estimates of shape (T,), aligned by alt_id (1-based).
        If provided, both "beta_backend" and its complement "power_backend" are emitted
        for alt rows.
    fwer_backend:
        Optional backend FWER target/estimate to include for both null/alt rows.
    cdf_method:
        Backend CDF computation method tag for metadata purposes.
    mc_samples:
        Number of MC samples used by the backend (if applicable).
    mc_seed:
        RNG seed used by the backend (if applicable).

    Returns
    -------
    pd.DataFrame
        One row per true_id with columns, including:
        - Common: scenario_name, scenario_slug, method, n, true_kind, true_id, L, k, m_used, alpha_global
        - Null rows: alpha_target_per_null, alpha_target_method, alpha_hat_empirical, alpha_ci_low, alpha_ci_high,
          alpha_backend, fwer_backend, p_correct, p_reject_all, p_misclass, cdf_method, mc_samples, mc_seed
        - Alt rows: power_hat_empirical, power_ci_low, power_ci_high, type2_hat_empirical, beta_backend, power_backend,
          fwer_backend, mjsd_target, mjsd, mjsd_error, closest_null, cdf_method, mc_samples, mc_seed

    Raises
    ------
    ValueError
        If `true_kind` is not "null" or "alt", or if `alts_df` is invalid for alt rows.
    """
    alpha_vec: FloatArray = np.asarray(a=scenario.alpha_vector, dtype=np.float64)
    alpha_global: float = float(np.max(alpha_vec))
    n_nulls: int = int(null_p.shape[0])

    rows: list[dict[str, Any]] = []

    if true_kind == "null":
        for ell, dec in decisions_by_id.items():
            dec = np.asarray(a=dec, dtype=np.int64).reshape(-1)
            m: int = int(dec.size)

            p_reject: float = float(np.mean(dec == REJECT_DECISION))
            p_correct: float = float(np.mean(dec == int(ell)))
            p_misclass: float = float(1.0 - p_reject - p_correct)

            ci_low, ci_high = normal_approx_binomial_ci(success_fraction=p_reject, num_trials=m, confidence_level=0.99)

            alpha_backend: float = np.nan
            if alpha_backend_by_ell is not None:
                alpha_backend = float(alpha_backend_by_ell[int(ell) - 1])

            alpha_target_per_null: float = float(alpha_vec[int(ell) - 1])
            alpha_target_method: float = (
                alpha_target_per_null if method.startswith(METHOD_JSD_PREFIX) else alpha_global
            )


            rows.append(
                {
                    "scenario_name": scenario.name,
                    "scenario_slug": scenario_slug,
                    "method": method,
                    "n": int(n),
                    "true_kind": "null",
                    "true_id": int(ell),
                    "L": n_nulls,
                    "k": int(null_p.shape[1]),
                    "m_used": m,
                    "alpha_global": alpha_global,
                    # Exp01-style
                    "alpha_target_per_null": alpha_target_per_null,
                    "alpha_target_method": alpha_target_method,
                    "alpha_hat_empirical": p_reject,
                    "alpha_ci_low": ci_low,
                    "alpha_ci_high": ci_high,
                    "alpha_backend": alpha_backend,
                    "fwer_backend": float(fwer_backend) if fwer_backend is not None else np.nan,
                    # Exp02 null-consistency
                    "p_correct": p_correct,
                    "p_reject_all": p_reject,
                    "p_misclass": p_misclass,
                    # method backend meta
                    "cdf_method": cdf_method if cdf_method is not None else np.nan,
                    "mc_samples": mc_samples if mc_samples is not None else np.nan,
                    "mc_seed": mc_seed if mc_seed is not None else np.nan,
                }
            )

    elif true_kind == "alt":
        if not alts_df["alt_id"].is_unique:
            raise ValueError("alts_df['alt_id'] must be unique.")
        alts_idx: pd.DataFrame = alts_df.set_index(keys="alt_id", drop=False)

        for alt_id, dec in decisions_by_id.items():
            dec = np.asarray(a=dec, dtype=np.int64).reshape(-1)
            m = int(dec.size)

            power_hat: float = float(np.mean(dec == REJECT_DECISION))
            type2_hat: float = float(1.0 - power_hat)
            ci_low, ci_high = normal_approx_binomial_ci(success_fraction=power_hat, num_trials=m, confidence_level=0.99)

            beta_backend: float = np.nan
            power_backend: float = np.nan
            if beta_backend_by_alt is not None:
                beta_backend = float(beta_backend_by_alt[int(alt_id) - 1])
                power_backend = float(1.0 - beta_backend)

            rows.append(
                {
                    "scenario_name": scenario.name,
                    "scenario_slug": scenario_slug,
                    "method": method,
                    "n": int(n),
                    "true_kind": "alt",
                    "true_id": int(alt_id),
                    "L": n_nulls,
                    "k": int(null_p.shape[1]),
                    "m_used": m,
                    "alpha_global": alpha_global,
                    # Exp02 alt-power
                    "power_hat_empirical": power_hat,
                    "power_ci_low": ci_low,
                    "power_ci_high": ci_high,
                    "type2_hat_empirical": type2_hat,
                    # Backend beta/power for our test
                    "beta_backend": beta_backend,
                    "power_backend": power_backend,
                    "fwer_backend": float(fwer_backend) if fwer_backend is not None else np.nan,
                    # alt meta
                    "mjsd_target": float(alts_idx.at[int(alt_id), "mjsd_target"]),
                    "mjsd": float(alts_idx.at[int(alt_id), "mjsd"]),
                    "mjsd_error": float(alts_idx.at[int(alt_id), "mjsd_error"]),
                    "closest_null": int(alts_idx.at[int(alt_id), "closest_null"]),
                    # method backend meta
                    "cdf_method": cdf_method if cdf_method is not None else np.nan,
                    "mc_samples": mc_samples if mc_samples is not None else np.nan,
                    "mc_seed": mc_seed if mc_seed is not None else np.nan,
                }
            )
    else:
        raise ValueError(f"true_kind must be 'null' or 'alt', got {true_kind!r}")

    return pd.DataFrame(data=rows)


def _as_2d_prob_array(x: FloatArray, name: str) -> FloatArray:
    """
    Coerce x into a 2D float64 array of shape (m, k). Accepts shape (k,) or (m, k). Rejects other shapes.

    Parameters
    ----------
    x:
        Input array to coerce.
    name:
         Name of the input array (used for error messages).

    Returns
    -------
    Two-dimensional numpy array of shape (m, k) and dtype float64.

    Raises
    ------
    ValueError
        If the input array does not have a shape (k,) or (m, k).
    """
    arr: FloatArray = np.asarray(a=x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"`{name}` must have shape (k,) or (m, k); got shape {arr.shape}.")


def _entropy_base2(probabilities: FloatArray) -> FloatArray:
    """
    Compute the Shannon entropy H(p) in bits for a batch of discrete distributions. Zero entries are treated in the
    usual convention `0 * log(0) = 0`.

    Parameters
    ----------
    probabilities:
        Array of shape (k,) or (m, k). If (k,), it is treated as (1, k).

    Returns
    -------
    entropies:
        Array of shape (m,), where entropies[i] = H(probabilities[i, :]).
    """
    p: FloatArray = _as_2d_prob_array(x=probabilities, name="probabilities")

    with np.errstate(divide="ignore", invalid="ignore"):
        logp: FloatArray = np.where(p > 0.0, np.log2(p), 0.0)
        entropies: FloatArray = -np.sum(a=p * logp, axis=1)

    return np.asarray(a=entropies, dtype=np.float64)

def jensen_shannon_divergence(p: FloatArray, q: FloatArray) -> FloatArray:
    """
    Computes a Jensen–Shannon divergence (JSD) matrix between the entries of two batches of probability vectors.

    JSD is defined as:

    .. math::

        \\operatorname{JSD}(p[i], q[j]) = H\\left(\\tfrac{p[i] + q[j]}{2}\\right) - \\tfrac{H(p[i]) + H(q[j])}{2}

    where H(·) denotes the Shannon entropy in bits.

    This implementation is robust to zero probabilities in p and q.

    Parameters
    ----------
    p:
        Array of shape (k,) or (m1, k).
    q:
        Array of shape (k,) or (m2, k).

    Returns
    -------
    jsd:
        Array of shape (m1, m2), where jsd[i, j] = JSD(p[i], q[j]) in bits. If p and q were both of shape (k,), the
        output is of shape (1, 1).

    Raises
    ------
    ValueError
        If the input arrays have different trailing dimensions (k).
    """
    p_mat: FloatArray = _as_2d_prob_array(x=p, name="p")
    q_mat: FloatArray = _as_2d_prob_array(x=q, name="q")

    if p_mat.shape[1] != q_mat.shape[1]:
        raise ValueError(f"Probability vectors must have same k; got {p_mat.shape[1]} and {q_mat.shape[1]}.")

    # Entropies
    entropy_p: FloatArray = _entropy_base2(probabilities=p_mat)  # (m1,)
    entropy_q: FloatArray = _entropy_base2(probabilities=q_mat)  # (m2,)

    # Mixtures: shape (m1, m2, k)
    m: FloatArray = 0.5 * (p_mat[:, np.newaxis, :] + q_mat[np.newaxis, :, :])

    # Entropy of mixtures: flatten to (m1*m2, k) then reshape back
    m1, m2, k = m.shape
    entropy_m: FloatArray = _entropy_base2(probabilities=m.reshape(m1 * m2, k)).reshape(m1, m2)

    jsd: FloatArray = entropy_m - 0.5 * (entropy_p[:, np.newaxis] + entropy_q[np.newaxis, :])

    # Numerical guard: tiny negatives may appear
    jsd = np.maximum(jsd, 0.0)

    return np.asarray(a=jsd, dtype=np.float64)

def jensen_shannon_distance(p: FloatArray, q: FloatArray) -> FloatArray:
    """
    Compute the Jensen–Shannon distance (JSd) matrix between the entries of two batches of probability vectors.

    The Jensen–Shannon distance is defined as the square root of the Jensen–Shannon divergence:

    .. math::

        \\operatorname{JSd}(p[i], q[j]) = \\sqrt{\\operatorname{JSD}(p[i], q[j])}.

    With base-2 logarithms, JSd takes values in the interval [0, 1].

    Parameters
    ----------
    p:
        Array of shape (k,) or (m1, k).
    q:
        Array of shape (k,) or (m2, k).

    Returns
    -------
    jsd_distance:
        Array of Jensen–Shannon distance in bits (square root of divergence). If p and q were both of shape (k,), the
        output is of shape (1, 1).
    """
    return np.sqrt(jensen_shannon_divergence(p=p, q=q))

def minimum_js_distance(
    candidates: FloatArray,
    null_probabilities: FloatArray,
    jsd_fn: Callable[[FloatArray, FloatArray], FloatArray] | None = None,
) -> tuple[FloatArray, IntArray]:
    """
    Compute the minimum Jensen–Shannon distance of a candidate batch to a set of nulls.

    Given a probability vector `candidate` and a collection of null probability vectors `(p_ℓ)_{ℓ=1}^L`, this function
    computes:

    .. math::

        \\mathrm{mJSd}(candidate) = \\min_{\\ell \\in [L]} \\operatorname{JSd}(candidate, p_{\\ell}),

    and returns both the value of the minimum distance and the index ℓ* at which it is reached.

    Parameters
    ----------
    candidates:
        Array of shape (k,) or (m, k). If (k,), treated as (1, k).

    null_probabilities:
        Two-dimensional numpy array of shape (L, k) containing the null probability vectors. All rows must have the
        same length as `candidate`.

    jsd_fn:
        Function that returns a distance matrix of shape (m, L) given candidates (m,k) and null_probabilities (L,k). If
        None, uses this module's `jensen_shannon_distance`.

    Returns
    -------
    A tuple (min_distances, argmin_indexes). The first term is the minimum distance between the candidates and the rows
    of `null_probabilities`. The second element is the array of indexes ℓ* (0-based) of the null probability vectors
    that reach the minimum distance associdated with each candidate.

    Raises
    ------
    ValueError
        If the shapes of the inputs are incompatible.
    """
    cand: FloatArray = _as_2d_prob_array(x=candidates, name="candidates")
    nulls: FloatArray = np.asarray(a=null_probabilities, dtype=np.float64)

    if nulls.ndim != 2:
        raise ValueError(f"`null_probabilities` must be 2D (L,k); got shape {nulls.shape}.")
    if cand.shape[1] != nulls.shape[1]:
        raise ValueError(f"Dimension mismatch: candidates k={cand.shape[1]} vs nulls k={nulls.shape[1]}.")

    dist_fn: Callable[[FloatArray, FloatArray], FloatArray] = jensen_shannon_distance if jsd_fn is None else jsd_fn

    dist_mat: FloatArray = np.asarray(a=dist_fn(cand, nulls), dtype=np.float64)  # expected (m, L)
    if dist_mat.ndim != 2 or dist_mat.shape != (cand.shape[0], nulls.shape[0]):
        raise ValueError(
            "Distance function returned incompatible shape: "
            f"expected {(cand.shape[0], nulls.shape[0])}, got {dist_mat.shape}."
        )

    argmin: IntArray = np.argmin(a=dist_mat, axis=1).astype(dtype=np.int64, copy=False)
    mins: FloatArray = dist_mat[np.arange(dist_mat.shape[0]), argmin].astype(dtype=np.float64, copy=False)
    return mins, argmin


def estimate_type_i_error(decisions: IntArray, reject_all_label: int = REJECT_DECISION) -> float:
    """
    Estimate Type-I error from decisions simulated under a fixed null hypothesis.

    In the multi-null setting of the paper, Type-I error for a given null hypothesis H0^ℓ is defined as:

    .. math::

        \\alpha_{n, \\ell} = P(\\varphi_n(H) = -1 \\mid H_0^\\ell),

    i.e., the probability that the decision rule returns the "reject all" label when H0^ℓ is true.

    This function estimates that quantity from a sequence of decisions produced under a fixed null.

    Parameters
    ----------
    decisions:
        One-dimensional numpy array of integer decisions, typically of shape (num_replications,). The decisions must
        all correspond to simulations under the same true null hypothesis.
    reject_all_label:
        Integer label representing the decision "reject all null hypotheses". Defaults to `REJECT_DECISION` (-1).

    Returns
    -------
    Empirical Type-I error estimate, i.e., the fraction of decisions that are equal to `reject_all_label`.
    """
    dec: FloatArray = np.asarray(a=decisions, dtype=np.int64)
    if dec.ndim != 1:
        raise ValueError(f"`decisions` must be a one-dimensional array; got shape {dec.shape}.")
    if dec.size == 0:
        raise ValueError("`decisions` array is empty; cannot estimate Type-I error.")

    num_rejections: int = int(np.sum(dec == reject_all_label))
    return float(num_rejections / dec.size)


def estimate_power(decisions: IntArray, reject_all_label: int = REJECT_DECISION) -> float:
    """
    Estimate power from decisions simulated under a fixed alternative.

    In the multi-null setting of the paper, power for an alternative base probability q is defined as:

    .. math::

        \\pi_n^q = P(\\varphi_n(H) = -1 \\mid H_1^q),

    i.e., the probability that the decision rule returns the "reject all" label when the true model is not any of the
    nulls.

    This function estimates that quantity from a sequence of decisions produced under a fixed alternative.

    Parameters
    ----------
    decisions:
        One-dimensional numpy array of integer decisions, typically of shape (num_replications,). The decisions must
        all correspond to simulations under the same true alternative hypothesis.
    reject_all_label:
        Integer label representing the decision "reject all null hypotheses". Defaults to `REJECT_DECISION` (-1).

    Returns
    -------
    power_hat:
        Empirical power estimate, i.e., the fraction of decisions that are equal to `reject_all_label`.
    """
    # Power has the same estimator as Type-I error, but under an alternative.
    return estimate_type_i_error(decisions=decisions, reject_all_label=reject_all_label)


def confusion_matrix_from_simulations(
    true_labels: IntArray,
    predicted_labels: IntArray,
    label_order: Sequence[int] | None = None,
) -> IntArray:
    """
    Build a confusion matrix for multi-null decisions (with rejection).

    This helper function is intended for summarizing the output of Monte Carlo experiments where decisions are made
    among L null hypotheses and a "reject all" label.

    Parameters
    ----------
    true_labels:
        One-dimensional integer array of shape (N,) containing the true labels for each simulated histogram. Values may
        include any integers, e.g., 1..L for null hypotheses and -1 for explicit alternative bins, depending on the
        experiment design.
    predicted_labels:
        One-dimensional integer array of shape (N,) containing the corresponding predicted decision labels. Values are
         usually in {1, ..., L, REJECT_DECISION}.
    label_order:
        Optional explicit ordering of labels to define the rows and columns of the confusion matrix. If None, the union
        of labels present in `true_labels` and `predicted_labels` is used, sorted in ascending numerical order.

    Returns
    -------
    confusion_matrix:
        Two-dimensional numpy array of shape (n_labels, n_labels), where `confusion_matrix[i, j]` is the number of
        samples with true label `labels[i]` and predicted label `labels[j]`, and `labels` is the effective label
        ordering used.

        The label ordering can be recovered as:

        >>> label_ordering = (
        ...     np.array(label_order)
        ...     if label_order is not None else
        ...     np.unique(np.concatenate([true_labels, predicted_labels]))
        ... )

    Raises
    ------
    ValueError
        If the input arrays have incompatible shapes or are not one-dimensional.
    """
    true_arr: IntArray = np.asarray(a=true_labels, dtype=np.int64)
    pred_arr: IntArray = np.asarray(a=predicted_labels, dtype=np.int64)

    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"`true_labels` and `predicted_labels` must have the same shape; "
            f"got {true_arr.shape} and {pred_arr.shape}."
        )
    if true_arr.ndim != 1:
        raise ValueError(f"`true_labels` and `predicted_labels` must be one-dimensional; got ndim={true_arr.ndim}.")
    if true_arr.size == 0:
        raise ValueError("Input label arrays are empty; cannot build confusion matrix.")

    if label_order is None:
        labels: IntArray = np.unique(ar=np.concatenate([true_arr, pred_arr]))
    else:
        labels = np.asarray(a=label_order, dtype=np.int64)

    n_labels: int = labels.shape[0]
    label_to_index: dict[int, int] = {int(lbl): idx for idx, lbl in enumerate(labels)}  # noqa

    cm: IntArray = np.zeros(shape=(n_labels, n_labels), dtype=np.int64)

    for t, p in zip(true_arr, pred_arr, strict=True):
        i: int = label_to_index[int(t)]
        j: int = label_to_index[int(p)]
        cm[i, j] += 1

    return cm
