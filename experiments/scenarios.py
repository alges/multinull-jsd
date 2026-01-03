"""
Scenario configuration and sampling utilities for multi-null experiments.

Defines dataclasses describing:
- how to sample families of null probability vectors (Dirichlet dense/sparse, border-like),
- per-method sampling plans over n grids and replication counts, and
- complete experiment scenarios (null config, alpha vector, alternative targets, and backend settings).

Also provides primitives to sample Dirichlet probabilities and to generate null probability matrices according to the
configured family.
"""
from dataclasses import dataclass, field

import numpy as np

from settings import M_ALTERNATIVE_SAMPLES, M_MONTE_CARLO, FloatArray, IntArray
from utils import make_generator, validate_probability_vector
from enum import Enum, auto


TRUE_KINDS: tuple[str, ...] = ("null", "alt")


class NullFamily(Enum):
    """
    Enumeration of probability families for sampling null hypotheses.

    The intent is to capture high-level regimes of probability vectors
    that are relevant for experiments (e.g., "dense", "sparse", "border").

    Members
    -------
    UNIFORM_DIRICHLET
        Null probabilities are sampled from a symmetric Dirichlet distribution with concentration parameter α >= 1,
        which yields relatively "dense" probability vectors.

    SPARSE_DIRICHLET
        Null probabilities are sampled from a symmetric Dirichlet distribution with concentration parameter α in
        (0, 1), which yields relatively "spiky" or "sparse" probability vectors.

    BORDER_EXTREME
        Null probabilities are sampled near the vertices of the simplex, i.e., one category carries most of the mass,
        and the remaining categories share a small residual probability.
    """
    UNIFORM_DIRICHLET = auto()
    SPARSE_DIRICHLET = auto()
    BORDER_EXTREME = auto()


@dataclass(slots=True)
class NullSamplingConfig:
    """
    Configuration for sampling a collection of null probability vectors.

    This data class summarizes the key parameters that define a small "scenario" of null hypotheses H0^1, ..., H0^L.

    Attributes
    ----------
    num_categories:
        Number of categories `k` in each probability vector. Must be >= 1.
    num_nulls:
        Number of null hypotheses `L` to sample.
    family:
        High-level family of probability vectors to sample. See `NullFamily` for semantics. Defaults to
        `NullFamily.UNIFORM_DIRICHLET`.
    dirichlet_alpha:
        Concentration parameter α for the symmetric Dirichlet distribution used in `UNIFORM_DIRICHLET` and
        `SPARSE_DIRICHLET` families.
        - For `UNIFORM_DIRICHLET`, it is typical to use α = 1.0 or α > 1.0.
        - For `SPARSE_DIRICHLET`, it is typical to use α in (0, 1).
        This value is ignored for `BORDER_EXTREME`.
    border_high_mass:
        For the `BORDER_EXTREME` family, probability mass assigned to each of the high-mass categories (see
        `border_num_high`). The total mass assigned to high-mass categories is `border_num_high * border_high_mass`,
        and must be strictly less than 1. Ignored for other families.
    border_num_high:
        Number of categories that will receive the high mass `border_high_mass`. These indices are chosen uniformly at
        random without replacement among the `num_categories` positions.
    border_num_zeros:
        Number of categories that will be forced to have a probability of exactly zero. These indices are chosen
        uniformly at random without replacement among the remaining positions (i.e., disjoint from the high-mass
        indices). The leftover categories (those that are neither high-mass nor zero) share the remaining probability
        mass in a random, non-uniform way (via a Dirichlet draw). Ignored for other families.
    """
    num_categories: int
    num_nulls: int
    family: NullFamily = NullFamily.UNIFORM_DIRICHLET
    dirichlet_alpha: float = 1.0
    border_high_mass: float = 0.9
    border_num_high: int = 1
    border_num_zeros: int = 0


@dataclass(slots=True)
class SamplingPlan:
    """
    Sampling plan across a grid of sample sizes.

    This data class specifies:
    - The grid of n values to evaluate.
    - The default number of replications M per truth kind ("null" or "alt").
    - Optional per-point overrides for specific (n, truth-kind) pairs.

    Attributes
    ----------
    n_grid:
        List of sample sizes n to simulate.
    m_by_true_kind:
        Mapping with keys "null" and/or "alt" indicating the number of histograms/replications to generate per point
        for each truth kind.
    m_overrides:
        Optional mapping whose key is (n, true_kind) and whose value is the number of replications to use for that
        specific point, overriding `m_by_true_kind`.
    """
    n_grid: list[int]
    m_by_true_kind: dict[str, int]
    m_overrides: dict[tuple[int, str], int] = field(default_factory=dict)

    def m_for(self, n: int, true_kind: str) -> int:
        """
        Return the number of replications M for a given n and truth kind.

        If an override exists in `m_overrides` for (n, true_kind), it is used;
        otherwise the value for `true_kind` in `m_by_true_kind` is returned.

        Parameters
        ----------
        n:
            Sample size.
        true_kind:
            Truth kind; must be "null" or "alt".

        Returns
        -------
        int
            Number of replications to use.

        Raises
        ------
        ValueError
            If `true_kind` is not one of {"null", "alt"}.
        """
        if true_kind not in TRUE_KINDS:
            raise ValueError(f"Invalid true_kind={true_kind!r}")
        return int(self.m_overrides.get((int(n), true_kind), self.m_by_true_kind[true_kind]))


def _plan_lookup(plans: dict[str, SamplingPlan], method_name: str) -> SamplingPlan:
    """
    Resolve a SamplingPlan for a given method name.

    Resolution order:
    1) Exact match on the key.
    2) Prefix match: if a key ends with '*', it is treated as a prefix wildcard.
       For example, the key 'ExactMultinom-*' will match 'ExactMultinom-LLR+Holm'.

    Parameters
    ----------
    plans:
        Dictionary of plans indexed by method name or prefix with '*'.
    method_name:
        Full method name to resolve.

    Returns
    -------
    SamplingPlan
        The plan associated with the method.

    Raises
    ------
    KeyError
        If no plan is found via exact or prefix match.
    """
    if method_name in plans:
        return plans[method_name]
    for k, v in plans.items():
        if k.endswith("*") and method_name.startswith(k[:-1]):
            return v
    raise KeyError(
        f"No SamplingPlan found for method={method_name!r} (add it to scenario.method_plans)."
    )

@dataclass(slots=True)
class ExperimentScenario:
    """
    Configuration bundle for a full experiment scenario.

    A scenario samples one set of L null probability vectors in Δ_k, builds a set
    of alternatives (q) matched to a grid of target mJSd values, and evaluates a
    list of methods across a set of sample sizes n.

    This object is designed to support:
      - Exp02: Type-I control + strong consistency (log-spaced n grids).
      - Exp03: power curves as a function of mJSd and n (typically linear n grid + denser mJSd targets).
      - Exp04: runtime analysis by treating different JSD CDF backends as different "methods"
              (e.g., exact vs. MC-multinomial vs. MC-normal), plus baselines.

    Attributes
    ----------
    name:
        Human-readable identifier for the scenario.
    null_sampling_config:
        Controls how the null probabilities (p_ℓ)_{ℓ=1}^L are sampled.
    alpha_vector:
        Shape (L,) per-null target significance levels \bar{α}_ℓ.
    n_grid:
        Optional canonical n-grid; method-specific grids come from `method_plans`.
    method_plans:
        Dict mapping method name (or prefix wildcard ending with '*') to a SamplingPlan.
        This controls which n values each method runs on and how many replications (m_null/m_alt)
        are simulated per point.
    ignore_baselines:
        If True, only our method is run.
    mjsd_targets:
        1-D array of target mJSd levels used to construct alternatives.
    alt_num_candidate_samples:
        Number of candidates q sampled to match each mJSd target.
    alt_dirichlet_alpha:
        Dirichlet alpha used when sampling candidate alternative probabilities.
    """
    name: str
    null_sampling_config: NullSamplingConfig
    alpha_vector: FloatArray
    n_grid: list[int]

    method_plans: dict[str, SamplingPlan] = field(default_factory=dict)

    ignore_baselines: bool = False

    # Alternative selection
    mjsd_targets: FloatArray = None
    alt_num_candidate_samples: int = M_ALTERNATIVE_SAMPLES
    alt_dirichlet_alpha: float = 1.0

    cdf_method: str = "mc_multinomial"
    mc_samples: int | None = M_MONTE_CARLO
    mc_seed: int | None = None

    def __post_init__(self) -> None:
        """
        Validate required attributes.

        Raises
        ------
        ValueError
            If the `method_plans` attribute is not set.
        """
        if not self.method_plans:
            raise ValueError(
                "ExperimentScenario.method_plans must be provided "
                "(build_default_scenarios should set it)."
            )


def sample_dirichlet_probabilities(
    alpha: float | FloatArray,
    num_categories: int,
    num_samples: int = 1,
    rng: int | np.random.Generator | None = None,
) -> FloatArray:
    """
    Sample probability vectors from a symmetric or general Dirichlet distribution.

    Parameters
    ----------
    alpha:
        Concentration parameter(s) of the Dirichlet distribution:
        - If a float, a symmetric Dirichlet with parameter α is used, i.e., all coordinates share the same
          concentration.
        - If a one-dimensional numpy array of shape (num_categories,), it is used as the full parameter vector.
    num_categories:
        Number of categories `k`. Must be >= 1.
    num_samples:
        Number of independent probability vectors to draw. Must be >= 1.
    rng:
        Random number generator or seed. See `make_generator` for accepted values.

    Returns
    -------
    Two-dimensional numpy array of shape (num_samples, num_categories), where each row is a probability vector sampled
    from the requested Dirichlet distribution.

    Raises
    ------
    ValueError
        If `num_categories` or `num_samples` are invalid or if `alpha` has an incompatible shape.
    """
    if num_categories <= 0:
        raise ValueError(f"`num_categories` must be >= 1; got {num_categories}.")
    if num_samples <= 0:
        raise ValueError(f"`num_samples` must be >= 1; got {num_samples}.")

    rng_obj: np.random.Generator = make_generator(seed_or_rng=rng)
    alpha_arr: FloatArray = np.asarray(a=alpha, dtype=np.float64)

    if alpha_arr.ndim == 0:
        if alpha_arr <= 0.0:
            raise ValueError("Dirichlet concentration parameter α must be positive.")
        alpha_vec: FloatArray = np.full(shape=(num_categories,), fill_value=float(alpha_arr), dtype=np.float64)
    elif alpha_arr.ndim == 1:
        if alpha_arr.shape[0] != num_categories:
            raise ValueError(f"Alpha vector must have length {num_categories}; got {alpha_arr.shape[0]}.")
        if np.any(alpha_arr <= 0.0):
            raise ValueError("All Dirichlet concentration parameters must be positive.")
        alpha_vec = alpha_arr
    else:
        raise ValueError(
            "`alpha` must be a scalar or a one-dimensional array; got array with shape {alpha_arr.shape}."
        )

    probabilities: FloatArray = rng_obj.dirichlet(alpha=alpha_vec, size=num_samples)
    # Ensure float64 dtype
    return np.asarray(a=probabilities, dtype=np.float64)

def _sample_border_probability_vector(
    num_categories: int,
    high_mass: float,
    num_high: int,
    num_zeros: int,
    rng: np.random.Generator,
) -> FloatArray:
    """
    Internal helper: sample a probability vector near the border of the simplex.

    This function constructs a probability vector with the following structure:
    - `num_high` categories receive a fixed high mass equal to `high_mass` each. These indices are chosen uniformly at
      random without replacement.
    - `num_zeros` categories are forced to have probability exactly zero. These indices are also chosen uniformly at
      random without replacement among the remaining positions.
    - The remaining categories (those that are neither high-mass nor zero) share the remaining probability mass in a
      random, non-uniform way: a Dirichlet(1, ..., 1) vector is drawn on the corresponding coordinates and then scaled
      to the leftover mass.

    This produces probability vectors that are "border-like" (one or several dominant categories, some categories with
    exactly a zero probability) but with randomness in the tail instead of a uniform residual distribution.

    Parameters
    ----------
    num_categories:
        Number of categories `k`. Must be >= 1.
    high_mass:
        Probability mass assigned to each high-mass category. Must be strictly positive, and `num_high * high_mass`
        must be strictly less than 1.
    num_high:
        Number of high-mass categories. Must be >= 1 and `num_high + num_zeros` must be <= `num_categories`.
    num_zeros:
        Number of categories that will be forced to have a probability of exactly zero. Must be >= 0 and
        `num_high + num_zeros` must be <= `num_categories`.
    rng:
        Random number generator.

    Returns
    -------
    One-dimensional array of shape (num_categories,) representing the sampled probability vector.

    Raises
    ------
    ValueError
        If any of the constraints above is violated.
    """
    if num_categories <= 0:
        raise ValueError(f"`num_categories` must be >= 1; got {num_categories}.")
    if num_high <= 0:
        raise ValueError(f"`num_high` must be >= 1; got {num_high}.")
    if num_zeros < 0:
        raise ValueError(f"`num_zeros` must be >= 0; got {num_zeros}.")
    if num_high + num_zeros > num_categories:
        raise ValueError(
            "The sum `num_high + num_zeros` must not exceed `num_categories`; "
            f"got num_high={num_high}, num_zeros={num_zeros}, num_categories={num_categories}."
        )
    if not (0.0 < high_mass < 1.0):
        raise ValueError(f"`high_mass` must be in (0.0, 1.0); got {high_mass}.")

    total_high_mass: float = num_high * high_mass
    if total_high_mass >= 1.0:
        raise ValueError(
            "Total mass assigned to high-mass categories must be strictly less than 1; "
            f"got num_high * high_mass = {total_high_mass}."
        )

    remaining_mass: float = 1.0 - total_high_mass
    # Number of categories available for low-mass (non-zero, non-high) entries
    num_low: int = num_categories - num_high - num_zeros

    if num_low == 0 and remaining_mass > 0.0:
        raise ValueError(
            "No categories left for low-mass entries (num_low == 0) but remaining_mass > 0. "
            f"Ensure that num_high * high_mass == 1 when num_low == 0. Got remaining_mass={remaining_mass}."
        )
    if remaining_mass < 0.0:
        # This should not happen given the previous check, but we keep it explicit.
        raise ValueError(f"Remaining probability mass must be non-negative; got {remaining_mass}.")

    probabilities: FloatArray = np.zeros(shape=(num_categories,), dtype=np.float64)

    # Randomly permute indices to assign roles (high / zero / low)
    permuted_indices: IntArray = rng.permutation(num_categories).astype(np.int64, copy=False)

    high_indices = permuted_indices[:num_high]
    low_indices = permuted_indices[num_high + num_zeros :]

    # Assign high-mass entries
    probabilities[high_indices] = high_mass

    # Zero indices are already zero by initialization

    # Assign non-uniform low-mass tail via Dirichlet if there is a remaining mass
    if num_low > 0 and remaining_mass > 0.0:
        # Draw a Dirichlet(1, ..., 1) over the low indices
        low_dirichlet: FloatArray = rng.dirichlet(
            alpha=np.ones(shape=(num_low,), dtype=np.float64)
        ).astype(dtype=np.float64, copy=False)
        probabilities[low_indices] = remaining_mass * low_dirichlet

    # Optional final sanity check (tolerant to tiny numerical deviations)
    total: float = float(np.sum(probabilities))
    if not np.isclose(a=total, b=1.0, atol=1e-12):
        # Normalize defensively if numerical error accumulated
        probabilities /= total

    return probabilities

def generate_null_probabilities(
    config: NullSamplingConfig,
    rng: int | np.random.Generator | None = None,
) -> FloatArray:
    """
    Sample a collection of null probability vectors according to a configuration.

    This is a convenience wrapper that covers the three main families of null models used in the experiments: "dense"
    Dirichlet, "sparse" Dirichlet, and border/vertex-like probabilities.

    Parameters
    ----------
    config:
        Instance of `NullSamplingConfig` describing the number of categories, number of nulls, and the probability
        family to be used.
    rng:
        Random number generator or seed. See `make_generator` for accepted values.

    Returns
    -------
    Two-dimensional numpy array of shape `(config.num_nulls, config.num_categories)`. Each row is a sampled probability
    vector on the simplex.

    Raises
    ------
    ValueError
        If the configuration is inconsistent.
    """
    if config.num_categories <= 0:
        raise ValueError(f"`num_categories` must be >= 1; got {config.num_categories}.")
    if config.num_nulls <= 0:
        raise ValueError(f"`num_nulls` must be >= 1; got {config.num_nulls}.")

    rng_obj: np.random.Generator = make_generator(seed_or_rng=rng)
    k: int = config.num_categories
    num_nulls: int = config.num_nulls

    if config.family in (NullFamily.UNIFORM_DIRICHLET, NullFamily.SPARSE_DIRICHLET):
        probabilities: FloatArray = sample_dirichlet_probabilities(
            alpha=config.dirichlet_alpha,
            num_categories=k,
            num_samples=num_nulls,
            rng=rng_obj,
        )
    elif config.family is NullFamily.BORDER_EXTREME:
        probabilities = np.empty(shape=(num_nulls, k), dtype=np.float64)
        for idx in range(num_nulls):
            probabilities[idx] = _sample_border_probability_vector(
                num_categories=k,
                high_mass=config.border_high_mass,
                num_high=config.border_num_high,
                num_zeros=config.border_num_zeros,
                rng=rng_obj,
            )
    else:
        raise ValueError(f"Unsupported null family: {config.family!r}.")

    return probabilities


def sample_multinomial_histograms_for_null(
    base_probabilities: FloatArray,
    num_observations: int,
    num_histograms: int,
    rng: int | np.random.Generator | None = None,
) -> IntArray:
    """
    Sample multinomial histograms under a single null hypothesis.

    Given a base probability vector p in Δ_k and a sample size n, this function draws `num_histograms` independent
    histograms H^(i) ~ Multinomial(n, p).

    Parameters
    ----------
    base_probabilities:
        One-dimensional numpy array of shape (k,) representing the base probability vector p.
    num_observations:
        Number of observations n in each histogram. Must be >= 0. If n = 0, all histograms will be identically zero.
    num_histograms:
        Number of independent histograms to draw. Must be >= 1.
    rng:
        Random number generator or seed. See `make_generator` for accepted values.

    Returns
    -------
    Two-dimensional numpy array of shape (num_histograms, k) with dtype int64. Each row is a histogram whose entries
    sum to `num_observations`.

    Raises
    ------
    ValueError
        If any of the input parameters is invalid.
    """
    if num_histograms <= 0:
        raise ValueError(f"`num_histograms` must be >= 1; got {num_histograms}.")
    if num_observations < 0:
        raise ValueError(f"`num_observations` must be >= 0; got {num_observations}.")

    p: FloatArray = np.asarray(a=base_probabilities, dtype=np.float64)
    validate_probability_vector(probabilities=p)

    rng_obj: np.random.Generator = make_generator(seed_or_rng=rng)

    if num_observations == 0:
        # Degenerate multinomial: all mass at zero vector
        zeros: IntArray = np.zeros(shape=(num_histograms, p.shape[0]), dtype=np.int64)
        return zeros

    samples: IntArray = rng_obj.multinomial(n=num_observations, pvals=p, size=num_histograms)
    return np.asarray(a=samples, dtype=np.int64)


def sample_multinomial_histograms_for_nulls(
    base_probabilities: FloatArray,
    num_observations: int,
    num_histograms_per_null: int,
    rng: int | np.random.Generator | None = None,
) -> IntArray:
    """
    Sample multinomial histograms under each of several null hypotheses.

    This function is a thin wrapper over `sample_multinomial_histograms_for_null`
    that works with a collection of base probabilities `(p_ℓ)_{ℓ=1}^L`.

    Parameters
    ----------
    base_probabilities:
        Two-dimensional numpy array of shape (num_nulls, k) containing the base probability vectors for each null
        hypothesis H0^ℓ.
    num_observations:
        Number of observations n in each histogram. Must be >= 0.
    num_histograms_per_null:
        Number of histograms to draw for each null hypothesis. Must be >= 1.
    rng:
        Random number generator or seed. See `make_generator` for accepted values.

    Returns
    -------
    Three-dimensional numpy array of shape (num_nulls, num_histograms_per_null, k) with dtype int64. Entry
    `histograms[ℓ, i, :]` is the i-th histogram drawn under null ℓ and sums to `num_observations`.

    Raises
    ------
    ValueError
        If the shapes of the inputs are incompatible or any parameter is invalid.
    """
    nulls: FloatArray = np.asarray(a=base_probabilities, dtype=np.float64)
    if nulls.ndim != 2:
        raise ValueError(f"`base_probabilities` must be a 2D array of shape (L, k); got shape {nulls.shape}.")

    num_nulls, k = nulls.shape
    if num_nulls <= 0:
        raise ValueError("Number of null hypotheses L must be >= 1.")
    if k <= 0:
        raise ValueError("Number of categories k must be >= 1.")

    rng_obj: np.random.Generator = make_generator(seed_or_rng=rng)
    histograms: IntArray = np.empty(shape=(num_nulls, num_histograms_per_null, k), dtype=np.int64)

    for idx in range(num_nulls):
        histograms[idx] = sample_multinomial_histograms_for_null(
            base_probabilities=nulls[idx],
            num_observations=num_observations,
            num_histograms=num_histograms_per_null,
            rng=rng_obj,
        )

    return histograms
