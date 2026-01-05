# `experiments/`

This folder contains the **experiment harness** used to benchmark and validate the multi-null JSd hypothesis test
implementation against multiple-testing baselines.

The experiments are designed to be:

- **Scenario-based**: each scenario specifies the number of null hypotheses $L$, the probability dimensionality $k$, 
  null-generation settings, target values of mJSd for the probabilities of the alternative hypotheses, sample sizes
  $n$, and evaluation budgets ($m$, specifically, `m_null`, `m_alt` for the null and alternative hypotheses).
- **Reproducible**: seeds are recorded in outputs; optional Monte Carlo backends record `mc_seed` and `mc_samples`.
  Some baselines do not allow for exact reproducibility.
- **Fair across methods**: for each scenario and `n`, the framework generates **shared histogram draws** (under nulls
  and alternatives) so all methods are evaluated on the same data.

## What this module does

At a high level, for each scenario, it performs the following steps:

1. Samples a set of $L$ null multinomial probability vectors $(\mathbf{p}_1,\ldots,\mathbf{p}_{L})$ on the $k$-simplex
   ($\Delta_k$).
2. Constructs alternative distributions $\mathbf{q}$ aiming to match requested **mJSd targets** (minimum JS distance to
   the null set).
3. For each $n$ in the scenario grid, samples:
   * `m_null` histograms under each null, and
   * `m_alt` histograms under each alternative.
4. Runs each method and records:
   * Decision counts/proportions (long format),
   * Empirical error metrics ($\alpha$/FWER under null; power/$\beta$ under alternatives),
   * Backend-reported estimates (when available),
   * Runtime breakdowns.

## Output layout (`experiments/results/`)

The experiment runner writes outputs under:

- `experiments/results/experiment.csv`
  **Main aggregated results file** (long format; includes metadata + decision proportions + summary metrics + runtime
  fields).

Additionally, many runs also write scenario artifacts under a scenario-specific prefixes:

- **Null cache**: the sampled null probability vectors: `[scenario_prefix]_null_probabilities.npy`.
- **Alternatives cache**: the constructed alternatives table with metadata: `[scenario_prefix]_alternatives.pkl`.
- **Optional histogram arrays**: cached null/alternative histograms when you enable caching for heavy runs:
  `[scenario_prefix]_histograms[n].npz`.

> The CSV and cached nulls/alternatives are intended to be sufficient for an article-grade analysis.

---

## `experiment.csv` schema

The CSV primary key is `scenario_name`, `method`, `n`, `true_kind`, `true_id`, `decision`.

The remainder of this document describes the columns in the CSV.

### CSV Columns

- `scenario_name`: Human-readable name of the scenario (the experimental design point). Expect the following values:

  - `Scenario 1 — Balanced`.
  - `Scenario 2 — Unbalanced`.
  - `Scenario 3 — Border/Extreme`.
  - `Scenario 4 — Heterogeneous-alpha`.

- `method`: Method label; all baselines are extended with the Bonferroni-Holm methodology to address multiple nulls.
  Expect the following values:

  - `MultinullJSD`: our method.
  - `Chi2-Pearson+Holm`: chi-square test.
  - `G-test-LLR+Holm`: G-test; log-likelihood ratio test.
  - `MMD-Gaussian+Holm`: MMD test: Gaussian kernel.
  - `MMD-Laplacian+Holm`: MMD test: Laplacian kernel.
  - `ExactMultinom-Prob+Holm`: exact multinomial test: probability distribution.
  - `ExactMultinom-Chisq+Holm`: exact multinomial test: chi-square.
  - `ExactMultinom-LLR+Holm`: exact multinomial test: log-likelihood ratio.
  
  For `scenario_name == "Scenario 1 — Balanced"`, expect only `MultinullJSD` to be present.

- `n`: Histogram sample size $n$ (number of multinomial trials / total count per histogram).

- `true_kind`: Data-generating regime: `"null"` or `"alt"` for null and alternative hypotheses, respectively.

- `true_id`: ID of the true generating distribution:
  - if `true_kind=="null"`: which null index: a number in $\{1,\ldots,L\}$, where $L$ is the number of nulls associated
    with the scenario.
  - if `true_kind=="alt"`: which alternative ID: a natural number.

- `decision`: Method decision code for a histogram. By convention:
  - $-1$: reject all nulls (declare alternative)
  - $1$ to $L$: accept/choose null of the expressed index.

- `count`: Number of histograms that produced this `decision`.

- `prop`: Proportion of histograms producing this `decision` (computed as `count / m_used`).

- `m`: Number of histograms simulated for this condition (valued as `m_null` for null rows, `m_alt` for alternative
  rows).

- `scenario_slug`: Filesystem-safe identifier derived from `scenario_name`.

- `alpha_global`: Target global family-wise error rate (FWER) level for the scenario.

- `mjsd_target`: Target minimum Jensen-Shannon distance used to design the alternative distribution.
F
- `mjsd`: Achieved minimum JS distance of the evaluated alternative to the null set.

- `mjsd_error`: Difference between achieved and targeted mJSd (`mjsd - mjsd_target`).

- `closest_null`: Index ($1$ to $L$) of the null distribution closest to the alternative (in JS distance).

- `L`: Number of null hypotheses in the family $L$.

- `k`: Number of multinomial categories (dimension: $k$).

- `m_used`: Number of histograms actually used after any filtering/skips (e.g., method constraints or chunk failures).

- `alpha_target_per_null`: Scenario-specified per-null target $\alpha_{\ell}$ level. Null for alt rows (rows such that
  `true_kind == "alt"`.

- `alpha_target_method`: The per-null $\alpha_{\ell}$ actually used by the method after any method-specific
  mapping/adjustments. NaN for alt rows.

- `alpha_hat_empirical`: Empirical type I error estimate from null runs. NaN for alt rows.

- `alpha_ci_low`: Lower confidence bound for `alpha_hat_empirical`.

- `alpha_ci_high`: Upper confidence bound for `alpha_hat_empirical`.

- `alpha_backend`: Backend-computed type I error estimate (if available).

- `fwer_backend`: Backend-computed FWER estimate (if available).

- `p_correct`: Empirical probability of correct behavior:

  - under null: selecting the true null (`decision == true_id`)
  - not defined for alternatives (NaN).

- `p_reject_all`: Empirical probability of rejecting all nulls (`decision == -1`). NaN for alternatives.

- `p_misclass`: Empirical probability of choosing the wrong null. NaN for alternatives.

- `cdf_method`: Backend/CDF computation mode (e.g., exact vs Monte Carlo-based approximation). Expect values `exact`,
  `mc_multinomial` or `mc_normal` for `method == MultiNullJSD`; else, expect NaN.

- `mc_samples`: Number of Monte Carlo samples used by the backend (when applicable: `cdf_method == mc_multinomial` or
  `cdf_method == mc_normal`).

- `mc_seed`: Random seed used for backend Monte Carlo (when applicable: `cdf_method == mc_multinomial` or
  `cdf_method == mc_normal`).

- `power_hat_empirical`: Empirical power estimate under alternatives. NaN for null rows (rows such that
  `true_kind == "null"`).

- `power_ci_low`: Lower confidence bound for `power_hat_empirical`. NaN for null rows.

- `power_ci_high`: Upper confidence bound for `power_hat_empirical`. NaN for null rows.

- `type2_hat_empirical`: Empirical type II error. NaN for null rows.

- `beta_backend`: Backend-computed type II error estimate (if available). NaN for null rows.

- `power_backend`: Backend-computed power estimate (if available). NaN for null rows and when `method != MultiNullJSD`.

- `runtime_infer_s`: Time spent evaluating decisions on histograms (inference phase).

- `runtime_per_hist_s`: Average inference time per histogram.

- `runtime_make_test_s`: Time spent constructing/initializing the test object. NaN for `method != MultiNullJSD`.

- `runtime_backend_s`: Time spent inside the backend computation (exact/MC CDF/p-values). NaN for
  `method != MultiNullJSD`.

- `runtime_total_method_s`: Total time attributable to the method for the scenario+`n` (setup + backend + inference).

- `m_null`: Configured number of histograms per null.

- `m_alt`: Configured number of histograms per alternative.

- `runtime_total_s`: Total wall-clock time for the overall scenario run at that `n` (including orchestration).

- `fwer_hat_empirical`: Empirical FWER estimate from null runs (project-defined aggregation across nulls).

- `fwer_target_method`: The global FWER target level the method is meant to control (usually equals `alpha_global`).

- `n_jobs`: Number of parallel workers requested/used.

- `n_chunks`: Number of chunks used to batch work for parallelism.

- `parallel_backend`: Parallel backend identifier (e.g., multiprocessing/joblib backend name). Expect `serial`,
  `threads`, or `processes`.

- `show_progress`: Whether progress reporting was enabled.

- `mp_start_method`: Multiprocessing start method (`spawn`, `forkserver`, `fork`, etc.).
