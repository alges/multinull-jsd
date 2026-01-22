# Multi-Null Non-Parametric (MN²) Jensen–Shannon Distance (JSd) Hypothesis Test (*MN-squared test*)

`mn-squared` implements a **m**ulti-**n**ull **n**on-parametric (MN²) Jensen–Shannon distance (JSd) based hypothesis
test for multinomial data. Given **multiple candidate null distributions** ($\mathbf{p}_{\ell}$) and observed
histograms ($\mathbf{h}$), it computes p-values using an exact or Monte-Carlo CDF backend and returns **decisions that
control per-null significance levels** and the overall **family-wise error rate (FWER)**.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
[![pre-commit](https://img.shields.io/badge/pre--commit-Enabled-success?logo=pre-commit)](https://pre-commit.com/)

> **Status:** **Stable (v1.0)** — public API is stable.

## Installation

```bash
pip install mn-squared
````

Python **≥ 3.10**, NumPy **≥ 1.24**.

## Quickstart

```python
from mn_squared import MNSquaredTest

test = MNSquaredTest(evidence_size=100, prob_dim=3, cdf_method="mc_multinomial", mc_samples=50_000, seed=1234)
test.add_nulls([0.5, 0.3, 0.2], target_alpha=0.05)
test.add_nulls([0.4, 0.4, 0.2], target_alpha=0.01)

histograms = [[55, 22, 23], [40, 39, 21], [0, 3, 97]]
p_vals = test.infer_p_values(histograms)
decisions = test.infer_decisions(histograms)

print("p-values:", p_vals)
print("decisions:", decisions)
```

## Concepts

* **JSd statistic:** divergence between empirical histogram ($\mathbf{h}/n$) and a reference ($\mathbf{p}$).
* **Multi-null setting:** several candidate vectors$. A query, $\mathbf{h}$, is either assigned to the “least-rejected”
  null or **rejects all**
* **Backends:**
  * `exact`: enumerate histograms in $\Delta'_{k,n}$ to compute the empirical CDF (ECDF); complexity $O(n^{k-1})$ for
    fixed $k$.
  * `mc_multinomial`: draw $\mathbf{H}\sim \text{Multinomial}(n, \mathbf{p})$; ECDF converges by strong law of large
    numbers (SLLN).
  * `mc_normal`: CLT proxy with $\mathcal{N}(n\mathbf{p}, n(\mathrm{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top))$.
* **Error control:** per-null $\alpha$; overall family-wise error rate (FWER); worst-case ($\beta$) at a query
  $\mathbf{q}$.

## Public API

```python
from mn_squared import MNSquaredTest, available_cdf_backends
```

Advanced imports:

```
mn_squared.cdf_backends     # ExactCDFBackend, NormalMCCDFBackend, MultinomialMCCDFBackend
mn_squared.null_structures  # IndexedHypotheses, NullHypothesis
```

## Performance & numerics

* Exact backend is intended for small-to-moderate ((n,k)) due to combinatorial growth.
* Monte-Carlo backends scale well and are recommended for larger regimes.
* Validations use a small floating tolerance for simplex / integer-like checks.
* MC results are **deterministic** under a fixed `seed`.

### Project layout

```
mn_squared/
  cdf_backends/        # CDF backends (exact, MC multinomial, MC normal)
  null_structures/     # NullHypothesis & IndexedHypotheses containers
  _validators.py       # Shared validation helpers (implemented)
  core.py              # MNSquaredTest orchestrator (stub)
tests/                 # Unit + property tests and backend contract tests
docs/                  # Documentation (Sphinx)
experiments/           # Optional: benchmarking / comparison scripts (if present)
```

## Versioning & license

* Versioning: **SemVer**.
* License: Apache-2.0.

## Citation

There is an associated preprint describing the methodology being written up. In the meantime, if you use this project in research, please cite:

```bibtex
@software{mn_squared,
  title = {mn-squared: Multi-Null Non-Parametric Jensen–Shannon Distance Hypothesis Test in Python},
  author = {ALGES},
  year = {2026},
  url = {https://github.com/alges/mn-squared}
}
```
