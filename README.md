# Multi-Null Jensen–Shannon Distance (JSd) Hypothesis Test

`multinull-jsd`: An implementation for the multi-null (multiple null hypotheses) Jensen–Shannon Distance (JSd) based
Hypothesis Test. It computes the statistic, via a backend that may be exact or Monte-Carlo, and makes decisions
controlling per-null significances and the overall family-wise error rate.

a typed Python toolkit for managing multiple nulls with per-null α, computing JSd statistics, estimating CDFs (exact or Monte-Carlo), and making decisions with FWER/β insights.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-To%20be%20defined-red)
[![pre-commit](https://img.shields.io/badge/pre--commit-To%20be%20enabled-red?logo=pre-commit)](https://pre-commit.com/)

> **Status:** Pre-alpha API scaffold. Validators and tests are implemented; core orchestration and CDF backends are stubs raising `NotImplementedError`.

## Installation

```bash
pip install -e .
````

Python **≥ 3.10**, NumPy **≥ 1.24**.

## Quickstart (API scaffold)

```python
from multinull_jsd import MultiNullJSDTest

# NOTE: API scaffold — methods raise NotImplementedError until implemented.
test = MultiNullJSDTest(evidence_size=100, prob_dim=3, cdf_method="mc_multinomial", mc_samples=10_000, seed=1234)
test.add_nulls([0.5, 0.3, 0.2], target_alpha=0.05)
test.add_nulls([0.4, 0.4, 0.2], target_alpha=0.01)

histograms = [[55, 22, 23], [40, 39, 21], [0, 3, 97]]
p_vals = test.infer_p_values(histograms)
decisions = test.infer_decisions(histograms)
```

## Concepts

* **JSd statistic:** divergence between empirical histogram $H/n$ and a reference $p$.
* **Multi-null setting:** several candidate $p$ vectors; choose the least-rejected null or reject all.
* **Backends:**
  * `exact`: enumerate histograms in $\Delta'_{k,n}$ to compute the empirical CDF (ECDF); complexity $O(n^{k-1})$ for fixed $k$.
  * `mc_multinomial`: draw $H\sim \text{Multinomial}(n, p)$; ECDF converges by strong law of large numbers (SLLN).
  * `mc_normal`: CLT proxy with $\mathcal{N}(n p, n(\mathrm{diag}(p)-pp^\top))$.
* **Error metrics:** per-null $\alpha$; overall family-wise error rate (FWER); worst-case ($\beta$) at a query $q$.

## Public API

```python
from multinull_jsd import MultiNullJSDTest, available_cdf_backends
```

Advanced users may import:

```
multinull_jsd.cdf_backends     # ExactCDFBackend, NormalMCCDFBackend, MultinomialMCCDFBackend
multinull_jsd.null_structures  # IndexedHypotheses, NullHypothesis
```

## Performance & numerics

* Exact backend scales roughly as $O(n^{k-1})$; use MC for larger $n$ or $k$.
* Tolerance: `FLOAT_TOL = 1e-12` for sum-to-one and integer-like checks.
* MC paths will be deterministic under a fixed `seed`.

### Project layout

```
multinull_jsd/
  cdf_backends/        # CDF backends (exact, MC multinomial, MC normal)
  null_structures/     # NullHypothesis & IndexedHypotheses containers
  _validators.py       # Shared validation helpers (implemented)
  core.py              # MultiNullJSDTest orchestrator (stub)
tests/                 # Unit + property tests and backend contract tests
```

## Documentation

Local build:

```bash
cd docs
pip install -r requirements.txt  # or: pip install -e ".[docs]"
make html
# open _build/html/index.html
```

## Versioning & license

* Versioning: SemVer (pre-1.0 may introduce breaking changes).
* License: to be defined.

## Citation

There is an associated preprint describing the methodology being written up. In the meantime, if you use this project in research, please cite:

```bibtex
@software{multinull_jsd,
  title = {multinull-jsd: Multi-Null Jensen–Shannon Distance Hypothesis Test in Python},
  author = {ALGES},
  year = {2025},
  url = {https://github.com/alges/multinull-jsd-test}
}
```
