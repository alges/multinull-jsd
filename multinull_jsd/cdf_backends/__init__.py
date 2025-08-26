"""
cdf_backends package
====================

Pluggable cumulative-distribution-function estimators supporting the Multi-Null JSd test.

Each back-end is a subclass of `multinull_jsd.cdf_backends.base.CDFBackend` and is automatically selected via the
``cdf_method`` argument in ``multinull_jsd.core.MultiNullJSDTest``.
"""
from .mc_multinomial import MultinomialMCCDFBackend
from .mc_normal import NormalMCCDFBackend
from .exact import ExactCDFBackend
from .base import CDFBackend

from typing import Callable, Final

#: Factory for deterministic (non-MC) CDF backends.
NON_MC_CDF_BACKEND_FACTORY: Final[dict[str, Callable[[int], CDFBackend]]] = {
    "exact": lambda n: ExactCDFBackend(evidence_size=n)
}
#: Factory for Monte-Carlo (MC) CDF backends.
MC_CDF_BACKEND_FACTORY: Final[dict[str, Callable[[int, int, int], CDFBackend]]] = {
    "mc_multinomial": lambda n, m, s: MultinomialMCCDFBackend(evidence_size=n, mc_samples=m, seed=s),
    "mc_normal": lambda n, m, s: NormalMCCDFBackend(evidence_size=n, mc_samples=m, seed=s)
}

__all__ = [
    "NON_MC_CDF_BACKEND_FACTORY", "MC_CDF_BACKEND_FACTORY", "CDFBackend", "ExactCDFBackend", "NormalMCCDFBackend",
    "MultinomialMCCDFBackend"
]
