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

__all__ = ["CDFBackend", "ExactCDFBackend", "NormalMCCDFBackend", "MultinomialMCCDFBackend"]
