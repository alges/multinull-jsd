"""
null_structures package
=======================

Internal containers used by *multinull_jsd*.

This sub-package is **not** intended for end-users; the high-level class `multinull_jsd.core.MultiNullJSDTest`
re-exports everything needed for typical workflows. Still, advanced users may import directly for custom
pipelines:

>>> from multinull_jsd.null_structures import IndexedHypotheses, NullHypothesis
"""
from .indexed_hypotheses import IndexedHypotheses
from .null_hypothesis import NullHypothesis

__all__ = ["IndexedHypotheses", "NullHypothesis"]
