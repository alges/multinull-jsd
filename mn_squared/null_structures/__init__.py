"""
null_structures package
=======================

Internal containers used by *mn_squared*.

This sub-package is **not** intended for end-users; the high-level class `mn_squared.core.MNSquaredTest` re-exports
everything needed for typical workflows. Still, advanced users may import directly for custom pipelines:

>>> from mn_squared.null_structures import IndexedHypotheses, NullHypothesis

Public re-exports
-----------------
- ``IndexedHypotheses`` – container that maintains 1-based consecutive indices
- ``NullHypothesis`` – lightweight holder for a single null hypothesis
"""
from typing import Any, TYPE_CHECKING

__all__ = ["IndexedHypotheses", "NullHypothesis"]

if TYPE_CHECKING:
    from .indexed_hypotheses import IndexedHypotheses
    from .null_hypothesis import NullHypothesis


def __getattr__(name: str) -> Any:
    if name == "IndexedHypotheses":
        from .indexed_hypotheses import IndexedHypotheses as _IndexedHypotheses
        return _IndexedHypotheses
    if name == "NullHypothesis":
        from .null_hypothesis import NullHypothesis as _NullHypothesis
        return _NullHypothesis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
