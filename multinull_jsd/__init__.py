"""
multinull_jsd
=============

Python implementation of the *Multi-Null Jensen-Shannon Distance (JSd) hypothesis test*.

Public re-export
----------------
``MultiNullJSDTest``
    High-level interface that wraps null-hypothesis management, JSd statistic calculation, p-value inference,
    decision-making, and operating-characteristic inspection (Type-I and Type-II error rates).

Notes
-----
The sub-packages

* :pymod:`multinull_jsd.cdf_backends`
* :pymod:`multinull_jsd.null_structures`

provide pluggable CDF estimation back-ends and internal data structures. These remain available for advanced users via
the normal import path.
"""
from typing import Any, TYPE_CHECKING

__all__ = ["MultiNullJSDTest", "available_cdf_backends"]

if TYPE_CHECKING:
    from .core import MultiNullJSDTest

def available_cdf_backends() -> tuple[str, ...]:
    """
    List the names of available CDF backends.

    Returns
    -------
    tuple of str
        Names of available CDF backends.
    """
    from .cdf_backends import CDF_BACKEND_FACTORY
    return tuple(sorted(CDF_BACKEND_FACTORY.keys()))

def __getattr__(name: str) -> Any:
    if name == "MultiNullJSDTest":
        from .core import MultiNullJSDTest as _MultiNullJSDTest
        return _MultiNullJSDTest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
