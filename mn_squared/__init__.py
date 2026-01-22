"""
mn_squared
=============

Python implementation of the *Multi-Null Jensen-Shannon Distance (JSd) hypothesis test*.

Public re-export
----------------
``MNSquaredTest``
    High-level interface that wraps null-hypothesis management, JSd statistic calculation, p-value inference,
    decision-making, and operating-characteristic inspection (Type-I and Type-II error rates).

Notes
-----
The sub-packages

* :py:mod:`mn_squared.cdf_backends`
* :py:mod:`mn_squared.null_structures`

provide pluggable CDF estimation back-ends and internal data structures. These remain available for advanced users via
the normal import path.
"""
from importlib.metadata import PackageNotFoundError, version as _dist_version
from typing import Any, TYPE_CHECKING

__all__ = ["MNSquaredTest", "available_cdf_backends", "__version__"]

if TYPE_CHECKING:
    from .core import MNSquaredTest


def available_cdf_backends() -> tuple[str, ...]:
    """
    List the names of available CDF backends.

    Returns
    -------
    tuple of str
        Names of available CDF backends.
    """
    from .cdf_backends import NON_MC_CDF_BACKEND_FACTORY, MC_CDF_BACKEND_FACTORY
    return tuple(sorted(NON_MC_CDF_BACKEND_FACTORY.keys() | MC_CDF_BACKEND_FACTORY.keys()))


def __getattr__(name: str) -> Any:
    if name == "MNSquaredTest":
        from .core import MNSquaredTest as _MNSquaredTest
        globals()["MNSquaredTest"] = _MNSquaredTest
        return _MNSquaredTest
    if name == "__version__":
        v: str = _package_version()
        globals()["__version__"] = v
        return v
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def _package_version() -> str:
    """
    Determines the version of the package by inspecting the distribution name.

    This function attempts to retrieve the version of the package from available distribution names. If the package is
    not installed, it will return a fallback version "0.0.0". The function uses a set of candidate names derived from
    the current module's name and checks for the corresponding distribution name to find the version.

    Raises
    ------
    PackageNotFoundError:
        If the candidate package name is not found.

    Returns
    -------
    The version string of the package if found, otherwise "0.0.0".
    """
    candidates: tuple[str, ...] = ("mn-squared", __name__, __name__.replace("_", "-"))
    for dist_name in candidates:
        try:
            return _dist_version(distribution_name=dist_name)
        except PackageNotFoundError:
            continue
    # If the package isn't installed (e.g., running from a source checkout without pip install), avoid crashing.
    return "0.0.0"

