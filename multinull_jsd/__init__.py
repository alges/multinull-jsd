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
from .core import MultiNullJSDTest

__all__ = ["MultiNullJSDTest"]
