"""
Exact CDF backend
=================

Exhaustively enumerates all histograms in the non-normalized histogram space :math:`\\Delta'_{k,n}` to obtain the exact
distribution of JSd.

Complexity
----------
* For fixed :math:`k`: :math:`O(n^{k-1})` (stars-and-bars).
* For fixed :math:`n`: :math:`O(k^n)`.

Notes
-----
* Enumeration should be **cached per probability vector** so repeated calls avoid re-computation.
"""
from .base import CDFBackend

from multinull_jsd._validators import validate_probability_vector
from multinull_jsd.types import FloatArray, CDFCallable


class ExactCDFBackend(CDFBackend):
    """
    Exhaustively enumerates all histograms in the non-normalized histogram space :math:`\\Delta'_{k,n}` to obtain the
    exact distribution of JSd.

    Complexity
    ----------
    :math:`O(n^{k-1})` for fixed :math:`k` (stars-and-bars enumeration) or :math:`O(k^n)` for fixed :math:`n`.

    Parameters
    ----------
    evidence_size
        Number of samples :math:`n`. See ``CDFBackend`` for details.

    Notes
    -----
    Enumeration is cached **per probability vector** so repeated calls with the same vector avoid re-computation.
    """
    def __init__(self, evidence_size: int):
        super().__init__(evidence_size=evidence_size)
        raise NotImplementedError

    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        validate_probability_vector(name="prob_vector", value=prob_vector, n_categories=None)
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
