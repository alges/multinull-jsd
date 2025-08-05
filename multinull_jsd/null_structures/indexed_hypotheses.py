"""
A thin wrapper around ``list`` that stores *NullHypothesis* objects and assigns **stable, consecutive integer indices**
to them. The container enforces invariants (index validity, recycling policy) that would otherwise clutter the
high-level logic inside the core package.
"""
from multinull_jsd.null_structures.null_hypothesis import NullHypothesis
from multinull_jsd.cdf_backends import CDFBackend
from multinull_jsd.types import FloatArray
from typing import Iterator


class IndexedHypotheses:
    """
    Container that assigns stable integer indices to ``NullHypothesis`` instances and exposes list-like access.

    The data structure ensures **index continuity**: deleted indices are recycled only when the container is empty,
    mimicking the behaviour expected by users in statistical software.

    Parameters
    ----------
    cdf_backend
        Back-end shared by all contained null hypotheses.
    """
    def __init__(self, cdf_backend: CDFBackend) -> None:
        raise NotImplementedError

    def add_null(self, prob_vector: FloatArray, target_alpha: float) -> int:
        """
        Append a new null and return its index.

        Parameters
        ----------
        prob_vector
            Probability vector (1-D, non-negative, sums to one).
        target_alpha
            Desired significance level in :math:`[0,1]`.

        Returns
        -------
        int
            One-based index assigned to the new null.
        """
        raise NotImplementedError

    def __getitem__(self, idx) -> NullHypothesis:
        raise NotImplementedError

    def __delitem__(self, idx) -> None:
        raise NotImplementedError

    def __contains__(self, prob_vector: FloatArray) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Iterator[NullHypothesis]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
