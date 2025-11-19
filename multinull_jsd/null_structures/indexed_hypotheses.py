"""
A thin wrapper around ``list`` that stores *NullHypothesis* objects and assigns **consecutive integer indices** to
them. The container enforces invariants (index validity, recycling policy) that would otherwise clutter the high-level
logic inside the core package.

Indexing policy
---------------
* Public indices are **1-based** and consecutive: 1,2,3,...,L.
* On index deletion, subsequent indices are shifted left, preserving continuity.
* ``__getitem__`` and ``__delitem__`` expect 1-based integers, slices, or iterables.
"""
from multinull_jsd.null_structures.null_hypothesis import NullHypothesis
from multinull_jsd.cdf_backends import CDFBackend
from multinull_jsd._validators import (
    validate_int_value, validate_probability_vector, validate_bounded_value, validate_null_indices, validate_null_slice
)
from multinull_jsd.types import ScalarInt
from typing import Iterator, Iterable, Any, overload

import numpy.typing as npt


class IndexedHypotheses:
    """
    Container that assigns integer indices to ``NullHypothesis`` instances and exposes list-like access.

    The data structure ensures **index continuity**: deleted indices are shifted left, and new indices are assigned
    consecutively, starting from 1. This allows for efficient lookups and deletions without gaps in the index space.

    Parameters
    ----------
    cdf_backend
        Back-end shared by all contained null hypotheses.
    prob_dim
        Number of categories (``k``) in each probability vector.

    Notes
    -----
    * Public indices are **1-based**.
    """
    def __init__(self, cdf_backend: CDFBackend, prob_dim: int) -> None:
        if not isinstance(cdf_backend, CDFBackend):
            raise TypeError("cdf_backend must be an instance of CDFBackend.")
        self._k: int = validate_int_value(name="prob_dim", value=prob_dim, min_value=1)
        raise NotImplementedError

    def add_null(self, prob_vector: npt.ArrayLike, target_alpha: float) -> ScalarInt:
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
        ScalarInt
            One-based index assigned to the new null.
        """
        validate_probability_vector(name="prob_vector", value=prob_vector, n_categories=self._k)
        validate_bounded_value(name="target_alpha", value=target_alpha, min_value=0.0, max_value=1.0)
        raise NotImplementedError

    @overload
    def __getitem__(self, idx: ScalarInt) -> NullHypothesis: ...
    @overload
    def __getitem__(self, idx: slice | Iterable[ScalarInt]) -> list[NullHypothesis]: ...

    def __getitem__(self, idx: Any) -> NullHypothesis | list[NullHypothesis]:
        if isinstance(idx, slice):
            validate_null_slice(name="idx", value=idx, n_nulls=len(self))
        else:
            validate_null_indices(name="idx", value=idx, n_nulls=len(self), keep_duplicates=True)
        raise NotImplementedError

    def __delitem__(self, idx: Any) -> None:
        if isinstance(idx, slice):
            validate_null_slice(name="idx", value=idx, n_nulls=len(self))
        else:
            validate_null_indices(name="idx", value=idx, n_nulls=len(self), keep_duplicates=False)
        raise NotImplementedError

    def __contains__(self, null_item: Any) -> bool:
        if not isinstance(null_item, NullHypothesis):
            validate_probability_vector(name="null_item", value=null_item, n_categories=self._k)
        raise NotImplementedError

    def __iter__(self) -> Iterator[NullHypothesis]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
