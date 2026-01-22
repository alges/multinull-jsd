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
from mn_squared.null_structures.null_hypothesis import NullHypothesis
from mn_squared.cdf_backends import CDFBackend
from mn_squared._validators import (
    validate_int_value, validate_probability_vector, validate_bounded_value, validate_null_indices, validate_null_slice
)
from mn_squared.types import ScalarInt, FloatArray
from typing import Iterator, Iterable, Any, overload

import numpy.typing as npt
import numpy as np


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
        self._backend: CDFBackend = cdf_backend
        self._nulls: list[NullHypothesis] = []

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
        nh: NullHypothesis = NullHypothesis(
            prob_vector=validate_probability_vector(name="prob_vector", value=prob_vector, n_categories=self._k),
            cdf_backend=self._backend
        )
        nh.set_target_alpha(
            target_alpha=validate_bounded_value(name="target_alpha", value=target_alpha, min_value=0.0, max_value=1.0)
        )
        self._nulls.append(nh)
        return len(self)

    @overload
    def __getitem__(self, idx: ScalarInt) -> NullHypothesis: ...
    @overload
    def __getitem__(self, idx: slice | Iterable[ScalarInt]) -> list[NullHypothesis]: ...

    def __getitem__(self, idx: Any) -> NullHypothesis | list[NullHypothesis]:
        n_nulls: int = len(self)
        if isinstance(idx, slice):
            validated_slice: slice = validate_null_slice(name="idx", value=idx, n_nulls=n_nulls)
            return self._nulls[validated_slice.start - 1:validated_slice.stop - 1:validated_slice.step]
        validated_idx_tuple: tuple[ScalarInt, ...] = validate_null_indices(
            name="idx", value=idx, n_nulls=n_nulls, keep_duplicates=True
        )
        if isinstance(idx, (int, np.integer)):
            return self._nulls[int(validated_idx_tuple[0]) - 1]
        return [self._nulls[int(i) - 1] for i in validated_idx_tuple]

    def __delitem__(self, idx: Any) -> None:
        n_nulls: int = len(self)
        if isinstance(idx, slice):
            validated_slice: slice = validate_null_slice(name="idx", value=idx, n_nulls=n_nulls)
            one_based_indices: list[int] = list(
                range(validated_slice.start, validated_slice.stop, validated_slice.step)
            )
        else:
            validated_idx_tuple: tuple[ScalarInt, ...] = validate_null_indices(
                name="idx", value=idx, n_nulls=n_nulls, keep_duplicates=False
            )
            one_based_indices = [int(i) for i in validated_idx_tuple]
        # Sorted deletion to preserve continuity
        for one_based_idx in sorted(one_based_indices, reverse=True):
            del self._nulls[one_based_idx - 1]

    def __contains__(self, null_item: Any) -> bool:
        if isinstance(null_item, NullHypothesis):
            return null_item in self._nulls
        try:
            null_p_vector: FloatArray = validate_probability_vector(
                name="null_item", value=null_item, n_categories=self._k
            )
        except (ValueError, TypeError):
            return False
        return any(np.array_equal(a1=nh.probability_vector, a2=null_p_vector) for nh in self._nulls)

    def __iter__(self) -> Iterator[NullHypothesis]:
        return iter(self._nulls)

    def __len__(self) -> int:
        return len(self._nulls)

    def __repr__(self) -> str:
        return f"IndexedHypotheses(size(L)={len(self)}, prob_dim(k)={self._k})"
