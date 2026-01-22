"""
Unit tests for the validate_null_indices and validate_null_slice functions.
"""
# noinspection PyProtectedMember
from mn_squared._validators import validate_null_indices, validate_null_slice

from typing import TypeAlias

import numpy as np
import pytest

ScalarInt: TypeAlias = int | np.integer


def test_validate_null_indices_requires_positive_n_nulls() -> None:
    """
    Test that validate_null_indices rejects n_nulls == 0 (must have at least one null).
    """
    with pytest.raises(expected_exception=ValueError):
        validate_null_indices(name="idx", value=1, n_nulls=0, keep_duplicates=False)


def test_validate_null_indices_accepts_single_int_and_returns_tuple() -> None:
    """
    Test that validate_null_indices accepts a single integer and returns a tuple with that index.
    """
    output: tuple[ScalarInt, ...] = validate_null_indices(name="idx", value=1, n_nulls=3, keep_duplicates=False)
    assert output == (1,)


def test_validate_null_indices_accepts_iterables_and_deduplicates_preserving_order() -> None:
    """
    Test that validate_null_indices accepts an iterable and deduplicates indices preserving first occurrence order.
    """
    output: tuple[ScalarInt, ...] = validate_null_indices(
        name="idx", value=[2, 1, 2, 3, 1], n_nulls=3, keep_duplicates=False
    )
    assert output == (2, 1, 3)


def test_validate_null_indices_keep_duplicates_true_preserves_duplicates() -> None:
    """
    Test that validate_null_indices preserves duplicates when keep_duplicates=True.
    """
    output: tuple[ScalarInt, ...] = validate_null_indices(
        name="idx", value=[2, 1, 2, 3, 1], n_nulls=3, keep_duplicates=True
    )
    assert output == (2, 1, 2, 3, 1)


def test_validate_null_indices_bounds_and_type_checks() -> None:
    """
    Test that validate_null_indices enforces 1-based bounds and rejects bad types.
    """
    # Out-of-bounds
    with pytest.raises(expected_exception=ValueError):
        validate_null_indices(name="idx", value=0, n_nulls=3, keep_duplicates=False)
    with pytest.raises(expected_exception=ValueError):
        validate_null_indices(name="idx", value=4, n_nulls=3, keep_duplicates=False)

    # Bad types: bool, str, bytes
    with pytest.raises(expected_exception=TypeError):
        validate_null_indices(name="idx", value=True, n_nulls=3, keep_duplicates=False)
    with pytest.raises(expected_exception=TypeError):
        validate_null_indices(name="idx", value="1,2", n_nulls=3, keep_duplicates=False)
    with pytest.raises(expected_exception=TypeError):
        validate_null_indices(name="idx", value=b"\x01\x02", n_nulls=3, keep_duplicates=False)

    # Iterable like numpy array is accepted
    assert validate_null_indices(
        name="idx", value=np.array([1, 3], dtype=np.int64), n_nulls=3, keep_duplicates=False
    ) == (1, 3)


def test_validate_null_slice_fills_defaults() -> None:
    """
    Test that validate_null_slice fills None start/stop/step with 1, n_nulls+1, and 1 respectively.
    """
    slice_object: slice = validate_null_slice(name="slice_object", value=slice(None, None, None), n_nulls=5)
    assert (slice_object.start, slice_object.stop, slice_object.step) == (1, 6, 1)


def test_validate_null_slice_accepts_explicit_bounds() -> None:
    """
    Test that validate_null_slice accepts explicit in-range start/stop/step.
    """
    slice_object = validate_null_slice(name="slice_object", value=slice(2, 5, 2), n_nulls=5)
    assert (slice_object.start, slice_object.stop, slice_object.step) == (2, 5, 2)


def test_validate_null_slice_rejects_out_of_bounds_and_bad_step() -> None:
    """
    Test that validate_null_slice rejects out-of-bounds indices and invalid step values.
    """
    # start below 1
    with pytest.raises(expected_exception=ValueError):
        validate_null_slice(name="s", value=slice(0, 3, 1), n_nulls=5)
    # stop above n_nulls
    with pytest.raises(expected_exception=ValueError):
        validate_null_slice(name="s", value=slice(1, 7, 1), n_nulls=5)
    # step <= 0
    with pytest.raises(expected_exception=ValueError):
        validate_null_slice(name="s", value=slice(1, 5, 0), n_nulls=5)
    with pytest.raises(expected_exception=ValueError):
        validate_null_slice(name="s", value=slice(1, 5, -1), n_nulls=5)


def test_validate_null_slice_rejects_non_slice() -> None:
    """
    Test that validate_null_slice rejects non-slice objects.
    """
    with pytest.raises(expected_exception=TypeError):
        validate_null_slice(name="s", value="1:3", n_nulls=5)
