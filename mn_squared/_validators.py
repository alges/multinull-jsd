"""
**Internal validation helpers** used across the *mn-squared* code-base. The module contains **only light-weight,
side-effect-free checks** so that importing it never triggers heavy numerical work (NumPy is imported lazily and only
for datatype inspection).
"""
from mn_squared.types import FloatArray, FloatDType, IntArray, IntDType, ScalarInt, ScalarFloat, TNumber
from typing import Any, Optional

import numpy.typing as npt
import numpy as np

import numbers


Number = ScalarInt | ScalarFloat

FLOAT_TOL: float = 1e-12


def validate_bounded_value(
    name: str,
    value: TNumber,
    min_value: Optional[ScalarInt | ScalarFloat] = None,
    max_value: Optional[ScalarInt | ScalarFloat] = None
) -> TNumber:
    """
    Check that the given value is a real number within the defined bounds (inclusive).

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    min_value
        Optional lower bound (inclusive). If not provided, no lower bound is enforced.
    max_value
        Optional upper bound (inclusive). If not provided, no upper bound is enforced.

    Raises
    ------
    TypeError
        If *value* is not a number.
    ValueError
        If *value* is outside the defined bounds or if the bounds are inconsistent (e.g., `min_value > max_value`).
    """
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number. Got {type(value).__name__}.")
    if min_value is not None and max_value is not None and float(min_value) > float(max_value):
        raise ValueError(f"Inconsistent bounds for {name}: min_value ({min_value}) > max_value ({max_value}).")
    if min_value is not None and float(value) < float(min_value):
        raise ValueError(f"{name} must be at least {min_value}. Got {value!r}.")
    if max_value is not None and float(value) > float(max_value):
        raise ValueError(f"{name} must be at most {max_value}. Got {value!r}.")
    return value


def validate_int_value(name: str, value: Any, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    """
    Check that the given value is an integer within the defined bounds (inclusive).

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    min_value
        Optional lower bound (inclusive). If not provided, no lower bound is enforced.
    max_value
        Optional upper bound (inclusive). If not provided, no upper bound is enforced.

    Raises
    ------
    TypeError
        If *value* is not an ``int``.
    ValueError
        If *value* is outside the defined bounds (or if the bounds are inconsistent, e.g., `min_value > max_value`).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        # bool is a subclass of int, so we need to exclude it explicitly
        raise TypeError(f"{name} must be an integer. Got {type(value).__name__}.")
    return int(validate_bounded_value(name=name, value=int(value), min_value=min_value, max_value=max_value))


def validate_finite_array(name: str, value: Any) -> npt.NDArray:
    """
    Check that the given value is a numeric array-like object with finite entries.

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.

    Raises
    ------
    TypeError
        If *value* is not a numeric array-like object.
    ValueError
        If *value* contains non-finite values (NaN or Inf).

    Returns
    -------
    npt.NDArray
        The validated array, converted to a numpy array.
    """
    array: npt.NDArray = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a numeric array-like object. Got {array.dtype.name}.")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must be real-valued, not complex.")
    if array.dtype == np.bool_:
        raise TypeError(f"{name} must not be a boolean array-like object.")
    if not np.all(a=np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values; not NaN or Inf.")
    return array


def validate_non_negative_batch(name: str, value: Any, n_categories: Optional[int]) -> npt.NDArray:
    """
    Check that the given value is a non-negative 1-D or 2-D array-like object.

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    n_categories
        Expected number of categories (columns) in the array. Every row must have exactly this many entries. If not
        provided, the number of categories is not checked.

    Raises
    ------
    TypeError
        If *value* is not a numeric array-like object.
    ValueError
        If *value* is not a 1-D or 2-D array-like object, if it does not have exactly *n_categories* columns, or if it
        contains negative values.

    Returns
    -------
    npt.NDArray
        The validated array, converted to a numpy array.
    """
    array: npt.NDArray = validate_finite_array(name=name, value=value)
    if array.ndim == 1:
        array = np.expand_dims(a=array, axis=0)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D array-like object.")
    if n_categories is not None:
        n_categories = validate_int_value(name="n_categories", value=n_categories, min_value=1)
        if array.shape[1] != n_categories:
            raise ValueError(f"{name} must have exactly {n_categories} columns. Got {array.shape[1]}.")
    if np.any(a=array < 0):
        raise ValueError(f"{name} must contain non-negative values.")
    return array


def validate_probability_vector(name: str, value: Any, n_categories: Optional[int]) -> FloatArray:
    """
    Check that the given value is a non-negative 1-D array-like object representing a probability distribution.

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    n_categories
        Expected number of categories (entries) in the probability distribution. The array must have exactly this many
        entries. If not provided, the number of categories is not checked.

    Raises
    ------
    TypeError
        If *value* is not a numeric array-like object.
    ValueError
        If *value* is not a 1-D array-like object ((k, ) or (1, k) shaped), if it does not have exactly *n_categories*
        entries, if it contains negative values, or if it does not sum to one.

    Returns
    -------
    FloatArray
        The validated probability vector, converted to a numpy array.
    """
    if n_categories is not None:
        n_categories = validate_int_value(name="n_categories", value=n_categories, min_value=1)
    value = validate_probability_batch(name=name, value=value, n_categories=n_categories)
    if value.shape[0] != 1:
        raise ValueError(f"{name} must be a 1-D array-like object.")
    return value[0]


def validate_probability_batch(name: str, value: Any, n_categories: Optional[int]) -> FloatArray:
    """
    Check that the given value is a non-negative 1-D or 2-D array-like object representing a probability distribution
    or a batch of probability distributions.

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    n_categories
        Expected number of categories (columns) in the probability distribution. Every row must have exactly this many
        entries. If not provided, the number of categories is not checked.

    Raises
    ------
    TypeError
        If *value* is not a numeric array-like object.
    ValueError
        If *value* is not a 1-D or 2-D array-like object, if it does not have exactly *n_categories* columns, if it
        contains negative values, or if the rows do not sum to one.

    Returns
    -------
    FloatArray
        The validated probability batch, converted to a numpy array.
    """
    if n_categories is not None:
        n_categories = validate_int_value(name="n_categories", value=n_categories, min_value=1)
    array: npt.NDArray = validate_non_negative_batch(name=name, value=value, n_categories=n_categories)
    if not np.allclose(a=np.sum(a=array, axis=1), b=1.0, atol=FLOAT_TOL, rtol=0.0):
        raise ValueError(f"{name} must contain probability distributions that sum to one in each row.")
    return array.astype(dtype=FloatDType)


def validate_histogram_batch(name: str, value: Any, n_categories: int, histogram_size: int) -> IntArray:
    """
    Check that the given value is a non-negative 1-D or 2-D array-like object representing a histogram or a batch of
    histograms.

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    n_categories
        Expected number of categories (columns) in the histogram. Every row must have exactly this many entries.
    histogram_size
        Expected number of samples in each histogram. This is the number of draws :math:`n` in the multinomial model.

    Raises
    ------
    TypeError
        If *value* is not a numeric array-like object.
    ValueError
        If *value* is not a 1-D or 2-D array-like object, if it does not have exactly *n_categories* columns, or if it
        contains negative values.

    Returns
    -------
    npt.NDArray
        The validated histogram batch, converted to a numpy array.
    """
    n_categories = validate_int_value(name="n_categories", value=n_categories, min_value=1)
    histogram_size = validate_int_value(name="histogram_size", value=histogram_size, min_value=1)
    array: npt.NDArray = validate_non_negative_batch(name=name, value=value, n_categories=n_categories)
    if (
        not np.issubdtype(array.dtype, np.integer)
        and np.any(a=~np.isclose(a=array, b=np.floor(array), atol=FLOAT_TOL, rtol=0.0))
    ):
        raise ValueError(f"{name} must contain histograms with integer counts in each row.")
    int_array: IntArray = array.astype(dtype=IntDType)
    if np.any(a=int_array.sum(axis=1) != histogram_size):
        raise ValueError(f"{name} must contain histograms with exactly {histogram_size} samples in each row.")
    return int_array


def validate_null_indices(name: str, value: Any, n_nulls: int, keep_duplicates: bool) -> tuple[ScalarInt, ...]:
    """
    Check that the given value is a sequence of integers representing null indices.

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    n_nulls
        Total number of null hypotheses in the container. The indices must be in the integer interval [1,n_nulls].
    keep_duplicates
        If ``True``, duplicate indices are kept in the returned tuple; if ``False``, duplicates are removed.

    Raises
    ------
    TypeError
        If *value* is not a sequence of integers.
    ValueError
        If the sequence contains indices outside the range [1,n_nulls].

    Returns
    -------
    tuple[ScalarInt, ...]
        A tuple of validated indices.
    """
    n_nulls = validate_int_value(name="n_nulls", value=n_nulls, min_value=0)
    if n_nulls == 0:
        raise ValueError("There should be at least one null hypothesis in the container (n_nulls > 0).")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or an iterable of integers. Got {type(value).__name__}.")
    value_seq: tuple[int, ...]
    if isinstance(value, numbers.Integral):
        value_seq = (int(value),)
    elif isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an integer or an iterable of integers. Got {type(value).__name__}.")
    else:
        try:
            value_seq = tuple(value)
        except TypeError:
            raise TypeError(f"{name} must be an integer or an iterable of integers. Got {type(value).__name__}.")
    value_list: list[int] = list()
    idx: int
    for idx in value_seq:
        if idx not in value_list or keep_duplicates:
            value_list.append(
                validate_int_value(name=f"{idx} in {name}", value=idx, min_value=1, max_value=n_nulls)
            )
    return tuple(value_list)


def validate_null_slice(name: str, value: Any, n_nulls: int) -> slice:
    """
    Check that the given value is a slice object representing null indices.

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    n_nulls
        Total number of null hypotheses in the container. The slice must be valid within the range [1,n_nulls].

    Raises
    ------
    TypeError
        If *value* is not a slice object.
    ValueError
        If the slice is invalid or contains indices outside the range [1,n_nulls].

    Returns
    -------
    slice
        A validated slice object with adjusted start and stop indices.
    """
    n_nulls = validate_int_value(name="n_nulls", value=n_nulls, min_value=0)
    if n_nulls == 0:
        raise ValueError("There should be at least one null hypothesis in the container (n_nulls > 0).")
    if not isinstance(value, slice):
        raise TypeError(f"{name} must be a slice object. Got {type(value).__name__}.")

    start: int = 1 if value.start is None else validate_int_value(
        name=f"{name}.start", value=value.start, min_value=1, max_value=n_nulls + 1
    )
    stop: int = n_nulls + 1 if value.stop is None else validate_int_value(
        name=f"{name}.stop", value=value.stop, min_value=1, max_value=n_nulls + 1
    )
    step: int = 1 if value.step is None else validate_int_value(
        name=f"{name}.step", value=value.step, min_value=1
    )

    return slice(start, stop, step)
