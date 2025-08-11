"""
**Internal validation helpers** used across the *multinull-jsd* code-base. The module contains **only light-weight,
side-effect-free checks** so that importing it never triggers heavy numerical work (NumPy is imported lazily and only
for datatype inspection).
"""
from multinull_jsd.types import FloatArray, FloatDType, IntArray, IntDType
from typing import Any, Optional

import numpy.typing as npt
import numpy as np

import numbers


FLOAT_TOL: float = 1e-12

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
        If *value* is outside the defined bounds.
    """
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(f"Inconsistent bounds for {name}: min_value ({min_value}) > max_value ({max_value}).")
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        # bool is a subclass of int, so we need to exclude it explicitly
        raise TypeError(f"{name} must be an integer. Got {type(value).__name__}.")
    value = int(value)
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be at least {min_value}. Got {value!r}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be at most {max_value}. Got {value!r}.")
    return value

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

def validate_non_negative_batch(name: str, value: Any, n_categories: int) -> npt.NDArray:
    """
    Check that the given value is a non-negative 1-D or 2-D array-like object.

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    n_categories
        Expected number of categories (columns) in the array. Every row must have exactly this many entries.

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
    n_categories = validate_int_value(name="n_categories", value=n_categories, min_value=1)
    array: npt.NDArray = validate_finite_array(name=name, value=value)
    if array.ndim == 1:
        array = np.expand_dims(a=array, axis=0)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D array-like object.")
    if array.shape[1] != n_categories:
        raise ValueError(f"{name} must have exactly {n_categories} columns. Got {array.shape[1]}.")
    if np.any(a=array < 0):
        raise ValueError(f"{name} must contain non-negative values.")
    return array

def validate_probability_batch(name: str, value: Any, n_categories: int) -> FloatArray:
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
        entries.

    Raises
    ------
    TypeError
        If *value* is not a numeric array-like object.
    ValueError
        If *value* is not a 1-D or 2-D array-like object, if it does not have exactly *n_categories* columns, if it
        contains negative values, or if the rows do not sum to one.

    Returns
    -------
    npt.NDArray
        The validated probability batch, converted to a numpy array.
    """
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
    type_limit: int = np.iinfo(IntDType).max
    n_categories = validate_int_value(name="n_categories", value=n_categories, min_value=1)
    histogram_size = validate_int_value(
        name="histogram_size", value=histogram_size, min_value=1, max_value=int(type_limit)
    )
    array: npt.NDArray = validate_non_negative_batch(name=name, value=value, n_categories=n_categories)
    if (
        not np.issubdtype(array.dtype, np.integer)
        and np.any(a=~np.isclose(a=array, b=np.floor(array), atol=FLOAT_TOL, rtol=0.0))
    ):
        raise ValueError(f"{name} must contain histograms with integer counts in each row.")
    if np.any(a=array > type_limit):
        raise ValueError(
            f"{name} must contain histograms with counts that fit into {IntDType.__name__} (max {type_limit})."
        )
    int_array: IntArray = array.astype(dtype=IntDType)
    if np.any(a=int_array.sum(axis=1) != histogram_size):
        raise ValueError(f"{name} must contain histograms with exactly {histogram_size} samples in each row.")
    return int_array

def validate_null_indices(name: str, value: Any, n_nulls: int) -> tuple[int, ...]:
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

    Raises
    ------
    TypeError
        If *value* is not a sequence of integers.
    ValueError
        If the sequence contains indices outside the range [1,n_nulls].

    Returns
    -------
    tuple[int]
        A tuple of unique, validated indices.
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
        try :
            value_seq = tuple(value)
        except TypeError:
            raise TypeError(f"{name} must be an integer or an iterable of integers. Got {type(value).__name__}.")
    value_list: list[int] = list()
    for idx in value_seq:
        if idx not in value_list:
            value_list.append(validate_int_value(name=f"{idx} in {name}", value=idx, min_value=1, max_value=n_nulls))

    return tuple(value_list)
