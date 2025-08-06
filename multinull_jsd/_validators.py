"""
**Internal validation helpers** used across the *multinull-jsd* code-base. The module contains **only light-weight,
side-effect-free checks** so that importing it never triggers heavy numerical work (NumPy is imported lazily and only
for datatype inspection).
"""
from typing import Any, Optional

import numpy.typing as npt
import numpy as np


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
        If the integer is not strictly positive.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer. Got {type(value).__name__}.")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be at least {min_value}. Got {value!r}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be at most {max_value}. Got {value!r}.")
    return value

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
    ValueError
        If *value* is not a 1-D or 2-D array-like object, if it does not have exactly *n_categories* columns, or if it
        contains negative values.

    Returns
    -------
    npt.NDArray
        The validated array, converted to a numpy array.
    """
    array: npt.NDArray = np.asarray(value)
    if array.ndim == 1:
        array = np.expand_dims(a=array, axis=0)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D array-like object.")
    if array.shape[1] != n_categories:
        raise ValueError(f"{name} must have exactly {n_categories} columns. Got {array.shape[1]}.")
    if np.any(a=array < 0):
        raise ValueError(f"{name} must contain non-negative values.")
    return array

def validate_probability_batch(name: str, value: Any, n_categories: int) -> npt.NDArray:
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
    ValueError
        If *value* is not a 1-D or 2-D array-like object, if it does not have exactly *n_categories* columns, if it
        contains negative values, or if the rows do not sum to one.

    Returns
    -------
    npt.NDArray
        The validated probability batch, converted to a numpy array.
    """
    array: npt.NDArray = validate_non_negative_batch(name=name, value=value, n_categories=n_categories)
    if not np.allclose(a=np.sum(a=array, axis=1), b=1.0, atol=FLOAT_TOL):
        raise ValueError(f"{name} must contain probability distributions that sum to one in each row.")
    return array

def validate_histogram_batch(name: str, value: Any, n_categories: int, histogram_size: int) -> npt.NDArray:
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
    ValueError
        If *value* is not a 1-D or 2-D array-like object, if it does not have exactly *n_categories* columns, or if it
        contains negative values.

    Returns
    -------
    npt.NDArray
        The validated histogram batch, converted to a numpy array.
    """
    array: npt.NDArray = validate_non_negative_batch(name=name, value=value, n_categories=n_categories)
    if np.any(a=array.sum(axis=1) != histogram_size):
        raise ValueError(f"{name} must contain histograms with exactly {histogram_size} samples in each row.")
    return array

def validate_null_index_set(name: str, value: Any, n_nulls: int) -> set[int]:
    """
    Check that the given value is a set of integers representing null indices.

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
        If *value* is not a set of integers.
    ValueError
        If the set contains indices outside the range [1,n_nulls].
    """
    if isinstance(value, int):
        value_set: set[int] = {value}
    else:
        try :
            value_set = set(value)
        except TypeError:
            raise TypeError(f"{name} must be an integer or a set of integers. Got {type(value).__name__}.")
    for idx in value_set:
        validate_int_value(name=f"{idx} in {name}", value=idx, min_value=1, max_value=n_nulls)
    return value_set
