"""
Small, shared utility helpers.

This module provides narrow, reusable helpers that are intentionally dependency-light and safe to import from anywhere
in the package. Current utilities include:

- `make_generator`: normalize "seed or RNG" inputs into a `numpy.random.Generator`.
- `validate_probability_vector`: sanity-check that a 1D vector lies on the simplex.

Design
------
These helpers are meant to reduce boilerplate and keep validation / RNG conventions consistent across experiments,
samplers, and test implementations.
"""
import numpy as np

from settings import FloatArray, IntArray


def make_generator(seed_or_rng: int | np.random.Generator | None = None) -> np.random.Generator:
    """
    Create or normalize a numpy random number generator.

    This helper centralizes the pattern of accepting either a seed (int), an existing `np.random.Generator`, or `None`,
    and always returning an `np.random.Generator` instance.

    Parameters
    ----------
    seed_or_rng:
        - If an int, a new `np.random.default_rng(seed_or_rng)` is created.
        - If a `np.random.Generator`, it is returned as-is.
        - If None, a new `np.random.default_rng()` is created with non-deterministic seeding.

    Returns
    -------
    A `numpy.random.Generator` instance suitable for use in simulations.
    """
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed=seed_or_rng)


def validate_probability_vector(probabilities: FloatArray, atol: float = 1e-9) -> None:
    """
    Validate that an array represents a probability vector on the simplex.

    This function checks that the input is one-dimensional, has non-negative entries, and sums (approximately) to one
    within a given tolerance. A `ValueError` is raised if any of these conditions is violated.

    Parameters
    ----------
    probabilities:
        One-dimensional numpy array of dtype float64 representing the probability vector.
    atol:
        Absolute tolerance for the check `abs(sum(probabilities) - 1) <= atol`.

    Raises
    ------
    ValueError
        If the vector has the wrong shape, contains negative entries, or does not sum to one within the specified
        tolerance.
    """
    if probabilities.ndim != 1:
        raise ValueError(f"Probability vector must be one-dimensional; got shape {probabilities.shape}.")
    if np.any(a=probabilities < 0.0):
        raise ValueError("Probability vector must have non-negative entries.")
    total: float = float(np.sum(probabilities))
    if not np.isfinite(total):
        raise ValueError("Probability vector sum is not finite.")
    if not np.isclose(a=total, b=1.0, atol=atol):
        raise ValueError(f"Probability vector must sum to 1 within tolerance {atol}; got sum {total}.")


def int_logspace_unique(n: int, start: int = 1, stop: int = 10_000) -> list[int]:
    """
    Generate a strictly increasing sequence of unique integers spaced roughly logarithmically over [start, stop].

    The routine creates a log-spaced grid between start and stop, rounds to integers, and enforces strict
    monotonicity while keeping the first and last elements fixed at the bounds. For extreme densities where
    unique log-rounded integers cannot be guaranteed, it falls back to a linear fill that respects the bounds.

    Parameters
    ----------
    n
        Number of points to generate. Must be >= 1 and not exceed the size of the integer range
        (stop - start + 1).
    start
        Lower bound of the range (inclusive).
    stop : int, default=10000
        Upper bound of the range (inclusive).

    Returns
    -------
    list[int]
        A list of length n with strictly increasing integers from start to stop (inclusive),
        approximately logarithmically spaced.

    Raises
    ------
    ValueError
        If n < 1, start > stop, or if n is larger than the number of available unique integers in [start, stop].

    Notes
    -----
    - For n == 1 returns [start]; for n == 2 returns [start, stop].
    - The internal enforcement pass guarantees uniqueness and bounds after rounding/clipping.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if start > stop:
        raise ValueError("start must be <= stop")
    if n > (stop - start + 1):
        raise ValueError(f"Need {n} unique ints, but only {stop - start + 1} exist in [{start}, {stop}].")

    if n == 1:
        return [start]
    if n == 2:
        return [start, stop]

    # initial (float) logspace, then round to ints
    vals: IntArray = np.rint(np.logspace(start=np.log10(start), stop=np.log10(stop), num=n)).astype(dtype=int)
    vals[0], vals[-1] = start, stop
    vals = np.clip(a=vals, a_min=start, a_max=stop)

    # enforce strict increasing (unique) while keeping bounds
    for _ in range(10):  # a few passes is enough in practice
        for i in range(1, n):
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + 1
        if vals[-1] > stop:
            vals[-1] = stop
        for i in range(n - 2, -1, -1):
            if vals[i] >= vals[i + 1]:
                vals[i] = vals[i + 1] - 1
        vals = np.clip(a=vals, a_min=start, a_max=stop)
        vals[0], vals[-1] = start, stop

    # final safety fallback (only triggers for extreme n): linear fill
    if np.unique(ar=vals).size != n:
        vals = np.arange(start=start, stop=start + n, dtype=int)
        vals[-1] = min(vals[-1], stop)  # noqa

    return vals.tolist()
