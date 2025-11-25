"""
Abstract backbone for CDF estimation backends.

Each concrete subclass must accept the **evidence size** :math:`n` at construction time and return a *callable* CDF for
a given null probability vector.

Design contract
---------------
* **Statelessness per call** – the object may cache expensive pre-computations (e.g., multinomial coefficient tables).
  but ``get_cdf`` must allow clean calls.
* **Thread-safety** – subclasses should not keep a mutable state that changes during evaluation of the returned
  callable.
* **CDF properties** – the callable returned by ``get_cdf`` MUST be:
  - vectorised (broadcasts over ``tau``),
  - monotone non-decreasing in ``tau``,
  - clipped to ``[0, 1]``.
"""
from multinull_jsd._validators import validate_int_value, validate_finite_array
from multinull_jsd.types import FloatDType, IntDType, FloatArray, IntArray, ScalarFloat, CDFCallable
from typing import Optional, cast
from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np


class CDFBackend(ABC):
    """
    Abstract interface for cumulative distribution function (CDF) backends.

    Parameters
    ----------
    evidence_size
        Number of draws :math:`n` in the multinomial model. Must be a positive integer.

    Raises
    ------
    TypeError
        If ``evidence_size`` is not an integer.
    ValueError
        If ``evidence_size`` is not positive.
    """
    def __init__(self, evidence_size: int) -> None:
        self._evidence_size: int = validate_int_value(name="evidence_size", value=evidence_size, min_value=1)

        # Cache for CDFs keyed by probability vector values (:math:`\\mathbf{p}`)
        self._cdf_cache: dict[tuple[float, ...], CDFCallable] = {}

    @property
    def evidence_size(self) -> int:
        """
        Returns the number of draws :math:`n` in the multinomial model. This is the number of samples in each
        histogram.

        Returns
        -------
        int
            The number of draws :math:`n`.
        """
        return self._evidence_size

    @staticmethod
    def _prob_vector_to_key(prob_vector: FloatArray) -> tuple[float, ...]:
        """
        Converts a probability vector into a hashable key represented as a tuple of floats. This key can then be used
        to uniquely identify the probability distribution in maps or dictionaries. The method ensures repeatable
        behavior for valid float arrays.

        Parameters
        ----------
        prob_vector
            The probability vector, represented as an array of floats.

        Returns
        -------
        A tuple containng the elements of the input `prob_vector`, preserving their numerical order for hashability and
        consistent mapping.
        """
        return tuple(float(x) for x in prob_vector)

    @staticmethod
    def _build_cdf_from_samples(distances: FloatArray, weights: Optional[FloatArray] = None) -> CDFCallable:
        """
        Build a piecewise-constant CDF callable from a set of sample distances and associated weights.

        Parameters
        ----------
        distances
            1-D array of JSd values.
        weights
            1-D array of non-negative weights with the same shape as ``distances``. If None, uniform weights are
            assumed (empirical CDF).

        Returns
        -------
        CDFCallable
            A callable F(tau) satisfying the contract:
            * scalar-in -> Python float,
            * array-in -> float64 ndarray with the same shape,
            * clipped to [0, 1],
            * monotone non-decreasing in tau.
        """
        distances = validate_finite_array(name="distances", value=distances).astype(dtype=FloatDType)
        if distances.ndim != 1:
            raise ValueError("distances must be a 1-D array.")
        if np.any(distances < 0.0) or np.any(distances > 1.0):
            raise ValueError("All entries in distances must lie in [0, 1].")
        m: int = distances.shape[0]
        if m == 0:
            raise ValueError("distances must contain at least one element.")

        if weights is None:
            # Uniform empirical CDF
            weights_arr: FloatArray = np.full(shape=m, fill_value=1.0 / m, dtype=FloatDType)
        else:
            weights_arr = validate_finite_array(name="weights", value=weights).astype(dtype=FloatDType)
            if weights_arr.shape != distances.shape:
                raise ValueError("weights and distances must have the same shape.")
            if np.any(weights_arr < 0.0):
                raise ValueError("All entries in weights must be non-negative.")
            total_weight: float = float(weights_arr.sum())
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                raise ValueError("weights must sum to a positive finite value.")
            weights_arr /= weights_arr.sum()

        # Sort by distance and build a cumulative sum of weights.
        order: IntArray = np.argsort(a=distances).astype(dtype=IntDType)
        distance_values: FloatArray = distances[order]
        cdf_values: FloatArray = np.clip(a=np.cumsum(weights_arr[order]), a_min=0.0, a_max=1.0)

        def cdf(tau: ScalarFloat | FloatArray) -> ScalarFloat | FloatArray:
            """
            Computes the cumulative distribution function (CDF) values for a given input.

            This function calculates the cumulative distribution function (CDF) using the input parameter tau. The
            parameter tau can be either a float or an array of floats. The returned value is an array of floats
            representing the calculated CDF values for the given input.

            Parameters
            ----------
            tau
                A float or an array of floats representing the input values for which the CDF is to be computed.

            Returns
            -------
            The CDF values for the given input.
            """
            if np.isscalar(tau):
                idx: int = int(np.searchsorted(distance_values, tau, side="right")) - 1
                if idx < 0:
                    return 0.0
                return float(cdf_values[idx])

            tau_array: FloatArray = np.asarray(tau, dtype=FloatDType)
            idx_array: IntArray = np.searchsorted(
                a=distance_values, v=tau_array, side="right"
            ).astype(dtype=IntDType) - 1

            out: FloatArray = np.zeros_like(a=tau_array, dtype=FloatDType)
            mask: npt.NDArray[np.bool_] = idx_array >= 0
            out[mask] = cdf_values[idx_array[mask]]
            return np.clip(a=out, a_min=0.0, a_max=1.0)

        return cast(CDFCallable, cdf)

    @abstractmethod
    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        """
        Returns a callable function
        :math:`F(\\tau) = \\mathbb{P}(\\mathrm{JSd}(\\mathbf{H}/n,\\mathbf{p}) \\leq \\tau)`, where
        :math:`\\mathbf{H}\\sim\\mathrm{Multinomial}(\\mathbf{p},n)`.

        Implementations may employ exact enumeration or approximations.

        Parameters
        ----------
        prob_vector
            1-D array of shape ``(k,)`` representing a probability vector. Every entry must lie in ``[0,1]`` and the
            vector must sum to one.

        Returns
        -------
        CDFCallable
            A monotone, vectorized cumulative-distribution function. Returned callable must accept either a Python
            scalar or a numpy array-like object and return a Python float or numpy array, respectively.

        Raises
        ------
        TypeError
            If *prob_vector* is not a numeric array-like object.
        ValueError
            If *prob_vector* is not 1-D, contains negative values, or does not sum to one.
        """

    @abstractmethod
    def __repr__(self) -> str:
        pass
