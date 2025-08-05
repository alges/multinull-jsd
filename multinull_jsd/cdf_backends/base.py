"""
Abstract backbone for CDF estimation backends.

Each concrete subclass must accept the **evidence size** :math:`n` at construction time and return a *callable* CDF for
a given null probability vector.

Design contract
---------------
* **Statelessness per call** – the object may cache expensive pre-computations (e.g. multinomial coefficient tables)
  but ``get_cdf`` must allow clean-calls.
* **Thread-safety** – subclasses should not keep mutable state that changes during evaluation of the returned callable.
"""
from multinull_jsd.types import FloatArray, CDFCallable
from abc import ABC, abstractmethod


class CDFBackend(ABC):
    """
    Abstract interface for cumulative distribution function (CDF) backends.

    Parameters
    ----------
    evidence_size
        Number of draws :math:`n` in the multinomial model. Must be a positive integer.

    Raises
    ------
    ValueError
        If ``evidence_size`` is not a positive integer.
    """
    def __init__(self, evidence_size: int) -> None:
        pass

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
            A monotone, vectorised cumulative-distribution function. Returned callable must accept either a Python
            scalar or a numpy array-like object and return a Python float or numpy array, respectively.

        Raises
        ------
        ValueError
            If *prob_vector* is not 1-D, contains negative values, or does not sum to one.
        """
