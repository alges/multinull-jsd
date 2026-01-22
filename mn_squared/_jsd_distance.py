"""
Internal utilities to compute the Jensen–Shannon distance (JSd) between discrete probability distributions.

All functions here use base-2 logarithms so that JSd lies in [0, 1].
"""
from mn_squared._validators import validate_probability_batch
from mn_squared.types import FloatArray, FloatDType

import numpy as np



def jsd(p: FloatArray, q: FloatArray) -> FloatArray:
    """
    Computes the Jensen-Shannon distance between two probability distributions; p and q may be batches or single
    probability vectors.

    Parameters
    ----------
    p
        Array of shape (k,) or (l,k) representing the first probability vector or a batch of probability vectors.
    q
        Array of shape (k,) or (l,k) representing the second probability vector or a batch of probability vectors.

    Returns
    -------
    FloatArray
        An array of shape (l,) containing the Jensen-Shannon distances for each pair of vectors in the batch. If both
        inputs are of shape (k,), the output is of shape (1,). If only one input is a batch, the output is of shape
        (l,) and output[i] is the JSD between p[i] and q or p and q[i], respectivaly. If both inputs are batches, the
        output is of shape (l,) and output[i] is the JSD between p[i] and q[i].
    """

    def _kl_divergence(a: FloatArray, b: FloatArray) -> FloatArray:
        """
        Compute the Kullback-Leibler divergence D_KL(a || b) for each row in a and b.

        Parameters
        ----------
        a
            2-D array of shape (l, k).
        b
            2-D array of shape (l, k).

        Returns
        -------
        FloatArray
            1-D array of shape (l,) containing the KL divergence for each row.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.maximum(np.where(a > 0.0, a * np.log2(a / b), 0.0).sum(axis=1), 0.0)

    p_batch: FloatArray = validate_probability_batch(name="p", value=p, n_categories=None)
    q_batch: FloatArray = validate_probability_batch(name="q", value=q, n_categories=p_batch.shape[-1])

    p_len: int = p_batch.shape[0]
    q_len: int = q_batch.shape[0]
    common_m: int = max(p_len, q_len)
    common_k: int = p_batch.shape[1]
    if common_m > 1:
        if p_len != 1 and q_len != 1 and p_len != q_len:
            raise ValueError("Probability batches p and q must have the same length or be single probability vectors.")
        p_batch = np.broadcast_to(array=p_batch, shape=(common_m, common_k))
        q_batch = np.broadcast_to(array=q_batch, shape=(common_m, common_k))

    p_batch = p_batch.astype(dtype=FloatDType, copy=False)
    q_batch = q_batch.astype(dtype=FloatDType, copy=False)
    m_batch: FloatArray = (p_batch + q_batch) / 2.0

    return np.clip(
        a=np.sqrt((_kl_divergence(a=p_batch, b=m_batch) + _kl_divergence(a=q_batch, b=m_batch)) / 2.0),
        a_min=0,
        a_max=1
    )
