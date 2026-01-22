"""
Holm-based multi-null decision rules and batched evaluation utilities.

This module implements:
- Holm (Bonferroni–Holm) step-down rejection on a vector of p-values.
- A multi-null *selector* that returns either:
  - a non-rejected null index (1-based), chosen as the non-rejected null with the largest raw p-value; or
  - `REJECT_DECISION` (-1) if all nulls are rejected.
- Batched execution over many histograms, with optional parallel backends (serial, threads, processes), plus a
  multi-statistic variant for backends that return several p-values per null (e.g., ExactMultinom).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from tqdm.auto import tqdm
from typing import Callable, Literal, Optional

import multiprocessing as mp
import numpy.typing as npt
import numpy as np

from experiments.settings import FloatArray, IntArray, REJECT_DECISION


ParallelBackend = Literal["serial", "threads", "processes"]
MPStartMethod = Literal["spawn", "forkserver", "fork"]


def _split_histograms_into_chunks(
    histograms: IntArray,
    n_jobs: int,
    n_chunks: int | None,
) -> tuple[list[IntArray], list[tuple[int, int]]]:
    """
    Split histograms into `num_chunks` contiguous chunks and return chunks + (start, end) spans.

    Notes
    -----
    - `num_chunks` is clamped to [1, m].
    - In parallel mode, we also clamp `num_chunks >= n_jobs` so every worker has something to do.

    Parameters
    ----------
    histograms:
        Two-dimensional array of shape (m, k) with m histograms.
    n_jobs:
        Number of workers.
    n_chunks:
        Desired number of chunks. If None, defaults to `n_jobs`.

    Returns
    -------
    Tuple with:
    - List of `num_chunks` arrays, each of shape (m_local, k).
    - List of `num_chunks` tuples (start, end) with the spans in the original array.
    """
    hist_arr: IntArray = np.asarray(a=histograms, dtype=np.int64)
    m: int = int(hist_arr.shape[0])

    if m == 0:
        return [], []

    n_workers: int = max(1, min(int(n_jobs), m))

    if n_chunks is None:
        n_chunks = n_workers
    else:
        n_chunks = int(n_chunks)
        if n_chunks <= 0:
            raise ValueError(f"`num_chunks` must be >= 1; got {n_chunks}.")
        # Make tqdm smoother but avoid degenerate cases.
        n_chunks = min(n_chunks, m)
        n_chunks = max(n_chunks, n_workers)

    chunks: list[IntArray] = np.array_split(ary=hist_arr, indices_or_sections=n_chunks, axis=0)

    sizes: list[int] = [int(c.shape[0]) for c in chunks]
    starts: IntArray = np.cumsum(a=[0] + sizes[:-1]).astype(dtype=int)
    spans: list[tuple[int, int]] = [(int(starts[i]), int(starts[i] + sizes[i])) for i in range(len(chunks))]

    return chunks, spans


def holm_step_down(p_values: FloatArray, alpha: float) -> npt.NDArray[np.bool_]:
    """
    Apply the Holm (Bonferroni–Holm) step-down procedure to a vector of p-values.

    Parameters
    ----------
    p_values:
        One-dimensional array of raw p-values p_ℓ, ℓ ∈ {1, …, n_nulls}.
    alpha:
        Global family-wise significance level α ∈ (0, 1) to be controlled.

    Returns
    -------
    Boolean array of shape (n_nulls,) where reject_mask[ℓ] is True if H₀^ℓ is rejected by Holm's procedure at level α,
    and False otherwise.
    """
    p_arr: FloatArray = np.asarray(a=p_values, dtype=np.float64)
    if p_arr.ndim != 1:
        raise ValueError(f"`p_values` must be one-dimensional; got shape {p_arr.shape}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must be in (0, 1); got {alpha}.")

    n_nulls: int = p_arr.shape[0]
    order: IntArray = np.argsort(a=p_arr).astype(dtype=np.int64, copy=False)

    reject_mask: np.ndarray = np.zeros(shape=(n_nulls,), dtype=bool)

    for j, idx in enumerate(order, start=1):
        threshold: float = alpha / float(n_nulls - j + 1)
        if p_arr[idx] <= threshold:
            reject_mask[idx] = True
        else:
            # As soon as a null is not rejected, all larger p-values remain accepted.
            break

    return reject_mask


def multinull_decision_holm_from_pvalues(p_values: FloatArray, alpha_global: float) -> int:
    """
    Multi-null decision rule based on Holm-corrected p-values (already computed).

    Returns the non-rejected null with the largest raw p-value, or REJECT_DECISION if all are rejected.

    Parameters
    ----------
    p_values:
        One-dimensional array of raw p-values p_ℓ, ℓ ∈ {1, …, n_nulls}.
    alpha_global:
        Global FWER control level for Holm's procedure, typically `alpha_global = float(np.max(alpha_vector))`.

    Returns
    -------
    Integer label ℓ* ∈ {1, …, n_nulls} or `REJECT_DECISION` (-1).
    """
    p_arr: FloatArray = np.asarray(a=p_values, dtype=np.float64)
    if p_arr.ndim != 1:
        raise ValueError(f"`p_values` must be one-dimensional; got shape {p_arr.shape}.")

    reject_mask: npt.NDArray[np.bool_] = holm_step_down(p_values=p_arr, alpha=alpha_global)

    if np.all(a=reject_mask):
        return int(REJECT_DECISION)

    non_rejected_idx: IntArray = np.nonzero(a=~reject_mask)[0]
    chosen_local: int = int(non_rejected_idx[np.argmax(a=p_arr[non_rejected_idx])])
    return chosen_local + 1  # convert 0-based to 1-based index


def multinull_decision_holm(
    histogram: IntArray,
    null_probabilities: FloatArray,
    alpha_global: float,
    single_null_pvalue_fn: Callable[[IntArray, FloatArray], float],
) -> int:
    """
    Multi-null decision rule based on Holm-corrected single-null p-values.

    This helper:

    1. Computes a p-value for each null hypothesis H₀^ℓ using `single_null_pvalue_fn`.
    2. Applies Holm's step-down multiple-testing procedure at level `alpha_global`.
    3. Produces a **single decision label** in the same convention as `MNSquaredTest`:
       - Returns ℓ* ∈ {1, …, n_nulls} if at least one null is not rejected; in that case, the chosen label is the
         **non-rejected null with the largest raw p-value**.
       - Returns `REJECT_DECISION` (-1) if **all nulls are rejected**.

    Parameters
    ----------
    histogram:
        One-dimensional integer array of shape (k,) with counts; must sum to n.
    null_probabilities:
        Two-dimensional array of shape (n_nulls, k) with the base probabilities p_ℓ for each null hypothesis.
    alpha_global:
        Global FWER control level for Holm's procedure, typically `alpha_global = float(np.max(alpha_vector))`.
    single_null_pvalue_fn:
        Callable that computes a single-null goodness-of-fit p-value with signature: [IntArray, FloatArray] -> float.

    Returns
    -------
    Integer label ℓ* ∈ {1, …, n_nulls} or `REJECT_DECISION` (-1).
    """
    h: IntArray = np.asarray(a=histogram, dtype=np.int64)
    nulls: FloatArray = np.asarray(a=null_probabilities, dtype=np.float64)

    if h.ndim != 1:
        raise ValueError(f"`histogram` must be one-dimensional; got shape {h.shape}.")
    if nulls.ndim != 2:
        raise ValueError(f"`null_probabilities` must have shape (n_nulls, k); got {nulls.shape}.")
    if h.shape[0] != nulls.shape[1]:
        raise ValueError(f"Dimension mismatch: histogram has length {h.shape[0]}, nulls have length {nulls.shape[1]}.")

    n_nulls: int = nulls.shape[0]
    p_values: FloatArray = np.empty(shape=(n_nulls,), dtype=np.float64)
    for idx in range(n_nulls):
        p_values[idx] = float(single_null_pvalue_fn(h, nulls[idx]))

    return multinull_decision_holm_from_pvalues(p_values=p_values, alpha_global=alpha_global)


def _decisions_chunk_worker(
    hist_chunk: IntArray,
    null_probabilities: FloatArray,
    alpha_global: float,
    single_null_pvalue_fn: Callable[[IntArray, FloatArray], float],
) -> IntArray:
    """
    Worker that processes a chunk of histograms sequentially.

    Intended for use with threads/processes to reduce per-task overhead.

    Parameters
    ----------
    hist_chunk:
        Two-dimensional array of shape (m_local, k) with m_local histograms.
    null_probabilities:
        Two-dimensional array of shape (L, k) with null base probabilities.
    alpha_global:
        Global FWER control level α.
    single_null_pvalue_fn:
        Single-null p-value function, see `multinull_decision_holm`.

    Returns
    -------
    One-dimensional integer array of shape (m_local,) with decision labels in {1, …, L, REJECT_DECISION}.
    """
    hist_chunk_arr: IntArray = np.asarray(a=hist_chunk, dtype=np.int64)
    m_local: int = hist_chunk_arr.shape[0]
    out: IntArray = np.empty(shape=(m_local,), dtype=np.int64)

    for i in range(m_local):
        out[i] = multinull_decision_holm(
            histogram=hist_chunk_arr[i],
            null_probabilities=null_probabilities,
            alpha_global=alpha_global,
            single_null_pvalue_fn=single_null_pvalue_fn,
        )

    return out


def multinull_decisions_holm_batch(
    histograms: IntArray,
    null_probabilities: FloatArray,
    alpha_global: float,
    single_null_pvalue_fn: Callable[[IntArray, FloatArray], float],
    n_jobs: int = 1,
    n_chunks: Optional[int] = None,
    parallel_backend: ParallelBackend = "serial",
    mp_start_method: MPStartMethod = "spawn",
    show_progress: bool = True,
    progress_desc: str = "Decisions",
) -> IntArray:
    """
    Batched version of multinull_decision_holm for many histograms.

    Parallelism
    -----------
    - serial: always runs in the current process (default)
    - threads: avoids pickling requirements, but may not speed up CPU-bound work
    - processes: true parallelism, but `single_null_pvalue_fn` must be pickleable. For rpy2 / R calls, prefer
      `processes` (R is not thread-safe)

    Parameters
    ----------
    histograms:
        Two-dimensional array of shape (m, k) with m histograms.
    null_probabilities:
        Two-dimensional array of shape (L, k) with null base probabilities.
    alpha_global:
        Global FWER control level α.
    single_null_pvalue_fn:
        Single-null p-value function, see `multinull_decision_holm`.
    n_jobs:
        Number of workers. If <= 1, runs serially.
    n_chunks:
        Number of chunks to split the histograms into for parallel processing. If None, defaults to `n_jobs`.
    parallel_backend:
        Parallel backend. See the parallelism section above.
    mp_start_method:
        Start method for process pools. "spawn" is the safest choice for R/rpy2 workloads.
    show_progress:
        Whether to show a tqdm progress bar.
    progress_desc:
        Description for the progress bar.

    Returns
    -------
    One-dimensional integer array of shape (m,) with decision labels in {1, …, L, REJECT_DECISION}.
    """
    histograms_array: IntArray = np.asarray(a=histograms, dtype=np.int64)
    if histograms_array.ndim != 2:
        raise ValueError(f"`histograms` must have shape (m, k); got {histograms_array.shape}.")

    m: int = histograms_array.shape[0]
    if m == 0:
        return np.empty(shape=(0,), dtype=np.int64)

    if n_jobs <= 1 or parallel_backend == "serial":
        decisions: IntArray = np.empty(shape=(m,), dtype=np.int64)
        pbar: Optional[tqdm] = None
        if show_progress:
            pbar = tqdm(desc=progress_desc, total=m, unit="histograms")
        else:
            print(progress_desc)
        for idx in range(m):
            decisions[idx] = multinull_decision_holm(
                histogram=histograms_array[idx],
                null_probabilities=null_probabilities,
                alpha_global=alpha_global,
                single_null_pvalue_fn=single_null_pvalue_fn,
            )
            if pbar is not None:
                pbar.update()
        return decisions

    n_workers: int = min(int(n_jobs), m)
    chunks, spans = _split_histograms_into_chunks(histograms=histograms_array, n_jobs=n_workers, n_chunks=n_chunks)
    decisions = np.empty(shape=(m,), dtype=np.int64)

    if parallel_backend == "threads":
        pbar = None
        if progress_desc:
            pbar = tqdm(total=m, desc=progress_desc, unit="histograms")
        else:
            print(progress_desc)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_span: dict[Future, tuple[int, int]] = {
                ex.submit(
                    _decisions_chunk_worker,
                    c,
                    null_probabilities,
                    alpha_global,
                    single_null_pvalue_fn,
                ): (a, b)
                for c, (a, b) in zip(chunks, spans)
            }
            for future in as_completed(fs=future_to_span):
                a, b = future_to_span[future]
                decisions[a:b] = future.result()
                if pbar is not None:
                    pbar.update(n=b - a)
        if pbar is not None:
            pbar.close()
        return decisions

    if parallel_backend == "processes":
        ctx = mp.get_context(mp_start_method)
        pbar = None
        if show_progress:
            pbar = tqdm(total=m, desc=progress_desc, unit="histograms")
        else:
            print(progress_desc)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            future_to_span = {
                ex.submit(
                    _decisions_chunk_worker,
                    c,
                    null_probabilities,
                    alpha_global,
                    single_null_pvalue_fn,
                ): (a, b)
                for c, (a, b) in zip(chunks, spans)
            }
            for future in as_completed(fs=future_to_span):
                a, b = future_to_span[future]
                decisions[a:b] = future.result()
                if pbar is not None:
                    pbar.update(n=b - a)
        if pbar is not None:
            pbar.close()
        return decisions

    raise ValueError(f"Unsupported parallel_backend={parallel_backend!r}.")


def _decisions_multistat_chunk_worker(
    hist_chunk: IntArray,
    null_probabilities: FloatArray,
    alpha_global: float,
    single_null_pvalues_fn: Callable[[IntArray, FloatArray], FloatArray],
) -> IntArray:
    """
    Worker that returns decisions for multiple statistics simultaneously.

    Output shape: (m_chunk, s), where s is len(pval_vector) returned by `single_null_pvalues_fn`.

    Parameters
    ----------
    hist_chunk:
        Two-dimensional array of shape (m_local, k) with m_local histograms.
    null_probabilities:
        Two-dimensional array of shape (L, k) with null base probabilities.
    alpha_global:
        Global FWER control level α.
    single_null_pvalues_fn:
        Callable that computes multiple single-null p-values with signature: [IntArray, FloatArray] -> FloatArray.

    Returns
    -------
    Two-dimensional integer array of shape (m_local, s) with decision labels in {1, …, L, REJECT_DECISION}.
    """
    hist_chunk_arr: IntArray = np.asarray(a=hist_chunk, dtype=np.int64)
    nulls: FloatArray = np.asarray(a=null_probabilities, dtype=np.float64)

    m_local: int = int(hist_chunk_arr.shape[0])
    n_nulls: int = int(nulls.shape[0])

    # Probe output dimension s once.
    probe: FloatArray = np.asarray(a=single_null_pvalues_fn(hist_chunk_arr[0], nulls[0]), dtype=np.float64)
    if probe.ndim != 1:
        raise ValueError("`single_null_pvalues_fn` must return a 1-D array of p-values.")
    s: int = int(probe.shape[0])

    out: IntArray = np.empty(shape=(m_local, s), dtype=np.int64)

    pvals_mat: FloatArray = np.empty(shape=(n_nulls, s), dtype=np.float64)

    for i in range(m_local):
        h: IntArray = hist_chunk_arr[i]
        for ell0 in range(n_nulls):
            pvec: FloatArray = np.asarray(a=single_null_pvalues_fn(h, nulls[ell0]), dtype=np.float64)
            if pvec.shape != (s,):
                raise ValueError(f"Inconsistent p-value vector shape: expected {(s,)}, got {pvec.shape}.")
            pvals_mat[ell0, :] = pvec

        for j in range(s):
            out[i, j] = multinull_decision_holm_from_pvalues(p_values=pvals_mat[:, j], alpha_global=alpha_global)

    return out


def multinull_decisions_holm_batch_multistat(
    histograms: IntArray,
    null_probabilities: FloatArray,
    alpha_global: float,
    single_null_pvalues_fn: Callable[[IntArray, FloatArray], FloatArray],
    n_jobs: int = 1,
    n_chunks: Optional[int] = None,
    parallel_backend: ParallelBackend= "serial",
    mp_start_method: MPStartMethod = "spawn",
    show_progress: bool = True,
    progress_desc: str = "Decisions"
) -> IntArray:
    """
    Batched Holm decisions where the single-null function returns multiple p-values at once.

    This is ideal for ExactMultinom, which returns p-values for [Prob, Chisq, LLR] in one call.

    Parameters
    ----------
    histograms:
        Two-dimensional array of shape (m, k) with m histograms.
    null_probabilities:
        Two-dimensional array of shape (L, k) with null base probabilities.
    alpha_global:
        Global FWER control level α.
    single_null_pvalues_fn:
        Callable that computes multiple single-null p-values with signature: [IntArray, FloatArray] -> FloatArray.
    n_jobs:
        Number of workers. If <= 1, runs serially.
    n_chunks:
        Number of chunks to split the histograms into for parallel processing. If None, defaults to `n_jobs`.
    parallel_backend:
        Parallel backend. See the parallelism section in `multinull_decisions_holm_batch`.
    mp_start_method:
        Start method for process pools. "spawn" is the safest choice for R/rpy2 workloads.
    show_progress:
        Whether to show a tqdm progress bar.
    progress_desc:
        Description for the progress bar.


    Returns
    -------
    Integer array of shape (m, s), where s is the number of p-values returned by `single_null_pvalues_fn`.
    """
    histograms_array: IntArray = np.asarray(a=histograms, dtype=np.int64)
    if histograms_array.ndim != 2:
        raise ValueError(f"`histograms` must have shape (m, k); got {histograms_array.shape}.")

    m: int = int(histograms_array.shape[0])
    if m == 0:
        return np.empty(shape=(0, 0), dtype=np.int64)

    if n_jobs <= 1 or parallel_backend == "serial":
        pbar: Optional[tqdm] = None
        if show_progress:
            pbar = tqdm(total=m, desc=progress_desc, unit="histograms")
        else:
            print(progress_desc)
        out: IntArray = _decisions_multistat_chunk_worker(
            hist_chunk=histograms_array,
            null_probabilities=null_probabilities,
            alpha_global=alpha_global,
            single_null_pvalues_fn=single_null_pvalues_fn,
        )
        if pbar is not None:
            pbar.update(n=m)
            pbar.close()
        return out

    n_workers: int = min(int(n_jobs), m)
    chunks, spans = _split_histograms_into_chunks(histograms=histograms_array, n_jobs=n_jobs, n_chunks=n_chunks)
    out_full: Optional[IntArray] = None  # Allocated after first result to learn `s`

    if parallel_backend == "threads":
        pbar = None
        if show_progress:
            pbar = tqdm(total=m, desc=progress_desc, unit="hist")
        else:
            print(progress_desc)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_span: dict[Future, tuple[int, int]] = {
                ex.submit(
                    _decisions_multistat_chunk_worker,
                    c,
                    null_probabilities,
                    alpha_global,
                    single_null_pvalues_fn,
                ): (a, b)
                for c, (a, b) in zip(chunks, spans)
            }
            for fut in as_completed(future_to_span):
                a, b = future_to_span[fut]
                out_chunk = fut.result()  # (b-a, s)
                if out_full is None:
                    s: int = int(out_chunk.shape[1])
                    out_full = np.empty(shape=(m, s), dtype=np.int64)
                out_full[a:b, :] = out_chunk
                if pbar is not None:
                    pbar.update(n=b - a)
        if pbar is not None:
            pbar.close()
        assert out_full is not None
        return out_full

    if parallel_backend == "processes":
        ctx = mp.get_context(mp_start_method)
        pbar = None
        if show_progress:
            pbar = tqdm(total=m, desc=progress_desc, unit="hist")
        else:
            print(progress_desc)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            future_to_span = {
                ex.submit(
                    _decisions_multistat_chunk_worker,
                    c,
                    null_probabilities,
                    alpha_global,
                    single_null_pvalues_fn,
                ): (a, b)
                for c, (a, b) in zip(chunks, spans)
            }
            for fut in as_completed(future_to_span):
                a, b = future_to_span[fut]
                out_chunk = fut.result()
                if out_full is None:
                    s = int(out_chunk.shape[1])
                    out_full = np.empty(shape=(m, s), dtype=np.int64)
                out_full[a:b, :] = out_chunk
                if pbar is not None:
                    pbar.update(n=b - a)
        if pbar is not None:
            pbar.close()
        assert out_full is not None
        return out_full

    raise ValueError(f"Unsupported parallel_backend={parallel_backend!r}.")
