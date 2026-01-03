"""
I/O helpers for experiment artifacts.

This module defines a canonical `RESULTS_DIR` under the current working directory and ensures it exists at import time.
It also provides:
- `_slug` for stable, filename-safe scenario identifiers, and
- `_maybe_save_histograms` for optional persistence of sampled histograms.
"""
from pathlib import Path

import numpy as np
import re

from settings import IntArray


ROOT: Path = Path.cwd()
RESULTS_DIR: Path = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _slug(s: str) -> str:
    """
    Replace non-alphanumeric characters with underscores and truncate to 80 chars.

    Parameters
    ----------
    s:
        String to slugify.

    Returns
    -------
    Slugified string
    """
    s = s.strip().lower()
    s = re.sub(pattern=r"[^a-z0-9]+", repl="_", string=s)
    return s.strip("_")[:80]


def _maybe_save_histograms(
    scenario_slug: str,
    n: int,
    h0_all: IntArray,
    hq_all: IntArray,
    alt_ids: IntArray,
    save_histograms: bool,
) -> None:
    """
    Optionally, persist sampled histograms to disk in compressed NPZ format.

    When `save_histograms` is True, writes a file named
    "{scenario_slug}_histograms_n{n}.npz" under RESULTS_DIR containing:
      - h0_all: int64 array of shape (L, m_null, k)
      - hq_all: int64 array of shape (T, m_alt, k)
      - alt_ids: int64 array of shape (T,)

    Parameters
    ----------
    scenario_slug:
        Slug identifying the scenario; used in the output filename.
    n:
        Sample size used to generate the histograms; used in the filename.
    h0_all:
        Null histograms array to store.
    hq_all:
        Alternative histograms array to store.
    alt_ids:
        Alternative IDs aligned with the first dimension of `hq_all`.
    save_histograms:
        If False, the function is a no-op; if True, data are saved with compression.

    Notes
    -----
    Files can become large. Compression is enabled via numpy.savez_compressed.
    """
    if not save_histograms:
        return
    out: Path = RESULTS_DIR / f"{scenario_slug}_histograms_n{int(n)}.npz"
    np.savez_compressed(
        out,
        h0_all=np.asarray(h0_all, dtype=np.int64),
        hq_all=np.asarray(hq_all, dtype=np.int64),
        alt_ids=np.asarray(alt_ids, dtype=np.int64),
    )
