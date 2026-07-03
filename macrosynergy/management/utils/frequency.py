"""
Infer per-observation release frequency from the spacing of end-of-period (eop) dates.
"""
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from macrosynergy.management.constants import ANNUALIZATION_FACTORS


def _reference_days(freqs: Sequence[str]) -> dict:
    # Calendar days per period, derived from periods-per-year. e.g. Q -> 365.25/4.
    return {f: 365.25 / ANNUALIZATION_FACTORS[f] for f in freqs}


def infer_release_frequency(
    eop: pd.Series,
    window: int = 3,
    freqs: Tuple[str, ...] = ("D", "W", "M", "Q", "A"),
) -> pd.Series:
    """
    Classify the release frequency of each observation from its local ``eop`` cadence.

    The gap (in days) between consecutive *distinct* eop dates is smoothed with a rolling
    median (``window``, ``min_periods=1``) and snapped to the nearest supported frequency
    by log-distance to the reference period length (``365.25 / ANNUALIZATION_FACTORS``).
    Observations sharing an eop (revisions) inherit that period's frequency.

    Parameters
    ----------
    eop : pd.Series
        per-observation end-of-period dates (datetime); the index is preserved.
    window : int
        rolling-median window over distinct-eop gaps. Default 3.
    freqs : Tuple[str, ...]
        candidate frequency labels. Default ("D", "W", "M", "Q", "A").

    Returns
    -------
    pd.Series
        per-observation frequency labels, aligned to the input index.
    """
    eop = pd.to_datetime(eop)
    ref = _reference_days(freqs)
    log_ref = {f: np.log(d) for f, d in ref.items()}

    # Distinct, sorted eop periods and their smoothed gaps (in days).
    distinct = pd.Series(sorted(pd.unique(eop.dropna())))
    if len(distinct) == 0:
        return pd.Series(index=eop.index, dtype=object)
    # Seed the leading NaN gap by back-filling from the first observed gap
    # (min_periods=1 covers the rest); avoids chained-assignment warnings.
    gaps = distinct.diff().dt.days.bfill()
    smoothed = gaps.rolling(window=window, min_periods=1).median()

    def _snap(g):
        if pd.isna(g) or g <= 0:
            return freqs[0]
        lg = np.log(g)
        return min(freqs, key=lambda f: abs(lg - log_ref[f]))

    period_freq = {d: _snap(g) for d, g in zip(distinct, smoothed)}
    return eop.map(period_freq)
