"""
Miscellaneous helpers for managing cached single-security data.
"""

import logging
from datetime import timedelta, datetime, date
from typing import Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def rescale_to_anchor(
    frame: pd.DataFrame,
    status: pd.DataFrame,
    ratio_tolerance: float = 0.05,
) -> pd.DataFrame:
    """
    Splice freshly fetched return index (RI) levels onto a cached series.

    The overlapping anchor date - the ticker's last cached date - is used as the
    splice point: the fetched levels are scaled by the ratio of the cached to the
    fetched value on that date. This corrects for any level rebasing the source may
    apply to historical data, which leaves the % change unaffected but shifts the
    level. The anchor row itself is then dropped, since the old cached value for that
    date remains authoritative. Tickers with no prior cache pass through unscaled.

    Parameters
    ----------
    frame : pd.DataFrame
        Freshly fetched long-format data with columns ``"ticker"``, ``"real_date"``
        and ``"value"`` (RI level).
    status : pd.DataFrame
        Cache state indexed by ticker, with columns ``"max_real_date"`` (the ticker's
        last cached date, i.e. the anchor) and ``"last_value"`` (the cached RI level
        on that date). Tickers missing from the index are treated as having no prior
        cache and are left unscaled.
    ratio_tolerance : float, default 0.05
        Absolute deviation of the rescaling ratio from one above which a warning is
        emitted for that ticker, flagging an RI level that may have shifted
        materially. Affects reporting only, not the rescaling itself.

    Returns
    -------
    pd.DataFrame
        ``frame`` with ``"value"`` rescaled onto the cached level, anchor-date rows
        removed, the merged helper columns dropped and the index reset.
    """
    frame = frame.merge(
        status[["max_real_date", "last_value"]],
        how="left",
        left_on="ticker",
        right_index=True,
    )
    is_anchor = frame["real_date"] == frame["max_real_date"]
    anchor_value = frame.loc[is_anchor].set_index("ticker")["value"]
    cached_value = frame.loc[is_anchor].set_index("ticker")["last_value"]
    scale = (cached_value / anchor_value).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    unusual = scale.loc[(scale - 1).abs() > ratio_tolerance]
    if len(unusual) > 0:
        for ticker, ratio in unusual.items():
            print(
                f"  Warning: unusual rescale ratio {ratio:.3f} for {ticker} at the anchor date — RI level may have shifted materially"
            )

    frame["value"] = frame["value"] * frame["ticker"].map(scale).fillna(1.0)
    return (
        frame.loc[~is_anchor]
        .drop(columns=["max_real_date", "last_value"])
        .reset_index(drop=True)
    )
