import logging
from datetime import timedelta, datetime, date
from typing import Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def rescale_to_anchor(frame: pd.DataFrame, status: pd.DataFrame, ratio_tolerance: float = 0.05):
    """Rescale freshly fetched RI levels onto the cached series using the overlapping
    anchor date (the ticker's last cached date) as a splice point — this corrects for
    any level rebasing the source may apply to historical data even though the % change
    is unaffected — then drop the anchor row itself, since the old cached value for that
    date remains authoritative. Tickers with no prior cache pass through unscaled."""
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
