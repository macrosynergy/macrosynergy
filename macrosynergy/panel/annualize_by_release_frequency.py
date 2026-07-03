"""
Annualize quantamental values by a time-varying weight inferred from release cadence.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from macrosynergy.management.constants import ANNUALIZATION_FACTORS
from macrosynergy.management.utils import infer_release_frequency
from macrosynergy.management.types import QuantamentalDataFrame


def annualize_by_release_frequency(
    df: pd.DataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
    eop_col: str = "eop",
    window: int = 3,
    freqs: Tuple[str, ...] = ("D", "W", "M", "Q", "A"),
    postfix: str = "A",
) -> QuantamentalDataFrame:
    """
    Multiply each value by ``sqrt(1 / ANNUALIZATION_FACTORS[freq])`` where ``freq`` is the
    contemporaneous release frequency inferred per observation from ``eop`` cadence.

    The weight is time-varying: a series whose cadence changes (e.g. quarterly -> monthly)
    is weighted quarterly before the break and monthly after it.

    Parameters
    ----------
    df : ~pandas.DataFrame
        QuantamentalDataFrame with columns 'cid', 'xcat', 'real_date', 'value' and an
        end-of-period column (``eop_col``). Emit it via
        ``InformationStateChanges.to_qdf(metrics=["eop", ...])``.
    xcats, cids : List[str]
        categories / cross-sections to transform. Default is all in ``df``.
    eop_col : str
        name of the end-of-period date column. Default "eop".
    window : int
        rolling-median window passed to ``infer_release_frequency``. Default 3.
    freqs : Tuple[str, ...]
        candidate frequency labels. Default ("D", "W", "M", "Q", "A").
    postfix : str
        suffix appended to each output category. Default "A".

    Returns
    -------
    QuantamentalDataFrame
        a standardized QuantamentalDataFrame in canonical column order ('real_date',
        'cid', 'xcat', 'value'); categories renamed ``{xcat}{postfix}``. This is always
        a genuine ``QuantamentalDataFrame`` instance, including when the selection is
        empty.
    """
    cols = ["cid", "xcat", "real_date", "value"]
    if eop_col not in df.columns:
        raise ValueError(
            f"`{eop_col}` column required. Emit it via "
            f"InformationStateChanges.to_qdf(metrics=['{eop_col}', ...])."
        )

    _as_categorical = QuantamentalDataFrame(df[cols]).InitializedAsCategorical

    # reduce_df strips non-standard columns, so subset on a plain frame to keep eop_col.
    work = df[cols + [eop_col]].copy()
    work["cid"] = work["cid"].astype(str)
    work["xcat"] = work["xcat"].astype(str)
    if xcats is not None:
        work = work[work["xcat"].isin(xcats)]
    if cids is not None:
        work = work[work["cid"].isin(cids)]

    weights = {v: np.sqrt(1 / ANNUALIZATION_FACTORS[v]) for v in freqs}

    frames = []
    for (cid, xcat), g in work.sort_values("real_date").groupby(["cid", "xcat"]):
        g = g.copy()
        freq = infer_release_frequency(g[eop_col], window=window, freqs=freqs)
        g["value"] = g["value"].to_numpy() * freq.map(weights).to_numpy()
        g["xcat"] = f"{xcat}{postfix}"
        frames.append(g[cols])

    if not frames:
        # Build directly in canonical (real_date, cid, xcat, value) order -- the same
        # order QuantamentalDataFrame.from_long_df produces on the non-empty path --
        # so the return contract does not depend on whether the selection is empty.
        # cid/xcat start out categorical so that QuantamentalDataFrame's own
        # `categorical` handling (and its InitializedAsCategorical bookkeeping) mirrors
        # what from_long_df does internally on the non-empty path.
        empty_df = pd.DataFrame(
            {
                "real_date": pd.Series([], dtype="datetime64[ns]"),
                "cid": pd.Series([], dtype="category"),
                "xcat": pd.Series([], dtype="category"),
                "value": pd.Series([], dtype="float64"),
            }
        )
        return QuantamentalDataFrame(empty_df, categorical=_as_categorical)

    df_out = pd.concat(frames, axis=0, ignore_index=True)
    return QuantamentalDataFrame.from_long_df(df_out, categorical=_as_categorical)
