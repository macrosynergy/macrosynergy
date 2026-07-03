"""
Annualize quantamental values by a time-varying weight inferred from release cadence.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from macrosynergy.management.constants import ANNUALIZATION_FACTORS
from macrosynergy.management.utils import infer_release_frequency
from macrosynergy.management.types import QuantamentalDataFrame


def _finalize_qdf(df_ordered: pd.DataFrame, is_categorical: bool) -> QuantamentalDataFrame:
    """
    Attach QDF metadata to an already column-ordered frame without disturbing that
    order.

    ``QuantamentalDataFrame.__init__`` re-sorts columns to its canonical
    ``(real_date, cid, xcat, ...)`` order whenever it is not handed an already-literal
    ``QuantamentalDataFrame`` instance -- which would silently override the
    ``cid, xcat, real_date, value`` order this function guarantees to callers.
    Setting ``InitializedAsCategorical`` directly (the same plain attribute assignment
    the constructor itself performs) avoids that reorder while still yielding an
    object that satisfies ``isinstance(..., QuantamentalDataFrame)`` (a structural
    check) and carries the attribute callers rely on.
    """
    df_ordered.InitializedAsCategorical = is_categorical
    return df_ordered


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
    ~pandas.DataFrame
        standardized QuantamentalDataFrame with columns 'cid', 'xcat', 'real_date',
        'value'; categories renamed ``{xcat}{postfix}``.
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
        empty_df = pd.DataFrame(
            {
                "cid": pd.Series([], dtype="category" if _as_categorical else "object"),
                "xcat": pd.Series([], dtype="category" if _as_categorical else "object"),
                "real_date": pd.Series([], dtype="datetime64[ns]"),
                "value": pd.Series([], dtype="float64"),
            }
        )[cols]
        return _finalize_qdf(empty_df, _as_categorical)

    df_out = pd.concat(frames, axis=0, ignore_index=True)
    return QuantamentalDataFrame.from_long_df(df_out, categorical=_as_categorical)
