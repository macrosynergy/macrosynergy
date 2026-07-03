"""
Fast exponential moving sum of quantamental panels on a business-day grid.
"""
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from macrosynergy.compat import PD_EWM_SUM, PD_FUTURE_STACK
from macrosynergy.management.utils import reduce_df, ewm_sum
from macrosynergy.management.types import QuantamentalDataFrame


def _ewm_sum(dfw: pd.DataFrame, halflife: Union[int, float]) -> pd.DataFrame:
    """
    Exponentially weighted moving sum of a wide (dense) frame.

    Uses the native ``ewm(halflife).sum()`` where available (pandas >= 1.4.0) and
    otherwise falls back to :func:`macrosynergy.management.utils.math.ewm_sum`, which
    reconstructs the same values as ``ewm().mean()`` scaled by the cumulative weights.
    """
    if PD_EWM_SUM:
        return dfw.ewm(halflife=halflife).sum()
    return ewm_sum(dfw, halflife)


def panel_ewm_sum(
    df: pd.DataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
    halflife: Union[int, float, List[Union[int, float]]] = 5,
    fillna: float = 0.0,
    mask_leading: bool = True,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    postfix: Optional[Union[str, List[str]]] = None,
) -> QuantamentalDataFrame:
    """
    Exponentially weighted moving sum of one or more category panels, computed on a
    dense business-day grid.

    Unlike :func:`macrosynergy.management.utils.math.ewm_sum` (which returns
    ``ewm().mean()`` scaled by cumulative weights), this uses the pandas
    ``ewm(halflife).sum()`` definition directly. The input is reindexed to a business-day
    grid, so ``halflife`` is measured in business days. Any structural gaps introduced by
    that reindexing (e.g. cross-sections with different date ranges) are filled with
    ``fillna``; explicit ``NaN`` values in the input ``value`` column are rejected, as a
    standardised panel is expected to be dense apart from blacklisted ranges.

    Parameters
    ----------
    df : ~pandas.DataFrame
        standardized QuantamentalDataFrame with columns 'cid', 'xcat', 'real_date',
        'value'.
    xcats : List[str]
        categories to transform. Default is all categories in ``df``.
    cids : List[str]
        cross-sections to transform. Default is all cross-sections in ``df``.
    halflife : int | float | List
        EWM half-life in business days. A list produces one output category per value.
    fillna : float
        value used for structural gaps introduced by reindexing to the business-day grid
        (i.e. dates a cross-section has no observation for because another cross-section
        does). Default 0.0 (a business day with no release contributes zero to the moving
        sum). Explicit ``NaN`` values already present in the input are not filled -- they
        raise a ``ValueError``.
    mask_leading : bool
        if True (default) output before each series' first real observation is excluded
        from the output entirely (rather than present as NaN).
    start, end : str
        date bounds (ISO). Default None uses the range in ``df``.
    blacklist : dict
        cross-sections with date ranges to exclude.
    postfix : str | List[str]
        output category suffix. Default None -> ``f"{h}DXMS"`` per half-life. A single
        string is allowed only for a scalar ``halflife``; a list must match its length.

    Returns
    -------
    ~pandas.DataFrame
        standardized QuantamentalDataFrame with columns 'real_date', 'cid', 'xcat',
        'value'; new categories named ``{xcat}_{h}DXMS`` (or ``{xcat}_{postfix}``).

    Raises
    ------
    ValueError
        if the selected input contains explicit ``NaN`` values in the ``value`` column.
    """
    cols = ["cid", "xcat", "real_date", "value"]
    assert set(cols).issubset(set(df.columns)), f"df must contain columns: {cols}."

    qdf = QuantamentalDataFrame(df[cols])
    _as_categorical = qdf.InitializedAsCategorical

    hls = [halflife] if isinstance(halflife, (int, float)) else list(halflife)
    assert all(
        isinstance(h, (int, float)) and not isinstance(h, bool) and h > 0 for h in hls
    ), "halflife must be a positive number or a list of positive numbers."
    if postfix is None:
        postfixes = [f"{h}DXMS" for h in hls]
    elif isinstance(postfix, str):
        assert len(hls) == 1, "A string postfix requires a scalar halflife."
        postfixes = [postfix]
    else:
        assert len(postfix) == len(hls), "postfix list must match halflife length."
        postfixes = list(postfix)

    dfr = reduce_df(
        qdf, xcats=xcats, cids=cids, start=start, end=end, blacklist=blacklist
    )
    if dfr.empty:
        empty_df = pd.DataFrame(
            {
                "real_date": pd.Series([], dtype="datetime64[ns]"),
                "cid": pd.Series([], dtype="object"),
                "xcat": pd.Series([], dtype="object"),
                "value": pd.Series([], dtype="float64"),
            }
        )
        return QuantamentalDataFrame(empty_df, categorical=_as_categorical)

    # A standardised panel is expected to be dense apart from blacklisted ranges (already
    # stripped by `reduce_df`). An explicit NaN in `value` is therefore a data-quality
    # signal, not a gap to zero-fill: silently treating it as 0 would corrupt the moving
    # sum. Reject it up front (this also covers all-NaN series).
    if dfr["value"].isna().any():
        raise ValueError(
            "Input `value` column contains NaN(s). `panel_ewm_sum` expects a dense panel "
            "(gaps only from blacklisted ranges); resolve or drop missing values first."
        )

    dfr = dfr.assign(
        ticker=dfr["cid"].astype(str) + "_" + dfr["xcat"].astype(str)
    )
    p = dfr.pivot(index="real_date", columns="ticker", values="value")
    first_valid = {c: p[c].first_valid_index() for c in p.columns}

    grid = pd.date_range(p.index.min(), p.index.max(), freq="B")
    p = p.reindex(grid)
    p.index.name = "real_date"
    p = p.fillna(fillna)

    frames = []
    for h, pf in zip(hls, postfixes):
        out = _ewm_sum(p, h)
        if mask_leading:
            for c in out.columns:
                out.loc[out.index < first_valid[c], c] = np.nan
        out.columns = [f"{c}_{pf}" for c in out.columns]
        # `PD_FUTURE_STACK` keeps NaNs on stack across pandas versions; drop the
        # masked leading rows explicitly so the behaviour does not depend on the
        # deprecated `stack(dropna=...)` default.
        tmp = out.stack(**PD_FUTURE_STACK).to_frame("value").reset_index()
        tmp.columns = ["real_date", "ticker", "value"]
        tmp = tmp.dropna(subset=["value"])
        tmp[["cid", "xcat"]] = tmp["ticker"].str.split("_", n=1, expand=True)
        frames.append(tmp[cols])

    df_out = pd.concat(frames, axis=0, ignore_index=True)
    qdf_out = QuantamentalDataFrame.from_long_df(df_out, categorical=_as_categorical)
    if blacklist is not None:
        # The zero-fill/reindex above re-fills blacklisted windows with `fillna`, which
        # would let the EWM sum decay through -- and reappear in -- excluded dates.
        # Re-apply the blacklist to the output so blacklisted rows stay absent, matching
        # blacklist semantics elsewhere in the package (e.g. `reduce_df`, `make_blacklist`).
        qdf_out = reduce_df(qdf_out, blacklist=blacklist)
    return qdf_out
