"""
Fast exponential moving sum of quantamental panels on a business-day grid.
"""
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from macrosynergy.management.utils import reduce_df
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
    ``ewm(halflife).sum()`` definition directly. Sparse inputs are reindexed to a
    business-day grid and zero-filled between observations, so ``halflife`` is measured
    in business days.

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
        value used for interior gaps after reindexing to the business-day grid.
        Default 0.0 (a business day with no release contributes zero to the moving sum).
    mask_leading : bool
        if True (default) output before each series' first real observation is excluded
        from the output entirely (rather than present as NaN), since ``stack()`` drops
        those rows.
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
        standardized QuantamentalDataFrame with columns 'cid', 'xcat', 'real_date',
        'value'; new categories named ``{xcat}_{h}DXMS`` (or ``{xcat}_{postfix}``).
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
                "cid": pd.Series([], dtype="category" if _as_categorical else "object"),
                "xcat": pd.Series([], dtype="category" if _as_categorical else "object"),
                "real_date": pd.Series([], dtype="datetime64[ns]"),
                "value": pd.Series([], dtype="float64"),
            }
        )[cols]
        return _finalize_qdf(empty_df, _as_categorical)

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
        out = p.ewm(halflife=h).sum()
        if mask_leading:
            for c in out.columns:
                out.loc[out.index < first_valid[c], c] = np.nan
        out.columns = [f"{c}_{pf}" for c in out.columns]
        tmp = out.stack().to_frame("value").reset_index()
        tmp.columns = ["real_date", "ticker", "value"]
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
    return _finalize_qdf(qdf_out[cols], _as_categorical)
