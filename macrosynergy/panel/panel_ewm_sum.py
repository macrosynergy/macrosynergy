"""
Fast exponential moving sum of quantamental panels on a business-day grid.
"""
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from macrosynergy.compat import PD_NEW_MAP, PD_FUTURE_STACK
from macrosynergy.management.utils import reduce_df, ewm_sum
from macrosynergy.management.types import QuantamentalDataFrame


def _ewm_sum(dfw: pd.DataFrame, halflife: Union[int, float]) -> pd.DataFrame:
    """
    Exponentially weighted moving sum of a wide (dense) frame.

    ``ExponentialMovingWindow.sum`` only exists on pandas >= 1.4.0, below the package's
    1.3.5 floor. Rather than add a dedicated version flag, gate on the existing
    ``PD_NEW_MAP`` (pandas >= 2.1.0): modern pandas uses the native
    ``ewm(halflife).sum()``, and anything older falls back to
    :func:`macrosynergy.management.utils.math.ewm_sum`, which reconstructs the identical
    values as ``ewm().mean()`` scaled by the cumulative weights.
    """
    if PD_NEW_MAP:
        return dfw.ewm(halflife=halflife).sum()
    return ewm_sum(dfw, halflife)


def panel_ewm_sum(
    df: pd.DataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
    halflife: Union[int, float, List[Union[int, float]]] = 5,
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
    ``ewm(halflife).sum()`` definition directly. Each series is reindexed to a business-day
    grid, so ``halflife`` is measured in business days.

    The input is expected to be dense on that grid: a standardised panel should have an
    observation on every business day between a series' first and last release, apart from
    blacklisted ranges. A ``NaN`` (whether explicit in ``value`` or an implicit gap in the
    business-day grid) inside a series' observed span therefore signals a data-quality
    problem and raises a ``ValueError`` rather than being silently filled. The leading and
    trailing regions outside each series' first/last observation are excluded from the
    output.

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
    start, end : str
        date bounds (ISO). Default None uses the range in ``df``.
    blacklist : dict
        cross-sections with date ranges to exclude. Blacklisted ranges are the one allowed
        source of gaps: they are excluded from the moving sum and absent from the output,
        and never trigger the interior-gap check.
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
        if a series contains a ``NaN`` (explicit or an implicit business-day gap) within
        its observed span, i.e. between its first and last observation and outside any
        blacklisted range.
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

    # Select without the blacklist first: this is the panel whose density we validate and
    # whose first/last observation defines each series' output span. The blacklist is the
    # one sanctioned source of gaps, so it must not make the interior-gap check fire.
    dfr = reduce_df(qdf, xcats=xcats, cids=cids, start=start, end=end)
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

    # Explicit NaNs in `value` are always a data-quality signal (this also makes every
    # series' first/last valid index well-defined below); implicit business-day gaps are
    # caught by the interior-span check further down.
    if dfr["value"].isna().any():
        raise ValueError(
            "Input `value` column contains NaN(s). `panel_ewm_sum` expects a dense panel "
            "(gaps only from blacklisted ranges); resolve or drop missing values first."
        )

    dfr = dfr.assign(
        ticker=dfr["cid"].astype(str) + "_" + dfr["xcat"].astype(str)
    )
    p = dfr.pivot(index="real_date", columns="ticker", values="value")

    grid = pd.date_range(p.index.min(), p.index.max(), freq="B")
    p = p.reindex(grid)
    p.index.name = "real_date"

    # Per series the output span is [first observation, last observation]; anything outside
    # is a leading/trailing region excluded from the output. A missing business day *inside*
    # that span is a data-quality signal: silently zero-filling it would corrupt the moving
    # sum, so reject it. Blacklisted ranges are handled separately below and never reach
    # here.
    first_valid = {c: p[c].first_valid_index() for c in p.columns}
    last_valid = {c: p[c].last_valid_index() for c in p.columns}
    for c in p.columns:
        span = p[c].loc[first_valid[c] : last_valid[c]]
        if span.isna().any():
            gap = span.index[span.isna()][0].date()
            raise ValueError(
                f"'{c}' has a gap within its observed span (first missing business day "
                f"{gap}). `panel_ewm_sum` expects a dense panel apart from blacklisted "
                "ranges; resolve the missing value(s) first."
            )

    if blacklist is not None:
        # Exclude blacklisted (cid, date) cells from the moving sum by dropping them before
        # the sum -- reindexing then leaves them NaN, which the fill below turns into a
        # zero contribution, matching blacklist semantics elsewhere in the package. The
        # span bounds above are kept from the pre-blacklist panel; blacklisted output rows
        # are dropped at the very end.
        kept = reduce_df(dfr[cols], blacklist=blacklist).assign(
            ticker=lambda d: d["cid"].astype(str) + "_" + d["xcat"].astype(str)
        )
        p = kept.pivot(index="real_date", columns="ticker", values="value").reindex(
            index=grid, columns=p.columns
        )
        p.index.name = "real_date"

    # Zero-fill the padding (leading/trailing regions and blacklisted cells). These are
    # masked out or dropped later; filling them keeps the native and fallback ewm routes
    # numerically identical, since a NaN and a 0 both contribute nothing to the sum.
    p = p.fillna(0.0)

    frames = []
    for h, pf in zip(hls, postfixes):
        out = _ewm_sum(p, h)
        for c in out.columns:
            outside = (out.index < first_valid[c]) | (out.index > last_valid[c])
            out.loc[outside, c] = np.nan
        out.columns = [f"{c}_{pf}" for c in out.columns]
        # `PD_FUTURE_STACK` keeps NaNs on stack across pandas versions; drop the masked
        # leading/trailing rows explicitly so the behaviour does not depend on the
        # deprecated `stack(dropna=...)` default.
        tmp = out.stack(**PD_FUTURE_STACK).to_frame("value").reset_index()
        tmp.columns = ["real_date", "ticker", "value"]
        tmp = tmp.dropna(subset=["value"])
        tmp[["cid", "xcat"]] = tmp["ticker"].str.split("_", n=1, expand=True)
        frames.append(tmp[cols])

    df_out = pd.concat(frames, axis=0, ignore_index=True)
    qdf_out = QuantamentalDataFrame.from_long_df(df_out, categorical=_as_categorical)
    if blacklist is not None:
        # The zero-fill/reindex above lets the EWM sum decay through -- and reappear in --
        # blacklisted dates. Re-apply the blacklist to the output so blacklisted rows stay
        # absent, matching blacklist semantics elsewhere (e.g. `reduce_df`, `make_blacklist`).
        qdf_out = reduce_df(qdf_out, blacklist=blacklist)
    return qdf_out
