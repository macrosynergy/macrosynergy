"""Estimate period-performance metrics directly from daily arithmetic PnL."""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd


def _wide(pnl):
    if isinstance(pnl, pd.Series):
        pnl = pnl.to_frame(name=pnl.name if pnl.name is not None else "pnl")
    if not isinstance(pnl, pd.DataFrame):
        raise TypeError("pnl must be a pandas Series or DataFrame.")
    if not isinstance(pnl.index, pd.DatetimeIndex):
        raise TypeError("pnl must have a DatetimeIndex.")
    if pnl.index.has_duplicates or pnl.index.hasnans:
        raise ValueError("PnL timestamps must be unique and nonmissing.")
    if pnl.columns.has_duplicates or pnl.empty:
        raise ValueError("pnl must have observations and unique column names.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in pnl.dtypes):
        raise TypeError("PnL values must be numeric.")
    if (
        np.iscomplexobj(pnl.to_numpy())
        or np.isinf(pnl.to_numpy(dtype=float, na_value=np.nan)).any()
    ):
        raise ValueError("PnL values must be real, without infinities.")
    return pnl.sort_index()


def _year_boundary(year, last=False, tz=None):
    return pd.Timestamp(
        year=year, month=12 if last else 1, day=31 if last else 1, tz=tz
    )


def period_metrics(
    pnl: Union[pd.Series, pd.DataFrame],
    *,
    annualization_factor: float = 252,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate annualised metrics from daily PnL in full three-year blocks.

    Parameters
    ----------
    pnl : pandas.Series or pandas.DataFrame
        Daily arithmetic excess PnL, one column per strategy. Return and
        volatility preserve input units; Sharpe is invariant to positive scaling.
        Timestamps must be unique, with at most one observation per calendar day.
        Coarser-than-daily observations are rejected.
    annualization_factor : float
        Daily observations per year, default 252. Return is daily mean times
        this factor; volatility is daily sample sd times its square root.
        These conventional annualisations do not correct serial dependence.
    end_year : int, optional
        Final calendar year of the latest three-year block. Defaults to the
        latest covered year. Earlier anchors support partition sensitivity.
        Blocks are non-overlapping and any initial remainder is discarded.

    Returns
    -------
    pandas.DataFrame
        Columns: variant, period_start, period_end, period_years, n_observations,
        ann_ret, ann_vol, sharpe. Boundaries are January 1 and December 31;
        period_years is always three. No monthly aggregation is performed.

    Notes
    -----
    Calendar coverage is a declared heuristic, not exchange-calendar validation:
    observations must reach within seven calendar days of each block boundary,
    and gaps of seven days or more invalidate a block. This tolerates weekends
    and short holiday closures while rejecting materially incomplete history.
    An ending year whose final observation precedes December 25 is incomplete.
    Explicit NaNs, including those at either edge, invalidate their whole block
    with a warning; they never shift boundaries. Different exchange calendars
    may require preparation upstream. All columns use the same dated span and
    anchor, so comparisons refer to the same periods.
    """
    if not np.isfinite(annualization_factor) or annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive and finite.")
    wide = _wide(pnl)
    if wide.index.normalize().has_duplicates:
        raise ValueError(
            "Daily input must have at most one observation per calendar day."
        )
    if len(wide) > 1:
        gaps = wide.index[1:] - wide.index[:-1]
        if np.median(gaps.asi8) > pd.Timedelta(1, unit="D").value:
            raise ValueError(
                "Supply daily PnL observations; coarser input cannot "
                "be annualised as daily data."
            )
    first, last = wide.index[0].normalize(), wide.index[-1].normalize()
    latest_year = last.year if last.month == 12 and last.day >= 25 else last.year - 1
    if end_year is None:
        end_year = latest_year
    elif (
        isinstance(end_year, bool)
        or not isinstance(end_year, (int, np.integer))
        or not first.year <= end_year <= latest_year
    ):
        raise ValueError("end_year must be an integer covered by the daily history.")
    factor, rows = float(annualization_factor), []
    for final_year in range(end_year, first.year + 1, -3):
        start = _year_boundary(final_year - 2, tz=wide.index.tz)
        end = _year_boundary(final_year, last=True, tz=wide.index.tz)
        # The partial initial remainder is excluded rather than promoted to a block.
        if first > start + pd.Timedelta(6, unit="D"):
            continue
        block = wide.loc[
            (wide.index >= start) & (wide.index < end + pd.Timedelta(1, unit="D"))
        ]
        covered = (
            len(block) >= 2
            and block.index[0].normalize() <= start + pd.Timedelta(6, unit="D")
            and block.index[-1].normalize() >= end - pd.Timedelta(6, unit="D")
            and ((block.index[1:] - block.index[:-1]) < pd.Timedelta(7, unit="D")).all()
        )
        for name in wide:
            values = block[name].to_numpy(dtype=float, na_value=np.nan)
            if not covered or not np.isfinite(values).all():
                warnings.warn(
                    "{}: block ending {} has incomplete daily coverage or missing values; skipped.".format(
                        name, end.date()
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                continue
            ret = factor * values.mean()
            vol = np.sqrt(factor) * values.std(ddof=1)
            rows.append(
                dict(
                    variant=name,
                    period_start=start,
                    period_end=end,
                    period_years=3,
                    n_observations=len(values),
                    ann_ret=ret,
                    ann_vol=vol,
                    sharpe=ret / vol if vol > 0 else np.nan,
                )
            )
    columns = [
        "variant",
        "period_start",
        "period_end",
        "period_years",
        "n_observations",
        "ann_ret",
        "ann_vol",
        "sharpe",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(["variant", "period_end"]).reset_index(drop=True)
    result.attrs.update(metric_freq="D", periods_per_year=factor, end_year=end_year)
    return result
