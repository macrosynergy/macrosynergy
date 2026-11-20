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
    if np.iscomplexobj(pnl.to_numpy()) or np.isinf(pnl.to_numpy(dtype=float)).any():
        raise ValueError("PnL values must be real, without infinities.")
    return pnl.sort_index()


def period_metrics(
    pnl: Union[pd.Series, pd.DataFrame],
    period_months: int = 36,
    step_months: Optional[int] = None,
    *,
    annualization_factor: float = 252,
) -> pd.DataFrame:
    """Estimate annual return, volatility and Sharpe in end-anchored blocks.

    Parameters
    ----------
    pnl : pandas.Series or pandas.DataFrame
        Daily arithmetic PnL, with one column per strategy.
        Return/volatility output uses the input's units. For percentage PnL,
        volatility 10 means 10 percent per year; decimal input instead uses 0.10.
        Sharpe assumes the PnL already represents excess returns over the desired
        benchmark. Constant positive scaling preserves Sharpe.
        Supply daily observations: weekly or coarser dated streams are rejected.
        Missing individual dates are not reconstructed against a holiday calendar.
    period_months : int
        Calendar months per block; default 36. The initial remainder is dropped.
    step_months : int, optional
        Distance between blocks. Defaults to period_months. Overlap warns because
        metrics cannot then be treated as independent without further modelling.
    annualization_factor : float
        Daily observations per year, default 252. Return is the daily mean
        multiplied by this factor; volatility is the daily sample standard
        deviation multiplied by its square root. No monthly aggregation is used.

    Returns
    -------
    pandas.DataFrame
        variant, period_start, period_end, n_months, ann_ret, ann_vol and sharpe.
        Dates label the first/last calendar month ends, not individual trade dates.
        Blocks with absent months or explicit missing daily values are skipped
        with a warning. No serial-dependence correction is estimated here.
    """
    step = period_months if step_months is None else step_months
    for name, value, minimum in (
        ("period_months", period_months, 2),
        ("step_months", step, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(
                "{} must be an integer of at least {}.".format(name, minimum)
            )
    if not np.isfinite(annualization_factor) or annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive and finite.")
    wide = _wide(pnl)
    if wide.index.normalize().has_duplicates:
        raise ValueError(
            "Daily input must have at most one observation per calendar day."
        )
    factor = float(annualization_factor)
    rows = []
    for name in wide:
        series = wide[name]
        first, last = series.first_valid_index(), series.last_valid_index()
        if first is None:
            continue
        series = series.loc[first:last]
        if len(series) > 1 and (
            series.index[1:] - series.index[:-1]
        ).median() >= pd.Timedelta(days=7):
            raise ValueError(
                "Supply daily PnL observations; weekly or coarser input cannot "
                "be annualised as daily data."
            )
        observation_months = series.index.to_period("M")
        calendar = pd.period_range(
            observation_months[0], observation_months[-1], freq="M"
        )
        # Counts check calendar coverage only; metrics always use daily values.
        groups = series.groupby(observation_months)
        complete = (groups.count() == groups.size()).reindex(calendar, fill_value=False)
        for end in range(len(calendar), period_months - 1, -step):
            block = calendar[end - period_months : end]
            if not complete.iloc[end - period_months : end].all():
                warnings.warn(
                    "{}: block ending {} has missing months or daily values; skipped.".format(
                        name, block[-1].to_timestamp("M").date()
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                continue
            mask = (observation_months >= block[0]) & (observation_months <= block[-1])
            values = series.loc[mask].to_numpy(dtype=float)
            if len(values) < 2:
                continue
            ret = factor * values.mean()
            vol = np.sqrt(factor) * values.std(ddof=1)
            rows.append(
                dict(
                    variant=name,
                    period_start=block[0].to_timestamp("M"),
                    period_end=block[-1].to_timestamp("M"),
                    n_months=period_months,
                    ann_ret=ret,
                    ann_vol=vol,
                    sharpe=ret / vol if vol > 0 else np.nan,
                )
            )
    columns = [
        "variant",
        "period_start",
        "period_end",
        "n_months",
        "ann_ret",
        "ann_vol",
        "sharpe",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(["variant", "period_end"]).reset_index(drop=True)
    if step < period_months:
        warnings.warn(
            "Overlapping periods are dependent; do not treat them as independent observations.",
            UserWarning,
            stacklevel=2,
        )
    result.attrs.update(
        overlapping=step < period_months,
        metric_freq="D",
        periods_per_year=factor,
    )
    return result
