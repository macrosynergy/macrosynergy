import logging
from typing import Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils.df_utils import _long_to_wide, _wide_to_long

from macrosynergy.securities.validate import (
    _validate_frequency,
    _validate_constituents,
    _validate_returns,
    _validate_index_returns,
)

logger = logging.getLogger(__name__)


def _resolve_reconstitution_freq(
    rebalance_freq: str,
    reconstitution_freq: Optional[str],
) -> str:
    """
    Return the effective reconstitution frequency, defaulting to the rebalance frequency.

    Parameters
    ----------
    rebalance_freq : str
        Base rebalancing frequency used when ``reconstitution_freq`` is ``None``.
    reconstitution_freq : str or None
        Explicit reconstitution frequency, or ``None`` to inherit from
        ``rebalance_freq``.

    Returns
    -------
    str
        ``reconstitution_freq`` if not ``None``; otherwise ``rebalance_freq``.
    """
    return reconstitution_freq if reconstitution_freq is not None else rebalance_freq


def _assign_period_labels(dates: pd.DatetimeIndex, freq: str) -> pd.PeriodIndex:
    """
    Convert a DatetimeIndex to a PeriodIndex at the given frequency.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates to label.
    freq : str
        Pandas period frequency alias, e.g. ``"M"`` for month-end periods.

    Returns
    -------
    pd.PeriodIndex
        Period labels corresponding to each date in ``dates``.
    """
    return dates.to_period(freq)


def _build_reconstitution_membership(
    membership_wide: pd.DataFrame,
    recon_freq: str,
) -> pd.DataFrame:
    """
    Snap membership to the first trading day of each reconstitution period.

    For each period defined by ``recon_freq``, the membership values recorded on
    the period's first day are broadcast forward across every day in that period,
    so the effective composition is held constant within a period.

    Parameters
    ----------
    membership_wide : pd.DataFrame
        Wide-format binary membership matrix with a DatetimeIndex (business days)
        and one column per security (cid).
    recon_freq : str
        Pandas period frequency alias defining the reconstitution cadence,
        e.g. ``"M"`` for monthly snapshots.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with the same shape as ``membership_wide``, where
        each cell reflects the membership recorded on the first day of its period.
    """
    periods = _assign_period_labels(membership_wide.index, recon_freq)
    first_day_idx = (
        pd.Series(membership_wide.index, index=periods).groupby(level=0).first()
    )
    first_day_mem = membership_wide.loc[first_day_idx.values]
    first_day_mem.index = first_day_idx.index
    result = membership_wide.copy()
    result.values[:] = first_day_mem.loc[periods].values
    return result


def _apply_er_formula(
    stock_returns: pd.DataFrame,
    bench_returns: pd.Series,
    method: str,
) -> pd.DataFrame:
    """
    Apply the chosen excess-return formula element-wise to stock and benchmark returns.

    Parameters
    ----------
    stock_returns : pd.DataFrame
        Wide-format returns for individual stocks (rows = dates, columns = cids),
        expressed as decimal fractions (not percentage points).
    bench_returns : pd.Series
        Benchmark return series aligned to the same date index as ``stock_returns``,
        expressed as decimal fractions.
    method : str
        Excess-return formula: ``"ratio"``, ``"log"``, or ``"diff"``.

    Returns
    -------
    pd.DataFrame
        Wide-format excess returns with the same shape as ``stock_returns``,
        expressed as decimal fractions.
    """
    if method == "ratio":
        return stock_returns.add(1).div(bench_returns + 1, axis=0) - 1
    elif method == "log":
        return np.log1p(stock_returns).sub(np.log1p(bench_returns), axis=0)
    elif method == "diff":
        return stock_returns.sub(bench_returns, axis=0)


def compute_daily_weights(
    constituents: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_freq: str = "M",
    reconstitution_freq: Optional[str] = None,
    blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
) -> pd.DataFrame:
    """
    Compute daily float-adjusted equal weights for an index constituent set.

    Starting from an equal-weighted portfolio at the beginning of each rebalancing
    period, weights drift with realized returns within the period.  At each rebalance
    date the portfolio is reset to equal weights over the current constituent set.
    Reconstitution (membership changes) can be snapped to a coarser frequency than
    rebalancing via ``reconstitution_freq``.

    Parameters
    ----------
    constituents : pd.DataFrame or QuantamentalDataFrame
        Long-format DataFrame with columns ``"cid"``, ``"real_date"``, and
        ``"membership"`` (binary 0/1).  Each row records whether a security was a
        constituent on a given date.
    returns : pd.DataFrame or QuantamentalDataFrame
        Long-format DataFrame with columns ``"cid"``, ``"real_date"``, ``"xcat"``,
        and ``"value"`` (daily return in percentage points).  Must be filtered to a
        single xcat before passing.
    rebalance_freq : str, default ``"M"``
        Pandas period alias controlling how often the portfolio is reset to equal
        weights.  Must be one of ``{"B", "W", "M", "Q", "Y"}``.
    reconstitution_freq : str or None, default ``None``
        Pandas period alias controlling how often membership changes take effect.
        If ``None``, defaults to ``rebalance_freq``.
    blacklist : dict or None, default ``None``
        Mapping of ``cid`` → ``(start, end)`` :class:`pd.Timestamp` pairs
        identifying securities to exclude.  Exclusions are snapped to rebalance
        period starts, matching the weight-reset cadence.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns ``["real_date", "cid", "value"]``
        containing the daily portfolio weight for each constituent, with
        zero-weight rows dropped.
    """
    _validate_frequency(rebalance_freq, "rebalance_freq")
    if reconstitution_freq is not None:
        _validate_frequency(reconstitution_freq, "reconstitution_freq")
    _validate_constituents(constituents)
    _validate_returns(returns)

    recon_freq = _resolve_reconstitution_freq(rebalance_freq, reconstitution_freq)

    # Pivot to wide
    constituents["real_date"] = pd.to_datetime(constituents["real_date"])
    mem_wide = (
        _long_to_wide(constituents[["cid", "real_date", "membership"]], "membership")
        .fillna(0)
        .astype(int)
    )
    returns["real_date"] = pd.to_datetime(returns["real_date"])
    ret_wide = _long_to_wide(returns[["cid", "real_date", "value"]], "value")

    # Align columns
    common_cids = mem_wide.columns.intersection(ret_wide.columns)
    assert (
        len(common_cids) > 0
    ), "No common cids between constituents and returns DataFrames."
    mem_wide = mem_wide[common_cids]
    ret_wide = ret_wide[common_cids]

    # Reindex both to a common complete business-day calendar, then align
    all_dates = mem_wide.index.union(ret_wide.index).sort_values()
    full_bdays = pd.bdate_range(all_dates.min(), all_dates.max(), freq="B")

    mem_wide = mem_wide.reindex(full_bdays).ffill().fillna(0).astype(int)
    ret_wide = ret_wide.reindex(full_bdays).fillna(0.0) / 100.0  # pct -> decimal

    # Apply reconstitution: snapshot at period start, hold through period
    mem_effective = _build_reconstitution_membership(mem_wide, recon_freq)

    # Apply blacklist: zero out blacklisted securities on each rebalance date.
    # Snapshot blacklist state at the first day of each rebalancing period and
    # hold through the period, so a blacklisted security stays excluded until
    # the next rebalance where it is no longer blacklisted.
    if blacklist:
        rebal_periods_pre = _assign_period_labels(full_bdays, rebalance_freq)
        active_periods_before = (
            mem_effective.groupby(rebal_periods_pre)
            .first()
            .gt(0)
            .sum()
            .rename("active_periods_before")
        )

        bl_mask = pd.DataFrame(False, index=full_bdays, columns=mem_effective.columns)
        for cid, (start, end) in blacklist.items():
            if cid in bl_mask.columns:
                bl_mask.loc[(bl_mask.index >= start) & (bl_mask.index <= end), cid] = (
                    True
                )
            else:
                logger.info(
                    "Blacklist cid '%s' not found in constituent universe — skipped.",
                    cid,
                )
        bl_effective = _build_reconstitution_membership(
            bl_mask.astype(int), rebalance_freq
        ).astype(bool)
        mem_effective = mem_effective.where(~bl_effective, 0)

        active_periods_after = (
            mem_effective.groupby(rebal_periods_pre)
            .first()
            .gt(0)
            .sum()
            .rename("active_periods_after")
        )
        summary = pd.concat([active_periods_before, active_periods_after], axis=1)
        summary["periods_removed"] = (
            summary["active_periods_before"] - summary["active_periods_after"]
        )
        affected = summary[summary["periods_removed"] > 0]
        if affected.empty:
            logger.info("Blacklist applied but no rebalancing periods were affected.")
        else:
            logger.warning(
                "Blacklist reduced active rebalancing periods for %d security(ies):\n%s",
                len(affected),
                affected.to_string(),
            )

    # Assign rebalancing periods
    rebal_periods = _assign_period_labels(full_bdays, rebalance_freq)

    # Vectorized weight drift using cumprod within each rebalancing period.
    #
    # On rebalancing day 1, weight_i = (1/N) * membership_i.
    # On day d within the period, the unnormalized weight is:
    #   w_i(d) = w_i(0) * prod_{t=0}^{d-1}(1 + r_i(t))
    #
    # We shift the cumulative product so that day 0 uses the initial weight
    # (cumprod hasn't started yet) and day d reflects returns through day d-1.
    # Then we normalize row-wise so weights sum to 1.

    # Initial equal weights per period: 1/N for members, 0 for non-members
    n_members = mem_effective.groupby(rebal_periods).transform("first").sum(axis=1)
    initial_w = mem_effective.div(n_members.replace(0, np.nan), axis=0).fillna(0.0)

    # Cumulative growth factor within each period, shifted so day 0 = 1.0
    growth = (1 + ret_wide).groupby(rebal_periods).cumprod()
    growth_shifted = growth.groupby(rebal_periods).shift(1).fillna(1.0)

    # Unnormalized drifted weights
    weights_raw = initial_w * growth_shifted

    # Normalize so each row sums to 1
    row_sums = weights_raw.sum(axis=1).replace(0, np.nan)
    weights = weights_raw.div(row_sums, axis=0).fillna(0.0)

    # Convert to long, drop zero-weight rows
    weights_long = _wide_to_long(weights, value_name="value")
    weights_long = weights_long[weights_long["value"] > 0].reset_index(drop=True)

    return weights_long


def compute_index_returns(
    constituents: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_freq: str = "M",
    reconstitution_freq: Optional[str] = None,
    blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily index-level returns from constituent weights and individual returns.

    Wraps :func:`compute_daily_weights` to obtain daily float-adjusted weights, then
    computes the weighted-average return across constituents for each business day.

    Parameters
    ----------
    constituents : pd.DataFrame or QuantamentalDataFrame
        Long-format DataFrame with columns ``"cid"``, ``"real_date"``, and
        ``"membership"`` (binary 0/1).
    returns : pd.DataFrame or QuantamentalDataFrame
        Long-format DataFrame with columns ``"cid"``, ``"real_date"``, ``"xcat"``,
        and ``"value"`` (daily return in percentage points).  Must be filtered to a
        single xcat before passing.
    rebalance_freq : str, default ``"M"``
        Portfolio rebalancing frequency.  Must be one of ``{"B", "W", "M", "Q", "Y"}``.
    reconstitution_freq : str or None, default ``None``
        Membership reconstitution frequency.  Defaults to ``rebalance_freq`` when
        ``None``.
    blacklist : dict or None, default ``None``
        Mapping of ``cid`` → ``(start, end)`` :class:`pd.Timestamp` pairs for
        securities to exclude.

    Returns
    -------
    daily_index : pd.DataFrame
        DataFrame with columns ``["real_date", "value"]`` containing the daily
        index return in percentage points.
    weights_long : pd.DataFrame
        Long-format weight DataFrame as returned by :func:`compute_daily_weights`.
    """
    weights_long = compute_daily_weights(
        constituents, returns, rebalance_freq, reconstitution_freq, blacklist
    )

    # Pivot weights to wide for multiplication
    w_wide = weights_long.pivot(
        index="real_date", columns="cid", values="value"
    ).fillna(0.0)

    # Align returns to same dates/cids
    ret_wide = _long_to_wide(returns[["cid", "real_date", "value"]], "value")
    full_bdays = w_wide.index
    common_cids = w_wide.columns
    ret_wide = (
        ret_wide.reindex(index=full_bdays, columns=common_cids).fillna(0.0) / 100.0
    )

    # Daily index return = sum(w_i * r_i)
    daily_ret = 100 * (w_wide * ret_wide).sum(axis=1)

    daily_index = pd.DataFrame(
        {
            "real_date": full_bdays,
            "value": daily_ret.values,
        }
    )

    return daily_index, weights_long


def compute_excess_returns(
    returns: pd.DataFrame,
    index_returns: pd.DataFrame,
    method: str = "ratio",
    output_freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute per-stock excess (active) returns relative to a benchmark index.

    Three excess-return formulations are supported:

    - ``"ratio"`` : ``(1 + r_stock) / (1 + r_bench) - 1``
    - ``"log"``   : ``log(1 + r_stock) - log(1 + r_bench)``
    - ``"diff"``  : ``r_stock - r_bench``

    Optionally compounds daily returns to a lower frequency before computing the
    excess return.

    Parameters
    ----------
    returns : pd.DataFrame or QuantamentalDataFrame
        Long-format DataFrame with columns ``"cid"``, ``"real_date"``, ``"xcat"``,
        and ``"value"`` (daily return in percentage points).
    index_returns : pd.DataFrame or QuantamentalDataFrame
        DataFrame with columns ``"real_date"`` and ``"value"`` (daily index return
        in percentage points).  Must contain one row per date (no duplicates).
    method : str, default ``"ratio"``
        Excess-return formula.  One of ``{"ratio", "log", "diff"}``.
    output_freq : str or None, default ``None``
        If provided, daily returns are compounded to this frequency before the
        excess-return formula is applied.  Must be one of
        ``{"B", "W", "M", "Q", "Y"}``.  When ``None``, excess returns are
        computed at daily frequency.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns ``["real_date", "cid", "value"]``
        containing excess returns in percentage points.  Dates correspond to
        period-end timestamps when ``output_freq`` is set.
    """
    _validate_returns(returns)
    _validate_index_returns(index_returns)
    if output_freq is not None:
        _validate_frequency(output_freq, "output_freq")
    assert method in (
        "ratio",
        "log",
        "diff",
    ), f"method must be 'ratio', 'log', or 'diff', got '{method}'."

    # Pivot stock returns to wide, convert to decimal
    ret_wide = _long_to_wide(returns[["cid", "real_date", "value"]], "value") / 100.0
    index_returns = index_returns.copy(deep=True)
    index_returns["value"] = index_returns["value"] / 100.0  # pct -> decimal

    # Benchmark as Series
    bench = index_returns.set_index("real_date")["value"]

    # Align to common business-day index
    all_dates = ret_wide.index.union(bench.index).sort_values()
    full_bdays = pd.bdate_range(all_dates.min(), all_dates.max(), freq="B")

    # Keep NaN for stocks on dates they have no return (preserves sparsity)
    ret_wide = ret_wide.reindex(full_bdays)
    bench = bench.reindex(full_bdays).fillna(0.0)

    if output_freq is not None:
        periods = _assign_period_labels(full_bdays, output_freq)

        # Compound daily -> period; NaN days treated as no return (1+NaN -> NaN)
        # Use skipna=False via manual approach: fill NaN with 0 for prod, but
        # track which stock-periods have any data
        has_data = ret_wide.notna().groupby(periods).any()
        ret_filled = ret_wide.fillna(0.0)

        stock_period = (1 + ret_filled).groupby(periods).prod() - 1
        bench_period = (1 + bench).groupby(periods).prod() - 1

        er_wide = _apply_er_formula(stock_period, bench_period, method)
        er_wide = er_wide.where(has_data)

        er_wide.index = er_wide.index.to_timestamp(how="end")
        er_long = _wide_to_long(er_wide, value_name="value")
        er_long["value"] = er_long["value"] * 100.0  # decimal -> pct
        return er_long

    else:
        er_wide = _apply_er_formula(ret_wide, bench, method)
        # NaN propagation: stocks with no return on a date stay NaN
        er_long = _wide_to_long(er_wide, value_name="value")
        er_long["value"] = er_long["value"] * 100.0  # decimal -> pct
        return er_long
