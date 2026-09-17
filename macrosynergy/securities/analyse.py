"""
Diagnostics for a portfolio of single securities: size, concentration, turnover and
return attribution, measured for the portfolio as a whole, for user-defined subgroups
of securities, and - where a benchmark is supplied - for the active position against
that benchmark.
"""

import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils.df_utils import _long_to_wide, _wide_to_long

from macrosynergy.securities.index import _apply_weight_drift, _assign_period_labels
from macrosynergy.securities.validate import _validate_frequency

logger = logging.getLogger(__name__)


#: Statistics reported for a standalone portfolio, i.e. ``active=False``.
STANDALONE_WEIGHT_STATS: List[str] = [
    "n_holdings",
    "effective_n",
    "weight",
    "gross_weight",
    "turnover",
    "turnover_annualised",
    "weight_autocorr",
]

#: Statistics reported against a benchmark, i.e. ``active=True``.
ACTIVE_WEIGHT_STATS: List[str] = [
    "n_active_holdings",
    "effective_active_n",
    "active_weight",
    "active_share",
    "active_turnover",
    "active_turnover_annualised",
    "active_weight_turnover",
    "active_weight_turnover_annualised",
    "active_weight_autocorr",
    "off_benchmark_n",
    "off_benchmark_weight",
    "benchmark_only_n",
]

#: Display label of every statistic, keyed by its column name. Use
#: :func:`weight_stat_labels` to key these by category name instead.
WEIGHT_STAT_LABELS: Dict[str, str] = {
    "n_holdings": "Non-zero holdings",
    "effective_n": "Effective holdings",  # inverse of the HHI concentration
    "weight": "Portfolio net weight, %",
    "gross_weight": "Portfolio gross weight, %",  # sum of absolute weights
    "turnover": "Portfolio turnover, %",  # traded at each rebalancing
    "turnover_annualised": "Portfolio turnover, % p.a.",
    "weight_autocorr": "Weight 1-period autocorrelation",
    "n_active_holdings": "Active holdings",
    "effective_active_n": "Effective active holdings",  # inverse participation ratio
    "active_weight": "Active net weight, %",  # over- or underweight vs the benchmark
    "active_share": "Active share, %",  # half the sum of absolute active weights
    "active_turnover": "Signal-driven turnover, %",  # portfolio minus benchmark
    "active_turnover_annualised": "Signal-driven turnover, % p.a.",
    "active_weight_turnover": "Active weight turnover, %",
    "active_weight_turnover_annualised": "Active weight turnover, % p.a.",
    "active_weight_autocorr": "Active weight 1-period autocorrelation",
    "off_benchmark_n": "Off-benchmark holdings",  # held, absent from the benchmark
    "off_benchmark_weight": "Off-benchmark weight, %",
    "benchmark_only_n": "Benchmark-only holdings",  # in the benchmark, never held
}

# Concentration statistics carry over verbatim from the standalone to the active
# calculation - only the matrix they are measured on changes - so they are computed
# once and renamed. "gross_weight" is the exception: halved, it becomes active share.
_ACTIVE_RENAME: Dict[str, str] = {
    "n_holdings": "n_active_holdings",
    "effective_n": "effective_active_n",
    "weight": "active_weight",
}

# Rebalancings a year implied by each supported cadence, i.e. the factor turning a
# per-rebalancing turnover into an annualised one. Deliberately not taken from
# `macrosynergy.management.constants.ANNUALIZATION_FACTORS`: that map keys the annual
# cadence as "A" where this module validates "Y", and it is iterated - not merely keyed
# - in `management.utils.frequency` to rank the single-letter aliases, so adding "Y" to
# it would silently reshuffle that ranking for unrelated callers.
_REBALANCINGS_PER_YEAR: Dict[str, float] = {
    "B": 252.0,  # trading days, i.e. already stated net of market holidays
    "W": 52.0,
    "M": 12.0,
    "Q": 4.0,
    "Y": 1.0,
}

# Gross exposure above which the weights are more likely to be percentage points than
# fractions. A long/short book can run well above 1, hence the generous threshold.
_PCT_WEIGHT_THRESHOLD: float = 5.0

# Cross-sectional dispersion, relative to a weight vector's own magnitude, below which
# the vector counts as flat and carries no correlation.
_FLAT_VECTOR_TOL: float = 1e-12


def _stat_xcat(xcat_prefix: str, stat: str) -> str:
    """
    Category name a statistic is written under when converted to a panel.

    Parameters
    ----------
    xcat_prefix : str
        Prefix naming the portfolio the statistics belong to, e.g. "PORT".
    stat : str
        Statistic's column name, e.g. "active_share".

    Returns
    -------
    str
        e.g. "PORT_ACTIVE_SHARE".
    """
    if not isinstance(xcat_prefix, str) or not xcat_prefix:
        raise TypeError("`xcat_prefix` must be a non-empty string.")
    return f"{xcat_prefix}_{stat.upper()}"


def weight_stat_labels(
    xcat_prefix: str = "PORT",
    benchmark: Optional[str] = None,
    stats: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Display labels for the statistics of :meth:`PortfolioAnalyser.weight_stats`, keyed
    by the category names they carry once converted to a panel.

    Parameters
    ----------
    xcat_prefix : str, default "PORT"
        Prefix the statistics were written under, matching the ``xcat_prefix``
        argument of :meth:`PortfolioAnalyser.weight_stats`.
    benchmark : str, optional
        Name of the benchmark the active statistics are measured against, appended to
        their labels as " (vs ...)". Standalone labels are left alone, since they do
        not depend on a benchmark.
    stats : list of str, optional
        Statistics to label. Default is every entry of
        :data:`STANDALONE_WEIGHT_STATS` followed by :data:`ACTIVE_WEIGHT_STATS`.

    Raises
    ------
    KeyError
        If ``stats`` names a statistic with no label in :data:`WEIGHT_STAT_LABELS`.

    Returns
    -------
    dict
        Mapping of category name to display label.

    Examples
    --------
    Labelling one portfolio measured against two different benchmarks:

    >>> labels = {
    ...     **weight_stat_labels("PORT", benchmark="SP500"),
    ...     **weight_stat_labels(
    ...         "PORTEW", benchmark="equal wgt", stats=ACTIVE_WEIGHT_STATS
    ...     ),
    ... }
    >>> labels["PORT_ACTIVE_SHARE"], labels["PORTEW_ACTIVE_SHARE"]
    ('Active share, % (vs SP500)', 'Active share, % (vs equal wgt)')
    """
    if stats is None:
        stats = STANDALONE_WEIGHT_STATS + ACTIVE_WEIGHT_STATS

    missing = [stat for stat in stats if stat not in WEIGHT_STAT_LABELS]
    if missing:
        raise KeyError(f"No label defined for statistic(s): {sorted(missing)}.")

    suffix = f" (vs {benchmark})" if benchmark is not None else ""
    return {
        _stat_xcat(xcat_prefix, stat): WEIGHT_STAT_LABELS[stat]
        + (suffix if stat in ACTIVE_WEIGHT_STATS else "")
        for stat in stats
    }


def _as_wide(df: pd.DataFrame, name: str, value_col: str = "value") -> pd.DataFrame:
    """
    Coerce a panel of security-level data to a wide (dates x cids) float matrix.

    Parameters
    ----------
    df : pd.DataFrame or QuantamentalDataFrame
        Either long format - columns ``"cid"``, ``"real_date"`` and a value column,
        optionally with ``"xcat"`` - or a wide frame with a date index (or a
        ``"real_date"`` column) and one column per security.
    name : str
        Name of the calling argument, used in error messages.
    value_col : str, default "value"
        Column holding the values when ``df`` is in long format.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If ``df`` is empty, is missing required columns, spans more than one
        ``"xcat"``, holds duplicate (cid, real_date) pairs, or has an index that
        cannot be read as dates.

    Returns
    -------
    pd.DataFrame
        Float matrix indexed by ``"real_date"`` with one column per ``"cid"``,
        sorted by date.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`{name}` must be a pandas DataFrame.")
    if df.empty:
        raise ValueError(f"`{name}` is empty.")

    if "cid" in df.columns:
        missing = {"real_date", value_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"Long-format `{name}` is missing columns: {sorted(missing)}."
            )
        if "xcat" in df.columns and df["xcat"].nunique() > 1:
            raise ValueError(
                f"`{name}` spans more than one xcat "
                f"({sorted(map(str, df['xcat'].unique()))}); reduce it to a single "
                "category before passing it in."
            )
        # Copied, so that coercing the dtypes below cannot write back into the
        # caller's frame.
        long = df[["real_date", "cid", value_col]].copy()
        long["real_date"] = pd.to_datetime(long["real_date"])
        # A QuantamentalDataFrame holds `cid` as categorical, which pivots into a
        # CategoricalIndex of columns; casting keeps the result identical to the one
        # the wide branch produces, so the same portfolio gives the same matrix
        # whichever format it arrives in.
        long["cid"] = long["cid"].astype(str)
        if long.duplicated(["real_date", "cid"]).any():
            raise ValueError(
                f"`{name}` holds duplicate (cid, real_date) pairs; each security must "
                "have at most one observation per date."
            )
        wide = _long_to_wide(long, value_col)
    else:
        wide = df.copy()
        if "real_date" in wide.columns:
            wide = wide.set_index("real_date")
        if not isinstance(wide.index, pd.DatetimeIndex):
            # A numeric index would otherwise be read as epoch nanoseconds, silently
            # relabelling the panel; it almost always signals a long frame whose
            # "cid" column is missing or differently named.
            if pd.api.types.is_numeric_dtype(wide.index):
                raise ValueError(
                    f"`{name}` was read as a wide frame but its index is numeric. A "
                    "wide frame must be indexed by date; a long frame must carry "
                    "'cid', 'real_date' and 'value' columns."
                )
            try:
                wide.index = pd.to_datetime(wide.index)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"`{name}` was read as a wide frame but its index could not be "
                    f"interpreted as dates: {exc}"
                ) from exc
        wide.columns = wide.columns.astype(str)

    try:
        wide = wide.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` holds non-numeric values: {exc}") from exc

    wide = wide.sort_index()
    wide.index.name = "real_date"
    wide.columns.name = "cid"
    return wide


def _align_active(
    weights: pd.DataFrame, benchmark: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align portfolio and benchmark weights onto a common universe and date index.

    The union of securities is taken and missing entries are filled with zero: a name
    held on only one side is a full active position against a zero weight on the
    other, not a missing observation. Dates are intersected, since an active weight is
    only defined where both sides are observed.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide portfolio weights (dates x cids).
    benchmark : pd.DataFrame
        Wide benchmark weights (dates x cids).

    Raises
    ------
    ValueError
        If the two share no dates, leaving no date an active weight is defined on.

    Returns
    -------
    weights, benchmark : pd.DataFrame
        Both sides on the common universe and dates.
    active : pd.DataFrame
        ``weights - benchmark``.
    """
    cids = weights.columns.union(benchmark.columns)
    dates = weights.index.intersection(benchmark.index)
    if len(dates) == 0:
        raise ValueError(
            "`weights` and `benchmark` share no dates; an active position cannot be "
            "formed."
        )
    w = weights.reindex(index=dates, columns=cids).fillna(0.0)
    b = benchmark.reindex(index=dates, columns=cids).fillna(0.0)
    return w, b, w - b


def _carry_targets_forward(
    target: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    rebalance_freq: str,
) -> pd.DataFrame:
    """
    Carry target weights onto a calendar, expiring them one rebalancing period on.

    A target panel records the book as it is set, typically once per rebalancing, so a
    target has to be carried over the days that do not re-state it. How far is the
    question a missing value cannot answer: carried without limit, the last target a
    security ever receives is held to the end of the sample, and a stalled signal feed
    reads as a live book. The cadence the caller already supplies bounds it - a target
    is in force for the rest of the period it was recorded in and for the one that
    follows, by which point a rebalancing has been and gone.

    Parameters
    ----------
    target : pd.DataFrame
        Wide target weights (dates x cids), NaN where none was recorded. Dates off the
        calendar are read rather than dropped, since a panel resampled to calendar
        month ends states a third of its targets on a weekend.
    calendar : pd.DatetimeIndex
        Business days the targets are wanted on.
    rebalance_freq : str
        Pandas period alias setting both the expiry and the periods whose coverage is
        checked.

    Raises
    ------
    ValueError
        If any rebalancing period on the calendar ends up with no target in force at
        all. Such a period cannot be traded, and carrying the previous one across it
        would annualise one rebalancing as though it were several.

    Returns
    -------
    pd.DataFrame
        The targets on ``calendar``, NaN where none is in force.
    """
    index = pd.DatetimeIndex(target.index.union(calendar))
    # Period ordinals are consecutive at every supported cadence, so "expired" is a
    # difference of more than one whatever the alias.
    ordinals = np.asarray(
        _assign_period_labels(index, rebalance_freq).astype("int64")
    )[:, None]

    observed = target.reindex(index=index)
    # The period a target was recorded in travels forward alongside the target itself,
    # so that expiry is counted in rebalancings rather than in days.
    recorded_in = pd.DataFrame(
        np.where(observed.notna(), ordinals, np.nan),
        index=index,
        columns=observed.columns,
    ).ffill()
    carried = (
        observed.ffill().where(recorded_in.ge(ordinals - 1)).reindex(index=calendar)
    )

    periods = _assign_period_labels(calendar, rebalance_freq)
    covered = carried.notna().any(axis=1).groupby(periods).any()
    if not covered.all():
        empty = [str(p) for p in covered.index[~covered]]
        shown = ", ".join(empty[:10]) + (", ..." if len(empty) > 10 else "")
        raise ValueError(
            f"`weights` records no target in force for {len(empty)} '{rebalance_freq}' "
            f"rebalancing period(s): {shown}. A target is carried forward for one "
            "period beyond the one it was recorded in and then expires, so a gap this "
            "long leaves a rebalancing with nothing to trade to. Record a target in "
            "each period, or coarsen `rebalance_freq`."
        )
    return carried


def _universe_mask(
    universe: pd.DataFrame,
    cids: pd.Index,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Coerce an investable-universe frame to a boolean mask on a given calendar.

    Membership is a state rather than an observation, so unlike a target it is carried
    forward without limit: an exit is stated as ``False`` and not as an absence, so the
    ambiguity that forces :func:`_carry_targets_forward` to expire its input does not
    arise. What the frame leaves unsaid - dates before its first observation, a
    security it never mentions - is read as investable, so that a partial frame narrows
    the book rather than emptying it.

    Parameters
    ----------
    universe : pd.DataFrame or QuantamentalDataFrame
        Boolean membership per security and date, long or wide, truthy where the
        security is investable.
    cids : pd.Index
        Securities the mask must cover.
    calendar : pd.DatetimeIndex
        Dates the mask is returned on.

    Raises
    ------
    ValueError
        If ``universe`` does not name exactly the securities in ``cids``. A membership
        frame that is stale, or whose labels are spelled differently, would otherwise
        silently drop the securities it fails to mention.

    Returns
    -------
    pd.DataFrame
        Boolean matrix indexed by ``calendar`` with one column per cid in ``cids``.
    """
    wide = _as_wide(universe, "universe")

    missing = pd.Index(cids).difference(wide.columns)
    unknown = wide.columns.difference(pd.Index(cids))
    if len(missing) or len(unknown):
        raise ValueError(
            "`universe` must name exactly the securities covered by `weights` and "
            "`returns`. Missing from `universe`: "
            f"{sorted(map(str, missing))}; unknown to `weights` and `returns`: "
            f"{sorted(map(str, unknown))}."
        )

    wide = wide.reindex(columns=cids)
    # Carried on the union of the two calendars, so that a membership record dated off
    # the business-day grid, or before it starts, still takes effect.
    wide = (
        wide.reindex(index=wide.index.union(calendar)).ffill().reindex(index=calendar)
    )
    return wide.fillna(1.0).ne(0.0)


def _concentration_stats(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Per-date size and concentration statistics of a wide weight matrix.

    Only a present, non-zero weight counts as held, so that the zero-filled remainder
    of the universe never registers as a holding.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide weight matrix (dates x cids). Weights are expected as fractions.

    Returns
    -------
    pd.DataFrame
        Indexed by ``"real_date"``, with columns "n_holdings" (count of non-zero
        positions), "effective_n" (reciprocal of the Herfindahl index of gross weight
        shares, i.e. the inverse participation ratio), "weight" (net weight in
        percentage points) and "gross_weight" (sum of absolute weights in percentage
        points). Dates with no position at all carry no reading beyond a zero count.
    """
    held = weights.notna() & weights.ne(0.0)
    n_holdings = held.sum(axis=1)

    abs_w = weights.abs().where(held)
    gross = abs_w.sum(axis=1)
    hhi = abs_w.div(gross.replace(0.0, np.nan), axis=0).pow(2).sum(axis=1)

    stats = pd.DataFrame(
        {
            "n_holdings": n_holdings,
            "effective_n": 1.0 / hhi.replace(0.0, np.nan),
            "weight": 100.0 * weights.where(held).sum(axis=1),
            "gross_weight": 100.0 * gross,
        }
    )
    stats.loc[n_holdings.eq(0), ["effective_n", "weight", "gross_weight"]] = np.nan
    stats.index.name = "real_date"
    return stats


def _assert_unbroken_schedule(periods: pd.PeriodIndex, rebalance_freq: str) -> None:
    """
    Raise if any rebalancing period in the sample carries no observation at all.

    Every turnover reading is taken between two consecutive trade dates and is charged
    to exactly one rebalancing. A period the weights skip entirely silently widens one
    of those gaps to span two periods or more, so the reading picks up the trading of
    several rebalancings and its annualisation then scales it as though it were one.
    The error is invisible in the output - a plausible number, simply too large - which
    is why this raises rather than warns.

    Business-day rebalancing is exempt: there, each period is a single observation, so
    an absent period is a market holiday rather than missing data, and the 252-day
    annualisation factor is already stated net of holidays.

    Parameters
    ----------
    periods : pd.PeriodIndex
        Period label of every observed date, from
        :func:`macrosynergy.securities.index._assign_period_labels`.
    rebalance_freq : str
        Pandas period alias defining the rebalancing cadence, one of
        {"B", "W", "M", "Q", "Y"}.

    Raises
    ------
    ValueError
        If a period between the first and last observation holds no date.
    """
    if rebalance_freq == "B":
        return

    observed = periods.unique()
    expected = pd.period_range(observed.min(), observed.max(), freq=observed.freq)
    missing = expected.difference(observed)
    if len(missing) == 0:
        return

    shown = ", ".join(str(period) for period in missing[:5])
    if len(missing) > 5:
        shown += f", ... ({len(missing)} in total)"
    raise ValueError(
        f"The weights have no observation in {len(missing)} '{rebalance_freq}' "
        f"rebalancing period(s) between {observed.min()} and {observed.max()}: "
        f"{shown}. Turnover is charged to one rebalancing apiece and annualised on "
        "that basis, so a gap would overstate both. Trim the sample with `start` and "
        "`end`, or supply the missing dates."
    )


def _trade_dates(index: pd.DatetimeIndex, rebalance_freq: str) -> pd.DatetimeIndex:
    """
    First observed date of each rebalancing period, i.e. the dates the book is set on.

    Trading only happens where the portfolio is reset to its targets. In between, a
    daily weight matrix still changes every day as positions drift with returns, which
    is market movement rather than turnover.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Dates covered by the weight matrix.
    rebalance_freq : str
        Pandas period alias defining the rebalancing cadence, one of
        {"B", "W", "M", "Q", "Y"}. Matches the argument of the same name on
        :func:`macrosynergy.securities.index.compute_daily_weights`.

    Raises
    ------
    ValueError
        If a rebalancing period in the sample holds no observation. See
        :func:`_assert_unbroken_schedule`.

    Returns
    -------
    pd.DatetimeIndex
        One date per rebalancing period, in ascending order.
    """
    periods = _assign_period_labels(index, rebalance_freq)
    _assert_unbroken_schedule(periods, rebalance_freq)
    firsts = pd.Series(index, index=periods).groupby(level=0).first()
    return pd.DatetimeIndex(firsts.values, name="real_date")


def _no_trade_weights(
    weights: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Weights each rebalancing would have inherited had nothing been traded since the
    previous one.

    Anchoring on the previous trade date and compounding returns forward to the
    current one reconstructs the book the market alone would have produced. It is the
    right baseline whether or not the supplied weights already drift: weights that do
    drift are reproduced exactly, so the difference against them is zero on every
    non-trading date, while static target weights are grown over the whole holding
    period rather than left flat.

    Capital is grown by the exposure-weighted return rather than by the row sum, so
    that a book which is not fully invested - a long/short or market-neutral one -
    is not rescaled by a denominator that can approach zero.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide weight matrix (dates x cids), with weights expressed as fractions.
    trade_dates : pd.DatetimeIndex
        Dates on which the book is reset, as returned by :func:`_trade_dates`.
    returns : pd.DataFrame, optional
        Wide returns (dates x cids) in percentage points. When None the anchor weights
        are carried forward unchanged, so the comparison against them measures the
        change in target weights with drift left in.

    Returns
    -------
    pd.DataFrame
        Indexed by every trade date bar the first, which has no prior anchor, with the
        same columns as ``weights``.
    """
    later, earlier = trade_dates[1:], trade_dates[:-1]
    anchor = weights.reindex(index=earlier).fillna(0.0).to_numpy(dtype=float)

    if returns is None:
        carried = anchor
    else:
        rets = (
            returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
            / 100.0
        )
        # Growth from the start of the sample through the prior date, so that the
        # growth between any two dates is the ratio of two of its rows.
        growth = (1.0 + rets).cumprod().shift(1)
        growth.iloc[0] = 1.0
        ratio = growth.reindex(index=later).to_numpy(dtype=float) / growth.reindex(
            index=earlier
        ).to_numpy(dtype=float)

        capital = 1.0 + (anchor * (ratio - 1.0)).sum(axis=1)
        capital = np.where(capital != 0.0, capital, np.nan)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            carried = anchor * ratio / capital

    return pd.DataFrame(carried, index=later, columns=weights.columns)


def _turnover_against(
    weights: pd.DataFrame,
    carried: pd.DataFrame,
    columns: Optional[pd.Index] = None,
) -> pd.Series:
    """
    One-way turnover traded at each rebalancing, in percentage points.

    Measured against the book the previous rebalancing would have left untouched, so
    that only the buys and sells are counted and the drift in between is not. Missing
    weights are read as "not held", so entries and exits register as a full move from
    or to a zero weight.

    Restricting ``columns`` measures one subgroup's share of the portfolio's trading.
    The baseline must have been built on the whole portfolio and sliced here rather
    than rebuilt per subgroup, so that each subgroup is drifted against the
    portfolio's capital and the readings stay additive across subgroups.

    Parameters
    ----------
    weights : pd.DataFrame
        Wide weight matrix (dates x cids), with weights expressed as fractions.
    carried : pd.DataFrame
        No-trade baseline indexed by trade date, from :func:`_no_trade_weights`.
    columns : pd.Index, optional
        Securities to measure. Default is every column of ``weights``.

    Returns
    -------
    pd.Series
        Indexed by ``"real_date"`` over every date of ``weights``, carrying a reading
        on each trade date bar the first and NaN everywhere else.
    """
    cols = weights.columns if columns is None else columns
    turnover = pd.Series(np.nan, index=weights.index, name="turnover")
    turnover.index.name = "real_date"
    if carried.empty:
        return turnover

    traded = weights.loc[carried.index, cols].fillna(0.0) - carried[cols]
    turnover.loc[carried.index] = 100.0 * 0.5 * traded.abs().sum(axis=1)
    return turnover


def _masked_rowwise_corr(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """
    Correlation of each row of ``cur`` against the matching row of ``prev``.

    Each pair is restricted to the columns carrying a position in at least one of the
    two rows. Without that restriction the zero-filled remainder of the universe
    dominates the cross-section and pushes the correlation trivially towards one.

    Parameters
    ----------
    cur : np.ndarray
        Two-dimensional array of weight vectors, one row per observation.
    prev : np.ndarray
        Array of the same shape holding the vectors to correlate against.

    Returns
    -------
    np.ndarray
        One correlation per row, NaN where fewer than two columns are relevant or
        either vector is flat across them.
    """
    mask = ((cur != 0.0) | (prev != 0.0)).astype(float)
    n_relevant = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Deviations from the masked cross-sectional mean; the mask zeroes out every
        # irrelevant security so it contributes to neither moment.
        mean_c = (cur * mask).sum(axis=1, keepdims=True) / n_relevant[:, None]
        mean_p = (prev * mask).sum(axis=1, keepdims=True) / n_relevant[:, None]
        dev_c = (cur - mean_c) * mask
        dev_p = (prev - mean_p) * mask

        spread_c = np.sqrt((dev_c**2).sum(axis=1))
        spread_p = np.sqrt((dev_p**2).sum(axis=1))
        # A vector with no cross-sectional dispersion has no correlation to report -
        # an equal-weighted book is the standard case. Judged against the vector's own
        # magnitude rather than against zero, so that rounding in the weights cannot
        # leave a flat vector looking dispersed and hand back an arbitrary reading.
        varies = (
            spread_c > _FLAT_VECTOR_TOL * np.sqrt(((cur * mask) ** 2).sum(axis=1))
        ) & (spread_p > _FLAT_VECTOR_TOL * np.sqrt(((prev * mask) ** 2).sum(axis=1)))
        corr = (dev_c * dev_p).sum(axis=1) / (spread_c * spread_p)
    return np.where(varies & (n_relevant > 1), corr, np.nan)


def _weight_autocorr(
    weights: pd.DataFrame, trade_dates: pd.DatetimeIndex
) -> pd.Series:
    """
    Correlation of each rebalancing's weight vector with the previous rebalancing's.

    Measured across consecutive trade dates rather than consecutive days: between
    rebalancings the book only drifts, so a daily reading answers "does today's
    position look like yesterday's", which is trivially yes, instead of "does this
    rebalancing keep the last one's positions".

    Parameters
    ----------
    weights : pd.DataFrame
        Wide weight matrix (dates x cids).
    trade_dates : pd.DatetimeIndex
        Dates on which the book is reset, as returned by :func:`_trade_dates`.

    Returns
    -------
    pd.Series
        Indexed by ``"real_date"`` over every date of ``weights``, carrying a reading
        on each trade date bar the first and NaN everywhere else.
    """
    autocorr = pd.Series(np.nan, index=weights.index, name="weight_autocorr")
    autocorr.index.name = "real_date"
    if len(trade_dates) < 2 or weights.shape[1] < 2:
        return autocorr

    later, earlier = trade_dates[1:], trade_dates[:-1]
    cur = weights.reindex(index=later).fillna(0.0).to_numpy(dtype=float)
    prev = weights.reindex(index=earlier).fillna(0.0).to_numpy(dtype=float)
    autocorr.loc[later] = _masked_rowwise_corr(cur, prev)
    return autocorr


def _group_labels(
    columns: pd.Index,
    group_map: Optional[Dict[str, str]],
    other_label: str,
) -> pd.Series:
    """
    Resolve the subgroup label of every security in ``columns``.

    Parameters
    ----------
    columns : pd.Index
        Security identifiers (cids) to label.
    group_map : dict or None
        Mapping of cid to subgroup label.
    other_label : str
        Fallback for a security the mapping omits, and for one it maps to a missing
        label.

    Returns
    -------
    pd.Series
        Group label per security, indexed by cid.
    """
    mapping = dict(group_map or {})
    labels = pd.Series(
        [mapping.get(cid, other_label) for cid in columns], index=columns, dtype=object
    )
    labels = labels.where(labels.notna(), other_label).astype(str)

    unmapped = [cid for cid in columns if cid not in mapping]
    if unmapped:
        logger.info(
            "%d security(ies) are absent from `groups` and were assigned to '%s': %s",
            len(unmapped),
            other_label,
            sorted(map(str, unmapped)),
        )
    return labels


def _cid_label_map(labels: Iterable[str]) -> Dict[str, str]:
    """
    Map subgroup labels onto valid cross-section identifiers.

    A cid is the part of a ticker preceding the first underscore, so an underscore in
    a label would corrupt the ticker it ends up in; underscores are replaced with
    hyphens.

    Parameters
    ----------
    labels : iterable of str
        Labels to convert.

    Raises
    ------
    ValueError
        If two distinct labels collide once underscores are replaced.

    Returns
    -------
    dict
        Mapping of label to cid.
    """
    mapping = {str(lbl): str(lbl).replace("_", "-") for lbl in dict.fromkeys(labels)}

    collisions: Dict[str, List[str]] = {}
    for raw, cid in mapping.items():
        collisions.setdefault(cid, []).append(raw)
    clashing = {cid: raws for cid, raws in collisions.items() if len(raws) > 1}
    if clashing:
        raise ValueError(
            "Subgroup labels collide once underscores are replaced with hyphens, so "
            f"they cannot be used as cross-sections: {clashing}."
        )
    return mapping


class PortfolioAnalyser:
    """
    Weight and return diagnostics for a portfolio of single securities.

    The same statistics are available along three axes, combined freely:

    - the portfolio as a whole, which is the default;
    - subgroups of securities defined by ``groups``, selected with ``by_group=True``,
      measured on the subgroup's own columns of the portfolio weight matrix;
    - the active position against ``benchmark``, selected with ``active=True``, which
      applies the same measurements to the portfolio's weights net of the benchmark's.

    Which statistics aggregate across subgroups, and which are daily rather than
    per-rebalancing readings, is set out in :meth:`weight_stats`.

    Parameters
    ----------
    weights : pd.DataFrame or QuantamentalDataFrame
        Portfolio weights per security and date, as fractions of the portfolio. Either
        long format with columns ``"cid"``, ``"real_date"`` and ``"value"`` - the
        output of :func:`macrosynergy.securities.index.compute_daily_weights` - or a
        wide frame indexed by date with one column per security. A missing weight is
        read as "not held". Weights may be static targets held flat between
        rebalancings or already drifting with returns; see
        :meth:`adjust_weights_with_drift` to turn the former into the latter. That
        method requires a ``universe``, since a target panel alone cannot say whether a
        security stopped being re-stated or stopped being investable, and holding one
        that has left the universe at its last weight never sells it.
    rebalance_freq : str
        Pandas period alias defining how often the book is reset to its targets, one
        of {"B", "W", "M", "Q", "Y"}. Matches the argument of the same name on
        :func:`macrosynergy.securities.index.compute_daily_weights`, and must describe
        the *portfolio's* schedule even when measuring against a benchmark: the
        benchmark reconstitutes on its own, and only the portfolio's rebalancings say
        when the book was actually traded.
    benchmark : pd.DataFrame or QuantamentalDataFrame, optional
        Benchmark weights in the same format and units as ``weights``. Required for
        any ``active=True`` call. Portfolio and benchmark are aligned on the union of
        their securities and the intersection of their dates.
    returns : pd.DataFrame or QuantamentalDataFrame, optional
        Single-security returns in the same format as ``weights``, in percentage
        points as JPMaQS return categories are. Required by :meth:`attribution`, whose
        contributions come back in the same units. Also used to remove drift from the
        turnover readings; without it, turnover measures the change in weights between
        rebalancings with drift left in.
    groups : dict or pd.Series, optional
        Mapping of security (cid) to subgroup label - sector, region, book, or any
        other partition. Required for any ``by_group=True`` call.
    start, end : str or pd.Timestamp, optional
        Bounds of the date window to retain. Default is None on either side, i.e. the
        earliest and latest dates available.
    other_label : str, default "OTHER"
        Subgroup label for securities missing from ``groups``.
    portfolio_name : str, default "PORTFOLIO"
        Name of the portfolio as a whole. Used as the cross-section of whole-portfolio
        statistics converted to a QuantamentalDataFrame, and as the label of the total
        column in :meth:`attribution`.

    Raises
    ------
    TypeError
        If ``groups`` is neither a mapping nor a pandas Series.
    ValueError
        If any input frame is empty, malformed, or cannot be aligned, or if the
        weights skip a whole ``rebalance_freq`` period, which would let one turnover
        reading cover several rebalancings.

    Attributes
    ----------
    weights, benchmark, returns : pd.DataFrame
        The inputs as wide (dates x cids) float matrices, trimmed to the date window.
        ``benchmark`` and ``returns`` are None when not supplied.
    active_weights : pd.DataFrame
        Portfolio weights net of the benchmark's. None when no benchmark was supplied.
    groups : pd.Series
        Subgroup label per security, indexed by cid. None when no mapping was supplied.
    cids : list of str
        Every security covered, i.e. the portfolio's universe widened by the
        benchmark's.
    trade_dates : pd.DatetimeIndex
        The dates the book is reset on, implied by ``rebalance_freq``.
    rebalancings_per_year : float
        Rebalancings a year implied by ``rebalance_freq``, i.e. the factor the
        annualised turnovers are scaled by.
    start, end : pd.Timestamp
        First and last date of the portfolio weights actually retained.

    Notes
    -----
    Weights are expected as fractions, i.e. a fully invested long-only portfolio sums
    to one. All reported weights, turnovers and active shares are in percentage points.

    The annualised turnover answers what the book costs to run over a year and is the
    one to compare across cadences - but it is not a cadence-neutral measure of how
    active a strategy is, and the difference matters. Turnover is the length of the
    path the weights travel, not the distance between their endpoints. Only where every
    weight moves monotonically between rebalancings does the path length telescope, and
    the same journey then annualises to the same figure however finely it is cut.
    Movement that reverses does not: signal noise, and the drift a fixed target has to
    be pulled back from, lengthen the measured path the more often it is measured. That
    component behaves like a random walk, whose path length grows with the square root
    of the number of steps, so its annualised turnover scales with the square root of
    the rebalancing frequency - a daily and an annual rebalancing of identical targets
    over identical markets differ by a factor of around ``sqrt(252)`` on it.
    """

    def __init__(
        self,
        weights: pd.DataFrame,
        rebalance_freq: str,
        benchmark: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        groups: Optional[Union[Dict[str, str], pd.Series]] = None,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        other_label: str = "OTHER",
        portfolio_name: str = "PORTFOLIO",
    ):
        if not isinstance(other_label, str):
            raise TypeError("`other_label` must be a string.")
        if not isinstance(portfolio_name, str):
            raise TypeError("`portfolio_name` must be a string.")
        if groups is not None and not isinstance(groups, (dict, pd.Series)):
            raise TypeError("`groups` must be a dict or a pandas Series.")
        _validate_frequency(rebalance_freq, "rebalance_freq")

        self.other_label = other_label
        self.portfolio_name = portfolio_name
        self.rebalance_freq = rebalance_freq
        self.rebalancings_per_year = _REBALANCINGS_PER_YEAR[rebalance_freq]

        self.weights = self._trim(_as_wide(weights, "weights"), start, end)
        self.benchmark = (
            self._trim(_as_wide(benchmark, "benchmark"), start, end)
            if benchmark is not None
            else None
        )
        self.returns = (
            self._trim(_as_wide(returns, "returns"), start, end)
            if returns is not None
            else None
        )
        self.start, self.end = self.weights.index.min(), self.weights.index.max()
        # Derived from the portfolio's own calendar: when measuring against a
        # benchmark, the benchmark drifts and reconstitutes on its own schedule, so
        # only the portfolio's rebalancings say when the book was actually traded.
        self.trade_dates = _trade_dates(self.weights.index, rebalance_freq)
        if len(self.trade_dates) < 2:
            logger.warning(
                "The weights span fewer than two '%s' rebalancing periods, so no "
                "turnover or autocorrelation can be measured.",
                rebalance_freq,
            )

        # Held alongside the raw weights: the standalone statistics must be measured
        # on the portfolio's own universe and calendar, unaffected by the benchmark.
        if self.benchmark is not None:
            self._w_aligned, self._b_aligned, self.active_weights = _align_active(
                self.weights, self.benchmark
            )
        else:
            self._w_aligned = self._b_aligned = self.active_weights = None

        self._group_map = (
            {str(k): v for k, v in dict(groups).items()} if groups is not None else None
        )
        universe = (
            self.weights.columns
            if self.benchmark is None
            else self.weights.columns.union(self.benchmark.columns)
        )
        self.cids = list(universe)
        self.groups = (
            _group_labels(universe, self._group_map, other_label)
            if self._group_map is not None
            else None
        )

        self._warn_if_percentage_weights()

    @staticmethod
    def adjust_weights_with_drift(
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        universe: pd.DataFrame,
        rebalance_freq: str = "M",
    ) -> pd.DataFrame:
        """
        Let static target weights drift with returns between rebalancings.

        Target weights recorded at each rebalancing describe the book as it is set,
        not as it is held: between rebalancings the positions move with the market.
        This reconstructs the daily path - the portfolio is reset to its targets on the
        first business day of each period and drifts with returns until the next - so
        that concentration and attribution are measured on the book actually held.

        Statistics measured on the result are unaffected by the difference for
        ``turnover``, which removes the drift either way, but ``n_holdings``,
        ``effective_n`` and the weight columns all read differently on a drifting book
        than on a flat one.

        Parameters
        ----------
        weights : pd.DataFrame or QuantamentalDataFrame
            Target weights per security and date, long or wide, as fractions. Values
            need not be normalised, and are carried over the days that do not re-state
            them - for the rest of the rebalancing period they were recorded in and for
            the one that follows, after which they expire. A panel that skips a whole
            period is therefore rejected rather than carried across it.

            The carry alone cannot tell a security that is merely not re-stated today
            from one that has left the investable set, which is why ``universe`` is
            required and not inferred.
        returns : pd.DataFrame or QuantamentalDataFrame
            Single-security returns in the same format, in percentage points.
        universe : pd.DataFrame or QuantamentalDataFrame
            Boolean investable universe per security and date, long or wide, truthy
            where the security can be held. A security is dropped from the book where
            it is false, mid-period as readily as on a rebalancing, and the survivors
            are rescaled to keep the row summing to one. Membership is carried forward
            without limit, so it need only be recorded when it changes, and what it
            leaves unsaid counts as investable: a partial frame narrows the book rather
            than emptying it. Pass an all-true frame for a fixed universe.
        rebalance_freq : str, default "M"
            Pandas period alias defining how often the book is reset to its targets,
            one of {"B", "W", "M", "Q", "Y"}. Also bounds the carry above.

        Returns
        -------
        pd.DataFrame
            Wide daily weights (business days x cids), each row summing to one where
            any weight is in force, ready to pass back in as ``weights``. The calendar
            opens on the first date carrying a target, not on the first return: a
            rebalancing period preceding every target has no book to report. A security
            outside the universe carries NaN rather than 0.0 - both read as "not held"
            by every statistic, so the distinction is there to be reported on, not to
            change a reading.

        Raises
        ------
        ValueError
            If ``rebalance_freq`` is not a supported alias, if ``universe`` does not
            name exactly the securities covered by ``weights`` and ``returns``, or if
            ``weights`` leaves a whole rebalancing period without a target.

        See Also
        --------
        macrosynergy.securities.index.compute_daily_weights : the same drift applied to
            an index constituent set. Membership is explicit there, in
            ``constituents``; this was the one entry point where it was implicit.
        """
        _validate_frequency(rebalance_freq, "rebalance_freq")
        target = _as_wide(weights, "weights")
        rets = _as_wide(returns, "returns")

        cids = target.columns.union(rets.columns)
        # Opening on the first target rather than the first return. A period that
        # precedes every target carries none on its first row, which is the row
        # `_apply_weight_drift` normalises the period by, so the period would be
        # divided by zero and silently deleted.
        calendar = pd.bdate_range(
            target.index.min(), target.index.union(rets.index).max(), freq="B"
        )
        target = _carry_targets_forward(
            target.reindex(columns=cids), calendar, rebalance_freq
        )
        rets = rets.reindex(index=calendar, columns=cids).fillna(0.0) / 100.0

        # Masked before the drift, not after: `_apply_weight_drift` closes on a
        # row-wise normalisation, which then rescales the survivors itself.
        investable = _universe_mask(universe, cids, calendar)
        drifted = _apply_weight_drift(
            target.where(investable).fillna(0.0),
            rets,
            _assign_period_labels(calendar, rebalance_freq),
        )

        drifted = drifted.where(investable)
        drifted.index.name = "real_date"
        drifted.columns.name = "cid"
        return drifted

    @staticmethod
    def _trim(
        wide: pd.DataFrame,
        start: Optional[Union[str, pd.Timestamp]],
        end: Optional[Union[str, pd.Timestamp]],
    ) -> pd.DataFrame:
        """
        Restrict a wide frame to the ``[start, end]`` date window.

        Parameters
        ----------
        wide : pd.DataFrame
            Wide frame indexed by date.
        start, end : str or pd.Timestamp or None
            Bounds of the window, either of them None for no bound on that side.

        Raises
        ------
        ValueError
            If the window leaves no dates, which otherwise surfaces much later as an
            empty statistics frame.

        Returns
        -------
        pd.DataFrame
            The trimmed frame.
        """
        if start is None and end is None:
            return wide
        trimmed = wide.loc[
            pd.Timestamp(start) if start is not None else None : (
                pd.Timestamp(end) if end is not None else None
            )
        ]
        if trimmed.empty:
            raise ValueError(
                f"No dates remain between start={start} and end={end}; the data spans "
                f"{wide.index.min():%Y-%m-%d} to {wide.index.max():%Y-%m-%d}."
            )
        return trimmed

    def _warn_if_percentage_weights(self) -> None:
        """
        Warn when the weights look like percentage points rather than fractions.
        """
        gross = self.weights.abs().sum(axis=1).replace(0.0, np.nan)
        typical = float(gross.median()) if gross.notna().any() else np.nan
        if np.isfinite(typical) and typical > _PCT_WEIGHT_THRESHOLD:
            logger.warning(
                "Gross exposure of `weights` has a median of %.1f, suggesting "
                "percentage points rather than fractions; reported weights and "
                "turnovers will be overstated by a factor of 100.",
                typical,
            )

    def _resolve_frames(self, active: bool) -> Tuple[pd.DataFrame, ...]:
        """
        Return the weight matrices a statistics call should be measured on.

        Parameters
        ----------
        active : bool
            If True, measure against the benchmark.

        Raises
        ------
        ValueError
            If ``active`` is True but no benchmark was supplied.

        Returns
        -------
        tuple of pd.DataFrame
            The raw portfolio weights alone, or the benchmark-aligned portfolio,
            benchmark and active matrices. Callers read the last element, which is the
            matrix to measure either way.
        """
        if not active:
            return (self.weights,)
        w, b, active_w = self._w_aligned, self._b_aligned, self.active_weights
        if w is None or b is None or active_w is None:
            raise ValueError(
                "`benchmark` must be supplied to PortfolioAnalyser for active "
                "statistics."
            )
        return (w, b, active_w)

    def _no_trade_baselines(
        self, frames: Tuple[pd.DataFrame, ...]
    ) -> Tuple[pd.DataFrame, ...]:
        """
        No-trade baseline for every weight matrix a statistics call will measure.

        Built once on the whole portfolio, so that a subgroup can be sliced out of it
        and still be drifted against the portfolio's capital rather than its own.

        Parameters
        ----------
        frames : tuple of pd.DataFrame
            Output of :meth:`_resolve_frames`.

        Returns
        -------
        tuple of pd.DataFrame
            One baseline per input frame, each indexed by trade date. The active
            baseline is the portfolio's less the benchmark's, differenced after the
            fact because an active book nets to roughly zero and a zero-sum vector has
            no capital base to grow.
        """
        trade_dates = self.trade_dates.intersection(frames[-1].index)
        if len(frames) == 1:
            return (_no_trade_weights(frames[0], trade_dates, self.returns),)

        carry_w = _no_trade_weights(frames[0], trade_dates, self.returns)
        carry_b = _no_trade_weights(frames[1], trade_dates, self.returns)
        return (carry_w, carry_b, carry_w - carry_b)

    def _weight_stats_block(
        self,
        frames: Tuple[pd.DataFrame, ...],
        carries: Tuple[pd.DataFrame, ...],
        columns: pd.Index,
    ) -> pd.DataFrame:
        """
        Weight statistics for one set of securities, indexed by date.

        Parameters
        ----------
        frames : tuple of pd.DataFrame
            Output of :meth:`_resolve_frames`.
        carries : tuple of pd.DataFrame
            Matching no-trade baselines from :meth:`_no_trade_baselines`.
        columns : pd.Index
            Securities to measure, i.e. the whole universe or one subgroup's columns.

        Returns
        -------
        pd.DataFrame
            Indexed by ``"real_date"``, with the columns listed in
            :data:`ACTIVE_WEIGHT_STATS` when ``frames`` carries a benchmark and
            :data:`STANDALONE_WEIGHT_STATS` otherwise.
        """
        if len(frames) == 1:
            w = frames[0][columns]
            trade_dates = self.trade_dates.intersection(w.index)
            stats = _concentration_stats(w)
            stats["turnover"] = _turnover_against(frames[0], carries[0], columns)
            stats["turnover_annualised"] = (
                stats["turnover"] * self.rebalancings_per_year
            )
            stats["weight_autocorr"] = _weight_autocorr(w, trade_dates)
            return stats[STANDALONE_WEIGHT_STATS]

        w, b, active_w = frames
        trade_dates = self.trade_dates.intersection(active_w.index)

        stats = _concentration_stats(active_w[columns]).rename(columns=_ACTIVE_RENAME)
        # Active share is the one-way gross active position, hence half.
        stats["active_share"] = 0.5 * stats.pop("gross_weight")

        # How much more the portfolio traded than the benchmark reconstituted, as
        # distinct from the turnover of the active weight vector itself.
        stats["active_turnover"] = _turnover_against(
            w, carries[0], columns
        ) - _turnover_against(b, carries[1], columns)
        stats["active_weight_turnover"] = _turnover_against(
            active_w, carries[2], columns
        )
        for stat in ("active_turnover", "active_weight_turnover"):
            stats[f"{stat}_annualised"] = stats[stat] * self.rebalancings_per_year
        stats["active_weight_autocorr"] = _weight_autocorr(
            active_w[columns], trade_dates
        )

        # Which side of the universe each name sits on, read off the two weight
        # matrices rather than the active weight: the sign of an active weight says
        # which way the bet runs, not whether the other side holds the name at all.
        # `_align_active` has zero-filled both, so a zero is "not held" - a security
        # sitting in the universe at a zero weight is not an off-benchmark position.
        held = w[columns].ne(0.0)
        in_benchmark = b[columns].ne(0.0)
        off_benchmark = held & ~in_benchmark

        stats["off_benchmark_n"] = off_benchmark.sum(axis=1).astype(float)
        stats["off_benchmark_weight"] = 100.0 * w[columns].where(off_benchmark).sum(
            axis=1
        )
        stats["benchmark_only_n"] = (in_benchmark & ~held).sum(axis=1).astype(float)

        # A date the portfolio sits out carries no weight reading, as in
        # `_concentration_stats`; the count still stands at zero. Where the portfolio
        # is held but sits entirely inside the benchmark, the zero weight is a real
        # reading and is kept.
        stats.loc[~held.any(axis=1), "off_benchmark_weight"] = np.nan
        # Measured on the whole benchmark rather than on `columns`: a subgroup the
        # benchmark does not reach is a genuine off-benchmark allocation, but a date
        # the benchmark is absent on gives nothing to be off - every holding would
        # register as off-benchmark, which is true and tells you nothing.
        stats.loc[
            ~b.ne(0.0).any(axis=1),
            ["off_benchmark_n", "off_benchmark_weight", "benchmark_only_n"],
        ] = np.nan
        return stats[ACTIVE_WEIGHT_STATS]

    def weight_stats(
        self,
        by_group: bool = False,
        active: bool = False,
        as_qdf: bool = False,
        xcat_prefix: str = "PORT",
    ) -> pd.DataFrame:
        """
        Size, concentration and turnover statistics of the portfolio's weights.

        Parameters
        ----------
        by_group : bool, default False
            If True, report one set of statistics per subgroup of ``groups`` rather
            than one for the portfolio as a whole.
        active : bool, default False
            If True, measure the active position against ``benchmark`` instead of the
            portfolio's own weights.
        as_qdf : bool, default False
            If True, return a QuantamentalDataFrame instead of the tidy frame: one
            cross-section per subgroup - or ``portfolio_name`` when ``by_group`` is
            False - and one category per statistic. Underscores in subgroup labels
            are replaced with hyphens so that the labels are valid cross-sections.
        xcat_prefix : str, default "PORT"
            Prefix of the category names when ``as_qdf`` is True, e.g. a prefix of
            "PORT" yields "PORT_N_HOLDINGS".

        Raises
        ------
        ValueError
            If ``by_group`` is True but no ``groups`` mapping was supplied, if
            ``active`` is True but no ``benchmark`` was supplied, or if ``as_qdf`` is
            True and every statistic is missing.

        Returns
        -------
        pd.DataFrame
            Tidy frame with a ``"real_date"`` column, a ``"group"`` column when
            ``by_group`` is True, and one column per statistic - the names in
            :data:`ACTIVE_WEIGHT_STATS` when ``active`` is True, :data:`STANDALONE_WEIGHT_STATS`
            otherwise. Returned as a QuantamentalDataFrame when ``as_qdf`` is True.

        Notes
        -----
        Subgroup statistics are measured on the subgroup's columns of the
        portfolio-level weight matrix, so ``n_holdings``, ``weight``,
        ``gross_weight``, ``active_weight``, ``active_share``, the off-benchmark
        split - ``off_benchmark_n``, ``off_benchmark_weight`` and
        ``benchmark_only_n`` - and the turnovers, annualised or not, are contributions
        that sum across subgroups to the whole-portfolio figure. ``effective_n`` and
        the autocorrelations are normalised within the subgroup and do not aggregate.

        The off-benchmark split reports how far the portfolio's universe departs from
        the benchmark's, which ``n_active_holdings`` and ``active_share`` fold in
        without distinguishing: ``off_benchmark_n`` counts securities held with no
        benchmark weight, ``off_benchmark_weight`` sums their portfolio weight in
        percentage points, and ``benchmark_only_n`` counts benchmark members the
        portfolio does not hold. A security sitting in the universe at a zero weight
        counts as held by neither side, so ``off_benchmark_n`` and the number of
        holdings shared with the benchmark add back to the standalone ``n_holdings``.
        On a date with no position the counts still stand at zero but
        ``off_benchmark_weight`` is NaN, as ``weight`` is; a zero there means the book
        is held and sits entirely inside the benchmark. On a date the benchmark itself
        carries no weight all three are NaN, since every holding would otherwise
        register as off-benchmark against nothing.

        The concentration columns and the off-benchmark split are daily readings. The
        turnovers and the autocorrelations are reported only on the rebalancing dates
        implied by ``rebalance_freq`` and are NaN on every other date: between
        rebalancings the book changes because positions drift with returns, not
        because anything was traded. Drop those rows with
        ``.dropna(subset=["turnover"])`` to get one row per rebalancing.

        Every turnover is paired with an ``_annualised`` column holding the same
        reading multiplied by ``rebalancings_per_year``, so that books rebalanced on
        different cadences can be set side by side. Read the class Notes before
        comparing across cadences: annualisation makes the yearly cost comparable, not
        the amount of signal churn, and the drift component of turnover grows with the
        square root of the rebalancing frequency rather than staying put.
        """
        frames = self._resolve_frames(active)
        carries = self._no_trade_baselines(frames)
        columns = frames[-1].columns

        if by_group:
            if self._group_map is None:
                raise ValueError(
                    "`groups` must be supplied to PortfolioAnalyser for subgroup "
                    "statistics."
                )
            labels = _group_labels(columns, self._group_map, self.other_label)
            blocks = []
            for label in sorted(labels.unique()):
                block = self._weight_stats_block(
                    frames, carries, labels.index[labels == label]
                )
                blocks.append(block.reset_index().assign(group=label))
            stats = pd.concat(blocks, axis=0, ignore_index=True)
            stat_cols = [c for c in stats.columns if c not in ("real_date", "group")]
            stats = (
                stats[["real_date", "group"] + stat_cols]
                .sort_values(["group", "real_date"])
                .reset_index(drop=True)
            )
        else:
            stats = (
                self._weight_stats_block(frames, carries, columns)
                .reset_index()
                .sort_values("real_date")
                .reset_index(drop=True)
            )

        if as_qdf:
            return self._stats_to_qdf(stats, xcat_prefix)
        return stats

    def _stats_to_qdf(
        self, stats: pd.DataFrame, xcat_prefix: str
    ) -> QuantamentalDataFrame:
        """
        Convert a tidy statistics frame to a QuantamentalDataFrame.

        Parameters
        ----------
        stats : pd.DataFrame
            Output of :meth:`weight_stats`.
        xcat_prefix : str
            Prefix prepended to the upper-cased statistic name to form the category.

        Raises
        ------
        ValueError
            If every statistic is missing, leaving nothing to convert.

        Returns
        -------
        QuantamentalDataFrame
            Standard panel with columns "cid", "xcat", "real_date" and "value".
        """
        id_vars = ["real_date"] + (["group"] if "group" in stats.columns else [])
        long = stats.melt(
            id_vars=id_vars, var_name="stat", value_name="value"
        ).dropna(subset=["value"])
        if long.empty:
            raise ValueError("No statistics available to convert to a panel.")

        # Named through the same helper as `weight_stat_labels`, so that the labels
        # cannot fall out of step with the categories they are meant to describe.
        long["xcat"] = long["stat"].map(
            {stat: _stat_xcat(xcat_prefix, stat) for stat in long["stat"].unique()}
        )
        if "group" in long.columns:
            long["cid"] = long["group"].map(_cid_label_map(long["group"].unique()))
        else:
            long["cid"] = _cid_label_map([self.portfolio_name])[self.portfolio_name]

        return QuantamentalDataFrame.from_long_df(
            long[["cid", "xcat", "real_date", "value"]]
        )

    def attribution(
        self,
        by_group: bool = False,
        active: bool = False,
        lag: int = 1,
        include_total: bool = True,
        as_qdf: bool = False,
        xcat: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Contribution of each security, or each subgroup, to the portfolio's return.

        Weights are lagged before being applied to same-day returns: a weight set on
        date ``t - lag`` earns the return realised on date ``t``. A security missing a
        weight or a return on a given date contributes zero rather than NaN, so that
        the contributions always sum to the portfolio return implied by the weights.

        Parameters
        ----------
        by_group : bool, default False
            If True, sum contributions within each subgroup of ``groups`` rather than
            reporting one column per security.
        active : bool, default False
            If True, attribute the active return by applying the active weights
            against ``benchmark`` rather than the portfolio's own weights.
        lag : int, default 1
            Number of dates the weights are lagged by. The default matches the one-day
            lag used throughout the package's signal pipelines.
        include_total : bool, default True
            If True, append a column holding the sum across securities or subgroups,
            named after ``portfolio_name``.
        as_qdf : bool, default False
            If True, return a QuantamentalDataFrame instead of the wide contribution
            matrix, with one cross-section per security or subgroup. Underscores in
            subgroup labels are replaced with hyphens.
        xcat : str, optional
            Category assigned when ``as_qdf`` is True. Defaults to "ACTIVE_CONTRIB"
            when ``active`` is True and "CONTRIB" otherwise.

        Raises
        ------
        ValueError
            If no ``returns`` were supplied, if ``by_group`` is True without a
            ``groups`` mapping, if ``active`` is True without a ``benchmark``, or if
            ``portfolio_name`` collides with a security or subgroup name.
        TypeError
            If ``lag`` is not a non-negative integer.

        Returns
        -------
        pd.DataFrame
            Indexed by ``"real_date"``, with one column per security - or per subgroup
            when ``by_group`` is True - plus the total column when ``include_total`` is
            True. Contributions are in the units of ``returns``. Returned as a
            QuantamentalDataFrame when ``as_qdf`` is True.
        """
        if self.returns is None:
            raise ValueError(
                "`returns` must be supplied to PortfolioAnalyser for attribution."
            )
        if not isinstance(lag, (int, np.integer)) or isinstance(lag, bool) or lag < 0:
            raise TypeError("`lag` must be a non-negative integer.")

        weights = self._resolve_frames(active)[-1]

        # Union of columns so that a security held without a return, or returning
        # without a holding, still appears - contributing zero either way.
        cids = weights.columns.union(self.returns.columns)
        lagged = weights.reindex(columns=cids).shift(lag).fillna(0.0)
        rets = self.returns.reindex(columns=cids).fillna(0.0)
        dates = lagged.index.intersection(rets.index)
        if len(dates) == 0:
            raise ValueError(
                "`weights` and `returns` share no dates; no contribution can be "
                "attributed."
            )
        contrib = lagged.loc[dates] * rets.loc[dates]

        if by_group:
            if self._group_map is None:
                raise ValueError(
                    "`groups` must be supplied to PortfolioAnalyser for subgroup "
                    "attribution."
                )
            labels = _group_labels(contrib.columns, self._group_map, self.other_label)
            contrib = contrib.T.groupby(labels).sum().T
            contrib = contrib[sorted(contrib.columns)]

        if include_total:
            if self.portfolio_name in contrib.columns:
                raise ValueError(
                    f"`portfolio_name` '{self.portfolio_name}' collides with an "
                    "existing column; choose another name or set include_total=False."
                )
            contrib[self.portfolio_name] = contrib.sum(axis=1)

        contrib.index.name = "real_date"
        contrib.columns.name = "cid"

        if as_qdf:
            if xcat is None:
                xcat = "ACTIVE_CONTRIB" if active else "CONTRIB"
            if not isinstance(xcat, str) or not xcat:
                raise TypeError("`xcat` must be a non-empty string.")
            renamed = contrib.rename(columns=_cid_label_map(contrib.columns))
            return QuantamentalDataFrame.from_long_df(
                _wide_to_long(renamed, value_name="value"), xcat=xcat
            )
        return contrib
