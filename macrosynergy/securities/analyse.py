# Portfolio weight concentration diagnostics
import os
from typing import Union, Dict, List, Tuple

import pandas as pd




# ---------------------------------------------------------------------------
# Core: turnover and active-weight alignment
# ---------------------------------------------------------------------------


def weight_turnover(weights_wide: pd.DataFrame) -> pd.Series:
    # NaN means "not held" -> 0, so entries/exits register as a full move from/to 0.
    w = weights_wide.fillna(0.0)
    turnover = 100.0 * 0.5 * w.diff().abs().sum(axis=1)
    turnover.iloc[0] = np.nan  # no prior date to compare against
    turnover[turnover == 0] = np.nan  # every security unchanged -> treat as no reading
    turnover.name = "turnover"
    return turnover


def align_active(
    weights_wide: pd.DataFrame, benchmark_wide: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Union of columns + fillna(0): a security only one side holds is a full active
    # position against a 0 weight on the other side, not a dropped/NaN entry.
    cols = sorted(set(weights_wide.columns) | set(benchmark_wide.columns))
    idx = weights_wide.index.intersection(benchmark_wide.index)
    w = weights_wide.reindex(index=idx, columns=cols).fillna(0.0)
    b = benchmark_wide.reindex(index=idx, columns=cols).fillna(0.0)
    return w, b, w - b


def _hhi_effective_n(magnitudes: pd.Series) -> Tuple[int, float]:
    n = len(magnitudes)
    if n == 0:
        return 0, np.nan
    shares = magnitudes / magnitudes.sum()
    hhi = float((shares**2).sum())
    return n, 1.0 / hhi


# ---------------------------------------------------------------------------
# Raw portfolio stats (not vs. benchmark) -- whole portfolio or one sector's columns
# ---------------------------------------------------------------------------


def portfolio_stats_raw(weights_wide: pd.DataFrame) -> pd.DataFrame:
    def _row(row: pd.Series) -> pd.Series:
        w = row.dropna()
        w = w[w != 0]
        n, effective_n = _hhi_effective_n(w.abs())
        return pd.Series(
            {
                "n_holdings": n,
                "effective_n": effective_n,
                "weight": 100.0 * w.sum() if n else np.nan,
            }
        )

    stats = weights_wide.apply(_row, axis=1)
    stats["turnover"] = weight_turnover(weights_wide)
    return stats


def group_portfolio_stats_raw(
    weights_wide: pd.DataFrame,
    group_map: Dict[str, str],
    other_label: str = "OTHER",
) -> pd.DataFrame:
    groups = pd.Series({c: group_map.get(c, other_label) for c in weights_wide.columns})
    frames = []
    for label in sorted(groups.unique()):
        sub = weights_wide[groups.index[groups == label]]
        stats = portfolio_stats_raw(sub)
        stats.index.name = "real_date"
        frames.append(stats.reset_index().assign(group=label))
    out = pd.concat(frames, axis=0, ignore_index=True)
    stat_cols = [c for c in out.columns if c not in ("real_date", "group")]
    return (
        out[["real_date", "group"] + stat_cols]
        .sort_values(["group", "real_date"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Active portfolio stats (vs. benchmark)
# ---------------------------------------------------------------------------


def autocorr_active_weight(active_wide: pd.DataFrame) -> pd.Series:
    # Cross-sectional correlation of the active-weight vector against its own prior
    # date, restricted each pair to securities relevant in at least one of the two
    # dates (a!=0 or b!=0) -- otherwise the ambient zero-filled universe dominates and
    # trivially inflates the correlation toward 1.
    vals = active_wide.to_numpy()
    idx = active_wide.index
    out = np.full(len(idx), np.nan)
    for i in range(1, len(idx)):
        cur, prev = vals[i], vals[i - 1]
        mask = (cur != 0) | (prev != 0)
        if mask.sum() > 1 and np.std(cur[mask]) > 0 and np.std(prev[mask]) > 0:
            out[i] = np.corrcoef(cur[mask], prev[mask])[0, 1]
    return pd.Series(out, index=idx, name="active_weight_autocorr")


def _active_row_stats(active: pd.DataFrame) -> pd.DataFrame:
    def _row(row: pd.Series) -> pd.Series:
        a = row[row != 0]
        n, effective_n = _hhi_effective_n(a.abs())
        return pd.Series(
            {
                "n_active_holdings": n,
                "effective_active_n": effective_n,  # inverse participation ratio
                "active_share": (
                    50.0 * a.abs().sum() if n else np.nan
                ),  # 0.5 * sum|active w|
            }
        )

    return active.apply(_row, axis=1)


def portfolio_stats_active(
    weights_wide: pd.DataFrame, benchmark_wide: pd.DataFrame
) -> pd.DataFrame:
    w, b, active = align_active(weights_wide, benchmark_wide)
    stats = _active_row_stats(active)
    stats["active_turnover"] = weight_turnover(w) - weight_turnover(b)
    stats["active_weight_turnover"] = weight_turnover(active)
    stats["active_weight_autocorr"] = autocorr_active_weight(active)
    return stats


def group_portfolio_stats_active(
    weights_wide: pd.DataFrame,
    benchmark_wide: pd.DataFrame,
    group_map: Dict[str, str],
    other_label: str = "OTHER",
) -> pd.DataFrame:
    w, b, active = align_active(weights_wide, benchmark_wide)
    groups = pd.Series({c: group_map.get(c, other_label) for c in active.columns})
    frames = []
    for label in sorted(groups.unique()):
        cols = groups.index[groups == label]
        sub_w, sub_b, sub_active = w[cols], b[cols], active[cols]
        stats = _active_row_stats(sub_active)
        stats["active_turnover"] = weight_turnover(sub_w) - weight_turnover(sub_b)
        stats["active_weight_turnover"] = weight_turnover(sub_active)
        stats["active_weight_autocorr"] = autocorr_active_weight(sub_active)
        stats.index.name = "real_date"
        frames.append(stats.reset_index().assign(group=label))
    out = pd.concat(frames, axis=0, ignore_index=True)
    stat_cols = [c for c in out.columns if c not in ("real_date", "group")]
    return (
        out[["real_date", "group"] + stat_cols]
        .sort_values(["group", "real_date"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Regression/correlation inputs: target stays a level, only the factors transform
# ---------------------------------------------------------------------------


def factor_target_frames(
    df: pd.DataFrame, factor_xcats: List[str], target_xcats: List[str], cid: str
) -> Dict[str, pd.DataFrame]:
    factors = monthly_wide(df, factor_xcats, cid)
    targets = monthly_wide(
        df, target_xcats, cid
    )  # left as computed, never diffed/squared

    frames = {
        "level": pd.concat([factors, targets], axis=1),
        "change": pd.concat([factors.diff(), targets], axis=1),
        "sq_change": pd.concat([factors.diff() ** 2, targets], axis=1),
    }
    return {name: wide_to_qdf(wide, cid) for name, wide in frames.items()}


# Single-security portfolio return attribution


def attribute_portfolio_return(
    weights_wide: pd.DataFrame, returns_wide: pd.DataFrame
) -> pd.DataFrame:
    """
    Per-security contribution to total portfolio return.

    Weights are lagged one day before being applied to same-day returns, matching the
    `lag=1` used throughout the learning pipeline: a weight set on date t-1 earns the
    return realised on date t. A stock missing a weight or a return on a given day
    contributes 0 rather than NaN.

    Parameters
    ----------
    weights_wide : pd.DataFrame
        Daily portfolio weights, as returned by `stitch_vintage_weights`.
    returns_wide : pd.DataFrame
        Daily single-security returns, index "real_date", one column per stock cid, same
        units as the `EQCTR_NSA` category the weights were trained against.

    Returns
    -------
    pd.DataFrame
        Index "real_date", one column per stock plus "TOTAL" (the sum, i.e. the realised
        portfolio return implied by the stitched weights).
    """
    cols = sorted(set(weights_wide.columns) | set(returns_wide.columns))
    w_lag = weights_wide.reindex(columns=cols).shift(1).fillna(0.0)
    r = returns_wide.reindex(columns=cols).fillna(0.0)
    idx = w_lag.index.intersection(r.index)
    contrib = w_lag.loc[idx] * r.loc[idx]
    contrib["TOTAL"] = contrib.sum(axis=1)
    return contrib


def group_attribution(
    weights_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    group_map: Dict[str, str],
    other_label: str = "OTHER",
) -> pd.DataFrame:
    contrib = attribute_portfolio_return(weights_wide, returns_wide).drop(
        columns="TOTAL"
    )
    groups = pd.Series({c: group_map.get(c, other_label) for c in contrib.columns})
    grouped = contrib.T.groupby(groups).sum().T
    grouped["TOTAL"] = grouped.sum(axis=1)  # sums back to the portfolio-level TOTAL
    return grouped


# Position statistics as a QDF, joined to the macro factor panel


def position_stats_to_qdf(
    position_stats: pd.DataFrame, cid: str, xcat_prefix: str
) -> pd.DataFrame:
    dfa = (
        position_stats.rename_axis("real_date")
        .reset_index()
        .melt(id_vars="real_date", var_name="stat", value_name="value")
    )
    dfa["xcat"] = xcat_prefix + "_" + dfa["stat"].str.upper()
    dfa["cid"] = cid
    return dfa.loc[dfa["value"].notna(), ["cid", "xcat", "real_date", "value"]]


def group_stats_to_qdf(stats: pd.DataFrame, xcat_prefix: str) -> pd.DataFrame:
    stat_cols = [c for c in stats.columns if c not in ("real_date", "group")]
    dfa = stats.melt(
        id_vars=["real_date", "group"],
        value_vars=stat_cols,
        var_name="stat",
        value_name="value",
    )
    dfa["xcat"] = xcat_prefix + "_" + dfa["stat"].str.upper()
    dfa["cid"] = dfa["group"].astype(str).str.replace("_", "-", regex=False)
    return dfa.loc[dfa["value"].notna(), ["cid", "xcat", "real_date", "value"]]


# Monthly-resampled inputs for the change / level / squared-level correlation matrices


def monthly_wide(df: pd.DataFrame, xcats: List[str], cid: str) -> pd.DataFrame:
    """
    Month-end mean level for a set of categories under a single cross-section.

    Parameters
    ----------
    df : pd.DataFrame
        Standard QDF with "cid", "xcat", "real_date", "value" columns.
    xcats : list of str
        Categories to pivot into columns, in the given order.
    cid : str
        Single cross-section the categories are stored under.

    Returns
    -------
    pd.DataFrame
        Index "real_date" (month-end), one column per xcat: the monthly mean of the
        native-frequency values.
    """
    wide = (
        df.loc[
            (df["cid"] == cid) & (df["xcat"].isin(xcats)),
            ["real_date", "xcat", "value"],
        ]
        .pivot(index="real_date", columns="xcat", values="value")
        .reindex(columns=xcats)
    )
    return wide.resample("M").mean()


def wide_to_qdf(wide: pd.DataFrame, cid: str) -> pd.DataFrame:
    """
    Standard QDF from a wide (real_date x xcat) frame, single cross-section.

    Parameters
    ----------
    wide : pd.DataFrame
        Index "real_date", one column per xcat.
    cid : str
        Cross-section to assign every row.

    Returns
    -------
    pd.DataFrame
        Standard QDF columns "cid", "xcat", "real_date", "value".
    """
    out = wide.reset_index().melt(
        id_vars="real_date", var_name="xcat", value_name="value"
    )
    out["cid"] = cid
    return out.loc[out["value"].notna(), ["cid", "xcat", "real_date", "value"]]
