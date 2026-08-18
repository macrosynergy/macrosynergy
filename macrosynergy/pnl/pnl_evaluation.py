from numbers import Number
from typing import Dict, Optional

import numpy as np
import pandas as pd

from macrosynergy.management import reduce_df
from macrosynergy.management.types import NoneType
from macrosynergy.management.utils import _map_to_business_day_frequency
from macrosynergy.pnl.sharpe_stability_ratio import sharpe_stability_ratio


def evaluate_pnl(
    df_pnl: pd.DataFrame,
    aum: Number,
    df_pnle: Optional[pd.DataFrame] = None,
    df_tcosts: Optional[pd.DataFrame] = None,
    label_dict: Optional[Dict[str, str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    benchmark_data: Optional[pd.DataFrame] = None,
    portfolio_name: str = "GLB",
) -> pd.DataFrame:
    """
    Compute summary performance statistics for the proxy PnL.

    The PnL series is converted to a percentage return on AUM and annualized
    statistics are computed assuming 252 trading days per year. The method
    requires that `proxy_pnl_calc` has already been run; `pnl_excl_costs` and
    `txn_costs_df` are required only when the corresponding flags are set.

    Parameters
    ----------
    aum : Number
        Assets under management used to scale the PnL into percentage returns.
    df_pnle : bool
        If True, include the PnL excluding transaction costs (`self.pnl_excl_costs`)
        as an additional column in the output.
    include_tcosts : bool
        If True, include total transaction costs as a row in the output. Requires
        `self.txn_costs_df` to be available.
    label_dict : dict
        Mapping from raw column names (xcat values) to display labels used in the
        output columns.
    start : str
        Start date (ISO format) used to filter the PnL prior to computing statistics.
        If not provided, no lower bound is applied.
    end : str
        End date (ISO format) used to filter the PnL prior to computing statistics.
        If not provided, no upper bound is applied.
    benchmark_data : pd.DataFrame
        QuantamentalDataFrame of benchmark series. If provided, the correlation
        between each PnL column and each benchmark ticker (cid_xcat) is added as
        a row in the output.

    Returns
    -------
    pd.DataFrame
        Summary statistics with one column per PnL series. Rows include the
        annualized return and standard deviation (in %), Sharpe and Sortino
        ratios, Sharpe stability, maximum 21-day, 6-month and peak-to-trough
        drawdowns (in %), the share of total PnL contributed by the top 5%
        of months, optional benchmark correlations, optional transaction
        costs, and the number of traded months.
    """
    # Input validation
    for arg, value, types in [
        ("aum", aum, Number),
        ("df_pnl", df_pnl, pd.DataFrame),
        ("df_pnle", df_pnl, (pd.DataFrame, NoneType)),
        ("df_tcosts", df_tcosts, (pd.DataFrame, NoneType)),
        ("label_dict", label_dict, (dict, NoneType)),
        ("start", start, (str, NoneType)),
        ("end", end, (str, NoneType)),
        ("benchmark_data", benchmark_data, (pd.DataFrame, NoneType)),
    ]:
        if not isinstance(value, types):
            raise TypeError(f"Argument {arg} must be one of: {types}")

    # Data preparation
    df = df_pnl if df_pnle is None else pd.concat((df_pnl, df_pnle), ignore_index=True)
    df = reduce_df(df, cids=[portfolio_name], start=start, end=end)

    dfw = df.pivot(index="real_date", columns="xcat", values="value")
    dfw = 100 * dfw / aum  # percentage return instead of $
    dfw = dfw.rename(columns=label_dict if label_dict is not None else {})

    # Summary statistics
    ## Annualized mean and std
    mean = dfw.mean(axis=0) * 252
    std = dfw.std(axis=0) * np.sqrt(252)

    ## Sharpes and Sortino
    sharpe = mean / std
    sortino = np.divide(
        mean,
        dfw.apply(lambda x: np.sqrt(np.sum(x[x < 0] ** 2) / len(x))) * np.sqrt(252),
    )
    sharpe_stability = [
        sharpe_stability_ratio(
            dfw[col].dropna(),
            window=252,
            benchmark_sr=0.0,
            annualization_factor=252,
        )
        for col in dfw.columns
    ]

    ## Draws
    draw_21_day = dfw.rolling(21).sum().min()
    draw_6_month = dfw.rolling(6 * 21).sum().min()
    draw_peak_to_trough = -(dfw.cumsum().cummax() - dfw.cumsum()).max()

    ## PnL share
    mfreq = _map_to_business_day_frequency("M")
    monthly_pnl = dfw.resample(mfreq).sum()
    total_pnl = monthly_pnl.sum(axis=0)
    n_top = int(max(np.ceil(len(monthly_pnl) * 0.05), 1))
    n_top_pnl = -np.sort(-monthly_pnl.values, axis=0)[:n_top].sum(0)
    pnl_share = n_top_pnl / total_pnl

    ## Number of traded months
    n_traded_months = dfw.notna().resample(mfreq).sum().ne(0).sum()

    ## Benchmark correlations
    correlations = {}
    if benchmark_data is not None and not benchmark_data.empty:
        bm_data = benchmark_data.copy()
        bm_data["ticker"] = bm_data["cid"] + "_" + bm_data["xcat"]
        bm_data_w = bm_data.pivot(index="real_date", columns="ticker", values="value")
        shared_idx = dfw.index.intersection(bm_data_w.index)
        correlations = {
            f"{bm} correl": dfw.loc[shared_idx].corrwith(
                other=bm_data_w.loc[shared_idx][bm],
                drop=True,
            )
            for bm in bm_data_w.columns
        }

    ## Transaction costs
    tcosts = {}
    if df_tcosts is not None:
        txn_costs = reduce_df(df=df_tcosts, cids=[portfolio_name])
        total_txn_costs = txn_costs["value"].sum()
        total_txn_cost = (
            [total_txn_costs, 0] if df_pnle is not None else [total_txn_costs]
        )
        tcosts["Transaction Cost"] = total_txn_cost

    # Format output
    summary_statistics = {
        "Return %": mean,
        "St. Dev. %": std,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Sharpe Stability": sharpe_stability,
        "Max 21-Day Draw %": draw_21_day,
        "Max 6-Month Draw %": draw_6_month,
        "Peak to Trough Draw %": draw_peak_to_trough,
        "Top 5% Monthly PnL Share": pnl_share,
        **correlations,
        **tcosts,
        "Traded Months": n_traded_months,
    }

    summary_statistics = pd.DataFrame(summary_statistics).T
    summary_statistics.columns = dfw.columns

    return summary_statistics
