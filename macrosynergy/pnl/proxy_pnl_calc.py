"""
Module for calculating an approximate nominal PnL under consideration of transaction costs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple, Optional, Dict, Callable
from numbers import Number

from macrosynergy.management.simulate import make_test_df
from macrosynergy.download.transaction_costs import download_transaction_costs
from macrosynergy.management.utils import (
    reduce_df,
    get_cid,
    get_xcat,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
)
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.pnl.transaction_costs import TransactionCosts


def get_diff_index(df_wide: pd.DataFrame) -> pd.Index:
    df_diff = df_wide.diff(axis=0)
    change_index = df_diff.index[(df_diff.abs() > 0).any(axis=1)]
    return change_index


def _replace_strs(
    list_of_strs: List[str], old_str: str, new_str: str = ""
) -> List[str]:
    return [_.replace(old_str, new_str) for _ in list_of_strs]


def _prep_dfs_for_pnl_calcs(
    df: QuantamentalDataFrame,
    spos: str,
    rstring: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[pd.Timestamp]]:
    df_wide = qdf_to_ticker_df(df=df.copy())

    # Filter
    u_tickers: List[str] = list(set(df["cid"] + "_" + df["xcat"]))
    filt_xrs: List[str] = [tx for tx in u_tickers if tx.endswith(rstring)]
    # do xrs contain `spos` at this point?
    filt_pos = [tx for tx in u_tickers if tx.endswith(spos)]

    assert set(_replace_strs(filt_xrs, rstring)) == set(
        _replace_strs(filt_pos, f"_{spos}")
    )

    # Pivot the dataframes
    pivot_returns: pd.DataFrame = df_wide.loc[:, filt_xrs]
    pivot_pos: pd.DataFrame = df_wide.loc[:, filt_pos]

    # warn about NAs
    dfx: pd.DataFrame
    for dfx, dfname in [(pivot_returns, "returns"), (pivot_pos, "positions")]:
        # for each column warns for dates of nas
        for col in dfx.columns:
            nas_idx = dfx[col].loc[dfx[col].isna()]
            if not nas_idx.empty:
                print(
                    f"Warning: Series {col} has NAs at the following dates: {nas_idx.index}"
                )

    # Get the diff index for positions
    pos_diff_index: pd.DatetimeIndex = get_diff_index(df_wide=pivot_pos)
    start: str = pivot_pos.first_valid_index()
    end: str = pivot_pos.last_valid_index()

    # List of rebal_dates - with end date as the last date of the PNL calc

    rebal_dates = sorted(set(pos_diff_index) | {pd.Timestamp(end)})

    # rename cols in pivot_pos and pivot_returns so that they match on mul.
    pivot_pos.columns = _replace_strs(pivot_pos.columns, f"_{spos}")
    pivot_returns.columns = _replace_strs(pivot_returns.columns, rstring)
    pivot_pos = pivot_pos[sorted(pivot_pos.columns)]
    pivot_returns = pivot_returns[sorted(pivot_returns.columns)]

    return_df_cols = pivot_pos.columns.tolist()
    pnl_df = pd.DataFrame(index=pd.bdate_range(start, end), columns=return_df_cols)

    return pnl_df, pivot_pos, pivot_returns, rebal_dates


def pnl_excl_costs(
    df: QuantamentalDataFrame,
    spos: str,
    rstring: str,
    pnl_name: str = "PNL",
) -> pd.DataFrame:

    pnl_df, pivot_pos, pivot_returns, rebal_dates = _prep_dfs_for_pnl_calcs(
        df=df, spos=spos, rstring=rstring
    )

    # between each rebalancing date
    for dt1, dt2 in zip(rebal_dates[:-1], rebal_dates[1:]):
        curr_pos: pd.Series = pivot_pos.loc[dt1]
        curr_rets: pd.DataFrame = pivot_returns.loc[dt1:dt2]
        cumprod_rets: pd.Series = (1 + curr_rets).cumprod()
        pnl_df.loc[dt1:dt2] = curr_pos * cumprod_rets

    # sum cols, ignore nans
    pnl_df[spos + pnl_name] = pnl_df.sum(axis=1, skipna=True)

    return pnl_df


def pnl_incl_costs(
    df: QuantamentalDataFrame,
    spos: str,
    rstring: str,
    fids: List[str],
    tcost_n: str,
    rcost_n: str,
    size_n: str,
    tcost_l: str,
    rcost_l: str,
    size_l: str,
    roll_freqs: Optional[dict] = None,
    pnl_name: str = "PNL",
) -> pd.DataFrame:

    pnl_df, pivot_pos, pivot_returns, rebal_dates = _prep_dfs_for_pnl_calcs(
        df=df, spos=spos, rstring=rstring
    )

    # Initialize the TransactionCosts class
    transcation_costs: TransactionCosts = TransactionCosts(
        df=df,
        fids=fids,
        tcost_n=tcost_n,
        rcost_n=rcost_n,
        size_n=size_n,
        tcost_l=tcost_l,
        rcost_l=rcost_l,
        size_l=size_l,
    )

    for dt1, dt2 in zip(rebal_dates[:-1], rebal_dates[1:]):
        curr_pos: pd.Series = pivot_pos.loc[dt1]
        next_pos: pd.Series = pivot_pos.loc[dt2]
        curr_rets: pd.DataFrame = pivot_returns.loc[dt1:dt2]
        cumprod_rets: pd.Series = (1 + curr_rets).cumprod()
        pnl_df.loc[dt1:dt2] = curr_pos * cumprod_rets
        avg_pos_size = curr_pos.mean()
        delta_pos = sum(abs(next_pos.values - curr_pos.values))
        ...


def proxy_pnl_calc(
    df: QuantamentalDataFrame,
    spos: str,
    rstring: str,
    fids: List[str],
    tcost_n: str,
    rcost_n: str,
    size_n: str,
    tcost_l: str,
    rcost_l: str,
    size_l: str,
    roll_freqs: Optional[dict] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    pnl_name: str = "PNL",
):
    """
    Calculates an approximate nominal PnL under consideration of transaction costs

    :param <QuantamentalDataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain the contract-specific signals and possibly
        related return series (for vol-targeting).
    :param <str> spos: the name of the strategy positions in the dataframe in
        the format "<sname>_<pname>".
        This must correspond to contract positions in the dataframe, which are categories
        of the format "<cid>_<ctype>_<sname>_<pname>". The strategy name <sname> has
        usually been set by the `contract_signals` function and the string for <pname> by
        the `notional_positions` function.
    :param <list[str]> fids: list of contract identifiers in the format
        "<cid>_<ctype>". It must correspond to contract signals in the dataframe in the
        format "<cid>_<ctype>_<sname>_<pname>".
    :param <str> tcost_n: the postfix of the trading cost category for normal size. Values
        are defined as the full bid-offer spread for a normal position size.
        This must correspond to trading a cost category "<cid>_<ctype>_<tcost_n>"
        in the dataframe.
        Default is None: no trading costs are considered.
    :param <str> rcost_n: the postfix of the roll cost category for normal size. Values
        are defined as the roll charges for a normal position size.
        This must correspond to a roll cost category "<cid>_<ctype>_<rcost_n>"
        in the dataframe.
        Default is None: no trading costs are considered.
    :param <str> size_n: Normal size in USD million This must correspond to a normal
        trade size category "<cid>_<ctype>_<size_n>" in the dataframe.
        Default is None: all costs are are applied independent of size.
    :param <str> tcost_l: the postfix of the trading cost category for large size.
        Large here is defined as 90% percentile threshold of trades in the market.
        Default is None: trading costs are are applied independent of size.
    :param <str> rcost_l: the postfix of the roll cost category for large size. Values
        are defined as the roll charges for a large position size.
        This must correspond to a roll cost category "<cid>_<ctype>_<rcost_l>"
        in the dataframe.
        Default is None: no trading costs are considered.
    :param <str> size_l: Large size in USD million. Default is None: all costs are
        are applied independent of size.
    :param <dict> roll_freqs: dictionary of roll frequencies for each contract type.
        This must use the contract types as keys and frequency string ("w", "m", or "q")
        as values. The default frequency for all contracts not in the dictionary is
        "m" for monthly. Default is None: all contracts are rolled monthly.
    :param <str> start: the start date of the data. Default is None, which means that
        the start date is taken from the dataframe.
    :param <str> end: the end date of the data. Default is None, which means that
        the end date is taken from the dataframe.
    :param <dict> blacklist: a dictionary of contract identifiers to exclude from
        the calculation. Default is None, which means that no contracts are excluded.

    return: <pd.DataFrame> of the standard JPMaQS format with "GLB" (Global) as cross
        section and three categories "<sname>_<pname>_PNL" (total PnL in USD terms under
        consideration of transaction costs), "<sname>_<pname>_COST" (all imputed trading
        and roll costs in USD terms), and "<sname>_<pname>_PNLX" (total PnL in USD terms
        without consideration of transaction costs).

    N.B.: Transaction costs as % of notional are considered to be a linear function of
        size, with the slope determined by the normal and large positions, if all relevant
        series are applied.
    """

    for _varx, _namex, _typex in [
        (df, "df", QuantamentalDataFrame),
        (spos, "spos", str),
        (fids, "fids", list),
        (tcost_n, "tcost_n", str),
        (rcost_n, "rcost_n", str),
        (size_n, "size_n", str),
        (tcost_l, "tcost_l", str),
        (rcost_l, "rcost_l", str),
        (size_l, "size_l", str),
        (roll_freqs, "roll_freqs", (dict, type(None))),
        (start, "start", (str, type(None))),
        (end, "end", (str, type(None))),
        (blacklist, "blacklist", (dict, type(None))),
    ]:
        if not isinstance(_varx, _typex):
            raise TypeError(f"{_namex} must be {_typex}")

        if _typex in [list, str, dict] and len(_varx) == 0:
            raise ValueError(f"`{_namex}` must not be an empty {str(_typex)}")

    if start is None:
        start = df["real_date"].min().strftime("%Y-%m-%d")
    if end is None:
        end = df["real_date"].max().strftime("%Y-%m-%d")

    # Reduce the dataframe - keep only the txn costs, and the spos xcats
    df = reduce_df(
        df=df,
        start=start,
        end=end,
        blacklist=blacklist,
    )

    # Initialize the TransactionCosts class
    transcation_costs: TransactionCosts = TransactionCosts(
        df=df,
        fids=fids,
        tcost_n=tcost_n,
        rcost_n=rcost_n,
        size_n=size_n,
        tcost_l=tcost_l,
        rcost_l=rcost_l,
        size_l=size_l,
    )

    # Create the wide dataframe
    df_wide = qdf_to_ticker_df(df=df)


if __name__ == "__main__":
    ...
    dfxcn = pd.read_pickle("data/dfxcn.pkl")
    pnl_excl_costs(
        df=dfxcn,
        spos="STRAT_POS",
        rstring="XR_NSA",
        # start=min(dfxcn.real_date),
        # end=max(dfxcn.real_date),
    )
