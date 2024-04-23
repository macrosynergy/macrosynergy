"""
Module for calculating an approximate nominal PnL under consideration of transaction costs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple, Optional, Dict, Callable
from numbers import Number
import warnings
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
from macrosynergy.pnl.transaction_costs import TransactionCosts, get_fids


def _replace_strs(
    list_of_strs: List[str], old_str: str, new_str: str = ""
) -> List[str]:
    return [_.replace(old_str, new_str) for _ in list_of_strs]


def _split_returns_positions_tickers(
    tickers: List[str], spos: str, rstring: str
) -> Tuple[List[str], List[str]]:
    # Filter tickers based on the specific suffixes
    returns_tickers: List[str] = [
        ticker for ticker in tickers if ticker.endswith(rstring)
    ]
    positions_tickers: List[str] = [
        ticker for ticker in tickers if ticker.endswith(spos)
    ]

    set_returns = set(_replace_strs(returns_tickers, rstring))
    set_positions = set(_replace_strs(positions_tickers, f"_{spos}"))
    assert len(set_positions - set_returns) == 0
    returns_tickers: List[str] = [
        ticker.replace(f"_{spos}", rstring) for ticker in positions_tickers
    ]

    return returns_tickers, positions_tickers


def _check_df(df: QuantamentalDataFrame, spos: str, rstring: str) -> None:
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    returns_tickers, positions_tickers = _split_returns_positions_tickers(
        tickers=list(set(df["cid"] + "_" + df["xcat"])),
        spos=spos,
        rstring=rstring,
    )

    err_msg = "The following tickers are missing in the dataframe: \n"
    missing_tickers = []
    for ticker in returns_tickers:
        if ticker.replace(rstring, f"_{spos}") not in positions_tickers:
            missing_tickers.append(ticker)
    for ticker in positions_tickers:
        if ticker.replace(f"_{spos}", rstring) not in returns_tickers:
            missing_tickers.append(ticker)

    if missing_tickers:
        raise ValueError(err_msg + ", ".join(missing_tickers))


def _split_returns_positions_df(
    df_wide: pd.DataFrame, spos: str, rstring: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Filter tickers based on the specific suffixes
    returns_tickers, positions_tickers = _split_returns_positions_tickers(
        tickers=df_wide.columns.tolist(),
        spos=spos,
        rstring=rstring,
    )

    # Pivot the dataframes
    pivot_returns: pd.DataFrame = df_wide.loc[:, returns_tickers]
    pivot_pos: pd.DataFrame = df_wide.loc[:, positions_tickers]

    assert set(_replace_strs(pivot_returns.columns, rstring)) == set(
        _replace_strs(pivot_pos.columns, f"_{spos}")
    )

    return pivot_returns, pivot_pos


def _get_rebal_dates(df_wide: pd.DataFrame) -> List[pd.Timestamp]:
    # get the diff along long axis
    df_diff = df_wide.diff(axis=0)

    # change_index -- where there is any value change across rows
    change_index = df_diff.index[(df_diff.abs() > 0).any(axis=1)]

    # rows where the previous row was all NaN
    # but the current row has at least one non-NaN value
    prev_all_na = df_wide.shift(1).isna().all(axis=1)
    curr_any_value = df_wide.notna().any(axis=1)
    from_na_to_value_index = df_wide.index[prev_all_na & curr_any_value]

    # combine indices
    combined_index = change_index.union(from_na_to_value_index)
    rebal_dates = sorted(combined_index.tolist())
    return rebal_dates


def _warn_and_drop_nans(df_wide: pd.DataFrame) -> pd.DataFrame:
    # get rows that are all nans
    all_nan_rows = df_wide.loc[df_wide.isna().all(axis=1)]
    if not all_nan_rows.empty:
        warnings.warn(
            f"Warning: The following rows are all NaNs and have been dropped: {all_nan_rows.index}"
        )
        df_wide = df_wide.dropna(how="all")

    all_nan_cols = df_wide.loc[:, df_wide.isna().all(axis=0)]
    if not all_nan_cols.empty:
        warnings.warn(
            f"Warning: The following columns are all NaNs and have been dropped: {all_nan_cols.columns}"
        )
        df_wide = df_wide.dropna(how="all", axis=1)

    return df_wide


def _prep_dfs_for_pnl_calcs(
    df_wide: QuantamentalDataFrame,
    spos: str,
    rstring: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[pd.Timestamp]]:

    # Split the returns and positions dataframes
    pivot_returns, pivot_pos = _split_returns_positions_df(
        df_wide=df_wide, spos=spos, rstring=rstring
    )

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
    start: str = pivot_pos.first_valid_index()
    end: str = pivot_pos.last_valid_index()

    # List of rebal_dates
    rebal_dates = _get_rebal_dates(pivot_pos)

    # rename cols in pivot_pos and pivot_returns so that they match on mul.
    pivot_pos.columns = _replace_strs(pivot_pos.columns, f"_{spos}")
    pivot_returns.columns = _replace_strs(pivot_returns.columns, rstring)
    pivot_pos = pivot_pos[sorted(pivot_pos.columns)]
    pivot_returns = pivot_returns[sorted(pivot_returns.columns)]
    assert pivot_pos.index.name == pivot_returns.index.name == "real_date"
    return_df_cols = pivot_pos.columns.tolist()
    pnl_df = pd.DataFrame(index=pd.bdate_range(start, end), columns=return_df_cols)
    pnl_df.index.name = "real_date"
    return pnl_df, pivot_pos, pivot_returns, rebal_dates


def pnl_excl_costs(
    df_wide: pd.DataFrame,
    spos: str,
    rstring: str,
    pnlx_name: str = "PNLx",
) -> pd.DataFrame:

    pnl_df, pivot_pos, pivot_returns, rebal_dates = _prep_dfs_for_pnl_calcs(
        df_wide=df_wide, spos=spos, rstring=rstring
    )
    # Add last end date - as position taken on the last rebal date,
    # is held until notional_positions data is available
    _end = pd.Timestamp(pivot_pos.last_valid_index())
    rebal_dates = sorted(set(rebal_dates + [_end]))

    # loop between each rebalancing date
    for dt1, dt2 in zip(rebal_dates[:-1], rebal_dates[1:]):
        dt2x = dt2 - pd.offsets.BDay(1)
        curr_pos: pd.Series = pivot_pos.loc[dt1]
        curr_rets: pd.DataFrame = pivot_returns.loc[dt1:dt2x]
        cumprod_rets: pd.Series = (1 + curr_rets).cumprod(axis=0)
        pnl_df.loc[dt1:dt2x] = curr_pos * cumprod_rets

    # on dt2, we need to hold the position
    pnl_df.loc[rebal_dates[-1] :] = pivot_pos.loc[rebal_dates[-1]] * (
        1 + pivot_returns.loc[rebal_dates[-1] :]
    ).cumprod(axis=0)

    # append <spos>_<pnl_name> to all columns
    pnl_df.columns = [f"{col}_{spos}_{pnlx_name}" for col in pnl_df.columns]

    return pnl_df


def calculate_trading_costs(
    df_wide: pd.DataFrame,
    spos: str,
    rstring: str,
    transaction_costs: TransactionCosts,
    tc_name: str,
) -> pd.DataFrame:

    pivot_returns, pivot_pos = _split_returns_positions_df(
        df_wide=df_wide, spos=spos, rstring=rstring
    )
    rebal_dates = _get_rebal_dates(pivot_pos)
    # Add last end date - as position taken on the last rebal date,
    # is held until notional_positions data is available
    _end = pd.Timestamp(pivot_pos.last_valid_index())
    rebal_dates = sorted(set(rebal_dates + [_end]))
    tc_df = pd.DataFrame(index=pivot_pos.index, columns=pivot_pos.columns)
    tickers = pivot_pos.columns.tolist()
    for dt1, dt2 in zip(rebal_dates[:-1], rebal_dates[1:]):
        dt2x = dt2 - pd.offsets.BDay(1)
        prev_pos, next_pos = pivot_pos.loc[dt1], pivot_pos.loc[dt2]
        curr_pos = pivot_pos.loc[dt1:dt2x]

        avg_pos = curr_pos.mean(axis=0)
        delta_pos = (next_pos - prev_pos).abs()

        for ticker in tickers:
            _fid = ticker.replace(f"_{spos}", "")
            bidoffer = transaction_costs.bidoffer(
                trade_size=delta_pos[ticker],
                fid=_fid,
                real_date=dt2,
            )
            rollcost = transaction_costs.rollcost(
                trade_size=avg_pos[ticker],
                fid=_fid,
                real_date=dt1,
            )
            tc_df.loc[dt1:dt2x, ticker] = bidoffer + rollcost
            assert not any(tc_df.loc[dt1:dt2x, ticker] < 0)

    tc_df.columns = [f"{col}_{tc_name}" for col in tc_df.columns]

    # check that all non-nan values are positive
    assert not (tc_df < 0).any().any()

    return tc_df


def apply_trading_costs(
    pnlx_wide_df: pd.DataFrame,
    tc_wide_df: pd.DataFrame,
    spos: str,
    tc_name: str,
    pnl_name: str,
    pnlx_name: str,
) -> pd.DataFrame:
    pnls_list = sorted(pnlx_wide_df.columns.tolist())
    tcs_list = sorted(tc_wide_df.columns.tolist())

    assert len(pnls_list) == len(tcs_list)
    assert all(
        a.replace(f"_{spos}_{pnlx_name}", "") == b.replace(f"_{spos}_{tc_name}", "")
        for a, b in zip(pnls_list, tcs_list)
    )

    out_df = pnlx_wide_df.copy()
    for pnl_col, tc_col in zip(pnls_list, tcs_list):
        assert pnl_col.replace(f"_{spos}_{pnlx_name}", "") == tc_col.replace(
            f"_{spos}_{tc_name}", ""
        )
        out_df[pnl_col] = out_df[pnl_col] - tc_wide_df[tc_col]

    rename_pnl = lambda x: x.replace(f"_{spos}_{pnlx_name}", f"_{spos}_{pnl_name}")
    out_df = out_df.rename(columns=rename_pnl)

    return out_df


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
    tc_name: str = "TCOST",
    return_pnl_excl_costs: bool = False,
    return_costs: bool = False,
) -> Union[
    QuantamentalDataFrame,
    Tuple[QuantamentalDataFrame, pd.DataFrame],
    Tuple[QuantamentalDataFrame, pd.DataFrame, pd.DataFrame],
]:
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

    if roll_freqs is not None:
        raise NotImplementedError(
            "Functionality to support `roll_freqs` is not yet implemented."
        )

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
    _check_df(df=df, spos=spos, rstring=rstring)
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

    df_wide = qdf_to_ticker_df(df)

    pnlx_name = pnl_name + "x"

    # Calculate the PnL excluding costs
    df_outs: Dict[str, pd.DataFrame] = {}
    df_outs["pnl_excl_costs"] = pnl_excl_costs(
        df_wide=df_wide,
        spos=spos,
        rstring=rstring,
        pnlx_name=pnlx_name,
    )

    # tc_wide_df: pd.DataFrame = calculate_trading_costs(
    df_outs["tc_wide"] = calculate_trading_costs(
        df_wide=df_wide,
        spos=spos,
        rstring=rstring,
        transaction_costs=transcation_costs,
        tc_name=tc_name,
    )

    df_outs["pnl_incl_costs"] = apply_trading_costs(
        pnlx_wide_df=df_outs["pnl_excl_costs"],
        tc_wide_df=df_outs["tc_wide"],
        spos=spos,
        tc_name=tc_name,
        pnl_name=pnl_name,
        pnlx_name=pnlx_name,
    )

    # # Convert to QDFs
    for key in df_outs.keys():
        df_outs[key] = ticker_df_to_qdf(df_outs[key])

    if not (return_pnl_excl_costs or return_costs):
        return df_outs["pnl_incl_costs"]
    elif return_pnl_excl_costs and return_costs:
        return (
            df_outs["pnl_incl_costs"],
            df_outs["pnl_excl_costs"],
            df_outs["tc_wide"],
        )
    elif return_pnl_excl_costs:
        return df_outs["pnl_incl_costs"], df_outs["pnl_excl_costs"]
    elif return_costs:
        return df_outs["pnl_incl_costs"], df_outs["tc_wide"]


if __name__ == "__main__":

    cids_dmca = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK", "USD"]
    cids_dmec = ["DEM", "ESP", "FRF", "ITL"]
    cids_nofx: List[str] = ["USD", "EUR", "CNY", "SGD"]
    cids_dmfx: List[str] = list(set(cids_dmca) - set(cids_nofx))

    dfx = pd.read_pickle(r"C:\Users\PalashTyagi\Code\ms\macrosynergy\data\r.pkl")
    df_pnl, df_pnlx, df_costs = proxy_pnl_calc(
        df=dfx,
        spos="STRAT_POS",
        rstring="XR_NSA",
        fids=[f"{cc:s}_FX" for cc in cids_dmfx],
        tcost_n="BIDOFFER_MEDIAN",
        rcost_n="ROLLCOST_MEDIAN",
        size_n="SIZE_MEDIAN",
        tcost_l="BIDOFFER_90PCTL",
        rcost_l="SIZE_90PCTL",
        size_l="SIZE_90PCTL",
        pnl_name="PNL",
        tc_name="TCOST",
        return_pnl_excl_costs=True,
        return_costs=True,
    )
