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
    return [ticker.replace(old_str, new_str) for ticker in list_of_strs]


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
            nas_idx = (
                dfx[col]
                .loc[dfx[col].isna()]
                .loc[dfx[col].first_valid_index() : dfx[col].last_valid_index()]
            )
            if not nas_idx.empty:
                warnings.warn(
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



####################################################################################
# __________________________________________________________________________________
# In psuedo code:
# [1] initial period (t=0)
# pnl(0) = 0 (no position)
# price(0) = 1 (value change exactly position)
# pos(0) = pos*(0) (end-of-period position - that earns returns at the end of the next business day)
# [2] End of next period (t=1) - no rebalancing
# Daily PnL pnl(1) = pos(0) * price(0) * r(1) = pos*(0) * r(1) 
# Value change of position: price(1) = price(0) * (1+ r(1)) = (1+ r(1))
# Position: pos(1) = pos(0) = pos*(0) 
# __________________________________________________________________________________
# More generally:
# [a] Non-rebalancing date t:
# pnl(t) =  pos(t-1) * price(t-1) * r(t)
# price(t) = price(t-1) * (1+r(t))
# pos(t) = pos(t-1)
# [b] Rebalancing date t:
# pnl(t) =  pos(t-1) * price(t-1) * r(t)
# price(t) = 1
# pos(t) = pos*(t)
# where pos*(t) is the "optimal" position for a rebalance on that day.
# __________________________________________________________________________________
####################################################################################



def _pnl_excl_costs(
    df_wide: pd.DataFrame, spos: str, rstring: str, pnl_name: str
) -> pd.DataFrame:

    pnl_df, pivot_pos, pivot_returns, rebal_dates = _prep_dfs_for_pnl_calcs(
        df_wide=df_wide, spos=spos, rstring=rstring
    )
    # Add last end date - as position taken on the last rebal date,
    # is held until notional_positions data is available
    _end = pd.Timestamp(pivot_pos.last_valid_index())
    rebal_dates = sorted(set(rebal_dates + [_end]))

    # loop between each rebalancing date (month start)
    # there are returns and positions for each date in between and on the rebalancing date
    for dt1, dt2 in zip(rebal_dates[:-1], rebal_dates[1:]):
        # dt1 is the first day of current (new) position
        # dt2 is the next rebalancing date, i.e.  position changes on dt2.
        dt2x = dt2 - pd.offsets.BDay(1) # last day to hold the position
        dt1x = dt1 + pd.offsets.BDay(1) # first day the current position made returns
        
        # current position is the position held from dt1 to dt2x
        # current returns is the returns from dt1x to dt2
        curr_pos = pivot_pos.loc[dt1:dt2x]
        curr_returns = pivot_returns.loc[dt1x:dt2]
        
        # calculate prices separately for each day
        prices = (1 + curr_returns).cumprod()
        # reset the first price to 1
        prices.iloc[0] = 1

        # calculate the daily pnl
        daily_pnl = curr_pos.shift(1) * prices.shift(1) * curr_returns
        x = curr_pos.shift(1).values * prices.shift(1).values * curr_returns.values
        pnl_df.loc[dt1:dt2x] = x

    # Drop rows with no pnl
    nan_count_rows = pnl_df.isna().all(axis=1).sum()
    pnl_df = pnl_df.loc[pnl_df.abs().sum(axis=1) > 0]

    return pnl_df



def _calculate_trading_costs(
    df_wide: pd.DataFrame,
    spos: str,
    rstring: str,
    transaction_costs: TransactionCosts,
    tc_name: str,
    bidoffer_name: str = "BIDOFFER",
    rollcost_name: str = "ROLLCOST",
) -> pd.DataFrame:

    pivot_returns, pivot_pos = _split_returns_positions_df(
        df_wide=df_wide, spos=spos, rstring=rstring
    )
    rebal_dates = _get_rebal_dates(pivot_pos)
    # Add last end date - as position taken on the last rebal date,
    # is held until notional_positions data is available
    _end = pd.Timestamp(pivot_pos.last_valid_index())
    rebal_dates = sorted(set(rebal_dates + [_end]))
    pos_cols = pivot_pos.columns.tolist()
    tc_cols = [
        f"{col}_{tc_name}_{cost_type}"
        for col in pos_cols
        for cost_type in [bidoffer_name, rollcost_name]
    ]
    # Create a dataframe to store the trading costs with all 0s
    tc_df = pd.DataFrame(data=0.0, index=pivot_pos.index, columns=tc_cols)

    tickers = pivot_pos.columns.tolist()

    ## Taking the 1st position
    ## Here, only the bidoffer is considered, as there is nothing to roll
    first_pos = pivot_pos.loc[rebal_dates[0]]
    for ticker in tickers:
        _fid = ticker.replace(f"_{spos}", "")
        bidoffer = transaction_costs.bidoffer(
            trade_size=first_pos[ticker],
            fid=_fid,
            real_date=rebal_dates[0],
        )
        # Add a 0 for rollcost
        tc_df.loc[rebal_dates[0], f"{ticker}_{tc_name}_{rollcost_name}"] = 0
        tc_df.loc[rebal_dates[0], f"{ticker}_{tc_name}_{bidoffer_name}"] = (
            bidoffer / 100
        )

    for ix, (dt1, dt2) in enumerate(zip(rebal_dates[:-1], rebal_dates[1:])):
        dt2x = dt2 - pd.offsets.BDay(1)
        prev_pos, next_pos = pivot_pos.loc[dt1], pivot_pos.loc[dt2]
        curr_pos = pivot_pos.loc[dt1:dt2x]
        avg_pos: pd.Series = curr_pos.abs().mean(axis=0)
        delta_pos = (next_pos - prev_pos).abs()
        for ticker in tickers:
            _fid = ticker.replace(f"_{spos}", "")
            _rcn = f"{ticker}_{tc_name}_{rollcost_name}"
            _bon = f"{ticker}_{tc_name}_{bidoffer_name}"
            rollcost = transaction_costs.rollcost(
                trade_size=avg_pos[ticker],
                fid=_fid,
                real_date=dt2,
            )
            bidoffer = transaction_costs.bidoffer(
                trade_size=delta_pos[ticker],
                fid=_fid,
                real_date=dt2,
            )
            # delta_pos and avg_pos are already in absolute terms
            tc_df.loc[dt2, _rcn] = avg_pos[ticker] * rollcost / 100
            tc_df.loc[dt2, _bon] = delta_pos[ticker] * bidoffer / 100

    # Sum TICKER_TCOST_BIDOFFER and TICKER_TCOST_ROLLCOST into TICKER_TCOST
    for ticker in tickers:
        tc_df[f"{ticker}_{tc_name}"] = tc_df[
            [
                f"{ticker}_{tc_name}_{bidoffer_name}",
                f"{ticker}_{tc_name}_{rollcost_name}",
            ]
        ].sum(axis=1)

    # Drop rows with no trading costs
    tc_df = tc_df.loc[tc_df.abs().sum(axis=1) > 0]

    # check that remaining dates are part of rebal_dates
    assert set(tc_df.index) <= set(rebal_dates)
    assert not (tc_df < 0).any().any()

    return tc_df


def _apply_trading_costs(
    pnlx_wide_df: pd.DataFrame,
    tc_wide_df: pd.DataFrame,
    spos: str,
    tc_name: str,
    pnl_name: str,
    pnlx_name: str,
    bidoffer_name: str = "BIDOFFER",
    rollcost_name: str = "ROLLCOST",
) -> pd.DataFrame:
    pnls_list = sorted(pnlx_wide_df.columns.tolist())
    tcs_list = sorted(tc_wide_df.columns.tolist())
    # remove all that ends with tc_name_bidoffer or tc_name_rollcost
    tcs_list = sorted(
        set(
            [
                tc
                for tc in tcs_list
                if not any(
                    [
                        tc.endswith(f"_{tc_name}_{cost_type}")
                        for cost_type in [bidoffer_name, rollcost_name]
                    ]
                )
            ]
        )
    )

    assert len(pnls_list) == len(tcs_list)
    assert set(_replace_strs(pnls_list, f"_{spos}_{pnl_name}")) == set(
        _replace_strs(tcs_list, f"_{spos}_{tc_name}")
    )

    out_df = pnlx_wide_df.copy()
    for pnl_col, tc_col in zip(pnls_list, tcs_list):
        assert pnl_col.replace(f"_{spos}_{pnl_name}", "") == tc_col.replace(
            f"_{spos}_{tc_name}", ""
        )

        out_df[pnl_col] = out_df[pnl_col].sub(tc_wide_df[tc_col], fill_value=0)

    rename_pnl = lambda x: str(x).replace(f"_{spos}_{pnl_name}", f"_{spos}_{pnlx_name}")
    out_df = out_df.rename(columns=rename_pnl)

    return out_df


def _portfolio_sums(
    df_outs: Dict[str, pd.DataFrame],
    spos: str,
    portfolio_name: str,
    pnl_name: str,
    tc_name: str,
    pnlx_name: str,
    bidoffer_name: str,
    rollcost_name: str,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate the sum of the PnLs and costs across all contracts in the portfolio
    """
    glb_pnl_incl_costs = df_outs["pnl_incl_costs"].sum(axis=1, skipna=True)
    glb_pnl_excl_costs = df_outs["pnl_excl_costs"].sum(axis=1, skipna=True)

    # Remove all that ends with tc_name_bidoffer or tc_name_rollcost
    tcs_list = sorted(
        set(
            [
                tc
                for tc in df_outs["tc_wide"].columns.tolist()
                if not any(
                    [
                        tc.endswith(f"_{tc_name}_{cost_type}")
                        for cost_type in [bidoffer_name, rollcost_name]
                    ]
                )
            ]
        )
    )

    # Sum the trading costs
    glb_tcosts = df_outs["tc_wide"].loc[:, tcs_list].sum(axis=1, skipna=True)

    df_outs["pnl_incl_costs"].loc[
        :, f"{portfolio_name}_{spos}_{pnl_name}"
    ] = glb_pnl_incl_costs

    df_outs["pnl_excl_costs"].loc[
        :, f"{portfolio_name}_{spos}_{pnlx_name}"
    ] = glb_pnl_excl_costs

    df_outs["tc_wide"].loc[:, f"{portfolio_name}_{spos}_{tc_name}"] = glb_tcosts

    return df_outs


def proxy_pnl_calc(
    df: QuantamentalDataFrame,
    spos: str,
    rstring: str,
    transaction_costs_object: TransactionCosts,
    roll_freqs: Optional[dict] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    portfolio_name: str = "GLB",
    pnl_name: str = "PNL",
    tc_name: str = "TCOST",
    bidoffer_name: str = "BIDOFFER",
    rollcost_name: str = "ROLLCOST",
    return_pnl_excl_costs: bool = False,
    return_costs: bool = False,
) -> Union[QuantamentalDataFrame, Tuple[QuantamentalDataFrame, ...]]:
    """
    Calculates an approximate nominal PnL under consideration of transaction costs

    :param <QuantamentalDataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain the notional positions and related notional return
        series (for PnL calculations).
    :param <str> spos: the name of the strategy positions in the dataframe in
        the format "<sname>_<pname>".
        This must correspond to contract positions in the dataframe, which are categories
        of the format "<cid>_<ctype>_<sname>_<pname>". The strategy name <sname> has
        usually been set by the `contract_signals` function and the string for <pname> by
        the `notional_positions` function.
    :param <list[str]> fids: list of contract identifiers in the format
        "<cid>_<ctype>". It must correspond to contract signals in the dataframe in the
        format "<cid>_<ctype>_<sname>_<pname>".
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
        (transaction_costs_object, "transaction_costs", TransactionCosts),
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

    df_wide = qdf_to_ticker_df(df)

    pnlx_name = pnl_name + "x"

    # Calculate the PnL excluding costs
    df_outs: Dict[str, pd.DataFrame] = {}
    df_outs["pnl_excl_costs"] = _pnl_excl_costs(
        df_wide=df_wide,
        spos=spos,
        rstring=rstring,
        pnl_name=pnl_name,
    )

    # tc_wide_df: pd.DataFrame = calculate_trading_costs(
    df_outs["tc_wide"] = _calculate_trading_costs(
        df_wide=df_wide,
        spos=spos,
        rstring=rstring,
        transaction_costs=transaction_costs_object,
        tc_name=tc_name,
    )

    df_outs["pnl_incl_costs"] = _apply_trading_costs(
        pnlx_wide_df=df_outs["pnl_excl_costs"],
        tc_wide_df=df_outs["tc_wide"],
        spos=spos,
        tc_name=tc_name,
        pnlx_name=pnlx_name,
        pnl_name=pnl_name,
    )

    df_outs = _portfolio_sums(
        df_outs=df_outs,
        spos=spos,
        portfolio_name=portfolio_name,
        pnl_name=pnl_name,
        tc_name=tc_name,
        pnlx_name=pnlx_name,
        bidoffer_name=bidoffer_name,
        rollcost_name=rollcost_name,
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
    import macrosynergy.management as msm
    import os, pickle

    cids_dmca = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK", "USD"]
    cids_dmec = ["DEM", "ESP", "FRF", "ITL"]
    cids_nofx: List[str] = ["USD", "EUR", "CNY", "SGD"]
    cids_dmfx: List[str] = list(set(cids_dmca) - set(cids_nofx))

    if not os.path.exists("data/txn.obj.pkl"):
        with open("data/txn.obj.pkl", "wb") as f:
            pickle.dump(TransactionCosts.download(), f)

    with open("data/txn.obj.pkl", "rb") as f:
        tx = pickle.load(f)

    dfx = pd.read_pickle("data/dfx.pkl")

    df_pnlx, df_pnl, df_costs = proxy_pnl_calc(
        df=dfx,
        spos="STRAT_POS",
        rstring="XR_NSA",
        start='2001-01-01',
        end='2020-01-01',
        transaction_costs_object=tx,
        portfolio_name="GLB",
        pnl_name="PNL",
        tc_name="TCOST",
        return_pnl_excl_costs=True,
        return_costs=True,
    )
    df_all = pd.concat(
        [
            df_pnlx,
            df_pnl,
        ],
        axis=0,
    )

    import macrosynergy.visuals as msv, numpy as np, PIL.Image as Image, matplotlib.pyplot as plt

    df_wide = qdf_to_ticker_df(df_all)
    df_wide = df_wide.loc[:, df_wide.columns.str.startswith("GLB_")]
    assert len(df_wide.columns) == 2

    pnlx_s = [col for col in df_wide.columns if str(col).endswith("PNLx")]
    assert len(pnlx_s) == 1
    pnlx = pnlx_s[0]

    # subtract pnlx from pnl to get the costs
    df_wide[pnlx + "_COST"] = df_wide[pnlx] - df_wide[pnlx.replace("PNLx", "PNL")]

    df_all = ticker_df_to_qdf(df_wide)

    msv.FacetPlot(df=df_all).lineplot(xcat_grid=True)
