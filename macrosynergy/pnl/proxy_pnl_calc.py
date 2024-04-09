"""
Module for calculating an approximate nominal PnL under consideration of transaction costs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple, Optional, Dict, Callable
from numbers import Number
import functools

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import (
    reduce_df,
    get_cid,
    get_xcat,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
)
from macrosynergy.management.types import QuantamentalDataFrame


def extrapolate_cost(
    trade_size: Number,
    median_size: Number,
    median_cost: Number,
    pct90_size: Number,
    pct90_cost: Number,
) -> Number:
    err_msg = "{k} must be a number > 0"
    for k, v in [
        ("trade_size", trade_size),
        ("median_size", median_size),
        ("median_cost", median_cost),
        ("pct90_size", pct90_size),
        ("pct90_cost", pct90_cost),
    ]:
        if not isinstance(v, Number):
            raise TypeError(err_msg.format(k=k))
        if v <= 0:
            raise ValueError(err_msg.format(k=k))

    if trade_size <= median_size:
        cost = median_cost
    else:
        b = (pct90_cost - median_cost) / (pct90_size - median_size)
        cost = median_cost + b * (trade_size - median_size)
    return cost


class PartialCalc:
    def __init__(
        self,
        df_wide: pd.DataFrame,
        target_arg: str,
        func: Callable,
        params: Dict[str, str],
        diff_index: pd.Index = None,
    ):
        self.func = func
        reqd_args = self.func.__code__.co_varnames[: self.func.__code__.co_argcount]
        if not set(reqd_args).issubset(set(params.keys())):
            raise ValueError(
                f"Function {self.func.__name__} is missing required arguments: "
                + str(set(reqd_args) - set(params.keys()))
            )
        self.params = params
        self.diff_index = diff_index
        if not set(list(params.values()) + [target_arg]).issubset(set(df_wide.columns)):
            raise ValueError(
                f"Columns missing in df_wide: "
                + str(set(list(params.values()) + [target_arg]) - set(df_wide.columns))
            )

        self.partials, self.params_index = self._create_partials(df_wide)

    def _create_partials(self, df_wide: pd.DataFrame) -> Dict[str, Callable]:
        partials = {
            index: functools.partial(
                self.func, **{k: row[v] for k, v in self.params.items()}
            )
            for index, row in df_wide.iterrows()
        }
        params_index = {
            index: {k: row[v] for k, v in df_wide.columns}
            for index, row in df_wide.iterrows()
        }
        return partials, params_index

    def calc(
        self,
        date: pd.Timestamp,
        target_arg_val: Number,
    ):
        if self.diff_index is not None:
            if date not in self.diff_index:
                return None
        partial = self.partials[date]
        params = self.params_index[date]
        params[self.params["target_arg"]] = target_arg_val
        return partial(**params)


def check_df_for_txn_stats(
    df: QuantamentalDataFrame,
    fids: List[str],
    tcost_n: str,
    rcost_n: str,
    size_n: str,
    tcost_l: str,
    rcost_l: str,
    size_l: str,
) -> None:
    u_cids = list(set(map(get_cid, fids)))
    expected_tickers = [
        f"{cid}_{txn_ticker}"
        for cid in u_cids
        for txn_ticker in [tcost_n, rcost_n, size_n, tcost_l, rcost_l, size_l]
    ]
    found_tickers = list(set(df["cid"] + "_" + df["xcat"]))
    if not set(expected_tickers).issubset(set(found_tickers)):
        raise ValueError(
            "The dataframe is missing the following tickers: "
            + ", ".join(set(expected_tickers) - set(found_tickers))
        )

    last_avails = None
    for ticker in expected_tickers:
        avail_dates = (
            df.loc[df["cid"] + "_" + df["xcat"] == ticker]
            .drop(columns=["cid", "xcat"])
            .set_index("real_date")
            .dropna()
            .index
        )
        if last_avails is None:
            last_avails = avail_dates
        else:
            if not last_avails.equals(avail_dates):
                raise ValueError(
                    f"The available dates for {ticker} do not match the available dates for the other tickers"
                )


def get_diff_index(df_wide: pd.DataFrame, freq: str) -> pd.Index:
    df_diff = df_wide.diff(axis=0)
    change_index = df_diff.index[((df_diff.abs() > 0) | df_diff.isnull()).any(axis=1)]
    return change_index


def proxy_pnl_calc(
    df: pd.DataFrame,
    spos: str,
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
):
    """
    Calculates an approximate nominal PnL under consideration of transaction costs

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
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

    # Reduce the dataframe
    _xcats = [tcost_n, rcost_n, size_n, tcost_l, rcost_l, size_l]
    df = reduce_df(df=df, xcats=_xcats, start=start, end=end, blacklist=blacklist)

    # Check the dataframe for the necessary tickers
    check_df_for_txn_stats(df, fids, tcost_n, rcost_n, size_n, tcost_l, rcost_l, size_l)

    # Create the wide dataframe
    df_wide = qdf_to_ticker_df(df=df)
