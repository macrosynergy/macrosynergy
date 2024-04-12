import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple, Optional, Dict, Callable
from numbers import Number
import functools
from macrosynergy.management.simulate import make_test_df
from macrosynergy.download.transaction_costs import download_transaction_costs, get_fids
from macrosynergy.management.utils import (
    reduce_df,
    get_cid,
    get_xcat,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
)
from macrosynergy.management.types import QuantamentalDataFrame


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


def get_diff_index(df_wide: pd.DataFrame, freq: str = "D") -> pd.Index:
    df_diff = df_wide.diff(axis=0)
    change_index = df_diff.index[((df_diff.abs() > 0) | df_diff.isnull()).any(axis=1)]
    return change_index


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


class SparseCosts(object):
    """
    Interface to query transaction statistics dataframe.
    """

    def __init__(
        self,
        df: QuantamentalDataFrame,
    ) -> None:
        if not isinstance(df, QuantamentalDataFrame):
            raise TypeError("df must be a QuantamentalDataFrame")
        df_wide = qdf_to_ticker_df(df)
        change_index = get_diff_index(df_wide)  # drop rows with no change
        df_wide = df_wide.loc[change_index]
        self.df_wide = df_wide

    def get_costs(self, fid: str, real_date: str) -> pd.Series:
        last_valid_index = self.df_wide.loc[fid].loc[:real_date].last_valid_index()
        return self.df_wide.loc[fid].loc[last_valid_index]


class TransactionCosts(object):
    """
    Interface to query transaction statistics dataframe.
    """

    DEFAULT_ARGS = dict(
        tcost_n="BIDOFFER_MEDIAN",
        rcost_n="ROLLCOST_MEDIAN",
        size_n="SIZE_MEDIAN",
        tcost_l="BIDOFFER_90PCTL",
        rcost_l="ROLLCOST_90PCTL",
        size_l="SIZE_90PCTL",
    )

    @classmethod
    def download(self) -> "TransactionCosts":
        df = download_transaction_costs()
        return self.__init__(df=df, fids=get_fids(df), **self.DEFAULT_ARGS)

    def __init__(
        self,
        df: QuantamentalDataFrame,
        fids: List[str],
        tcost_n: str,
        rcost_n: str,
        size_n: str,
        tcost_l: str,
        rcost_l: str,
        size_l: str,
    ) -> None:
        self.sparse_costs = SparseCosts(
            df, fids, tcost_n, rcost_n, size_n, tcost_l, rcost_l, size_l
        )
        self.tcost_n = tcost_n
        self.rcost_n = rcost_n
        self.size_n = size_n
        self.tcost_l = tcost_l
        self.rcost_l = rcost_l
        self.size_l = size_l

    @staticmethod
    def extrapolate_cost(
        trade_size: Number,
        median_size: Number,
        median_cost: Number,
        pct90_size: Number,
        pct90_cost: Number,
    ) -> Number:
        return extrapolate_cost(
            trade_size=trade_size,
            median_size=median_size,
            median_cost=median_cost,
            pct90_size=pct90_size,
            pct90_cost=pct90_cost,
        )

    def bidoffer(self, fid: str, trade_size: Number, real_date: str) -> Number:
        row = self.sparse_costs.get_costs(fid=fid, real_date=real_date)
        d = dict(
            trade_size=trade_size,
            median_size=row[self.size_n],
            median_cost=row[self.tcost_n],
            pct90_size=row[self.size_l],
            pct90_cost=row[self.tcost_l],
        )
        return self.extrapolate_cost(**d)

    def rollcost(self, fid: str, trade_size: Number, real_date: str) -> Number:
        row = self.sparse_costs.get_costs(fid=fid, real_date=real_date)
        d = dict(
            trade_size=trade_size,
            median_size=row[self.size_n],
            median_cost=row[self.rcost_n],
            pct90_size=row[self.size_l],
            pct90_cost=row[self.rcost_l],
        )
        return self.extrapolate_cost(**d)


class ExampleAdapter(TransactionCosts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def extrapolate_cost(
        trade_size: Number,
        median_size: Number,
        median_cost: Number,
        pct90_size: Number,
        pct90_cost: Number,
    ) -> Number:
        # just as an example
        u = trade_size * median_cost * median_cost / (median_size * median_size)
        v = trade_size * pct90_cost * pct90_cost / (pct90_size * pct90_size)
        return (u + v) / 2

    def bidoffer(self, fid: str, trade_size: Number, real_date: str) -> Number:
        return super().bidoffer(fid, trade_size, real_date)

    def somecalc(
        self, fid: str, trade_size: Number, real_date: str, factor=1
    ) -> Number:
        # some random computation
        row = self.sparse_costs.get_costs(fid=fid, real_date=real_date)
        d = dict(
            trade_size=trade_size,
            median_size=row[self.size_n],
            median_cost=row[self.rcost_n],
            pct90_size=row[self.size_l],
            pct90_cost=row[self.rcost_l],
        )
        d["roll_cost"] = d["roll_cost"] * factor
        return self.extrapolate_cost(**d)


if __name__ == "__main__":
    import time, random

    txn_costs_df: pd.DataFrame = download_transaction_costs(verbose=True)

    txn_costs_obj: TransactionCosts = TransactionCosts.download()

    assert txn_costs_obj.sparse_costs.get_costs(
        fid="GBP_FX", real_date="2011-01-01"
    ).to_dict() == {
        "GBP_FXBIDOFFER_MEDIAN": 0.0224707153696722,
        "GBP_FXROLLCOST_MEDIAN": 0.0022470715369672,
        "GBP_FXSIZE_MEDIAN": 50.0,
        "GBP_FXBIDOFFER_90PCTL": 0.0449414307393445,
        "GBP_FXROLLCOST_90PCTL": 0.0052431669195902,
        "GBP_FXSIZE_90PCTL": 200.0,
    }

    tx_costs_dates = pd.bdate_range(
        txn_costs_df["real_date"].min(), txn_costs_df["real_date"].max()
    )

    start = time.time()
    test_iters = 1000
    for i in range(test_iters):
        txn_costs_obj.bidoffer(
            fid="GBP_FX",
            trade_size=random.randint(1, 100),
            real_date=random.choice(tx_costs_dates).strftime("%Y-%m-%d"),
        )
    end = time.time()
    print(f"Time taken: {end - start}")
    print(f"Time per iteration: {(end - start) / test_iters}")
