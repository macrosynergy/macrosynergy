"""Test historical volatility estimates with simulate returns from random normal distribution"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional
from numbers import Number
from unittest import mock
import warnings

from macrosynergy.download.transaction_costs import (
    AVAIALBLE_COSTS,
    AVAILABLE_STATS,
    AVAILABLE_CTYPES,
    AVAILABLE_CATS,
)
from macrosynergy.pnl.transaction_costs import (
    get_fids,
    check_df_for_txn_stats,
    get_diff_index,
    extrapolate_cost,
    SparseCosts,
    TransactionCosts,
)
from macrosynergy.management.utils import (
    qdf_to_ticker_df,
    get_sops,
    ticker_df_to_qdf,
    _map_to_business_day_frequency,
)
from macrosynergy.management.types import QuantamentalDataFrame, NoneType
from macrosynergy.management.simulate import make_test_df, simulate_returns_and_signals

KNOWN_FID_ENDINGS = [f"{t}_{s}" for t in AVAIALBLE_COSTS for s in AVAILABLE_STATS]


def make_tx_cost_df(
    cids: List[str] = None,
    tickers: List[str] = None,
    start="2015-01-01",
    end="2020-01-01",
) -> pd.DataFrame:
    err = "Either cids or tickers must be provided (not both)"
    assert bool(cids) or bool(tickers), err
    assert bool(cids) ^ bool(tickers), err

    if cids is None:
        tiks = tickers
    else:
        tiks = [f"{c}_{k}" for c in cids for k in AVAILABLE_CATS]
        fids = [f"{c}_{ct}" for c in cids for ct in AVAILABLE_CTYPES]

    date_range = pd.bdate_range(start=start, end=end)
    total_dates = len(date_range)
    date_batches = [date_range[i : i + 30] for i in range(0, len(date_range), 30)]

    dataframes = [
        pd.DataFrame(
            data=float(np.random.rand()),
            index=date_batch,
            columns=["value"],
        )
        .reset_index()
        .rename(columns={"index": "real_date"})
        for date_batch in date_batches
    ]

    outs = []
    for tik in tiks:
        cid, xcat = tik.split("_", 1)
        out_df = (
            pd.concat(dataframes, axis=0)
            .reset_index(drop=True)
            .assign(cid=cid, xcat=xcat)
        )
        out_df["value"] = out_df["value"] * np.random.randint(1, 100)
        outs.append(out_df)

    return QuantamentalDataFrame.from_qdf_list(outs, categorical=False)


class TestFunctions(unittest.TestCase):
    def setUp(self):
        cids = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD"]
        tiks = [f"{c}_{k}" for c in cids for k in AVAILABLE_CATS]
        self.cids = cids
        self.fids = [f"{c}_{ct}" for c in cids for ct in AVAILABLE_CTYPES]
        self.df = make_tx_cost_df(tickers=tiks)
        self.dfw = QuantamentalDataFrame(self.df).to_wide()
        self.change_index = get_diff_index(self.dfw)
        self.tcost_n: str = "BIDOFFER_MEDIAN"
        self.rcost_n: str = "ROLLCOST_MEDIAN"
        self.size_n: str = "SIZE_MEDIAN"
        self.tcost_l: str = "BIDOFFER_90PCTL"
        self.rcost_l: str = "ROLLCOST_90PCTL"
        self.size_l: str = "SIZE_90PCTL"

    def test_get_fids(self):
        fids = get_fids(self.df)
        self.assertTrue(all([fid in fids for fid in self.fids]))

    def test_check_df_for_txn_stats(self):
        check_df_for_txn_stats(
            self.df,
            fids=self.fids,
            tcost_n=self.tcost_n,
            tcost_l=self.tcost_l,
            rcost_n=self.rcost_n,
            rcost_l=self.rcost_l,
            size_n=self.size_n,
            size_l=self.size_l,
        )

        # check invalid fid
        inv_fid = "USD_FX_BIDOFFER_MEDIAN"
        with self.assertRaises(ValueError):
            check_df_for_txn_stats(
                self.df,
                fids=self.fids + [inv_fid],
                tcost_n=self.tcost_n,
                tcost_l=self.tcost_l,
                rcost_n=self.rcost_n,
                rcost_l=self.rcost_l,
                size_n=self.size_n,
                size_l=self.size_l,
            )

    def test_get_diff_index(self):
        change_index = get_diff_index(self.dfw)
        self.assertTrue(len(change_index) < len(self.dfw.index))
        df_diff = self.dfw.diff(axis=0)
        chg_idx = df_diff.index[((df_diff.abs() > 0) | df_diff.isnull()).any(axis=1)]
        self.assertTrue((change_index == chg_idx).all())

    def test_extrapolate_cost(self):
        good_args = {
            "trade_size": 1e7,  # 10mil
            "median_size": 3e6,  # 3mil
            "median_cost": 5e4,  # 50k
            "pct90_size": 1.5e7,  # 15mil
            "pct90_cost": 1e5,  # 100k
        }

        # test with good args
        cost = extrapolate_cost(**good_args)
        assert isinstance(cost, Number)
        cost = good_args["median_cost"] + (
            good_args["trade_size"] - good_args["median_size"]
        ) * (good_args["pct90_cost"] - good_args["median_cost"]) / (
            good_args["pct90_size"] - good_args["median_size"]
        )
        self.assertAlmostEqual(cost, extrapolate_cost(**good_args))

        # test with negative trade size, result should be the same
        bad_args = good_args.copy()
        bad_args["trade_size"] = bad_args["trade_size"] * -1
        self.assertAlmostEqual(cost, extrapolate_cost(**bad_args))

        # test with bad args
        for k in good_args:
            bad_args = good_args.copy()
            bad_args[k] = "a"
            with self.assertRaises(TypeError):
                extrapolate_cost(**bad_args)
            if k == "trade_size":
                continue
            bad_args[k] = -1
            with self.assertRaises(ValueError):
                extrapolate_cost(**bad_args)

        # try with trade size less than median size
        good_args["trade_size"] = 1e6
        self.assertEqual(good_args["median_cost"], extrapolate_cost(**good_args))


class TestSparseCosts(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD"]
        self.tiks = [f"{c}_{k}" for c in self.cids for k in AVAILABLE_CATS]
        self.fids = [f"{c}_{ct}" for c in self.cids for ct in AVAILABLE_CTYPES]
        self.df = make_tx_cost_df(tickers=self.tiks)
        self.dt_range = pd.bdate_range(
            self.df["real_date"].min(), self.df["real_date"].max()
        )

    def test_init(self):
        sc = SparseCosts(self.df)
        self.assertTrue(all([fid in sc._all_fids for fid in self.fids]))
        self.assertTrue(set(self.tiks) == set(sc.df_wide.columns))

    def test_get_costs(self):
        sc = SparseCosts(self.df)
        for _ in range(10):
            rd = np.random.choice(self.dt_range)
            fid = np.random.choice(self.fids)
            result = sc.get_costs(fid, rd)
            self.assertIsInstance(result, pd.Series)

            tiks_with_fid = [tik for tik in self.tiks if tik.startswith(fid)]
            self.assertEqual(set(result.index.tolist()), set(tiks_with_fid))

            expected_df = sc.df_wide.loc[sc.df_wide.index <= rd, tiks_with_fid]
            last_valid_index = expected_df.last_valid_index()
            expected_result = expected_df.loc[last_valid_index]
            self.assertTrue(expected_result.equals(result))

        # test with invalid fid
        inv_fid = "GBP_RANDOM_BIDOFFER_MEDIAN"
        result = sc.get_costs(inv_fid, rd)
        self.assertIsNone(result)

        # test with valid fid, invalid cost type
        inv_fid = "GBP_FX_RANDOM_MEDIAN"
        result = sc.get_costs(inv_fid, rd)
        self.assertIsNone(result)

        # test with 2021 date, should return last valid date
        fid = np.random.choice(self.fids)
        rd = "2021-01-01"
        result = sc.get_costs(fid, rd)

        tiks_with_fid = [tik for tik in self.tiks if tik.startswith(fid)]
        self.assertEqual(set(result.index.tolist()), set(tiks_with_fid))

        expected_df = sc.df_wide.loc[sc.df_wide.index <= rd, tiks_with_fid]
        last_valid_index = expected_df.last_valid_index()
        expected_result = expected_df.loc[last_valid_index]

        self.assertTrue(expected_result.equals(result))


class TestTransactionCosts(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD"]
        self.tiks = [f"{c}_{k}" for c in self.cids for k in AVAILABLE_CATS]
        self.fids = [f"{c}_{ct}" for c in self.cids for ct in AVAILABLE_CTYPES]
        self.df = make_tx_cost_df(tickers=self.tiks)
        self.dt_range = pd.bdate_range(
            self.df["real_date"].min(), self.df["real_date"].max()
        )

    def test_init(self):
        tc = TransactionCosts(self.df)

        self.assertTrue(tc.change_index.equals(tc.sparse_costs.change_index))
        self.assertTrue(tc.df_wide.equals(tc.sparse_costs.df_wide))
        self.assertTrue(tc.qdf.equals(tc.sparse_costs.qdf))


if __name__ == "__main__":
    unittest.main()
