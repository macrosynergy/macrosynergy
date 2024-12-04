"""Test historical volatility estimates with simulate returns from random normal distribution"""

import unittest
import unittest.mock
import matplotlib.pyplot
import pandas as pd
import numpy as np
import matplotlib

from typing import List
from numbers import Number

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
    _plot_costs_func,
    SparseCosts,
    TransactionCosts,
)

from macrosynergy.management.utils import (  # noqa
    qdf_to_ticker_df,
    get_sops,
    ticker_df_to_qdf,
    _map_to_business_day_frequency,
)
from macrosynergy.management.types import QuantamentalDataFrame, NoneType  # noqa
from macrosynergy.management.simulate import (  # noqa
    make_test_df,
    simulate_returns_and_signals,
)

KNOWN_FID_ENDINGS = [f"{t}_{s}" for t in AVAIALBLE_COSTS for s in AVAILABLE_STATS]


# def make_tx_cost_df(
#     cids: List[str] = None,
#     tickers: List[str] = None,
#     start="2020-01-01",
#     end="2025-01-01",
# ) -> pd.DataFrame:
#     err = "Either cids or tickers must be provided (not both)"
#     assert bool(cids) or bool(tickers), err
#     assert bool(cids) ^ bool(tickers), err

#     if cids is None:
#         tiks = tickers
#     else:
#         tiks = [f"{c}_{k}" for c in cids for k in AVAILABLE_CATS]

#     date_range = pd.bdate_range(start=start, end=end, freq="BME")

#     val_dict = {
#         "BIDOFFER_MEDIAN": (0.1, 0.2),
#         "BIDOFFER_90PCTL": (0.5, 2),
#         "ROLLCOST_MEDIAN": (0.001, 0.006),
#         "ROLLCOST_90PCTL": (0.007, 0.01),
#         "SIZE_MEDIAN": (10, 20),
#         "SIZE_90PCTL": (50, 70),
#     }

#     ct_map = {
#         "FX": (10, 20),
#         "IRS": (100, 150),
#         "CDS": (1000, 1500),
#     }

#     df = pd.DataFrame(index=date_range)
#     for tik in tiks:
#         cid = tik.split("_")[0]
#         # add all cols in val_dict
#         for cost_type, (mn, mx) in val_dict.items():
#             for fid_type, (rn, rx) in ct_map.items():
#                 df[f"{cid}_{fid_type}{cost_type}"] = np.random.uniform(
#                     mn, mx, len(df)
#                 ) * np.random.uniform(rn, rx)

#     df.index.name = "real_date"
#     # forward will this to complete for every day
#     new_index = pd.bdate_range(start=start, end=end, freq="B")
#     df = df.reindex(new_index).ffill().bfill()


#     return QuantamentalDataFrame.from_wide(df)


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

    date_range = pd.bdate_range(start=start, end=end)
    date_batches = [date_range[i : i + 30] for i in range(0, len(date_range), 30)]

    dataframes = [
        pd.DataFrame(
            data=abs(float(np.random.rand())),
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


def benchmark_extrapolate_cost(
    trade_size: float,
    median_size: float,
    median_cost: float,
    pct90_size: float,
    pct90_cost: float,
) -> float:
    err_msg = "`{k}` must be a number > 0"
    if not isinstance(trade_size, Number):
        raise TypeError(err_msg.format(k="trade_size"))
    trade_size = abs(trade_size)

    for k, v in [
        ("trade_size", trade_size),
        ("median_size", median_size),
        ("median_cost", median_cost),
        ("pct90_size", pct90_size),
        ("pct90_cost", pct90_cost),
    ]:
        if not isinstance(v, Number):
            raise TypeError(err_msg.format(k=k))
        if v < 0:
            raise ValueError(err_msg.format(k=k))

    if trade_size <= median_size:
        cost = median_cost
    else:
        b = (pct90_cost - median_cost) / (pct90_size - median_size)
        cost = median_cost + b * (trade_size - median_size)
    return cost


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

    def test_benchmark_extrapolate_cost(self):
        for _ in range(100):
            trade_size = np.random.randint(1, 1e7)
            median_size = np.random.randint(1, 1e7)
            median_cost = np.random.randint(1, 1e7)
            pct90_size = np.random.randint(1, 1e7)
            pct90_cost = np.random.randint(1, 1e7)

            result = benchmark_extrapolate_cost(
                trade_size=trade_size,
                median_size=median_size,
                median_cost=median_cost,
                pct90_size=pct90_size,
                pct90_cost=pct90_cost,
            )

            expected_result = extrapolate_cost(
                trade_size=trade_size,
                median_size=median_size,
                median_cost=median_cost,
                pct90_size=pct90_size,
                pct90_cost=pct90_cost,
            )

            self.assertAlmostEqual(expected_result, result)


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

        self.assertRaises(TypeError, SparseCosts, qdf_to_ticker_df(self.df))

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
        inv_fid = "AUD_RANDOM"
        result = sc.get_costs(inv_fid, rd)
        self.assertIsNone(result)

        test_sc = SparseCosts(self.df)
        # drop one fid
        sel_fid = test_sc._all_fids[0]
        test_sc._all_fids = test_sc._all_fids[1:]
        for icol in list(test_sc.df_wide.columns):
            if icol.startswith(sel_fid):
                test_sc.df_wide.drop(columns=icol, inplace=True)

        # test with missing fid
        result = test_sc.get_costs(sel_fid, rd)
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
        tc = TransactionCosts(df=self.df, fids=self.fids)

        self.assertTrue(tc.change_index.equals(tc.sparse_costs.change_index))
        self.assertTrue(tc.df_wide.equals(tc.sparse_costs.df_wide))
        self.assertTrue(tc.qdf.equals(tc.sparse_costs.df))

    def test_download(self):
        with unittest.mock.patch(
            "macrosynergy.pnl.transaction_costs.download_transaction_costs",
            return_value=self.df,
        ):
            tc = TransactionCosts.download()
            self.assertTrue(tc.qdf.eq(self.df).all().all())

    def test_from_qdf(self):
        tc = TransactionCosts.from_qdf(self.df, fids=self.fids)
        self.assertTrue(tc.qdf.eq(self.df).all().all())

    def test_get_costs(self):
        tc = TransactionCosts(df=self.df, fids=self.fids)
        for _ in range(10):
            rd = np.random.choice(self.dt_range)
            fid = np.random.choice(self.fids)
            result = tc.get_costs(fid, rd)
            expected_result = tc.sparse_costs.get_costs(fid, rd)
            self.assertTrue(expected_result.equals(result))

        # test with invalid fid
        inv_fid = "AUD_RANDOM"
        self.assertIsNone(tc.get_costs(inv_fid, rd))

    def test_extrapolate_cost(self):
        tc = TransactionCosts(df=self.df, fids=self.fids)

        for _ in range(100):
            trade_size = np.random.randint(1, 1e7)
            median_size = np.random.randint(1, 1e7)
            median_cost = np.random.randint(1, 1e7)
            pct90_size = np.random.randint(1, 1e7)
            pct90_cost = np.random.randint(1, 1e7)

            result = tc.extrapolate_cost(
                trade_size=trade_size,
                median_size=median_size,
                median_cost=median_cost,
                pct90_size=pct90_size,
                pct90_cost=pct90_cost,
            )

            expected_result = extrapolate_cost(
                trade_size=trade_size,
                median_size=median_size,
                median_cost=median_cost,
                pct90_size=pct90_size,
                pct90_cost=pct90_cost,
            )

            if np.isnan(expected_result):
                self.assertTrue(np.isnan(result))
            else:
                self.assertAlmostEqual(expected_result, result)

        self.assertEqual(tc.extrapolate_cost(None, 1, 1, 1, 1), 0)

    def test_bidoffer(self):
        tc = TransactionCosts(df=self.df, fids=self.fids)
        # mock the tc.sparse_costs.get_costs method
        for _ in range(100):
            fid = np.random.choice(self.fids)
            trade_size = np.random.randint(1, 1e7)
            real_date = np.random.choice(self.dt_range)

            result = tc.bidoffer(fid=fid, trade_size=trade_size, real_date=real_date)

            rel_row = tc.sparse_costs.get_costs(fid=fid, real_date=real_date)
            expected_result = tc.extrapolate_cost(
                trade_size=trade_size,
                median_size=rel_row[fid + tc.size_n],
                median_cost=rel_row[fid + tc.tcost_n],
                pct90_size=rel_row[fid + tc.size_l],
                pct90_cost=rel_row[fid + tc.tcost_l],
            )
            if np.isnan(expected_result):
                self.assertTrue(np.isnan(result))
            else:
                self.assertAlmostEqual(expected_result, result)

    def test_rollcost(self):
        tc = TransactionCosts(df=self.df, fids=self.fids)
        for _ in range(100):
            fid = np.random.choice(self.fids)
            trade_size = np.random.randint(1, 1e7)
            real_date = np.random.choice(self.dt_range)

            result = tc.rollcost(fid=fid, trade_size=trade_size, real_date=real_date)

            rel_row = tc.sparse_costs.get_costs(fid=fid, real_date=real_date)
            expected_result = tc.extrapolate_cost(
                trade_size=trade_size,
                median_size=rel_row[fid + tc.size_n],
                median_cost=rel_row[fid + tc.rcost_n],
                pct90_size=rel_row[fid + tc.size_l],
                pct90_cost=rel_row[fid + tc.rcost_l],
            )
            if np.isnan(expected_result):
                self.assertTrue(np.isnan(result))
            else:
                self.assertAlmostEqual(expected_result, result)

    def test_plot_costs_func(self):
        matplotlib.pyplot.close("all")
        curr_backend = matplotlib.pyplot.get_backend()
        matplotlib.use("Agg")
        tc = TransactionCosts(df=self.df, fids=self.fids)

        good_args = {
            "fids": self.fids[:3],
            "cost_type": "BIDOFFER",
            "ncol": 2,
            "x_axis_label": "Real Date",
            "y_axis_label": "Cost",
            "title": "Test",
            "title_fontsize": 28,
            "facet_title_fontsize": 20,
        }

        with self.assertRaises(ValueError):
            bad_args = good_args.copy()
            bad_args["fids"] = [1, 2]
            _plot_costs_func(tco=tc, **bad_args)

        with self.assertRaises(ValueError):
            bad_tc = TransactionCosts(df=self.df, fids=self.fids[:3])
            bad_tc.__dict__.pop("sparse_costs")
            _plot_costs_func(tco=bad_tc, **good_args)

        tc.plot_costs(**good_args)
        matplotlib.pyplot.close("all")
        matplotlib.use(curr_backend)


if __name__ == "__main__":
    unittest.main()
