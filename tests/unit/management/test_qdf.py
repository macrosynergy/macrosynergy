import unittest
import numpy as np
import pandas as pd
import random
import string
from typing import List
import warnings
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.constants import JPMAQS_METRICS
from macrosynergy.management.utils import get_cid, get_xcat
from macrosynergy.management.types.qdf.methods import (
    get_col_sort_order,
    change_column_format,
    qdf_to_categorical,
    qdf_to_string_index,
    check_is_categorical,
    _get_tickers_series,
    apply_blacklist,
    reduce_df,
    reduce_df_by_ticker,
    update_df,
    update_tickers,
    update_categories,
    qdf_to_wide_df,
    add_ticker_column,
    rename_xcats,
    _add_index_str_column,
    add_nan_series,
    drop_nan_series,
    qdf_from_timseries,
    create_empty_categorical_qdf,
    concat_qdfs,
)


from macrosynergy.management.simulate import make_test_df


def helper_random_tickers(n: int = 10) -> List[str]:
    cids = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "SEK", "NOK", "DKK"]
    cids += ["NZD", "ZAR", "BRL", "CNY", "INR", "RUB", "TRY", "KRW", "IDR", "MXN"]
    xcats = ["FXXR", "IR", "EQXR", "CDS", "PPP", "CTOT", "CPI", "PMI", "GDP", "UNR"]
    xcats += ["HSP", "CREDIT", "INDPROD", "RETAIL", "SENTIMENT", "TRADE"]
    adjs = ["", "_NSA", "_SA", "_SJA"]

    all_tickers = [
        f"{cid}_{xcat}{adj}" for cid in cids for xcat in xcats for adj in adjs
    ]
    while n > len(all_tickers):
        all_tickers += all_tickers
    return random.sample(all_tickers, n)


def helper_split_df_by_metrics(
    df: QuantamentalDataFrame,
) -> List[QuantamentalDataFrame]:
    return [
        df[QuantamentalDataFrame.IndexCols + [m]].reset_index(drop=True)
        for m in (set(df.columns) - set(QuantamentalDataFrame.IndexCols))
    ]


def helper_split_df_by_ticker(df: QuantamentalDataFrame) -> List[QuantamentalDataFrame]:
    return [sdf for (c, x), sdf in df.groupby(["cid", "xcat"], observed=True)]


class TestQDFBasic(unittest.TestCase):
    def test_quantamental_dataframe_type(self):
        test_df: pd.DataFrame = make_test_df()
        self.assertTrue(isinstance(test_df, QuantamentalDataFrame))
        # rename xcat to xkat
        df: pd.DataFrame = test_df.copy()
        df.columns = df.columns.str.replace("xcat", "xkat")
        self.assertFalse(isinstance(df, QuantamentalDataFrame))

        # rename cid to xid
        df: pd.DataFrame = test_df.copy()
        df.columns = df.columns.str.replace("cid", "xid")
        self.assertFalse(isinstance(df, QuantamentalDataFrame))

        # change one date to a string
        df: pd.DataFrame = test_df.copy()
        self.assertTrue(isinstance(df, QuantamentalDataFrame))

        # change one date to a string -- remember the caveats for pd arrays
        df: pd.DataFrame = test_df.copy()
        nseries: List[pd.Timestamp] = df["real_date"].tolist()
        nseries[0] = "2020-01-01"
        df["real_date"] = pd.Series(nseries, dtype=object).copy()
        self.assertFalse(isinstance(df, QuantamentalDataFrame))

        # Check if subclassing works as expected
        df: pd.DataFrame = test_df.copy()

        dfx: QuantamentalDataFrame = QuantamentalDataFrame(df)

        self.assertTrue(isinstance(dfx, QuantamentalDataFrame))
        self.assertTrue(isinstance(dfx, pd.DataFrame))

        df_Q = (
            QuantamentalDataFrame(df)
            .sort_values(["cid", "xcat", "real_date"])
            .reset_index(drop=True)
        )
        df_S = df.sort_values(["cid", "xcat", "real_date"]).reset_index(drop=True)
        self.assertTrue((df_S == df_Q).all().all())

        # test with categorical=False
        df_Q = (
            QuantamentalDataFrame(df, categorical=False)
            .sort_values(["cid", "xcat", "real_date"])
            .reset_index(drop=True)
        )
        df_S = df.sort_values(["cid", "xcat", "real_date"]).reset_index(drop=True)
        self.assertTrue(df_Q.equals(df_S))


class TestMethods(unittest.TestCase):
    def test_get_col_sort_order(self):
        expc_order = QuantamentalDataFrame.IndexCols + JPMAQS_METRICS

        test_df: pd.DataFrame = make_test_df(metrics=JPMAQS_METRICS)

        torder = get_col_sort_order(test_df)
        self.assertEqual(expc_order, torder)

        # shuffle the col order and check
        scols = random.sample(expc_order, len(expc_order))
        test_df = test_df[scols]

        torder = get_col_sort_order(test_df)
        self.assertEqual(expc_order, torder)

        with self.assertRaises(TypeError):
            get_col_sort_order(pd.DataFrame())

    def test_change_column_format(self):
        test_df: pd.DataFrame = make_test_df(metrics=JPMAQS_METRICS)

        # change dtype of eop_lag to int
        tdf = change_column_format(test_df, cols=["eop_lag"], dtype="int64")
        self.assertEqual(tdf["eop_lag"].dtype.name, "int64")

        # change dtype of eop_lag to float
        tdf = change_column_format(test_df, cols=["eop_lag"], dtype="float")
        self.assertEqual(tdf["eop_lag"].dtype, np.float64)

        # change the string columns to category
        tdf = change_column_format(test_df, cols=["cid", "xcat"], dtype="category")
        self.assertEqual(tdf["cid"].dtype.name, "category")
        self.assertTrue(check_is_categorical(tdf))

        with self.assertRaises(TypeError):
            change_column_format(
                test_df.rename(columns={"cid": "xid"}), ["cid"], "category"
            )

        with self.assertRaises(TypeError):
            change_column_format(test_df, 1, "category")

        with warnings.catch_warnings(record=True) as w:
            change_column_format(test_df, ["random-col"], "category")
            self.assertTrue(len(w) == 1)

    def test_qdf_to_categorical(self):
        test_df: pd.DataFrame = make_test_df(metrics=JPMAQS_METRICS)

        curr_dtypes = test_df.dtypes.to_dict()

        tdf = qdf_to_categorical(test_df)
        # check the types of cid, xcat columns
        self.assertEqual(tdf["cid"].dtype.name, "category")
        self.assertEqual(tdf["xcat"].dtype.name, "category")
        self.assertTrue(check_is_categorical(tdf))

        # check that the other columns are not changed
        for col in JPMAQS_METRICS:
            self.assertEqual(tdf[col].dtype, curr_dtypes[col])

        # check that is fails if not a QuantamentalDataFrame
        with self.assertRaises(TypeError):
            qdf_to_categorical(test_df.rename(columns={"cid": "xid"}))

    def test_qdf_to_string_index(self):
        test_df: pd.DataFrame = make_test_df(metrics=JPMAQS_METRICS)

        curr_dtypes = test_df.dtypes.to_dict()

        tdf = qdf_to_string_index(test_df)
        # check the types of cid, xcat columns
        self.assertEqual(tdf["cid"].dtype.name, "object")
        self.assertEqual(tdf["xcat"].dtype.name, "object")
        self.assertFalse(check_is_categorical(tdf))

        # check that the other columns are not changed
        for col in JPMAQS_METRICS:
            self.assertEqual(tdf[col].dtype, curr_dtypes[col])

        # check that is fails if not a QuantamentalDataFrame
        with self.assertRaises(TypeError):
            qdf_to_string_index(test_df.rename(columns={"cid": "xid"}))

    def test_check_is_categorical(self):
        test_df: pd.DataFrame = make_test_df()
        self.assertFalse(check_is_categorical(test_df))

        test_df = qdf_to_categorical(test_df)
        self.assertTrue(check_is_categorical(test_df))

    def test_get_tickers_series(self):
        test_df: pd.DataFrame = make_test_df()

        tickers = test_df["cid"] + "_" + test_df["xcat"]
        tseries = _get_tickers_series(QuantamentalDataFrame(test_df))

        self.assertTrue(tseries.tolist() == tickers.tolist())

        # now test for non categorical
        tseries = _get_tickers_series(test_df)
        self.assertTrue(tseries.tolist() == tickers.tolist())

        with self.assertRaises(ValueError):
            _get_tickers_series(df=test_df, cid_column="A")
        with self.assertRaises(ValueError):
            _get_tickers_series(df=test_df, xcat_column="A")

    def test_create_empty_categorical_qdf(self):
        dt_range = pd.bdate_range("2000-01-01", "2020-01-01")

        good_args = dict(cid="A", xcat="X", metrics=["a", "b"], date_range=dt_range)

        df = create_empty_categorical_qdf(**good_args)

        alt_args = dict(
            ticker="A_X",
            metrics=["a", "b"],
            start_date="2000-01-01",
            end_date="2020-01-01",
        )

        df2 = create_empty_categorical_qdf(**alt_args)

        self.assertTrue(df.equals(df2))

        test_ticker = (
            (df2["cid"].astype(str) + "_" + df2["xcat"].astype(str))
            .unique()
            .tolist()[0]
        )

        self.assertEqual(test_ticker, "A_X")

        self.assertTrue(
            set(df.columns) - set(QuantamentalDataFrame.IndexCols) == set(["a", "b"])
        )

        with self.assertRaises(ValueError):
            create_empty_categorical_qdf(
                cid="A", metrics=["a", "b"], date_range=dt_range
            )

        with self.assertRaises(ValueError):
            create_empty_categorical_qdf(
                xcat="X", metrics=["a", "b"], date_range=dt_range
            )

        with self.assertRaises(ValueError):
            create_empty_categorical_qdf(
                cid="A", xcat="X", ticker="A_X", date_range=dt_range
            )

        with self.assertRaises(ValueError):
            create_empty_categorical_qdf(cid="A", xcat="X", metrics=["a", "b"])

        with self.assertRaises(TypeError):
            create_empty_categorical_qdf(
                cid="A", xcat="X", metrics=["a", 1], date_range=dt_range
            )

    def test_qdf_from_timseries(self):
        ts = pd.Series(
            np.random.randn(100), index=pd.bdate_range("2020-01-01", periods=100)
        )

        # test with cid and xcat
        qdf = qdf_from_timseries(ts, cid="A", xcat="X")
        self.assertTrue(check_is_categorical(qdf))

        # test with ticker
        qdf = qdf_from_timseries(ts, ticker="A_X")
        self.assertTrue(check_is_categorical(qdf))

        # test with ticker
        qdf = qdf_from_timseries(ts, ticker="A_X")
        self.assertTrue(check_is_categorical(qdf))

        # test with non datetime index

        with self.assertRaises(ValueError):
            ts_copy = ts.copy()
            ts_copy.index = ts_copy.index.astype(str)
            qdf_from_timseries(ts_copy, ticker="A_X")

        # test with non pd.Series
        with self.assertRaises(TypeError):
            qdf_from_timseries(ts.values, ticker="A_X")

        # test with non string metric
        with self.assertRaises(TypeError):
            qdf_from_timseries(ts, ticker="A_X", metric=1)

        # test with no cid or xcat
        with self.assertRaises(ValueError):
            qdf_from_timseries(ts, ticker="A_X", cid="A", xcat="X")

        with self.assertRaises(ValueError):
            qdf_from_timseries(ts, ticker="A_X", cid="A")

        with self.assertRaises(ValueError):
            qdf_from_timseries(ts, xcat="X")


class TestQDFMethods(unittest.TestCase):
    def test_drop_nan_series(self):

        tickers = helper_random_tickers(50)
        sel_tickers = random.sample(tickers, 10)
        test_df: pd.DataFrame = make_test_df(tickers=tickers)

        for (cid, xcat), sdf in test_df.groupby(["cid", "xcat"], observed=True):
            if f"{cid}_{xcat}" in sel_tickers:
                test_df.loc[sdf.index, "value"] = np.nan

        # randomly make 200 more nans
        n_random_nans = 200
        for _ in range(n_random_nans):
            rrow = random.randint(0, test_df.shape[0] - 1)
            test_df.loc[rrow, "value"] = np.nan

        qdf = QuantamentalDataFrame(test_df)

        with warnings.catch_warnings(record=True) as w:
            out_qdf = drop_nan_series(qdf, raise_warning=True)
            self.assertTrue(len(w) == len(sel_tickers))

        out_tickers = list(_get_tickers_series(out_qdf).unique())

        self.assertTrue(set(tickers) - set(out_tickers) == set(sel_tickers))

        # test return when no nans
        test_qdf = make_test_df(tickers=helper_random_tickers(10))
        self.assertTrue(drop_nan_series(test_qdf).equals(test_qdf))

        # test with non QuantamentalDataFrame
        with self.assertRaises(TypeError):
            drop_nan_series(test_df.rename(columns={"cid": "xid"}))

        with self.assertRaises(ValueError):
            drop_nan_series(test_df, column="random-col")

        with self.assertRaises(TypeError):
            drop_nan_series(test_df, raise_warning="True")

    def test_add_nan_series(self):
        tickers = helper_random_tickers(10)
        test_df: pd.DataFrame = make_test_df(tickers=tickers)
        sel_ticker = random.choice(tickers)
        # add a series of nans
        new_df = add_nan_series(test_df, ticker=sel_ticker)

        nan_loc = new_df[new_df["value"].isna()]
        found_cids = nan_loc["cid"].unique()
        self.assertEqual(len(found_cids), 1)
        found_cid = found_cids[0]

        found_xcats = nan_loc["xcat"].unique()
        self.assertEqual(len(found_xcats), 1)
        found_xcat = found_xcats[0]

        self.assertEqual(f"{found_cid}_{found_xcat}", sel_ticker)

        # test with custom date range
        start = "2020-01-01"
        end = "2020-01-10"

        new_df = add_nan_series(test_df, ticker=sel_ticker, start=start, end=end)

        nan_loc = new_df[new_df["value"].isna()]
        # check that the dates are pd.bdate_range(start, end)
        self.assertTrue(
            (nan_loc["real_date"].unique() == pd.bdate_range(start, end)).all()
        )

        with self.assertRaises(TypeError):
            add_nan_series(df=1, ticker=sel_ticker)

    def test_qdf_to_wide_df(self):
        tickers = helper_random_tickers(10)
        test_df: pd.DataFrame = make_test_df(tickers=tickers)

        qdf = QuantamentalDataFrame(test_df)
        wide_df = qdf_to_wide_df(qdf)

        self.assertTrue(wide_df.shape[0] == len(qdf["real_date"].unique()))
        self.assertTrue(wide_df.shape[1] == len(tickers))
        self.assertTrue((wide_df.index == qdf["real_date"].unique()).all())

        # test with non QuantamentalDataFrame
        with self.assertRaises(TypeError):
            qdf_to_wide_df(test_df.rename(columns={"cid": "xid"}))

        with self.assertRaises(TypeError):
            qdf_to_wide_df(test_df, value_column=1)

        with self.assertRaises(ValueError):
            qdf_to_wide_df(test_df, value_column="random-col")

    def test_apply_blacklist(self):
        tickers = [
            f"{cid}_{xcat}"
            for cid in ["USD", "EUR", "GBP", "JPY"]
            for xcat in ["FX", "IR", "EQ"]
        ]
        test_df: pd.DataFrame = make_test_df(
            tickers=tickers,
            cids=None,
            xcats=None,
            start="2010-01-01",
            end="2010-12-31",
        )

        qdf = QuantamentalDataFrame(test_df)
        bl_start, bl_end = pd.Timestamp("2010-06-06"), pd.Timestamp("2010-07-23")
        sel_ticker = random.choice(tickers)
        sel_cid, sel_xcat = sel_ticker.split("_", 1)
        blacklist = {sel_cid: [bl_start, bl_end]}

        new_df = apply_blacklist(qdf, blacklist)

        # check that the dates are not in the new df
        self.assertTrue(
            new_df[
                (new_df["cid"] == sel_cid)
                & (new_df["xcat"] == sel_xcat)
                & (new_df["real_date"] >= bl_start)
                & (new_df["real_date"] <= bl_end)
            ].empty
        )

        # check that all other entries are intact are unchanged by checking if all of them are in the new df
        expected_remaining_entries = (
            test_df[
                ~test_df["cid"].isin([sel_cid])
                | ~test_df["real_date"].between(bl_start, bl_end)
            ]
            .sort_values(by=QuantamentalDataFrame.IndexColsSortOrder)
            .reset_index(drop=True)
        )

        self.assertTrue(expected_remaining_entries.eq(new_df).all().all())

        ## test with very long periiod of time

        tickers = [
            f"{cid}_{xcat}"
            for cid in ["USD", "EUR", "GBP", "JPY"]
            for xcat in ["FX", "IR", "EQ"]
        ]
        bl_start, bl_end = pd.Timestamp("2000-01-01"), pd.Timestamp("2020-01-01")

        test_df: pd.DataFrame = make_test_df(
            tickers=tickers,
            cids=None,
            xcats=None,
            start="2005-01-01",
            end="2015-01-01",
        )

        blacklist = {sel_cid: [bl_start, bl_end]}
        new_df = apply_blacklist(qdf, blacklist)

        self.assertTrue(new_df[new_df["cid"] == sel_cid].empty)

        with self.assertRaises(TypeError):
            apply_blacklist(df=1, blacklist=blacklist)

        with self.assertRaises(TypeError):
            apply_blacklist(df=test_df, blacklist={1: [bl_start, bl_end]})

        with self.assertRaises(TypeError):
            apply_blacklist(df=test_df, blacklist={sel_cid: 1})

        with self.assertRaises(TypeError):
            apply_blacklist(df=test_df, blacklist={sel_cid: [bl_start]})

        with self.assertRaises(TypeError):
            apply_blacklist(df=test_df, blacklist=[])


class TestReduceDF(unittest.TestCase):
    def test_reduce_df_basic(self):
        tickers = helper_random_tickers(20)
        test_df: pd.DataFrame = make_test_df(tickers=tickers)

        qdf = QuantamentalDataFrame(test_df)
        orig_cids = set(qdf["cid"].unique())
        orig_xcats = set(qdf["xcat"].unique())
        # test with no cids or xcats
        new_df: QuantamentalDataFrame = reduce_df(qdf)
        self.assertTrue(new_df.equals(qdf))

        # test with out_all
        new_df, xcats, cids = reduce_df(qdf, out_all=True)
        self.assertTrue(new_df.equals(qdf))

        found_cids = new_df["cid"].unique()
        found_xcats = new_df["xcat"].unique()

        # should be the same as the original
        self.assertTrue(set(found_cids) == orig_cids)
        self.assertTrue(set(found_xcats) == orig_xcats)

    def test_reduce_df_across_cid(self):
        tickers = helper_random_tickers(20)
        start = "2010-01-01"
        end = "2010-12-31"
        test_df: pd.DataFrame = make_test_df(tickers=tickers, start=start, end=end)

        qdf = QuantamentalDataFrame(test_df)
        orig_cids = set(qdf["cid"].unique())
        orig_xcats = set(qdf["xcat"].unique())

        reduce_cids = random.sample(list(orig_cids), random.randint(2, 5))
        rel_tickers = sorted(set([t for t in tickers if get_cid(t) in reduce_cids]))
        reduce_start = pd.Timestamp("2010-06-06")
        reduce_end = pd.Timestamp("2010-08-25")
        # test with cids
        new_df: QuantamentalDataFrame = reduce_df(
            qdf, cids=reduce_cids, start=reduce_start, end=reduce_end
        )

        # check that the cids are the same
        self.assertTrue(set(new_df["cid"].unique()) == set(reduce_cids))

        found_tickers = new_df["cid"].astype(str) + "_" + new_df["xcat"].astype(str)
        self.assertTrue(set(found_tickers) == set(rel_tickers))

        # check that the dates are within the range
        self.assertEqual(
            set(new_df["real_date"]),
            set(pd.bdate_range(reduce_start, reduce_end)),
        )

    def test_reduce_df_across_xcat(self):
        tickers = helper_random_tickers(20)
        start = "2010-01-01"
        end = "2010-12-31"
        test_df: pd.DataFrame = make_test_df(tickers=tickers)

        qdf = QuantamentalDataFrame(test_df)
        orig_cids = set(qdf["cid"].unique())
        orig_xcats = set(qdf["xcat"].unique())

        reduce_xcats = random.sample(list(orig_xcats), random.randint(2, 5))
        rel_tickers = sorted(set([t for t in tickers if get_xcat(t) in reduce_xcats]))
        reduce_start = pd.Timestamp("2010-06-01")
        reduce_end = pd.Timestamp("2010-08-25")
        # test with cids
        new_df: QuantamentalDataFrame = reduce_df(
            qdf, xcats=reduce_xcats, start=reduce_start, end=reduce_end
        )

        # check that the cids are the same
        self.assertTrue(set(new_df["xcat"].unique()) == set(reduce_xcats))

        found_tickers = new_df["cid"].astype(str) + "_" + new_df["xcat"].astype(str)
        self.assertTrue(set(found_tickers) == set(rel_tickers))

        # check that the dates are within the range
        self.assertEqual(
            set(new_df["real_date"]),
            set(pd.bdate_range(reduce_start, reduce_end)),
        )

    def test_reduce_df_intersect(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]
        tickers = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]

        rm_tickers = ["GBP_FX", "GBP_EQ", "AUD_IR", "JPY_CDS"]
        tickers = [t for t in tickers if t not in rm_tickers]

        start = "2010-01-01"
        end = "2010-12-31"
        test_df: pd.DataFrame = make_test_df(tickers=tickers, metrics=JPMAQS_METRICS)

        qdf = QuantamentalDataFrame(test_df)
        orig_cids = set(qdf["cid"].unique())
        orig_xcats = set(qdf["xcat"].unique())

        reduce_xcats = ["FX", "IR", "EQ", "CDS"]

        expected_cids = sorted(
            set([get_cid(t) for t in tickers if get_cid(t) not in get_cid(rm_tickers)])
        )

        new_df: QuantamentalDataFrame = reduce_df(
            qdf, xcats=reduce_xcats, intersect=True, start=start, end=end
        )

        found_cids = sorted(new_df["cid"].unique())
        self.assertEqual(set(found_cids), set(expected_cids))

        found_xcats = sorted(new_df["xcat"].unique())
        self.assertEqual(set(found_xcats), set(reduce_xcats))

        # check that the dates are within the range
        dtrange = pd.bdate_range(start, end)
        for (_cid, _xcat), sdf in new_df.groupby(["cid", "xcat"], observed=True):
            self.assertEqual(set(sdf["real_date"]), set(dtrange))

    def test_reduce_df_blacklist(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]

        start, end = "2010-01-01", "2015-01-01"

        blacklist = {
            "GBP": [pd.Timestamp("2010-06-06"), pd.Timestamp("2010-07-23")],
            "AUD": [pd.Timestamp("2000-01-01"), pd.Timestamp("2025-01-01")],
        }

        df = make_test_df(
            cids=cids, xcats=xcats, start=start, end=end, metrics=JPMAQS_METRICS
        )

        qdf = QuantamentalDataFrame(df)

        new_df = reduce_df(qdf, blacklist=blacklist)

        # test the result = apply_blacklist(qdf, blacklist)
        expected_df = apply_blacklist(qdf, blacklist)

        self.assertTrue(new_df.equals(expected_df))

        # test with some reduced as well
        reduce_cids = ["USD", "AUD", "GBP"]
        reduce_xcats = ["FX", "IR", "EQ"]

        new_df = reduce_df(
            qdf,
            blacklist=blacklist,
            cids=reduce_cids,
            xcats=reduce_xcats,
        )

        expected_df = reduce_df(
            apply_blacklist(df=qdf, blacklist=blacklist),
            cids=reduce_cids,
            xcats=reduce_xcats,
        )

        self.assertTrue(new_df.equals(expected_df))

    def test_reduce_df_str_args(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]
        df = make_test_df(cids=cids, xcats=xcats, metrics=JPMAQS_METRICS)

        qdf = QuantamentalDataFrame(df)

        rdf = reduce_df(qdf, cids="USD", xcats="FX")

        expected_df = qdf[(qdf["cid"] == "USD") & (qdf["xcat"] == "FX")].reset_index(
            drop=True
        )

        self.assertTrue(rdf.equals(expected_df))

    def test_reduce_df_by_ticker(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]
        tickers = [f"{c}_{x}" for c in cids for x in xcats]
        start = "2010-01-01"
        end = "2010-12-31"
        test_df: pd.DataFrame = make_test_df(
            cids=cids, xcats=xcats, start=start, end=end
        )

        qdf = QuantamentalDataFrame(test_df)

        sel_tickers = random.sample(tickers, 10)
        sel_start = "2010-06-10"
        sel_end = "2010-06-20"
        new_df: QuantamentalDataFrame = reduce_df_by_ticker(
            df=qdf, tickers=sel_tickers, start=sel_start, end=sel_end
        )

        self.assertTrue(set(_get_tickers_series(new_df).unique()) == set(sel_tickers))

        df_ts = _get_tickers_series(qdf)
        expected_df = qdf.loc[
            (df_ts.isin(sel_tickers))
            & (qdf["real_date"] >= pd.Timestamp(sel_start))
            & (qdf["real_date"] <= pd.Timestamp(sel_end)),
        ].reset_index(drop=True)

        self.assertTrue(new_df.equals(expected_df))

        # test with non QuantamentalDataFrame
        with self.assertRaises(TypeError):
            reduce_df_by_ticker(df=1, tickers=sel_tickers)

        with self.assertRaises(TypeError):
            reduce_df_by_ticker(df=test_df, tickers=1)

        empty_df = reduce_df_by_ticker(df=test_df, tickers=["random-ticker"])
        self.assertTrue(empty_df.empty)

        # test with tickers=None
        new_df = reduce_df_by_ticker(df=qdf, tickers=None)

        self.assertTrue(new_df.equals(qdf))

        # test with blacklist of AUD
        blacklist = {"AUD": [pd.Timestamp("2010-06-06"), pd.Timestamp("2010-07-23")]}
        new_df = reduce_df_by_ticker(df=qdf, blacklist=blacklist, tickers=None)

        expected_df = apply_blacklist(qdf, blacklist)

        self.assertTrue(new_df.equals(expected_df))


class TestUpdateDF(unittest.TestCase):
    def test_upate_df_basic(self):
        tickers = helper_random_tickers(20)
        dfa = make_test_df(tickers=tickers[:10])
        dfb = make_test_df(tickers=tickers[10:])

        qdfa = QuantamentalDataFrame(dfa)
        qdfb = QuantamentalDataFrame(dfb)

        new_df = update_df(qdfa, qdfb)

        expected_df = (
            pd.concat([qdfa, qdfb], axis=0, ignore_index=True)
            .sort_values(QuantamentalDataFrame.IndexColsSortOrder)
            .drop_duplicates(subset=QuantamentalDataFrame.IndexCols, keep="last")
            .reset_index(drop=True)
        )

        self.assertTrue(new_df.equals(expected_df))
        
        self.assertRaises(TypeError, update_df, 1, dfa)
        self.assertRaises(TypeError, update_df, dfa, 1)
        self.assertRaises(TypeError, update_df, dfa, dfa, 'string')

    def test_update_df_with_nans(self):
        tickers = helper_random_tickers(20)
        dfa = make_test_df(tickers=tickers[:10], metrics=JPMAQS_METRICS)
        dfb = make_test_df(tickers=tickers[10:], metrics=JPMAQS_METRICS)

        for _ in range(100):
            rrow = random.randint(0, dfa.shape[0] - 1)
            rcol = random.choice(JPMAQS_METRICS)
            dfa.loc[rrow, rcol] = np.nan

            rrow = random.randint(0, dfb.shape[0] - 1)
            rcol = random.choice(JPMAQS_METRICS)
            dfb.loc[rrow, rcol] = np.nan

        qdfa = QuantamentalDataFrame(dfa)
        qdfb = QuantamentalDataFrame(dfb)

        new_df = update_df(qdfa, qdfb)

        expected_df = (
            pd.concat([qdfa, qdfb], axis=0, ignore_index=True)
            .sort_values(QuantamentalDataFrame.IndexColsSortOrder)
            .drop_duplicates(subset=QuantamentalDataFrame.IndexCols, keep="last")
            .reset_index(drop=True)
        )

        self.assertTrue(new_df.equals(expected_df))

    def test_update_df_short_circuited(self):
        tickers = helper_random_tickers(20)
        dfa = make_test_df(tickers=tickers[:10], metrics=JPMAQS_METRICS)
        dfb = make_test_df(tickers=tickers[10:], metrics=JPMAQS_METRICS)

        empty_df = pd.DataFrame(
            columns=QuantamentalDataFrame.IndexCols + JPMAQS_METRICS
        )

        test_df = update_df(empty_df, empty_df)
        self.assertTrue(test_df.empty)

        test_df = update_df(dfa, empty_df)
        self.assertTrue(test_df.equals(dfa))

        test_df = update_df(empty_df, dfb)
        self.assertTrue(test_df.equals(dfb))


class TestConcatQDFs(unittest.TestCase):

    def test_concat_qdfs_simple(self):
        tickers = helper_random_tickers()
        cargs = dict(
            metrics=JPMAQS_METRICS,
            cids=None,
            xcats=None,
        )
        dfA = make_test_df(tickers=tickers[:5], **cargs)
        dfB = make_test_df(tickers=tickers[5:], **cargs)

        qdfA = QuantamentalDataFrame(dfA)
        qdfB = QuantamentalDataFrame(dfB)

        qdfC = concat_qdfs([qdfA, qdfB])

        # assert that all tickers are in the new qdf
        self.assertEqual(set(QuantamentalDataFrame.list_tickers(qdfC)), set(tickers))

        expc_df = (
            pd.concat([qdfA, qdfB], axis=0)
            .sort_values(by=QuantamentalDataFrame.IndexColsSortOrder + JPMAQS_METRICS)
            .reset_index(drop=True)
        )

        self.assertTrue(expc_df.equals(qdfC))

    def test_concat_single_metric_qdfs(self):
        tickers = helper_random_tickers(50)
        cargs = dict(
            metrics=JPMAQS_METRICS,
            cids=None,
            xcats=None,
        )
        dfo = QuantamentalDataFrame(make_test_df(tickers=tickers, **cargs))

        n_random_nans = 219
        for _ in range(n_random_nans):
            rrow = random.randint(0, dfo.shape[0] - 1)
            rcol = random.choice(JPMAQS_METRICS)
            dfo.loc[rrow, rcol] = np.nan

        split_dfs: List[QuantamentalDataFrame] = []
        for sdf in helper_split_df_by_ticker(dfo):
            split_dfs.extend(helper_split_df_by_metrics(sdf))

        for sdf in split_dfs:
            # sort my the metric just for fun
            sdf = sdf.sort_values(
                by=QuantamentalDataFrame.IndexCols + [sdf.columns[-1]]
            )

        output_df = concat_qdfs(split_dfs)

        self.assertTrue(dfo.equals(output_df))

        # get nan locs
        in_nan_rows = dfo[dfo.isna().any(axis=1)]
        out_nan_rows = output_df[output_df.isna().any(axis=1)]

        non_eq_mask = in_nan_rows != out_nan_rows
        nan_mask = dfo[dfo.isna().any(axis=1)].isna()

        self.assertTrue((non_eq_mask == nan_mask).all().all())

    def test_concat_single_metric_qdfs_partial(self):
        tickers = helper_random_tickers(50)
        cargs = dict(
            metrics=JPMAQS_METRICS,
            cids=None,
            xcats=None,
        )
        dfo = QuantamentalDataFrame(make_test_df(tickers=tickers, **cargs))

        n_random_nans = random.randint(100, 1000)
        for _ in range(n_random_nans):
            rrow = random.randint(0, dfo.shape[0] - 1)
            rcol = random.choice(JPMAQS_METRICS)
            dfo.loc[rrow, rcol] = np.nan

        drop_pairs = [
            (t, m) for t in tickers for m in JPMAQS_METRICS if random.random() < 0.1
        ]
        split_dfs: List[QuantamentalDataFrame] = []
        for sdf in helper_split_df_by_ticker(dfo):
            split_dfs.extend(helper_split_df_by_metrics(sdf))

        for sdf in split_dfs:
            sdf = sdf.sort_values(
                by=QuantamentalDataFrame.IndexCols + [sdf.columns[-1]]
            )

        orig_len = len(split_dfs)
        for idf, _ in enumerate(split_dfs):
            cid = split_dfs[idf]["cid"].unique().tolist()[0]
            xcat = split_dfs[idf]["xcat"].unique().tolist()[0]
            _metric = split_dfs[idf].columns[-1]
            for t, m in drop_pairs:
                if t == f"{cid}_{xcat}" and m == _metric:
                    # split_dfs.pop()
                    split_dfs.pop(idf)

                    # dfo[(dfo["cid"] == cid) & (dfo["xcat"] == xcat)][m] = np.nan
                    dfo.loc[(dfo["cid"] == cid) & (dfo["xcat"] == xcat), m] = np.nan

        output_df = concat_qdfs(split_dfs)

        self.assertTrue(dfo.equals(output_df))

        # get nan locs
        in_nan_rows = dfo[dfo.isna().any(axis=1)]
        out_nan_rows = output_df[output_df.isna().any(axis=1)]

        non_eq_mask = in_nan_rows != out_nan_rows
        nan_mask = dfo[dfo.isna().any(axis=1)].isna()

        self.assertTrue((non_eq_mask == nan_mask).all().all())

    def test_concat_qdfs_errors(self):

        dfs = [make_test_df() for _ in range(5)]
        with self.assertRaises(ValueError):
            concat_qdfs([])

        with self.assertRaises(TypeError):
            concat_qdfs(dfs + [pd.DataFrame()])

        with self.assertRaises(TypeError):
            concat_qdfs(dfs + [1])

        with self.assertRaises(TypeError):
            concat_qdfs(1)


if __name__ == "__main__":
    unittest.main()
