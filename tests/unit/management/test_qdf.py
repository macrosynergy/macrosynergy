import unittest
import numpy as np
import pandas as pd
import random
from typing import List
import warnings
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.compat import PD_2_0_OR_LATER
from macrosynergy.management.constants import JPMAQS_METRICS
from macrosynergy.management.utils import (
    get_cid,
    get_xcat,
    qdf_to_ticker_df,
)
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
    update_categories,
    qdf_to_wide_df,
    add_ticker_column,
    rename_xcats,
    add_nan_series,
    drop_nan_series,
    qdf_from_timeseries,
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
            start="2000-01-01",
            end="2020-01-01",
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
        qdf = qdf_from_timeseries(ts, cid="A", xcat="X")
        self.assertTrue(check_is_categorical(qdf))

        # test with ticker
        qdf = qdf_from_timeseries(ts, ticker="A_X")
        self.assertTrue(check_is_categorical(qdf))

        # test with ticker
        qdf = qdf_from_timeseries(ts, ticker="A_X")
        self.assertTrue(check_is_categorical(qdf))

        # test with non datetime index

        with self.assertRaises(ValueError):
            ts_copy = ts.copy()
            ts_copy.index = ts_copy.index.astype(str)
            qdf_from_timeseries(ts_copy, ticker="A_X")

        # test with non pd.Series
        with self.assertRaises(TypeError):
            qdf_from_timeseries(ts.values, ticker="A_X")

        # test with non string metric
        with self.assertRaises(TypeError):
            qdf_from_timeseries(ts, ticker="A_X", metric=1)

        # test with no cid or xcat
        with self.assertRaises(ValueError):
            qdf_from_timeseries(ts, ticker="A_X", cid="A", xcat="X")

        with self.assertRaises(ValueError):
            qdf_from_timeseries(ts, ticker="A_X", cid="A")

        with self.assertRaises(ValueError):
            qdf_from_timeseries(ts, xcat="X")

    def test_add_ticker_column(self):
        test_df: pd.DataFrame = make_test_df()
        test_df = add_ticker_column(test_df)

        tickers = test_df["cid"] + "_" + test_df["xcat"]
        self.assertTrue((test_df["ticker"] == tickers).all())

        # test with non QuantamentalDataFrame
        with self.assertRaises(TypeError):
            add_ticker_column(test_df.rename(columns={"cid": "xid"}))

        ## test with categorical df
        test_df = make_test_df()
        test_df = QuantamentalDataFrame(test_df)

        tickers = test_df["cid"].astype(str) + "_" + test_df["xcat"].astype(str)

        test_df = add_ticker_column(test_df)

        self.assertTrue((test_df["ticker"] == tickers).all())


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
        # orig_xcats = set(qdf["xcat"].unique())

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
        test_df: pd.DataFrame = make_test_df(tickers=tickers)

        qdf = QuantamentalDataFrame(test_df)
        # orig_cids = set(qdf["cid"].unique())
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

        self.assertTrue((expected_df.values == expected_df.values).all())
        self.assertTrue(
            qdf_to_string_index(new_df).equals(qdf_to_string_index(expected_df))
        )

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

        self.assertTrue((rdf.values == expected_df.values).all())
        self.assertTrue(
            qdf_to_string_index(rdf).equals(qdf_to_string_index(expected_df))
        )

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

        self.assertTrue((new_df.values == expected_df.values).all())
        self.assertTrue(
            qdf_to_string_index(new_df).equals(qdf_to_string_index(expected_df))
        )

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

    def test_reduce_df_out_all(self):
        # tickers = helper_random_tickers(20)
        test_cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        test_xcats = ["FX", "IR", "EQ", "CDS", "PPP"]
        test_df: pd.DataFrame = make_test_df(cids=test_cids, xcats=test_xcats)

        qdf = QuantamentalDataFrame(test_df)
        orig_cids = set(qdf["cid"].unique())
        orig_xcats = set(qdf["xcat"].unique())

        sel_cids = random.sample(list(orig_cids), 2)
        sel_xcats = random.sample(list(orig_xcats), 2)

        new_df, out_xcats, out_cids = reduce_df(
            qdf, cids=sel_cids, xcats=sel_xcats, out_all=True
        )

        found_cids = new_df["cid"].unique()
        found_xcats = new_df["xcat"].unique()

        # should be the same as the original
        self.assertTrue(set(found_cids) == set(sel_cids))
        self.assertTrue(set(found_xcats) == set(sel_xcats))

        self.assertTrue(set(out_cids) == set(sel_cids))
        self.assertTrue(set(out_xcats) == set(sel_xcats))


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
        self.assertRaises(TypeError, update_df, dfa, dfa, "string")

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

    def test_update_categories(self):

        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcatsa = ["FX", "IR", "EQ"]
        xcatsb = ["FX", "PPP", "IR"]

        cargs = dict(cids=cids, metrics=JPMAQS_METRICS, style="linear")

        dfa = make_test_df(xcats=xcatsa, **cargs)
        dfb = make_test_df(xcats=xcatsb, **cargs)

        # select USD_FX and EUR_FX and make their values -5000
        for im, mt in enumerate(JPMAQS_METRICS):
            dfb.loc[(dfb["cid"] == "USD") & (dfb["xcat"] == "FX"), mt] = -1e3 * (im + 1)
            dfb.loc[(dfb["cid"] == "EUR") & (dfb["xcat"] == "FX"), mt] = -1e3 * (im + 1)

        qdfa = QuantamentalDataFrame(dfa)
        qdfb = QuantamentalDataFrame(dfb)

        new_df = update_df(qdfa, qdfb, xcat_replace=True)

        # check that the values of USD_FX and EUR_FX are -5000
        for im, mt in enumerate(JPMAQS_METRICS):
            found_values = new_df.loc[
                (new_df["cid"].isin(["USD", "EUR"])) & (new_df["xcat"] == "FX"), mt
            ].unique()

            self.assertEqual(len(found_values), 1)
            self.assertEqual(found_values[0], -1e3 * (im + 1))

        direct_call = update_categories(qdfa, qdfb)

        self.assertTrue(new_df.equals(direct_call))

    def test_update_categories_no_overlap(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcatsa = ["FX", "IR", "EQ"]
        xcatsb = ["CDS", "PPP", "GDP"]

        cargs = dict(cids=cids, metrics=JPMAQS_METRICS, style="linear")

        dfa = make_test_df(xcats=xcatsa, **cargs)
        dfb = make_test_df(xcats=xcatsb, **cargs)

        qdfa = QuantamentalDataFrame(dfa)
        qdfb = QuantamentalDataFrame(dfb)

        new_df = update_df(qdfa, qdfb, xcat_replace=True)

        expected_df = update_df(qdfa, qdfb)

        self.assertTrue(new_df.equals(expected_df))

        self.assertRaises(TypeError, update_categories, 1, dfa)
        self.assertRaises(TypeError, update_categories, dfa, 1)


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

        for idf, _ in enumerate(split_dfs):
            cid = split_dfs[idf]["cid"].unique().tolist()[0]
            xcat = split_dfs[idf]["xcat"].unique().tolist()[0]
            _metric = split_dfs[idf].columns[-1]
            for t, m in drop_pairs:
                if t == f"{cid}_{xcat}" and m == _metric:
                    split_dfs.pop(idf)
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


class TestRenameXCATs(unittest.TestCase):
    def test_rename_xcats_xcat_map(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]

        dfo = make_test_df(cids=cids, xcats=xcats)
        qdf = QuantamentalDataFrame(dfo.copy())

        xcat_map = {xc: xc[::-1] for xc in xcats}

        new_df = rename_xcats(qdf, xcat_map)

        found_xcats = new_df["xcat"].unique()
        for fxc in found_xcats:
            self.assertTrue(fxc[::-1] in xcats)
            self.assertEqual(xcats.count(fxc[::-1]), 1)

        self.assertTrue(new_df.equals(qdf))
        dfo_copy = dfo.copy()
        dfo_copy["xcat"] = dfo_copy["xcat"].map(xcat_map)
        self.assertTrue(new_df.equals(QuantamentalDataFrame(dfo_copy)))

    def test_rename_xcats_postfix(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]

        dfo = make_test_df(cids=cids, xcats=xcats)
        qdf = QuantamentalDataFrame(dfo.copy())

        postfix = "_new"
        new_df = rename_xcats(qdf, postfix=postfix)

        found_xcats: List[str] = new_df["xcat"].unique().tolist()
        for fxc in found_xcats:
            self.assertTrue(fxc.endswith(postfix))
            self.assertEqual(xcats.count(fxc[: -len(postfix)]), 1)

        self.assertTrue(new_df.equals(qdf))
        dfo_copy = dfo.copy()
        dfo_copy["xcat"] = dfo_copy["xcat"].astype(str) + postfix
        self.assertTrue(new_df.equals(QuantamentalDataFrame(dfo_copy)))

        qdf = QuantamentalDataFrame(dfo.copy())

        postfix = "_new"
        sel_xcats = ["EQ", "PPP"]

        new_df = rename_xcats(qdf, select_xcats=sel_xcats, postfix=postfix)

        found_xcats = new_df["xcat"].unique().tolist()

        for fxc in found_xcats:
            if fxc not in xcats:
                self.assertTrue(fxc.endswith(postfix))
                self.assertTrue(fxc[: -len(postfix)] in sel_xcats)

        unchanged_dfo = dfo[~dfo["xcat"].isin(sel_xcats)]
        unchanged_new = new_df[~new_df["xcat"].str.endswith(postfix)]

        self.assertTrue(unchanged_dfo.eq(unchanged_new).all().all())

    def test_rename_xcats_prefix(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]

        dfo = make_test_df(cids=cids, xcats=xcats)
        qdf = QuantamentalDataFrame(dfo.copy())

        prefix = "new_"
        new_df = rename_xcats(qdf, prefix=prefix)

        found_xcats: List[str] = new_df["xcat"].unique().tolist()
        for fxc in found_xcats:
            self.assertTrue(fxc.startswith(prefix))
            self.assertEqual(xcats.count(fxc[len(prefix) :]), 1)

        self.assertTrue(new_df.equals(qdf))
        dfo_copy = dfo.copy()
        dfo_copy["xcat"] = prefix + dfo_copy["xcat"].astype(str)
        self.assertTrue(new_df.equals(QuantamentalDataFrame(dfo_copy)))

        qdf = QuantamentalDataFrame(dfo.copy())

        prefix = "new_"
        sel_xcats = ["FX", "IR"]

        new_df = rename_xcats(qdf, select_xcats=sel_xcats, prefix=prefix)

        found_xcats = new_df["xcat"].unique().tolist()

        for fxc in found_xcats:
            if fxc not in xcats:
                self.assertTrue(fxc.startswith(prefix))
                self.assertTrue(fxc[len(prefix) :] in sel_xcats)

        unchanged_dfo = dfo[~dfo["xcat"].isin(sel_xcats)]
        unchanged_new = new_df[~new_df["xcat"].str.startswith(prefix)]

        self.assertTrue(unchanged_dfo.eq(unchanged_new).all().all())

    def test_rename_xcats_name_all(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]

        dfo = make_test_df(cids=cids, xcats=xcats)
        qdf = QuantamentalDataFrame(dfo.copy())

        new_name = "new_name"
        new_df = rename_xcats(qdf, name_all=new_name)

        found_xcats: List[str] = new_df["xcat"].unique().tolist()
        for fxc in found_xcats:
            self.assertEqual(fxc, new_name)

        self.assertTrue(new_df.equals(qdf))
        dfo_copy = dfo.copy()
        dfo_copy["xcat"] = new_name
        self.assertTrue(new_df.equals(QuantamentalDataFrame(dfo_copy)))

        qdf = QuantamentalDataFrame(dfo.copy())

        new_name = "new_name"
        sel_xcats = ["CDS", "FX"]

        new_df = rename_xcats(qdf, select_xcats=sel_xcats, name_all=new_name)

        found_xcats = new_df["xcat"].unique().tolist()

        for fxc in found_xcats:
            if fxc not in xcats:
                self.assertEqual(fxc, new_name)

        unchanged_dfo = dfo[~dfo["xcat"].isin(sel_xcats)]
        unchanged_new = new_df[new_df["xcat"] != new_name]
        # sort these by unchanged_dfo.columns.tolist()
        # unchanged_dfo = unchanged_dfo.sort_values(
        #     by=QuantamentalDataFrame.IndexColsSortOrder
        # )
        # unchanged_new = unchanged_new.sort_values(
        #     by=QuantamentalDataFrame.IndexColsSortOrder
        # )
        self.assertTrue(unchanged_dfo.eq(unchanged_new).all().all())

    def test_rename_xcats_fmt_string(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]

        dfo = make_test_df(cids=cids, xcats=xcats)
        qdf = QuantamentalDataFrame(dfo.copy())

        fmt_string = "new_{}_name"
        new_df = rename_xcats(qdf, fmt_string=fmt_string)

        found_xcats: List[str] = new_df["xcat"].unique().tolist()

        for fxc in found_xcats:
            self.assertTrue(fxc.startswith("new_"))
            self.assertTrue(fxc.endswith("_name"))
            self.assertTrue(fxc[4:-5] in xcats)

        self.assertTrue(new_df.equals(qdf))

        # now only with a selected xcat
        qdf = QuantamentalDataFrame(dfo.copy())

        fmt_string = "new_{}_name"
        sel_xcats = ["CDS", "FX"]

        new_df = rename_xcats(qdf, select_xcats=sel_xcats, fmt_string=fmt_string)

        found_xcats = new_df["xcat"].unique().tolist()

        for fxc in found_xcats:
            if fxc not in xcats:
                self.assertTrue(fxc.startswith("new_"))
                self.assertTrue(fxc.endswith("_name"))
                og_xcat = fxc[4:-5]
                self.assertTrue(og_xcat in sel_xcats)

        unchanged_dfo = dfo[~dfo["xcat"].isin(sel_xcats)]
        unchanged_new = new_df[~new_df["xcat"].str.startswith("new_")]

        self.assertTrue(unchanged_dfo.eq(unchanged_new).all().all())

    def test_rename_xcats_errors(self):
        cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        xcats = ["FX", "IR", "EQ", "CDS", "PPP"]
        dfo = make_test_df(cids=cids, xcats=xcats)
        qdf = QuantamentalDataFrame(dfo)

        # provide non-qdf
        with self.assertRaises(TypeError):
            rename_xcats(dfo.rename(columns={"cid": "xid"}), name_all="new_name")

        # provide xcat_map and select_xcats
        with self.assertRaises(ValueError):
            rename_xcats(
                qdf,
                xcat_map={"FX": "new_name"},
                select_xcats=["IR"],
                name_all="new_name",
            )

        # provide non-dict xcat_map
        with self.assertRaises(TypeError):
            rename_xcats(qdf, xcat_map=1)

        # provide non-str dict xcat_map
        with self.assertRaises(TypeError):
            rename_xcats(qdf, xcat_map={"FX": 1})

        # provide select_xcats with postfix and prefix
        with self.assertRaises(ValueError):
            rename_xcats(qdf, select_xcats=["FX"], prefix="new_", postfix="_new")

        # provide select_xcats with name_all and fmt_string
        with self.assertRaises(ValueError):
            rename_xcats(
                qdf, select_xcats=["FX"], name_all="new_name", fmt_string="new_{}_name"
            )

        # provide with non-compatible fmt string
        with self.assertRaises(ValueError):
            rename_xcats(qdf, fmt_string="new_name")

        with self.assertRaises(ValueError):
            rename_xcats(qdf, fmt_string="new_{}_name_{}")

        with self.assertRaises(ValueError):
            rename_xcats(qdf, fmt_string="new_{xcat}_name_")


class TestQDFClass(unittest.TestCase):
    def setUp(self) -> None:
        self.tickers = helper_random_tickers(50)
        self.test_df: pd.DataFrame = make_test_df(
            tickers=self.tickers, metrics=JPMAQS_METRICS
        )

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

    def test_qdf_minimal(self):
        test_df = self.test_df.copy()
        qdf = QuantamentalDataFrame(test_df.copy())
        in_dtypes = test_df.dtypes.to_dict()

        qdf_copy = qdf.copy()
        qdf_copy[["cid", "xcat"]] = qdf_copy[["cid", "xcat"]].astype("object")

        self.assertTrue(qdf_copy.equals(test_df))

        self.assertTrue(qdf.dtypes.to_dict()["cid"].name == "category")
        self.assertTrue(qdf.dtypes.to_dict()["xcat"].name == "category")

        for col in qdf.columns:
            if col not in ["cid", "xcat"]:
                self.assertTrue(qdf[col].dtype == in_dtypes[col])

    def test_qdf_str_cols(self):
        tickers = helper_random_tickers(10)
        test_df: pd.DataFrame = make_test_df(tickers=tickers, metrics=JPMAQS_METRICS)

        qdf = QuantamentalDataFrame(test_df, categorical=False)

        self.assertTrue(qdf.equals(test_df))
        self.assertTrue(qdf.dtypes.to_dict()["cid"].name == "object")
        self.assertTrue(qdf.dtypes.to_dict()["xcat"].name == "object")

    def test_qdf_is_categorical(self):
        test_df = self.test_df.copy()
        qdf = QuantamentalDataFrame(test_df)
        self.assertTrue(qdf.is_categorical())
        self.assertEqual(check_is_categorical(qdf), qdf.is_categorical())

        qdf = QuantamentalDataFrame(test_df, categorical=False)
        self.assertFalse(qdf.is_categorical())
        self.assertEqual(check_is_categorical(qdf), qdf.is_categorical())

    def test_qdf_to_categorical(self):
        test_df = self.test_df.copy()
        qdf = QuantamentalDataFrame(test_df, categorical=False)
        self.assertFalse(qdf.is_categorical())
        qdf = qdf.to_categorical()
        self.assertTrue(qdf.is_categorical())

        test_df = self.test_df.copy()
        qdf = QuantamentalDataFrame(test_df)
        self.assertTrue(qdf.is_categorical())
        qdf = qdf.to_categorical()
        self.assertTrue(qdf.is_categorical())

    def test_qdf_to_string_type(self):
        test_df = self.test_df.copy()
        qdf = QuantamentalDataFrame(test_df, categorical=False)
        self.assertFalse(qdf.is_categorical())
        qdf = qdf.to_string_type()
        self.assertFalse(qdf.is_categorical())

        test_df = self.test_df.copy()
        qdf = QuantamentalDataFrame(test_df)
        self.assertTrue(qdf.is_categorical())
        qdf = qdf.to_string_type()
        self.assertFalse(qdf.is_categorical())

    def test_qdf_to_original_dtypes(self):
        test_df = self.test_df.copy()

        qdf = QuantamentalDataFrame(test_df)
        self.assertTrue(qdf.is_categorical())
        qdf = qdf.to_original_dtypes()
        self.assertFalse(qdf.is_categorical())
        self.assertTrue(qdf.equals(test_df))

        # now try with an already categorical df
        new_test_df = qdf_to_categorical(self.test_df.copy())
        qdf = QuantamentalDataFrame(new_test_df)
        qdf = qdf.to_original_dtypes()
        self.assertTrue(qdf.is_categorical())

    def test_qdf_list_tickers(self):
        test_df = self.test_df.copy()
        qdf = QuantamentalDataFrame(test_df)
        self.assertEqual(set(qdf.list_tickers()), set(self.tickers))

        self.assertTrue(pd.DataFrame(qdf).eq(self.test_df).all().all())

    def test_add_ticker_column(self):
        test_df = self.test_df.copy()
        qdf = QuantamentalDataFrame(test_df)
        orig_cols = qdf.columns

        qdf = qdf.add_ticker_column()

        self.assertTrue("ticker" in qdf.columns)

        self.assertTrue(qdf[orig_cols].eq(test_df).all().all())

        expc_column = qdf["cid"].astype(str) + "_" + qdf["xcat"].astype(str)
        self.assertTrue((qdf["ticker"] == expc_column).all())

    def test_drop_ticker_column(self):
        test_df = self.test_df.copy()
        qdf = QuantamentalDataFrame(test_df)
        orig_cols = qdf.columns

        qdf = qdf.add_ticker_column()
        assert "ticker" in qdf.columns, "Test logic failed"

        qdf = qdf.drop_ticker_column()
        self.assertTrue("ticker" not in qdf.columns)
        self.assertTrue(qdf[orig_cols].eq(test_df).all().all())

        new_qdf = QuantamentalDataFrame(test_df)

        with self.assertRaises(ValueError):
            new_qdf.drop_ticker_column()

    def test_reduce_df(self):
        test_cids = ["USD", "EUR", "GBP", "JPY", "AUD"]
        test_xcats = ["FX", "IR", "EQ", "CDS", "PPP"]

        test_df: pd.DataFrame = make_test_df(
            cids=test_cids, xcats=test_xcats, metrics=JPMAQS_METRICS
        )

        qdf = QuantamentalDataFrame(test_df)
        red_cids = random.sample(test_cids, 3)
        red_xcats = random.sample(test_xcats, 3)

        new_df = qdf.reduce_df(cids=red_cids, xcats=red_xcats)

        expc_df = reduce_df(qdf, cids=red_cids, xcats=red_xcats)

        if PD_2_0_OR_LATER:
            self.assertTrue(new_df.eq(expc_df).all().all())
        else:
            self.assertTrue(
                pd.DataFrame(new_df).eq(pd.DataFrame(expc_df)).all().all(),
                msg="Test failing on pandas < 2.0",
            )
        # test reduce_df out_all

        qdf = QuantamentalDataFrame(test_df)
        red_cids = random.sample(test_cids, 3)
        red_xcats = random.sample(test_xcats, 3)

        new_df, out_xcats, out_cids = qdf.reduce_df(
            cids=red_cids, xcats=red_xcats, out_all=True
        )

        expc_df, expc_out_xcats, expc_out_cids = reduce_df(
            qdf, cids=red_cids, xcats=red_xcats, out_all=True
        )

        if PD_2_0_OR_LATER:
            self.assertTrue(new_df.eq(expc_df).all().all())
        else:
            self.assertTrue(
                pd.DataFrame(new_df).eq(pd.DataFrame(expc_df)).all().all(),
                msg="Test failing on pandas < 2.0",
            )

        self.assertEqual(out_cids, expc_out_cids)
        self.assertEqual(out_xcats, expc_out_xcats)

    def test_reduce_df_by_ticker(self):

        qdf = QuantamentalDataFrame(self.test_df)
        sel_tickers = random.sample(self.tickers, 10)

        new_df = qdf.reduce_df_by_ticker(tickers=sel_tickers)

        expc_df = reduce_df_by_ticker(df=qdf, tickers=sel_tickers)

        if PD_2_0_OR_LATER:
            self.assertTrue(new_df.eq(expc_df).all().all())
        else:
            self.assertTrue(
                pd.DataFrame(new_df).eq(pd.DataFrame(expc_df)).all().all(),
                msg="Test failing on pandas < 2.0",
            )

    def test_apply_blacklist(self):
        blacklist = {
            "GBP": [pd.Timestamp("2010-06-06"), pd.Timestamp("2010-07-23")],
            "AUD": [pd.Timestamp("2000-01-01"), pd.Timestamp("2025-01-01")],
        }

        qdf = QuantamentalDataFrame(self.test_df)

        new_df = qdf.apply_blacklist(blacklist=blacklist)

        expc_df = apply_blacklist(df=qdf, blacklist=blacklist)

        if PD_2_0_OR_LATER:
            self.assertTrue(new_df.eq(expc_df).all().all())
        else:
            self.assertTrue(
                pd.DataFrame(new_df).eq(pd.DataFrame(expc_df)).all().all(),
                msg="Test failing on pandas < 2.0",
            )

        # test with reduce_df

        qdf = QuantamentalDataFrame(self.test_df)

        new_df = qdf.reduce_df(blacklist=blacklist)

        expc_df = reduce_df(qdf, blacklist=blacklist)

        if PD_2_0_OR_LATER:
            self.assertTrue(new_df.eq(expc_df).all().all())
        else:
            self.assertTrue(
                pd.DataFrame(new_df).eq(pd.DataFrame(expc_df)).all().all(),
                msg="Test failing on pandas < 2.0",
            )

    def test_update_df(self):
        tickers = helper_random_tickers(20)
        dfa = make_test_df(tickers=tickers[:10])
        dfb = make_test_df(tickers=tickers[10:])

        qdfa = QuantamentalDataFrame(dfa)
        qdfb = QuantamentalDataFrame(dfb)

        new_df = qdfa.update_df(qdfb)

        expected_df = update_df(qdfa, qdfb)

        if PD_2_0_OR_LATER:
            self.assertTrue(new_df.eq(expected_df).all().all())
        else:
            self.assertTrue(
                pd.DataFrame(new_df).eq(pd.DataFrame(expected_df)).all().all(),
                msg="Test failing on pandas < 2.0",
            )

    def test_add_nan_series(self):
        test_df = self.test_df.copy()

        qdf = QuantamentalDataFrame(test_df)
        curr_min_date, curr_max_date = qdf["real_date"].min(), qdf["real_date"].max()

        new_cid, new_xcat = "NCID", "NXCAT"
        new_ticker = f"{new_cid}_{new_xcat}"

        qdf = qdf.add_nan_series(ticker=new_ticker)

        test_added = qdf.loc[(qdf["cid"] == new_cid) & (qdf["xcat"] == new_xcat)]

        # test that the min and max dates are the same
        self.assertEqual(test_added["real_date"].min(), curr_min_date)
        self.assertEqual(test_added["real_date"].max(), curr_max_date)

        # test that all the metrics are nan
        self.assertTrue(test_added[JPMAQS_METRICS].isna().all().all())

    def test_drop_nan_series(self):
        test_df = self.test_df.copy()

        qdf = QuantamentalDataFrame(test_df)
        qdf_copy = qdf.copy()
        rand_xcats = ["RAND1", "RAND2", "RAND3"]
        rand_cids = ["RAND4", "RAND5", "RAND6"]
        rand_tickers = [f"{rc}_{rx}" for rc in rand_cids for rx in rand_xcats]

        for rt in rand_tickers:
            qdf = qdf.add_nan_series(ticker=rt)

        with warnings.catch_warnings(record=True) as wcatch:
            qdf = qdf.drop_nan_series()
            wlist = [w for w in wcatch if issubclass(w.category, UserWarning)]
            self.assertEqual(len(wlist), len(rand_tickers))

            ticker_dict = {t: False for t in rand_tickers}
            for w in wlist:
                for _t in rand_tickers:
                    if _t in str(w.message):
                        ticker_dict[_t] = True

            self.assertTrue(all(ticker_dict.values()))

        tickers = qdf.list_tickers()
        self.assertTrue(set(tickers).isdisjoint(set(rand_tickers)))

        # sort both dfs by the index cols
        qdf = qdf.sort_values(QuantamentalDataFrame.IndexColsSortOrder).reset_index(
            drop=True
        )
        qdf_copy = qdf_copy.sort_values(
            QuantamentalDataFrame.IndexColsSortOrder
        ).reset_index(drop=True)

        qdf[["cid", "xcat"]] = qdf[["cid", "xcat"]].astype("object")
        qdf_copy[["cid", "xcat"]] = qdf_copy[["cid", "xcat"]].astype("object")

        self.assertTrue(qdf.equals(qdf_copy))

    def test_rename_xcats(self):
        test_df = self.test_df.copy()

        qdf = QuantamentalDataFrame(test_df)

        xcat_map = {xc: xc[::-1] for xc in qdf["xcat"].unique()}

        new_df = qdf.rename_xcats(xcat_map=xcat_map)

        expc_df = rename_xcats(qdf, xcat_map=xcat_map)

        if PD_2_0_OR_LATER:
            self.assertTrue(new_df.eq(expc_df).all().all())
        else:
            self.assertTrue(
                pd.DataFrame(new_df).eq(pd.DataFrame(expc_df)).all().all(),
                msg="Test failing on pandas < 2.0",
            )

    def test_qdf_to_wide(self):
        test_df = self.test_df.copy()

        qdf = QuantamentalDataFrame(test_df)

        wide_df = qdf.to_wide()

        expc_df = qdf_to_ticker_df(test_df)

        # check the columns are the same
        self.assertTrue(wide_df.columns.equals(expc_df.columns))

        for col in wide_df.columns:
            self.assertTrue(wide_df[col].equals(expc_df[col]))

        # test with categorical df
        test_df = self.test_df.copy()
        expc_df = qdf_to_ticker_df(test_df)

        _qdf = QuantamentalDataFrame(test_df, _initialized_as_categorical=True)
        qdf = QuantamentalDataFrame(_qdf)

        wide_df = qdf.to_wide()

        # assert that the wide sdf
        isinstance(wide_df.columns, pd.CategoricalIndex)

        # check that the set of columns is the same
        self.assertTrue(set(wide_df.columns) == set(expc_df.columns))

        for col in wide_df.columns:
            self.assertTrue(wide_df[col].equals(expc_df[col]))


class TestQDFClassInit(unittest.TestCase):
    def setUp(self) -> None:
        self.tickers = helper_random_tickers(10)
        self.test_df: pd.DataFrame = make_test_df(tickers=self.tickers)

    def test_qdf_errors(self):
        with self.assertRaises(TypeError):
            QuantamentalDataFrame(1)

        with self.assertRaises(ValueError):
            QuantamentalDataFrame(pd.DataFrame())

    def test_qdf_init_as_categorical_flag(self):
        cat_df = pd.DataFrame(QuantamentalDataFrame(self.test_df.copy()))
        qdf = QuantamentalDataFrame(cat_df, _initialized_as_categorical=True)

        new_qdf = QuantamentalDataFrame(qdf)
        self.assertTrue(new_qdf.InitializedAsCategorical)

        with self.assertRaises(TypeError):
            QuantamentalDataFrame(self.test_df, _initialized_as_categorical="banana")

        qdf = QuantamentalDataFrame(self.test_df)
        self.assertFalse(qdf.InitializedAsCategorical)

    def test_qdf_col_sort_order(self):
        test_df = self.test_df.copy()
        test_df = test_df[test_df.columns.tolist()[::-1]]

        qdf = QuantamentalDataFrame(test_df)
        self.assertTrue(qdf.columns.tolist() == get_col_sort_order(test_df))

    def test_qdf_is_cat_check(self):
        cat_df = pd.DataFrame(QuantamentalDataFrame(self.test_df.copy()))
        qdf = QuantamentalDataFrame(cat_df)

        self.assertTrue(qdf.is_categorical())
        self.assertTrue(qdf.InitializedAsCategorical)

    def test_qdf_init_real_date_strs(self):
        # test that it still works when the real_date is a column of strings
        test_df = self.test_df.copy()
        test_df["real_date"] = test_df["real_date"].astype(str)

        qdf = QuantamentalDataFrame(test_df)
        self.assertTrue(qdf["real_date"].dtype == "datetime64[ns]")


class TestQDFInitializationMethods(unittest.TestCase):

    def test_qdf_from_timeseries(self):
        ts = pd.Series(
            np.random.randn(100), index=pd.bdate_range("2020-01-01", periods=100)
        )

        # test with cid and xcat
        cid, xcat = "A", "X"
        ticker = f"{cid}_{xcat}"
        qdf = qdf_from_timeseries(ts, ticker=ticker)

        self.assertTrue(isinstance(qdf, QuantamentalDataFrame))

        self.assertTrue(qdf["cid"].unique().tolist() == ["A"])
        self.assertTrue(qdf["xcat"].unique().tolist() == ["X"])

        self.assertTrue(qdf["real_date"].eq(ts.index).all())

        # test same output

        qdf_a = qdf_from_timeseries(ts, ticker=ticker, metric="value")
        qdf_b = QuantamentalDataFrame.from_timeseries(ts, ticker=ticker, metric="value")

        self.assertTrue(qdf_a.equals(qdf_b))

    def test_from_long_df(self):

        df = pd.DataFrame(
            {
                "real_date": pd.bdate_range("2020-01-01", periods=100),
                "value": np.random.randn(100),
            }
        )

        qdf = QuantamentalDataFrame.from_long_df(df, cid="A", xcat="X")

        self.assertTrue(isinstance(qdf, QuantamentalDataFrame))
        self.assertTrue(qdf["cid"].unique().tolist() == ["A"])
        self.assertTrue(qdf["xcat"].unique().tolist() == ["X"])

        self.assertTrue(qdf["real_date"].eq(df["real_date"]).all())

        # test ticker args work the same way
        new_qdf = QuantamentalDataFrame.from_long_df(df, ticker="A_X")
        self.assertTrue(new_qdf.equals(qdf))

        # test works when cid col is missing
        test_df = make_test_df()
        no_cid = test_df.drop(columns=["cid"])
        qdf = QuantamentalDataFrame.from_long_df(no_cid, cid="A")

        with self.assertRaises(ValueError):
            QuantamentalDataFrame.from_long_df(no_cid)

        # test works when xcat col is missing
        no_xcat = test_df.drop(columns=["xcat"])
        qdf = QuantamentalDataFrame.from_long_df(no_xcat, xcat="X")

        with self.assertRaises(ValueError):
            QuantamentalDataFrame.from_long_df(no_xcat)

        # test works when real_date is named differently
        diff_real_date = test_df.rename(columns={"real_date": "date"})

        qdf = QuantamentalDataFrame.from_long_df(
            diff_real_date, cid="A", xcat="X", real_date_column="date"
        )

        with self.assertRaises(ValueError):
            QuantamentalDataFrame.from_long_df(diff_real_date, cid="A", xcat="X")

        # test works when value is named differently
        diff_value = test_df.rename(columns={"value": "val"})
        qdf = QuantamentalDataFrame.from_long_df(
            diff_value, cid="A", xcat="X", value_column="val"
        )

        with self.assertRaises(ValueError):
            QuantamentalDataFrame.from_long_df(diff_value, cid="A", xcat="X")

        # test a error states
        test_df = make_test_df()
        test_df = test_df.iloc[0:0]

        with self.assertRaises(ValueError):
            QuantamentalDataFrame.from_long_df(test_df, cid="A", xcat="X")

        # test with ticker & cid+xcat
        test_df = make_test_df()

        with self.assertRaises(ValueError):
            QuantamentalDataFrame.from_long_df(test_df, cid="A", xcat="X", ticker="A_X")

    def test_from_qdf_list(self):
        tickers = helper_random_tickers(50)
        df_list = [make_test_df(tickers=tickers[i : i + 10]) for i in range(0, 50, 10)]

        qdf = QuantamentalDataFrame.from_qdf_list(df_list)
        expc_df = concat_qdfs(df_list)

        self.assertTrue(isinstance(qdf, QuantamentalDataFrame))
        self.assertTrue(qdf.equals(expc_df))

        # test non categorical return
        qdf = QuantamentalDataFrame.from_qdf_list(df_list, categorical=False)
        expc_df = QuantamentalDataFrame(expc_df, categorical=False)

        self.assertTrue(qdf.equals(expc_df))

        # test errors
        with self.assertRaises(ValueError):
            QuantamentalDataFrame.from_qdf_list([])

        with self.assertRaises(TypeError):
            QuantamentalDataFrame.from_qdf_list([1])

    def test_qdf_from_wide_df(self):
        tickers = helper_random_tickers(50)
        df = make_test_df(tickers=tickers)

        wdf = qdf_to_ticker_df(df)
        qdf: QuantamentalDataFrame = QuantamentalDataFrame.from_wide(wdf)

        self.assertTrue(isinstance(qdf, QuantamentalDataFrame))

        qdf["cid"] = qdf["cid"].astype("object")
        qdf["xcat"] = qdf["xcat"].astype("object")

        # sort the dfs by the index cols
        qdf = qdf.sort_values(QuantamentalDataFrame.IndexColsSortOrder).reset_index(
            drop=True
        )
        df = df.sort_values(QuantamentalDataFrame.IndexColsSortOrder).reset_index(
            drop=True
        )

        self.assertTrue(qdf.equals(df))

        # test with non-categorical
        qdf = QuantamentalDataFrame.from_wide(wdf, categorical=False)

        qdf = qdf.sort_values(QuantamentalDataFrame.IndexColsSortOrder).reset_index(
            drop=True
        )

        df = df.sort_values(QuantamentalDataFrame.IndexColsSortOrder).reset_index(
            drop=True
        )

        self.assertTrue(qdf.equals(df))

        # test errors
        with self.assertRaises(TypeError):
            QuantamentalDataFrame.from_wide(1)

        with self.assertRaises(TypeError):
            df = qdf_to_ticker_df(make_test_df())
            QuantamentalDataFrame.from_wide(df, value_column=1)

        with self.assertRaises(ValueError):
            df = qdf_to_ticker_df(make_test_df())
            df.index = df.index.astype(str)
            QuantamentalDataFrame.from_wide(df)

        with self.assertRaises(ValueError):
            df = qdf_to_ticker_df(make_test_df())
            df.columns = df.columns.astype(str)
            df.columns = [c.replace("_", "/") for c in df.columns]
            QuantamentalDataFrame.from_wide(df)

    def test_qdf_create_empty_df(self):

        test_cid, test_xcat = "A", "X"
        test_ticker = f"{test_cid}_{test_xcat}"
        test_metrics = JPMAQS_METRICS[:-1]
        test_start_date, test_end_date = "2020-01-01", "2020-01-10"
        qdf = QuantamentalDataFrame.create_empty_df(
            cid=test_cid,
            xcat=test_xcat,
            metrics=test_metrics,
            start=test_start_date,
            end=test_end_date,
        )

        expc_df = create_empty_categorical_qdf(
            cid=test_cid,
            xcat=test_xcat,
            metrics=test_metrics,
            start=test_start_date,
            end=test_end_date,
        )

        self.assertTrue(qdf.equals(expc_df))

        # test with ticker and date range args
        dt_range = pd.bdate_range(test_start_date, test_end_date)
        qdf = QuantamentalDataFrame.create_empty_df(
            ticker=test_ticker,
            metrics=test_metrics,
            date_range=dt_range,
        )
        self.assertTrue(qdf.equals(expc_df))

        # test with categorical = False
        qdf = QuantamentalDataFrame.create_empty_df(
            cid=test_cid,
            xcat=test_xcat,
            metrics=test_metrics,
            start=test_start_date,
            end=test_end_date,
            categorical=False,
        )

        expc_df = create_empty_categorical_qdf(
            cid=test_cid,
            xcat=test_xcat,
            metrics=test_metrics,
            start=test_start_date,
            end=test_end_date,
        )

        expc_df = qdf_to_string_index(expc_df)

        self.assertTrue(qdf.equals(expc_df))


if __name__ == "__main__":
    unittest.main()
