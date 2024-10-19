import unittest
import numpy as np
import pandas as pd
import random
import string
from typing import Any, List
import warnings
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.constants import JPMAQS_METRICS
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


@staticmethod
def helper_random_tickers(n: int = 10) -> List[str]:
    def rstr(n: int = 3, m: int = 5) -> str:
        return "".join(random.sample(string.ascii_uppercase, random.randint(n, m)))

    cids = [rstr() for _ in range(n)]
    xcats = [rstr(4, 5) for _ in range(n)]

    all_tickers = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
    return random.sample(all_tickers, n)


@staticmethod
def helper_split_df_by_metrics(
    df: QuantamentalDataFrame,
) -> List[QuantamentalDataFrame]:
    return [
        df[QuantamentalDataFrame.IndexCols + [m]].reset_index(drop=True)
        for m in (set(df.columns) - set(QuantamentalDataFrame.IndexCols))
    ]


@staticmethod
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
        test_df: pd.DataFrame = make_test_df(tickers=tickers, cids=None, xcats=None)

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
        test_qdf = make_test_df(
            tickers=helper_random_tickers(10), cids=None, xcats=None
        )
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
        test_df: pd.DataFrame = make_test_df(tickers=tickers, cids=None, xcats=None)
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
        test_df: pd.DataFrame = make_test_df(tickers=tickers, cids=None, xcats=None)

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
