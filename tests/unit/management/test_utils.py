import unittest
import pandas as pd
import warnings
import datetime
import numpy as np
from parameterized import parameterized

from typing import List, Tuple, Dict, Union, Set, Any
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.download.jpmaqs import timeseries_to_qdf, construct_expressions
from macrosynergy.management.utils import (
    get_cid,
    get_xcat,
    downsample_df_on_real_date,
    concat_single_metric_qdfs,
    get_dict_max_depth,
    rec_search_dict,
    is_valid_iso_date,
    convert_dq_to_iso,
    convert_iso_to_dq,
    form_full_url,
    generate_random_date,
    common_cids,
    drop_nan_series,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
    get_eops,
    get_sops,
    _map_to_business_day_frequency,
    apply_slip,
    merge_categories,
    estimate_release_frequency,
    Timer,
    rotate_cid_xcat,
)
from macrosynergy.management.utils.df_utils import _long_to_wide, _wide_to_long
from macrosynergy.management.constants import FREQUENCY_MAP
from macrosynergy.management.utils.math import expanding_mean_with_nan
from macrosynergy.compat import PD_NEW_DATE_FREQ
from tests.simulate import make_qdf
from tests.unit.download.mock_helpers import mock_request_wrapper


class TestFunctions(unittest.TestCase):
    def test_get_dict_max_depth(self):
        d: dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        self.assertEqual(get_dict_max_depth(d), 3)

        d: int = 10
        self.assertEqual(get_dict_max_depth(d), 0)

        dx: dict = {0: "a"}
        for i in range(1, 100):
            dx = {i: dx}
        self.assertEqual(get_dict_max_depth(dx), 100)

    def test_rec_search_dict(self):
        d: dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        self.assertEqual(rec_search_dict(d, "e"), 3)

        self.assertEqual(rec_search_dict("Some string", "KEY"), None)

        dx: dict = {0: "a"}
        for i in range(1, 100):
            dx = {i: dx}
        self.assertEqual(rec_search_dict(dx, 0), "a")

        d = {"12": 1, "123": 2, "234": 3, "1256": 4, "246": 5}
        self.assertEqual(rec_search_dict(d=d, key="25", match_substring=True), 4)
        self.assertEqual(rec_search_dict(d=d, key="99", match_substring=True), None)

        d = {"12": 1, "123": [2], "234": "3", "1256": 4.0, "246": {"a": 1}}
        for k in d.keys():
            self.assertEqual(
                rec_search_dict(
                    d=d, key=k, match_substring=True, match_type=type(d[k])
                ),
                d[k],
            )

    def test_is_valid_iso_date(self):
        good_case: str = "2020-01-01"

        bad_cases: List[str] = [
            "2020-01-01T00:00:00",
            "2020-01-01T00:00:00Z",
            "12-900-56",
            "foo",
            "bar",
            "Ze-ld-a",
        ]

        type_error_cases: List[Any] = [
            1,
            2.0,
            {},
            [],
            None,
            True,
            False,
            ["2020-01-01", "2020-01-02", "2020-01-03"],
            {"a": "2020-01-01", "b": "2020-01-02", "c": "2020-01-03"},
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-01 00:00:00"),
            1 + 2j,
        ]

        self.assertTrue(is_valid_iso_date(good_case))
        for bcase in bad_cases:
            self.assertFalse(is_valid_iso_date(bcase))

        for tcase in type_error_cases:
            self.assertRaises(TypeError, is_valid_iso_date, tcase)

    def test_convert_dq_to_iso(self):
        d: List[Tuple[str, str]] = [
            ("20200101", "2020-01-01"),
            ("20110101", "2011-01-01"),
            ("20120101", "2012-01-01"),
            ("20130101", "2013-01-01"),
            ("20140101", "2014-01-01"),
            ("20150101", "2015-01-01"),
        ]

        for i in range(len(d)):
            self.assertEqual(convert_dq_to_iso(d[i][0]), d[i][1])

        # generate 20 random dates
        dts = [generate_random_date() for i in range(20)]
        for dt in dts:
            self.assertEqual(convert_iso_to_dq(dt), dt.replace("-", ""))
            self.assertEqual(convert_dq_to_iso(dt.replace("-", "")), dt)

    def test_convert_iso_to_dq(self):
        d: List[Tuple[str, str]] = [
            ("2020-01-01", "20200101"),
            ("2011-01-01", "20110101"),
            ("2012-01-01", "20120101"),
            ("2013-01-01", "20130101"),
            ("2014-01-01", "20140101"),
            ("2015-01-01", "20150101"),
        ]

        for i in range(len(d)):
            self.assertEqual(convert_iso_to_dq(d[i][0]), d[i][1])

        # generate 20 random dates
        dts: List[str] = [generate_random_date() for i in range(20)]
        for dt in dts:
            self.assertEqual(convert_iso_to_dq(dt), dt.replace("-", ""))

    def test_form_full_url(self):
        url: str = "https://www.google.com"
        params: Dict[str, Union[str, int]] = {"a": 1, "b": 2}
        # html safe url
        self.assertEqual(form_full_url(url, params), "https://www.google.com?a=1&b=2")
        # url = http://foo.bar
        # params = {'banana': '!@#$%^&*()_+{}|:"<>?'}', 'apple': '><?>?<?><?'}
        url = "http://foo.bar"
        params = {"banana": """!@#$%^&*()_+{}|:"<>?}""", "apple": "><?>?<?><?"}

        exp_out: str = (
            "http://foo.bar?banana=%21%40%23%24%25%5E%26%"
            "2A%28%29_%2B%7B%7D%7C%3A%22%3C%3E%3F%7D&apple"
            "=%3E%3C%3F%3E%3F%3C%3F%3E%3C%3F"
        )
        self.assertEqual(form_full_url(url, params), exp_out)

    def test_generate_random_date(self):
        # get 20 random dates
        strts: List[str] = [generate_random_date() for i in range(10)]
        ends: List[str] = [generate_random_date() for i in range(10)]

        for st, ed in zip(strts, ends):
            stD = datetime.datetime.strptime(st, "%Y-%m-%d")
            edD = datetime.datetime.strptime(ed, "%Y-%m-%d")
            if stD > edD:
                stD, edD = edD, stD
                # generate random date between st and ed
                rd = generate_random_date(stD, edD)
                rdD = datetime.datetime.strptime(rd, "%Y-%m-%d")
                self.assertTrue(stD <= rdD <= edD)

        strts = ["2020-01-01", "2023-05-02", "2021-12-31"]
        ends = ["2020-01-03", "2023-05-03", "2021-12-31"]
        endst = ["2020-01-02", "2023-05-03", "2021-12-31"]
        for st, ed, edt in zip(strts, ends, endst):
            stD = datetime.datetime.strptime(st, "%Y-%m-%d")
            edD = datetime.datetime.strptime(ed, "%Y-%m-%d")
            edtD = datetime.datetime.strptime(edt, "%Y-%m-%d")
            if stD > edD:
                stD, edD = edD, stD
            # generate random date between st and ed
            rd = generate_random_date(stD, edD)
            rdD = datetime.datetime.strptime(rd, "%Y-%m-%d")
            self.assertTrue(stD <= rdD <= edD)
            self.assertTrue(rdD <= edtD)

            # when generate st=ed, rd=ed
            rdD = generate_random_date(edD, edD)
            self.assertEqual(rdD, edD.strftime("%Y-%m-%d"))

    def test_common_cids(self):
        cids: List[str] = ["AUD", "USD", "GBP", "EUR", "CAD"]
        xcats: List[str] = ["FXXR", "IR", "EQXR", "CRY", "FXFW"]
        df: pd.DataFrame = make_test_df(cids=cids, xcats=xcats)

        # check normal case
        com_cids: List[str] = common_cids(df=df, xcats=xcats)
        self.assertEqual(set(com_cids), set(cids))

        self.assertRaises(TypeError, common_cids, df=1, xcats=xcats)
        self.assertRaises(TypeError, common_cids, df=df, xcats=1)
        self.assertRaises(ValueError, common_cids, df=df, xcats=["xcat"])
        self.assertRaises(ValueError, common_cids, df=df, xcats=["apple", "banana"])
        self.assertRaises(TypeError, common_cids, df=df, xcats=[1, 2, 3])

        # test A
        dfA: pd.DataFrame = df.copy()
        dfA = dfA[~((dfA["cid"] == "USD") & (dfA["xcat"].isin(["FXXR", "IR"])))]
        dfA = dfA[~((dfA["cid"] == "CAD") & (dfA["xcat"].isin(["FXXR", "IR"])))]

        com_cids: List[str] = common_cids(df=dfA, xcats=xcats)
        self.assertEqual(set(com_cids), set(["AUD", "GBP", "EUR"]))

        comm_cids: List[str] = common_cids(df=dfA, xcats=["FXXR", "IR"])
        self.assertEqual(
            set(comm_cids),
            set(
                [
                    "AUD",
                    "GBP",
                    "EUR",
                ]
            ),
        )

        # test B
        dfB: pd.DataFrame = df.copy()
        # remove "FXXR", "IR", "EQXR" from "AUD", "USD"
        dfB = dfB[~((dfB["cid"] == "AUD") & (dfB["xcat"].isin(["FXXR", "IR", "EQXR"])))]
        dfB = dfB[~((dfB["cid"] == "USD") & (dfB["xcat"].isin(["FXXR", "IR", "EQXR"])))]

        com_cids: List[str] = common_cids(df=dfB, xcats=xcats)
        self.assertEqual(set(com_cids), set(["GBP", "EUR", "CAD"]))

        comm_cids: List[str] = common_cids(df=dfB, xcats=["FXFW", "CRY"])
        self.assertEqual(set(comm_cids), set(cids))

    def test_drop_nan_series(self):
        cids: List[str] = ["AUD", "USD", "GBP", "EUR", "CAD"]
        xcats: List[str] = ["FXXR", "IR", "EQXR", "CRY", "FXFW"]
        df_orig: pd.DataFrame = make_test_df(cids=cids, xcats=xcats)

        # set warnings to error. test if a warning is raised in the obvious "clean" case
        warnings.simplefilter("error")
        for boolx in [True, False]:
            try:
                dfx: pd.DataFrame = drop_nan_series(df=df_orig, raise_warning=boolx)
                self.assertTrue(dfx.equals(df_orig))
            except Warning as w:
                self.fail("Warning raised unexpectedly")

        df_test: pd.DataFrame = df_orig.copy()

        with warnings.catch_warnings(record=True) as w:
            # all warnings will raise errors, this raises deprecated warning
            # with older pd/np versions. the ignore filter is to suppress it
            warnings.simplefilter("ignore")
            df_test.loc[
                (df_test["cid"] == "AUD") & (df_test["xcat"].isin(["FXXR", "IR"])),
                "value",
            ] = pd.NA

        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as w:
            dfx: pd.DataFrame = drop_nan_series(df=df_test, raise_warning=True)
            self.assertEqual(len(w), 2)
            for ww in w:
                self.assertTrue(issubclass(ww.category, UserWarning))

            found_tickers: Set = set(dfx["cid"] + "_" + dfx["xcat"])
            if any([x in found_tickers for x in ["AUD_FXXR", "AUD_IR"]]):
                self.fail("NaN series not dropped")

        with warnings.catch_warnings(record=True) as w:
            dfx: pd.DataFrame = drop_nan_series(df=df_test, raise_warning=False)
            self.assertEqual(len(w), 0)
            found_tickers: Set = set(dfx["cid"] + "_" + dfx["xcat"])
            if any([x in found_tickers for x in ["AUD_FXXR", "AUD_IR"]]):
                self.fail("NaN series not dropped")

        self.assertRaises(TypeError, drop_nan_series, df=1, raise_warning=True)
        self.assertRaises(TypeError, drop_nan_series, df=df_test, raise_warning=1)

        df_test_q = df_test.dropna(how="any")
        with warnings.catch_warnings(record=True) as w:
            dfx: pd.DataFrame = drop_nan_series(df=df_test_q, raise_warning=True)
            dfu: pd.DataFrame = drop_nan_series(df=df_test_q, raise_warning=False)
            self.assertEqual(len(w), 0)
            self.assertTrue(dfx.equals(df_test_q))
            self.assertTrue(dfu.equals(df_test_q))

        df_test: pd.DataFrame = df_orig.copy()
        bcids: List[str] = [
            "AUD",
            "USD",
            "GBP",
        ]
        bxcats: List[str] = [
            "FXXR",
            "IR",
            "EQXR",
        ]
        df_test.loc[
            (df_test["cid"].isin(bcids)) & (df_test["xcat"].isin(bxcats)), "value"
        ] = pd.NA
        with warnings.catch_warnings(record=True) as w:
            dfx: pd.DataFrame = drop_nan_series(df=df_test, raise_warning=True)
            self.assertEqual(len(w), 9)
            for ww in w:
                self.assertTrue(issubclass(ww.category, UserWarning))

            found_tickers: Set = set(dfx["cid"] + "_" + dfx["xcat"])
            if any(
                [
                    x in found_tickers
                    for x in [f"{cid}_{xcat}" for cid in bcids for xcat in bxcats]
                ]
            ):
                self.fail("NaN series not dropped")

            dfu: pd.DataFrame = drop_nan_series(df=df_test, raise_warning=False)
            self.assertEqual(len(w), 9)

        # test with non existant column name
        df_test: pd.DataFrame = df_orig.copy()
        with self.assertRaises(ValueError):
            drop_nan_series(df=df_test.rename(columns={"value": "val"}))

        warnings.resetwarnings()

    def test_qdf_to_ticker_df(self):
        cids: List[str] = ["AUD", "USD", "GBP", "EUR", "CAD"]
        xcats: List[str] = ["FXXR", "IR", "EQXR", "CRY", "FXFW"]
        start_date: str = "2010-01-01"
        end_date: str = "2020-01-31"
        bdtrange: pd.DatetimeIndex = pd.bdate_range(start_date, end_date)

        tickers: List[str] = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]

        test_df: pd.DataFrame = make_test_df(
            cids=cids, xcats=xcats, start=start_date, end=end_date
        )

        # test case 0 - does it work?
        rdf: pd.DataFrame = qdf_to_ticker_df(df=test_df.copy())

        # test 0.1  - are all tickers present as columns?
        self.assertEqual(set(rdf.columns), set(tickers))

        # test 0.2 - is the df indexed by "real_date"?
        self.assertTrue(rdf.index.name == "real_date")

        # test 0.3 - are all dates present?
        self.assertEqual(set(rdf.index), set(bdtrange))

        # test 0.4 - are the axes unnamed - should be

        self.assertTrue(rdf.columns.name is None)

        # test case 1 - type error on df
        self.assertRaises(TypeError, qdf_to_ticker_df, df=1)

        # test case 2 - value error, thrown by standardise_df
        bad_df: pd.DataFrame = test_df.copy()
        # rename xcats to xkats
        bad_df.rename(columns={"xcat": "xkat"}, inplace=True)
        self.assertRaises(TypeError, qdf_to_ticker_df, df=bad_df)

        # test case 3 - NO side effects
        # test case 3.1 - df is not modified
        df: pd.DataFrame = test_df.copy()
        rdf: pd.DataFrame = qdf_to_ticker_df(df=df)
        self.assertTrue(df.equals(test_df))

        #  test case 4 - bad args
        self.assertRaises(TypeError, qdf_to_ticker_df, df=test_df, value_column=1)

        with warnings.catch_warnings(record=True) as w:
            qdf_to_ticker_df(df=test_df, value_column="banana")
            self.assertEqual(len(w), 1)
            # check that 'Value column specified in `value_column`' is in the warning
            self.assertTrue(
                "Value column specified in `value_column`" in str(w[0].message)
            )

        # test case 5 - with categorical qdf
        qdf = QuantamentalDataFrame(test_df.copy())
        rdf = qdf_to_ticker_df(qdf)

        expc_df = qdf_to_ticker_df(test_df)

        self.assertTrue(rdf.equals(expc_df))

    def test_ticker_df_to_qdf(self):
        cids: List[str] = ["AUD", "USD", "GBP", "EUR", "CAD"]
        xcats: List[str] = ["FXXR", "IR", "EQXR", "CRY", "FXFW"]
        tickers: List[str] = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
        start_date: str = "2010-01-01"
        end_date: str = "2020-01-31"
        bdtrange: pd.DatetimeIndex = pd.bdate_range(start_date, end_date)
        test_df: pd.DataFrame = qdf_to_ticker_df(
            df=make_test_df(cids=cids, xcats=xcats, start=start_date, end=end_date)
        )

        # test case 0 - does it work?
        rdf: pd.DataFrame = ticker_df_to_qdf(df=test_df.copy())

        # test 0.1  - are all tickers successfully converted to cid and xcat?
        found_tickers: List[str] = (
            (rdf["cid"] + "_" + rdf["xcat"]).drop_duplicates().tolist()
        )
        self.assertEqual(set(found_tickers), set(tickers))

        # test 0.2 - is the df unindexed?
        self.assertTrue(rdf.index.name is None)

        # test 0.3 - are all dates present?
        self.assertEqual(set(rdf["real_date"]), set(bdtrange))

        # test 0.4 - isinstance
        self.assertTrue(isinstance(rdf, QuantamentalDataFrame))
        self.assertTrue(
            set(rdf.columns), set(QuantamentalDataFrame.IndexCols + ["value"])
        )
        # test case 1 - type error on df
        self.assertRaises(TypeError, ticker_df_to_qdf, df=1)

        # test case 2 - there should only be cid, xcat, real_date, value columns in rdf
        self.assertEqual(set(rdf.columns), set(["cid", "xcat", "real_date", "value"]))

        # test case 3 - NO side effects
        # test case 3.1 - df is not modified
        df: pd.DataFrame = test_df.copy()
        rdf: pd.DataFrame = ticker_df_to_qdf(df=df)
        self.assertTrue(df.equals(test_df))

        # pass an integer as a metric
        df: pd.DataFrame = test_df.copy()
        with self.assertRaises(TypeError):
            ticker_df_to_qdf(df=df, metric=1)

        # test the metric acctually works
        df: pd.DataFrame = test_df.copy()
        rdf: pd.DataFrame = ticker_df_to_qdf(df=df, metric="hibiscus")
        self.assertTrue(
            set(rdf.columns) - set(QuantamentalDataFrame.IndexCols) == set(["hibiscus"])
        )

    def test_concat_single_metric_qdfs(self):

        with self.assertRaises(TypeError):
            concat_single_metric_qdfs("")

        with self.assertRaises(ValueError):
            concat_single_metric_qdfs([], errors=True)

        cids: List[str] = ["GBP", "EUR", "CAD"]
        xcats: List[str] = ["FXXR_NSA", "EQXR_NSA"]
        metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        expressions = construct_expressions(cids=cids, xcats=xcats, metrics=metrics)
        dicts_lists = mock_request_wrapper(
            dq_expressions=expressions,
            start_date="2019-01-01",
            end_date="2019-01-31",
        )

        qdfs = [timeseries_to_qdf(dicts) for dicts in dicts_lists]
        combined = concat_single_metric_qdfs(qdfs)
        self.assertIsInstance(combined, QuantamentalDataFrame)

        bad_qdfs = qdfs.copy()
        bad_qdfs[0]["newcol"] = 1
        with self.assertRaises(ValueError):
            concat_single_metric_qdfs(bad_qdfs)

        ## different metrics

        test_exprs = [
            "DB(JPMAQS,GBP_FXXR_NSA,value)",
            "DB(JPMAQS,GBP_EQXR_NSA,grading)",
        ]
        dicts_lists = mock_request_wrapper(
            dq_expressions=test_exprs,
            start_date="2019-01-01",
            end_date="2019-01-31",
        )

        qdfs = [timeseries_to_qdf(dicts) for dicts in dicts_lists]
        combined = concat_single_metric_qdfs(qdfs)
        self.assertIsInstance(combined, QuantamentalDataFrame)
        self.assertEqual(set(combined["xcat"].unique()), set(["FXXR_NSA", "EQXR_NSA"]))
        self.assertEqual(set(combined["cid"].unique()), set(["GBP"]))
        metrics = list(set(combined.columns) - set(QuantamentalDataFrame.IndexCols))
        self.assertEqual(set(metrics), set(["value", "grading"]))

        for xcat in ["FXXR_NSA", "EQXR_NSA"]:
            tgt = "grading" if xcat == "FXXR_NSA" else "value"
            dfn = combined[combined["xcat"] == xcat].copy()
            self.assertTrue(dfn[tgt].isna().all())

        with self.assertRaises(TypeError):
            concat_single_metric_qdfs(qdfs + [None], errors="raise")

        self.assertIsNone(concat_single_metric_qdfs([None, None], errors="ignore"))

    def test_get_cid(self):
        good_cases: List[Tuple[str, str]] = [
            ("AUD_FXXR", "AUD"),
            ("USD_IR", "USD"),
            ("GBP_EQXR_NSA", "GBP"),
            ("EUR_CRY_ABC", "EUR"),
            ("CAD_FXFW", "CAD"),
        ]
        # test good cases
        for case in good_cases:
            self.assertEqual(get_cid(case[0]), case[1])

        # test type errors
        for case in [1, 1.0, None, True, False]:
            self.assertRaises(TypeError, get_cid, case)

        # test value errors for empty lists
        for case in [[], (), {}, set()]:
            self.assertRaises(ValueError, get_cid, case)

        # test overloading for iterables
        cases: List[str] = [case[0] for case in good_cases]
        fresults: List[str] = get_cid(cases)
        self.assertTrue(isinstance(fresults, list))
        self.assertEqual(fresults, [case[1] for case in good_cases])

        # cast to pd.Series and test
        cases: pd.Series = pd.Series(cases)
        self.assertEqual(get_cid(cases), [case[1] for case in good_cases])

        # test value errors for bad tickers
        bad_cases: List[str] = ["AUD", "USD-IR-FXXR", ""]
        for case in bad_cases:
            self.assertRaises(ValueError, get_cid, case)

    def test_get_xcat(self):
        good_cases: List[Tuple[str, str]] = [
            ("AUD_FXXR", "FXXR"),
            ("USD_IR", "IR"),
            ("GBP_EQXR_NSA", "EQXR_NSA"),
            ("EUR_CRY_ABC", "CRY_ABC"),
            ("CAD_FXFW", "FXFW"),
        ]
        # test good cases
        for case in good_cases:
            self.assertEqual(get_xcat(case[0]), case[1])

        # test type errors
        for case in [1, 1.0, None, True, False]:
            self.assertRaises(TypeError, get_xcat, case)

        # test value errors for empty lists
        for case in [[], (), {}, set()]:
            self.assertRaises(ValueError, get_xcat, case)

        # test overloading for iterables
        cases: List[str] = [case[0] for case in good_cases]
        fresults: List[str] = get_xcat(cases)
        self.assertTrue(isinstance(fresults, list))
        self.assertEqual(fresults, [case[1] for case in good_cases])

        # cast to pd.Series and test
        cases: pd.Series = pd.Series(cases)
        self.assertEqual(get_xcat(cases), [case[1] for case in good_cases])

        # test value errors for bad tickers
        bad_cases: List[str] = ["AUD", "USD-IR-FXXR", ""]
        for case in bad_cases:
            self.assertRaises(ValueError, get_xcat, case)

    def test_downsample_df_on_real_date(self):
        test_cids: List[str] = ["USD"]  # ,  "EUR", "GBP"]
        test_xcats: List[str] = ["FX"]
        df: pd.DataFrame = make_test_df(
            cids=test_cids,
            xcats=test_xcats,
            style="any",
            start="2010-01-01",
            end="2010-12-31",
        )

        freq = "M"
        agg_method = "mean"

        downsampled_df: pd.DataFrame = downsample_df_on_real_date(
            df=df, groupby_columns=["cid", "xcat"], freq=freq, agg=agg_method
        )
        assert downsampled_df.shape[0] == 12

        with self.assertRaises(ValueError):
            downsample_df_on_real_date(
                df=df, groupby_columns=["xid", "xcat"], freq=freq, agg=agg_method
            )

    def test_downsample_df_on_real_date_multiple_xcats(self):
        test_cids: List[str] = ["USD", "EUR", "GBP"]
        test_xcats: List[str] = ["FX"]
        df: pd.DataFrame = make_test_df(
            cids=test_cids,
            xcats=test_xcats,
            style="any",
            start="2010-01-01",
            end="2010-12-31",
        )

        freq = "M"
        agg_method = "mean"

        downsampled_df: pd.DataFrame = downsample_df_on_real_date(
            df=df, groupby_columns=["cid", "xcat"], freq=freq, agg=agg_method
        )
        assert downsampled_df.shape[0] == 36

    def test_downsample_df_on_real_date_invalid_freq(self):
        test_cids: List[str] = ["USD"]
        test_xcats: List[str] = ["FX"]
        df: pd.DataFrame = make_test_df(
            cids=test_cids,
            xcats=test_xcats,
            style="any",
            start="2010-01-01",
            end="2010-12-31",
        )

        freq = 0
        agg_method = "mean"

        with self.assertRaises(TypeError):
            downsample_df_on_real_date(
                df=df, groupby_columns=["cid", "xcat"], freq=freq, agg=agg_method
            )

        freq = "INVALID_FREQ"
        agg_method = "mean"

        with self.assertRaises(ValueError):
            downsample_df_on_real_date(
                df=df, groupby_columns=["cid", "xcat"], freq=freq, agg=agg_method
            )

    def test_downsample_df_on_real_date_invalid_agg(self):
        test_cids: List[str] = ["USD"]
        test_xcats: List[str] = ["FX"]
        df: pd.DataFrame = make_test_df(
            cids=test_cids,
            xcats=test_xcats,
            style="any",
            start="2010-01-01",
            end="2010-12-31",
        )

        freq = "M"
        agg_method = 0

        with self.assertRaises(TypeError):
            downsample_df_on_real_date(
                df=df, groupby_columns=["cid", "xcat"], freq=freq, agg=agg_method
            )

        freq = "M"
        agg_method = "INVALID_AGG"

        with self.assertRaises(ValueError):
            downsample_df_on_real_date(
                df=df, groupby_columns=["cid", "xcat"], freq=freq, agg=agg_method
            )

    def test_rolling_mean(self):
        self.__dict__["cids"] = ["AUD", "CAD", "GBP", "NZD"]
        self.__dict__["xcats"] = ["XR", "CRY", "GROWTH", "INFL"]
        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
        df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
        df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
        df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]

        df_xcats = pd.DataFrame(
            index=self.xcats,
            columns=[
                "earliest",
                "latest",
                "mean_add",
                "sd_mult",
                "ar_coef",
                "back_coef",
            ],
        )

        df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
        df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
        df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__["dfd"] = dfd

        dfd_xr = dfd[dfd["xcat"] == "XR"]
        self.__dict__["dfd_xr"] = dfd_xr

        dfw = dfd_xr.pivot(index="real_date", columns="cid", values="value")
        self.__dict__["dfw"] = dfw
        no_rows = dfw.shape[0]

        self.__dict__["no_timestamps"] = no_rows

        ar_neutral = expanding_mean_with_nan(dfw=self.dfw)

        benchmark_pandas = [
            self.dfw.iloc[0 : (i + 1), :].stack().mean()
            for i in range(self.no_timestamps)
        ]

        self.assertTrue(len(ar_neutral) == len(benchmark_pandas))

        for i, elem in enumerate(ar_neutral):
            bm_elem = round(benchmark_pandas[i], 4)
            self.assertTrue(round(elem, 4) == bm_elem)

        bm_expanding = self.dfw.mean(axis=1)
        bm_expanding = bm_expanding.expanding(min_periods=1).mean()

        # Test on another category to confirm the logic.
        dfd_cry = self.dfd[self.dfd["xcat"] == "CRY"]
        dfw_cry = dfd_cry.pivot(index="real_date", columns="cid", values="value")

        ar_neutral = expanding_mean_with_nan(dfw=dfw_cry)
        benchmark_pandas_cry = [
            dfw_cry.iloc[0 : (i + 1), :].stack().mean()
            for i in range(self.no_timestamps)
        ]

        self.assertTrue(len(ar_neutral) == len(benchmark_pandas_cry))
        for i, elem in enumerate(ar_neutral):
            bm_elem_cry = round(benchmark_pandas_cry[i], 4)
            self.assertTrue(round(elem, 4) == bm_elem_cry)

    def test_get_eops(self):
        daterange1 = pd.bdate_range(start="2023-01-28", end="2023-02-02")
        test_case_1 = pd.DataFrame({"real_date": pd.Series(daterange1)})
        # NOTE: get_eops(freq=...) is case insensitive
        test_result_1 = get_eops(dates=test_case_1, freq="M")
        # expected results : 2023-01-31 (last of cycle), last index

        expc_vals = set([pd.Timestamp("2023-01-31"), daterange1[-1]])
        test_vals = set(test_result_1.tolist())
        self.assertEqual(expc_vals, test_vals)

        daterange2 = pd.bdate_range(start="2023-03-20", end="2023-07-10")
        test_case_2 = pd.DataFrame({"real_date": pd.Series(daterange2)})
        test_result_2 = get_eops(dates=test_case_2, freq="q")
        # expected results : 2023-03-31 (last of cycle), 2023-06-30 (last of cycle), last index

        expc_vals = set(
            [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-06-30"), daterange2[-1]]
        )
        test_vals = set(test_result_2.tolist())
        self.assertEqual(expc_vals, test_vals)

        daterange3 = pd.bdate_range(start="2000-01-01", end="2023-07-10")
        test_case_3 = pd.DataFrame({"real_date": pd.Series(daterange3)})
        test_result_3 = get_eops(dates=test_case_3, freq="m")

        # expc_vals = set(
        # print(test_case_3[test_result_3]["real_date"].values.tolist())
        expc_vals = []
        r_start_date = pd.Timestamp("2000-01-01")
        r_end_date = pd.Timestamp("2023-07-10")

        expc_no_cycles = (r_end_date.year - r_start_date.year) * 12 + (
            r_end_date.month - r_start_date.month
        )
        self.assertEqual(expc_no_cycles, len(test_result_3) - 1)
        # -1 as len(test_result_3) includes the last index
        set_test_result_3 = set(test_result_3)

        test_vals = [
            [pd.Timestamp("2023-01-31"), True],
            [pd.Timestamp("2016-02-29"), True],
            # [pd.Timestamp("2023-02-29"), False],
            # this timestamp doesn't exist and will raise an Exception
            [pd.Timestamp("2023-02-28"), True],
            [pd.Timestamp("2023-03-20"), False],
            [pd.Timestamp("2005-12-30"), True],
            [pd.Timestamp("2000-12-29"), True],
        ]
        for tval in test_vals:
            self.assertEqual(tval[0] in set_test_result_3, tval[1])

        test_start_date = "2000-01-01"
        test_end_date = "2001-10-31"

        test_result_4 = get_eops(
            start_date=test_start_date, end_date=test_end_date, freq="Q"
        )
        expected_rs_4 = [
            "2000-03-31",
            "2000-06-30",
            "2000-09-29",
            "2000-12-29",
            "2001-03-30",
            "2001-06-29",
            "2001-09-28",
            "2001-10-31",
        ]
        self.assertTrue(
            set(test_result_4) == set([pd.Timestamp(dx) for dx in expected_rs_4])
        )

        stdt5, enddt5 = "2023-01-01", "2024-01-01"
        daterange5 = pd.bdate_range(start=stdt5, end=enddt5)
        test_case_5 = pd.DataFrame({"real_date": pd.Series(daterange5)})
        test_result_5 = get_eops(dates=test_case_5, freq="D")

        expc_vals = set(daterange5.tolist())
        test_vals = set(test_result_5.tolist())
        self.assertEqual(expc_vals, test_vals)

        stdt5, enddt5 = enddt5, stdt5  # swap
        test_result_6 = get_eops(start_date=stdt5, end_date=enddt5, freq="D")
        self.assertEqual(expc_vals, set(test_result_6))

        with self.assertRaises(TypeError):
            get_eops(start_date=1, end_date="2024-01-01", freq="D")

        with self.assertRaises(ValueError):
            get_eops(start_date="0", end_date="2024-01-01", freq="D")

        with self.assertRaises(ValueError):
            get_eops(start_date="2023-01-01", end_date="2024-01-01", freq="X")

        with self.assertRaises(TypeError):
            get_eops(start_date="2023-01-01", freq=1)

        with self.assertRaises(ValueError):
            get_eops(start_date="2023-01-01", dates=test_case_5, freq="D")

        test_result_7 = get_eops(dates=test_case_5, freq="A")
        self.assertTrue(len(test_result_7) == 2)  # as there are 2 years

    def test_get_sops(self):
        daterange = pd.bdate_range(start="2023-01-01", end="2023-04-01")
        test_case = pd.DataFrame({"real_date": pd.Series(daterange)})
        test_result = get_sops(dates=test_case, freq="M")
        expected_dates = set(
            [
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-02-01"),
                pd.Timestamp("2023-03-01"),
            ]
        )
        result_dates = set(test_result.tolist())
        self.assertEqual(expected_dates, result_dates)

        daterange2 = pd.bdate_range(start="2023-01-01", end="2023-12-31")
        test_case_2 = pd.DataFrame({"real_date": pd.Series(daterange2)})
        test_result_2 = get_sops(dates=test_case_2, freq="Q")
        expected_dates_2 = set(
            [
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-04-03"),
                pd.Timestamp("2023-07-03"),
                pd.Timestamp("2023-10-02"),
            ]
        )
        result_dates_2 = set(test_result_2.tolist())
        self.assertEqual(expected_dates_2, result_dates_2)

        daterange3 = pd.bdate_range(start="2022-01-01", end="2023-01-01")
        test_case_3 = pd.DataFrame({"real_date": pd.Series(daterange3)})
        test_result_3 = get_sops(dates=test_case_3, freq="A")
        expected_dates_3 = set([pd.Timestamp("2022-01-03")])
        result_dates_3 = set(test_result_3.tolist())
        self.assertEqual(expected_dates_3, result_dates_3)

        daterange4 = pd.bdate_range(start="2023-03-25", end="2023-04-05")
        test_case_4 = pd.DataFrame({"real_date": pd.Series(daterange4)})
        test_result_4 = get_sops(dates=test_case_4, freq="D")
        expected_dates_4 = set(daterange4)
        result_dates_4 = set(test_result_4.tolist())
        self.assertEqual(expected_dates_4, result_dates_4)

        with self.assertRaises(TypeError):
            get_sops(start_date=1, end_date="2023-12-31", freq="D")

        with self.assertRaises(ValueError):
            get_sops(start_date="invalid-date", end_date="2023-12-31", freq="D")

        with self.assertRaises(ValueError):
            get_sops(start_date="2022-01-01", end_date="2023-01-01", freq="unknown")

        with self.assertRaises(TypeError):
            get_sops(dates="not-a-date-series", freq="M")

    def test_map_to_business_day_frequency(self):
        fm_copy = FREQUENCY_MAP.copy()
        if PD_NEW_DATE_FREQ:
            fm_copy["M"] = "BME"
            fm_copy["Q"] = "BQE"

        for k in fm_copy.keys():
            self.assertEqual(
                _map_to_business_day_frequency(k), fm_copy[k], f"Failed for {k}"
            )

        with self.assertRaises(ValueError):
            _map_to_business_day_frequency("X")

        with self.assertRaises(TypeError):
            _map_to_business_day_frequency(1)

        with self.assertRaises(ValueError):
            _map_to_business_day_frequency("D", valid_freqs=["W", "M"])

        with self.assertRaises(ValueError):
            _map_to_business_day_frequency("D", valid_freqs=["X", "Y", "Z"])

        # check reverse mapping
        for k, v in fm_copy.items():
            self.assertEqual(_map_to_business_day_frequency(v), v, f"Failed for {v}")

        with self.assertRaises(TypeError):
            _map_to_business_day_frequency("M", valid_freqs=1)

    def test_apply_slip(self):
        warnings.simplefilter("always")
        # pick 3 random cids
        sel_xcats: List[str] = ["XR", "CRY"]
        sel_cids: List[str] = ["AUD", "CAD", "GBP"]
        sel_dates: pd.DatetimeIndex = pd.bdate_range(
            start="2020-01-01", end="2020-02-01"
        )
        cids: List[str] = ["AUD", "CAD", "GBP", "NZD", "JPY", "CHF"]
        xcats: List[str] = ["XR", "CRY", "GROWTH", "INFL"]
        # reduce the dataframe to the selected cids and xcats
        test_df: pd.DataFrame = make_test_df(cids=cids, xcats=xcats)
        test_df = test_df[
            test_df["cid"].isin(sel_cids)
            & test_df["xcat"].isin(sel_xcats)
            & test_df["real_date"].isin(sel_dates)
        ].reset_index(drop=True)

        df: pd.DataFrame = test_df.copy()

        # Test Case 1

        # for every unique cid, xcat pair add a column "vx" which is just an integer 0→n ,
        # where n is the number of unique dates for that cid, xcat pair
        df["vx"] = (
            df.groupby(["cid", "xcat"])["real_date"].rank(method="dense").astype(int)
        )
        test_slip: int = 5
        # apply the slip method
        out_df = apply_slip(
            df=df,
            slip=test_slip,
            xcats=sel_xcats,
            cids=sel_cids,
            metrics=["value", "vx"],
        )

        # NOTE: casting df.vx to int as pandas casts it to float64
        self.assertEqual(int(df["vx"].max()) - test_slip, int(out_df["vx"].max()))

        for cid in sel_cids:
            for xcat in sel_xcats:
                inan_count = (
                    df[(df["cid"] == cid) & (df["xcat"] == xcat)]["vx"].isna().sum()
                )
                onan_count = (
                    out_df[(out_df["cid"] == cid) & (out_df["xcat"] == xcat)]["vx"]
                    .isna()
                    .sum()
                )
                self.assertEqual(inan_count, onan_count - test_slip)

        # Test Case 2 - slip is greater than the number of unique dates for a cid, xcat pair

        df: pd.DataFrame = test_df.copy()
        df["vx"] = (
            df.groupby(["cid", "xcat"])["real_date"].rank(method="dense").astype(int)
        )

        test_slip = int(max(df["vx"])) + 1

        out_df = apply_slip(
            df=df,
            slip=test_slip,
            xcats=sel_xcats,
            cids=sel_cids,
            metrics=["value", "vx"],
        )

        self.assertTrue(out_df["vx"].isna().all())
        self.assertTrue(out_df["value"].isna().all())

        out_df = apply_slip(
            df=df,
            slip=test_slip,
            xcats=sel_xcats,
            cids=sel_cids,
            metrics=["value"],
        )

        self.assertTrue((df["vx"] == out_df["vx"]).all())
        self.assertTrue(out_df["value"].isna().all())

        # case 3 - slip is negative
        df: pd.DataFrame = test_df.copy()

        with self.assertRaises(ValueError):
            apply_slip(
                df=df,
                slip=-1,
                xcats=sel_xcats,
                cids=sel_cids,
                metrics=["value"],
            )

        # case 4 - slip works with tickers
        sel_tickers = [f"{cid}_{xcat}" for cid in sel_cids[:-2] for xcat in sel_xcats]
        df: pd.DataFrame = test_df.copy()
        out_df = apply_slip(
            df=df,
            slip=2,
            tickers=sel_tickers,
            metrics=["value"],
        )
        out_tickers = list(set(out_df["cid"] + "_" + out_df["xcat"]))
        self.assertTrue(len(set(sel_tickers) - set(out_tickers)) == 0)

        # iteratively for each cid_xcat pair, check that only sel_tickers changed and the rest are the same
        for cid in sel_cids[:-2]:
            for xcat in sel_xcats:
                if f"{cid}_{xcat}" in sel_tickers:
                    eq_series = out_df[
                        (out_df["cid"] == cid) & (out_df["xcat"] == xcat)
                    ]["value"] == df[(df["cid"] == cid) & (df["xcat"] == xcat)][
                        "value"
                    ].shift(
                        2
                    )
                    self.assertTrue(
                        (
                            eq_series
                            ^ out_df[(out_df["cid"] == cid) & (out_df["xcat"] == xcat)][
                                "value"
                            ].isna()
                        ).all()
                    )
                else:
                    self.assertTrue(
                        (
                            out_df[(out_df["cid"] == cid) & (out_df["xcat"] == xcat)][
                                "value"
                            ]
                            == df[(df["cid"] == cid) & (df["xcat"] == xcat)]["value"]
                        ).all()
                    )

        # case 5 - check effect of extend_dates flag on the output
        df: pd.DataFrame = test_df.copy()

        test_slip = 1
        # get the last date in the dataframe
        last_date = df["real_date"].max()
        new_last_date = df["real_date"].max() + pd.offsets.BDay(test_slip)
        out_df = apply_slip(
            df=df,
            slip=test_slip,
            xcats=df["xcat"].unique().tolist(),
            cids=df["cid"].unique().tolist(),
            metrics=["value"],
            extend_dates=True,
        )

        valid_cid_xcat_pairs = df[["cid", "xcat"]].drop_duplicates()
        for cid, xcat in df[["cid", "xcat"]].drop_duplicates().values.tolist():
            ld_in = df[
                (df["cid"] == cid)
                & (df["xcat"] == xcat)
                & (df["real_date"] == last_date)
            ]

            ld_out = out_df[
                (out_df["cid"] == cid)
                & (out_df["xcat"] == xcat)
                & (out_df["real_date"] == new_last_date)
            ]
            self.assertTrue(
                (
                    ld_in[["cid", "xcat", "value"]].values
                    == ld_out[["cid", "xcat", "value"]].values
                ).all()
            )
            self.assertEqual(
                len(
                    pd.bdate_range(
                        ld_in["real_date"].tolist()[0],
                        ld_out["real_date"].tolist()[0],
                    )
                ),
                test_slip + 1,
            )

        # check that a value error is raised when cids and xcats are not in the dataframe
        with warnings.catch_warnings(record=True) as w:
            apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=["ac_dc"],
                metrics=["value"],
                raise_error=False,
            )
            apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=sel_cids,
                metrics=["value"],
                raise_error=False,
            )
            apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=sel_cids,
                metrics=["value"],
                raise_error=False,
            )
            efilter = (
                "Tickers targetted for applying slip are not present in the DataFrame."
            )
            wlist = [_w for _w in w if efilter in str(_w.message)]

            self.assertEqual(len(wlist), 3)
        warnings.resetwarnings()

        with self.assertRaises(ValueError):
            apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=["ac_dc"],
                metrics=["value"],
            )
        with self.assertRaises(ValueError):
            apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=sel_cids,
                metrics=["value"],
            )
        with self.assertRaises(ValueError):
            apply_slip(
                df=df,
                slip=2,
                xcats=["metallica"],
                cids=sel_cids,
                metrics=["value"],
            )

        with self.assertRaises(ValueError):
            apply_slip(
                df=df,
                slip=2,
                xcats=sel_xcats,
                cids=sel_cids,
                metrics=["random_metric"],
            )
        with self.assertRaises(TypeError):
            apply_slip(
                df=df,
                slip=2,
                xcats=sel_xcats,
                cids=sel_cids,
                metrics=[1],
            )

    def test_merge_categories(self):
        cids: List[str] = ["AUD", "CAD", "GBP", "NZD", "JPY", "CHF"]
        xcats: List[str] = ["XR", "CRY", "GROWTH", "INFL"]
        test_df: pd.DataFrame = make_test_df(cids=cids, xcats=xcats)

        # Ensure that test dataframe has differing values for XR and CRY on 2015-01-01
        test_df.loc[
            (test_df["cid"] == "AUD")
            & (test_df["xcat"] == "XR")
            & (test_df["real_date"] == "2015-01-01"),
            "value",
        ] = 1.0

        test_df.loc[
            (test_df["cid"] == "AUD")
            & (test_df["xcat"] == "CRY")
            & (test_df["real_date"] == "2015-01-01"),
            "value",
        ] = 2.0
        # Typings are correct

        with self.assertRaises(TypeError):
            merge_categories("AUD", 1)

        with self.assertRaises(TypeError):
            merge_categories(test_df, xcats=1, new_xcat="NEW_CAT")

        with self.assertRaises(TypeError):
            merge_categories(
                test_df, xcats=["XR", "CRY"], new_xcat="NEW_CAT", cids="AUD"
            )

        with self.assertRaises(TypeError):
            merge_categories(test_df, xcats=["XR", "CRY"], cids=["AUD"], new_xcat=1)

        # Check dataframe has correct format

        with self.assertRaises(TypeError):
            merge_categories(
                test_df.drop(columns=["value", "real_date"]),
                xcats=["XR", "CRY"],
                new_xcat="NEW_CAT",
                cids=["AUD"],
            )

        # Check values specified exist in Dataframe

        with self.assertRaises(ValueError):
            merge_categories(
                test_df,
                xcats=["XR", "CRY", "NOT_PRESENT", "INFL"],
                new_xcat="NEW_CAT",
                cids=["AUD"],
            )

        with self.assertRaises(ValueError):
            merge_categories(
                test_df, xcats=["XR", "CRY", "INFL"], new_xcat="NEW_CAT", cids=["NOPE"]
            )

        # Check that the new category is created

        new_xcat = "NEW_CAT"
        new_df = merge_categories(
            test_df, xcats=["XR", "CRY"], cids=["AUD"], new_xcat=new_xcat
        )
        self.assertTrue(new_xcat in new_df["xcat"].unique())

        # Check that the new category is equal to preference 1

        new_df = merge_categories(
            test_df, xcats=["XR", "CRY"], cids=["AUD"], new_xcat=new_xcat
        )
        new_df_values = new_df[new_df["xcat"] == new_xcat]["value"].reset_index(
            drop=True
        )

        test_df_values = test_df[(test_df["xcat"] == "XR") & (test_df["cid"] == "AUD")][
            "value"
        ].reset_index(drop=True)

        self.assertTrue(new_df_values.equals(test_df_values))

        # Check that the new category is equal to preference 2 if preference 1 does not exist

        mask = ~(
            (test_df["cid"] == "AUD")
            & (test_df["xcat"] == "XR")
            & (test_df["real_date"] == "2015-01-01")
        )
        df_filtered = test_df[mask].reset_index(drop=True)

        new_df = merge_categories(
            df_filtered, xcats=["XR", "CRY"], cids=["AUD"], new_xcat=new_xcat
        )

        new_df_value = new_df[
            (new_df["xcat"] == new_xcat)
            & (new_df["cid"] == "AUD")
            & (new_df["real_date"] == "2015-01-01")
        ]["value"].reset_index(drop=True)
        self.assertTrue(
            new_df_value.equals(
                test_df[
                    (test_df["cid"] == "AUD")
                    & (test_df["xcat"] == "CRY")
                    & (test_df["real_date"] == "2015-01-01")
                ]["value"].reset_index(drop=True)
            )
        )
        self.assertTrue(
            not new_df_value.equals(
                test_df[
                    (test_df["cid"] == "AUD")
                    & (test_df["xcat"] == "XR")
                    & (test_df["real_date"] == "2015-01-01")
                ]["value"].reset_index(drop=True)
            )
        )

        with self.assertRaises(TypeError):
            merge_categories(
                df=1, xcats=["XR", "CRY"], cids=["AUD"], new_xcat="NEW_CAT"
            )

        with self.assertRaises(TypeError):
            merge_categories(
                df=test_df, xcats=["XR", 1], cids=["AUD"], new_xcat="NEW_CAT"
            )

        with self.assertRaises(TypeError):
            merge_categories(
                df=test_df, xcats=["XR", "CRY"], cids=["AUD", 1], new_xcat="NEW_CAT"
            )


class TestMergeCategoriesHierarchy(unittest.TestCase):
    """Mirrors the extend_history test suite to ensure merge_categories covers the
    same hierarchy/backfill/start behavior."""

    @classmethod
    def setUpClass(cls):
        xcats = ["INFL1", "INFL2", "INFL3"]
        cids = ["GBP", "CAD"]
        end_date = pd.Timestamp.now()

        date_ranges = {
            "INFL1": pd.date_range(start="2015-01-01", end=end_date, freq="D"),
            "INFL2": pd.date_range(start="2010-01-01", end=end_date, freq="D"),
            "INFL3": pd.date_range(start="2005-01-01", end=end_date, freq="D"),
        }

        data = []
        for xcat, dates in date_ranges.items():
            for cid in cids:
                for date in dates:
                    value = int(xcat[-1])
                    data.append(
                        {"real_date": date, "xcat": xcat, "cid": cid, "value": value}
                    )

        cls.df = pd.DataFrame(data)

    def setUp(self):
        self.valid_args = {
            "df": self.df,
            "new_xcat": "TEST",
            "cids": ["GBP", "CAD"],
            "hierarchy": ["INFL1", "INFL2"],
            "backfill": True,
            "start": "1995-01-01",
        }

    def test_valid_args(self):
        try:
            merge_categories(**self.valid_args)
        except Exception as e:
            self.fail(f"merge_categories raised {e} unexpectedly")

    def test_invalid_args(self):
        type_error_args = {
            "df": 1,
            "new_xcat": 1,
            "cids": 1,
            "hierarchy": 1,
            "backfill": 1,
            "start": 1,
        }

        for key, value in type_error_args.items():
            with self.assertRaises(TypeError):
                invalid_args = self.valid_args.copy()
                invalid_args[key] = value
                merge_categories(**invalid_args)

    def test_backfill_start_args(self):
        with self.assertRaises(ValueError):
            invalid_args = self.valid_args.copy()
            invalid_args["backfill"] = True
            invalid_args["start"] = None
            merge_categories(**invalid_args)

        try:
            valid_args = self.valid_args.copy()
            valid_args["backfill"] = True
            valid_args["start"] = "1995-01-01"
            merge_categories(**valid_args)
        except Exception as e:
            self.fail(f"merge_categories raised {e} unexpectedly")

        try:
            valid_args = self.valid_args.copy()
            valid_args["backfill"] = False
            valid_args["start"] = None
            merge_categories(**valid_args)
        except Exception as e:
            self.fail(f"merge_categories raised {e} unexpectedly")

    def test_invalid_cids(self):
        with self.assertRaises(TypeError):
            invalid_args = self.valid_args.copy()
            invalid_args["cids"] = [1, 2, 3]
            merge_categories(**invalid_args)

        with self.assertRaises(ValueError):
            invalid_args = self.valid_args.copy()
            invalid_args["cids"] = ["bad_cid"]
            merge_categories(**invalid_args)

    def test_one_xcat(self):
        df = self.df
        new_xcat = "NEW_XCAT"
        cids = ["GBP", "CAD"]
        hierarchy = ["INFL1"]
        start = "1995-01-02"

        result_df = merge_categories(
            df,
            hierarchy=hierarchy,
            new_xcat=new_xcat,
            cids=cids,
            backfill=True,
            start=start,
        )

        self._check_result_df(result_df, new_xcat, cids, start)

        for cid in cids:
            cid_df = df[(df["cid"] == cid) & (df["xcat"] == hierarchy[0])]
            result_cid_df = result_df[
                (result_df["cid"] == cid) & (result_df["xcat"] == new_xcat)
            ]
            backfill_value = cid_df.loc[cid_df["real_date"].idxmin(), "value"]
            self.assertTrue(
                (
                    result_cid_df.loc[
                        result_cid_df["real_date"] < cid_df["real_date"].min(),
                        "value",
                    ]
                    == backfill_value
                ).all(),
                f"Backfill values for cid {cid} are not consistent.",
            )

    @parameterized.expand(
        [
            (["INFL1", "INFL2", "INFL3"],),
            (["INFL1", "INVALID", "INFL3"],),
        ]
    )
    def test_multiple_xcats(self, hierarchy):
        df = self.df
        new_xcat = "NEW_XCAT"
        cids = ["GBP", "CAD"]
        start = "1995-01-02"

        # 'INVALID' is not in the df; merge_categories raises on missing xcats,
        # so substitute it with INFL2 to keep the spirit of the test (mid-tier
        # category between INFL1 and INFL3).
        effective_hierarchy = [
            h if h in df["xcat"].unique() else "INFL2" for h in hierarchy
        ]

        result_df = merge_categories(
            df,
            hierarchy=effective_hierarchy,
            new_xcat=new_xcat,
            cids=cids,
            backfill=True,
            start=start,
        )

        self._check_result_df(result_df, new_xcat, cids, start)

        for cid in cids:
            cid_df3 = df[(df["cid"] == cid) & (df["xcat"] == effective_hierarchy[2])]
            result_cid_df = result_df[result_df["cid"] == cid]
            backfill_value = cid_df3.loc[cid_df3["real_date"].idxmin(), "value"]
            self.assertTrue(
                (
                    result_cid_df.loc[
                        result_cid_df["real_date"] < cid_df3["real_date"].min(),
                        "value",
                    ]
                    == backfill_value
                ).all(),
                f"Backfill values for cid {cid} are not consistent.",
            )

    def test_no_backfill(self):
        df = self.df
        new_xcat = "NEW_XCAT"
        cids = ["GBP", "CAD"]
        hierarchy = ["INFL1", "INFL2", "INFL3"]

        result_df = merge_categories(
            df, hierarchy=hierarchy, new_xcat=new_xcat, cids=cids, backfill=False
        )

        self._check_result_df(result_df, new_xcat, cids, start=None)

        self.assertTrue(
            set(result_df["real_date"].unique()) == set(df["real_date"].unique()),
            "The real_date column is not consistent with the original DataFrame.",
        )

        for cid in cids:
            cid_df1 = df[(df["cid"] == cid) & (df["xcat"] == hierarchy[0])]
            cid_df2 = df[(df["cid"] == cid) & (df["xcat"] == hierarchy[1])]
            cid_df3 = df[(df["cid"] == cid) & (df["xcat"] == hierarchy[2])]
            result_cid_df = result_df[result_df["cid"] == cid]

            backfill_value2 = cid_df2.loc[cid_df2["real_date"].idxmin(), "value"]
            backfill_value3 = cid_df3.loc[cid_df3["real_date"].idxmin(), "value"]
            self.assertTrue(
                (
                    result_cid_df.loc[
                        (result_cid_df["real_date"] < cid_df1["real_date"].min())
                        & (result_cid_df["real_date"] >= cid_df2["real_date"].min()),
                        "value",
                    ]
                    == backfill_value2
                ).all(),
                f"INFL2-tier values for cid {cid} are not consistent.",
            )
            self.assertTrue(
                (
                    result_cid_df.loc[
                        result_cid_df["real_date"] < cid_df2["real_date"].min(),
                        "value",
                    ]
                    == backfill_value3
                ).all(),
                f"INFL3-tier values for cid {cid} are not consistent.",
            )

    def test_missing_cid(self):
        # merge_categories raises on missing cids (vs extend_history which warns).
        df = self.df
        with self.assertRaises(ValueError):
            merge_categories(
                df,
                hierarchy=["INFL1", "INFL2", "INFL3"],
                new_xcat="NEW_XCAT",
                cids=["GBP", "CAD", "AUD"],
                backfill=True,
                start="1995-01-02",
            )

    def test_no_cids(self):
        df = self.df
        new_xcat = "NEW_XCAT"
        hierarchy = ["INFL1", "INFL2", "INFL3"]
        start = "1995-01-02"

        result_df = merge_categories(
            df,
            hierarchy=hierarchy,
            new_xcat=new_xcat,
            cids=None,
            backfill=True,
            start=start,
        )

        self._check_result_df(result_df, new_xcat, ["GBP", "CAD"], start)

    def test_backfill_skips_leading_nans(self):
        df = self.df.copy()
        cids = ["GBP", "CAD"]
        hierarchy = ["INFL1"]
        new_xcat = "NEW_XCAT"
        start = "1995-01-02"

        expected_value = int(hierarchy[0][-1])
        leading_dates = sorted(
            df.loc[df["xcat"] == hierarchy[0], "real_date"].unique()
        )[:5]
        df.loc[
            (df["xcat"] == hierarchy[0]) & (df["real_date"].isin(leading_dates)),
            "value",
        ] = np.nan
        first_valid_date = df.loc[
            (df["xcat"] == hierarchy[0]) & df["value"].notna(), "real_date"
        ].min()

        result_df = merge_categories(
            df,
            hierarchy=hierarchy,
            new_xcat=new_xcat,
            cids=cids,
            backfill=True,
            start=start,
        )

        self._check_result_df(result_df, new_xcat, cids, start)

        for cid in cids:
            backfilled = result_df.loc[
                (result_df["cid"] == cid) & (result_df["real_date"] < first_valid_date),
                "value",
            ]
            self.assertFalse(
                backfilled.isna().any(),
                f"Backfilled region for cid {cid} contains NaNs.",
            )
            self.assertTrue(
                (backfilled == expected_value).all(),
                f"Backfilled values for cid {cid} do not equal first valid observation.",
            )

    def _check_result_df(self, result_df, new_xcat, cids, start):
        expected_columns = {"real_date", "xcat", "cid", "value"}
        self.assertTrue(
            set(result_df.columns) == expected_columns,
            "Result DataFrame columns do not match expected columns.",
        )

        self.assertTrue(
            set(result_df["xcat"].unique()) == {new_xcat},
            "xcat column does not contain the expected value.",
        )

        self.assertTrue(
            set(result_df["cid"].unique()) == set(cids),
            "cid column does not contain the expected cids.",
        )

        if start is not None:
            min_date = pd.to_datetime(start)
            self.assertTrue(
                result_df["real_date"].min() == min_date,
                "The history is not extended to the specified start date.",
            )


class TestTimer(unittest.TestCase):
    def test_timer(self):
        t = Timer()
        self.assertIsInstance(t, Timer)

    def test_timer_str(self):
        t = Timer()
        self.assertIsInstance(str(t), str)

        self.assertIsInstance(f"{t:s}", str)

        self.assertIn(" seconds", str(t))

    def test_timer_repr(self):
        t = Timer()
        self.assertIsInstance(repr(t), str)

        self.assertIn(" seconds>", repr(t))

        self.assertIsInstance(f"{t!r}", str)

    def test_timer_float(self):
        t = Timer()
        self.assertIsInstance(float(t), float)

        self.assertIsInstance(f"{t:0.2f}", str)


class TestEstimateReleaseFrequency(unittest.TestCase):
    def setUp(self):
        # Simulated time series data for testing different frequencies using business day ranges
        # Daily frequency for 10 years using business day range

        start_date = "2010-01-01"
        end_date = "2020-01-01"
        freqs = ["D", "W", "M", "Q", "A"]
        daily_dates = pd.bdate_range(start=start_date, end=end_date)
        ts_dict = {}
        for freq in freqs:
            dt_range = pd.bdate_range(
                start=start_date,
                end=end_date,
                freq=_map_to_business_day_frequency(freq),
            )
            new_ts = (
                pd.Series(data=np.random.normal(0, 1, len(dt_range)), index=dt_range)
                .reindex(daily_dates)
                .ffill()
            )
            ts_dict[freq] = new_ts

        self.timeseries_daily = ts_dict["D"]
        self.timeseries_weekly = ts_dict["W"]
        self.timeseries_monthly = ts_dict["M"]
        self.timeseries_quarterly = ts_dict["Q"]
        self.timeseries_annual = ts_dict["A"]

        self.df_wide = pd.concat(
            ts_dict.values(),
            axis=1,
        )
        self.df_wide.columns = ["daily", "weekly", "monthly", "quarterly", "annual"]
        self.df_wide.index.name = "real_date"

    def test_single_timeseries_daily(self):
        result = estimate_release_frequency(timeseries=self.timeseries_daily)
        self.assertEqual(result, "D")

    def test_df_wide(self):
        result = estimate_release_frequency(df_wide=self.df_wide)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["daily"], "D")
        self.assertEqual(result["weekly"], "W")
        self.assertEqual(result["monthly"], "M")
        self.assertEqual(result["quarterly"], "Q")
        self.assertEqual(result["annual"], "A")

    def test_invalid_timeseries_and_df_wide(self):
        with self.assertRaises(ValueError):
            estimate_release_frequency(
                timeseries=self.timeseries_daily, df_wide=self.df_wide
            )

    def test_invalid_df_wide_type(self):
        with self.assertRaises(TypeError):
            estimate_release_frequency(df_wide="not_a_dataframe")

    def test_empty_df_wide(self):
        with self.assertRaises(ValueError):
            empty_df = pd.DataFrame()
            estimate_release_frequency(df_wide=empty_df)

    def test_invalid_atol_rtol_both_passed(self):
        with self.assertRaises(ValueError):
            estimate_release_frequency(
                timeseries=self.timeseries_daily, atol=0.1, rtol=0.1
            )

    def test_invalid_atol_type(self):
        with self.assertRaises(TypeError):
            estimate_release_frequency(
                timeseries=self.timeseries_daily, atol="not_a_number"
            )

    def test_invalid_rtol_type(self):
        with self.assertRaises(TypeError):
            estimate_release_frequency(
                timeseries=self.timeseries_daily, rtol="not_a_number"
            )

    def test_invalid_rtol_value(self):
        with self.assertRaises(ValueError):
            estimate_release_frequency(timeseries=self.timeseries_daily, rtol=1.5)

    def test_timeseries_with_atol(self):
        result = estimate_release_frequency(
            timeseries=self.timeseries_monthly, atol=0.01
        )
        self.assertEqual(result, "M")

    def test_timeseries_with_rtol(self):
        result = estimate_release_frequency(
            timeseries=self.timeseries_monthly, rtol=0.01
        )
        self.assertEqual(result, "M")


class TestLongToWide(unittest.TestCase):
    def _make_long_df(self, cids=None, dates=None):
        cids = cids or ["AUD", "USD", "GBP"]
        dates = dates or pd.bdate_range("2020-01-01", periods=5)
        rows = [
            {"real_date": d, "cid": c, "value": float(i + j)}
            for j, c in enumerate(cids)
            for i, d in enumerate(dates)
        ]
        return pd.DataFrame(rows)

    def test_normal_case(self):
        df = self._make_long_df()
        wide = _long_to_wide(df, value_col="value")
        self.assertIsInstance(wide, pd.DataFrame)
        self.assertEqual(wide.index.name, "real_date")
        self.assertEqual(set(wide.columns), {"AUD", "USD", "GBP"})
        self.assertEqual(len(wide), 5)

    def test_roundtrip_with_wide_to_long(self):
        df = self._make_long_df()
        wide = _long_to_wide(df, value_col="value")
        long = _wide_to_long(wide, value_name="value")
        self.assertEqual(set(long.columns), {"real_date", "cid", "value"})
        self.assertEqual(len(long), len(df))

    def test_missing_value_col_raises(self):
        df = self._make_long_df()
        with self.assertRaises(ValueError):
            _long_to_wide(df, value_col="nonexistent")

    def test_missing_real_date_raises(self):
        df = self._make_long_df().rename(columns={"real_date": "date"})
        with self.assertRaises(ValueError):
            _long_to_wide(df, value_col="value")

    def test_missing_cid_raises(self):
        df = self._make_long_df().rename(columns={"cid": "ticker"})
        with self.assertRaises(ValueError):
            _long_to_wide(df, value_col="value")

    def test_error_message_lists_missing_columns(self):
        df = self._make_long_df().drop(columns=["cid", "value"])
        with self.assertRaises(ValueError) as ctx:
            _long_to_wide(df, value_col="value")
        self.assertIn("cid", str(ctx.exception))
        self.assertIn("value", str(ctx.exception))

    def test_single_cid(self):
        df = self._make_long_df(cids=["AUD"])
        wide = _long_to_wide(df, value_col="value")
        self.assertEqual(list(wide.columns), ["AUD"])

    def test_custom_value_col(self):
        df = self._make_long_df()
        df = df.rename(columns={"value": "grading"})
        wide = _long_to_wide(df, value_col="grading")
        self.assertIsInstance(wide, pd.DataFrame)
        self.assertEqual(set(wide.columns), {"AUD", "USD", "GBP"})


class TestWideToLong(unittest.TestCase):
    def _make_wide_df(self, cids=None, periods=5):
        cids = cids or ["AUD", "USD", "GBP"]
        dates = pd.bdate_range("2020-01-01", periods=periods)
        data = {c: range(periods) for c in cids}
        df = pd.DataFrame(data, index=dates)
        df.index.name = "real_date"
        return df

    def test_normal_case(self):
        wide = self._make_wide_df()
        long = _wide_to_long(wide, value_name="value")
        self.assertIsInstance(long, pd.DataFrame)
        self.assertEqual(set(long.columns), {"real_date", "cid", "value"})
        self.assertEqual(set(long["cid"].unique()), {"AUD", "USD", "GBP"})
        self.assertEqual(len(long), 5 * 3)

    def test_custom_value_name(self):
        wide = self._make_wide_df()
        long = _wide_to_long(wide, value_name="grading")
        self.assertIn("grading", long.columns)
        self.assertNotIn("value", long.columns)

    def test_nan_rows_dropped(self):
        wide = self._make_wide_df()
        wide.iloc[0, 0] = float("nan")
        long = _wide_to_long(wide, value_name="value")
        self.assertEqual(len(long), 5 * 3 - 1)
        self.assertFalse(long["value"].isna().any())

    def test_sorted_by_cid_then_date(self):
        wide = self._make_wide_df()
        long = _wide_to_long(wide, value_name="value")
        cids_order = long["cid"].tolist()
        self.assertEqual(cids_order, sorted(cids_order))
        for cid in long["cid"].unique():
            dates = long.loc[long["cid"] == cid, "real_date"].tolist()
            self.assertEqual(dates, sorted(dates))

    def test_empty_columns_raises(self):
        empty = pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=3))
        with self.assertRaises(ValueError):
            _wide_to_long(empty)

    def test_single_cid(self):
        wide = self._make_wide_df(cids=["AUD"])
        long = _wide_to_long(wide, value_name="value")
        self.assertEqual(list(long["cid"].unique()), ["AUD"])
        self.assertEqual(len(long), 5)


class TestRotateCidXcat(unittest.TestCase):
    def _make_df(self, cids, xcats, periods=3):
        dates = pd.bdate_range("2020-01-01", periods=periods)
        rows = [
            {"real_date": d, "cid": c, "xcat": x, "value": float(i)}
            for c in cids
            for x in xcats
            for i, d in enumerate(dates)
        ]
        return pd.DataFrame(rows)

    # --- direction validation ---

    def test_invalid_direction_raises(self):
        df = self._make_df(["AUD"], ["EQXR_AUD_NSA"])
        with self.assertRaises(ValueError):
            rotate_cid_xcat(df, "sideways", "EQXR_{cid}_NSA", "EQ")

    # --- xcat_template validation ---

    def test_template_without_placeholder_raises(self):
        df = self._make_df(["AUD"], ["EQXR_AUD_NSA"])
        with self.assertRaises(ValueError):
            rotate_cid_xcat(df, "to_xcats", "EQXR_NSA", "EQ")

    # --- panel shape validation ---

    def test_multiple_cids_and_xcats_raises(self):
        df = self._make_df(["AUD", "USD"], ["EQXR_AUD_NSA", "EQXR_USD_NSA"])
        with self.assertRaises(ValueError):
            rotate_cid_xcat(df, "to_xcats", "EQXR_{cid}_NSA", "EQ")

    # --- to_xcats direction ---

    def test_to_xcats_sets_cid_to_fixed_value(self):
        df = self._make_df(["AUD", "USD", "GBP"], ["EQXR_NSA"])
        result = rotate_cid_xcat(df, "to_xcats", "EQXR_{cid}_NSA", "EQ")
        self.assertTrue((result["cid"] == "EQ").all())

    def test_to_xcats_derives_xcat_from_template(self):
        df = self._make_df(["AUD", "USD"], ["EQXR_NSA"])
        result = rotate_cid_xcat(df, "to_xcats", "EQXR_{cid}_NSA", "EQ")
        self.assertEqual(set(result["xcat"].unique()), {"EQXR_AUD_NSA", "EQXR_USD_NSA"})

    def test_to_xcats_does_not_mutate_input(self):
        df = self._make_df(["AUD", "USD"], ["EQXR_NSA"])
        original_cids = set(df["cid"].unique())
        rotate_cid_xcat(df, "to_xcats", "EQXR_{cid}_NSA", "EQ")
        self.assertEqual(set(df["cid"].unique()), original_cids)

    # --- to_cids direction ---

    def test_to_cids_extracts_cid_from_xcat(self):
        df = self._make_df(["EQ"], ["EQXR_AUD_NSA", "EQXR_USD_NSA", "EQXR_GBP_NSA"])
        result = rotate_cid_xcat(df, "to_cids", "EQXR_{cid}_NSA", "EQXR_NSA")
        self.assertEqual(set(result["cid"].unique()), {"AUD", "USD", "GBP"})

    def test_to_cids_sets_xcat_to_fixed_value(self):
        df = self._make_df(["EQ"], ["EQXR_AUD_NSA", "EQXR_USD_NSA"])
        result = rotate_cid_xcat(df, "to_cids", "EQXR_{cid}_NSA", "EQXR_NSA")
        self.assertTrue((result["xcat"] == "EQXR_NSA").all())

    def test_to_cids_does_not_mutate_input(self):
        df = self._make_df(["EQ"], ["EQXR_AUD_NSA", "EQXR_USD_NSA"])
        original_xcats = set(df["xcat"].unique())
        rotate_cid_xcat(df, "to_cids", "EQXR_{cid}_NSA", "EQXR_NSA")
        self.assertEqual(set(df["xcat"].unique()), original_xcats)

    # --- roundtrip ---

    def test_to_xcats_then_to_cids_roundtrip(self):
        cids = ["AUD", "USD", "GBP"]
        df = self._make_df(cids, ["EQXR_NSA"])
        template = "EQXR_{cid}_NSA"
        rotated = rotate_cid_xcat(df, "to_xcats", template, "EQ")
        restored = rotate_cid_xcat(rotated, "to_cids", template, "EQXR_NSA")
        self.assertEqual(set(restored["cid"].unique()), set(cids))
        self.assertTrue((restored["xcat"] == "EQXR_NSA").all())


if __name__ == "__main__":
    unittest.main()
