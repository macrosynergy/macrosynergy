import unittest
import pandas as pd
import warnings
import datetime

from typing import List, Tuple, Dict, Union, Set, Any
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (
    get_cid,
    get_xcat,
    downsample_df_on_real_date,
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
)


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
        df_test.loc[
            (df_test["cid"] == "AUD") & (df_test["xcat"].isin(["FXXR", "IR"])), "value"
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


if __name__ == "__main__":
    unittest.main()
