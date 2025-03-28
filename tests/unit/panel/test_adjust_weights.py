import unittest
from typing import Callable
import warnings
import numpy as np
import pandas as pd
from macrosynergy.compat import PD_2_0_OR_LATER, PYTHON_3_8_OR_LATER
from macrosynergy.panel.adjust_weights import (
    adjust_weights,
    check_types,
    check_missing_cids_xcats,
    split_weights_adj_zns,
    normalize_weights,
    generic_weights_backend,
    lincomb_backend,
)

from macrosynergy.panel.lincomb_adjust import linear_combination_adjustment
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.utils import (
    ticker_df_to_qdf,
    reduce_df,
    get_cid,
    qdf_to_ticker_df,
)


class TestAdjustWeightsTypeChecks(unittest.TestCase):
    def setUp(self):
        def _sigmoid(x, a=1, b=1, c=0):
            return a / (1 + np.exp(-b * (x - c)))

        _params = {"a": 1, "b": 1, "c": 0}
        self.valid_args = {
            "weights": "weights",
            "adj_zns": "adj_zns",
            "method": "generic",
            "adj_func": _sigmoid,
            "params": _params,
            "cids": ["USD", "GBP", "JPY"],
        }

    def test_valid_input(self):
        """Check that valid input does not raise any exception."""
        try:
            check_types(**self.valid_args)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_invalid_types_for_non_list_params(self):
        """Loop over non-list parameters and test with invalid types."""
        invalids = {
            "weights": [123, None, []],
            "adj_zns": [456, None, {}],
            "method": [123, None, []],
            "adj_func": [42, "not callable", []],
            "params": ["string", None, [{"a": 1}]],
        }
        for param, bad_values in invalids.items():
            for bad_val in bad_values:
                with self.subTest(param=param, bad_value=bad_val):
                    args = self.valid_args.copy()
                    args[param] = bad_val
                    with self.assertRaises(TypeError):
                        check_types(**args)

    def test_invalid_cids_type(self):
        """Test that passing a non-list for 'cids' raises TypeError."""
        for bad_val in [123, "not a list", []]:
            with self.subTest(bad_value=bad_val):
                args = self.valid_args.copy()
                args["cids"] = bad_val
                with self.assertRaises(TypeError):
                    check_types(**args)

    def test_invalid_cids_contents(self):
        """Test that passing a list with non-string elements in 'cids' raises TypeError."""
        args = self.valid_args.copy()
        args["cids"] = ["USD", 123]
        with self.assertRaises(TypeError):
            check_types(**args)

    def test_invalid_qdf(self):
        # this tests uses adjust_returns directly
        with self.assertRaises(TypeError):
            adjust_weights(df=pd.DataFrame(), **self.valid_args)

    def test_method_provided(self):
        # test that the method is 'generic' it should except adj_func to be non-None
        args = self.valid_args.copy()
        args["method"] = "generic"
        args["adj_func"] = None

        with self.assertRaises(ValueError):
            check_types(**args)

    def test_method_lincomb(self):
        # test that the method is 'lincomb' it should except adj_func to be None
        args = self.valid_args.copy()
        args["method"] = "lincomb"
        args["adj_func"] = None

        try:
            check_types(**args)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_method_invalid(self):
        # test that the method is 'lincomb' it should except adj_func to be None
        args = self.valid_args.copy()
        args["method"] = "invalid"

        with self.assertRaises(ValueError):
            check_types(**args)


class TestAdjustReturnsMissingLogic(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP"]
        self.xcats = ["X1", "X2", "X3", "WG", "AZ"]
        self.wa_xcats = ["WG", "AZ"]
        self.valid_args = {
            "weights": "WG",
            "adj_zns": "AZ",
            "cids": self.cids,
            "r_xcats": self.wa_xcats,
            "r_cids": self.cids,
        }
        self.tickers = [
            f"{cid}_{xcat}"
            for cid in self.valid_args["cids"]
            for xcat in self.valid_args["r_xcats"]
        ]

    def test_no_missing(self):
        """Check that no exception is raised if all required xcats and cids are present."""
        try:
            check_missing_cids_xcats(**self.valid_args)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_missing_xcats(self):
        # remove one xcat at a time and check that the exception is raised
        for missing in self.wa_xcats:
            with self.subTest(missing=missing):
                args = self.valid_args.copy()
                args["r_xcats"] = [x for x in args["r_xcats"] if x != missing]
                with self.assertRaises(ValueError) as context:
                    check_missing_cids_xcats(**args)
                self.assertIn("Missing xcats", str(context.exception))

    def test_missing_cids(self):
        # remove one cid at a time and check that the exception is raised
        for missing in self.cids:
            with self.subTest(missing=missing):
                args = self.valid_args.copy()
                args["r_cids"] = [x for x in args["r_cids"] if x != missing]
                with self.assertRaises(ValueError) as context:
                    check_missing_cids_xcats(**args)
                self.assertIn("Missing cids", str(context.exception))


class TestNormalizeWeights(unittest.TestCase):
    def setUp(self):
        # self.df_valid = pd.DataFrame({"A": [1, 2, 3], "B": [1, 3, 4]})
        cids = ["USD", "EUR", "JPY"]
        xcats = ["X1", "X2", "X3"]
        tickers = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
        self.df_valid = qdf_to_ticker_df(make_test_df(tickers=tickers))

    def test_normalization_valid(self):
        normalized_df = normalize_weights(self.df_valid)
        for idx, row in normalized_df.iterrows():
            with self.subTest(row=idx):
                # Use np.nansum to ignore any potential NaNs (though there are none here).
                self.assertTrue(
                    np.isclose(np.nansum(row), 1.0),
                    msg=f"Row {idx} normalized nansum is {np.nansum(row)} (expected 1).",
                )
                # Also compare with expected results computed manually.
                expected = self.df_valid.loc[idx] / self.df_valid.loc[idx].sum()
                pd.testing.assert_series_equal(row, expected)

    def test_normalization_with_nans(self):
        df_with_nans = self.df_valid.copy()
        nan_mask = np.random.random(df_with_nans.shape) < 0.2
        df_with_nans = df_with_nans.mask(nan_mask)
        normalized_df = normalize_weights(df_with_nans)

        for idx, row in normalized_df.iterrows():
            with self.subTest(row=idx):
                # If there is at least one non-NaN value, then the non-NaN entries should sum to 1.
                if np.nansum(df_with_nans.loc[idx]) > 0:
                    self.assertTrue(
                        np.isclose(np.nansum(row), 1.0),
                        msg=f"Row {idx} normalized nansum is {np.nansum(row)} (expected 1).",
                    )

        # verify that NaNs are preserved
        self.assertTrue(normalized_df.isna().equals(df_with_nans.isna()))

    def test_normalization_zero_sum(self):
        df_zero = self.df_valid.copy()

        # select a random day from the index and make it 0
        zero_idx = np.random.choice(df_zero.index)
        df_zero.loc[zero_idx] = 0

        normalized_df = normalize_weights(df_zero)
        with self.subTest("Row with zero non-NaN sum"):
            self.assertTrue(normalized_df.loc[zero_idx].isna().all())
        with self.subTest("Rows with non-zero sum"):
            for idx, row in normalized_df.iterrows():
                if np.nansum(df_zero.loc[idx]) > 0:
                    expected = df_zero.loc[idx] / df_zero.loc[idx].sum()
                    pd.testing.assert_series_equal(row, expected)

    def test_normalize_random_exception(self):
        self.df_valid[np.random.random(self.df_valid.shape) < 0.2] = np.inf

        with self.assertRaises(Exception):
            normalize_weights(self.df_valid)

    def test_norm_to_pct(self):
        df = self.df_valid.copy()
        df = normalize_weights(df, normalize_to_pct=True)
        self.assertTrue(np.allclose(df.sum(axis=1), 100))


class TestSplitWeightsAdjZns(unittest.TestCase):
    def setUp(self):

        self.cids = ["USD", "EUR", "JPY", "GBP"]
        self.xcats = ["X1", "X2", "X3", "WG", "AZ"]
        self.df = make_test_df(cids=self.cids, xcats=self.xcats)
        self.adj_zns = "AZ"
        self.weights = "WG"

    def test_split_success(self):
        df_weights_wide, df_adj_zns_wide = split_weights_adj_zns(
            self.df, self.weights, self.adj_zns
        )
        self.assertEqual(df_weights_wide.shape[1], len(self.cids))
        self.assertEqual(df_adj_zns_wide.shape[1], len(self.cids))
        self.assertEqual(set(df_weights_wide.columns), set(df_adj_zns_wide.columns))

    def test_split_failure(self):
        # remove one cid at a time and check that the exception is raised
        for miss_cid in self.cids:
            for miss_xcat in [self.weights, self.adj_zns]:
                with self.subTest(missing=f"{miss_cid}_{miss_xcat}"):
                    df = self.df.copy()
                    df = df[~((df["cid"] == miss_cid) & (df["xcat"] == miss_xcat))]
                    with self.assertRaises(ValueError) as context:
                        split_weights_adj_zns(df, self.weights, self.adj_zns)
                    self.assertIn("Missing tickers", str(context.exception))

    def test_missing_dates_filling(self):
        # choose 20% of the dates to be missing
        all_dates = self.df["real_date"].unique().tolist()
        missing_dates = pd.to_datetime(
            np.random.choice(all_dates, int(0.1 * len(all_dates)), replace=False)
        )

        # remove the missing dates from the dataframe for adj_zns
        self.df.loc[
            (self.df["real_date"].isin(missing_dates))
            & (self.df["xcat"] == self.adj_zns),
            "value",
        ] = np.nan

        with warnings.catch_warnings(record=True) as w:
            _, df_adj_zns_wide = split_weights_adj_zns(
                self.df, self.weights, self.adj_zns
            )
            if PYTHON_3_8_OR_LATER:
                self.assertTrue(len(w) > 0)
                last_warn = w[-1].message.args[0]
                for date in missing_dates:
                    with self.subTest(date=date):
                        self.assertIn(
                            pd.Timestamp(date).strftime("%Y-%m-%d"), last_warn
                        )

        # check that the missing dates have been filled with 1
        for date in missing_dates:
            with self.subTest(date=date):
                self.assertTrue(np.allclose(df_adj_zns_wide.loc[pd.Timestamp(date)], 1))

    def test_negative_weights(self):
        # choose 20% of the dates to be missing
        all_dates = self.df["real_date"].unique().tolist()
        negative_date = pd.Timestamp(np.random.choice(all_dates, 1, replace=False)[0])

        # random cid
        cid = np.random.choice(self.cids)
        self.df.loc[
            (self.df["real_date"] == negative_date)
            & (self.df["cid"] == cid)
            & (self.df["xcat"] == self.weights),
            "value",
        ] = -1

        with self.assertRaises(ValueError) as context:
            split_weights_adj_zns(self.df, self.weights, self.adj_zns)

            # chec that the and the cid are in the error message
            self.assertIn(cid, str(context.exception))
            self.assertIn(negative_date.strftime("%Y-%m-%d"), str(context.exception))


def get_primes(n):
    """Return a list of the first n prime numbers."""
    primes = []
    num = 2
    while len(primes) < n:
        for p in primes:
            if num % p == 0:
                break
        else:
            primes.append(num)
        num += 1
    return primes


class TestAdjustWeightsBackend(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY"]
        self.xcats = ["WG", "AZ"]
        tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats]
        self.tickers = list(np.random.permutation(tickers))

        # prime numbers have been chosen so the weights can be easily tested
        self.ticker_weights = dict(zip(self.tickers, get_primes(len(self.tickers))))

        self.expected_results = {
            _cid: np.prod(
                [
                    self.ticker_weights[ticker]
                    for ticker in self.tickers
                    if get_cid(ticker) == _cid
                ]
            )
            for _cid in self.cids
        }

        start, end = "2020-01-01", "2021-02-01"
        temp_df = make_test_df(tickers=self.tickers, start=start, end=end)
        wdf = qdf_to_ticker_df(temp_df)
        for ticker, weight in self.ticker_weights.items():
            wdf[ticker] = weight
        self.wdf = wdf
        self.qdf = ticker_df_to_qdf(wdf)

        self.df_weights_wide, self.df_adj_zns_wide = split_weights_adj_zns(
            self.qdf, weights="WG", adj_zns="AZ"
        )

    def test_adjust_weights_backend(self):
        adjusted = generic_weights_backend(
            self.df_weights_wide,
            self.df_adj_zns_wide,
            lambda x: x,
            # no params
        )
        for cid, expected in self.expected_results.items():
            with self.subTest(cid=cid):
                # check that there is one unique value in the colum
                uval = list(set(adjusted[cid]))
                self.assertEqual(len(uval), 1)
                self.assertAlmostEqual(uval[0], expected)

    def test_adjust_weights_backend_nan(self):
        # randomly set some values to NaN
        nan_weights_mask = np.random.random(self.df_weights_wide.shape) < 0.2
        self.df_weights_wide[nan_weights_mask] = np.nan

        nan_adj_zns_mask = np.random.random(self.df_adj_zns_wide.shape) < 0.2
        self.df_adj_zns_wide[nan_adj_zns_mask] = np.nan

        def _method(x):
            return x

        adjusted = generic_weights_backend(
            df_adj_zns_wide=self.df_adj_zns_wide,
            df_weights_wide=self.df_weights_wide,
            adj_func=_method,
        )

        expc_res = self.df_weights_wide * self.df_adj_zns_wide.apply(_method) * 1
        self.assertTrue(adjusted.equals(expc_res))

    def test_adjust_weights_backend_params(self):
        def _method(x, a=1, b=1):
            return x * a * b

        _params = {"a": 0}

        adjusted = generic_weights_backend(
            df_adj_zns_wide=self.df_adj_zns_wide,
            df_weights_wide=self.df_weights_wide,
            adj_func=_method,
            params=_params,
        )

        expc_res = self.df_weights_wide * self.df_adj_zns_wide.apply(_method, **_params)
        self.assertTrue(adjusted.equals(expc_res))
        self.assertTrue((adjusted == 0).all().all())


class TestLinCombBackend(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY"]
        self.xcats = ["WG", "AZ"]
        tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats]
        self.tickers = list(np.random.permutation(tickers))

        # prime numbers have been chosen so the weights can be easily tested
        self.ticker_weights = dict(zip(self.tickers, get_primes(len(self.tickers))))

        self.expected_results = {
            _cid: np.prod(
                [
                    self.ticker_weights[ticker]
                    for ticker in self.tickers
                    if get_cid(ticker) == _cid
                ]
            )
            for _cid in self.cids
        }

        start, end = "2020-01-01", "2021-02-01"
        temp_df = make_test_df(tickers=self.tickers, start=start, end=end)
        wdf = qdf_to_ticker_df(temp_df)
        for ticker, weight in self.ticker_weights.items():
            wdf[ticker] = weight
        self.wdf = wdf
        self.qdf = ticker_df_to_qdf(wdf)

        self.df_weights_wide, self.df_adj_zns_wide = split_weights_adj_zns(
            self.qdf, weights="WG", adj_zns="AZ"
        )

    def test_lincomb_backend_full(self):
        min_score = self.df_adj_zns_wide.min().min()
        res1 = lincomb_backend(
            df_adj_zns_wide=self.df_adj_zns_wide,
            df_weights_wide=self.df_weights_wide,
            min_score=min_score,
            coeff_new=1,
        )

        expc_result = self.df_adj_zns_wide - min_score
        expc_result = normalize_weights(expc_result)
        diff = np.allclose((res1) - (expc_result), 0, atol=1e-12)
        # due to multiple floating point operations, there is *some* floating point error
        self.assertTrue(diff)

    def test_lincomb_backend_no_min_score(self):

        with warnings.catch_warnings(record=True) as w:
            res1 = lincomb_backend(
                df_adj_zns_wide=self.df_adj_zns_wide,
                df_weights_wide=self.df_weights_wide,
                coeff_new=0,
            )
            self.assertTrue(len(w) > 0)
            last_warn = w[-1].message.args[0]
            self.assertIn("`min_score` not provided", last_warn)

        expc_result = normalize_weights(self.df_weights_wide)
        diff = np.allclose((res1) - (expc_result), 0, atol=1e-12)
        self.assertTrue(diff)

    def test_different_min_score(self):
        min_score = -1
        res1 = lincomb_backend(
            df_adj_zns_wide=self.df_adj_zns_wide,
            df_weights_wide=self.df_weights_wide,
            min_score=min_score,
            coeff_new=1,
        )

        expc_result = self.df_adj_zns_wide - min_score
        expc_result = normalize_weights(expc_result)
        diff = np.allclose((res1) - (expc_result), 0, atol=1e-12)
        self.assertTrue(diff)


def expected_adjusted_weights(
    df: pd.DataFrame,
    weights_xcat: str,
    adj_zns_xcat: str,
    method: str,
    adj_func: Callable,
    params: dict,
    adj_name: str,
    normalize: bool = True,
    normalize_to_pct: bool = False,
) -> pd.DataFrame:
    cids = list(set(df["cid"]))
    check_types(weights_xcat, adj_zns_xcat, method, adj_func, params, cids)
    df, r_xcats, r_cids = reduce_df(
        df, cids=cids, xcats=[weights_xcat, adj_zns_xcat], intersect=True, out_all=True
    )
    check_missing_cids_xcats(weights_xcat, adj_zns_xcat, cids, r_xcats, r_cids)
    df_weights_wide, df_adj_zns_wide = split_weights_adj_zns(
        df, weights_xcat, adj_zns_xcat
    )
    nan_rows = df_adj_zns_wide.isna().all(axis="columns")
    df_adj_zns_wide.loc[nan_rows] = 1

    if method == "generic":
        dfw_result = generic_weights_backend(
            df_weights_wide, df_adj_zns_wide, adj_func, params
        )
    elif method == "lincomb":
        dfw_result = lincomb_backend(
            dfw_adj_zns=df_weights_wide, dfw_weights=df_adj_zns_wide, **params
        )
    dfw_result = dfw_result.dropna(how="all", axis="rows")
    if normalize:
        dfw_result = normalize_weights(dfw_result) * (100 if normalize_to_pct else 1)
    dfw_result.columns = list(map(lambda x: f"{x}_{adj_name}", dfw_result.columns))
    return ticker_df_to_qdf(dfw_result).dropna(how="any", axis=0).reset_index(drop=True)


class TestAdjustWeightsMain(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP", "AUD"]
        self.xcats = ["WG", "AZ"]
        tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats]
        self.tickers = list(np.random.permutation(tickers))
        self.ticker_weights = dict(zip(self.tickers, get_primes(len(self.tickers))))

        self.expected_results = {
            _cid: np.prod(
                [
                    self.ticker_weights[ticker]
                    for ticker in self.tickers
                    if get_cid(ticker) == _cid
                ]
            )
            for _cid in self.cids
        }

        self.start, self.end = "2020-01-01", "2021-02-01"
        temp_df = make_test_df(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            style="linear",
        )
        wdf = qdf_to_ticker_df(temp_df)
        for ticker, weight in self.ticker_weights.items():
            wdf[ticker] *= weight
        self.wdf = wdf
        self.qdf = ticker_df_to_qdf(wdf)

    def test_adjust_weights(self):
        args = {
            "weights_xcat": "WG",
            "adj_zns_xcat": "AZ",
            "method": "generic",
            "adj_func": lambda x: x,
            "params": {},
            "adj_name": "ADJWGT",
        }

        expc_result = expected_adjusted_weights(df=self.qdf, **args)
        adjusted = adjust_weights(df=self.qdf, **args)

        if PD_2_0_OR_LATER:
            self.assertTrue(adjusted.equals(expc_result))
        else:
            self.assertTrue(adjusted.eq(expc_result).all().all())

    def test_adjust_weights_with_nans(self):
        all_nan_date: pd.Timestamp = np.random.choice(self.qdf["real_date"].unique())
        self.qdf.loc[
            (self.qdf["real_date"] == all_nan_date) & self.qdf["xcat"].eq("AZ"), "value"
        ] = np.nan

        args = {
            "weights_xcat": "WG",
            "adj_zns_xcat": "AZ",
            "method": "generic",
            "adj_func": lambda x: x,
            "params": {},
            "adj_name": "ADJWGT",
        }

        with warnings.catch_warnings(record=True) as w:
            expc_result = expected_adjusted_weights(df=self.qdf, **args)

        with warnings.catch_warnings(record=True) as w:
            adjusted = adjust_weights(df=self.qdf, **args)

            last_warn = w[-1].message.args[0]
            ts_str = pd.Timestamp(all_nan_date).strftime("%Y-%m-%d")
            err_str = "Missing ZNs data (will be filled with 1"
            self.assertIn(ts_str, last_warn)
            self.assertIn(err_str, last_warn)

        self.assertFalse(adjusted.isna().any().any())
        self.assertTrue(np.allclose(adjusted.groupby("real_date")["value"].sum(), 1))

        if PD_2_0_OR_LATER:
            self.assertTrue(adjusted.equals(expc_result))
        else:
            self.assertTrue(adjusted.eq(expc_result).all().all())

    def test_adjust_weights_all_zeros(self):
        nan_weights_mask = np.random.random(len(self.qdf)) < 0.2
        self.qdf.loc[nan_weights_mask, "value"] = np.nan

        all_nan_date: pd.Timestamp = np.random.choice(self.qdf["real_date"].unique())
        self.qdf.loc[
            (self.qdf["real_date"] == all_nan_date) & self.qdf["xcat"].eq("AZ"), "value"
        ] = np.nan

        def _method(x, a=1):
            return x * a

        args = {
            "weights_xcat": "WG",
            "adj_zns_xcat": "AZ",
            "method": "generic",
            "adj_func": _method,
            "params": {"a": 0},
            "adj_name": "ADJWGT",
        }

        with self.assertRaises(ValueError) as context:
            with warnings.catch_warnings(record=True):
                adjust_weights(df=self.qdf, **args)
            self.assertIn("The resulting DataFrame is empty", str(context.exception))

    def test_adjust_weights_nans_and_no_normalize(self):
        nan_weights_mask = np.random.random(len(self.qdf)) < 0.25
        self.qdf.loc[nan_weights_mask, "value"] = np.nan
        # make sure the sum is not 100
        self.qdf.loc[:, "value"] = self.qdf["value"] * 1e7

        all_nan_date: pd.Timestamp = np.random.choice(self.qdf["real_date"].unique())
        self.qdf.loc[
            (self.qdf["real_date"] == all_nan_date) & self.qdf["xcat"].eq("AZ"), "value"
        ] = np.nan

        args = {
            "weights_xcat": "WG",
            "adj_zns_xcat": "AZ",
            "adj_func": lambda x: x,
            "method": "generic",
            "params": {},
            "adj_name": "ADJWGT",
            "normalize": False,
        }

        with warnings.catch_warnings(record=True) as w:
            expc_result = expected_adjusted_weights(df=self.qdf, **args)

        with warnings.catch_warnings(record=True) as w:
            adjusted = adjust_weights(df=self.qdf, **args)

            split_warning = w[-2].message.args[0]
            ts_str = pd.Timestamp(all_nan_date).strftime("%Y-%m-%d")
            err_str = "Missing ZNs data (will be filled with 1"
            self.assertIn(ts_str, split_warning)
            self.assertIn(err_str, split_warning)

            miss_dates = set(pd.bdate_range(self.start, self.end)) - set(
                adjusted["real_date"]
            )
            nan_date_warning = w[-1].message.args[0]
            for mdt in miss_dates:
                self.assertIn(mdt.strftime("%Y-%m-%d"), nan_date_warning)

        self.assertFalse(adjusted.isna().any().any())
        self.assertFalse(any(adjusted.groupby("real_date")["value"].sum() == 100))

        if PD_2_0_OR_LATER:
            self.assertTrue(adjusted.equals(expc_result))
        else:
            self.assertTrue(adjusted.eq(expc_result).all().all())

    def test_adjust_weights_lincomb_failure(self):
        args = {
            "weights_xcat": "WG",
            "adj_zns_xcat": "AZ",
            "method": "lincomb",
            "params": {"min_score": 0},
            "adj_name": "ADJWGT",
        }

        with self.assertRaises(ValueError):
            adjust_weights(df=self.qdf, **args)

    def test_adjust_weights_missing_cids(self):
        args = {
            "weights_xcat": "WG",
            "adj_zns_xcat": "AZ",
            "method": "lincomb",
            "params": {},
            "adj_name": "ADJWGT",
        }

        df = self.qdf.copy()

        with warnings.catch_warnings(record=True) as w:
            with self.assertRaises(ValueError):
                adjust_weights(df=df, cids=self.cids, **args)
            self.assertTrue(len(w) > 0)
            last_warn = w[-1].message.args[0]
            self.assertIn("`min_score` not provided.", last_warn)

        df = df[df["cid"] != "USD"]

        with self.assertRaises(ValueError):
            args["params"] = {"coeff_new": -1}
            adjust_weights(df=df, cids=self.cids, **args)

    def test_adjust_weights_cids_not_specified(self):
        args = {
            "weights_xcat": "WG",
            "adj_zns_xcat": "AZ",
            "method": "generic",
            "adj_func": lambda x: x,
            "params": {},
            "adj_name": "ADJWGT",
        }

        df = self.qdf.copy()
        df = df[df["cid"] != "USD"]

        try:
            adjust_weights(df=df, **args)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")


class TestLinearCombinationAdjustmentMapping(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY"]
        self.xcats = ["WG", "AZ"]
        tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats]
        self.tickers = list(np.random.permutation(tickers))

        # prime numbers have been chosen so the weights can be easily tested
        self.ticker_weights = dict(zip(self.tickers, get_primes(len(self.tickers))))

        self.expected_results = {
            _cid: np.prod(
                [
                    self.ticker_weights[ticker]
                    for ticker in self.tickers
                    if get_cid(ticker) == _cid
                ]
            )
            for _cid in self.cids
        }

        start, end = "2020-01-01", "2021-02-01"
        self.qdf = make_test_df(tickers=self.tickers, start=start, end=end)

        self.df_weights_wide, self.df_adj_zns_wide = split_weights_adj_zns(
            self.qdf, weights="WG", adj_zns="AZ"
        )

    def test_mapping(self):
        common_args = {
            "weights_xcat": "WG",
            "adj_zns_xcat": "AZ",
            "adj_name": "ADJWGT",
            "cids": self.cids,
            "normalize": True,
        }
        coeff_new = 0.5

        with warnings.catch_warnings(record=True):
            res1 = adjust_weights(
                df=self.qdf,
                method="lincomb",
                params=dict(coeff_new=coeff_new),
                **common_args,
            )

            res2 = linear_combination_adjustment(
                df=self.qdf,
                coeff_new=coeff_new,
                **common_args,
            )

        self.assertTrue(res1.equals(res2))


if __name__ == "__main__":
    unittest.main()
