import unittest
from numbers import Number
from typing import Callable, List

import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.panel.adjust_weights import (
    adjust_weights,
    check_types,
    check_missing_cids_xcats,
    split_weights_adj_zns,
    normalize_weights,
    adjust_weights_backend,
)
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.utils import (
    ticker_df_to_qdf,
    reduce_df,
    get_cid,
    get_xcat,
    qdf_to_ticker_df,
)


class TestAdjustReturnsTypeChecks(unittest.TestCase):
    def setUp(self):
        self.valid_args = {
            "weights": "weights",
            "adj_zns": "adj_zns",
            "method": lambda x: x,  # valid callable
            "param": 3.14,  # valid number
            "cids": ["US", "UK"],
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
            "method": [42, "not callable", None],
            "param": ["string", None, {}],
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
        for bad_val in [123, "not a list", None]:
            with self.subTest(bad_value=bad_val):
                args = self.valid_args.copy()
                args["cids"] = bad_val
                with self.assertRaises(TypeError):
                    check_types(**args)

    def test_invalid_cids_contents(self):
        """Test that passing a list with non-string elements in 'cids' raises TypeError."""
        args = self.valid_args.copy()
        args["cids"] = ["US", 123]
        with self.assertRaises(TypeError):
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
        df_with_nans = df_with_nans.mask(nan_mask < 0.2)
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
        adjusted = adjust_weights_backend(
            self.df_weights_wide, self.df_adj_zns_wide, lambda x: x, 1
        )
        for cid, expected in self.expected_results.items():
            with self.subTest(cid=cid):
                # check that there is one unique value in the colum
                uval = list(set(adjusted[cid]))
                self.assertEqual(len(uval), 1)
                self.assertAlmostEqual(uval[0], expected)


class TestAdjustWeightsMain(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY"]
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

        start, end = "2020-01-01", "2021-02-01"
        temp_df = make_test_df(tickers=self.tickers, start=start, end=end)
        wdf = qdf_to_ticker_df(temp_df)
        for ticker, weight in self.ticker_weights.items():
            wdf[ticker] = weight
        self.wdf = wdf
        self.qdf = ticker_df_to_qdf(wdf)

    def test_adjust_weights(self):
        args = {
            "df": self.qdf,
            "weights": "WG",
            "adj_zns": "AZ",
            "method": lambda x: x,
            "param": 1,
        }
        
        
if __name__ == "__main__":
    unittest.main()