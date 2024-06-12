"""Test historical volatility estimates with simulate returns from random normal distribution"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional
from unittest import mock

from macrosynergy.pnl.historic_portfolio_volatility import (
    historic_portfolio_vol,
    flat_weights_arr,
    expo_weights_arr,
    _weighted_covariance,
    estimate_variance_covariance,
    _check_est_args,
    _check_missing_data,
    _check_frequency,
    _check_input_arguments,
)
from macrosynergy.management.utils import qdf_to_ticker_df, ticker_df_to_qdf
from macrosynergy.management.types import QuantamentalDataFrame, NoneType
from macrosynergy.management.simulate import make_test_df


class TestWeightedCovariance(unittest.TestCase):
    # testing `weighted_covariance` function
    def setUp(self):
        self.good_args: Dict[str, Any] = {
            "half_life": 10,
            "lback_periods": 100,
            "x": np.arange(100) / 100,
            "y": np.arange(100) / 100,
            "weights_func": expo_weights_arr,
        }

    def tearDown(self): ...

    def test_weighted_covariance(self):
        # Test good args
        res = _weighted_covariance(**self.good_args)
        self.assertTrue(isinstance(res, float))
        self.assertTrue(np.isclose(res, 0.019827, atol=1e-6))

    def test_x_y_length(self):
        ## X and Y must be same length
        for argn in ["x", "y"]:
            bad_args = self.good_args.copy()
            bad_args[argn] = np.arange(99) / 100
            with self.assertRaises(AssertionError):
                _weighted_covariance(**bad_args)

        ## X and Y must be 1D
        for argn in ["x", "y"]:
            bad_args = self.good_args.copy()
            bad_args[argn] = np.arange(100).reshape((10, 10))
            with self.assertRaises(AssertionError):
                _weighted_covariance(**bad_args)

    def test_nan_handling(self):
        ## For either being all nan, the result should be nan
        for argn in ["x", "y"]:
            bad_args = self.good_args.copy()
            bad_args[argn] = np.full(100, np.nan)
            res = _weighted_covariance(**bad_args)
            self.assertTrue(np.isnan(res))

        bad_args = self.good_args.copy()
        for argn in [["x", [1, 11]], ["y", [7, 42]]]:
            bad_args[argn[0]] = np.full(100, np.nan)
            bad_args[argn[0]][argn[1]] = np.random.rand(2)
        res = _weighted_covariance(**bad_args)
        self.assertTrue(np.isnan(res))


class TestEstimateVarianceCovariance(unittest.TestCase):
    # testing `estimate_variance_covariance` function
    def setUp(self):
        piv_ret = qdf_to_ticker_df(
            make_test_df(
                cids=["A", "B", "C", "D"],
                xcats=["Z", "Y", "X", "W"],
                start="2020-01-01",
                end="2021-01-01",
            )
        )
        self.good_args: Dict[str, Any] = {
            "piv_ret": piv_ret,
            "remove_zeros": True,
            "weights_func": expo_weights_arr,
            "lback_periods": 100,
            "half_life": 10,
        }

    def tearDown(self): ...

    def test_estimate_variance_covariance(self):
        # Test good args
        res = estimate_variance_covariance(**self.good_args)
        self.assertTrue(isinstance(res, pd.DataFrame))
        self.assertEqual(res.shape[0], self.good_args["piv_ret"].shape[1])
        self.assertEqual(res.shape[0], res.shape[1])
        self.assertEqual(set(res.columns), set(self.good_args["piv_ret"].columns))


class TestMisc(unittest.TestCase):
    def setUp(self): ...

    def tearDown(self): ...

    def test_flat_weights_arr(self):
        # Test good args
        res = flat_weights_arr(10)
        self.assertTrue(isinstance(res, np.ndarray))
        self.assertEqual(res.shape[0], 10)
        self.assertTrue(np.allclose(res, np.full(10, 1 / 10)))

    def test_expo_weights_arr(self):
        # Test good args
        res = expo_weights_arr(10, 10)
        self.assertTrue(isinstance(res, np.ndarray))
        self.assertEqual(res.shape[0], 10)

    def test_check_frequency(self):
        # Test good args
        for freq in ["D", "W", "M", "Q", "A"]:
            _check_frequency(freq, "freq-type")
        for freq in ["X", "Y", "Z"]:
            with self.assertRaises(ValueError):
                _check_frequency(freq, "freq-type")

    def test_check_missing_data(self):
        # Test good args
        sname = "SNAME"
        rstring = "RSTRING"
        cids = ["USD", "EUR", "GBP"]
        fids = [f"{c}_FX" for c in cids]
        xcats = [f"FX{rstring}_CSIG_{sname}", f"FX{rstring}"]
        good_df = make_test_df(
            cids=cids,
            xcats=xcats,
            start="2020-01-01",
            end="2021-01-01",
        )
        good_df["ticker"] = good_df["cid"] + "_" + good_df["xcat"]
        _check_missing_data(df=good_df, fids=fids, rstring=rstring, sname=sname)

        # Test bad args
        bad_df = good_df.copy()
        bad_df["xcat"] = bad_df["xcat"].str.replace("CSIG", "BAR")
        bad_df["ticker"] = bad_df["cid"] + "_" + bad_df["xcat"]
        with self.assertRaises(ValueError):
            _check_missing_data(df=bad_df, fids=fids, rstring=rstring, sname=sname)

        # test dropping cid=USD, xcat.endswith("_CSIG_SNAME")
        bad_df = good_df.copy()
        bad_df = bad_df[
            ~(
                (bad_df["cid"] == "USD")
                & (bad_df["xcat"].str.endswith(f"_CSIG_{sname}"))
            )
        ].reset_index(drop=True)
        with self.assertRaises(ValueError):
            _check_missing_data(df=bad_df, fids=fids, rstring=rstring, sname=sname)

        # drop all rows with cid=USD, xcat=="FXRSTRING"
        bad_df = good_df.copy()
        bad_df = bad_df[
            ~((bad_df["cid"] == "USD") & (bad_df["xcat"] == f"FX{rstring}"))
        ].reset_index(drop=True)
        with self.assertRaises(ValueError):
            _check_missing_data(df=bad_df, fids=fids, rstring=rstring, sname=sname)

    def test_check_input_arguments(self):
        arguments = [
            ("df", pd.DataFrame),
            ("sname", str),
            ("fids", list),
            ("rstring", str),
            ("rebal_freq", str),
            ("lback_meth", str),
            ("lback_periods", list),
            ("half_life", list),
            ("est_freqs", list),
            ("est_weights", list),
            ("start", (str, NoneType)),
            ("end", (str, NoneType)),
            ("blacklist", (dict, NoneType)),
            ("nan_tolerance", float),
            ("remove_zeros", bool),
            ("return_variance_covariance", bool),
        ]
        good_args = {
            "df": make_test_df(),
            "sname": "SNAME",
            "fids": ["FID1", "FID2"],
            "rstring": "RSTRING",
            "rebal_freq": "M",
            "lback_meth": "ma",
            "lback_periods": [10, 20],
            "half_life": [5, 10],
            "est_freqs": ["D", "W"],
            "est_weights": [0.5, 0.5],
            "start": "2020-01-01",
            "end": "2021-01-01",
            "blacklist": {"A": ["B", "C"]},
            "nan_tolerance": 0.1,
            "remove_zeros": True,
            "return_variance_covariance": False,
        }

        # Test good args
        _check_input_arguments(
            [(good_args[argn], argn, argt) for argn, argt in arguments]
        )

        # Test bad args
        # pass an int for all arguments
        for argn, argt in arguments:
            # pass an int instead of the expected type
            bad_args = good_args.copy()
            bad_args[argn] = -1
            with self.assertRaises(TypeError):
                _check_input_arguments(
                    [(bad_args[argn], argn, argt) for argn, argt in arguments]
                )

            if isinstance(argt, (list, dict, str)):
                bad_args = good_args.copy()
                if isinstance(argt, list):
                    bad_args[argn] = []
                elif isinstance(argt, dict):
                    bad_args[argn] = {}
                elif isinstance(argt, str):
                    bad_args[argn] = ""
                with self.assertRaises(ValueError):
                    _check_input_arguments(
                        [(bad_args[argn], argn, argt) for argn, argt in arguments]
                    )


if __name__ == "__main__":
    unittest.main()
