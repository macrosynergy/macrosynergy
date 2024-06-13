"""Test historical volatility estimates with simulate returns from random normal distribution"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional
from numbers import Number
from unittest import mock
import warnings
from macrosynergy.pnl.historic_portfolio_volatility import (
    historic_portfolio_vol,
    _calculate_portfolio_volatility,
    flat_weights_arr,
    _downsample_returns,
    expo_weights_arr,
    _weighted_covariance,
    estimate_variance_covariance,
    get_max_lookback,
    _check_est_args,
    _check_missing_data,
    _check_frequency,
    _check_input_arguments,
)
from macrosynergy.management.utils import (
    qdf_to_ticker_df,
    get_sops,
    ticker_df_to_qdf,
    _map_to_business_day_frequency,
)
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


class TestArgChecks(unittest.TestCase):
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

    def test_check_est_args(self):
        def __check_results(tpl: Tuple, good_args: Dict[str, Any], order: List[str]):
            for i, argn in enumerate(order):
                gargs = good_args[argn].copy()
                if argn == "est_weights":
                    gargs = list(np.array(gargs) / np.sum(gargs))
                self.assertEqual(tpl[i], gargs)

        def good_args():
            return {
                "est_freqs": ["D", "W", "M"],
                "est_weights": [0.2, 0.3, 0.5],
                "lback_periods": [15, 8, 5],
                "half_life": [10, 5, 2],
            }

        good_args_order = ["est_freqs", "est_weights", "lback_periods", "half_life"]
        numeric_list_args = ["est_weights", "lback_periods", "half_life"]
        # Test good args
        __check_results(
            good_args=good_args(),
            order=good_args_order,
            tpl=_check_est_args(**good_args()),
        )

        for argn in good_args().keys():
            bad_args = good_args()
            bad_args[argn] = bad_args[argn][:-1]
            with self.assertRaises(ValueError):
                _check_est_args(**bad_args)

        # check that it works works with a single value for the rest of the arguments
        for argn in numeric_list_args:
            bad_args = good_args()
            test_args = bad_args.copy()
            bad_args[argn] = [bad_args[argn][0]]
            test_args[argn] = [bad_args[argn][0]] * len(test_args[argn])
            __check_results(
                good_args=test_args,
                order=good_args_order,
                tpl=_check_est_args(**bad_args),
            )

        # test bad numeric values
        for argn in numeric_list_args:
            bad_args = good_args()
            bad_args[argn][np.random.randint(0, len(bad_args[argn]))] = "w"
            with self.assertRaises(ValueError):
                _check_est_args(**bad_args)

        # test negative weights
        for argn in numeric_list_args:
            bad_args = good_args()
            bad_args[argn][np.random.randint(0, len(bad_args[argn]))] = -0.1
            with self.assertRaises(ValueError):
                _check_est_args(**bad_args)

        # check that lback allows -1
        bad_args = good_args()
        bad_args["lback_periods"] = [-1]
        _check_est_args(**bad_args)


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

    def test_get_max_lookback(self):
        def _get_max_lookback_mock(lb: int, nt: float) -> int:
            return int(np.ceil(lb * (1 + nt))) if lb > 0 else 0

        # nt between 0 and 1, lb > 0
        for _ in range(1000):
            lb = np.random.randint(1, 100)
            nt = np.random.rand()
            res = get_max_lookback(lb, nt)
            self.assertEqual(res, _get_max_lookback_mock(lb, nt))

    def test_downsample_returns(self):
        def _downsample_returns_mock(piv_df: pd.DataFrame, freq: str) -> pd.DataFrame:
            freq = _map_to_business_day_frequency(freq)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                piv_new_freq: pd.DataFrame = (
                    (1 + piv_df / 100).resample(freq).prod() - 1
                ) * 100

                warnings.resetwarnings()

            return piv_new_freq

        cols = ["A", "B", "C", "D"]
        idx = pd.bdate_range(start="2010-01-01", end="2015-01-01")
        freqs = ["D", "W", "M", "Q", "A"]

        def _gen_df(idx, cols):
            return pd.DataFrame(
                np.random.rand(len(idx), len(cols)), columns=cols, index=idx
            )

        for freq in freqs:
            piv_df = _gen_df(idx, cols)
            res = _downsample_returns(piv_df, freq)
            res_mock = _downsample_returns_mock(piv_df, freq)
            self.assertTrue(res.equals(res_mock))


class TestCalculatePortfolioVolatility(unittest.TestCase):
    def setUp(self):
        mkdf_args = dict(
            cids=["USD", "EUR", "GBP", "JPY", "CHF"],
            xcats=["EQ"],
            start="2020-01-01",
            end="2021-01-01",
        )
        _dft = make_test_df(**mkdf_args)
        _dft["value"] = 1
        _dft = qdf_to_ticker_df(_dft)
        self.good_args: Dict[str, Any] = {
            "pivot_returns": _dft,
            "pivot_signals": _dft,
            "weights_func": flat_weights_arr,
            "rebal_freq": "M",
            "est_freqs": ["D", "W"],
            "est_weights": [0.5, 0.5],
            "half_life": [10, 2],
            "lback_periods": [15, 5],
            "nan_tolerance": 0.1,
            "remove_zeros": True,
            "portfolio_return_name": "PORTFOLIO",
        }

    @staticmethod
    def expected_rebal_dates(dt_range: pd.DatetimeIndex, freq: str) -> pd.Series:
        return get_sops(dates=dt_range, freq=freq)

    def tearDown(self): ...

    def test_calculate_portfolio_volatility(self):
        # Test good args
        res = _calculate_portfolio_volatility(**self.good_args)
        # res must be a tuple of 2 elements
        self.assertTrue(isinstance(res, tuple))
        self.assertEqual(len(res), 2)
        # both elements must be pandas dataframes
        self.assertTrue(isinstance(res[0], pd.DataFrame))
        # the first element must have 1 column and real_date as index
        expc_rebal_dates = self.expected_rebal_dates(
            res[0].index, self.good_args["rebal_freq"]
        )
        self.assertTrue(res[0].index.tolist() == expc_rebal_dates.tolist())
        self.assertTrue(res[0].shape[1] == 1)
        # column name must be the same as the portfolio_return_name
        self.assertTrue(res[0].columns[0] == self.good_args["portfolio_return_name"])

        # the second element must be a pandas dataframe
        # must be a df with 'real_date', 'fid1', 'fid2', 'value' as columns
        self.assertTrue(isinstance(res[1], pd.DataFrame))
        self.assertTrue(res[1].shape[1] == 4)
        self.assertTrue(set(res[1].columns) == {"real_date", "fid1", "fid2", "value"})


if __name__ == "__main__":
    unittest.main()
