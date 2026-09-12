"""Test historical volatility estimates with simulate returns from random normal distribution"""

import logging
import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from unittest import mock
from macrosynergy.pnl.historic_portfolio_volatility import (
    historic_portfolio_vol,
    unstack_covariances,
    _hist_vol,
    _calculate_portfolio_volatility,
    _calculate_multi_frequency_vcv_for_period,
    _cov_matrix_history,
    flat_weights_arr,
    _downsample_returns,
    expo_weights_arr,
    _weighted_covariance,
    estimate_variance_covariance,
    _check_est_args,
    _check_missing_data,
    _check_frequency,
    _check_input_arguments,
    _get_first_usable_date,
    _bdays_per_period,
    _nan_frac_since_first_obs,
    _blend_frequency_vcvs,
    RETURN_SERIES_XCAT,
)
from macrosynergy.management.utils import qdf_to_ticker_df, get_sops
from macrosynergy.management.constants import ANNUALIZATION_FACTORS
from macrosynergy.management.types import QuantamentalDataFrame, NoneType
from macrosynergy.management.simulate import (
    make_test_df,
    simulate_returns_and_signals,
    SignalsAndReturnsGenerator,
)


class TestWeightedCovariance(unittest.TestCase):
    # testing `weighted_covariance` function
    def setUp(self): ...

    @property
    def good_args(self):
        return {
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
            "lback_min_obs": 1,
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

            if argt in [list, dict, str]:
                bad_args = good_args.copy()
                if argt == list:
                    bad_args[argn] = []
                elif argt == dict:
                    bad_args[argn] = {}
                elif argt == str:
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
                "lback_min_obs": [1, 1, 1],
            }

        good_args_order = [
            "est_freqs",
            "est_weights",
            "lback_periods",
            "half_life",
            "lback_min_obs",
        ]
        numeric_list_args = [
            "est_weights",
            "lback_periods",
            "half_life",
            "lback_min_obs",
        ]
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
        bad_args["lback_min_obs"] = [1]
        _check_est_args(**bad_args)


class TestMisc(unittest.TestCase):
    def setUp(self):
        data = {
            "real_date": [
                "2022-01-01",
                "2022-01-01",
                "2022-01-01",
                "2022-01-02",
                "2022-01-02",
                "2022-01-02",
            ],
            "fid1": ["A", "A", "B", "A", "B", "B"],
            "fid2": ["A", "B", "B", "A", "A", "B"],
            "value": [1.0, 0.5, 1.0, 1.0, 0.8, 1.0],
        }
        self.vcv_df = pd.DataFrame(data)

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

    @staticmethod
    def _series_df(n_rows: int, cols: List[str] = ["A"]) -> pd.DataFrame:
        # distinct, non-zero returns so any mis-bucketing changes the answer
        idx = pd.bdate_range("2020-01-01", periods=n_rows)
        vals = np.arange(1, n_rows * len(cols) + 1, dtype=float).reshape(
            n_rows, len(cols)
        )
        return pd.DataFrame(vals, index=idx, columns=cols)

    def test_downsample_returns_compounds_within_backward_buckets(self):
        # buckets are counted back from the most recent row rather than anchored to
        # calendar period ends, so every bucket holds exactly `n` business days.
        n = _bdays_per_period("W-FRI")
        piv_df = self._series_df(2 * n + 1)

        res = _downsample_returns(piv_df, "W-FRI")

        self.assertEqual(res.shape, (2, 1))
        vals = piv_df["A"].to_numpy()
        expected = [
            100 * (np.prod(1 + vals[k : k + n] / 100) - 1)
            for k in (1, 1 + n)  # row 0 is the dropped partial block
        ]
        np.testing.assert_allclose(res["A"].to_numpy(), expected, rtol=1e-12)
        self.assertEqual(res.index.tolist(), [0, 1])

    def test_downsample_returns_drops_oldest_partial_block(self):
        n = _bdays_per_period("W-FRI")
        piv_df = self._series_df(2 * n + 3)

        res = _downsample_returns(piv_df, "W-FRI")
        self.assertEqual(res.shape[0], len(piv_df) // n)

        # the 3 oldest rows fall outside a whole bucket, so perturbing them has
        # no effect
        perturbed = piv_df.copy()
        perturbed.iloc[:3] += 1000.0
        np.testing.assert_allclose(
            _downsample_returns(perturbed, "W-FRI").to_numpy(),
            res.to_numpy(),
            rtol=1e-12,
        )

    def test_downsample_returns_daily_preserves_values(self):
        piv_df = self._series_df(20, ["A", "B"])
        for freq in ("B", "D"):
            res = _downsample_returns(piv_df, freq)
            self.assertEqual(res.shape, piv_df.shape)
            np.testing.assert_allclose(
                res.to_numpy(), piv_df.to_numpy(), rtol=1e-12
            )

    def test_downsample_returns_nan_handling(self):
        n = _bdays_per_period("W-FRI")
        piv_df = self._series_df(2 * n)

        # a bucket with no observations at all cannot be compounded
        all_nan = piv_df.copy()
        all_nan.iloc[:n] = np.nan
        res = _downsample_returns(all_nan, "W-FRI")
        self.assertTrue(np.isnan(res["A"].iloc[0]))
        self.assertFalse(np.isnan(res["A"].iloc[1]))

        # a partially observed bucket compounds the observations it does have
        partial = piv_df.copy()
        partial.iloc[0] = np.nan
        res = _downsample_returns(partial, "W-FRI")
        expected = 100 * (np.prod(1 + piv_df["A"].to_numpy()[1:n] / 100) - 1)
        self.assertAlmostEqual(res["A"].iloc[0], expected, places=10)

    def test_downsample_returns_shorter_than_one_bucket_is_empty(self):
        piv_df = self._series_df(_bdays_per_period("W-FRI") - 1)
        self.assertEqual(_downsample_returns(piv_df, "W-FRI").shape[0], 0)

    def test_downsample_returns_sorts_index(self):
        piv_df = self._series_df(2 * _bdays_per_period("W-FRI"))
        np.testing.assert_allclose(
            _downsample_returns(piv_df.iloc[::-1], "W-FRI").to_numpy(),
            _downsample_returns(piv_df, "W-FRI").to_numpy(),
            rtol=1e-12,
        )

    def test_unstack_covariances_no_fillna(self):
        result = unstack_covariances(self.vcv_df, fillna=False)

        expected_result = {
            "2022-01-01": pd.DataFrame(
                {"A": {"A": 1.0, "B": 0.5}, "B": {"A": None, "B": 1.0}}
            ),
            "2022-01-02": pd.DataFrame(
                {"A": {"A": 1.0, "B": None}, "B": {"A": 0.8, "B": 1.0}}
            ),
        }

        for dt in result:
            self.assertTrue(result[dt].equals(expected_result[dt]))

    def test_unstack_covariances_with_fillna(self):
        result = unstack_covariances(self.vcv_df, fillna=True)

        expected_result = {
            "2022-01-01": pd.DataFrame(
                {"A": {"A": 1.0, "B": 0.5}, "B": {"A": 0.5, "B": 1.0}}
            ),
            "2022-01-02": pd.DataFrame(
                {"A": {"A": 1.0, "B": 0.8}, "B": {"A": 0.8, "B": 1.0}}
            ),
        }

        for dt in result:
            self.assertTrue(result[dt].equals(expected_result[dt]))


class TestGetFirstUsableDate(unittest.TestCase):
    def setUp(self):
        self.est_freqs = ["D"]
        self.lback_periods = [15]
        self.rebal_freq = "M"
        self.idx = pd.bdate_range(start="2010-01-01", end="2013-01-01")
        self.rebal_dates = get_sops(dates=self.idx, freq=self.rebal_freq)

    def tearDown(self): ...

    def _pivot(self, starts: Dict[str, str]) -> pd.DataFrame:
        # build a wide panel of ones, NaN before each column's start date
        df = pd.DataFrame(1.0, index=self.idx, columns=list(starts.keys()))
        for col, start in starts.items():
            df.loc[df.index < pd.Timestamp(start), col] = np.nan
        return df

    def _first_usable(
        self,
        pivot_returns: pd.DataFrame,
        est_freqs: List[str] = None,
        lback_periods: List[int] = None,
    ) -> pd.Series:
        return _get_first_usable_date(
            pivot_returns=pivot_returns,
            rebal_dates=self.rebal_dates,
            est_freqs=est_freqs or self.est_freqs,
            lback_periods=lback_periods or self.lback_periods,
        )

    def _expected(self, ret_start: str, buffer_bdays: int) -> pd.Timestamp:
        # first rebalance date at or after the contract's return start plus buffer
        return self.rebal_dates[
            self.rebal_dates >= pd.Timestamp(ret_start) + pd.offsets.BDay(buffer_bdays)
        ].min()

    def test_each_contract_uses_its_own_return_start(self):
        pivot_returns = self._pivot({"USD_EQ": "2010-01-01", "EUR_EQ": "2011-06-01"})

        res = self._first_usable(pivot_returns)

        self.assertEqual(res["USD_EQ"], self._expected("2010-01-01", 15))
        self.assertEqual(res["EUR_EQ"], self._expected("2011-06-01", 15))
        self.assertGreater(res["EUR_EQ"], res["USD_EQ"])

    def test_multi_frequency_takes_the_longest_buffer(self):
        # with several estimation frequencies, the longest window must be chosen
        ret_start = "2010-01-01"
        pivot_returns = self._pivot({"USD_EQ": ret_start, "EUR_EQ": ret_start})

        res = self._first_usable(
            pivot_returns, est_freqs=["D", "BME"], lback_periods=[15, 3]
        )

        buffer_bdays = max(15 * _bdays_per_period("D"), 3 * _bdays_per_period("BME"))
        self.assertEqual(res["USD_EQ"], self._expected(ret_start, buffer_bdays))

    def test_full_lookback_requires_twice_as_many_periods_as_contracts(self):
        ret_start = "2010-01-01"
        pivot_returns = self._pivot(
            {f"C{i}_EQ": ret_start for i in range(4)}
        )
        n_fids = pivot_returns.shape[1]

        for est_freq in ("D", "W-FRI"):
            res = self._first_usable(
                pivot_returns, est_freqs=[est_freq], lback_periods=[-1]
            )
            buffer_bdays = 2 * n_fids * _bdays_per_period(est_freq)
            self.assertEqual(res["C0_EQ"], self._expected(ret_start, buffer_bdays))

    def test_contract_without_enough_history_gets_nat(self):
        pivot_returns = self._pivot(
            {"USD_EQ": "2010-01-01", "EUR_EQ": self.idx[-2].strftime("%Y-%m-%d")}
        )

        res = self._first_usable(pivot_returns)

        self.assertTrue(pd.isna(res["EUR_EQ"]))
        self.assertFalse(pd.isna(res["USD_EQ"]))


class TestCalculatePortfolioVolatility(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.idx = pd.bdate_range("2015-01-01", periods=400)
        self.fids = ["AUD_FX", "CAD_FX", "GBP_FX"]
        self.returns = pd.DataFrame(
            rng.normal(0, [0.2, 1.0, 3.0], size=(len(self.idx), 3)),
            index=self.idx,
            columns=self.fids,
        )
        self.signals = pd.DataFrame(1.0, index=self.idx, columns=self.fids)

        self.good_args = dict(
            pivot_returns=self.returns,
            pivot_signals=self.signals,
            weights_func=flat_weights_arr,
            rebal_freq="M",
            cov_freq="M",
            est_freqs=["D"],
            est_weights=[1.0],
            half_life=[10],
            lback_periods=[60],
            lback_min_obs=[1],
            nan_tolerance=0.1,
            remove_zeros=False,
            portfolio_return_name="PORTFOLIO",
        )

    def tearDown(self): ...

    def _vol(self, **overrides):
        pvol, _ = _calculate_portfolio_volatility(**{**self.good_args, **overrides})
        return pvol

    def test_output_properties(self):
        """Test the output dfs have the correct properties"""
        pvol, vcv = _calculate_portfolio_volatility(**self.good_args)

        rebal_dates = get_sops(dates=self.signals.index, freq=self.good_args["rebal_freq"])
        self.assertTrue(pvol.index.equals(pd.DatetimeIndex(rebal_dates)))
        self.assertEqual(pvol.index.name, "real_date")
        self.assertEqual(pvol.columns.tolist(), ["PORTFOLIO"])

        vals = pvol["PORTFOLIO"].to_numpy()
        self.assertTrue(np.all(vals[~np.isnan(vals)] >= 0))

        self.assertEqual(vcv.columns.tolist(), ["fid1", "fid2", "real_date", "value"])
        self.assertFalse(vcv["value"].isna().any())
        self.assertTrue(vcv["real_date"].isin(rebal_dates).all())

    def test_single_fid_vol_equals_annualized_return_vol(self):
        """
        Test that with one fid holding a unit signal the portfolio volatility is just
        that fid's annualized return volatility over the lookback window
        """
        returns = self.returns[["GBP_FX"]]
        signals = self.signals[["GBP_FX"]]
        pvol = self._vol(pivot_returns=returns, pivot_signals=signals)

        observed = pvol["PORTFOLIO"].dropna()
        self.assertGreater(len(observed), 0)

        expected = {}
        lback_periods = self.good_args["lback_periods"][0]
        est_freq = self.good_args["est_freqs"][0]
        for date in observed.index:
            window = returns.loc[
                (returns.index >= date - pd.offsets.BDay(lback_periods))
                & (returns.index < date)
            ]
            self.assertEqual(len(window), lback_periods)
            cov = estimate_variance_covariance(
                piv_ret=_downsample_returns(window, freq=est_freq),
                remove_zeros=False,
                weights_func=flat_weights_arr,
                lback_periods=lback_periods,
                half_life=10,
                lback_min_obs=1,
            )
            assert cov.shape == (1, 1)
            expected[date] = np.sqrt(ANNUALIZATION_FACTORS["D"] * cov.iloc[0, 0])
        np.testing.assert_allclose(
            observed.to_numpy(),
            np.array([expected[d] for d in observed.index]),
            rtol=1e-12,
        )

    def test_pvol_matches_returned_covariances(self):
        """
        Test that the reported volatility is the quadratic form of the signals against
        the covariance matrix that is reported alongside it
        """
        pvol, vcv = _calculate_portfolio_volatility(**self.good_args)
        pos = {fid: i for i, fid in enumerate(self.fids)}

        for date, block in vcv.groupby("real_date"):
            cov = np.zeros((len(self.fids), len(self.fids)))
            for fid1, fid2, value in block[["fid1", "fid2", "value"]].itertuples(
                index=False
            ):
                cov[pos[fid1], pos[fid2]] = cov[pos[fid2], pos[fid1]] = value
            sig = self.signals.loc[date, self.fids].to_numpy()

            np.testing.assert_allclose(
                pvol.loc[date, "PORTFOLIO"], np.sqrt(sig @ cov @ sig), rtol=1e-12
            )

    def test_scaling_signals_scales_vol(self):
        """Test that scaling signals scales volatility"""
        unscaled = self._vol()["PORTFOLIO"].to_numpy()
        scaled_pos = self._vol(pivot_signals=self.signals * 2)["PORTFOLIO"].to_numpy()
        scaled_neg = self._vol(pivot_signals=self.signals * -5)["PORTFOLIO"].to_numpy()

        np.testing.assert_allclose(scaled_pos, 2 * unscaled, rtol=1e-12)
        np.testing.assert_allclose(scaled_neg, 5 * unscaled, rtol=1e-12)

    def test_offsetting_positions_have_no_vol(self):
        returns = self.returns[["GBP_FX"]].copy()
        returns["GBP_FX_CLONE"] = returns["GBP_FX"]
        signals = pd.DataFrame({"GBP_FX": 1.0, "GBP_FX_CLONE": -1.0}, index=self.idx)

        pvol = self._vol(pivot_returns=returns, pivot_signals=signals)
        self.assertTrue(np.all(np.nan_to_num(pvol["PORTFOLIO"].to_numpy()) < 1e-8))

    def test_zero_signals_give_zero_vol_where_estimated(self):
        """
        Test that a portfolio wit 0 signals has no volatility, which is not the same
        as having no estimate (NaN)
        """
        pvol, vcv = _calculate_portfolio_volatility(
            **{**self.good_args, "pivot_signals": self.signals * 0.0}
        )

        estimated = pvol.index.isin(vcv["real_date"].unique())
        self.assertGreater(estimated.sum(), 0)
        self.assertGreater((~estimated).sum(), 0)
        np.testing.assert_allclose(pvol.loc[estimated, "PORTFOLIO"].to_numpy(), 0.0)
        self.assertTrue(pvol.loc[~estimated, "PORTFOLIO"].isna().all())

    def test_result_invariant_to_signal_column_order(self):
        """
        Test signals are matched to the covariance axes by contract, not by the order
        in which the caller happens to supply the columns
        """
        signals = self.signals * np.array([1.0, -2.0, 0.5])

        base = self._vol(pivot_signals=signals)
        permuted = self._vol(pivot_signals=signals[signals.columns[::-1]])

        np.testing.assert_allclose(
            permuted["PORTFOLIO"].to_numpy(),
            base["PORTFOLIO"].to_numpy(),
            rtol=1e-12,
        )

    def test_mismatched_fids_raise(self):
        """Test signals and returns with different fids raises error"""
        signals = self.signals.rename(columns={"CAD_FX": "CAD_EQ"})
        with self.assertRaises(ValueError):
            self._vol(pivot_signals=signals)

    def test_no_lookahead(self):
        """
        Test pvol is never estimated using a cov containing future data
        """
        # scale returns after a cutoff so if future data was used it would be obvious
        cutoff_dt = self.idx[250]
        shocked = self.returns.copy()
        shocked.loc[shocked.index > cutoff_dt] *= 100

        base = self._vol()["PORTFOLIO"]
        after = self._vol(pivot_returns=shocked)["PORTFOLIO"]

        early = base.index <= cutoff_dt
        self.assertGreater(early.sum(), 0)
        np.testing.assert_allclose(
            after[early].to_numpy(),
            base[early].to_numpy(),
            rtol=1e-12,
        )
        self.assertFalse(np.allclose(after.to_numpy(), base.to_numpy()))

    def test_stale_covariance_reused_between_estimates(self):
        """
        Test that rebalancing more often than the covariance is re-estimated
        reuses the most recent estimate until the next one is produced
        """
        _, vcv = _calculate_portfolio_volatility(
            **{**self.good_args, "rebal_freq": "W", "cov_freq": "Q"}
        )
        vcv_wide = vcv.pivot_table(
            index="real_date", columns=["fid1", "fid2"], values="value"
        )
        quarters = vcv_wide.index.to_period("Q")

        self.assertTrue((vcv_wide.groupby(quarters).nunique().max(axis=1) == 1).all())
        self.assertGreater(vcv_wide.groupby(quarters).ngroups, 1)
        self.assertGreater(vcv_wide.round(12).drop_duplicates().shape[0], 1)

    def test_warmup_dates_have_no_volatility(self):
        """
        Test early rebalance dates that don't have enough history to estimate
        a cov matrix hav NaN for volatility
        """
        pvol, vcv = _calculate_portfolio_volatility(**self.good_args)

        warmup = pvol.index < vcv["real_date"].min()
        self.assertGreater(warmup.sum(), 0)
        self.assertTrue(pvol.loc[warmup, "PORTFOLIO"].isna().all())

    def test_late_starting_fid_enters_when_it_has_history(self):

        returns = self.returns.copy()
        returns.loc[returns.index < self.idx[200], "CAD_FX"] = np.nan

        _, vcv = _calculate_portfolio_volatility(
            **{**self.good_args, "pivot_returns": returns}
        )
        cad = vcv[(vcv["fid1"] == "CAD_FX") | (vcv["fid2"] == "CAD_FX")]
        others = vcv[(vcv["fid1"] != "CAD_FX") & (vcv["fid2"] != "CAD_FX")]

        self.assertGreater(len(cad), 0)
        self.assertGreater(cad["real_date"].min(), others["real_date"].min())

    def test_est_weights_blend_variances_linearly(self):
        """
        Test that a multi freq estimation is a linear combination of
        single freq estimates
        """
        daily = self._vol(
            est_freqs=["D"], est_weights=[1.0], lback_periods=[60], half_life=[10]
        )["PORTFOLIO"]
        weekly = self._vol(
            est_freqs=["W"], est_weights=[1.0], lback_periods=[12], half_life=[4]
        )["PORTFOLIO"]
        blend = self._vol(
            est_freqs=["D", "W"],
            est_weights=[0.25, 0.75],
            lback_periods=[60, 12],
            half_life=[10, 4],
            lback_min_obs=[1, 1],
        )["PORTFOLIO"]

        defined = ~(daily.isna() | weekly.isna() | blend.isna())
        np.testing.assert_allclose(
            np.square(blend[defined].to_numpy()),
            0.25 * np.square(daily[defined].to_numpy())
            + 0.75 * np.square(weekly[defined].to_numpy()),
            rtol=1e-12,
        )

    def test_lookback_too_short_for_fid_count_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            expected_regex=r"`lback_periods` of 2 at est_freq 'W' cannot support "
                           r"3 contracts",
        ):
            self._vol(est_freqs=["W"], lback_periods=[2], half_life=[2])

    def test_lookback_is_checked_before_any_estimation(self):
        """
        The contract count grows as contracts accumulate history, so a lookback that is
        adequate at the first estimation dates can become impossible later
        """
        returns = self.returns.copy()
        # only AUD is available early on, where a 2-period lookback would be fine
        returns.loc[returns.index < self.idx[200], "CAD_FX"] = np.nan
        returns.loc[returns.index < self.idx[300], "GBP_FX"] = np.nan

        with mock.patch(
            "macrosynergy.pnl.historic_portfolio_volatility"
            "._calculate_multi_frequency_vcv_for_period"
        ) as mock_vcv:
            with self.assertRaises(ValueError):
                self._vol(
                    pivot_returns=returns,
                    est_freqs=["W"],
                    lback_periods=[2],
                    half_life=[2],
                )

        mock_vcv.assert_not_called()

    def test_recovers_known_annualised_vol(self):
        base_vol = np.array([0.01, 0.02, 0.05])
        dg = SignalsAndReturnsGenerator(
            n_fids=3, corr=np.eye(3), vol_of_vol=0, base_vol=base_vol
        )
        dg.simulate_signals_and_returns(
            n_periods=2000,
            signal_names=["CAD_FX", "GBP_FX", "JPY_FX"],
            return_names=["CAD_FX", "GBP_FX", "JPY_FX"],
            seed=29,
        )
        signals = dg.signals * 0 + 1 / 3 # equally weighted signals

        pvol, vcv = _calculate_portfolio_volatility(
            **{
                **self.good_args,
                "pivot_returns": dg.returns,
                "pivot_signals": signals,
                "rebal_freq": "W",
                "est_freqs": ["D", "W"],
                "est_weights": [0.5, 0.5],
                "half_life": [10, 2],
                "lback_periods": [60, 35],
                "lback_min_obs": [1, 1],
                "remove_zeros": True,
            }
        )

        expected = np.sqrt(252) * np.sqrt((1 / 9) * np.square(base_vol).sum())
        np.testing.assert_allclose(pvol["PORTFOLIO"].mean(), expected, rtol=0.07)

        # independent returns leave no systematic cross-contract covariance
        off_diag = vcv[vcv["fid1"] != vcv["fid2"]]["value"]
        self.assertTrue(np.isclose(off_diag.mean(), 0, atol=0.01))


class TestHistVolFunc(unittest.TestCase):
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
        self.portfolio_return_name = f"SNAME{RETURN_SERIES_XCAT}"
        self._dft = _dft

    @property
    def good_args(self):
        return {
            "pivot_returns": self._dft,
            "pivot_signals": self._dft,
            "sname": "SNAME",
            "rebal_freq": "M",
            "lback_meth": "ma",
            "lback_periods": [15, 6],
            "half_life": [10, 2],
            "lback_min_obs": [1, 1],
            "est_freqs": ["D", "W"],
            "est_weights": [0.5, 0.5],
            "nan_tolerance": 0.1,
            "remove_zeros": True,
            "return_variance_covariance": True,
        }

    def tearDown(self): ...

    def test_basic(self):
        # Test good args
        res = _hist_vol(**self.good_args)
        self.assertTrue(isinstance(res, list))
        self.assertEqual(len(res), 2)
        self.assertTrue(isinstance(res[0], pd.DataFrame))
        self.assertTrue(isinstance(res[1], pd.DataFrame))

        # check that the first dataframe is indexed with real_date
        self.assertTrue(isinstance(res[0].index, pd.DatetimeIndex))
        self.assertTrue(res[0].index.name == "real_date")
        # check that the first dataframe has 1 column called portfolio_return_name
        self.assertTrue(res[0].columns.tolist() == [self.portfolio_return_name])

        # test when called with return_variance_covariance=False
        res = _hist_vol(**{**self.good_args, "return_variance_covariance": False})
        self.assertTrue(isinstance(res, list))
        self.assertEqual(len(res), 1)
        self.assertTrue(isinstance(res[0], pd.DataFrame))
        # same checks on res0
        self.assertTrue(isinstance(res[0].index, pd.DatetimeIndex))
        self.assertTrue(res[0].index.name == "real_date")
        self.assertTrue(res[0].columns.tolist() == [self.portfolio_return_name])

    def test_fails(self):
        for lbmeth in ["ma", "xma"]:
            _hist_vol(**{**self.good_args, "lback_meth": lbmeth})
        for lbmeth in ["abc", "xyz"]:
            with self.assertRaises(NotImplementedError):
                _hist_vol(**{**self.good_args, "lback_meth": lbmeth})

    def _run_with_nan_dates(self, dates, nan_dates, **kwargs):
        """
        Run `_hist_vol` against a volatility series that has no estimate on `nan_dates`.
        """
        pvol = pd.DataFrame(1.0, index=dates, columns=[self.portfolio_return_name])
        pvol.loc[nan_dates] = np.nan
        with mock.patch(
            "macrosynergy.pnl.historic_portfolio_volatility"
            "._calculate_portfolio_volatility",
            side_effect=lambda **_: [pvol.copy(), None],
        ):
            return _hist_vol(**{**self.good_args, **kwargs})

    def test_dates_without_an_estimate_are_dropped(self):
        """
        A rebalance date the covariance history could not estimate arrives as NaN and
        must not reach the caller, while the dates that were estimated pass through.
        """
        dates = self._dft.index

        res = self._run_with_nan_dates(dates, dates[:5])
        self.assertEqual(res[0].index.tolist(), dates[5:].tolist())
        self.assertFalse(res[0][self.portfolio_return_name].isna().any())

        # nothing estimated anywhere leaves an empty frame, not a column of NaNs
        res = self._run_with_nan_dates(dates, dates)
        self.assertTrue(res[0].empty)

    def test_missing_estimates_are_reported_at_info(self):
        """
        Every warm-up date produces a NaN, so dropping them is routine rather than
        exceptional: it is reported, but must not raise a warning on an ordinary run.
        """
        module = "macrosynergy.pnl.historic_portfolio_volatility"
        dates = self._dft.index

        with self.assertLogs(module, level="INFO") as captured:
            self._run_with_nan_dates(dates, dates[:5])

        self.assertTrue(
            any("dropping all NaNs" in msg for msg in captured.output),
            "dropped rebalance dates were not reported at all",
        )
        self.assertEqual(
            [r.getMessage() for r in captured.records if r.levelno >= logging.WARNING],
            [],
        )


class TestHistVolEntrypoint(unittest.TestCase):
    def test_main(self):
        cids: List[str] = ["EUR", "GBP", "AUD", "CAD"]
        xcats: List[str] = ["EQ"]
        ctypes = xcats.copy()
        start: str = "2000-01-01"
        xr_tickers = [f"{cid}_{xcat}XR" for cid in cids for xcat in xcats]
        cs_tickers = [f"{cid}_{xcat}_CSIG_STRAT" for cid in cids for xcat in xcats]
        fids: List[str] = [f"{cid}_{ctype}" for cid in cids for ctype in ctypes]
        df = simulate_returns_and_signals(
            cids=cids,
            xcat=xcats[0],
            return_suffix="XR",
            signal_suffix="CSIG_STRAT",
            start=start,
            years=5,
        )
        end = df["real_date"].max().strftime("%Y-%m-%d")
        all_args = dict(
            df=df,
            sname="STRAT",
            fids=fids,
            rebal_freq="m",
            est_freqs=["D", "W", "M"],
            est_weights=[0.1, 0.2, 0.7],
            lback_periods=[30, 20, -1],
            half_life=[10, 5, 2],
            lback_min_obs=[1, 1, 1],
            lback_meth="xma",
            rstring="XR",
            start=start,
            end=end,
            return_variance_covariance=True,
        )

        df_vol, vcv_df = historic_portfolio_vol(**all_args)

        self.assertTrue(isinstance(df_vol, QuantamentalDataFrame))
        self.assertTrue(isinstance(vcv_df, pd.DataFrame))
        tdf = qdf_to_ticker_df(df_vol)
        self.assertTrue(tdf.columns.tolist() == [f"STRAT{RETURN_SERIES_XCAT}"])

        self.assertEqual(
            set(vcv_df.columns.tolist()), set(["fid1", "fid2", "value", "real_date"])
        )

        df_vol = historic_portfolio_vol(
            **{**all_args, "return_variance_covariance": False}
        )
        self.assertTrue(isinstance(df_vol, QuantamentalDataFrame))

        # test with 'difficult' args
        historic_portfolio_vol(
            **{
                **all_args,
                "lback_periods": 30,
                "half_life": 10,
                "est_weights": 0.8,
                "est_freqs": "D",
                "start": None,
                "end": None,
            }
        )

        # test raises TypeError with start=123
        with self.assertRaises(TypeError):
            historic_portfolio_vol(**{**all_args, "start": 123})

        for argx, inpx in zip(["start", "end"], ["5006-59-01", "2024-14-14"]):
            with self.assertRaises(ValueError):
                historic_portfolio_vol(**{**all_args, argx: inpx})


class TestCovMatrixHistory(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.idx = pd.bdate_range("2015-01-01", periods=500)
        self.fids = ["AUD_FX", "CAD_FX", "GBP_FX"]
        self.returns = pd.DataFrame(
            rng.normal(0, [0.5, 1.0, 2.0], size=(len(self.idx), 3)),
            index=self.idx,
            columns=self.fids,
        )
        self.estimation_dates = get_sops(dates=self.idx, freq="M")

        self.good_args = dict(
            pivot_returns=self.returns,
            estimation_dates=self.estimation_dates,
            est_freqs=["D"],
            est_weights=[1.0],
            lback_periods=[60],
            half_life=[10],
            nan_tolerance=0.1,
            remove_zeros=False,
            weights_func=flat_weights_arr,
            lback_min_obs=[1],
        )

    def tearDown(self): ...

    def _first_usable(self, **overrides) -> pd.Series:
        args = {**self.good_args, **overrides}
        return _get_first_usable_date(
            pivot_returns=args["pivot_returns"],
            rebal_dates=args["estimation_dates"],
            est_freqs=args["est_freqs"],
            lback_periods=args["lback_periods"],
        )

    def test_shape_and_symmetry(self):
        """Test output has correct dimensions and each cov matrix is symmetric"""
        history = _cov_matrix_history(**self.good_args)

        self.assertEqual(
            history.shape,
            (len(self.estimation_dates), len(self.fids), len(self.fids)),
        )

        estimated = ~np.isnan(history).all(axis=(1, 2))

        self.assertGreater(estimated.sum(), 0)
        for mat in history[estimated]:
            np.testing.assert_allclose(mat, mat.T, rtol=1e-12)
            self.assertTrue(np.all(np.diag(mat) >= 0))

    def test_warmup_covs_are_nan(self):
        """
        Test cov matrices on dates with insufficient history are all nan
        """
        history = _cov_matrix_history(**self.good_args)
        first_usable = self._first_usable().min()

        warmup = np.asarray(self.estimation_dates) < np.datetime64(first_usable)
        self.assertGreater(warmup.sum(), 0)
        self.assertTrue(np.isnan(history[warmup]).all())
        self.assertFalse(np.isnan(history[~warmup]).any())

    def test_slice_matches_single_period_estimate(self):
        """
        Test each slice is the vcv that the per-period helper produces for that date
        """
        history = _cov_matrix_history(**self.good_args)
        estimated = np.flatnonzero(~np.isnan(history).all(axis=(1, 2)))

        for i in (estimated[0], estimated[len(estimated) // 2], estimated[-1]):
            expected = _calculate_multi_frequency_vcv_for_period(
                pivot_returns=self.returns,
                rebal_date=self.estimation_dates.iloc[i],
                est_freqs=self.good_args["est_freqs"],
                est_weights=self.good_args["est_weights"],
                weights_func=self.good_args["weights_func"],
                lback_periods=self.good_args["lback_periods"],
                half_life=self.good_args["half_life"],
                nan_tolerance=self.good_args["nan_tolerance"],
                remove_zeros=self.good_args["remove_zeros"],
                lback_min_obs=self.good_args["lback_min_obs"],
            )
            np.testing.assert_allclose(
                history[i],
                expected.loc[self.fids, self.fids].to_numpy(),
                rtol=1e-12,
            )

    def test_axes_follow_pivot_returns_column_order(self):
        """
        Test that even if contracts become available at different dates, the axes of
        every slice stays in the same order as pivot_returns.columns
        """
        returns = self.returns.copy()
        returns.loc[returns.index < self.idx[300], "AUD_FX"] = np.nan

        history = _cov_matrix_history(**{**self.good_args, "pivot_returns": returns})
        aud = returns.columns.get_loc("AUD_FX")

        first_usable = self._first_usable(pivot_returns=returns)
        aud_starts = np.asarray(self.estimation_dates) >= np.datetime64(
            first_usable["AUD_FX"]
        )
        others_started = np.asarray(self.estimation_dates) >= np.datetime64(
            first_usable.drop("AUD_FX").min()
        )

        # before AUD is usable the other contracts are already estimated, so its
        # row/column is the only NaN part of the slice
        pre = others_started & ~aud_starts
        self.assertGreater(pre.sum(), 0)
        self.assertTrue(np.isnan(history[pre][:, aud, :]).all())
        self.assertTrue(np.isnan(history[pre][:, :, aud]).all())
        kept = [i for i in range(len(self.fids)) if i != aud]
        self.assertFalse(np.isnan(history[pre][:, kept][:, :, kept]).any())

        # once usable, its variance lands on the AUD diagonal position
        self.assertFalse(np.isnan(history[aud_starts][:, aud, aud]).any())

    def test_est_weights_blend_frequencies(self):
        """
        Test a blended history is the weighted sum of the single-frequency histories
        """
        daily = _cov_matrix_history(
            **{**self.good_args, "est_freqs": ["D"], "est_weights": [1.0],
               "lback_periods": [60], "half_life": [10]}
        )
        weekly = _cov_matrix_history(
            **{**self.good_args, "est_freqs": ["W-FRI"], "est_weights": [1.0],
               "lback_periods": [12], "half_life": [4]}
        )
        blend = _cov_matrix_history(
            **{**self.good_args, "est_freqs": ["D", "W-FRI"],
               "est_weights": [0.25, 0.75], "lback_periods": [60, 12],
               "half_life": [10, 4], "lback_min_obs": [1, 1]}
        )

        defined = ~(np.isnan(daily) | np.isnan(weekly) | np.isnan(blend))
        self.assertTrue(defined.any())
        np.testing.assert_allclose(
            blend[defined],
            0.25 * daily[defined] + 0.75 * weekly[defined],
            rtol=1e-12,
        )


class TestNanTolerance(unittest.TestCase):
    TOL = 0.25

    def setUp(self):
        self.idx = pd.bdate_range("2020-01-01", periods=200)
        self.fids = ["AUD_FX", "CAD_FX", "GBP_FX"]
        rng = np.random.default_rng(17)
        self.returns = pd.DataFrame(
            rng.normal(0, 1, (len(self.idx), 3)), index=self.idx, columns=self.fids
        )

    def _kept(self, returns: pd.DataFrame) -> List[str]:
        """The contracts that survive the tolerance filter, as the estimator sees it."""
        frac = _nan_frac_since_first_obs(returns)
        return returns.columns[frac <= self.TOL].tolist()

    def test_leading_gap_does_not_count_against_a_late_starter(self):
        """
        A contract that simply starts late has no missing observations of its own. If
        the fraction were taken across the whole window it would breach the tolerance
        and the contract would never be traded, however much history it accumulates.
        """
        returns = self.returns.copy()
        returns.loc[returns.index[:120], "CAD_FX"] = np.nan

        # the whole-window measure would reject it outright
        self.assertGreater(returns["CAD_FX"].isna().mean(), self.TOL)

        self.assertEqual(_nan_frac_since_first_obs(returns)["CAD_FX"], 0.0)
        self.assertIn("CAD_FX", self._kept(returns))

    def test_scattered_gaps_do_count_against_a_contract(self):
        """Holes after a contract has started are real missing data."""
        returns = self.returns.copy()
        returns.iloc[::3, returns.columns.get_loc("CAD_FX")] = np.nan

        frac = _nan_frac_since_first_obs(returns)["CAD_FX"]
        self.assertAlmostEqual(frac, 1 / 3, places=2)
        self.assertNotIn("CAD_FX", self._kept(returns))

    def test_every_contract_breaching_leaves_no_estimate_for_the_date(self):
        """
        If the tolerance empties the window there is nothing to estimate, and the date
        yields no covariance matrix at all rather than a partial one.
        """
        returns = self.returns.copy()
        returns.iloc[::3] = np.nan  # every contract over the tolerance

        vcv = _calculate_multi_frequency_vcv_for_period(
            pivot_returns=returns,
            rebal_date=self.idx[-1],
            est_freqs=["B"],
            est_weights=[1.0],
            weights_func=flat_weights_arr,
            lback_periods=[150],
            half_life=[10],
            nan_tolerance=self.TOL,
            remove_zeros=False,
            lback_min_obs=[1],
        )

        self.assertTrue(vcv.empty)

    def test_breaching_contract_is_absent_from_the_estimated_matrix(self):
        """
        a contract over the tolerance does not appear on either axis of
        the covariance matrix for that date.
        """
        returns = self.returns.copy()
        returns.iloc[::3, returns.columns.get_loc("CAD_FX")] = np.nan

        vcv = _calculate_multi_frequency_vcv_for_period(
            pivot_returns=returns,
            rebal_date=self.idx[-1],
            est_freqs=["B"],
            est_weights=[1.0],
            weights_func=flat_weights_arr,
            lback_periods=[150],
            half_life=[10],
            nan_tolerance=self.TOL,
            remove_zeros=False,
            lback_min_obs=[1],
        )

        self.assertNotIn("CAD_FX", vcv.columns)
        self.assertNotIn("CAD_FX", vcv.index)
        self.assertEqual(vcv.columns.tolist(), ["AUD_FX", "GBP_FX"])

    def test_late_starter_reaches_the_estimated_matrix(self):
        returns = self.returns.copy()
        returns.loc[returns.index[:120], "CAD_FX"] = np.nan

        vcv = _calculate_multi_frequency_vcv_for_period(
            pivot_returns=returns,
            rebal_date=self.idx[-1],
            est_freqs=["B"],
            est_weights=[1.0],
            weights_func=flat_weights_arr,
            lback_periods=[150],
            half_life=[10],
            nan_tolerance=self.TOL,
            remove_zeros=False,
            lback_min_obs=[1],
        )

        self.assertIn("CAD_FX", vcv.columns)
        self.assertFalse(np.isnan(vcv.loc["CAD_FX", "CAD_FX"]))


class TestFrequencyBlend(unittest.TestCase):
    FIDS = ["AUD_FX", "CAD_FX", "GBP_FX"]

    @staticmethod
    def _matrix(values, fids):
        return pd.DataFrame(np.array(values, dtype=float), index=fids, columns=fids)

    def _blend(self, dict_vcv, est_weights=(0.25, 0.75)):
        return _blend_frequency_vcvs(
            dict_vcv=dict_vcv,
            est_freqs=list(dict_vcv),
            est_weights=list(est_weights),
            column_order=pd.Index(self.FIDS),
        )

    def test_blend_is_the_annualized_weighted_mean(self):
        """With every entry estimated everywhere, no renormalization is in play."""
        fids = self.FIDS[:2]
        daily = self._matrix([[1.0, 0.5], [0.5, 2.0]], fids)
        monthly = self._matrix([[3.0, 1.0], [1.0, 4.0]], fids)

        out = self._blend({"D": daily, "M": monthly})

        expected = (
            0.25 * ANNUALIZATION_FACTORS["D"] * daily
            + 0.75 * ANNUALIZATION_FACTORS["M"] * monthly
        )
        np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), rtol=1e-12)

    def test_entry_missing_at_one_frequency_uses_the_survivor_alone(self):
        """
        The surviving weight is renormalized to 1 for that entry, so the result is the
        other frequency's annualized estimate rather than a fraction of it.
        """
        fids = self.FIDS[:2]
        daily = self._matrix([[1.0, np.nan], [np.nan, 2.0]], fids)
        monthly = self._matrix([[3.0, 1.0], [1.0, 4.0]], fids)

        out = self._blend({"D": daily, "M": monthly})

        # off-diagonal: monthly only, renormalized - not 0.75 * the monthly value
        np.testing.assert_allclose(
            out.loc["AUD_FX", "CAD_FX"], ANNUALIZATION_FACTORS["M"] * 1.0, rtol=1e-12
        )
        # diagonal: both available, so the plain weighted mean
        np.testing.assert_allclose(
            out.loc["AUD_FX", "AUD_FX"],
            0.25 * ANNUALIZATION_FACTORS["D"] * 1.0
            + 0.75 * ANNUALIZATION_FACTORS["M"] * 3.0,
            rtol=1e-12,
        )

    def test_contract_dropped_at_one_frequency_survives_via_another(self):
        kept = self.FIDS[:2]
        daily = self._matrix([[1.0, 0.5], [0.5, 2.0]], kept)  # GBP_FX absent
        monthly = self._matrix(
            [[3.0, 1.0, 0.2], [1.0, 4.0, 0.3], [0.2, 0.3, 5.0]], self.FIDS
        )

        out = self._blend({"D": daily, "M": monthly})

        self.assertEqual(out.columns.tolist(), self.FIDS)
        self.assertFalse(np.isnan(out.loc["GBP_FX", "GBP_FX"]))
        np.testing.assert_allclose(
            out.loc["GBP_FX", "GBP_FX"], ANNUALIZATION_FACTORS["M"] * 5.0, rtol=1e-12
        )

    def test_entry_estimated_at_no_frequency_stays_nan(self):
        """Renormalization must not invent an estimate where there is none."""
        fids = self.FIDS[:2]
        daily = self._matrix([[1.0, np.nan], [np.nan, 2.0]], fids)
        monthly = self._matrix([[3.0, np.nan], [np.nan, 4.0]], fids)

        out = self._blend({"D": daily, "M": monthly})

        self.assertTrue(np.isnan(out.loc["AUD_FX", "CAD_FX"]))
        self.assertFalse(np.isnan(out.loc["AUD_FX", "AUD_FX"]))

    def test_axes_follow_the_requested_column_order(self):
        """Signals are paired with the covariance axes positionally downstream."""
        reversed_fids = self.FIDS[::-1]
        monthly = self._matrix(np.eye(3) + 0.1, reversed_fids)

        out = _blend_frequency_vcvs(
            dict_vcv={"M": monthly},
            est_freqs=["M"],
            est_weights=[1.0],
            column_order=pd.Index(self.FIDS),
        )

        self.assertEqual(out.columns.tolist(), self.FIDS)
        self.assertEqual(out.index.tolist(), self.FIDS)


if __name__ == "__main__":
    unittest.main()
