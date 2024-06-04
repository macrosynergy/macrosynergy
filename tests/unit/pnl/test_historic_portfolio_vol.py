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
)
from macrosynergy.management.types import QuantamentalDataFrame


class TestHistoricPortfolioVol(unittest.TestCase):
    def setUp(self): ...
    def tearDown(self): ...

    def test_weighted_covariance(
        self,
    ):
        x_arr = np.arange(100) / 100
        y_arr = np.arange(100) / 100
        good_args: Dict[str, Any] = {
            "half_life": 10,
            "lback_periods": 100,
            "x": x_arr,
            "y": y_arr,
            "weights_func": expo_weights_arr,
        }

        # Test good args
        res = _weighted_covariance(**good_args)
        self.assertTrue(isinstance(res, float))
        self.assertTrue(np.isclose(res, 0.019827, atol=1e-6))

        ## Test bad args

        ## X and Y must be same length
        for argn in ["x", "y"]:
            bad_args = good_args.copy()
            bad_args[argn] = np.arange(99) / 100
            with self.assertRaises(AssertionError):
                _weighted_covariance(**bad_args)

        ## X and Y must be 1D
        for argn in ["x", "y"]:
            bad_args = good_args.copy()
            bad_args[argn] = np.arange(100).reshape((10, 10))
            with self.assertRaises(AssertionError):
                _weighted_covariance(**bad_args)

        ## For either being all nan, the result should be nan
        for argn in ["x", "y"]:
            bad_args = good_args.copy()
            bad_args[argn] = np.full(100, np.nan)
            res = _weighted_covariance(**bad_args)
            self.assertTrue(np.isnan(res))

        # @mock.patch("macrosynergy.pnl.historic_portfolio_volatility.expo_weights_arr")
        # def _test


if __name__ == "__main__":
    unittest.main()
