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


if __name__ == "__main__":
    unittest.main()
