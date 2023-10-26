import unittest
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from macrosynergy.management.simulate_quantamental_data import make_test_df
from macrosynergy.management.utils import qdf_to_ticker_df
from macrosynergy.panel.granger_causality_tests import (
    _get_tickers,
    _granger_causality_backend,
    granger_causality_test,
)


class TestGrangerCausality(unittest.TestCase):
    def setUp(self) -> None:
        self.cids: List[str] = ["USD", "EUR", "JPY", "GBP", "CAD"]
        self.xcats: List[str] = ["EQ", "IR", "FX", "CMD"]
        self.tickers: List[str] = [
            f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats
        ]
        self.df: pd.DataFrame = make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start="2000-01-01",
            end="2020-01-01",
        )

    def test_type_checks(self):
        # df: pd.DataFrame
        # tickers: Optional[List[str]]
        # cids: Optional[List[str]]
        # xcats: Optional[List[str]]
        # max_lag: Union[int, List[int]]
        # add_constant: bool
        # start: Optional[str]
        # end: Optional[str]
        # freq: str
        # agg: str
        # metric: str

        good_args: Dict[str, Union[pd.DataFrame, List[str], List[int], bool, str]] = {
            "df": self.df,
            "cids": self.cids[0],
            "xcats": self.xcats[:2],
            "tickers": None,
            "max_lag": 1,
            "add_constant": True,
            "start": "2000-01-01",
            "end": "2020-01-01",
            "freq": "M",
            "agg": "mean",
            "metric": "value",
        }

        # Test that good args work
        self.assertIsInstance(granger_causality_test(**good_args), dict)

        # Test bad args: all except "max_lag" should raise TypeError on integer input

        for key in good_args:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args[key] = 1
            if key == "max_lag":
                bad_args[key] = "apple"

            if key == "tickers":
                continue

            with self.assertRaises(TypeError, msg=f"Key: {key}={bad_args[key]}"):
                granger_causality_test(**bad_args)

        # Test specifying cids, xcats and tickers
        bad_args: Dict[str, Any] = good_args.copy()
        _tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats][:2]
        bad_args["tickers"] = _tickers

        with self.assertRaises(ValueError):
            granger_causality_test(**bad_args)

        # Test specifying more than 2 cids
        bad_args = good_args.copy()
        bad_args["cids"] = self.cids
        with self.assertRaises(ValueError):
            granger_causality_test(**bad_args)

        # Test specifying more than 2 xcats
        bad_args = good_args.copy()
        bad_args["xcats"] = self.xcats
        with self.assertRaises(ValueError):
            granger_causality_test(**bad_args)

        # Test specifying more than 2 tickers
        bad_args = good_args.copy()
        bad_args["tickers"] = [
            f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats
        ]

        bad_args.pop("cids")
        bad_args.pop("xcats")
        with self.assertRaises(ValueError):
            granger_causality_test(**bad_args)

        # pass a list of ints for tickers
        bad_args["tickers"] = [1, 2]
        with self.assertRaises(TypeError):
            granger_causality_test(**bad_args)

        # Pass random xcats to test ValueError
        bad_args = good_args.copy()
        bad_args["xcats"] = [u[::-1] for u in self.xcats[:2]]
        with self.assertRaises(ValueError):
            granger_causality_test(**bad_args)

        # Value error when metric="return"
        bad_args = good_args.copy()
        bad_args["metric"] = "return"
        with self.assertRaises(ValueError):
            granger_causality_test(**bad_args)

        # test with max_lag as a list of string
        bad_args = good_args.copy()
        bad_args["max_lag"] = ["apple", "orange"]
        with self.assertRaises(TypeError):
            granger_causality_test(**bad_args)

        # test random strings for dates
        bad_args = good_args.copy()
        bad_args["start"] = "apple"
        with self.assertRaises(ValueError):
            granger_causality_test(**bad_args)

    def test_get_tickers(self):
        ticks: List[str] = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats]
        # Test that tickers are simply returned
        self.assertEqual(
            _get_tickers(tickers=ticks, cids=self.cids, xcats=self.xcats), ticks
        )

        # test when specifying cids and xcats, `ticks` is returned
        self.assertEqual(_get_tickers(cids=self.cids, xcats=self.xcats), ticks)

        # pass an xcat as a string
        xc: str = self.xcats[0]
        ticks: List[str] = [f"{cid}_{xc}" for cid in self.cids]
        self.assertEqual(_get_tickers(cids=self.cids, xcats=xc), ticks)

    def test_gct_backend_asserts(self):
        wdf: pd.DataFrame = qdf_to_ticker_df(self.df)

        good_args: Dict[str, Any] = {
            "data": wdf[self.tickers[:2]],
            "max_lag": 1,
            "add_constant": True,
        }

        # Test that good args work
        _granger_causality_backend(**good_args)

        # Test whether the full wdf works
        bad_args: Dict[str, Any] = good_args.copy()
        bad_args["data"] = wdf
        with self.assertRaises(AssertionError):
            _granger_causality_backend(**bad_args)

        # Test whether the wrong max_lag works
        bad_args = good_args.copy()
        bad_args["max_lag"] = "apple"
        with self.assertRaises(AssertionError):
            _granger_causality_backend(**bad_args)

        # Test whether it works with a list of ints for max_lag
        good_args_2: Dict[str, Any] = good_args.copy()
        good_args_2["max_lag"] = [1, 1]
        # check if the output is a dict
        self.assertIsInstance(_granger_causality_backend(**good_args_2), dict)

        # Should not work with a list of strings or an empty list for max_lag
        bad_args = good_args.copy()
        bad_args["max_lag"] = ["apple", "orange"]
        with self.assertRaises(AssertionError):
            _granger_causality_backend(**bad_args)

        bad_args = good_args.copy()
        bad_args["max_lag"] = []
        with self.assertRaises(AssertionError):
            _granger_causality_backend(**bad_args)

        # should fail is add_constant is not a bool
        bad_args = good_args.copy()
        bad_args["add_constant"] = 1
        with self.assertRaises(AssertionError):
            _granger_causality_backend(**bad_args)
