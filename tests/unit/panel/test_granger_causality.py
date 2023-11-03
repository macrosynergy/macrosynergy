import unittest, unittest.mock
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.utils import qdf_to_ticker_df
from macrosynergy.panel.granger_causality_test import (
    _get_tickers,
    _granger_causality_backend,
    granger_causality_test,
    _type_checks,
)


def mock_wrapper(*args, **kwargs):
    return {1: 1}


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
            style="linear",
        )

        good_args: Dict[str, Any] = {
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
        self.good_args: Dict[str, Any] = good_args

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
        good_args: Dict[str, Any] = self.good_args.copy()
        # Test that good args work
        self.assertTrue(_type_checks(**good_args))

        good_args2 = good_args.copy()
        good_args2["cids"] = self.cids[:2]
        good_args2["xcats"] = self.xcats[0]
        self.assertTrue(_type_checks(**good_args2))

        # Test bad args: all except "max_lag" should raise TypeError on integer input

        for key in good_args:
            bad_args: Dict[str, Any] = good_args.copy()
            bad_args[key] = 1
            if key == "max_lag":
                bad_args[key] = "apple"

            if key == "tickers":
                continue

            with self.assertRaises(TypeError, msg=f"Key: {key}={bad_args[key]}"):
                _type_checks(**bad_args)

        # Test specifying cids, xcats and tickers
        bad_args: Dict[str, Any] = good_args.copy()
        _tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats][:2]
        bad_args["tickers"] = _tickers

        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

        # Test specifying more than 2 cids
        bad_args = good_args.copy()
        bad_args["cids"] = self.cids
        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

        # Test specifying more than 2 xcats
        bad_args = good_args.copy()
        bad_args["xcats"] = self.xcats
        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

        # test only specifying cids
        bad_args = good_args.copy()
        bad_args["cids"] = None
        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

        # Test specifying more than 2 tickers
        bad_args = good_args.copy()
        bad_args["tickers"] = [
            f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats
        ]

        bad_args["cids"] = None
        bad_args["xcats"] = None
        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

        # pass a list of ints for tickers
        bad_args["tickers"] = [1, 2]
        with self.assertRaises(TypeError):
            _type_checks(**bad_args)

        # Pass random xcats to test ValueError
        bad_args = good_args.copy()
        bad_args["xcats"] = [u[::-1] for u in self.xcats[:2]]
        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

        # Value error when metric="return"
        bad_args = good_args.copy()
        bad_args["metric"] = "return"
        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

        # test with max_lag as a list of string
        bad_args = good_args.copy()
        bad_args["max_lag"] = ["apple", "orange"]
        with self.assertRaises(TypeError):
            _type_checks(**bad_args)

        # test random strings for dates
        bad_args = good_args.copy()
        bad_args["start"] = "apple"
        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

        bad_args = good_args.copy()
        bad_args["df"] = bad_args["df"][
            ~((bad_args["df"]["cid"] == "USD") & (bad_args["df"]["xcat"] == "EQ"))
        ]

        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

        # reuse df from above
        bad_args["tickers"] = ["USD_EQ", "USD_FX"]
        bad_args["cids"] = None
        bad_args["xcats"] = None
        with self.assertRaises(ValueError):
            _type_checks(**bad_args)

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
        with unittest.mock.patch(
            "macrosynergy.panel.granger_causality_test._statsmodels_compatibility_wrapper",
            mock_wrapper,
        ):
            self.assertIsInstance(_granger_causality_backend(**good_args), dict)

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
        with unittest.mock.patch(
            "macrosynergy.panel.granger_causality_test._statsmodels_compatibility_wrapper",
            mock_wrapper,
        ):
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

    def test_gct_wrapper(self):
        # all failure cases are already tested in the backend tests or utils tests
        # run with good args to see if it works

        # mock macrosynergy.panel.granger_causality_test._statsmodels_compatibility_wrapper to return a dict

        with unittest.mock.patch(
            "macrosynergy.panel.granger_causality_test._statsmodels_compatibility_wrapper",
            mock_wrapper,
        ):
            self.assertIsInstance(granger_causality_test(**self.good_args), dict)

        bad_args = self.good_args.copy()

        # for 0th cid and 0th xcat, set all values to np.nan
        dfb: pd.DataFrame = self.df.copy()
        dfb.loc[
            (dfb["cid"] == self.cids[0]) & (dfb["xcat"] == self.xcats[0]), "value"
        ] = np.nan

        bad_args["df"] = dfb
        with self.assertRaises(ValueError):
            granger_causality_test(**bad_args)

        # run a test that will run with the backend
        cids: List[str] = ["USD"]
        xcats: List[str] = ["EQ", "IR"]
        dfc: pd.DataFrame = make_test_df(
            cids=cids,
            xcats=xcats[0],
            start="2000-01-01",
            end="2020-01-01",
            style="sine",
        )

        dfc = pd.concat(
            [
                dfc,
                make_test_df(
                    cids=cids,
                    xcats=xcats[1],
                    start="2000-01-01",
                    end="2020-01-01",
                    style="sine",
                ),
            ]
        )

        test_args: Dict[str, Any] = self.good_args.copy()
        test_args["df"] = dfc
        self.assertIsInstance(granger_causality_test(**test_args), dict)

        # mock grangercausalitytests to return a dict - solely to avoid real computation
        with unittest.mock.patch(
            "statsmodels.tsa.stattools.grangercausalitytests", mock_wrapper
        ):
            # mock sm.__version__ as 0.15.0
            with unittest.mock.patch("statsmodels.__version__", "0.15.0"):
                self.assertIsInstance(granger_causality_test(**test_args), dict)
