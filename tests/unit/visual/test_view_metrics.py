import unittest
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate import make_test_df
import macrosynergy.visuals as msv
import matplotlib
import warnings
from unittest.mock import patch
from matplotlib import pyplot as plt


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Prevents plots from being displayed during tests.
        plt.close("all")
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()
        warnings.simplefilter("ignore")

        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD"]
        self.xcats: List[str] = ["XR", "CRY", "INFL"]
        self.metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        self.start: str = "2010-01-01"
        self.end: str = "2020-12-31"

        self.df: pd.DataFrame = make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start=self.start,
            end=self.end,
            metrics=self.metrics,
        )

    @classmethod
    def tearDownClass(self) -> None:
        warnings.resetwarnings()
        plt.close("all")
        matplotlib.use(self.mpl_backend)

    def setUp(self):
        self.valid_args: Dict[str, Any] = {
            "df": self.df,
            "xcat": "XR",
            "cids": self.cids,
            "metric": "eop_lag",
            "start": self.start,
            "end": self.end,
            "freq": "M",
            "agg": "mean",
            "title": "Test plot",
            "figsize": (10, None),
        }

    def tearDown(self) -> None:
        plt.close("all")

    def test_view_metrics_no_error(self):
        try:
            msv.view_metrics(**self.valid_args)
        except Exception as e:
            self.fail(f"view_metrics raised {e} unexpectedly")

    def test_view_metrics_invalid_types(self):
        for arg in self.valid_args:
            invalid_args: Dict[str, Any] = self.valid_args.copy()
            invalid_args[arg] = 1
            with self.assertRaises(TypeError):
                msv.view_metrics(**invalid_args)

    def test_view_metrics_invalid_xcat(self):
        invalid_args: Dict[str, Any] = self.valid_args
        invalid_args["xcat"] = "bad_xcat"
        with self.assertRaises(ValueError):
            try:
                with warnings.catch_warnings(record=True) as w:
                    msv.view_metrics(**invalid_args)
            except Exception as e:
                self.assertIsInstance(e, ValueError)
                raise ValueError(e)

    def test_view_metrics_invalid_metric(self):
        bad_args = self.valid_args
        bad_args["metric"] = "bad_metric"
        with self.assertRaises(ValueError):
            msv.view_metrics(**bad_args)

    def test_view_metrics_empty_df(self):
        invalid_args: Dict[str, Any] = self.valid_args
        invalid_args["df"] = pd.DataFrame()
        with self.assertRaises(ValueError):
            msv.view_metrics(**invalid_args)

    def test_view_metrics_invalid_cids(self):
        invalid_args = self.valid_args.copy()
        invalid_args["cids"] = [1, 2, 3]
        # catch warnings from msm.validation.py
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaises(TypeError):
                msv.view_metrics(**invalid_args)

            invalid_args["cids"] = ["bad_cid"]
            with self.assertRaises(ValueError):
                msv.view_metrics(**invalid_args)

    def test_view_metrics_invalid_start_and_end(self):
        invalid_args = self.valid_args
        for arg in ["start", "end"]:
            invalid_args[arg] = "bad_date"
            with self.assertRaises(ValueError):
                msv.view_metrics(**invalid_args)

    def test_view_metrics_invalid_freq(self):
        invalid_args = self.valid_args.copy()
        invalid_args["freq"] = "z"
        with self.assertRaises(ValueError):
            msv.view_metrics(**invalid_args)

    def test_view_metrics_lowercase_freq(self):
        invalid_args = self.valid_args.copy()
        invalid_args["freq"] = "m"
        try:
            msv.view_metrics(**invalid_args)
        except Exception as e:
            self.fail(f"view_metrics raised {e} unexpectedly")

    def test_view_metrics_invalid_agg(self):
        # test agg
        invalid_args = self.valid_args.copy()
        invalid_args["agg"] = "bad_agg"

        with self.assertRaises(ValueError):
            msv.view_metrics(**invalid_args)

        invalid_args["agg"] = "MEAN"
        try:
            msv.view_metrics(**invalid_args)
        except Exception as e:
            self.fail(f"view_metrics raised {e} unexpectedly")

    def test_view_metrics_invalid_figsize(self):
        invalid_args = self.valid_args.copy()
        invalid_cases: List[Any] = [
            ["apple", 10],
            (10, "apple"),
            [None, 10],
        ]
        for case in invalid_cases:
            invalid_args["figsize"] = case
            with self.assertRaises(ValueError):
                msv.view_metrics(**invalid_args)

        invalid_args["figsize"] = [10, 10, 10]
        with self.assertRaises(TypeError):
            msv.view_metrics(**invalid_args)

        invalid_args["figsize"] = [10, None]
        try:
            msv.view_metrics(**invalid_args)
        except Exception as e:
            self.fail(f"view_metrics raised {e} unexpectedly")


if __name__ == "__main__":
    unittest.main()
