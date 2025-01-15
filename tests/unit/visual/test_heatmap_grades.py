import unittest
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate import make_qdf

# from macrosynergy.visuals import heatmap_grades
from macrosynergy.panel.view_grades import heatmap_grades
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

        # Set up test df with grading.
        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD"]
        self.xcats: List[str] = ["XR", "CRY", "INFL"]
        self.metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
        self.start: str = "2010-01-01"
        self.end: str = "2020-12-31"

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD",] = ["2000-01-01", "2020-12-31", 0.1, 1]
        df_cids.loc["CAD",] = ["2001-01-01", "2020-11-30", 0, 1]
        df_cids.loc["GBP",] = ["2002-01-01", "2020-11-30", 0, 2]
        df_cids.loc["NZD",] = ["2002-01-01", "2020-09-30", -0.1, 2]

        df_xcats = pd.DataFrame(
            index=self.xcats,
            columns=[
                "earliest",
                "latest",
                "mean_add",
                "sd_mult",
                "ar_coef",
                "back_coef",
            ],
        )
        df_xcats.loc["XR",] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CRY",] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
        df_xcats.loc["GROWTH",] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
        df_xcats.loc["INFL",] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

        dfd["grading"] = "3"
        filter_date = dfd["real_date"] >= pd.to_datetime("2010-01-01")
        filter_cid = dfd["cid"].isin(["NZD", "AUD"])
        dfd.loc[filter_date & filter_cid, "grading"] = "1"
        filter_date = dfd["real_date"] >= pd.to_datetime("2013-01-01")
        filter_xcat = dfd["xcat"].isin(["CRY", "GROWTH"])
        dfd.loc[filter_date & filter_xcat, "grading"] = "2.1"
        filter_xcat = dfd["xcat"] == "XR"
        dfd.loc[filter_xcat, "grading"] = 1
        self.df: pd.DataFrame = dfd
        warnings.simplefilter("ignore")

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(self.mpl_backend)
        warnings.resetwarnings()

    def test_heatmap_grades(self):
        heatmap_grades(self.df, xcats=self.xcats)

    def test_heatmap_no_grades_col(self):
        self.df = self.df.drop(columns=["grading"])
        with self.assertRaises(AssertionError):
            heatmap_grades(self.df, xcats=self.xcats)


if __name__ == "__main__":
    unittest.main()
