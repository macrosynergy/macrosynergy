import unittest
from matplotlib import pyplot as plt
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.utils.df_utils import downsample_df_on_real_date, reduce_df
from macrosynergy.visuals import FacetPlot
import numpy as np
import matplotlib
from unittest.mock import patch
from parameterized import parameterized


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Prevents plots from being displayed during tests.
        plt.close("all")
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()
        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD"]
        self.xcats: List[str] = ["XR", "CRY", "INFL"]
        self.metric = "value"
        self.metrics: List[str] = [self.metric]
        self.start: str = "2010-01-01"
        self.end: str = "2011-12-31"

        self.df: pd.DataFrame = make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start=self.start,
            end=self.end,
            metrics=self.metrics,
        )

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        patch.stopall()
        matplotlib.use(self.mpl_backend)

    def tearDown(self) -> None:
        plt.close("all")

    @parameterized.expand(["2010-01-05", "2010-05-15", "2010-07-15"])
    def test_insert_nans(self, blacklist_start_date: str):
        df = self.df[(self.df.cid == "AUD") & (self.df.xcat == "INFL")]
        fp = FacetPlot(df=df)
        black_list = {"AUD": [blacklist_start_date, "2010-12-10"]}

        fp.df = reduce_df(fp.df, blacklist=black_list)

        X, Y = fp._insert_nans(fp.df.real_date.to_numpy(), fp.df.value.to_numpy())

        idx = np.where(X == blacklist_start_date)[0][0]

        np.testing.assert_equal(Y[idx], np.nan)

    def test_insert_nans_multiple(self):

        first_date_start = "2010-01-05"
        first_date_end = "2010-02-10"
        second_date_start = "2010-05-15"
        second_date_end = "2010-07-15"

        df = self.df[(self.df.cid == "AUD") & (self.df.xcat == "INFL")]
        fp = FacetPlot(df=df)
        black_list = {
            "AUD": [first_date_start, first_date_end],
            "AUD_2": [second_date_start, second_date_end],
        }

        fp.df = reduce_df(fp.df, blacklist=black_list)
        X, Y = fp._insert_nans(fp.df.real_date.to_numpy(), fp.df.value.to_numpy())

        for date in [first_date_start, second_date_start]:
            idx = np.where(X == date)[0][0]
            np.testing.assert_equal(Y[idx], np.nan)

    def test_insert_nans_no_gaps(self):
        df = self.df[(self.df.cid == "AUD") & (self.df.xcat == "INFL")]
        fp = FacetPlot(df=df)

        X, Y = fp._insert_nans(fp.df.real_date.to_numpy(), fp.df.value.to_numpy())

        self.assertFalse(np.isnan(Y).any())

    def test_insert_nans_no_data(self):
        df = self.df[(self.df.cid == "AUD") & (self.df.xcat == "INFL")]
        fp = FacetPlot(df=df)

        X, Y = fp._insert_nans([], [])

        self.assertTrue(len(X) == 0)
        self.assertTrue(len(Y) == 0)

    def test_insert_nans_monthly_freq(self):
        df = self.df[(self.df.cid == "AUD") & (self.df.xcat == "INFL")]
        fp = FacetPlot(df=df)
        black_list = {
            "AUD": ["2010-02-05", "2010-05-05"],
        }

        fp.df = downsample_df_on_real_date(fp.df, ["xcat", "cid"], freq="M")
        fp.df = reduce_df(fp.df, blacklist=black_list)

        X, Y = fp._insert_nans(fp.df.real_date.to_numpy(), fp.df.value.to_numpy())

        np.testing.assert_equal(Y[1], np.nan)
