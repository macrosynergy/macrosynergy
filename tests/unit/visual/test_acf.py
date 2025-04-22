import unittest
from matplotlib import pyplot as plt
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate import make_test_df
from macrosynergy.visuals import plot_acf, plot_pacf
import matplotlib
from unittest.mock import patch


class TestACF(unittest.TestCase):
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
        patch.stopall()
        plt.close("all")
        patch.stopall()
        matplotlib.use(self.mpl_backend)

    def tearDown(self) -> None:
        plt.close("all")

    def test_plot_acf(self):
        try:
            plot_acf(self.df, xcat="INFL", cids=self.cids)
        except Exception as e:
            self.fail(f"plot_acf raised {e} unexpectedly")

    def test_plot_pacf(self):
        try:
            plot_pacf(self.df, xcat="INFL", cids=self.cids)
        except Exception as e:
            self.fail(f"plot_acf raised {e} unexpectedly")


if __name__ == "__main__":
    unittest.main()
