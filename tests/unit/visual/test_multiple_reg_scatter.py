import unittest
from matplotlib import pyplot as plt
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel.category_relations import CategoryRelations
from macrosynergy.visuals import multiple_reg_scatter
import matplotlib
from unittest.mock import patch


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

    def setUp(self):
        cr1 = CategoryRelations(
            self.df,
            xcats=["XR", "CRY"],
            freq="M",
            lag=1,
            cids=self.cids,
            xcat_aggs=["mean", "sum"],
            start=self.start,
            end=self.end,
        )
        cr2 = CategoryRelations(
            self.df,
            xcats=["XR", "INFL"],
            freq="M",
            lag=1,
            cids=self.cids,
            xcat_aggs=["mean", "sum"],
            start=self.start,
            end=self.end,
        )
        cr3 = CategoryRelations(
            self.df,
            xcats=["CRY", "INFL"],
            freq="M",
            lag=1,
            cids=self.cids,
            xcat_aggs=["mean", "sum"],
            start=self.start,
            end=self.end,
        )
        self.args = {
            "cat_rels": [cr1, cr2, cr3],
            "ncol": 3,
            "nrow": 1,
            "figsize": (20, 15),
            "title": "Test",
            "title_xadj": 0.5,
            "title_yadj": 0.99,
            "title_fontsize": 20,
            "xlab": "Test",
            "ylab": "Test",
            "fit_reg": True,
            "reg_ci": 95,
            "reg_order": 1,
            "reg_robust": False,
            "coef_box": None,
            "prob_est": "pool",
            "separator": None,
            "single_chart": False,
            "subplot_titles": None,
        }

    def test_multiple_reg_scatter(self):
        try:
            multiple_reg_scatter(**self.args)
        except Exception as e:
            self.fail(f"multiple_reg_scatter raised {e} unexpectedly")


if __name__ == "__main__":
    unittest.main()
