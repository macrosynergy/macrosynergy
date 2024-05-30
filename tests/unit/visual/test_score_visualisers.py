import unittest
from matplotlib import pyplot as plt
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel.category_relations import CategoryRelations
from macrosynergy.visuals.score_visualisers import ScoreVisualizers
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
            self.cids, self.xcats, self.start, self.end
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
        self.args = {
            "df": self.df,
        }

    def test_multiple_reg_scatter(self):
        try:
            sv = ScoreVisualizers(**self.args)
        except Exception as e:
            self.fail(f"multiple_reg_scatter raised {e} unexpectedly")
    


if __name__ == "__main__":
    unittest.main()
