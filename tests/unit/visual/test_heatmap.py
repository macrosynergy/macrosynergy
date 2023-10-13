import unittest
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate_quantamental_data import make_test_df
from macrosynergy.visuals import Heatmap
import matplotlib


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Prevents plots from being displayed during tests.
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")

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
        matplotlib.use(self.mpl_backend)

    def setUp(self):
        self.constructor_args: Dict[str, Any] = {
            "df": self.df,
            "xcats": self.xcats,
            "cids": self.cids,
            "metrics": self.metrics,
            "start": self.start,
            "end": self.end,
        }

        self.plot_args: Dict[str, Any] = {
            "figsize": (12, 12),
            "x_axis_label": "x_axis_label",
            "y_axis_label": "y_axis_label",
            "axis_fontsize": 12,
            "title": "title",
            "title_fontsize": 12,
            "title_xadjust": 0.5,
            "title_yadjust": 1,
            "vmin": 0,
            "vmax": 100,
            "show": True,
            "save_to_file": None,
            "dpi": 300,
            "return_figure": False,
            "on_axis": None,
            "max_xticks": 50,
        }

    def test_instantiate_heatmap_no_error(self):
        try:
            Heatmap(**self.constructor_args)
        except Exception as e:
            self.fail(f"Heatmap raised {e} unexpectedly")

    def test_plot_heatmap_no_error(self):
        try:
            heatmap = Heatmap(**self.constructor_args)
            heatmap.df: pd.DataFrame = heatmap.df.pivot_table(
                index="cid", columns="real_date", values=self.metric
            )
            heatmap.plot(**self.plot_args)
        except Exception as e:
            self.fail(f"Heatmap.plot raised {e} unexpectedly")


if __name__ == "__main__":
    unittest.main()
