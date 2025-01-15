import unittest
from matplotlib import pyplot as plt
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate import make_test_df
from macrosynergy.visuals import Heatmap
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
            "cmap": "vlag",
            "rotate_xticks": 0,
            "rotate_yticks": 0,
            "show_tick_lines": True,
            "show_colorbar": True,
            "show_annotations": False,
            "show_boundaries": False,
        }

    def test_instantiate_heatmap_no_error(self):
        try:
            Heatmap(**self.constructor_args)
        except Exception as e:
            self.fail(f"Heatmap raised {e} unexpectedly")

    def test_plot_heatmap_no_error(self):
        heatmap = Heatmap(**self.constructor_args)
        heatmap.df = heatmap.df.pivot_table(
            index="cid",
            columns="real_date",
            values=self.metric,
            observed=True,
        )
        try:
            heatmap.plot(heatmap.df, **self.plot_args)
        except Exception as e:
            self.fail(f"Heatmap.plot raised {e} unexpectedly")

        self.plot_args["x_axis_label"] = None
        self.plot_args["y_axis_label"] = None
        self.plot_args["vmin"] = None
        self.plot_args["vmax"] = None
        _, self.plot_args["on_axis"] = plt.subplots(
            figsize=self.plot_args["figsize"], layout="constrained"
        )
        self.plot_args["show"] = False
        self.plot_args["return_figure"] = True

        try:
            fig = heatmap.plot(heatmap.df, **self.plot_args)
        except Exception as e:
            self.fail(f"Heatmap.plot raised {e} unexpectedly")

        assert isinstance(fig, plt.Figure)


if __name__ == "__main__":
    unittest.main()
