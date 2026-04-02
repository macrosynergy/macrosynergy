import unittest
from macrosynergy.management.simulate import make_test_df

# import macrosynergy.visuals as msv
from macrosynergy.panel.view_ranges import view_ranges
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch


class TestAll(unittest.TestCase):
    def setUp(self):
        # Prevents plots from being displayed during tests.
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()
        self.test_cids = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD"]
        self.test_xcats = ["XR", "CRY", "INFL", "GROWTH"]
        self.df = make_test_df(cids=self.test_cids, xcats=self.test_xcats)

    def tearDown(self) -> None:
        plt.close("all")
        patch.stopall()
        matplotlib.use(self.mpl_backend)
        plt.close("all")

    def test_legend_loc_args_none(self):
        view_ranges(
            self.df,
            xcats=["XR", "CRY"],
            cids=self.test_cids,
            kind="bar",
        )

    def test_legend_loc_args_anchor(self):
        view_ranges(
            self.df,
            xcats=["XR", "CRY"],
            cids=self.test_cids,
            kind="bar",
            legend_bbox_to_anchor=(0.5, -0.15),
        )

    def test_footnote_forwarded(self):
        import sys
        vr_mod = sys.modules["macrosynergy.panel.view_ranges"]
        with patch.object(vr_mod.msv, "view_ranges") as mock_view_ranges:
            view_ranges(
                self.df,
                xcats=["XR", "CRY"],
                footnote="Source: test",
                footnote_fontsize=11,
            )

        self.assertEqual(mock_view_ranges.call_args[1]["footnote"], "Source: test")
        self.assertEqual(mock_view_ranges.call_args[1]["footnote_fontsize"], 11)


if __name__ == "__main__":
    unittest.main()
