from typing import List
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from macrosynergy.management.simulate.simulate_quantamental_data import make_test_df
from macrosynergy.management.utils.df_utils import reduce_df
from macrosynergy.visuals import (
    ScoreVisualisers,
)  # Adjust the import according to your module's name


class TestScoreVisualisers(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame to use in tests
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

    def test_initialization(self):
        sv = ScoreVisualisers(
            df=self.df, cids=["AUD", "CAD", "GBP"], xcats=["XR", "CRY"]
        )
        self.assertEqual(sv.cids, ["AUD", "CAD", "GBP"])
        self.assertEqual(set(sv.xcats), set(["Composite_ZN", "XR_ZN", "CRY_ZN"]))

    def test_initialization_no_zn(self):
        sv = ScoreVisualisers(
            df=self.df,
            cids=["AUD", "CAD", "GBP"],
            xcats=["XR", "CRY"],
            no_zn_scores=True,
        )

    def test_invalid_cids_type(self):
        with self.assertRaises(TypeError):
            ScoreVisualisers(df=self.df, cids="invalid_cid")

    def test_invalid_xcats_type(self):
        with self.assertRaises(TypeError):
            ScoreVisualisers(df=self.df, xcats="invalid_xcat")

    def test_invalid_xcat_comp_type(self):
        with self.assertRaises(TypeError):
            ScoreVisualisers(df=self.df, xcat_comp=123)

    def test_create_df_no_zn_scores(self):
        sv = ScoreVisualisers(df=self.df, xcats=["XR", "CRY"], no_zn_scores=True)
        self.assertTrue(set(sv.xcats) == set(["Composite", "XR", "CRY"]))

    def test_create_df_with_zn_scores(self):
        sv = ScoreVisualisers(df=self.df, xcats=["XR", "CRY"], no_zn_scores=False)
        self.assertTrue(set(sv.xcats) == set(["Composite_ZN", "XR_ZN", "CRY_ZN"]))

    def test_apply_postfix(self):
        sv = ScoreVisualisers(
            df=self.df, cids=["AUD", "CAD", "GBP"], xcats=["XR", "CRY"]
        )
        items = ["item1", "item2"]
        self.assertEqual(sv._apply_postfix(items), ["item1_ZN", "item2_ZN"])

    def test_strip_postfix(self):
        sv = ScoreVisualisers(
            df=self.df, cids=["AUD", "CAD", "GBP"], xcats=["XR", "CRY"]
        )
        items = ["item1_ZN", "item2_ZN"]
        self.assertEqual(sv._strip_postfix(items), ["item1", "item2"])

    @patch("matplotlib.pyplot.show")
    def test_view_snapshot(self, mock_plt_show):
        sv = ScoreVisualisers(
            df=self.df, cids=["AUD", "CAD", "GBP"], xcats=["XR", "CRY"]
        )
        sv.view_snapshot()
        self.assertTrue(mock_plt_show.called)

    @patch("matplotlib.pyplot.show")
    def test_view_score_evolution(self, mock_plt_show):
        sv = ScoreVisualisers(
            df=self.df, cids=["AUD", "CAD", "GBP"], xcats=["XR", "CRY"]
        )
        sv.view_score_evolution(xcat="XR", freq="Q")
        self.assertTrue(mock_plt_show.called)

    @patch("matplotlib.pyplot.show")
    def test_view_cid_evolution(self, mock_plt_show):
        sv = ScoreVisualisers(
            df=self.df, cids=["AUD", "CAD", "GBP"], xcats=["XR", "CRY"]
        )
        sv.view_cid_evolution(cid="AUD", xcats=["XR"], freq="Q")
        self.assertTrue(mock_plt_show.called)


if __name__ == "__main__":
    unittest.main()
