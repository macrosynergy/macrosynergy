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

    def test_view_score_evolution_cid_labels(self):
        sv = ScoreVisualisers(
            df=self.df, cids=["AUD", "CAD", "GBP"], xcats=["XR", "CRY"]
        )
        cid_labels = {"AUD": "Australia", "CAD": "Canada", "GBP": "UK"}
        result = sv.view_score_evolution(xcat="XR", freq="Q", cid_labels=cid_labels, return_as_df=True)
        self.assertIn("Australia", result.index)
        self.assertIn("Canada", result.index)
        self.assertNotIn("AUD", result.index)

    def test_view_score_evolution_cid_labels_from_constructor(self):
        cid_labels = {"AUD": "Australia", "CAD": "Canada", "GBP": "UK"}
        sv = ScoreVisualisers(
            df=self.df, cids=["AUD", "CAD", "GBP"], xcats=["XR", "CRY"],
            cid_labels=cid_labels,
        )
        result = sv.view_score_evolution(xcat="XR", freq="Q", return_as_df=True)
        self.assertIn("Australia", result.index)
        self.assertNotIn("AUD", result.index)

    def test_precomputed_composite_is_used(self):
        # Build a df that includes a pre-computed "Composite" xcat with sentinel values
        composite_df = self.df[self.df["xcat"] == "XR"].copy()
        composite_df["xcat"] = "Composite"
        composite_df["value"] = 999.0
        df_with_comp = pd.concat([self.df, composite_df], ignore_index=True)

        sv = ScoreVisualisers(
            df=df_with_comp,
            cids=["AUD", "CAD", "GBP"],
            xcats=["XR", "CRY"],
            no_zn_scores=True,
        )
        result = sv.view_score_evolution(xcat="Composite", freq="Q", return_as_df=True)
        # All values should be 999, confirming the pre-computed composite was used
        self.assertTrue((result.values == 999.0).all())


if __name__ == "__main__":
    unittest.main()
