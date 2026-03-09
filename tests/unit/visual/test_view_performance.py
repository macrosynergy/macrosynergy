import unittest
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import macrosynergy.visuals as msv
from macrosynergy.management.simulate import make_qdf


class TestViewPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mpl_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        cls.mock_show = patch("matplotlib.pyplot.show").start()

        cls.cids = ["AUD", "CAD", "GBP"]
        cls.xcats = ["FXXR_NSA", "EQXR_NSA"]

        df_cids = pd.DataFrame(
            index=cls.cids,
            columns=["earliest", "latest", "mean_add", "sd_mult"],
        )
        for cid in cls.cids:
            df_cids.loc[cid] = ["2018-01-01", "2019-12-31", 0.0, 1.0]

        df_xcats = pd.DataFrame(
            index=cls.xcats,
            columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
        )
        df_xcats.loc["FXXR_NSA"] = ["2018-01-01", "2019-12-31", 0.1, 1.0, 0.0, 0.2]
        df_xcats.loc["EQXR_NSA"] = ["2018-01-01", "2019-12-31", 0.2, 1.2, 0.0, 0.2]

        cls.df = make_qdf(df_cids, df_xcats, back_ar=0.6, seed=42)

    @classmethod
    def tearDownClass(cls):
        patch.stopall()
        plt.close("all")
        matplotlib.use(cls.mpl_backend)

    def tearDown(self):
        plt.close("all")

    def test_return_metrics_compare_cids(self):
        result = msv.view_performance(
            self.df,
            xcats=["FXXR_NSA"],
            cids=["AUD", "CAD", "GBP"],
            return_metrics=True,
        )

        self.assertEqual(result.index.tolist(), ["Return %", "St. Dev. %", "Sharpe Ratio", "Sortino Ratio"])
        self.assertEqual(set(result.columns.tolist()), {"AUD", "CAD", "GBP"})

    def test_return_metrics_compare_xcats_with_aggregation(self):
        result = msv.view_performance(
            self.df,
            xcats=["FXXR_NSA", "EQXR_NSA"],
            cids=["ALL"],
            return_metrics=True,
        )

        self.assertEqual(set(result.columns.tolist()), {"FXXR_NSA", "EQXR_NSA"})
        self.assertTrue((result.loc["St. Dev. %"] > 0).all())

    def test_return_metrics_with_tickers_and_benchmark(self):
        result = msv.view_performance(
            self.df,
            tickers=["AUD_FXXR_NSA", "CAD_EQXR_NSA"],
            bms="GBP_EQXR_NSA",
            return_metrics=True,
        )

        self.assertIn("GBP_EQXR_NSA correl", result.index.tolist())
        self.assertEqual(set(result.columns.tolist()), {"AUD_FXXR_NSA", "CAD_EQXR_NSA"})

    def test_return_fig(self):
        fig = msv.view_performance(
            self.df,
            xcats=["FXXR_NSA"],
            cids=["AUD", "CAD", "GBP"],
            return_fig=True,
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_invalid_dimension_combination_raises(self):
        with self.assertRaises(ValueError):
            msv.view_performance(
                self.df,
                xcats=["FXXR_NSA", "EQXR_NSA"],
                cids=["AUD", "CAD"],
                return_metrics=True,
            )

    def test_missing_comparison_dimension_raises(self):
        with self.assertRaises(ValueError):
            msv.view_performance(
                self.df,
                xcats=["FXXR_NSA"],
                cids=["AUD"],
                return_metrics=True,
            )

    def test_missing_ticker_raises(self):
        with self.assertRaises(ValueError):
            msv.view_performance(
                self.df,
                tickers=["AUD_MISSING_XCAT"],
                return_metrics=True,
            )

    def test_invalid_metrics_filter_raises(self):
        with self.assertRaises(ValueError):
            msv.view_performance(
                self.df,
                xcats=["FXXR_NSA"],
                cids=["AUD", "CAD", "GBP"],
                metrics=["Not a metric"],
                return_metrics=True,
            )

    def test_labels_list_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            msv.view_performance(
                self.df,
                xcats=["FXXR_NSA"],
                cids=["AUD", "CAD", "GBP"],
                labels=["Only one label"],
                return_metrics=True,
            )


if __name__ == "__main__":
    unittest.main()
