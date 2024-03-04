import unittest
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from unittest.mock import patch
from tests.simulate import make_qdf
from macrosynergy.panel.view_correlation import correl_matrix


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Prevents plots from being displayed during tests.
        plt.close("all")
        self.mpl_backend: str = matplotlib.get_backend()
        self.mock_show = patch("matplotlib.pyplot.show").start()

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        patch.stopall()
        matplotlib.use(self.mpl_backend)

    def setUp(self) -> None:
        self.cids = ["AUD", "CAD", "GBP"]
        self.xcats = ["XR", "CRY", "GROWTH", "INFL"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD", :] = ["2010-01-01", "2020-12-31", 0.5, 2]
        df_cids.loc["CAD", :] = ["2010-01-01", "2020-11-30", 0, 1]
        df_cids.loc["GBP", :] = ["2012-01-01", "2020-11-30", -0.2, 0.5]

        df_xcats = pd.DataFrame(
            index=self.xcats,
            columns=[
                "earliest",
                "latest",
                "mean_add",
                "sd_mult",
                "ar_coef",
                "back_coef",
            ],
        )
        df_xcats.loc["CRY", :] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
        df_xcats.loc["XR", :] = ["2011-01-01", "2020-12-31", 0, 1, 0, 0.3]
        df_xcats.loc["GROWTH", :] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
        df_xcats.loc["INFL", :] = ["2010-01-01", "2020-10-30", 0, 2, 0.9, 0.5]

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.dfd = dfd

        self.start = "2012-01-01"
        self.end = "2020-09-30"

        self.valid_args = {
            "df": self.dfd,
            "xcats": self.xcats,
            "cids": self.cids,
            "xcats_secondary": None,
            "cids_secondary": None,
            "start": self.start,
            "end": self.end,
            "val": "value",
            "freq": None,
            "cluster": True,
            "lags": None,
            "lags_secondary": None,
            "title": None,
            "size": (14, 8),
            "max_color": None,
            "xlabel": "",
            "ylabel": "",
            "return_values": False,
        }

    def tearDown(self) -> None:
        return super().tearDown()

    def test_correl_matrix_no_error(self):
        try:
            correl_matrix(**self.valid_args)
        except Exception as e:
            self.fail(f"correl_matrix raised {e} unexpectedly")

    def test_cross_category_corr(self):
        corr_df = correl_matrix(
            self.dfd,
            xcats=self.xcats,
            return_values=True,
            cluster=False,
        )
        self.assertEqual(corr_df.shape, (len(self.xcats), len(self.xcats)))
        self.assertEqual(corr_df.columns.to_list(), self.xcats)
        self.assertEqual(corr_df.index.to_list(), self.xcats)

    def test_cross_sectional_corr(self):
        corr_df = correl_matrix(
            self.dfd,
            xcats=["XR"],
            cids=self.cids,
            return_values=True,
        )
        self.assertEqual(corr_df.shape, (len(self.cids), len(self.cids)))
        self.assertEqual(corr_df.columns.to_list(), self.cids)
        self.assertEqual(corr_df.index.to_list(), self.cids)

    def test_cross_category_corr_with_xcats_secondary(self):
        xcats_secondary = ["XR"]
        corr_df = correl_matrix(
            self.dfd,
            xcats=self.xcats,
            xcats_secondary=xcats_secondary,
            return_values=True,
        )
        self.assertEqual(corr_df.shape, (len(xcats_secondary), len(self.xcats)))
        self.assertEqual(corr_df.columns.to_list(), self.xcats)
        self.assertEqual(corr_df.index.to_list(), xcats_secondary)

    def test_cross_sectional_corr_with_cids_secondary(self):
        cids_secondary = ["AUD"]
        corr_df = correl_matrix(
            self.dfd,
            xcats=["XR"],
            cids=self.cids,
            cids_secondary=cids_secondary,
            return_values=True,
        )
        self.assertEqual(corr_df.shape, (len(cids_secondary), len(self.cids)))
        self.assertEqual(corr_df.columns.to_list(), self.cids)
        self.assertEqual(corr_df.index.to_list(), cids_secondary)

    def test_cross_category_corr_with_both_secondary(self):
        xcats_secondary = ["XR"]
        corr_df = correl_matrix(
            self.dfd,
            xcats=self.xcats,
            xcats_secondary=xcats_secondary,
            cids=self.cids,
            cids_secondary=self.cids[:2],
            return_values=True,
        )
        self.assertEqual(corr_df.shape, (len(xcats_secondary), len(self.xcats)))
        self.assertEqual(corr_df.columns.to_list(), self.xcats)
        self.assertEqual(corr_df.index.to_list(), xcats_secondary)

    def test_cross_sectional_corr_with_both_secondary(self):
        cids_secondary = ["AUD"]
        corr_df = correl_matrix(
            self.dfd,
            xcats=["XR"],
            xcats_secondary=["CRY"],
            cids=self.cids,
            cids_secondary=cids_secondary,
            return_values=True,
        )
        self.assertEqual(corr_df.shape, (len(cids_secondary), len(self.cids)))
        self.assertEqual(corr_df.columns.to_list(), self.cids)
        self.assertEqual(corr_df.index.to_list(), cids_secondary)

    def test_correl_matrix(self):
        # Mainly test assertions given the function is used for visualisation. The
        # function can easily be tested through the graph returned.

        try:
            correl_matrix(self.dfd, xcats=["XR"], cids=self.cids, max_color=0.1)
        except Exception as e:
            self.fail(f"correl_matrix raised {e} unexpectedly")

        try:
            correl_matrix(
                self.dfd,
                xcats=["XR"],
                xcats_secondary=["CRY"],
                cids=self.cids,
                max_color=0.1,
            )
        except Exception as e:
            self.fail(f"correl_matrix raised {e} unexpectedly")

        try:
            correl_matrix(
                self.dfd,
                xcats=["XR"],
                xcats_secondary=["CRY"],
                cids=["AUD"],
                cids_secondary=["GBP"],
                max_color=0.1,
            )
        except Exception as e:
            self.fail(f"correl_matrix raised {e} unexpectedly")

        try:
            correl_matrix(
                self.dfd,
                xcats=["XR"],
                cids=["AUD"],
                cids_secondary=["GBP"],
                max_color=0.1,
            )
        except Exception as e:
            self.fail(f"correl_matrix raised {e} unexpectedly")

        lag_dict = {"INFL": [0, 1, 2, 5]}
        with self.assertRaises(ValueError):
            # Test the frequency options: either ['W', 'M', 'Q'].
            correl_matrix(
                self.dfd,
                xcats=["XR", "CRY"],
                cids=self.cids,
                freq="BW",
                lags=lag_dict,
                max_color=0.1,
            )

        with self.assertRaises(AssertionError):
            # Test the max_color value. Expects a floating point value.
            correl_matrix(
                self.dfd,
                xcats=["XR", "CRY"],
                cids=self.cids,
                lags=lag_dict,
                max_color=1,
            )

        with self.assertRaises(AssertionError):
            # Test the received lag data structure. Dictionary expected.
            lag_list = [0, 60]
            correl_matrix(
                self.dfd,
                xcats=["XR", "CRY"],
                cids=self.cids,
                lags=lag_list,
                max_color=1,
            )

        with self.assertRaises(AssertionError):
            # Test that the lagged categories are present in the received DataFrame.
            # The category, GROWTH, is not in the received categories.
            lag_dict = {"GROWTH": [0, 1, 2, 5]}
            correl_matrix(
                self.dfd,
                xcats=["XR", "CRY"],
                cids=self.cids,
                lags=lag_dict,
                max_color=1,
            )


if __name__ == "__main__":
    unittest.main()
