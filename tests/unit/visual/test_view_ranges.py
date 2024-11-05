import unittest
import pandas as pd
from typing import List, Dict, Any
from macrosynergy.management.simulate import make_test_df
import macrosynergy.visuals as msv
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

    def test_boxplot(self):
        # test with box plot
        msv.view_ranges(
            self.df,
            xcats=["XR", "CRY"],
            cids=self.test_cids,
            kind="box",
            start="2012-01-01",
            end="2018-01-01",
        )

    def test_barplot(self):
        # test with bar
        msv.view_ranges(
            self.df,
            xcats=["XR", "CRY"],
            cids=self.test_cids,
            kind="bar",
            start="2012-01-01",
            end="2018-01-01",
        )

    def test_barplot_1_xcat(self):
        # test with 1 xcat
        msv.view_ranges(
            self.df,
            xcats=["XR"],
            cids=["USD", "EUR"],
            kind="bar",
        )

    def test_barplot_1_xcat_no_cids(self):
        # sort cid by mean
        msv.view_ranges(
            self.df,
            xcats=["XR", "CRY"],
            kind="bar",
            sort_cids_by="mean",
        )

    def test_barplot_1_xcat_no_cids_std(self):
        # sort cid by std
        msv.view_ranges(
            self.df,
            xcats=["XR", "CRY"],
            kind="bar",
            sort_cids_by="std",
        )

    def test_invalid_xcats(self):
        with self.assertRaises(ValueError):
            msv.view_ranges(
                self.df,
                xcats=["banana", "apple"],
                kind="bar",
            )

    def test_invalid_cids(self):
        with self.assertRaises(ValueError):
            msv.view_ranges(
                self.df,
                xcats=["XR", "CRY"],
                cids=["banana", "apple"],
                kind="bar",
            )

    def test_invalid_sort_cids_by(self):
        with self.assertRaises(ValueError):
            msv.view_ranges(
                self.df,
                xcats=["XR", "CRY"],
                kind="bar",
                sort_cids_by="banana",
            )

        with self.assertRaises(TypeError):
            msv.view_ranges(
                self.df,
                xcats=["XR", "CRY"],
                kind="bar",
                sort_cids_by=1,
            )

    def test_invalid_xcat_labels(self):
        with self.assertRaises(ValueError):
            msv.view_ranges(
                self.df,
                xcats=["XR", "CRY"],
                kind="bar",
                xcat_labels=["banana"],
            )

    def test_valid_xcat_labels(self):
        msv.view_ranges(
            self.df,
            xcats=["XR", "CRY"],
            kind="bar",
            xcat_labels=["Return", "Carry"],
        )

    def test_missing_cid(self):
        df = self.df.copy()
        sel_cid, sel_xcat = self.test_cids[0], self.test_xcats[0]
        df = df.loc[~((df["cid"] == sel_cid) & (df["xcat"] == sel_xcat))]

        msv.view_ranges(
            df,
            xcats=[sel_xcat, "CRY"],
            cids=self.test_cids,
            kind="bar",
            sort_cids_by="std",
        )


if __name__ == "__main__":
    unittest.main()
