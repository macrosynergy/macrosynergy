import unittest
from typing import Dict, List, Tuple, Union
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from macrosynergy.management.simulate.simulate_quantamental_data import \
    make_test_df
from macrosynergy.management.utils import reduce_df
from macrosynergy.pnl.multi_pnl import MultiPnL
from macrosynergy.pnl.naive_pnl import NaivePnL


class TestMultiPnL(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        plt.close("all")
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        patch.stopall()
        matplotlib.use(self.mpl_backend)

    def setUp(self) -> None:

        self.cids: List[str] = ["AUD", "CAD", "GBP"]
        self.rets = ["EQXR", "FXXR"]
        self.xcats: List[str] = ["INFL", "CRY"] + self.rets

        self.PNL_XCAT_1 = "PNL_EQ"
        self.PNL_XCAT_2 = "PNL_FX"

        dfd: pd.DataFrame = make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start="2000-01-01",
            end="2010-12-31",
        )
        self.dfd: pd.DataFrame = dfd

        self.pnl1 = NaivePnL(
            self.dfd,
            ret=self.rets[0],
            sigs=self.xcats,
            cids=self.cids,
            start="2000-01-01",
        )

        self.pnl2 = NaivePnL(
            self.dfd,
            ret=self.rets[1],
            sigs=self.xcats,
            cids=self.cids,
            start="2000-01-01",
        )

        self.pnl1.make_pnl(
            sig="INFL",
            sig_op="zn_score_pan",
            sig_neg=True,
            sig_add=0.5,
            rebal_freq="monthly",
            vol_scale=5,
            rebal_slip=1,
            min_obs=250,
            thresh=2,
            pnl_name=self.PNL_XCAT_1,
        )

        self.pnl2.make_pnl(
            sig="CRY",
            sig_op="zn_score_pan",
            sig_neg=True,
            sig_add=0.5,
            rebal_freq="monthly",
            vol_scale=5,
            rebal_slip=1,
            min_obs=250,
            thresh=2,
            pnl_name=self.PNL_XCAT_2,
        )

        self.pnl1.make_long_pnl(vol_scale=10, label="LONG")
        self.pnl2.make_long_pnl(vol_scale=10, label="LONG")

    def test_add_pnl(self):
        ma_pnl = MultiPnL()
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])
        ma_pnl.add_pnl(self.pnl2, [self.PNL_XCAT_2, "LONG"])
        self.assertTrue(f"{self.PNL_XCAT_1}/EQXR" in ma_pnl.pnl_xcats)
        self.assertTrue(f"{self.PNL_XCAT_2}/FXXR" in ma_pnl.pnl_xcats)

    def test_add_pnl_duplicate(self):
        ma_pnl = MultiPnL()
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])
        df_len = len(ma_pnl.get_pnls())
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])

        self.assertEqual(len(ma_pnl.get_pnls()), df_len)

    def test_infer_return_by_xcat(self):
        ma_pnl = MultiPnL()
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])
        ma_pnl.add_pnl(self.pnl2, [self.PNL_XCAT_2, "LONG"])
        # assert raises error when calling _infer_return_by_xcat
        with self.assertRaises(ValueError):
            ma_pnl._infer_return_by_xcat("LONG")

        # assert returns correct return when calling _infer_return_by_xcat
        self.assertEqual(
            ma_pnl._infer_return_by_xcat(self.PNL_XCAT_1), f"{self.PNL_XCAT_1}/EQXR"
        )
        self.assertEqual(
            ma_pnl._infer_return_by_xcat(f"{self.PNL_XCAT_1}/EQXR"),
            f"{self.PNL_XCAT_1}/EQXR",
        )

    def test_duplicate_xcat_raises_error(self):
        ma_pnl = MultiPnL()
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])
        ma_pnl.add_pnl(self.pnl2, [self.PNL_XCAT_2, "LONG"])
        with self.assertRaises(ValueError):
            ma_pnl.combine_pnls(["LONG", "LONG"], composite_pnl_xcat="LONG")
        with self.assertRaises(ValueError):
            ma_pnl.evaluate_pnls(["LONG", "LONG"])
        with self.assertRaises(ValueError):
            ma_pnl.plot_pnls(["LONG", "LONG"])

    def test_combine_pnls(self):
        ma_pnl = MultiPnL()
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])
        ma_pnl.add_pnl(self.pnl2, [self.PNL_XCAT_2, "LONG"])
        ma_pnl.combine_pnls(["LONG/FXXR", "LONG/EQXR"], composite_pnl_xcat="LONG")
        self.assertTrue("LONG" in ma_pnl.pnl_xcats)
        self.assertTrue("LONG/EQXR" in ma_pnl.pnl_xcats)
        self.assertTrue("LONG/FXXR" in ma_pnl.pnl_xcats)

        # Assert combined pnl is overwritten rather than duplicated
        df_len = len(ma_pnl.get_pnls())
        ma_pnl.combine_pnls(["LONG/FXXR", "LONG/EQXR"], composite_pnl_xcat="LONG")
        self.assertEqual(len(ma_pnl.get_pnls()), df_len)

    def test_check_xcat_labels(self):

        ma_pnl = MultiPnL()
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])
        ma_pnl.add_pnl(self.pnl2, [self.PNL_XCAT_2, "LONG"])

        pnl_xcats = ["LONG/FXXR", "LONG/EQXR"]
        xcat_labels = ["FX", "EQ"]
        labels_result = ma_pnl._check_xcat_labels(pnl_xcats, xcat_labels)

        self.assertEqual(labels_result["LONG/FXXR"], "FX")
        self.assertEqual(labels_result["LONG/EQXR"], "EQ")


if __name__ == "__main__":
    unittest.main()
