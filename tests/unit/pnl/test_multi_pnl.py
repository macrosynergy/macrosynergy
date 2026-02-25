import unittest
from typing import Dict, List, Tuple, Union
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from macrosynergy.management.simulate.simulate_quantamental_data import \
    make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
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

    def test_init_with_benchmark_success(self):
        ma_pnl = MultiPnL(bms="AUD_EQXR", df=self.dfd)
        self.assertTrue(ma_pnl.bm_bool)
        self.assertIn("AUD_EQXR", ma_pnl._bm_dict)

    def test_init_with_bms_without_df_raises(self):
        with self.assertRaises(ValueError):
            MultiPnL(bms="AUD_EQXR")

    def test_init_with_missing_benchmark_raises(self):
        with self.assertRaises(ValueError):
            MultiPnL(bms="ZZZ_EQXR", df=self.dfd)

    def test_evaluate_pnls_no_benchmark_unchanged(self):
        ma_pnl = MultiPnL()
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])
        ma_pnl.add_pnl(self.pnl2, [self.PNL_XCAT_2, "LONG"])
        ma_pnl.combine_pnls(["LONG/FXXR", "LONG/EQXR"], composite_pnl_xcat="LONG")

        eval_df = ma_pnl.evaluate_pnls([f"{self.PNL_XCAT_1}/EQXR", "LONG"])
        self.assertFalse(any("correl" in idx for idx in eval_df.index.tolist()))

    def test_evaluate_pnls_with_benchmark_adds_rows_all_pnls(self):
        ma_pnl = MultiPnL(bms="AUD_EQXR", df=self.dfd)
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])
        ma_pnl.add_pnl(self.pnl2, [self.PNL_XCAT_2, "LONG"])
        ma_pnl.combine_pnls(["LONG/FXXR", "LONG/EQXR"], composite_pnl_xcat="LONG")

        eval_df = ma_pnl.evaluate_pnls([f"{self.PNL_XCAT_1}/EQXR", "LONG"])
        self.assertIn("AUD_EQXR correl", eval_df.index.tolist())
        self.assertIn(f"{self.PNL_XCAT_1}/EQXR", eval_df.columns.tolist())
        self.assertIn("LONG", eval_df.columns.tolist())

    def test_benchmark_row_position(self):
        ma_pnl = MultiPnL(bms="AUD_EQXR", df=self.dfd)
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1, "LONG"])

        eval_df = ma_pnl.evaluate_pnls([f"{self.PNL_XCAT_1}/EQXR"])
        idx = eval_df.index.tolist()
        self.assertLess(idx.index("AUD_EQXR correl"), idx.index("Traded Months"))

    def test_multiple_benchmarks(self):
        ma_pnl = MultiPnL(bms=["AUD_EQXR", "CAD_FXXR"], df=self.dfd)
        ma_pnl.add_pnl(self.pnl1, [self.PNL_XCAT_1])

        eval_df = ma_pnl.evaluate_pnls([f"{self.PNL_XCAT_1}/EQXR"])
        idx = eval_df.index.tolist()
        self.assertIn("AUD_EQXR correl", idx)
        self.assertIn("CAD_FXXR correl", idx)
        # both correl rows must appear before Traded Months
        traded_pos = idx.index("Traded Months")
        self.assertLess(idx.index("AUD_EQXR correl"), traded_pos)
        self.assertLess(idx.index("CAD_FXXR correl"), traded_pos)

    def test_correlation_value_sanity(self):
        dates = pd.bdate_range("2020-01-01", periods=120)
        vals = np.linspace(-1, 1, len(dates))

        bm_df = pd.DataFrame(
            {
                "cid": "AUD",
                "xcat": "EQXR",
                "real_date": dates,
                "value": vals,
            }
        )
        ma_pnl = MultiPnL(bms="AUD_EQXR", df=bm_df)

        pnl_df = pd.DataFrame(
            {
                "real_date": np.concatenate([dates.to_numpy(), dates.to_numpy()]),
                "xcat": ["SYNTH"] * len(dates) + ["SYNTH_NEG"] * len(dates),
                "value": np.concatenate([vals, -vals]),
                "cid": "ALL",
            }
        )
        ma_pnl.pnls_df = QuantamentalDataFrame(pnl_df)

        eval_df = ma_pnl.evaluate_pnls()
        self.assertTrue(np.isclose(eval_df.loc["AUD_EQXR correl", "SYNTH"], 1.0))
        self.assertTrue(np.isclose(eval_df.loc["AUD_EQXR correl", "SYNTH_NEG"], -1.0))


if __name__ == "__main__":
    unittest.main()
