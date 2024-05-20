import numpy as np
import pandas as pd 

from pandas.testing import assert_frame_equal, assert_series_equal

from macrosynergy.management.simulate import make_qdf
from macrosynergy.learning.marketbeta import BetaEstimator

import unittest

class TestBetaEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Simulate a panel dataset of benchmark and contract returns
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["BENCH_XR", "CONTRACT_XR"]
        cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

        df_cids = pd.DataFrame(
            index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
        df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
        df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
        df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

        df_xcats = pd.DataFrame(index=xcats, columns=cols)
        df_xcats.loc["BENCH_XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CONTRACT_XR"] = ["2001-01-01", "2020-12-31", 0.1, 1, 0, 0.3]

        self.dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.xcat = "CONTRACT_XR"
        self.cids = cids
        self.benchmark_return = "USD_BENCH_XR"

    def test_valid_init(self):
        be = BetaEstimator(
            df = self.dfd,
            xcat=self.xcat,
            cids=self.cids,
            benchmark_return=self.benchmark_return,
        )

        # Check class attributes are correctly initialised
        self.assertIsInstance(be, BetaEstimator)
        self.assertEqual(be.xcat, self.xcat)
        self.assertEqual(be.cids, self.cids)
        self.assertEqual(be.benchmark_return, self.benchmark_return)

        assert_frame_equal(be.beta_df, pd.DataFrame(columns=["cid", "real_date", "xcat", "value"]))
        assert_frame_equal(be.hedged_returns, pd.DataFrame(columns=["cid", "real_date", "xcat", "value"]))
        assert_frame_equal(
            be.chosen_models,
            pd.DataFrame(
                columns=["real_date", "xcat", "model_type", "hparams", "n_splits_used"],
            ),
        )

        # Check the format of final long format dataframes
        self.assertIsInstance(be.X, pd.Series)
        self.assertTrue(be.X.name == self.benchmark_return.split("_",maxsplit=1)[1])
        self.assertTrue(be.X.index.names == ["cid", "real_date"])
        self.assertIsInstance(be.X.index, pd.MultiIndex)

        self.assertIsInstance(be.y, pd.Series)
        self.assertTrue(be.y.name == self.xcat)
        self.assertTrue(be.y.index.names == ["cid", "real_date"])
        self.assertIsInstance(be.y.index, pd.MultiIndex)

        # Check that the long format dataframes are correctly calculated
        dfd = self.dfd
        dfd["ticker"] = dfd["cid"] + "_" + dfd["xcat"]
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])
        for cid in self.cids:
            dfd_portion = dfd[(dfd.ticker == cid + "_" + self.xcat) | (dfd.ticker == self.benchmark_return)]
            dfd_portion["cid"] = f"{cid}v{self.benchmark_return.split('_', maxsplit=1)[0]}"
            dfx = pd.concat([dfx, dfd_portion[["real_date", "cid", "xcat", "value"]]])

        dfx_long = dfx.pivot(index=["cid", "real_date"], columns="xcat", values="value").dropna()
        X = dfx_long[self.benchmark_return.split("_", maxsplit=1)[1]]
        y = dfx_long[self.xcat]

        assert_series_equal(be.X, dfx_long[self.benchmark_return.split("_", maxsplit=1)[1]])
        assert_series_equal(be.y, dfx_long[self.xcat])