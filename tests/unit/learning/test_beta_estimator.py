import numpy as np
import pandas as pd 

from pandas.testing import assert_frame_equal, assert_series_equal

from macrosynergy.management.simulate import make_qdf
from macrosynergy.learning import (
    BetaEstimator,
    ExpandingKFoldPanelSplit,
    neg_mean_abs_corr,
    LinearRegressionSystem,
)
from macrosynergy.learning.beta_estimator import BetaEstimator

import unittest

from parameterized import parameterized

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
        df_cids.loc["AUD"] = ["2017-01-01", "2020-12-31", 0, 1]
        df_cids.loc["CAD"] = ["2019-01-01", "2020-12-31", 0, 1]
        df_cids.loc["GBP"] = ["2017-01-01", "2020-12-31", 0, 1]
        df_cids.loc["USD"] = ["2017-01-01", "2020-12-31", 0, 1]

        df_xcats = pd.DataFrame(index=xcats, columns=cols)
        df_xcats.loc["BENCH_XR"] = ["2017-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CONTRACT_XR"] = ["2018-01-01", "2020-12-31", 0.1, 1, 0, 0.3]

        self.dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.xcat = "CONTRACT_XR"
        self.cids = sorted(cids)
        self.benchmark_return = "USD_BENCH_XR"

        self.be = BetaEstimator(
            df = self.dfd,
            xcat=self.xcat,
            cids=self.cids,
            benchmark_return=self.benchmark_return,
        )

        # Create some general betas and hedged returns
        self.be.estimate_beta(
            beta_xcat="BETA_NSA",
            hedged_return_xcat="HEDGED_RETURN_NSA",
            inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
            scorer=neg_mean_abs_corr,
            models={
                "OLS": LinearRegressionSystem(),
            },
            hparam_grid={
                "OLS": {"fit_intercept": [True, False]},
            },
            min_cids = 1,
            min_periods = 21 * 12,
            est_freq="M",
            use_variance_correction=False,
        )

    def test_valid_init(self):
        # Check class attributes are correctly initialised
        be = BetaEstimator(
            df = self.dfd,
            xcat=self.xcat,
            cids=self.cids,
            benchmark_return=self.benchmark_return,
        )
        self.assertIsInstance(be, BetaEstimator)
        self.assertEqual(be.xcat, self.xcat)
        self.assertEqual(be.cids, self.cids)
        self.assertEqual(be.benchmark_return, self.benchmark_return)

        assert_frame_equal(be.betas, pd.DataFrame(columns=["cid", "real_date", "xcat", "value"]))
        assert_frame_equal(be.hedged_returns, pd.DataFrame(columns=["cid", "real_date", "xcat", "value"]))
        assert_frame_equal(
            be.chosen_models,
            pd.DataFrame(
                columns=["real_date", "xcat", "model_type", "hparams", "n_splits"],
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

    def test_types_init(self):
        # dataframe df
        with self.assertRaises(TypeError):
            be = BetaEstimator(
                df = "not a dataframe",
                xcat=self.xcat,
                cids=self.cids,
                benchmark_return=self.benchmark_return,
            )
        with self.assertRaises(ValueError):
            be = BetaEstimator(
                df = self.dfd.iloc[:,:-1],
                xcat=self.xcat,
                cids=self.cids,
                benchmark_return=self.benchmark_return,
            )
        with self.assertRaises(ValueError):
            be = BetaEstimator(
                df = self.dfd[self.dfd.xcat != self.xcat],
                xcat=self.xcat,
                cids=self.cids,
                benchmark_return=self.benchmark_return,
            )
        # xcat
        with self.assertRaises(TypeError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=1,
                cids=self.cids,
                benchmark_return=self.benchmark_return,
            )
        with self.assertRaises(ValueError):
            be = BetaEstimator(
                df = self.dfd,
                xcat="not a valid xcat",
                cids=self.cids,
                benchmark_return=self.benchmark_return,
            )
        # cids
        with self.assertRaises(TypeError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=self.xcat,
                cids="not a list",
                benchmark_return=self.benchmark_return,
            )
        with self.assertRaises(TypeError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=self.xcat,
                cids=[1],
                benchmark_return=self.benchmark_return,
            )
        with self.assertRaises(ValueError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=self.xcat,
                cids=self.cids + ["not a valid cid"],
                benchmark_return=self.benchmark_return,
            )
        with self.assertRaises(ValueError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=self.xcat,
                cids=["not a valid cid"],
                benchmark_return=self.benchmark_return,
            )
        with self.assertRaises(ValueError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=self.xcat,
                cids=self.cids + ["not a valid cid"],
                benchmark_return=self.benchmark_return,
            )
        # benchmark_return
        with self.assertRaises(TypeError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=self.xcat,
                cids=self.cids,
                benchmark_return=1,
            )
        with self.assertRaises(ValueError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=self.xcat,
                cids=self.cids,
                benchmark_return="not a valid benchmark_return",
            )
        with self.assertRaises(ValueError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=self.xcat,
                cids=self.cids,
                benchmark_return="GLB_DRBXR_NSA",
            )
        with self.assertRaises(ValueError):
            be = BetaEstimator(
                df = self.dfd,
                xcat=self.xcat,
                cids=self.cids,
                benchmark_return="USD_CONTRACT_XR",
            )
        

    def test_valid_estimate_beta(self):
        # Test the broad method
        # Betas dataframe
        self.assertIsInstance(self.be.betas, pd.DataFrame)
        self.assertTrue(self.be.betas.columns.tolist() == ["cid", "real_date", "xcat", "value"])
        self.assertTrue("BETA_NSA" in self.be.betas.xcat.unique())
        self.assertTrue(sorted(self.be.betas[self.be.betas.xcat == "BETA_NSA"].cid.unique()) == self.cids)
        self.assertTrue(all(~self.be.betas[self.be.betas.xcat == "BETA_NSA"].value.isna()))
        # Hedged returns dataframe
        self.assertIsInstance(self.be.hedged_returns, pd.DataFrame)
        self.assertTrue(self.be.hedged_returns.columns.tolist() == ["cid", "real_date", "xcat", "value"])
        self.assertTrue("HEDGED_RETURN_NSA" in self.be.hedged_returns.xcat.unique())
        self.assertTrue(sorted(self.be.hedged_returns[self.be.hedged_returns.xcat == "HEDGED_RETURN_NSA"].cid.unique()) == self.cids)
        self.assertTrue(all(~self.be.hedged_returns[self.be.hedged_returns.xcat == "HEDGED_RETURN_NSA"].value.isna()))