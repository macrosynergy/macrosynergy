import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from parameterized import parameterized

from macrosynergy.learning import (
    BetaEstimator,
    ExpandingKFoldPanelSplit,
    LinearRegressionSystem,
    neg_mean_abs_corr,
)
from macrosynergy.management.simulate import make_qdf


class TestBetaEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Simulate a panel dataset of benchmark and contract returns
        cids = ["AUD", "CAD", "GBP", "USD"]
        self.xcats = ["BENCH_XR", "CONTRACT_XR"]
        cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

        df_cids = pd.DataFrame(
            index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD"] = ["2017-01-01", "2020-12-31", 0, 1]
        df_cids.loc["CAD"] = ["2019-01-01", "2020-12-31", 0, 1]
        df_cids.loc["GBP"] = ["2017-01-01", "2020-12-31", 0, 1]
        df_cids.loc["USD"] = ["2017-01-01", "2020-12-31", 0, 1]

        df_xcats = pd.DataFrame(index=self.xcats, columns=cols)
        df_xcats.loc["BENCH_XR"] = ["2017-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CONTRACT_XR"] = ["2018-01-01", "2020-12-31", 0.1, 1, 0, 0.3]

        self.dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.xcat = ["CONTRACT_XR"]
        self.cids = sorted(cids)
        self.benchmark_return = "USD_BENCH_XR"
        self.benchmark_cid, self.benchmark_xcat = self.benchmark_return.split("_", 1)

        self.be = BetaEstimator(
            df=self.dfd,
            xcats=self.xcat,
            cids=self.cids,
            benchmark_return=self.benchmark_return,
        )

        self.inner_splitters = {
            "expandingkfold": ExpandingKFoldPanelSplit(n_splits=3),
        }
        self.scorers = {"scorer": neg_mean_abs_corr}
        self.models = {
            "OLS": LinearRegressionSystem(),
        }
        self.hyperparameters = {
            "OLS": {"fit_intercept": [True, False]},
        }

        self.beta_xcat = "BETA_NSA"
        self.hedged_return_xcat = "HEDGED_RETURN_NSA"
        # Create some general betas and hedged returns
        self.be.estimate_beta(
            beta_xcat=self.beta_xcat,
            hedged_return_xcat=self.hedged_return_xcat,
            inner_splitters=self.inner_splitters,
            scorers=self.scorers,
            models=self.models,
            hyperparameters=self.hyperparameters,
            min_cids=1,
            min_periods=21 * 12,
            est_freq="M",
            n_jobs_outer=1,
            n_jobs_inner=1,
        )

    # def test_valid_init(self):
    #     # Check class attributes are correctly initialised
    #     be = BetaEstimator(
    #         df=self.dfd,
    #         xcats=self.xcat,
    #         cids=self.cids,
    #         benchmark_return=self.benchmark_return,
    #     )
    #     self.assertIsInstance(be, BetaEstimator)
    #     self.assertEqual(be.xcats, self.xcats)
    #     self.assertEqual(
    #         set([cid + "v" + be.benchmark_cid for cid in self.cids]), set(be.cids)
    #     )
    #     self.assertEqual(be.benchmark_return, self.benchmark_return)
    #     self.assertEqual(be.benchmark_cid, self.benchmark_cid)
    #     self.assertEqual(be.benchmark_xcat, self.benchmark_xcat)

    #     assert_frame_equal(
    #         be.betas,
    #         pd.DataFrame(columns=["cid", "real_date", "xcat", "value"]).astype(
    #             {
    #                 "cid": "category",
    #                 "real_date": "datetime64[ns]",
    #                 "xcat": "category",
    #                 "value": "float32",
    #             }
    #         ),
    #     )
    #     assert_frame_equal(
    #         be.hedged_returns,
    #         pd.DataFrame(columns=["cid", "real_date", "xcat", "value"]).astype(
    #             {
    #                 "cid": "category",
    #                 "real_date": "datetime64[ns]",
    #                 "xcat": "category",
    #                 "value": "float32",
    #             }
    #         ),
    #     )
    #     assert_frame_equal(
    #         be.chosen_models,
    #         pd.DataFrame(
    #             columns=[
    #                 "real_date",
    #                 "name",
    #                 "model_type",
    #                 "score",
    #                 "hparams",
    #                 "n_splits_used",
    #             ],
    #         ).astype(
    #             {
    #                 "real_date": "datetime64[ns]",
    #                 "name": "category",
    #                 "model_type": "category",
    #                 "score": "float32",
    #                 "hparams": "object",
    #                 "n_splits_used": "object",
    #             }
    #         ),
    #     )

    #     # Check the format of final long format dataframes
    #     self.assertIsInstance(be.X, pd.DataFrame)
    #     self.assertTrue(
    #         be.X.columns[0] == self.benchmark_return.split("_", maxsplit=1)[1]
    #     )
    #     self.assertTrue(be.X.index.names == ["cid", "real_date"])
    #     self.assertIsInstance(be.X.index, pd.MultiIndex)

    #     self.assertIsInstance(be.y, pd.Series)
    #     self.assertTrue(be.y.name == self.xcat[0])
    #     self.assertTrue(be.y.index.names == ["cid", "real_date"])
    #     self.assertIsInstance(be.y.index, pd.MultiIndex)

    #     # Check that the long format dataframes are correctly calculated
    #     dfd = self.dfd
    #     dfd["ticker"] = dfd["cid"] + "_" + dfd["xcat"]
    #     dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])
    #     for cid in self.cids:
    #         dfd_portion = dfd[
    #             (dfd.ticker == cid + "_" + self.xcat[0])
    #             | (dfd.ticker == self.benchmark_return)
    #         ]
    #         dfd_portion.loc[:, "cid"] = (
    #             f"{cid}v{self.benchmark_return.split('_', maxsplit=1)[0]}"
    #         )
    #         dfx = pd.concat([dfx, dfd_portion[["real_date", "cid", "xcat", "value"]]])

    #     dfx_long = dfx.pivot(
    #         index=["cid", "real_date"], columns="xcat", values="value"
    #     ).dropna()
    #     X = dfx_long[[self.benchmark_return.split("_", maxsplit=1)[1]]]
    #     X.columns.name = None
    #     X = X.astype("float32")
    #     y = dfx_long[self.xcat]

    #     assert_frame_equal(be.X, X)
    #     assert_series_equal(be.y, dfx_long[self.xcat[0]])

    # def test_types_init(self):
    #     # dataframe df
    #     with self.assertRaises(TypeError):
    #         be = BetaEstimator(
    #             df="not a dataframe",
    #             xcats=self.xcat,
    #             cids=self.cids,
    #             benchmark_return=self.benchmark_return,
    #         )
    #     with self.assertRaises(ValueError):
    #         be = BetaEstimator(
    #             df=self.dfd.iloc[:, :-1],
    #             xcats=self.xcat,
    #             cids=self.cids,
    #             benchmark_return=self.benchmark_return,
    #         )
    #     with self.assertRaises(ValueError):
    #         be = BetaEstimator(
    #             df=self.dfd[self.dfd.xcat != self.xcat],
    #             xcats=self.xcat,
    #             cids=self.cids,
    #             benchmark_return=self.benchmark_return,
    #         )
    #     # xcat
    #     with self.assertRaises(TypeError):
    #         be = BetaEstimator(
    #             df=self.dfd,
    #             xcats=1,
    #             cids=self.cids,
    #             benchmark_return=self.benchmark_return,
    #         )
    #     with self.assertRaises(ValueError):
    #         be = BetaEstimator(
    #             df=self.dfd,
    #             xcats="not a valid xcat",
    #             cids=self.cids,
    #             benchmark_return=self.benchmark_return,
    #         )
    #     # cids
    #     with self.assertRaises(TypeError):
    #         be = BetaEstimator(
    #             df=self.dfd,
    #             xcats=self.xcat,
    #             cids="not a list",
    #             benchmark_return=self.benchmark_return,
    #         )
    #     with self.assertRaises(ValueError):
    #         be = BetaEstimator(
    #             df=self.dfd,
    #             xcats=self.xcat,
    #             cids=self.cids + ["not a valid cid"],
    #             benchmark_return=self.benchmark_return,
    #         )
    #     with self.assertRaises(ValueError):
    #         be = BetaEstimator(
    #             df=self.dfd,
    #             xcats=self.xcat,
    #             cids=["not a valid cid"],
    #             benchmark_return=self.benchmark_return,
    #         )
    #     with self.assertRaises(ValueError):
    #         be = BetaEstimator(
    #             df=self.dfd,
    #             xcats=self.xcat,
    #             cids=self.cids + ["not a valid cid"],
    #             benchmark_return=self.benchmark_return,
    #         )
    #     # benchmark_return
    #     with self.assertRaises(TypeError):
    #         be = BetaEstimator(
    #             df=self.dfd,
    #             xcats=self.xcat,
    #             cids=self.cids,
    #             benchmark_return=1,
    #         )
    #     with self.assertRaises(ValueError):
    #         be = BetaEstimator(
    #             df=self.dfd,
    #             xcats=self.xcat,
    #             cids=self.cids,
    #             benchmark_return="not a valid benchmark_return",
    #         )
    #     with self.assertRaises(ValueError):
    #         be = BetaEstimator(
    #             df=self.dfd,
    #             xcats=self.xcat,
    #             cids=self.cids,
    #             benchmark_return="GLB_DRBXR_NSA",
    #         )

    # def test_valid_estimate_beta(self):
    #     # Test the broad method
    #     # Betas dataframe
    #     self.assertIsInstance(self.be.betas, pd.DataFrame)
    #     self.assertTrue(
    #         self.be.betas.columns.tolist() == ["cid", "real_date", "xcat", "value"]
    #     )
    #     self.assertTrue("BETA_NSA" in self.be.betas.xcat.unique())
    #     self.assertTrue(
    #         sorted(self.be.betas[self.be.betas.xcat == "BETA_NSA"].cid.unique())
    #         == self.cids
    #     )
    #     self.assertTrue(
    #         all(~self.be.betas[self.be.betas.xcat == "BETA_NSA"].value.isna())
    #     )
    #     # Hedged returns dataframe
    #     self.assertIsInstance(self.be.hedged_returns, pd.DataFrame)
    #     self.assertTrue(
    #         self.be.hedged_returns.columns.tolist()
    #         == ["cid", "real_date", "xcat", "value"]
    #     )
    #     self.assertTrue("HEDGED_RETURN_NSA" in self.be.hedged_returns.xcat.unique())
    #     self.assertTrue(
    #         sorted(
    #             self.be.hedged_returns[
    #                 self.be.hedged_returns.xcat == "HEDGED_RETURN_NSA"
    #             ].cid.unique()
    #         )
    #         == self.cids
    #     )
    #     self.assertTrue(
    #         all(
    #             ~self.be.hedged_returns[
    #                 self.be.hedged_returns.xcat == "HEDGED_RETURN_NSA"
    #             ].value.isna()
    #         )
    #     )

    @parameterized.expand(["HEDGED_RETURN_NSA", None])
    def test_valid_get_hedged_returns_single(self, hedged_return_xcat):
        # Test that the output is a dataframe
        be = self.be

        df1 = be.get_hedged_returns(hedged_return_xcat=hedged_return_xcat)
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(df1.columns.to_list(), ["real_date", "cid", "xcat", "value"])
        self.assertEqual(df1.xcat.unique()[0], self.hedged_return_xcat)

    def test_valid_get_hedged_returns_multiple(self):
        hedged_return_xcat2 = "HEDGED_RETURN_NSA2"
        be = self.be
        be.estimate_beta(
            beta_xcat=self.beta_xcat,
            hedged_return_xcat=hedged_return_xcat2,
            inner_splitters=self.inner_splitters,
            scorers=self.scorers,
            models=self.models,
            hyperparameters=self.hyperparameters,
            min_cids=1,
            min_periods=21 * 12,
            est_freq="M",
            n_jobs_outer=1,
            n_jobs_inner=1,
        )

        df1 = be.get_hedged_returns(hedged_return_xcat=hedged_return_xcat2)
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(df1.columns.to_list(), ["real_date", "cid", "xcat", "value"])
        self.assertEqual(df1.xcat.unique()[0], hedged_return_xcat2)

        df2 = be.get_hedged_returns(
            hedged_return_xcat=[self.hedged_return_xcat, hedged_return_xcat2]
        )
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(df2.columns.to_list(), ["real_date", "cid", "xcat", "value"])
        self.assertTrue(self.hedged_return_xcat in df2.xcat.unique())
        self.assertTrue(hedged_return_xcat2 in df2.xcat.unique())

        df3 = be.get_hedged_returns()
        self.assertIsInstance(df3, pd.DataFrame)
        self.assertEqual(df3.columns.to_list(), ["real_date", "cid", "xcat", "value"])
        self.assertTrue(self.hedged_return_xcat in df3.xcat.unique())
        self.assertTrue(hedged_return_xcat2 in df3.xcat.unique())

    def test_get_hedged_returns_checks(self):
        be = self.be
        with self.assertRaises(ValueError):
            be.get_hedged_returns(hedged_return_xcat="not a valid xcat")
        with self.assertRaises(ValueError):
            be.get_hedged_returns(hedged_return_xcat=["not a valid xcat"])
        with self.assertRaises(ValueError):
            be.get_hedged_returns(
                hedged_return_xcat=["not a valid xcat", "not a valid xcat"]
            )
        with self.assertRaises(TypeError):
            be.get_hedged_returns(hedged_return_xcat=1)
        with self.assertRaises(TypeError):
            be.get_hedged_returns(hedged_return_xcat=[1])
        with self.assertRaises(TypeError):
            be.get_hedged_returns(hedged_return_xcat=[self.hedged_return_xcat, 1])

    @parameterized.expand(["BETA_NSA", None])
    def test_valid_get_betas(self, beta_xcat):
        # Test that the output is a dataframe
        be = self.be

        df1 = be.get_betas(beta_xcat=beta_xcat)
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(df1.columns.to_list(), ["real_date", "cid", "xcat", "value"])
        self.assertEqual(df1.xcat.unique()[0], self.beta_xcat)

    def test_valid_get_betas_multiple(self):
        beta_xcat2 = "BETA_NSA2"
        be = self.be
        be.estimate_beta(
            beta_xcat=beta_xcat2,
            hedged_return_xcat=self.hedged_return_xcat,
            inner_splitters=self.inner_splitters,
            scorers=self.scorers,
            models=self.models,
            hyperparameters=self.hyperparameters,
            min_cids=1,
            min_periods=21 * 12,
            est_freq="M",
            n_jobs_outer=1,
            n_jobs_inner=1,
        )

        df1 = be.get_betas(beta_xcat=beta_xcat2)
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(df1.columns.to_list(), ["real_date", "cid", "xcat", "value"])
        self.assertEqual(df1.xcat.unique()[0], beta_xcat2)

        df2 = be.get_betas(beta_xcat=[self.beta_xcat, beta_xcat2])
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(df2.columns.to_list(), ["real_date", "cid", "xcat", "value"])
        self.assertTrue(self.beta_xcat in df2.xcat.unique())
        self.assertTrue(beta_xcat2 in df2.xcat.unique())

        df3 = be.get_betas()
        self.assertIsInstance(df3, pd.DataFrame)
        self.assertEqual(df3.columns.to_list(), ["real_date", "cid", "xcat", "value"])
        self.assertTrue(self.beta_xcat in df3.xcat.unique())
        self.assertTrue(beta_xcat2 in df3.xcat.unique())

    def test_get_betas_checks(self):
        be = self.be
        with self.assertRaises(ValueError):
            be.get_betas(beta_xcat="not a valid xcat")
        with self.assertRaises(ValueError):
            be.get_betas(beta_xcat=["not a valid xcat"])
        with self.assertRaises(ValueError):
            be.get_betas(beta_xcat=["not a valid xcat", self.hedged_return_xcat])
        with self.assertRaises(TypeError):
            be.get_betas(beta_xcat=1)
        with self.assertRaises(TypeError):
            be.get_betas(beta_xcat=[1])
        with self.assertRaises(TypeError):
            be.get_betas(beta_xcat=[self.hedged_return_xcat, 1])

    def test_evaluate_hedged_returns_checks(self):
        be = self.be
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(correlation_types=1)
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(correlation_types=[1])
        with self.assertRaises(ValueError):
            be.evaluate_hedged_returns(
                correlation_types=["not a valid correlation type"]
            )
        with self.assertRaises(ValueError):
            be.evaluate_hedged_returns(
                hedged_return_xcat=["not a valid xcat", "not a valid xcat"]
            )
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(hedged_return_xcat=1)
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(hedged_return_xcat=[1])
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(hedged_return_xcat=[self.hedged_return_xcat, 1])

        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(cids=1)
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(cids=[1])
        with self.assertRaises(ValueError):
            be.evaluate_hedged_returns(cids=["not a valid cid"])

        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(start=1)

        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(end=1)

        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(blacklist=1)
        with self.assertRaises(ValueError):
            be.evaluate_hedged_returns(blacklist=dict())
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(blacklist={1: "invalid"})
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(blacklist={"key": "invalid"})
        with self.assertRaises(ValueError):
            be.evaluate_hedged_returns(blacklist={"key": ()})
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(blacklist={"key": (1, 2)})

        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(freqs=1)
        with self.assertRaises(TypeError):
            be.evaluate_hedged_returns(freqs=[1])
        with self.assertRaises(ValueError):
            be.evaluate_hedged_returns(freqs="A")
        with self.assertRaises(ValueError):
            be.evaluate_hedged_returns(freqs=["Q", "A"])

    def test_evaluate_hedged_returns(self):

        be = self.be
        corr_df = be.evaluate_hedged_returns(
            correlation_types=["spearman", "pearson"],
            hedged_return_xcat=self.hedged_return_xcat,
            cids=self.cids,
            start="2018-01-01",
            end="2020-12-31",
            # blacklist={"AUD": ("2019-01-01", "2019-12-31")},
            freqs=["M", "Q"],
        )

        self.assertTrue(isinstance(corr_df, pd.DataFrame))
        self.assertEqual(
            corr_df.index.get_level_values(0).unique()[0], self.benchmark_return
        )
        self.assertTrue(
            self.hedged_return_xcat in corr_df.index.get_level_values(1).unique()
        )
        self.assertTrue(self.xcat[0] in corr_df.index.get_level_values(1).unique())
        self.assertTrue(
            {"M", "Q"}.issubset(set(corr_df.index.get_level_values(2).unique()))
        )
        self.assertTrue({"spearman", "pearson"}.issubset(set(corr_df.columns.unique())))


if __name__ == "__main__":

    Test = TestBetaEstimator()
    Test.setUpClass()
    Test.test_valid_estimate_beta()
