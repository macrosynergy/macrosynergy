import numpy as np
import pandas as pd 

from pandas.testing import assert_frame_equal, assert_series_equal

from macrosynergy.management.simulate import make_qdf
from macrosynergy.learning import (
    BetaEstimator,
    ExpandingKFoldPanelSplit,
    RollingKFoldPanelSplit,
    neg_mean_abs_corr,
    LinearRegressionSystem,
    RidgeRegressionSystem,
    ExpandingFrequencyPanelSplit
)
from macrosynergy.learning.beta_estimator import BetaEstimator

import unittest

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, make_scorer

from parameterized import parameterized

class TestBetaEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Simulate a panel dataset of benchmark and contract returns
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["BENCH_XR", "CONTRACT_XR"]
        cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

        # Set up the dataset so that June 2019 is the earliest date in the dataset
        df_cids = pd.DataFrame(
            index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31", 0, 1]
        df_cids.loc["CAD"] = ["2020-06-01", "2020-12-31", 0, 1]
        df_cids.loc["GBP"] = ["2020-01-01", "2020-12-31", 0, 1]
        df_cids.loc["USD"] = ["2019-01-01", "2020-12-31", 0, 1]

        df_xcats = pd.DataFrame(index=xcats, columns=cols)
        df_xcats.loc["BENCH_XR"] = ["2019-06-01", "2020-12-31", 0.1, 1, 0, 0.3]
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
        # The last 6 months of 2019 should be the initial training set
        # Then betas are estimated monthly for cross sections with at least 21 observations
        self.be.estimate_beta(
            beta_xcat="BETA_NSA",
            hedged_return_xcat="HEDGED_RETURN_NSA",
            inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
            scorer=neg_mean_abs_corr,
            models={
                "OLS": LinearRegressionSystem(min_xs_samples=21),
            },
            hparam_grid={
                "OLS": {"fit_intercept": [True, False]},
            },
            min_cids = 1,
            min_periods = 21 * 6,
            est_freq="M",
            use_variance_correction=False,
            n_jobs_outer=1,
        )

        # Test that it works with the RollingKFoldPanelSplit
        self.be.estimate_beta(
            beta_xcat="BETA_ROLL_NSA",
            hedged_return_xcat="HEDGED_RETURN_ROLL_NSA",
            inner_splitter=RollingKFoldPanelSplit(n_splits=3),
            scorer=neg_mean_abs_corr,
            models={
                "OLS": LinearRegressionSystem(min_xs_samples=21),
            },
            hparam_grid={
                "OLS": {"fit_intercept": [True, False]},
            },
            min_cids = 1,
            min_periods = 21 * 6,
            est_freq="M",
            use_variance_correction=False,
            n_jobs_outer=1,
        )

        # Test that a different model works

        self.be.estimate_beta(
            beta_xcat="BETA_RIDGE_NSA",
            hedged_return_xcat="HEDGED_RETURN_RIDGE_NSA",
            inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
            scorer=neg_mean_abs_corr,
            models={
                "Ridge": RidgeRegressionSystem(min_xs_samples=21),
            },
            hparam_grid={
                "Ridge": {"fit_intercept": [True, False], "alpha": [0.1, 1, 10]},
            },
            min_cids = 1,
            min_periods = 21 * 6,
            est_freq="M",
            use_variance_correction=False,
            n_jobs_outer=1,
        )

        # Test that using a VotingRegressor works

        self.be.estimate_beta(
            beta_xcat="BETA_VOTE_NSA",
            hedged_return_xcat="HEDGED_RETURN_VOTE_NSA",
            inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
            scorer=neg_mean_abs_corr,
            models={
                "Vote": VotingRegressor(
                    [
                        ("OLS1", LinearRegressionSystem(min_xs_samples=21, data_freq="unadjusted")),
                        ("OLS2", LinearRegressionSystem(min_xs_samples=21, data_freq="W")),
                    ],
               ),
            },
            hparam_grid={
                "Vote": {"OLS1__fit_intercept": [True, False], "OLS2__fit_intercept": [True, False]},
            },
            min_cids = 1,
            min_periods = 21 * 6,
            est_freq="M",
            use_variance_correction=False,
            n_jobs_outer=1,
        )

        # Test use_variance_correction works
        self.be.estimate_beta(
            beta_xcat="BETA_VC_NSA",
            hedged_return_xcat="HEDGED_RETURN_VC_NSA",
            inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
            scorer=neg_mean_abs_corr,
            models={
                "Ridge": RidgeRegressionSystem(min_xs_samples=21),
            },
            hparam_grid={
                "Ridge": {"fit_intercept": [True, False], "alpha": [0.1, 1, 10]},
            },
            min_cids = 1,
            min_periods = 21 * 6,
            est_freq="M",
            use_variance_correction=True,
            n_jobs_outer=1,
        )

        # Test that it works with the ExpandingFrequencyPanelSplit
        #self.be.estimate_beta(
        #    beta_xcat="BETA_CHG_NSA",
        #    hedged_return_xcat="HEDGED_RETURN_CHG_NSA",
        #    inner_splitter=ExpandingFrequencyPanelSplit(
        #        expansion_freq="Q",
        #        test_freq="M",
        #        min_cids = 1,
        #        min_periods = 21 * 3,
        #    ),
        #    scorer=neg_mean_abs_corr,
        #    models={
        #        "OLS": LinearRegressionSystem(min_xs_samples=21),
        #    },
        #    hparam_grid={
        #        "OLS": {"fit_intercept": [True, False]},
        #    },
        #    min_cids = 1,
        #    min_periods = 21 * 6,
        #    est_freq="M",
        #    use_variance_correction=False,
        #    n_jobs_outer=1,
        #)

        self.beta_names = ["BETA_NSA", "BETA_ROLL_NSA", "BETA_RIDGE_NSA", "BETA_VOTE_NSA", "BETA_VC_NSA"]
        self.hedged_return_names = ["HEDGED_RETURN_NSA", "HEDGED_RETURN_ROLL_NSA", "HEDGED_RETURN_RIDGE_NSA", "HEDGED_RETURN_VOTE_NSA", "HEDGED_RETURN_VC_NSA"]
        self.models_init = [
            LinearRegressionSystem(min_xs_samples=21),
            LinearRegressionSystem(min_xs_samples=21),
            RidgeRegressionSystem(min_xs_samples=21),
            VotingRegressor([("OLS1", LinearRegressionSystem(min_xs_samples=21, data_freq="unadjusted")),("OLS2", LinearRegressionSystem(min_xs_samples=21, data_freq="W"))]),
            RidgeRegressionSystem(min_xs_samples=21),
        ]

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
        
    def test_types_estimate_beta(self):
        """ beta_xcat """
        # Should fail if beta_xcat is not a string
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = 1,
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        # Should fail if beta_xcat is already in the class instance
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        """ hedged_return_xcat """
        # Should fail if hedged_return_xcat is not a string
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat=1,
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        # Should fail if hedged_return_xcat is already in the class instance
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        """ inner_splitter """
        # Should fail if inner_splitter is not a splitter
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=1,
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        # Should fail if inner_splitter doesn't subclass BasePanelSplit
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=TimeSeriesSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        """ scorer """
        # Should fail if scorer is not a callable
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=1,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        """ models """
        # Should fail if models is not a dictionary
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models=1,
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models=[LinearRegressionSystem(min_xs_samples=63)],
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if models is an empty dictionary
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={},
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if the keys in the models dictionary are not strings
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={1: LinearRegressionSystem(min_xs_samples=21)},
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if the values in the models dictionary are general scikit-learn 
        # regressors and aren't systems of linear models
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={"OLS": LinearRegression()},
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # should fail if the values in the models dictionary are not models at all
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={"OLS": 1},
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # should fail if the values in the models dictionary are VotingRegressors but 
        # the regressors are not systems of linear models
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={"OLS": VotingRegressor(
                        estimators=[("OLS1", LinearRegression()), ("OLS2", LinearRegression())]
                    )},
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={"OLS": VotingRegressor(
                        estimators=[("OLS1", LinearRegressionSystem(min_xs_samples=21)), ("OLS2", LinearRegression())]
                    )},
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={"OLS": VotingRegressor(
                        estimators=[("OLS1", LinearRegression()), ("OLS2", LinearRegressionSystem(min_xs_samples=21))]
                    )},
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        """ hparam_grid """
        # Should fail if hparam_grid is not a dictionary
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid=1,
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if hparam_grid is an empty dictionary
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={},
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if the keys in the hparam_grid dictionary differ from the keys in the models dictionary
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS2": {"fit_intercept": [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                ) 
        # Should fail if the values of the hyperparameter grid are neither dictionaries
        # nor lists of dictionaries
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": 1,
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": [1],
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if, when a dictionary of hyperparameters is provided, not all keys
        # in the inner dictionary are strings
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": {1: [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False], 1: [True, False]},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if, when a dictionary of hyperparameters is provided, not all values
        # in the inner dictionary are lists
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": {"fit_intercept": 1},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": {"fit_intercept": [True, False], "positive": 1},
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if, when a list of dictionaries of hyperparameters is provided, not all
        # lists contain dictionaries
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": [1],
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": [{"fit_intercept": [True, False]}, 2],
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if, when a list of dictionaries of hyperparameters is provided, not all
        # dictionaries contain strings as keys
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": [{1: [True, False]}],
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": [{"fit_intercept": [True, False], 1: [True, False]}],
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        # Should fail if, when a list of dictionaries of hyperparameters is provided, not all
        # dictionaries contain lists as values
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": [{"fit_intercept": 1}],
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                    beta_xcat = "BETA_NSA2",
                    hedged_return_xcat="HEDGED_RETURN_NSA2",
                    inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                    scorer=neg_mean_abs_corr,
                    models={
                        "OLS": LinearRegressionSystem(min_xs_samples=21),
                    },
                    hparam_grid={
                        "OLS": [{"fit_intercept": [True, False], "positive": 1}],
                    },
                    min_cids = 1,
                    min_periods = 21 * 6,
                    est_freq="M",
                    use_variance_correction=False,
                    n_jobs_outer=1,
                )
        
        """ min_cids """
        # Should fail if min_cids is not an integer
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = "not an integer",
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        # Should fail if min_cids is less than 1
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 0,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = -4,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        # Should fail if min_cids is a float
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1.2,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        
        """ min_periods """
        # Should fail if min_periods is not an integer
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = "not an integer",
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        # Should fail if min_periods is less than 1
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 0,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = -4,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        # Should fail if min_periods is a float
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 1.2,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
            )

        """ est_freq """
        # Should fail if est_freq is not a string
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq=1,
                use_variance_correction=False,
                n_jobs_outer=1,
            )
        # Should fail if est_freq is not a valid frequency string
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="not a valid frequency string",
                use_variance_correction=False,
                n_jobs_outer=1,
            )

        """ use_variance_correction """
        # Should fail if use_variance_correction is not a boolean
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=1,
                n_jobs_outer=1,
            )

        """ initial_n_splits """
        # Should fail if, when specified, initial_n_splits is not an integer
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                initial_n_splits="not an integer",
            )
        # Should fail if, when specified, initial_n_splits is less than 2
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                initial_n_splits=1,
            )
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                initial_n_splits=-3,
            )
        
        """ threshold_n_periods """
        # Should fail if, when specified, threshold_n_periods is not an integer
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                initial_n_splits = 5,
                threshold_n_periods="not an integer",
            )
        # Should fail if, when specified, threshold_n_periods is less than 1
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                initial_n_splits=5,
                threshold_n_periods=0,
            )
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                initial_n_splits=5,
                threshold_n_periods=-5,
            )
        
        """ hparam_type """
        # Should fail if hparam_type is not a string
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                hparam_type=1,
            )
        # Should fail if hparam_type is not a valid string
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                hparam_type="not a valid string",
            )
        # Should raise a NotImplementedError if hparam_type is "bayes"
        with self.assertRaises(NotImplementedError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                hparam_type="bayes",
            )

        """ n_iter """
        # Should fail if n_iter is not an integer
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                hparam_type="prior",
                n_iter="not an integer",
            )
        # Should fail if n_iter is less than 1
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                hparam_type="prior",
                n_iter=0,
            )
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                hparam_type="prior",
                n_iter=-5,
            )
        
        """ n_jobs_outer """
        # Should fail if n_jobs_outer is not an integer
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer="not an integer",
            )
        # Should fail if n_jobs_outer is less than 1 (other than -1)
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=0,
            )
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=-5,
            )

        """ n_jobs_inner """
        # Should fail if n_jobs_inner is not an integer
        with self.assertRaises(TypeError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                n_jobs_inner="not an integer",
            )

        # Should fail if n_jobs_outer is less than 1 (other than -1)
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                n_jobs_inner=0,
            )
        with self.assertRaises(ValueError):
            self.be.estimate_beta(
                beta_xcat = "BETA_NSA2",
                hedged_return_xcat="HEDGED_RETURN_NSA2",
                inner_splitter=ExpandingKFoldPanelSplit(n_splits=3),
                scorer=neg_mean_abs_corr,
                models={
                    "OLS": LinearRegressionSystem(min_xs_samples=21),
                },
                hparam_grid={
                    "OLS": {"fit_intercept": [True, False]},
                },
                min_cids = 1,
                min_periods = 21 * 6,
                est_freq="M",
                use_variance_correction=False,
                n_jobs_outer=1,
                n_jobs_inner=-2,
            )

    def test_valid_estimate_beta(self):
        determined_betas = self.be.betas.sort_values(by=["cid","xcat","real_date"]).drop_duplicates(subset=["value","xcat","cid"])
        determined_hedged_returns = self.be.hedged_returns.sort_values(by=["cid","xcat","real_date"])
        determined_optimal_models = self.be.get_optimal_models()

        # Loop through training and test splits, and determine correct training times, betas and hedged returns
        outer_splitter = ExpandingFrequencyPanelSplit(
            expansion_freq="M",
            test_freq="M",
            min_cids = 1,
            min_periods = 21 * 6,
        )

        estimation_dates = []

        correct_betas = {pipeline_name: {cross_section: [] for cross_section in ["AUDvUSD", "CADvUSD", "GBPvUSD", "USDvUSD"]} for pipeline_name in self.beta_names}
        correct_hedged_returns = {pipeline_name: {cross_section: {} for cross_section in ["AUDvUSD", "CADvUSD", "GBPvUSD", "USDvUSD"]} for pipeline_name in self.hedged_return_names}
        for idx, (train_idx, test_idx) in enumerate(outer_splitter.split(self.be.X, self.be.y)):
            # Get training and test sets
            X_train, y_train = self.be.X.iloc[train_idx], self.be.y.iloc[train_idx]
            X_test, y_test = self.be.X.iloc[test_idx], self.be.y.iloc[test_idx]
            # store the true re-estimation dates
            real_reest_date = X_train.index.get_level_values("real_date").max()
            estimation_dates.append(real_reest_date)
            # store the right betas based on the selected model
            hparams = determined_optimal_models[determined_optimal_models.real_date == real_reest_date][["xcat","hparams"]].reset_index(drop=True)
            for idx, row in hparams.iterrows():
                selected_model = self.models_init[idx].set_params(**row["hparams"])
                selected_model.fit(pd.DataFrame(X_train), y_train)
                if isinstance(selected_model, VotingRegressor):
                    # average the betas for simplicity
                    try:
                        betas = {beta_xs: np.mean([model.coefs_[beta_xs] for model in selected_model.estimators_]) for beta_xs in set(selected_model.estimators_[1].coefs_.keys()).intersection(selected_model.estimators_[0].coefs_.keys())}
                    except Exception as e:
                        pass
                else:
                    betas = selected_model.coefs_
                for beta_xs, beta in betas.items():
                    correct_betas[row["xcat"]][beta_xs].append(beta)
                # for this model, store the right out-of-sample hedged returns 
                oos_cross_sections = X_test.index.get_level_values("cid").unique()
                for cid in oos_cross_sections:
                    if cid in X_train.index.get_level_values("cid"):
                        if len(X_train.xs(cid)) > 21:
                            # Get the beta for this cross-section
                            beta = betas[cid]
                            # Create hedged return series
                            oos_hedged_returns = (y_test.xs(cid) - beta * X_test.xs(cid)).values
                            correct_hedged_returns[self.hedged_return_names[idx]][cid][real_reest_date] = oos_hedged_returns

        # check basic beta dataframe properties
        self.assertIsInstance(self.be.betas, pd.DataFrame)
        self.assertTrue(self.be.betas.columns.tolist() == ["cid", "real_date", "xcat", "value"])
        for beta_xcat in self.beta_names:
            self.assertTrue(beta_xcat in self.be.betas.xcat.unique())
            self.assertTrue(sorted(self.be.betas[self.be.betas.xcat == beta_xcat].cid.unique()) == self.cids)
            self.assertTrue(all(~self.be.betas[self.be.betas.xcat == beta_xcat].value.isna()))
        # check the estimation dates are as expected 
        real_reest_dates = sorted(estimation_dates)
        for beta_xcat in self.beta_names:
            determined_reest_dates = sorted(determined_betas[determined_betas.xcat == beta_xcat].real_date.unique())
            self.assertTrue(np.all(determined_reest_dates == real_reest_dates))
        # check the betas themselves are as expected
        for beta_xcat in self.beta_names:
            for beta_xs, beta in correct_betas[beta_xcat].items():
                try:
                    self.assertTrue(np.all(determined_betas[(determined_betas.cid + "vUSD" == beta_xs) & (determined_betas.xcat == beta_xcat)].value == beta))
                except Exception as e:
                    pass
        # check basic hedged return dataframe properties
        self.assertIsInstance(self.be.hedged_returns, pd.DataFrame)
        self.assertTrue(self.be.hedged_returns.columns.tolist() == ["cid", "real_date", "xcat", "value"])
        for hedged_return_xcat in self.hedged_return_names:
            self.assertTrue(hedged_return_xcat in self.be.hedged_returns.xcat.unique())
            self.assertTrue(sorted(self.be.hedged_returns[self.be.hedged_returns.xcat == hedged_return_xcat].cid.unique()) == self.cids)
            self.assertTrue(all(~self.be.hedged_returns[self.be.hedged_returns.xcat == hedged_return_xcat].value.isna()))
        # check the hedged returns themselves are as expected
        for hedged_return_idx, hedged_return_xcat in enumerate(self.hedged_return_names):
            for idx in range(len(real_reest_dates)-1):
                # Get all cross-sections that should have hedged returns between this estimation date and the next
                relevant_xss = determined_betas[(determined_betas.real_date == real_reest_dates[idx]) & (determined_betas.xcat == self.beta_names[hedged_return_idx])].cid.values
                # Get all calculated hedged returns between re-estimation dates
                for cid in relevant_xss:
                    subset_det_hedged_rets = determined_hedged_returns[(determined_hedged_returns.cid == cid) & (determined_hedged_returns.real_date > real_reest_dates[idx]) & (determined_hedged_returns.real_date <= real_reest_dates[idx+1]) & (determined_hedged_returns.xcat == hedged_return_xcat)]
                    self.assertTrue(len(correct_hedged_returns[hedged_return_xcat][cid+"vUSD"][real_reest_dates[idx]]) == len(subset_det_hedged_rets.value.values))
                    self.assertTrue(np.all(correct_hedged_returns[hedged_return_xcat][cid+"vUSD"][real_reest_dates[idx]]==subset_det_hedged_rets.value.values))

    def test_types_evaluate_hedged_returns(self):
        """ hedged_rets """
        # Should fail if hedged_rets is neither a string or a List of strings
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(hedged_rets=1)
        # Should fail if hedged_rets is a list of strings, but not all elements are strings
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(hedged_rets=[1, "HEDGED_RETURN_NSA2"])
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(hedged_rets=["HEDGED_RETURN_NSA1", 1])
        # Should fail if a string hedged return is not in the hedged returns dataframe
        with self.assertRaises(ValueError):
            self.be.evaluate_hedged_returns(hedged_rets="not a valid hedged return")
        # Should fail if a list of hedged returns contains a string that is not in the hedged returns dataframe
        with self.assertRaises(ValueError):
            self.be.evaluate_hedged_returns(hedged_rets=["HEDGED_RETURN_NSA1", "not a valid hedged return"])
        
        """ cids """
        # Should fail if cids is neither a string or a List of strings
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(hedged_rets="HEDGED_RETURN_NSA", cids=1)
        # Should fail if cids is a list of strings, but not all elements are strings
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(hedged_rets="HEDGED_RETURN_NSA", cids=[1, "AUD"])
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(hedged_rets="HEDGED_RETURN_NSA", cids=["AUD", 1])
        # Should fail if a string cid is not in the cids list
        with self.assertRaises(ValueError):
            self.be.evaluate_hedged_returns(hedged_rets="HEDGED_RETURN_NSA", cids="not a valid cid")
        # Should fail if a list of cids contains a string that is not in the cids list
        with self.assertRaises(ValueError):
            self.be.evaluate_hedged_returns(hedged_rets="HEDGED_RETURN_NSA", cids=["AUD", "not a valid cid"])
        
        """ correlation_types """
        # Should fail if correlation_types is not a string
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(correlation_types=1)
        # Should fail if correlation_types is not a valid string
        with self.assertRaises(ValueError):
            self.be.evaluate_hedged_returns(correlation_types="not a valid correlation type")
        
        """ title """
        # Should fail if title is not a string
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(title=1)
        
        """ start """
        # Should fail if start is not a string
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(start=1)

        # Should fail if start is not in ISO format
        with self.assertRaises(ValueError):
            self.be.evaluate_hedged_returns(start="hello")
        
        """ end """
        # Should fail if end is not a string
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(end=1)

        # Should fail if end is not in ISO format
        with self.assertRaises(ValueError):
            self.be.evaluate_hedged_returns(end="hello")

        """ blacklist """
        # Should fail if blacklist is not a dictionary
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(blacklist=1)
        # Should fail if blacklist is a dictionary, but not all keys are strings
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(blacklist={1: ["AUD"]})
        # Should fail if blacklist is a dictionary, but not all values are tuples
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(blacklist={"HEDGED_RETURN_NSA1": "AUD"})
        # Should fail if one of the elements in the tuple isn't a timestamp
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(
                blacklist={"AUD": (pd.Timestamp(year=1999,month=8,day=20), "not a valid timestamp")})

        """ freqs """
        # Should fail if freqs is not a string or a list of strings
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(freqs=1)
        # Should fail if freqs is a list of strings, but not all elements are strings
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(freqs=[1, "M"])
        with self.assertRaises(TypeError):
            self.be.evaluate_hedged_returns(freqs=["M", 1])
        # Should fail if freqs is a string, but not a valid frequency string
        with self.assertRaises(ValueError):
            self.be.evaluate_hedged_returns(freqs="not a valid frequency string")

    def test_valid_evaluate_hedged_returns(self):
        # Check that running evaluate_hedged_returns without any arguments
        # creates a dataframe as expected
        df = self.be.evaluate_hedged_returns()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) == len(self.hedged_return_names) + 1)
        self.assertTrue(np.all(df.index.names == ["benchmark return", "return category", "frequency"]))
        self.assertTrue(len(df.index.get_level_values(0).unique()) == 1)
        self.assertTrue(df.index.get_level_values(0)[0] == self.be.benchmark_return)
        self.assertTrue(np.all(df.index.get_level_values(1) == self.hedged_return_names + [self.xcat]))
        self.assertTrue(np.all(df.index.get_level_values(2) == "M"))
        self.assertTrue(len(df.columns) == 1)
        self.assertTrue("|pearson|" == df.columns[0])
        self.assertTrue(all(df.iloc[:,0].apply(lambda x: isinstance(x, float))))
        self.assertTrue(all(df.iloc[:,0] >= 0))
        # Check that this works with different correlation types
        df = self.be.evaluate_hedged_returns(correlation_types=["pearson", "spearman", "kendall"])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) == len(self.hedged_return_names) + 1)
        self.assertTrue(np.all(df.index.names == ["benchmark return", "return category", "frequency"]))
        self.assertTrue(len(df.index.get_level_values(0).unique()) == 1)
        self.assertTrue(df.index.get_level_values(0)[0] == self.be.benchmark_return)
        self.assertTrue(np.all(df.index.get_level_values(1) == self.hedged_return_names + [self.xcat]))
        self.assertTrue(np.all(df.index.get_level_values(2) == "M"))
        self.assertTrue(len(df.columns) == 3)
        self.assertTrue(np.all(["|pearson|","|spearman|","|kendall|"] == df.columns))
        self.assertTrue(all(df.iloc[:,0].apply(lambda x: isinstance(x, float))))
        self.assertTrue(all(df.iloc[:,1].apply(lambda x: isinstance(x, float))))
        self.assertTrue(all(df.iloc[:,2].apply(lambda x: isinstance(x, float))))
        self.assertTrue(all(df.iloc[:,0] >= 0))
        self.assertTrue(all(df.iloc[:,1] >= 0))
        self.assertTrue(all(df.iloc[:,2] >= 0))
        # Check that the default works with different frequencies
        df = self.be.evaluate_hedged_returns(freqs = ["D", "W", "M", "Q"])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) == 4*len(self.hedged_return_names) + 4)
        self.assertTrue(np.all(df.index.names == ["benchmark return", "return category", "frequency"]))
        self.assertTrue(len(df.index.get_level_values(0).unique()) == 1)
        self.assertTrue(df.index.get_level_values(0)[0] == self.be.benchmark_return)
        self.assertTrue(np.all(df.index.get_level_values(1) == [hedged_return_name for hedged_return_name in self.hedged_return_names for i in range(4)] + [self.xcat] * 4))
        self.assertTrue(np.all(df.index.get_level_values(2) == ["D", "W", "M", "Q"] * 6))
        self.assertTrue(len(df.columns) == 1)
        self.assertTrue("|pearson|" == df.columns[0])
        self.assertTrue(all(df.iloc[:,0].apply(lambda x: isinstance(x, float))))
        self.assertTrue(all(df.iloc[:,0] >= 0))
        # Check that both different frequencies and correlation types work
        df = self.be.evaluate_hedged_returns(freqs = ["D", "W", "M", "Q"], correlation_types=["pearson", "spearman", "kendall"])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) == 4*len(self.hedged_return_names) + 4)
        self.assertTrue(np.all(df.index.names == ["benchmark return", "return category", "frequency"]))
        self.assertTrue(len(df.index.get_level_values(0).unique()) == 1)
        self.assertTrue(df.index.get_level_values(0)[0] == self.be.benchmark_return)
        self.assertTrue(np.all(df.index.get_level_values(1) == [hedged_return_name for hedged_return_name in self.hedged_return_names for i in range(4)] + [self.xcat] * 4))
        self.assertTrue(np.all(df.index.get_level_values(2) == ["D", "W", "M", "Q"] * 6))
        self.assertTrue(len(df.columns) == 3)
        self.assertTrue(np.all(["|pearson|","|spearman|","|kendall|"] == df.columns))
        self.assertTrue(all(df.iloc[:,0].apply(lambda x: isinstance(x, float))))
        self.assertTrue(all(df.iloc[:,1].apply(lambda x: isinstance(x, float))))
        self.assertTrue(all(df.iloc[:,2].apply(lambda x: isinstance(x, float))))
        self.assertTrue(all(df.iloc[:,0] >= 0))
        self.assertTrue(all(df.iloc[:,1] >= 0))
        self.assertTrue(all(df.iloc[:,2] >= 0))