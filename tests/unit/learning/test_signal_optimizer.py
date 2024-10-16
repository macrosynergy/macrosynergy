import datetime
import itertools
import unittest
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from parameterized import parameterized
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from macrosynergy.learning import (
    ExpandingIncrementPanelSplit,
    ExpandingKFoldPanelSplit,
    LassoSelector,
    RollingKFoldPanelSplit,
    SignalOptimizer,
    regression_balanced_accuracy,
)
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils.df_utils import categories_df


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        self.cids = ["AUD", "CAD", "GBP", "USD"]
        self.xcats = ["XR", "CPI", "GROWTH", "RIR"]
        cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD"] = ["2014-01-01", "2020-12-31", 0, 1]
        df_cids.loc["CAD"] = ["2015-01-01", "2020-12-31", 0, 1]
        df_cids.loc["GBP"] = ["2015-01-01", "2020-12-31", 0, 1]
        df_cids.loc["USD"] = ["2015-01-01", "2020-12-31", 0, 1]

        df_xcats = pd.DataFrame(index=self.xcats, columns=cols)
        df_xcats.loc["XR"] = ["2014-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CPI"] = ["2015-01-01", "2020-12-31", 1, 2, 0.95, 1]
        df_xcats.loc["GROWTH"] = ["2015-01-01", "2020-12-31", 1, 2, 0.9, 1]
        df_xcats.loc["RIR"] = ["2015-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

        self.df = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.df["value"] = self.df["value"].astype("float32")

        # create blacklist dictionary
        self.black_valid = {
            "AUD": (
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2020, month=4, day=1),
            ),
            "GBP": (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2020, month=2, day=1),
            ),
        }
        self.black_invalid1 = {
            "AUD": ["2018-09-01", "2018-10-01"],
            "GBP": ["2019-06-01", "2100-01-01"],
        }
        self.black_invalid2 = {
            "AUD": ("2018-09-01", "2018-10-01"),
            "GBP": ("2019-06-01", "2100-01-01"),
        }
        self.black_invalid3 = {
            "AUD": [
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2018, month=10, day=1),
            ],
            "GBP": [
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2100, month=1, day=1),
            ],
        }
        self.black_invalid4 = {
            "AUD": (pd.Timestamp(year=2018, month=9, day=1),),
            "GBP": (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2100, month=1, day=1),
            ),
        }
        self.black_invalid5 = {
            1: (
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2018, month=10, day=1),
            ),
            2: (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2100, month=1, day=1),
            ),
        }

        # models dictionary
        self.models = {
            "linreg": LinearRegression(),
            "ridge": Ridge(),
        }
        self.hyperparameters = {
            "linreg": {},
            "ridge": {"alpha": [0.1, 1.0]},
        }
        self.scorers = {
            "r2": make_scorer(r2_score),
            "bac": make_scorer(regression_balanced_accuracy),
        }
        # # instantiate some splitters
        # expandingkfold = ExpandingKFoldPanelSplit(n_splits=4)
        # rollingkfold = RollingKFoldPanelSplit(n_splits=4)
        # expandingincrement = ExpandingIncrementPanelSplit(min_cids=2, min_periods=100)
        # self.splitters = {
        #     "expanding_kfold": expandingkfold,
        #     "rolling": rollingkfold,
        #     "expanding": expandingincrement,
        # }
        self.inner_splitters = {
            "ExpandingKFold": ExpandingKFoldPanelSplit(n_splits=4),
            "RollingKFold": RollingKFoldPanelSplit(n_splits=4),
        }
        self.single_inner_splitter = {
            "ExpandingKFold": ExpandingKFoldPanelSplit(n_splits=4),
        }

        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
        )
        # Now run calculate_predictions
        so.calculate_predictions(
            name="test",
            models=self.models,
            scorers=self.scorers,
            hyperparameters=self.hyperparameters,
            search_type="grid",
            n_jobs_outer=1,
            n_jobs_inner=1,
            inner_splitters=self.single_inner_splitter,
        )
        self.so_with_calculated_preds = so

        self.X, self.y, self.df_long = _get_X_y(so)

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(self.mpl_backend)

    @parameterized.expand(itertools.product([True, False], [True, False]))
    def test_valid_init(self, use_blacklist, use_cids):
        try:
            blacklist = self.black_valid if use_blacklist else None
            cids = self.cids if use_cids else None
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                cids=cids,
                blacklist=blacklist,
            )
        except Exception as e:
            self.fail(f"Instantiation of the SignalOptimizer raised an exception: {e}")
        self.assertIsInstance(so, SignalOptimizer)
        X, y, df_long = _get_X_y(so)
        pd.testing.assert_frame_equal(so.X, X)
        pd.testing.assert_series_equal(so.y, y)

        pd.testing.assert_frame_equal(
            so.chosen_models,
            pd.DataFrame(
                columns=[
                    "real_date",
                    "name",
                    "model_type",
                    "score",
                    "hparams",
                    "n_splits_used",
                ]
            ),
        )
        pd.testing.assert_frame_equal(
            so.preds,
            pd.DataFrame(columns=["cid", "real_date", "xcat", "value"]),
        )
        pd.testing.assert_frame_equal(
            so.ftr_coefficients,
            pd.DataFrame(columns=["real_date", "name"] + list(so.X.columns)),
        )
        pd.testing.assert_frame_equal(
            so.intercepts, pd.DataFrame(columns=["real_date", "name", "intercepts"])
        )

        min_date = min(so.unique_date_levels)
        max_date = max(so.unique_date_levels)
        forecast_date_levels = pd.date_range(start=min_date, end=max_date, freq="B")
        pd.testing.assert_index_equal(
            so.forecast_idxs,
            pd.MultiIndex.from_product(
                [so.unique_xs_levels, forecast_date_levels],
                names=["cid", "real_date"],
            ),
        )

    def test_types_init(self):
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=pd.Series([]),
                xcats=self.xcats,
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats="invalid",
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=[1, 2, 3],
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                cids="invalid",
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                cids=[1, 2, 3],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=pd.DataFrame([]),
                xcats=self.xcats,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=["invalid"],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats + ["invalid"],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                cids=self.cids + ["invalid"],
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                start=1,
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                end=1,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                start="invalid",
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                end="invalid",
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                blacklist=list(),
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                blacklist={1: "invalid"},
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                blacklist={"CAD": "invalid"},
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                blacklist={"CAD": tuple()},
            )

    def test_valid_calculate_predictions(self):
        so1 = self.so_with_calculated_preds
        df1 = so1.preds.copy()
        self.assertIsInstance(df1, pd.DataFrame)
        if len(df1.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df1.xcat.unique()[0], "test")
        # Test that blacklisting works as expected
        so2 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            blacklist=self.black_valid,
        )
        try:
            so2.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")

        df2 = so2.preds.copy()
        self.assertIsInstance(df2, pd.DataFrame)
        for cross_section, periods in self.black_valid.items():
            cross_section_key = cross_section.split("_")[0]
            self.assertTrue(
                len(
                    df2[
                        (df2.cid == cross_section_key)
                        & (df2.real_date >= periods[0])
                        & (df2.real_date <= periods[1])
                    ].dropna()
                )
                == 0
            )
        # Test that rolling models work as expected
        so3 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        try:
            so3.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="prior",
                n_iter=1,
                max_periods=21,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df3 = so3.preds.copy()
        self.assertIsInstance(df3, pd.DataFrame)
        if len(df3.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df3.xcat.unique()[0], "test")
        # Test that an unreasonably large roll is equivalent to no roll
        so4 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        try:
            so4.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                max_periods=int(1e6),  # million days roll
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df4 = so4.preds.copy()
        self.assertIsInstance(df4, pd.DataFrame)
        if len(df4.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df4.xcat.unique()[0], "test")
        self.assertTrue(df1.equals(df4))

    def test_types_run(self):
        # Training set only
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        outer_splitter = ExpandingIncrementPanelSplit()

        # Valid parameters
        valid_params = {
            "name": "test",
            "outer_splitter": outer_splitter,
            "inner_splitters": self.single_inner_splitter,
            "models": self.models,
            "hyperparameters": self.hyperparameters,
            "scorers": self.scorers,
            "normalize_fold_results": True,
            "search_type": "prior",
            "cv_summary": "median",
            "n_iter": 5,
            "split_functions": None,
            "n_jobs_outer": 1,
            "n_jobs_inner": 1,
        }

        # Check the valid case to ensure it passes without error
        so._check_run(**valid_params)

        # A mapping of parameters to invalid values
        invalid_params = {
            "name": 123,  # Invalid: should be a string
            "outer_splitter": "invalid_splitter",  # Invalid: should be a splitter object
            "inner_splitters": "invalid_splitter",  # Invalid: should be a list or splitter object
            "models": None,  # Invalid: should be a model list
            "hyperparameters": "invalid_hyperparameters",  # Invalid: should be a dict or iterable
            "scorers": None,  # Invalid: should be a scorer or list of scorers
            "normalize_fold_results": "invalid_boolean",  # Invalid: should be a boolean
            "search_type": 123,  # Invalid: should be a string
            "cv_summary": 123,  # Invalid: should be a string
            "n_iter": "invalid_integer",  # Invalid: should be an integer
            "split_functions": "invalid_split_functions",  # Invalid: should be a function or None
            "n_jobs_outer": "invalid_jobs",  # Invalid: should be an integer
            "n_jobs_inner": "invalid_jobs",  # Invalid: should be an integer
        }

        # Loop through the invalid cases
        for param, invalid_value in invalid_params.items():
            invalid_case = valid_params.copy()  # Start with valid params
            invalid_case[param] = invalid_value  # Introduce invalid value
            with self.assertRaises(
                TypeError,
                msg=f"Expected TypeError for invalid '{param}' but didn't get it. "
                f"Invalid value: {invalid_value}",
            ):
                so._check_run(**invalid_case)

    def test_types_get_optimized_signals(
        self,
    ):
        # Test that invalid names are caught
        so = self.so_with_calculated_preds
        with self.assertRaises(TypeError):
            so.get_optimized_signals(name=1)
        with self.assertRaises(TypeError):
            so.get_optimized_signals(name={})
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name=["test", "test2"])
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name="test2")
        # Test that if no signals have been calculated, an error is raised
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name="test2")

    def test_valid_get_optimized_signals(self):
        # Test that the output is a dataframe
        so = self.so_with_calculated_preds

        df1 = so.get_optimized_signals(name="test")
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(df1.shape[1], 4)
        self.assertEqual(df1.columns[0], "cid")
        self.assertEqual(df1.columns[1], "real_date")
        self.assertEqual(df1.columns[2], "xcat")
        self.assertEqual(df1.columns[3], "value")
        self.assertEqual(df1.xcat.unique()[0], "test")
        # Add a second signal and check that the output is a dataframe
        so.calculate_predictions(
            name="test2",
            models=self.models,
            scorers=self.scorers,
            hyperparameters=self.hyperparameters,
            search_type="grid",
            n_jobs_outer=1,
            n_jobs_inner=1,
            inner_splitters=self.inner_splitters,
        )
        df2 = so.get_optimized_signals(name="test2")
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(df2.shape[1], 4)
        self.assertEqual(df2.columns[0], "cid")
        self.assertEqual(df2.columns[1], "real_date")
        self.assertEqual(df2.columns[2], "xcat")
        self.assertEqual(df2.columns[3], "value")
        self.assertEqual(df2.xcat.unique()[0], "test2")
        df3 = so.get_optimized_signals()
        self.assertIsInstance(df3, pd.DataFrame)
        self.assertEqual(df3.shape[1], 4)
        self.assertEqual(df3.columns[0], "cid")
        self.assertEqual(df3.columns[1], "real_date")
        self.assertEqual(df3.columns[2], "xcat")
        self.assertEqual(df3.columns[3], "value")
        self.assertEqual(len(df3.xcat.unique()), 2)
        df4 = so.get_optimized_signals(name=["test", "test2"])
        self.assertIsInstance(df4, pd.DataFrame)
        self.assertEqual(df4.shape[1], 4)
        self.assertEqual(df4.columns[0], "cid")
        self.assertEqual(df4.columns[1], "real_date")
        self.assertEqual(df4.columns[2], "xcat")
        self.assertEqual(df4.columns[3], "value")
        self.assertEqual(len(df4.xcat.unique()), 2)

    def test_types_get_optimal_models(self):

        so = self.so_with_calculated_preds
        with self.assertRaises(TypeError):
            so.get_optimal_models(name=1)
        with self.assertRaises(TypeError):
            so.get_optimal_models(name={})
        with self.assertRaises(ValueError):
            so.get_optimal_models(name=["test", "test2"])
        with self.assertRaises(ValueError):
            so.get_optimal_models(name="test2")

        # Test that if no signals have been calculated, an error is raised
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        with self.assertRaises(ValueError):
            so.get_optimal_models(name="test2")

    # @parameterized.expand(itertools.product([1], [True, False]))

    def test_valid_get_optimal_models(self):

        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        so.calculate_predictions(
            name="test",
            models=self.models,
            scorers=self.scorers,
            hyperparameters=self.hyperparameters,
            search_type="grid",
            n_jobs_outer=1,
            n_jobs_inner=1,
            inner_splitters=self.single_inner_splitter,
        )
        df1 = so.get_optimal_models(name="test")
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(df1.shape[1], 6)
        self.assertEqual(df1.columns[0], "real_date")
        self.assertEqual(df1.columns[1], "name")
        self.assertEqual(df1.columns[2], "model_type")
        self.assertEqual(df1.columns[3], "score")
        self.assertEqual(df1.columns[4], "hparams")
        self.assertEqual(df1.columns[5], "n_splits_used")
        self.assertEqual(df1.name.unique()[0], "test")
        #     if not change_n_splits:
        #         self.assertTrue(
        #             np.all(df1.iloc[:, 4] == self.splitters[splitter_idx].n_splits)
        #         )
        #     else:
        #         num_new_dates = len(
        #             pd.bdate_range(df1.real_date.min(), df1.real_date.max())
        #         )
        #         self.assertTrue(np.min(df1.iloc[:, 4]) == initial_nsplits)
        #         self.assertTrue(
        #             np.max(df1.iloc[:, 4])
        #             == initial_nsplits + (num_new_dates // threshold_ndates)
        #         )

        # Add a second signal and check that the output is a dataframe
        so.calculate_predictions(
            name="test2",
            models=self.models,
            scorers=self.scorers,
            hyperparameters=self.hyperparameters,
            search_type="grid",
            n_jobs_outer=1,
            n_jobs_inner=1,
            inner_splitters=self.single_inner_splitter,
        )
        df2 = so.get_optimal_models(name="test2")
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(df2.shape[1], 6)
        self.assertEqual(df2.columns[0], "real_date")
        self.assertEqual(df2.columns[1], "name")
        self.assertEqual(df2.columns[2], "model_type")
        self.assertEqual(df2.columns[3], "score")
        self.assertEqual(df2.columns[4], "hparams")
        self.assertEqual(df2.columns[5], "n_splits_used")
        self.assertEqual(df2.name.unique()[0], "test2")
        #     if not change_n_splits:
        #         self.assertTrue(
        #             np.all(df2.iloc[:, 4] == self.splitters[splitter_idx].n_splits)
        #         )
        #     else:
        #         num_new_dates = len(
        #             pd.bdate_range(df2.real_date.min(), df2.real_date.max())
        #         )
        #         self.assertTrue(np.min(df2.iloc[:, 4]) == initial_nsplits)
        #         self.assertTrue(
        #             np.max(df2.iloc[:, 4])
        #             == initial_nsplits + (num_new_dates // threshold_ndates)
        #         )

        df3 = so.get_optimal_models()
        self.assertIsInstance(df3, pd.DataFrame)
        self.assertEqual(df3.shape[1], 6)
        self.assertEqual(df3.columns[0], "real_date")
        self.assertEqual(df3.columns[1], "name")
        self.assertEqual(df3.columns[2], "model_type")
        self.assertEqual(df3.columns[3], "score")
        self.assertEqual(df3.columns[4], "hparams")
        self.assertEqual(df3.columns[5], "n_splits_used")
        self.assertEqual(len(df3.name.unique()), 2)
        #     if not change_n_splits:
        #         self.assertTrue(
        #             np.all(df3.iloc[:, 4] == self.splitters[splitter_idx].n_splits)
        #         )
        #     else:
        #         num_new_dates = len(
        #             pd.bdate_range(df3.real_date.min(), df3.real_date.max())
        #         )
        #         self.assertTrue(np.min(df3.iloc[:, 4]) == initial_nsplits)
        #         self.assertTrue(
        #             np.max(df3.iloc[:, 4])
        #             == initial_nsplits + (num_new_dates // threshold_ndates)
        #         )

        df4 = so.get_optimal_models(name=["test", "test2"])
        self.assertIsInstance(df4, pd.DataFrame)
        self.assertEqual(df4.shape[1], 6)
        self.assertEqual(df4.columns[0], "real_date")
        self.assertEqual(df4.columns[1], "name")
        self.assertEqual(df4.columns[2], "model_type")
        self.assertEqual(df4.columns[3], "score")
        self.assertEqual(df4.columns[4], "hparams")
        self.assertEqual(df4.columns[5], "n_splits_used")
        self.assertEqual(len(df4.name.unique()), 2)

    #     if not change_n_splits:
    #         self.assertTrue(
    #             np.all(df4.iloc[:, 4] == self.splitters[splitter_idx].n_splits)
    #         )
    #     else:
    #         num_new_dates = len(
    #             pd.bdate_range(df4.real_date.min(), df4.real_date.max())
    #         )
    #         self.assertTrue(np.min(df4.iloc[:, 4]) == initial_nsplits)
    #         self.assertTrue(
    #             np.max(df4.iloc[:, 4])
    #             == initial_nsplits + (num_new_dates // threshold_ndates)
    #         )

    def test_types_feature_selection_heatmap(self):
        so = self.so_with_calculated_preds

        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name=1)
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name=["test"])
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test2")
        # title
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", figsize="figsize")
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", figsize=1)
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(
                name="test", figsize=[1.5, 2]
            )  # needs to be a tuple!
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", figsize=(1.5, "e"))
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test", figsize=(0,))
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test", figsize=(0, 1, 2))
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test", figsize=(2, -1))

    def test_valid_feature_selection_heatmap(self):
        so = self.so_with_calculated_preds
        try:
            so.feature_selection_heatmap(name="test")
        except Exception as e:
            self.fail(f"feature_selection_heatmap raised an exception: {e}")

    def test_types_models_heatmap(self):
        so = self.so_with_calculated_preds

        with self.assertRaises(TypeError):
            so.models_heatmap(name=1)
        with self.assertRaises(TypeError):
            so.models_heatmap(name=["test"])
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test2")
        # title
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", title=1)
        # cap
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", cap="cap")
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", cap=1.3)
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", cap=-1)
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", cap=0)
        # figsize
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize="figsize")
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize=1)
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize=[1.5, 2])  # needs to be a tuple!
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize=(1.5, "e"))
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(0,))
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(0, 1, 2))
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(2, -1))

    def test_valid_models_heatmap(self):
        so = self.so_with_calculated_preds
        try:
            so.models_heatmap(name="test")
        except Exception as e:
            self.fail(f"models_heatmap raised an exception: {e}")
        # Repeat but for when cap > 20
        try:
            so.models_heatmap(name="test", cap=21)
        except Exception as e:
            self.fail(f"models_heatmap raised an exception: {e}")

    @parameterized.expand([["grid", None], ["prior", 1]])
    def test_valid_worker(self, search_type, n_iter):
        # Check that the worker private method works as expected for a grid search
        outer_splitter = ExpandingIncrementPanelSplit(
            train_intervals=1,
            test_size=1,
            min_cids=4,
            min_periods=36,
            max_periods=None,
        )

        so1 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        for idx, (train_idx, test_idx) in enumerate(
            outer_splitter.split(X=self.X, y=self.y)
        ):
            try:
                split_result = so1._worker(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    name="test",
                    models=self.models,
                    scorers=self.scorers,
                    hyperparameters=self.hyperparameters,
                    search_type=search_type,
                    n_iter=n_iter,
                    n_jobs_inner=1,
                    inner_splitters=self.single_inner_splitter,
                    normalize_fold_results=False,
                    # n_iter=None,
                    n_splits_add=None,
                    cv_summary="median",
                )

            except Exception as e:
                self.fail(f"_worker raised an exception: {e}")

            self.assertIsInstance(split_result, dict)
            self.assertTrue(
                split_result.keys()
                == {
                    "model_choice",
                    "ftr_coefficients",
                    "intercepts",
                    "selected_ftrs",
                    "predictions",
                }
            )

            model_choice_data = split_result["model_choice"]
            self.assertIsInstance(model_choice_data, list)
            self.assertIsInstance(model_choice_data[0], datetime.date)
            self.assertTrue(model_choice_data[1] == "test")
            self.assertIsInstance(model_choice_data[2], str)
            self.assertTrue(model_choice_data[2] in ["linreg", "ridge"])
            self.assertIsInstance(model_choice_data[3], float)
            self.assertIsInstance(model_choice_data[4], dict)
            self.assertIsInstance(model_choice_data[5], dict)

            prediction_data = split_result["predictions"]
            self.assertTrue(prediction_data[0] == "test")
            self.assertIsInstance(prediction_data[1], pd.MultiIndex)
            self.assertIsInstance(prediction_data[2], np.ndarray)

            ftr_data = split_result["ftr_coefficients"]
            self.assertIsInstance(ftr_data, list)
            self.assertTrue(len(ftr_data) == 2 + 3)  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_data[0], datetime.date)
            self.assertTrue(ftr_data[1] == "test")
            for i in range(2, len(ftr_data)):
                if ftr_data[i] != np.nan:
                    self.assertIsInstance(ftr_data[i], np.float32)

            intercept_data = split_result["intercepts"]
            self.assertIsInstance(intercept_data, list)
            self.assertTrue(
                len(intercept_data) == 2 + 1
            )  # 1 intercept + 2 extra columns
            self.assertIsInstance(intercept_data[0], datetime.date)
            self.assertTrue(intercept_data[1] == "test")
            if intercept_data[2] is not None:
                self.assertIsInstance(intercept_data[2], np.float32)

            ftr_selection_data = split_result["selected_ftrs"]
            self.assertIsInstance(ftr_selection_data, list)
            self.assertTrue(
                len(ftr_selection_data) == 2 + 3
            )  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_selection_data[0], datetime.date)
            self.assertTrue(ftr_selection_data[1] == "test")
            for i in range(2, len(ftr_selection_data)):
                self.assertTrue(ftr_selection_data[i] in [0, 1])

    def test_types_get_intercepts(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_intercepts(name="test2")
        # Test that the wrong dtype of a signal name raises an error
        with self.assertRaises(TypeError):
            so.get_intercepts(name=1)

    def test_valid_get_intercepts(self):
        so = self.so_with_calculated_preds
        # Test that running get_intercepts on pipeline "test" works
        try:
            intercepts = so.get_intercepts(name="test")
        except Exception as e:
            self.fail(f"get_intercepts raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(intercepts, pd.DataFrame)
        self.assertEqual(intercepts.shape[1], 3)
        self.assertEqual(intercepts.columns[0], "real_date")
        self.assertEqual(intercepts.columns[1], "name")
        self.assertEqual(intercepts.columns[2], "intercepts")
        self.assertTrue(intercepts.name.unique()[0] == "test")
        self.assertTrue(intercepts.isna().sum().sum() == 0)

    def test_types_get_selected_features(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_selected_features(name="test2")
        # Test that the wrong dtype of a signal name raises an error
        with self.assertRaises(TypeError):
            so.get_selected_features(name=1)

    def test_valid_get_selected_features(self):
        so = self.so_with_calculated_preds
        # Test that running get_selected_features on pipeline "test" works
        try:
            selected_ftrs = so.get_selected_features(name="test")
        except Exception as e:
            self.fail(f"get_selected_features raised an exception: {e}")
        # Test that the output is as expected
        # Test that the output is as expected
        self.assertIsInstance(selected_ftrs, pd.DataFrame)
        self.assertEqual(selected_ftrs.shape[1], 5)
        self.assertEqual(selected_ftrs.columns[0], "real_date")
        self.assertEqual(selected_ftrs.columns[1], "name")
        for i in range(2, 5):
            self.assertEqual(selected_ftrs.columns[i], self.X.columns[i - 2])
        self.assertTrue(selected_ftrs.name.unique()[0] == "test")
        self.assertTrue(selected_ftrs.isna().sum().sum() == 0)

    def test_types_get_ftr_coefficients(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_ftr_coefficients(name="test2")
        # Test that the wrong dtype of a signal name raises an error
        with self.assertRaises(TypeError):
            so.get_ftr_coefficients(name=1)

    def test_valid_get_ftr_coefficients(self):
        so = self.so_with_calculated_preds
        # Test that running get_ftr_coefficients on pipeline "test" works
        try:
            ftr_coefficients = so.get_ftr_coefficients(name="test")
        except Exception as e:
            self.fail(f"get_ftr_coefficients raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(ftr_coefficients, pd.DataFrame)
        self.assertEqual(ftr_coefficients.shape[1], 5)
        self.assertEqual(ftr_coefficients.columns[0], "real_date")
        self.assertEqual(ftr_coefficients.columns[1], "name")
        for i in range(2, 5):
            self.assertEqual(ftr_coefficients.columns[i], self.X.columns[i - 2])
        self.assertTrue(ftr_coefficients.name.unique()[0] == "test")
        self.assertTrue(ftr_coefficients.isna().sum().sum() == 0)

    def test_types_coefs_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.coefs_timeplot(name="test2")
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", figsize="figsize")
        with self.assertRaises(ValueError):
            so.coefs_timeplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", figsize=("hello", "hello"))
        # ftrs_renamed
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", ftrs_renamed=1)
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", ftrs_renamed={1: "ftr1"})
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", ftrs_renamed={"ftr1": 1})
        with self.assertRaises(ValueError):
            so.coefs_timeplot(name="test", ftrs_renamed={"ftr1": "ftr2"})

    def test_valid_coefs_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that running coefs_timeplot on pipeline "test" works
        try:
            so.coefs_timeplot(name="test")
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        # Check that the legend is correct
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(np.all(sorted(self.X.columns) == sorted(labels)))
        # Now rerun coefs_timeplot but with a feature renaming dictionary
        ftr_dict = {"CPI": "inflation"}
        try:
            so.coefs_timeplot(name="test", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            np.all(
                sorted(self.X.rename(columns=ftr_dict).columns) == sorted(labels)
            )
        )
        # Now rename two features
        ftr_dict = {"CPI": "inflation", "GROWTH": "growth"}
        try:
            so.coefs_timeplot(name="test", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            np.all(
                sorted(self.X.rename(columns=ftr_dict).columns) == sorted(labels)
            )
        )
        # Now rename all features
        ftr_dict = {ftr: f"ftr{i}" for i, ftr in enumerate(self.X.columns)}
        try:
            so.coefs_timeplot(name="test", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            np.all(
                sorted(self.X.rename(columns=ftr_dict).columns) == sorted(labels)
            )
        )
        # Finally, test that the title works
        title = ax.get_title()
        self.assertTrue(title == "Feature coefficients for pipeline: test")
        # Try changing the title
        try:
            so.coefs_timeplot(name="test", title="hello")
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "hello")

    def test_types_intercepts_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.intercepts_timeplot(name="test2")
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize="figsize")
        with self.assertRaises(ValueError):
            so.intercepts_timeplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize=("hello", "hello"))

    def test_valid_intercepts_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that running intercepts_timeplot on pipeline "test" works
        try:
            so.intercepts_timeplot(name="test")
        except Exception as e:
            self.fail(f"intercepts_timeplot raised an exception: {e}")

    def test_types_coefs_stackedbarplot(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test2")
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize="figsize")
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize=("hello", "hello"))
        # ftrs_renamed
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed=1)
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed={1: "ftr1"})
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed={"ftr1": 1})
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed={"ftr1": "ftr2"})

    def test_valid_coefs_stackedbarplot(self):
        so = self.so_with_calculated_preds
        # Test that running coefs_stackedbarplot on pipeline "test" works
        try:
            so.coefs_stackedbarplot(name="test")
        except Exception as e:
            self.fail(f"coefs_stackedbarplot raised an exception: {e}")
        # Check that the title is correct
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "Stacked bar plot of model coefficients: test")
        # Change the title
        try:
            so.coefs_stackedbarplot(name="test", title="hello")
        except Exception as e:
            self.fail(f"coefs_stackedbarplot raised an exception: {e}")
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "hello")
        # Now rerun coefs_stackedbarplot but with a feature renaming dictionary
        ftr_dict = {"CPI": "inflation"}
        try:
            so.coefs_stackedbarplot(name="test", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"coefs_stackedbarplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = sorted([text.get_text() for text in legend.get_texts()])
        # Check that the legend is correct
        ftrcoef_df = so.get_ftr_coefficients(name="test")
        ftrcoef_df["year"] = ftrcoef_df["real_date"].dt.year
        ftrcoef_df = ftrcoef_df.drop(columns=["real_date", "name"])
        ftrcoef_df = ftrcoef_df.rename(columns=ftr_dict)
        avg_coefs = ftrcoef_df.groupby("year").mean()
        pos_coefs = avg_coefs.clip(lower=0)
        neg_coefs = avg_coefs.clip(upper=0)
        correct_labels = [
            col for col in list(pos_coefs.sum().index[pos_coefs.sum() > 0])
        ]
        correct_labels += [
            col for col in list(neg_coefs.sum().index[neg_coefs.sum() < 0])
        ]
        correct_labels = sorted(correct_labels)
        self.assertTrue(np.all(labels == correct_labels))

    # def test_types_nsplits_timeplot(self):
    #     so = self.so_with_calculated_preds
    #     # Test that a wrong signal name raises an error
    #     with self.assertRaises(ValueError):
    #         so.nsplits_timeplot(name="test2")
    #     with self.assertRaises(TypeError):
    #         so.nsplits_timeplot(name=1)
    #     # title
    #     with self.assertRaises(TypeError):
    #         so.nsplits_timeplot(name="test", title=1)
    #     # figsize
    #     with self.assertRaises(TypeError):
    #         so.nsplits_timeplot(name="test", figsize="figsize")
    #     with self.assertRaises(ValueError):
    #         so.nsplits_timeplot(name="test", figsize=(0, 1, 2))
    #     with self.assertRaises(TypeError):
    #         so.nsplits_timeplot(name="test", figsize=(10, "hello"))
    #     with self.assertRaises(TypeError):
    #         so.nsplits_timeplot(name="test", figsize=("hello", 6))
    #     with self.assertRaises(TypeError):
    #         so.nsplits_timeplot(name="test", figsize=("hello", "hello"))

    # def test_valid_nsplits_timeplot(self):
    #     so = self.so_with_calculated_preds
    #     # Test that running nsplits_timeplot on pipeline "test" works
    #     try:
    #         so.nsplits_timeplot(name="test")
    #     except Exception as e:
    #         self.fail(f"nsplits_timeplot raised an exception: {e}")
    #     # Test that setting the title works
    #     ax = plt.gca()
    #     title = ax.get_title()
    #     self.assertTrue(title == "Number of CV splits for pipeline: test")
    #     # Try changing the title
    #     try:
    #         so.nsplits_timeplot(name="test", title="hello")
    #     except Exception as e:
    #         self.fail(f"nsplits_timeplot raised an exception: {e}")
    #     ax = plt.gca()
    #     title = ax.get_title()
    #     self.assertTrue(title == "hello")

    # def test_invalid_plots(self):
    #     splitter_idx = 1
    #     so = SignalOptimizer(
    #         inner_splitters=self.splitters[splitter_idx],
    #         X=self.X,
    #         y=self.y,
    #     )
    #     # Test that an error is raised if calculate_predictions has not been run
    #     with self.assertRaises(ValueError):
    #         so.nsplits_timeplot(name="test")
    #     # Test that an error is raised if calculate_predictions has not been run
    #     with self.assertRaises(ValueError):
    #         so.coefs_stackedbarplot(name="test")
    #     # Test that an error is raised if calculate_predictions has not been run
    #     with self.assertRaises(ValueError):
    #         so.intercepts_timeplot(name="test")
    #     # Test that an error is raised if calculate_predictions has not been run
    #     with self.assertRaises(ValueError):
    #         so.coefs_timeplot(name="test")
    #     # Test that if no signals have been calculated, an error is raised
    #     with self.assertRaises(ValueError):
    #         so.get_parameter_stats(name="test", include_intercept=False)
    #     with self.assertRaises(ValueError):
    #         so.get_parameter_stats(name="test", include_intercept=True)
    #     # Test that if no signals have been calculated, an error is raised
    #     with self.assertRaises(ValueError):
    #         so.get_ftr_coefficients(name="test")
    #     # Test that if no signals have been calculated, an error is raised
    #     with self.assertRaises(ValueError):
    #         so.get_intercepts(name="test")
    #     # Test that if no signals have been calculated, an error is raised
    #     with self.assertRaises(ValueError):
    #         so.get_selected_features(name="test")
    #     # Test that invalid names are caught
    #     with self.assertRaises(TypeError):
    #         so.feature_selection_heatmap(name=1)
    #     with self.assertRaises(TypeError):
    #         so.feature_selection_heatmap(name=[1, 2, 3])
    #     with self.assertRaises(ValueError):
    #         so.feature_selection_heatmap(name="test")
    #     # Test that invalid names are caught
    #     with self.assertRaises(TypeError):
    #         so.models_heatmap(name=1)
    #     with self.assertRaises(TypeError):
    #         so.models_heatmap(name=[1, 2, 3])
    #     with self.assertRaises(ValueError):
    #         so.models_heatmap(name="test")


def _get_X_y(so: SignalOptimizer):
    df_long = (
        categories_df(
            df=so.df,
            xcats=so.xcats,
            cids=so.cids,
            start=so.start,
            end=so.end,
            blacklist=so.blacklist,
            freq=so.freq,
            lag=so.lag,
            xcat_aggs=so.xcat_aggs,
        )
        .dropna()
        .sort_index()
    )
    X = df_long.iloc[:, :-1]
    y = df_long.iloc[:, -1]
    return X, y, df_long


if __name__ == "__main__":
    Test = TestAll()
    Test.setUpClass()
    Test.test_valid_coefs_stackedbarplot()
