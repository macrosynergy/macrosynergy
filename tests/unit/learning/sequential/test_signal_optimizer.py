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
    RandomEffects,
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
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                freq = 1,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                freq = "invalid",
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                lag = "invalid",
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                lag = -1,
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs = "invalid",
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs = [1, "sum"],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs = ["last", 1],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs = [1, 2],
            )

    def test_types_calculate_predictions(self):
        # Name
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name=1,
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name=True,
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )

        # Models
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=1,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models={},
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models={1: LinearRegression()},
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models={"LR": 1},
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )

        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models={"LR": RandomEffects()},
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )

        # Hyperparameters
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=1,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # wrong length of hyperparameter dict
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={"linreg": {"fit_intercept": [True, False]}},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # check hyperparameter dict is a nested dict
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={"linreg": 1, "ridge": 1},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "linreg": 1,
                    "ridge": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "ridge": 1,
                    "linreg": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "ridge": {1: [True, False]},
                    "linreg": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # Check inner hparam dictionaries specify a grid
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "ridge": {"fit_intercept": 1},
                    "linreg": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "linreg": {"fit_intercept": 1},
                    "ridge": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "ridge": {"fit_intercept": 1},
                    "linreg": {"fit_intercept": 1},
                },
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # Scorers should be a dict
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=1,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # Scorers dict shouldn't be empty
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers={},
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # Scorer keys should be strings
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers={1: make_scorer(r2_score)},
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers={"R2": make_scorer(r2_score), 1 : make_scorer(r2_score)},
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # Scorer values should be scorers
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers={"R2": 1},
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers={"R2": r2_score},
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # Inner splitters should be a dict
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=1,
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=ExpandingKFoldPanelSplit(),
            )
        # Inner splitters dict shouldn't be empty
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters={},
            )
        # Inner splitters keys should be strings
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters={1: ExpandingKFoldPanelSplit()},
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters={"ExpandingKFold": ExpandingKFoldPanelSplit(n_splits=5), 1: ExpandingKFoldPanelSplit(n_splits=10)},
            )
        # Inner splitters values should be splitters
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters={"ExpandingKFold": 1},
            )
        # Search type should be a string
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type=1,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # Search type should be either "grid" or "prior"
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="invalid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(NotImplementedError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="bayes",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # Normalize_fold_results should be a boolean
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                normalize_fold_results=1,
            )
        # If normalize_fold_results is True, then at least 2 hparams should
        # be provided for each model
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={"ridge": {"alpha": [0.1]}, "linreg": {"fit_intercept": [True, False]}},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                normalize_fold_results=True,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={"ridge": {"alpha": [0.1, 1, 10]}, "linreg": {"fit_intercept": [True]}},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                normalize_fold_results=True,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={"ridge": {"alpha": [0.1]}, "linreg": {"fit_intercept": [True]}},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                normalize_fold_results=True,
            )
        # Test structure of the hyperparameter grid when 
        # search_type is "prior"
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=1,
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={},
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # wrong length of hyperparameter dict
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={"LR": {}},
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # wrong model names in hyperparameter dict
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={"LR": {}, "L2": {}},
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # invalid hyperparameter dictionary
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={"linreg": {1: {}}},
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={"linreg": {1: {}}, "ridge": {1: {}}},
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # hyperparameter values are neither grids nor distributions
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "linreg": {"fit_intercept": [True, False]},
                    "ridge": {"alpha": 1},
                },
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "linreg": {"fit_intercept": 1},
                    "ridge": {"alpha": stats.expon()},
                },
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # cv_summary should be str or callable
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                cv_summary=1,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                cv_summary="invalid",
            )
        # min_cids should be an int
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                min_cids="invalid",
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                min_cids=7.2,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                min_cids=-2,
            )
        # min_periods should be an int
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                min_periods="invalid",
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                min_periods=7.2,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                min_periods=-2,
            )
        # test size should be a positive int
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                test_size="invalid",
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                test_size=7.2,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                test_size=-2,
            )
        # max periods should be a positive int, if not None
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                max_periods="invalid",
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                max_periods=7.2,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                max_periods=-2,
            )
        # n_iter should be a positive int, if not None
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                n_iter="invalid",
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                n_iter=None,
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                n_iter=3.4,
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                n_iter=3.4,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="prior",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                n_iter=-3,
            )
        # n_jobs should be an integer >= -1
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer="invalid",
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=-2,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_inner="invalid",
                n_jobs_outer=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_inner=-2,
                n_jobs_outer=1,
                inner_splitters=self.single_inner_splitter,
            )

        # split_functions should be a dict with keys matching inner_splitters
        with self.assertRaises(TypeError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                split_functions=1,
            )

        with self.assertRaises(ValueError):
            self.so_with_calculated_preds.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters=self.hyperparameters,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
                split_functions={"ExpandingKFold": 1},
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

        # Test that a random search works as expected
        so5 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids = self.cids,
            blacklist = self.black_valid,
        )
        try:
            so5.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "linreg": {
                        "fit_intercept": [True, False],
                    },
                    "ridge": {
                        "alpha": stats.expon(),
                    },
                },
                search_type="prior",
                n_iter=7,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")

        df5 = so5.preds.copy()
        self.assertIsInstance(df5, pd.DataFrame)
        if len(df5.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df5.xcat.unique()[0], "test")

        # Tests normalize_fold_results, cv_summary, split_functions, multiple inner splitters
        # TODO: debug and discover why this test is failing
        so6 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids = self.cids,
            blacklist = self.black_valid,
        )

        try:
            so6.calculate_predictions(
                name="test",
                models=self.models,
                scorers=self.scorers,
                hyperparameters={
                    "linreg": {
                        "fit_intercept": [True, False],
                        "positive": [True, False],
                    },
                    "ridge": {
                        "alpha": stats.expon(),
                    },
                },
                search_type="prior",
                n_iter=5,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.inner_splitters,
                normalize_fold_results=True,
                cv_summary="mean/std",
                split_functions = {
                    "ExpandingKFold": None,
                    "RollingKFold": lambda n: 2 * (n // 12),
                },
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")

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
            "hyperparameters": {
                "linreg": {"fit_intercept": [True, False], "positive": [True, False]},
                "ridge": {"alpha": [0.1, 1, 10], "fit_intercept": [True, False]},
            },
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
