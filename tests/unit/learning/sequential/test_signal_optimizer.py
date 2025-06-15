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
from sklearn.decomposition import PCA
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge)
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             make_scorer, r2_score)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning import (ExpandingIncrementPanelSplit,
                                   ExpandingKFoldPanelSplit, LassoSelector,
                                   RandomEffects, RollingKFoldPanelSplit,
                                   SignalOptimizer,
                                   regression_balanced_accuracy)
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils.df_utils import categories_df


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        
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
        self.pipelines = {
            "linreg": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=3)),
                    ("linreg", LinearRegression()),
                ]
            ),
        }
        self.hyperparameters = {
            "linreg": {},
            "ridge": {"alpha": [0.1, 1.0]},
        }
        self.pipeline_hyperparameters = {
            "linreg": {
                "pca__n_components": [3],
            },
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

        self.X, self.y, self.df_long = _get_X_y(so, drop_nas= True)

        # Create SignalOptimizer instances without NA drops
        self.so_no_na = SignalOptimizer(
            df = self.df,
            xcats = self.xcats,
            cids = self.cids,
            drop_nas= False,
        )

        self.so_no_na.calculate_predictions(
            name = "RF",
            models = {
                "RF": RandomForestRegressor(n_estimators = 10, max_depth = 1)
            },
            hyperparameters = {
                "RF": {
                    "min_samples_leaf": [36, 60]
                }
            },
            scorers = self.scorers,
            n_jobs_outer = 1,
            inner_splitters = self.inner_splitters,
            min_cids = 1,
            min_periods = 12
        )

        self.so_no_na.calculate_predictions(
            name = "RIDGE",
            models = {
                "RIDGE": Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("ridge", Ridge())
                ])
            },
            hyperparameters = {
                "RIDGE": {
                    "ridge__alpha": [1, 100, 10000]
                }
            },
            scorers = self.scorers,
            n_jobs_outer = 1,
            inner_splitters = self.inner_splitters,
            min_cids = 1,
            min_periods = 12
        )

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(self.mpl_backend)

    @parameterized.expand(
        itertools.product(
            [True, False], [True, False], [None, lambda x: -1 if x < 0 else 1], [True, False]
        )
    )
    def test_valid_init(self, use_blacklist, use_cids, generate_labels, drop_nas):
        try:
            blacklist = self.black_valid if use_blacklist else None
            cids = self.cids if use_cids else None
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                cids=cids,
                blacklist=blacklist,
                generate_labels=generate_labels,
                drop_nas=drop_nas,
            )
        except Exception as e:
            self.fail(f"Instantiation of the SignalOptimizer raised an exception: {e}")
        self.assertIsInstance(so, SignalOptimizer)
        X, y, df_long = _get_X_y(so, drop_nas=drop_nas)
        pd.testing.assert_frame_equal(so.X, X)
        if generate_labels:
            pd.testing.assert_series_equal(so.y, y.apply(generate_labels))
        else:
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
            ).astype(
                {
                    "real_date": "datetime64[ns]",
                    "name": "category",
                    "model_type": "category",
                    "score": "float32",
                    "hparams": "object",
                    "n_splits_used": "object",
                }
            ),
        )
        pd.testing.assert_frame_equal(
            so.preds,
            pd.DataFrame(columns=["real_date", "cid", "xcat", "value"]).astype(
                {
                    "cid": "category",
                    "real_date": "datetime64[ns]",
                    "xcat": "category",
                    "value": "float32",
                }
            ),
        )
        pd.testing.assert_frame_equal(
            so.feature_importances,
            pd.DataFrame(columns=["real_date", "name"] + list(so.X.columns)).astype(
                {
                    **{col: "float32" for col in self.X.columns},
                    "real_date": "datetime64[ns]",
                    "name": "category",
                }
            ),
        )
        pd.testing.assert_frame_equal(
            so.intercepts,
            pd.DataFrame(columns=["real_date", "name", "intercepts"]).astype(
                {
                    "real_date": "datetime64[ns]",
                    "name": "category",
                    "intercepts": "float32",
                }
            ),
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
                freq=1,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                freq="invalid",
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                lag="invalid",
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                lag=-1,
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs="invalid",
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs=[1, "sum"],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs=["last", 1],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs=[1, 2],
            )
        # generate_labels should be a callable or None
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                generate_labels=1,
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                generate_labels="invalid",
            )
        # drop_nas should be a boolean
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                drop_nas="sdf",
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

        # Models when NAs aren't dropped cannot admit models that don't support NAs
        with self.assertRaises(ValueError):
            self.so_no_na.calculate_predictions(
                name="test",
                models={"Lasso": Lasso()},
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
                scorers={"R2": make_scorer(r2_score), 1: make_scorer(r2_score)},
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
        # with self.assertRaises(ValueError):
        #     self.so_with_calculated_preds.calculate_predictions(
        #         name="test",
        #         models=self.models,
        #         scorers={"R2": r2_score},
        #         hyperparameters=self.hyperparameters,
        #         search_type="grid",
        #         n_jobs_outer=1,
        #         n_jobs_inner=1,
        #         inner_splitters=self.single_inner_splitter,
        #     )
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
                inner_splitters={
                    "ExpandingKFold": ExpandingKFoldPanelSplit(n_splits=5),
                    1: ExpandingKFoldPanelSplit(n_splits=10),
                },
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
                hyperparameters={
                    "ridge": {"alpha": [0.1]},
                    "linreg": {"fit_intercept": [True, False]},
                },
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
                hyperparameters={
                    "ridge": {"alpha": [0.1, 1, 10]},
                    "linreg": {"fit_intercept": [True]},
                },
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
                hyperparameters={
                    "ridge": {"alpha": [0.1]},
                    "linreg": {"fit_intercept": [True]},
                },
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
        # include_train_folds should be a boolean
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
                include_train_folds="hello",
            )
        # If cv_summary is "mean-std-ge", then include_train_folds should be True
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
                cv_summary="mean-std-ge",
                include_train_folds=False,
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
                store_correlations=1,
            )

        # If store_correlations is True, then models must be pipelines
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
                store_correlations=True,
            )

    def test_valid_calculate_predictions(self):
        # Test that the function runs without error and produces expected results when NAs aren't dropped
        outer_splitter = list(
            ExpandingIncrementPanelSplit(
                train_intervals=1,
                test_size=1,
                min_cids=1,
                min_periods=12,
                drop_nas=False
            ).split(self.so_no_na.X, self.so_no_na.y)
        )
        first_date = (
            self.so_no_na.X.iloc[outer_splitter[0][0], :].index.get_level_values(1).max()
        )
        last_date = (
            self.so_no_na.X.iloc[outer_splitter[-1][1], :].index.get_level_values(1).max()
        )
        so0 = self.so_no_na
        df0 = so0.preds.copy()
        self.assertIsInstance(df0, pd.DataFrame)
        if len(df0.xcat.unique()) != 2:
            self.fail("The signal dataframe should have two xcats")
        self.assertEqual(df0.xcat.unique()[0], "RF")
        self.assertEqual(df0.xcat.unique()[1], "RIDGE")
        self.assertTrue(len(df0) != 0)
        self.assertTrue(df0.value.notnull().all())
        self.assertTrue(len(df0.cid.unique()) == 4)
        self.assertTrue(df0.real_date.min() == first_date)
        self.assertTrue(df0.real_date.max() == last_date)
        # Test that the function runs without error and prediction dataframe is as expected
        outer_splitter = list(
            ExpandingIncrementPanelSplit(
                train_intervals=1,
                test_size=1,
                min_cids=4,
                min_periods=36,
            ).split(self.X, self.y)
        )
        first_date = (
            self.X.iloc[outer_splitter[0][0], :].index.get_level_values(1).max()
        )
        last_date = (
            self.X.iloc[outer_splitter[-1][1], :].index.get_level_values(1).max()
        )
        so1 = self.so_with_calculated_preds
        df1 = so1.preds.copy()
        self.assertIsInstance(df1, pd.DataFrame)
        if len(df1.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df1.xcat.unique()[0], "test")
        self.assertTrue(len(df1) != 0)
        self.assertTrue(df1.value.notnull().all())
        self.assertTrue(len(df1.cid.unique()) == 4)
        self.assertTrue(df1.real_date.min() == first_date)
        self.assertTrue(df1.real_date.max() == last_date)

        # Test that a different retraining frequency works as expected
        so2 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
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
                test_size=3,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df2 = so2.preds.copy()
        self.assertIsInstance(df2, pd.DataFrame)
        if len(df2.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df2.xcat.unique()[0], "test")
        self.assertTrue(len(df2) != 0)
        self.assertTrue(df2.value.notnull().all())
        self.assertTrue(len(df2.cid.unique()) == 4)
        self.assertTrue(df2.real_date.min() == first_date)
        self.assertTrue(df2.real_date.max() == last_date)

        # Test that blacklisting works as expected
        so3 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            blacklist=self.black_valid,
        )
        try:
            so3.calculate_predictions(
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

        df3 = so3.preds.copy()
        self.assertIsInstance(df3, pd.DataFrame)
        for cross_section, periods in self.black_valid.items():
            cross_section_key = cross_section.split("_")[0]
            self.assertTrue(
                len(
                    df3[
                        (df3.cid == cross_section_key)
                        & (df3.real_date >= periods[0])
                        & (df3.real_date <= periods[1])
                    ].dropna()
                )
                == 0
            )
        self.assertTrue(len(df3.cid.unique()) == 4)
        self.assertTrue(df3.real_date.min() == first_date)
        self.assertTrue(df3.real_date.max() == last_date)

        # Test that rolling models work as expected
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
                search_type="prior",
                n_iter=1,
                max_periods=21,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df4 = so4.preds.copy()
        self.assertIsInstance(df4, pd.DataFrame)
        if len(df4.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df4.xcat.unique()[0], "test")
        self.assertTrue(len(df4.cid.unique()) == 4)
        self.assertTrue(df4.real_date.min() == first_date)
        self.assertTrue(df4.real_date.max() == last_date)

        # Test that an unreasonably large roll is equivalent to no roll
        so5 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        try:
            so5.calculate_predictions(
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
        df5 = so5.preds.copy()
        self.assertIsInstance(df5, pd.DataFrame)
        if len(df5.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df5.xcat.unique()[0], "test")
        self.assertTrue(df1.equals(df5))
        self.assertTrue(len(df5.cid.unique()) == 4)
        self.assertTrue(df5.real_date.min() == first_date)
        self.assertTrue(df5.real_date.max() == last_date)

        # Test that a random search works as expected
        so6 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
            blacklist=self.black_valid,
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
                n_iter=4,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")

        df6 = so6.preds.copy()
        self.assertIsInstance(df6, pd.DataFrame)
        if len(df6.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df6.xcat.unique()[0], "test")
        self.assertTrue(len(df6.cid.unique()) == 4)
        self.assertTrue(df6.real_date.min() == first_date)
        self.assertTrue(df6.real_date.max() == last_date)

        # Tests normalize_fold_results, cv_summary, split_functions, multiple inner splitters
        so7 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
            blacklist=self.black_valid,
        )

        try:
            so7.calculate_predictions(
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
                n_iter=4,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.inner_splitters,
                normalize_fold_results=True,
                cv_summary="mean/std",
                split_functions={
                    "ExpandingKFold": None,
                    "RollingKFold": lambda n: (
                        n // 24
                    ),  # Increases by one every two years
                },
                test_size=3,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df7 = so7.preds.copy()
        self.assertIsInstance(df7, pd.DataFrame)
        if len(df7.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df7.xcat.unique()[0], "test")
        self.assertTrue(len(df7.cid.unique()) == 4)
        self.assertTrue(df7.real_date.min() == first_date)
        self.assertTrue(df7.real_date.max() == last_date)

        # Test that include_train_folds works as expected. When cv_summary = "mean-std-ge"
        # an error should be thrown if include_train_folds is False and when cv_summary is
        # something else, a warning should be thrown
        so8 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
            blacklist=self.black_valid,
        )
        with self.assertRaises(ValueError):
            so8.calculate_predictions(
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
                n_iter=4,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.inner_splitters,
                normalize_fold_results=True,
                cv_summary="mean-std-ge",
                include_train_folds=False,
            )

        with self.assertWarns(UserWarning):
            so8.calculate_predictions(
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
                n_iter=4,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.inner_splitters,
                normalize_fold_results=True,
                cv_summary="mean",
                include_train_folds=True,
            )
        try:
            so8.calculate_predictions(
                name="test2",
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
                n_iter=4,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.inner_splitters,
                cv_summary="mean-std-ge",
                include_train_folds=True,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")

        df8 = so8.preds.copy()
        self.assertIsInstance(df8, pd.DataFrame)
        if len(df8.xcat.unique()) != 2:
            self.fail("The signal dataframe should only contain two xcats")
        self.assertEqual(df8.xcat.unique()[0], "test")
        self.assertEqual(df8.xcat.unique()[1], "test2")
        self.assertTrue(len(df7.cid.unique()) == 4)
        self.assertTrue(df8.real_date.min() == first_date)
        self.assertTrue(df8.real_date.max() == last_date)
        # Test generate_labels works with a logistic regression
        so8 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
            blacklist=self.black_valid,
            generate_labels=lambda x: -1 if x < 0 else 1,
        )

        try:
            so8.calculate_predictions(
                name="test",
                models={
                    "LogReg": LogisticRegression(),
                },
                scorers={"ACC": make_scorer(accuracy_score), "BAC": make_scorer(balanced_accuracy_score)},
                hyperparameters={
                    "LogReg": {
                        "fit_intercept": [True, False],
                        "C": stats.expon(),
                    },
                },
                search_type="prior",
                n_iter=4,
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.inner_splitters,
                normalize_fold_results=True,
                cv_summary="mean-std",
                split_functions={
                    "ExpandingKFold": None,
                    "RollingKFold": lambda n: (
                        n // 24
                    ),  # Increases by one every two years
                },
                test_size=3,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df8 = so8.preds.copy()
        self.assertIsInstance(df8, pd.DataFrame)
        if len(df8.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df8.xcat.unique()[0], "test")
        self.assertTrue(len(df8.cid.unique()) == 4)
        self.assertTrue(df8.real_date.min() == first_date)
        self.assertTrue(df8.real_date.max() == last_date)
        self.assertTrue(len(df8.value.value_counts()) == 2)

    @parameterized.expand([True, False])
    def test_optional_hparam_validity(self, drop_nas: bool):
        """
        I test that the pipelines run as expected when no hyperparameters are 
        entered. 
        """
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
            drop_nas=drop_nas,
        )
        if drop_nas:
            so.calculate_predictions(
                name="test",
                models={"LR": LinearRegression()},
                min_cids = 1, 
                min_periods = 12,
            )
        else:
            so.calculate_predictions(
                name="test",
                models={"LR": Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ])},
                min_cids = 1, 
                min_periods = 12
            )
        dfa = so.get_optimized_signals("test")
        self.assertIsInstance(dfa, pd.DataFrame)
        if len(dfa.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(dfa.xcat.unique()[0], "test")
        self.assertTrue(len(dfa) != 0)
        self.assertTrue(dfa.value.notnull().all())

        # Check the first and last date is as expected
        outer_splitter = list(
            ExpandingIncrementPanelSplit(
                train_intervals=1,
                test_size=1,
                min_cids=1,
                min_periods=12,
                drop_nas=drop_nas,
            ).split(so.X, so.y)
        )
        first_date = (
            so.X.iloc[outer_splitter[0][0], :].index.get_level_values(1).max()
        )
        last_date = (
            so.X.iloc[outer_splitter[-1][1], :].index.get_level_values(1).max()
        )
        self.assertTrue(dfa.real_date.min() == first_date)
        self.assertTrue(dfa.real_date.max() == last_date)

        # (2) Check that the model diagnostics are stored as expected in this instance
        dfa = so.get_optimal_models("test")
        self.assertIsInstance(dfa, pd.DataFrame)
        self.assertTrue(len(dfa) != 0)
        self.assertTrue(dfa.name.notnull().all())
        self.assertTrue(dfa.model_type.notnull().all())
        self.assertTrue(dfa.model_type.unique()[0] == "LR")
        self.assertTrue(dfa.name.unique()[0] == "test")
        self.assertTrue(all(dfa.score==0))
        self.assertTrue(all(dfa.hparams=={}))
        self.assertTrue(all(dfa.n_splits_used==0))

    @parameterized.expand([True, False])
    def test_types_run(self, drop_nas: bool):
        # Training set only
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            drop_nas=drop_nas
        )
        outer_splitter = ExpandingIncrementPanelSplit(drop_nas=drop_nas)

        # Valid parameters
        valid_params = {
            "name": "test",
            "outer_splitter": outer_splitter,
            "inner_splitters": self.single_inner_splitter,
            "models": {
                "linreg": Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]),
                "ridge": Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]),
            },
            "hyperparameters": {
                "linreg": {"model__fit_intercept": [True, False], "model__positive": [True, False]},
                "ridge": {"model__alpha": [0.1, 1, 10], "model__fit_intercept": [True, False]},
            },
            "scorers": self.scorers,
            "normalize_fold_results": True,
            "search_type": "prior",
            "cv_summary": "median",
            "include_train_folds": False,
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
            "scorers": "scorers",  # Invalid: should be a scorer or list of scorers
            "normalize_fold_results": "invalid_boolean",  # Invalid: should be a boolean
            "search_type": 123,  # Invalid: should be a string
            "cv_summary": 123,  # Invalid: should be a string
            "include_train_folds": "invalid_boolean",  # Invalid: should be a boolean
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

    @parameterized.expand([["grid", None, True], ["prior", 1, False]])
    def test_valid_worker(self, search_type, n_iter, drop_nas):
        search_type = "grid"
        n_iter = None
        store_correlations = False
        # Check that the worker private method works as expected for a grid search
        outer_splitter = ExpandingIncrementPanelSplit(
            train_intervals=1,
            test_size=1,
            min_cids=1,
            min_periods=12,
            max_periods=None,
            drop_nas=drop_nas,
        )

        so1 = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            drop_nas=drop_nas,
        )
        so1.store_correlations = store_correlations
        for idx, (train_idx, test_idx) in enumerate(
            outer_splitter.split(X=so1.X, y=so1.y)
        ):
            try:
                split_result = so1._worker(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    name="test",
                    models={
                        "linreg": Pipeline([
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                            ("model", LinearRegression()),
                        ]),
                        "ridge": Pipeline([
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("scaler", StandardScaler()),
                            ("model", Ridge()),
                        ]),
                    },
                    scorers=self.scorers,
                    hyperparameters={
                        "linreg": {"model__fit_intercept": [True, False], "model__positive": [True, False]},
                        "ridge": {"model__alpha": [0.1, 1, 10], "model__fit_intercept": [True, False]},
                    },
                    search_type=search_type,
                    n_iter=n_iter,
                    n_jobs_inner=1,
                    inner_splitters=self.single_inner_splitter,
                    normalize_fold_results=False,
                    n_splits_add=None,
                    cv_summary="median",
                    include_train_folds=False,
                    base_splits=None,
                )

            except Exception as e:
                self.fail(f"_worker raised an exception: {e}")

            self.assertIsInstance(split_result, dict)
            self.assertTrue(
                split_result.keys()
                == {
                    "model_choice",
                    "feature_importances",
                    "intercepts",
                    "selected_ftrs",
                    "predictions",
                    "ftr_corr",
                }
            )

            model_choice_data = split_result["model_choice"]
            self.assertIsInstance(model_choice_data, list)
            self.assertIsInstance(model_choice_data[0], datetime.date)
            self.assertIsInstance(model_choice_data[1], str)
            self.assertTrue(model_choice_data[1] in ["linreg", "ridge"])
            self.assertIsInstance(model_choice_data[2], float)
            self.assertIsInstance(model_choice_data[3], dict)
            self.assertIsInstance(model_choice_data[4], dict)

            prediction_data = split_result["predictions"]
            self.assertIsInstance(prediction_data[0], pd.MultiIndex)
            self.assertIsInstance(prediction_data[1], np.ndarray)

            ftr_data = split_result["feature_importances"]
            self.assertIsInstance(ftr_data, list)
            self.assertTrue(len(ftr_data) == 1 + 3)  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_data[0], datetime.date)
            for i in range(1, len(ftr_data)):
                if ftr_data[i] != np.nan:
                    self.assertIsInstance(ftr_data[i], (np.float64, np.float32, float))  # float or int

            intercept_data = split_result["intercepts"]
            self.assertIsInstance(intercept_data, list)
            self.assertTrue(
                len(intercept_data) == 2
            )  # 1 intercept + 2 extra columns
            self.assertIsInstance(intercept_data[0], datetime.date)
            if intercept_data[1] is not None:
                self.assertIsInstance(intercept_data[1], (np.float64, np.float32, float))

            ftr_selection_data = split_result["selected_ftrs"]
            self.assertIsInstance(ftr_selection_data, list)
            self.assertTrue(
                len(ftr_selection_data) == 1 + 3
            )  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_selection_data[0], datetime.date)
            for i in range(1, len(ftr_selection_data)):
                self.assertTrue(ftr_selection_data[i] in [0, 1])

    def test_valid_store_correlation(self):
        search_type = "grid"
        n_iter = None
        store_correlations = True
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
        so1.store_correlations = store_correlations
        for idx, (train_idx, test_idx) in enumerate(
            outer_splitter.split(X=self.X, y=self.y)
        ):
            try:
                split_result = so1._worker(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    name="test",
                    models=self.pipelines,
                    scorers=self.scorers,
                    hyperparameters=self.pipeline_hyperparameters,
                    search_type=search_type,
                    n_iter=n_iter,
                    n_jobs_inner=1,
                    inner_splitters=self.single_inner_splitter,
                    normalize_fold_results=False,
                    n_splits_add=None,
                    cv_summary="median",
                    include_train_folds=False,
                    base_splits=None,
                )

            except Exception as e:
                self.fail(f"_worker raised an exception: {e}")

            self.assertIsInstance(split_result, dict)
            self.assertTrue(
                split_result.keys()
                == {
                    "model_choice",
                    "feature_importances",
                    "intercepts",
                    "selected_ftrs",
                    "predictions",
                    "ftr_corr",
                }
            )

            model_choice_data = split_result["model_choice"]
            self.assertIsInstance(model_choice_data, list)
            self.assertIsInstance(model_choice_data[0], datetime.date)
            self.assertIsInstance(model_choice_data[1], str)
            self.assertTrue(model_choice_data[1] in ["linreg"])
            self.assertIsInstance(model_choice_data[2], float)
            self.assertIsInstance(model_choice_data[3], dict)
            self.assertIsInstance(model_choice_data[4], dict)

            prediction_data = split_result["predictions"]
            self.assertIsInstance(prediction_data[0], pd.MultiIndex)
            self.assertIsInstance(prediction_data[1], np.ndarray)

            ftr_data = split_result["feature_importances"]
            self.assertIsInstance(ftr_data, list)
            self.assertTrue(len(ftr_data) == 1 + 3)  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_data[0], datetime.date)
            for i in range(1, len(ftr_data)):
                if ftr_data[i] != np.nan:
                    self.assertIsInstance(ftr_data[i], np.float32)

            intercept_data = split_result["intercepts"]
            self.assertIsInstance(intercept_data, list)
            self.assertTrue(
                len(intercept_data) == 2
            )  # 1 intercept + 2 extra columns
            self.assertIsInstance(intercept_data[0], datetime.date)
            if intercept_data[1] is not None:
                self.assertIsInstance(intercept_data[1], np.float32)

            ftr_selection_data = split_result["selected_ftrs"]
            self.assertIsInstance(ftr_selection_data, list)
            self.assertTrue(
                len(ftr_selection_data) == 1 + 3
            )  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_selection_data[0], datetime.date)
            for i in range(1, len(ftr_selection_data)):
                self.assertTrue(ftr_selection_data[i] in [0, 1])

            ftr_correlation = split_result["ftr_corr"]
            self.assertIsInstance(ftr_correlation, list)
            self.assertTrue(len(ftr_correlation[0]) == 5)
            self.assertIsInstance(ftr_correlation[0][0], datetime.date)
            self.assertTrue(ftr_correlation[0][1] == "test")
            self.assertIsInstance(ftr_correlation[0][2], str)
            self.assertIsInstance(ftr_correlation[0][3], str)

    def test_optional_hparam_types(self):
        """
        Hyperparameter tuning is optional. This test checks that various Type/Value errors
        are called should optional tuning be abused.
        """
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
        )
        # Raise ValueError if scorers is None but inner_splitters is not None
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models={"LR":LinearRegression()},
                scorers=None,
                hyperparameters={"LR": {"fit_intercept": [True, False]}},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models={"LR":LinearRegression()},
                scorers=None,
                hyperparameters=None,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=self.single_inner_splitter,
            )
        # Raise ValueError if inner_splitters is None but scorers is not None
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models={"LR":LinearRegression()},
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={"LR": {"fit_intercept": [True, False]}},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=None,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models={"LR":LinearRegression()},
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=None,
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
                inner_splitters=None,
            )
        # Raise ValueError if no cross-validation is performed but multiple models specified
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models={"LR":LinearRegression(), "RIDGE":Ridge()},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
            )
        # Raise ValueError if no cross-validation is performed but hyperparameters specified
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models={"LR":LinearRegression()},
                hyperparameters={"LR": {"fit_intercept": [True, False]}},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
            )
        # Raise ValueError if no cross-validation is performed but
        # hyperparameters specified and multiple models specified
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models={"LR":LinearRegression(), "Ridge":Ridge()},
                hyperparameters={"LR": {"fit_intercept": [True, False]}, "Ridge": {"fit_intercept": [True, False]}},
                search_type="grid",
                n_jobs_outer=1,
                n_jobs_inner=1,
            )

    def test_get_ftr_corr_data_no_model(self):

        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        so.store_correlations = True
        ftr_corr_data = so._get_ftr_corr_data(
            pipeline_name="test",
            optimal_model=None,
            X_train=self.X,
            timestamp=datetime.date(2021, 1, 1),
        )

        self.assertIsInstance(ftr_corr_data, list)
        self.assertTrue(len(ftr_corr_data) == 3)

        for lst in ftr_corr_data[1:]:
            self.assertIsInstance(lst, list)
            self.assertTrue(len(lst) == 5)
            self.assertIsInstance(lst[0], datetime.date)
            self.assertEqual(lst[1], "test")
            self.assertEqual(lst[2], lst[3])
            self.assertEqual(lst[4], 1)

    def test_get_ftr_corr_data_no_corr(self):

        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        ftr_corr_data = so._get_ftr_corr_data(
            pipeline_name="test",
            optimal_model=None,
            X_train=self.X,
            timestamp=datetime.date(2021, 1, 1),
        )

        self.assertIsInstance(ftr_corr_data, list)
        self.assertTrue(len(ftr_corr_data) == 0)

    def test_get_feature_correlations(self):
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        so.store_correlations = True
        so.calculate_predictions(
            name="test",
            models=self.pipelines,
            scorers=self.scorers,
            hyperparameters=self.pipeline_hyperparameters,
            search_type="grid",
            n_jobs_outer=1,
            n_jobs_inner=1,
            inner_splitters=self.single_inner_splitter,
            store_correlations=True,
        )
        kwargs_list = [
            {"name": "test"},
            {"name": None},
            {"name": ["test"]},
        ]
        for kwargs in kwargs_list:
            ftr_corr_df = so.get_feature_correlations(**kwargs)

            self.assertIsInstance(ftr_corr_df, pd.DataFrame)
            self.assertTrue(ftr_corr_df.shape[1] == 5)
            self.assertTrue(ftr_corr_df.columns[0] == "real_date")
            self.assertTrue(ftr_corr_df.columns[1] == "name")
            self.assertTrue(ftr_corr_df.columns[2] == "predictor_input")
            self.assertTrue(ftr_corr_df.columns[3] == "pipeline_input")
            self.assertTrue(ftr_corr_df.columns[4] == "pearson")

        with self.assertRaises(TypeError):
            so.get_feature_correlations(name=1)

        with self.assertRaises(ValueError):
            so.get_feature_correlations(name=["invalid"])

    def test_types_get_optimized_signals(self):
        so = self.so_with_calculated_preds
        so_nas = self.so_no_na
        # Test invalid names are caught
        with self.assertRaises(TypeError):
            so.get_optimized_signals(name=1)
        with self.assertRaises(TypeError):
            so.get_optimized_signals(name={})
        with self.assertRaises(TypeError):
            so_nas.get_optimized_signals(name=1)
        with self.assertRaises(TypeError):
            so_nas.get_optimized_signals(name={})

        # Test an error is raised if a wrong name is passed
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name=["test", "test2"])
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name="test2")
        with self.assertRaises(ValueError):
            so_nas.get_optimized_signals(name=["test", "test2"])
        with self.assertRaises(ValueError):
            so_nas.get_optimized_signals(name="test2")

        # Test that if no signals have been calculated, an error is raised
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name="test2")

        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            drop_nas = False,
        )
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name="test2")

    def test_valid_get_optimized_signals(self):
        # Test that the output is a dataframe
        so = self.so_with_calculated_preds
        so2 = self.so_no_na

        df1 = so.get_optimized_signals(name="test")
        df2 = so2.get_optimized_signals()
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(df1.shape[1], 4)
        self.assertEqual(df2.shape[1], 4)
        self.assertEqual(df1.columns[0], "real_date")
        self.assertEqual(df2.columns[0], "real_date")
        self.assertEqual(df1.columns[1], "cid")
        self.assertEqual(df2.columns[1], "cid")
        self.assertEqual(df1.columns[2], "xcat")
        self.assertEqual(df2.columns[2], "xcat")
        self.assertEqual(df1.columns[3], "value")
        self.assertEqual(df2.columns[3], "value")
        self.assertEqual(df1.xcat.unique()[0], "test")
        self.assertEqual(df2.xcat.unique()[0], "RF")
        self.assertEqual(df2.xcat.unique()[1], "RIDGE")

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
        self.assertEqual(df2.columns[0], "real_date")
        self.assertEqual(df2.columns[1], "cid")
        self.assertEqual(df2.columns[2], "xcat")
        self.assertEqual(df2.columns[3], "value")
        self.assertEqual(df2.xcat.unique()[0], "test2")

        df3 = so.get_optimized_signals()
        self.assertIsInstance(df3, pd.DataFrame)
        self.assertEqual(df3.shape[1], 4)
        self.assertEqual(df3.columns[0], "real_date")
        self.assertEqual(df3.columns[1], "cid")
        self.assertEqual(df3.columns[2], "xcat")
        self.assertEqual(df3.columns[3], "value")
        self.assertEqual(len(df3.xcat.unique()), 2)

        df4 = so.get_optimized_signals(name=["test", "test2"])
        self.assertIsInstance(df4, pd.DataFrame)
        self.assertEqual(df4.shape[1], 4)
        self.assertEqual(df4.columns[0], "real_date")
        self.assertEqual(df4.columns[1], "cid")
        self.assertEqual(df4.columns[2], "xcat")
        self.assertEqual(df4.columns[3], "value")
        self.assertEqual(len(df4.xcat.unique()), 2)

        df5 = so2.get_optimized_signals(name=["RIDGE"])
        self.assertIsInstance(df5, pd.DataFrame)
        self.assertEqual(df5.shape[1], 4)
        self.assertEqual(df5.columns[0], "real_date")
        self.assertEqual(df5.columns[1], "cid")
        self.assertEqual(df5.columns[2], "xcat")
        self.assertEqual(df5.columns[3], "value")
        self.assertEqual(len(df5.xcat.unique()), 1)

    def test_types_get_selected_features(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_selected_features(name="test2")
        with self.assertRaises(ValueError):
            so.get_selected_features(name=["test", "test2"])

        # Test that the wrong dtype of a signal name raises an error
        with self.assertRaises(TypeError):
            so.get_selected_features(name=1)
        with self.assertRaises(TypeError):
            so.get_selected_features(name={})

    def test_valid_get_selected_features(self):
        so = self.so_with_calculated_preds
        # Test that running get_selected_features on pipeline "test" works
        try:
            selected_ftrs = so.get_selected_features(name="test")
        except Exception as e:
            self.fail(f"get_selected_features raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(selected_ftrs, pd.DataFrame)
        self.assertEqual(selected_ftrs.shape[1], 5)
        self.assertEqual(selected_ftrs.columns[0], "real_date")
        self.assertEqual(selected_ftrs.columns[1], "name")
        for i in range(2, 5):
            self.assertEqual(selected_ftrs.columns[i], self.X.columns[i - 2])
        self.assertTrue(selected_ftrs.name.unique()[0] == "test")
        self.assertTrue(selected_ftrs.isna().sum().sum() == 0)

        # Test that running get_selected_features without a name works
        try:
            selected_ftrs = so.get_selected_features()
        except Exception as e:
            self.fail(f"get_selected_features raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(selected_ftrs, pd.DataFrame)
        self.assertEqual(selected_ftrs.shape[1], 5)
        self.assertEqual(selected_ftrs.columns[0], "real_date")
        self.assertEqual(selected_ftrs.columns[1], "name")
        for i in range(2, 5):
            self.assertEqual(selected_ftrs.columns[i], self.X.columns[i - 2])
        self.assertTrue(selected_ftrs.name.unique()[0] == "test")
        self.assertTrue(selected_ftrs.isna().sum().sum() == 0)

    def test_types_get_feature_importances(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_feature_importances(name="test2")
            so2.get_feature_importances(name="test2")
        with self.assertRaises(ValueError):
            so.get_feature_importances(name=["test", "test2"])
            so2.get_feature_importances(name=["test", "test2"])
        # Test that the wrong dtype of a signal name raises an error
        with self.assertRaises(TypeError):
            so.get_feature_importances(name=1)
            so2.get_feature_importances(name=1)
        with self.assertRaises(TypeError):
            so.get_feature_importances(name={})
            so2.get_feature_importances(name={})

    def test_valid_get_feature_importances(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that running get_feature_importances on pipeline "test" works
        try:
            feature_importances = so.get_feature_importances(name="test")
            feature_importances2 = so2.get_feature_importances(name="RF")
            feature_importances3 = so2.get_feature_importances(name="RIDGE")
        except Exception as e:
            self.fail(f"get_feature_importances raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(feature_importances, pd.DataFrame)
        self.assertIsInstance(feature_importances2, pd.DataFrame)
        self.assertIsInstance(feature_importances3, pd.DataFrame)
        self.assertEqual(feature_importances.shape[1], 5)
        self.assertEqual(feature_importances2.shape[1], 5)
        self.assertEqual(feature_importances3.shape[1], 5)
        self.assertEqual(feature_importances.columns[0], "real_date")
        self.assertEqual(feature_importances2.columns[0], "real_date")
        self.assertEqual(feature_importances3.columns[0], "real_date")
        self.assertEqual(feature_importances.columns[1], "name")
        self.assertEqual(feature_importances2.columns[1], "name")
        self.assertEqual(feature_importances3.columns[1], "name")
        for i in range(2, 5):
            self.assertEqual(feature_importances.columns[i], self.X.columns[i - 2])
            self.assertEqual(feature_importances2.columns[i], self.X.columns[i - 2])
            self.assertEqual(feature_importances3.columns[i], self.X.columns[i - 2])
        self.assertTrue(feature_importances.name.unique()[0] == "test")
        self.assertTrue(feature_importances2.name.unique()[0] == "RF")
        self.assertTrue(feature_importances3.name.unique()[0] == "RIDGE")
        self.assertTrue(feature_importances.isna().sum().sum() == 0)
        self.assertTrue(feature_importances2.isna().sum().sum() == 0)
        self.assertTrue(feature_importances3.isna().sum().sum() == 0)

        # Test that running get_feature_importances without a name works
        try:
            feature_importances = so.get_feature_importances()
            feature_importances2 = so2.get_feature_importances()
        except Exception as e:
            self.fail(f"get_selected_features raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(feature_importances, pd.DataFrame)
        self.assertIsInstance(feature_importances2, pd.DataFrame)
        self.assertEqual(feature_importances.shape[1], 5)
        self.assertEqual(feature_importances2.shape[1], 5)
        self.assertEqual(feature_importances.columns[0], "real_date")
        self.assertEqual(feature_importances2.columns[0], "real_date")
        self.assertEqual(feature_importances.columns[1], "name")
        self.assertEqual(feature_importances2.columns[1], "name")
        for i in range(2, 5):
            self.assertEqual(feature_importances.columns[i], self.X.columns[i - 2])
            self.assertEqual(feature_importances2.columns[i], self.X.columns[i - 2])
        self.assertTrue(feature_importances.name.unique()[0] == "test")
        self.assertTrue(feature_importances2.name.unique()[0] == "RF")
        self.assertTrue(feature_importances.isna().sum().sum() == 0)
        self.assertTrue(feature_importances2.isna().sum().sum() == 0)

    def test_types_get_intercepts(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_intercepts(name="test2")
            so2.get_intercepts(name="test2")
        with self.assertRaises(ValueError):
            so.get_intercepts(name=["test", "test2"])
            so2.get_intercepts(name=["test", "test2"])
        # Test that the wrong dtype of a signal name raises an error
        with self.assertRaises(TypeError):
            so.get_intercepts(name=1)
            so2.get_intercepts(name=1)
        with self.assertRaises(TypeError):
            so.get_intercepts(name={})
            so2.get_intercepts(name={})

    def test_valid_get_intercepts(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that running get_intercepts on pipeline "test" works
        try:
            intercepts = so.get_intercepts(name="test")
            intercepts2 = so2.get_intercepts(name="RF")
            intercepts3 = so2.get_intercepts(name="RIDGE")
        except Exception as e:
            self.fail(f"get_intercepts raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(intercepts, pd.DataFrame)
        self.assertIsInstance(intercepts2, pd.DataFrame)
        self.assertIsInstance(intercepts3, pd.DataFrame)
        self.assertEqual(intercepts.shape[1], 3)
        self.assertEqual(intercepts2.shape[1], 3)
        self.assertEqual(intercepts3.shape[1], 3)
        self.assertEqual(intercepts.columns[0], "real_date")
        self.assertEqual(intercepts2.columns[0], "real_date")
        self.assertEqual(intercepts3.columns[0], "real_date")
        self.assertEqual(intercepts.columns[1], "name")
        self.assertEqual(intercepts2.columns[1], "name")
        self.assertEqual(intercepts3.columns[1], "name")
        self.assertEqual(intercepts.columns[2], "intercepts")
        self.assertEqual(intercepts2.columns[2], "intercepts")
        self.assertEqual(intercepts3.columns[2], "intercepts")
        self.assertTrue(intercepts.name.unique()[0] == "test")
        self.assertTrue(intercepts2.name.unique()[0] == "RF")
        self.assertTrue(intercepts3.name.unique()[0] == "RIDGE")
        self.assertTrue(intercepts.isna().sum().sum() == 0)
        self.assertTrue(intercepts2.isna().sum().sum() == len(intercepts2)) # RF has no intercepts
        self.assertTrue(intercepts3.isna().sum().sum() == 0)

        # Test that running get_intercepts without a name works
        try:
            intercepts = so.get_intercepts()
            intercepts2 = so2.get_intercepts()
        except Exception as e:
            self.fail(f"get_intercepts raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(intercepts, pd.DataFrame)
        self.assertIsInstance(intercepts2, pd.DataFrame)
        self.assertEqual(intercepts.shape[1], 3)
        self.assertEqual(intercepts2.shape[1], 3)
        self.assertEqual(intercepts.columns[0], "real_date")
        self.assertEqual(intercepts2.columns[0], "real_date")
        self.assertEqual(intercepts.columns[1], "name")
        self.assertEqual(intercepts2.columns[1], "name")
        self.assertEqual(intercepts.columns[2], "intercepts")
        self.assertEqual(intercepts2.columns[2], "intercepts")
        self.assertTrue(intercepts.name.unique()[0] == "test")
        self.assertTrue(intercepts2.name.unique()[0] == "RF")
        self.assertTrue(intercepts.isna().sum().sum() == 0)
        self.assertTrue(intercepts2.isna().sum().sum() == len(intercepts2)/2) # RF has no intercepts

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
        # ftrs_renamed
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", ftrs_renamed="invalid")
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", ftrs_renamed={1: "ftr1"})
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", ftrs_renamed={"ftr1": 1})
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test", ftrs_renamed={"ftr1": "ftr2"})

    def test_valid_feature_selection_heatmap(self):
        so = self.so_with_calculated_preds
        try:
            so.feature_selection_heatmap(name="test")
        except Exception as e:
            self.fail(f"feature_selection_heatmap raised an exception: {e}")

        try:
            so.feature_selection_heatmap(name="test", cap=1)
        except Exception as e:
            self.fail(f"feature_selection_heatmap raised an exception: {e}")

    def test_types_correlations_heatmap(self):

        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        so.calculate_predictions(
            name="test",
            models=self.pipelines,
            scorers=self.scorers,
            hyperparameters=self.pipeline_hyperparameters,
            search_type="grid",
            n_jobs_outer=1,
            n_jobs_inner=1,
            inner_splitters=self.single_inner_splitter,
            store_correlations=True,
        )

        with self.assertRaises(TypeError):
            so.correlations_heatmap(name="test")
        with self.assertRaises(TypeError):
            so.correlations_heatmap(name=1, feature_name="Feature 1")
        with self.assertRaises(TypeError):
            so.correlations_heatmap(name="test", feature_name=1)
        with self.assertRaises(ValueError):
            so.correlations_heatmap(name="invalid", feature_name="Feature 1")
        with self.assertRaises(ValueError):
            so.correlations_heatmap(name="test", feature_name="invalid")
        # title
        with self.assertRaises(TypeError):
            so.correlations_heatmap(name="test", feature_name="Feature 1", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.correlations_heatmap(
                name="test", feature_name="Feature 1", figsize="invalid"
            )
        with self.assertRaises(ValueError):
            so.correlations_heatmap(
                name="test", feature_name="Feature 1", figsize=(1, 2, 3)
            )
        with self.assertRaises(TypeError):
            so.correlations_heatmap(
                name="test", feature_name="Feature 1", figsize=(1, "invalid")
            )
        with self.assertRaises(TypeError):
            so.correlations_heatmap(
                name="test", feature_name="Feature 1", figsize=(1, True)
            )
        # cap
        with self.assertRaises(TypeError):
            so.correlations_heatmap(
                name="test", feature_name="Feature 1", cap="invalid"
            )
        with self.assertRaises(ValueError):
            so.correlations_heatmap(name="test", feature_name="Feature 1", cap=-1)
        # ftrs_renamed
        with self.assertRaises(TypeError):
            so.correlations_heatmap(
                name="test", feature_name="Feature 1", ftrs_renamed="invalid"
            )
        with self.assertRaises(TypeError):
            so.correlations_heatmap(
                name="test", feature_name="Feature 1", ftrs_renamed={1: "ftr1"}
            )
        with self.assertRaises(TypeError):
            so.correlations_heatmap(
                name="test", feature_name="Feature 1", ftrs_renamed={"ftr1": 1}
            )
        with self.assertRaises(ValueError):
            so.correlations_heatmap(
                name="test", feature_name="Feature 1", ftrs_renamed={"ftr1": "ftr2"}
            )

    def test_valid_correlations_heatmap(self):
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        so.calculate_predictions(
            name="test",
            models=self.pipelines,
            scorers=self.scorers,
            hyperparameters=self.pipeline_hyperparameters,
            search_type="grid",
            n_jobs_outer=1,
            n_jobs_inner=1,
            inner_splitters=self.single_inner_splitter,
            store_correlations=True,
        )
        try:
            so.correlations_heatmap(name="test", feature_name="Feature 1")
        except Exception as e:
            self.fail(f"correlations_heatmap raised an exception: {e}")

    def test_types_feature_importance_timeplot(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.feature_importance_timeplot(name="test2")
            so2.feature_importance_timeplot(name="test2")
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name=1)
            so2.feature_importance_timeplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", title=1)
            so2.feature_importance_timeplot(name="RF", title=1)
            so2.feature_importance_timeplot(name="RIDGE", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", figsize="figsize")
            so2.feature_importance_timeplot(name="RF", figsize="figsize")
            so2.feature_importance_timeplot(name="RIDGE", figsize="figsize")
        with self.assertRaises(ValueError):
            so.feature_importance_timeplot(name="test", figsize=(0, 1, 2))
            so2.feature_importance_timeplot(name="RF", figsize=(0, 1, 2))
            so2.feature_importance_timeplot(name="RIDGE", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", figsize=(10, "hello"))
            so2.feature_importance_timeplot(name="RF", figsize=(10, "hello"))
            so2.feature_importance_timeplot(name="RIDGE", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", figsize=("hello", 6))
            so2.feature_importance_timeplot(name="RF", figsize=("hello", 6))
            so2.feature_importance_timeplot(name="RIDGE", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", figsize=("hello", "hello"))
            so2.feature_importance_timeplot(name="RF", figsize=("hello", "hello"))
            so2.feature_importance_timeplot(name="RIDGE", figsize=("hello", "hello"))
        # ftrs_renamed
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", ftrs_renamed=1)
            so2.feature_importance_timeplot(name="RF", ftrs_renamed=1)
            so2.feature_importance_timeplot(name="RIDGE", ftrs_renamed=1)
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", ftrs_renamed={1: "ftr1"})
            so2.feature_importance_timeplot(name="RF", ftrs_renamed={1: "ftr1"})
            so2.feature_importance_timeplot(name="RIDGE", ftrs_renamed={1: "ftr1"})
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", ftrs_renamed={"ftr1": 1})
            so2.feature_importance_timeplot(name="RF", ftrs_renamed={"ftr1": 1})
            so2.feature_importance_timeplot(name="RIDGE", ftrs_renamed={"ftr1": 1})
        with self.assertRaises(ValueError):
            so.feature_importance_timeplot(name="test", ftrs_renamed={"ftr1": "ftr2"})
            so2.feature_importance_timeplot(name="RF", ftrs_renamed={"ftr1": "ftr2"})
            so2.feature_importance_timeplot(name="RIDGE", ftrs_renamed={"ftr1": "ftr2"})
        # ftrs
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", ftrs=1)
            so2.feature_importance_timeplot(name="RF", ftrs=1)
            so2.feature_importance_timeplot(name="RIDGE", ftrs=1)
        with self.assertRaises(ValueError):
            so.feature_importance_timeplot(name="test", ftrs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            so2.feature_importance_timeplot(name="RF", ftrs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            so2.feature_importance_timeplot(name="RIDGE", ftrs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        with self.assertRaises(TypeError):
            so.feature_importance_timeplot(name="test", ftrs=[1])
            so2.feature_importance_timeplot(name="RF", ftrs=[1])
            so2.feature_importance_timeplot(name="RIDGE", ftrs=[1])
        with self.assertRaises(ValueError):
            so.feature_importance_timeplot(name="test", ftrs=["invalid"])
            so2.feature_importance_timeplot(name="RF", ftrs=["invalid"])
            so2.feature_importance_timeplot(name="RIDGE", ftrs=["invalid"])

    def test_valid_feature_importance_timeplot(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that running feature_importance_timeplot on pipeline "test" works
        try:
            so.feature_importance_timeplot(name="test")
            so2.feature_importance_timeplot(name="RF")
            so2.feature_importance_timeplot(name="RIDGE")
        except Exception as e:
            self.fail(f"feature_importance_timeplot raised an exception: {e}")
        # Check that the legend is correct
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(np.all(sorted(self.X.columns) == sorted(labels)))
        # Now rerun feature_importance_timeplot but with a feature renaming dictionary
        ftr_dict = {"CPI": "inflation"}
        try:
            so.feature_importance_timeplot(name="test", ftrs_renamed=ftr_dict)
            so2.feature_importance_timeplot(name="RF", ftrs_renamed=ftr_dict)
            so2.feature_importance_timeplot(name="RIDGE", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"feature_importance_timeplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            np.all(sorted(self.X.rename(columns=ftr_dict).columns) == sorted(labels))
        )
        # Now rename two features
        ftr_dict = {"CPI": "inflation", "GROWTH": "growth"}
        try:
            so.feature_importance_timeplot(name="test", ftrs_renamed=ftr_dict)
            so2.feature_importance_timeplot(name="RF", ftrs_renamed=ftr_dict)
            so2.feature_importance_timeplot(name="RIDGE", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"feature_importance_timeplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            np.all(sorted(self.X.rename(columns=ftr_dict).columns) == sorted(labels))
        )
        # Now rename all features
        ftr_dict = {ftr: f"ftr{i}" for i, ftr in enumerate(self.X.columns)}
        try:
            so.feature_importance_timeplot(name="test", ftrs_renamed=ftr_dict)
            so2.feature_importance_timeplot(name="RF", ftrs_renamed=ftr_dict)
            so2.feature_importance_timeplot(name="RIDGE", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"feature_importance_timeplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            np.all(sorted(self.X.rename(columns=ftr_dict).columns) == sorted(labels))
        )
        # Finally, test that the title works
        title = ax.get_title()
        self.assertTrue(title == "Feature importances for pipeline: RIDGE")
        # Try changing the title
        try:
            so.feature_importance_timeplot(name="test", title="hello")
            so2.feature_importance_timeplot(name="RF", title="hello")
            so2.feature_importance_timeplot(name="RIDGE", title="hello")
        except Exception as e:
            self.fail(f"feature_importance_timeplot raised an exception: {e}")
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "hello")

    def test_types_intercepts_timeplot(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.intercepts_timeplot(name="test2")
        with self.assertRaises(ValueError):
            so2.intercepts_timeplot(name="test2")
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name=1)
        with self.assertRaises(TypeError):
            so2.intercepts_timeplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", title=1)
        with self.assertRaises(TypeError):
            so2.intercepts_timeplot(name="RIDGE", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize="figsize")
        with self.assertRaises(TypeError):
            so2.intercepts_timeplot(name="RIDGE", figsize="figsize")
        with self.assertRaises(ValueError):
            so.intercepts_timeplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(ValueError):
            so2.intercepts_timeplot(name="RIDGE", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so2.intercepts_timeplot(name="RIDGE", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so2.intercepts_timeplot(name="RIDGE", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize=("hello", "hello"))
        with self.assertRaises(TypeError):
            so2.intercepts_timeplot(name="RIDGE", figsize=("hello", "hello"))

    def test_valid_intercepts_timeplot(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that running intercepts_timeplot on pipeline "test" works
        try:
            so.intercepts_timeplot(name="test")
            so2.intercepts_timeplot(name="RIDGE")
        except Exception as e:
            self.fail(f"intercepts_timeplot raised an exception: {e}")

    def test_types_coefs_stackedbarplot(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test2")
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="test2")
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name=1)
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", title=1)
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", title=1)
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize="figsize")
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", figsize="figsize")
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", figsize="figsize")
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RF", figsize=(0, 1, 2))
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RIDGE", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize=("hello", "hello"))
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", figsize=("hello", "hello"))
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", figsize=("hello", "hello"))
        # ftrs_renamed
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed=1)
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", ftrs_renamed=1)
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", ftrs_renamed=1)
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed={1: "ftr1"})
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", ftrs_renamed={1: "ftr1"})
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", ftrs_renamed={1: "ftr1"})
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed={"ftr1": 1})
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", ftrs_renamed={"ftr1": 1})
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", ftrs_renamed={"ftr1": 1})
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed={"ftr1": "ftr2"})
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RF", ftrs_renamed={"ftr1": "ftr2"})
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RIDGE", ftrs_renamed={"ftr1": "ftr2"})
        # ftrs
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs=1)
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", ftrs=1)
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", ftrs=1)
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(
                name="test", ftrs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            )
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(
                name="RF", ftrs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            )
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(
                name="RIDGE", ftrs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            )
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs=[1])
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", ftrs=[1])
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", ftrs=[1])
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test", ftrs=["invalid"])
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RF", ftrs=["invalid"])
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RIDGE", ftrs=["invalid"])
        # cap
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", cap="invalid")
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RF", cap="invalid")
        with self.assertRaises(TypeError):
            so2.coefs_stackedbarplot(name="RIDGE", cap="invalid")
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test", cap=-1)
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RF", cap=-1)
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RIDGE", cap=-1)
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test", cap=11)
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RF", cap=11)
        with self.assertRaises(ValueError):
            so2.coefs_stackedbarplot(name="RIDGE", cap=11)

    def test_valid_coefs_stackedbarplot(self):
        so = self.so_with_calculated_preds
        so2 = self.so_no_na
        # Test that running coefs_stackedbarplot on pipeline "test" works
        try:
            so.coefs_stackedbarplot(name="test")
            so2.coefs_stackedbarplot(name="RF")
            so2.coefs_stackedbarplot(name="RIDGE")
        except Exception as e:
            self.fail(f"coefs_stackedbarplot raised an exception: {e}")
        # Check that the title is correct
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "Stacked bar plot of model coefficients: RIDGE")
        # Change the title
        try:
            so.coefs_stackedbarplot(name="test", title="hello")
            so2.coefs_stackedbarplot(name="RF", title="hello")
            so2.coefs_stackedbarplot(name="RIDGE", title="hello")
        except Exception as e:
            self.fail(f"coefs_stackedbarplot raised an exception: {e}")
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "hello")
        # Now rerun coefs_stackedbarplot but with a feature renaming dictionary
        ftr_dict = {"CPI": "inflation"}
        try:
            so.coefs_stackedbarplot(name="test", ftrs_renamed=ftr_dict)
            so2.coefs_stackedbarplot(name="RF", ftrs_renamed=ftr_dict)
            so2.coefs_stackedbarplot(name="RIDGE", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"coefs_stackedbarplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = sorted([text.get_text() for text in legend.get_texts()])
        # Check that the legend is correct for the pipeline with dropped NAs
        ftrcoef_df = so.get_feature_importances(name="test")
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
        correct_labels = sorted(list(set(correct_labels)))
        self.assertTrue(np.all(labels == correct_labels))
        # Check that the legend is correct for the RF pipeline with NAs
        ftrcoef_df = so2.get_feature_importances(name="RF")
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
        correct_labels = sorted(list(set(correct_labels)))
        self.assertTrue(np.all(labels == correct_labels))
        # Check that the legend is correct for the RIDGE pipeline with NAs
        ftrcoef_df = so.get_feature_importances(name="test")
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
        correct_labels = sorted(list(set(correct_labels)))
        self.assertTrue(np.all(labels == correct_labels))

    def test_invalid_plots(self):
        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
        )
        # # Test that an error is raised if calculate_predictions has not been run
        # with self.assertRaises(ValueError):
        #     so.nsplits_timeplot(name="test")
        # Test that an error is raised if calculate_predictions has not been run
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test")
        # Test that an error is raised if calculate_predictions has not been run
        with self.assertRaises(ValueError):
            so.intercepts_timeplot(name="test")
        # Test that an error is raised if calculate_predictions has not been run
        with self.assertRaises(ValueError):
            so.feature_importance_timeplot(name="test")
        # Test that if no signals have been calculated, an error is raised
        with self.assertRaises(ValueError):
            so.get_feature_importances(name="test")
        # Test that if no signals have been calculated, an error is raised
        with self.assertRaises(ValueError):
            so.get_intercepts(name="test")
        # Test that if no signals have been calculated, an error is raised
        with self.assertRaises(ValueError):
            so.get_selected_features(name="test")
        # Test that invalid names are caught
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name=1)
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name=[1, 2, 3])
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test")
        # Test that invalid names are caught
        with self.assertRaises(TypeError):
            so.models_heatmap(name=1)
        with self.assertRaises(TypeError):
            so.models_heatmap(name=[1, 2, 3])
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test")

    def test_types_models_heatmap(self):
        so = self.so_with_calculated_preds

        with self.assertRaises(TypeError):
            so.models_heatmap(name=1)
        with self.assertRaises(ValueError):
            so.models_heatmap(name="invalid")
        # cap
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", cap="invalid")
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", cap=-1)
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", cap=11)
        # title
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize="figsize")
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(1.5, 2, 3))
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize=(1.5, "e"))

    def test_valid_models_heatmap(self):

        so = self.so_with_calculated_preds
        try:
            so.models_heatmap(name="test")
        except Exception as e:
            self.fail(f"models_heatmap raised an exception: {e}")

    def test_types_nsplits_timeplot(self):
        so = self.so_with_calculated_preds

        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name=1)
        with self.assertRaises(ValueError):
            so.nsplits_timeplot(name="invalid")
        # title
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name="test", figsize="figsize")
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name="test", figsize=1)
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name="test", figsize=(1.5, "e"))
        with self.assertRaises(ValueError):
            so.nsplits_timeplot(name="test", figsize=(0,))
        with self.assertRaises(ValueError):
            so.nsplits_timeplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(ValueError):
            so.nsplits_timeplot(name="test", figsize=(2, -1))

    def test_valid_nsplits_timeplot(self):
        so = self.so_with_calculated_preds
        try:
            so.nsplits_timeplot(name="test")
        except Exception as e:
            self.fail(f"feature_selection_heatmap raised an exception: {e}")


def _get_X_y(so: SignalOptimizer, drop_nas: bool):
    df_long = categories_df(
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
    if drop_nas:
        df_long = df_long.dropna()
    else:
        df_long = df_long.dropna(subset=[so.xcats[-1]])

    df_long = df_long.sort_index()

    df_long.index.names = ["cid", "real_date"]
    new_outer_level = df_long.index.levels[0].astype("object")
    df_long.index = pd.MultiIndex(
        levels=[new_outer_level, df_long.index.levels[1]],
        codes=df_long.index.codes,
        names=df_long.index.names,
    )
    X = df_long.iloc[:, :-1]
    y = df_long.iloc[:, -1]
    return X, y, df_long


if __name__ == "__main__":
    Test = TestAll()
    Test.setUpClass()
    Test.test_types_feature_selection_heatmap()
