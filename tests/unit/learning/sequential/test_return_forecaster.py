import numpy as np
import pandas as pd
import unittest
import matplotlib
import matplotlib.pyplot as plt

import parameterized
import itertools

from macrosynergy.management import make_qdf
from unittest.mock import patch
import scipy.stats as stats

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer

from macrosynergy.learning import PanelPCA, RandomEffects, ExpandingKFoldPanelSplit, PanelStandardScaler, LassoSelector, SignalOptimizer, RollingKFoldPanelSplit, ReturnForecaster

class TestReturnForecaster(unittest.TestCase):
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

        # Set up different experiments
        # (1) Ridge regression 
        # (2) Variable selection + Linear regression
        # (3) Random forest regression
        # (4) PCA + Ridge regression
        # In addition to specialised tests for each of these models, I need to test that
        # SignalOptimizer gives the same results as ReturnForecaster. 

        self.pipelines = [
            {
                "Ridge": Pipeline([
                    ("scaler", PanelStandardScaler()),
                    ("ridge", Ridge(random_state = 42))
                ]),
            },
            {
                "Var+LR": Pipeline([
                    ("scaler", PanelStandardScaler()),
                    ("selector", LassoSelector(n_factors = 2)),
                    ("lr", LinearRegression())
                ]),
            },
            {
                "RF": RandomForestRegressor(random_state = 1, n_estimators=20, max_depth = 3),
            },
            {
                "PCA+Ridge": Pipeline([
                    ("scaler", PanelStandardScaler()),
                    ("pca", PanelPCA(n_components = 2)),
                    ("ridge", Ridge())
                ]),
            }
        ]

        self.hyperparameters = [
            {
                "Ridge": {
                    "ridge__alpha": [0.1, 1, 10]
                }
            },
            {
                "Var+LR": {}
            },
            {
                "RF": {
                    "min_samples_leaf": [1, 32]
                }
            },
            {
                "PCA+Ridge": {
                    "ridge__alpha": [0.1, 1, 10]
                }
            }
        ]

        self.names = ["Ridge", "Var+LR", "RF", "PCA+Ridge"]

        self.evaluation_date = "2020-11-30" # Aim to produce forecast for December 2020

        self.black_valid = {
            "AUD": (
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2020, month=4, day=1),
            ),
            "GBP": (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2020, month=12, day=31),
            ),
        }

        # Initialize signal optimizer and return forecaster classes
        self.so = SignalOptimizer(
            df = self.df,
            xcats = ["CPI", "GROWTH", "RIR", "XR"],
            blacklist = self.black_valid,
        )

        self.rf = ReturnForecaster(
            df = self.df,
            xcats = ["CPI", "GROWTH", "RIR", "XR"],
            real_date = self.evaluation_date,
            blacklist = self.black_valid,
        )

        for i in range(len(self.pipelines)):
            # For each pipeline, run a sequential process
            self.so.calculate_predictions(
                name = "SO_" + self.names[i],
                models = self.pipelines[i],
                hyperparameters = self.hyperparameters[i],
                scorers = {"R2": make_scorer(r2_score)},
                inner_splitters = {
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                store_correlations=True if i==3 else False,
            )

            # For each pipeline, run a forecast on November 30th 2020
            self.rf.calculate_predictions(
                name = "RF_" + self.names[i],
                models = self.pipelines[i],
                hyperparameters = self.hyperparameters[i],
                scorers = {"R2": make_scorer(r2_score)},
                inner_splitters = {
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                store_correlations=True if i==3 else False,
            )

        # Create invalid blacklist dictionaries for the different experiments
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

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(self.mpl_backend)

    def test_types_init(self):
        """
        Test appropriate errors are raised when the input types are incorrect.
        """
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=pd.Series([]),
                xcats=self.xcats,
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats="invalid",
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=[1, 2, 3],
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                cids="invalid",
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                cids=[1, 2, 3],
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=pd.DataFrame([]),
                xcats=self.xcats,
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=["invalid"],
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats + ["invalid"],
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                cids=self.cids + ["invalid"],
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                start=1,
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                end=1,
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                blacklist=list(),
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                blacklist={1: "invalid"},
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                blacklist={"CAD": "invalid"},
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                blacklist={"CAD": tuple()},
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                freq=1,
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                freq="invalid",
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                lag="invalid",
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                lag=-1,real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs="invalid",
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs=[1, "sum"],
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs=["last", 1],
                real_date = self.evaluation_date,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                xcat_aggs=[1, 2],
                real_date = self.evaluation_date,
            )
        # generate_labels should be a callable or None
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                generate_labels=1,
                real_date = self.evaluation_date,
            )
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                generate_labels="invalid",
                real_date = self.evaluation_date,
            )

        # real date
        with self.assertRaises(TypeError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                real_date=1,
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                real_date="hello",
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                real_date="2014-01-01",
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                real_date="2013-06-01",
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                real_date="2025-01-01",
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                real_date="2014-01-01",
            )
        with self.assertRaises(ValueError):
            rf = ReturnForecaster(
                df=self.df,
                xcats=self.xcats,
                real_date="2018-07-07", # weekend
            )

    def test_valid_init(self):
        """
        Test that the initialization of ReturnForecaster results in the same dataset
        as the initialization of SignalOptimizer.
        """
        # Return forecaster training and test sets
        X_train_rf = self.rf.X
        y_train_rf = self.rf.y
        X_test_rf = self.rf.X_test

        # Signal optimizer training and test sets
        X = self.so.X
        y = self.so.y
        X_train_so = X[X.index.get_level_values(1) <= pd.Timestamp(year=2020, month=11, day=30)]
        y_train_so = y[y.index.get_level_values(1) <= pd.Timestamp(year=2020, month=11, day=30)]
        X_test_so = X[X.index.get_level_values(1) > pd.Timestamp(year=2020, month=11, day=30)]

        # Check that each training set is the same 
        np.testing.assert_array_equal(X_train_rf.values, X_train_so.values)
        np.testing.assert_array_equal(y_train_rf.values, y_train_so.values)

        # Check that each test set is the same
        np.testing.assert_array_equal(X_test_rf.values, X_test_so.values)

    def test_types_calculate_predictions(self):
        # Name
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name=1,
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name=True,
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )

        # Models
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=1,
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models={},
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models={1: LinearRegression()},
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models={"LR": 1},
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )

        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models={"LR": RandomEffects()},
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )

        # Hyperparameters
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=1,
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={},
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # wrong length of hyperparameter dict
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={"linreg": {"fit_intercept": [True, False]}},
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # check hyperparameter dict is a nested dict
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={"linreg": 1, "ridge": 1},
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "linreg": 1,
                    "ridge": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "ridge": 1,
                    "linreg": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "ridge": {1: [True, False]},
                    "linreg": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # Check inner hparam dictionaries specify a grid
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "ridge": {"fit_intercept": 1},
                    "linreg": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "linreg": {"fit_intercept": 1},
                    "ridge": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "ridge": {"fit_intercept": 1},
                    "linreg": {"fit_intercept": 1},
                },
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # Scorers should be a dict
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers=1,
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # Scorers dict shouldn't be empty
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # Scorer keys should be strings
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={1: make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score), 1: make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # Scorer values should be scorers
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": 1},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # Inner splitters should be a dict
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters=1,
            )
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters=ExpandingKFoldPanelSplit(),
            )
        # Inner splitters dict shouldn't be empty
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={},
            )
        # Inner splitters keys should be strings
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={1: ExpandingKFoldPanelSplit()},
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "ExpandingKFold": ExpandingKFoldPanelSplit(n_splits=5),
                    1: ExpandingKFoldPanelSplit(n_splits=10),
                },
            )
        # Inner splitters values should be splitters
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={"ExpandingKFold": 1},
            )
        # Search type should be a string
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type=1,
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # Search type should be either "grid" or "prior"
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="invalid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(NotImplementedError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="bayes",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # Normalize_fold_results should be a boolean
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                normalize_fold_results=1,
            )
        # If normalize_fold_results is True, then at least 2 hparams should
        # be provided for each model
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "ridge": {"alpha": [0.1]},
                    "linreg": {"fit_intercept": [True, False]},
                },
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                normalize_fold_results=True,
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "ridge": {"alpha": [0.1, 1, 10]},
                    "linreg": {"fit_intercept": [True]},
                },
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                normalize_fold_results=True,
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "ridge": {"alpha": [0.1]},
                    "linreg": {"fit_intercept": [True]},
                },
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                normalize_fold_results=True,
            )
        # Test structure of the hyperparameter grid when
        # search_type is "prior"
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=1,
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={},
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # wrong length of hyperparameter dict
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={"LR": {}},
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # wrong model names in hyperparameter dict
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={"LR": {}, "L2": {}},
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # invalid hyperparameter dictionary
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={"linreg": {1: {}}},
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={"linreg": {1: {}}, "ridge": {1: {}}},
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # hyperparameter values are neither grids nor distributions
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "linreg": {"fit_intercept": [True, False]},
                    "ridge": {"alpha": 1},
                },
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters={
                    "linreg": {"fit_intercept": 1},
                    "ridge": {"alpha": stats.expon()},
                },
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        # cv_summary should be str or callable
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                cv_summary=1,
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                cv_summary="invalid",
            )
        # min_cids should be an int
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                min_cids="invalid",
            )
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                min_cids=7.2,
            )
        # n_iter should be a positive int, if not None
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                n_iter="invalid",
            )
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                n_iter=None,
            )
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                n_iter=3.4,
            )
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                n_iter=3.4,
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="prior",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                n_iter=-3,
            )
        # n_jobs should be an integer >= -1
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv="invalid",
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=-2,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_model="invalid",
                n_jobs_cv=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_model=-2,
                n_jobs_cv=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
            )

        # store_correlations
        with self.assertRaises(TypeError):
            self.rf.calculate_predictions(
                name="test",
                models=self.pipelines[0],
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                store_correlations=1,
            )
        # If store_correlations is True, then models must be pipelines
        with self.assertRaises(ValueError):
            self.rf.calculate_predictions(
                name="test",
                models={"LR": LinearRegression()},
                scorers={"R2": make_scorer(r2_score)},
                hyperparameters=self.hyperparameters[0],
                search_type="grid",
                n_jobs_cv=1,
                n_jobs_model=1,
                inner_splitters={
                    "Rolling": RollingKFoldPanelSplit(5),
                },
                store_correlations=True,
            )

    def test_valid_functionality(self):
        """
        Test that ReturnForecaster and SignalOptimizer give the same results when run with
        the same parameters at the end of November 2020.
        """
        # First check that the models selected are the same on the evaluation date
        so_models_selected = self.so.get_optimal_models()[self.so.get_optimal_models().real_date==pd.Timestamp(year=2020, month=11, day=30)]
        rf_models_selected = self.rf.get_optimal_models()

        # First rename the model names to match
        so_models_selected.iloc[:,1] = so_models_selected.iloc[:,1].str.split("_").str[1]
        rf_models_selected.iloc[:,1] = rf_models_selected.iloc[:,1].str.split("_").str[1]

        np.testing.assert_array_equal(so_models_selected.values, rf_models_selected.values)

        # Check that the predictions for each pipeline align
        so_preds = self.so.get_optimized_signals()[self.so.get_optimized_signals().real_date==pd.Timestamp(year=2020, month=11, day=30)].dropna()
        rf_preds = self.rf.get_optimized_signals()
        so_preds.iloc[:,2] = so_preds.iloc[:,2].str.split("_").str[1]
        rf_preds.iloc[:,2] = rf_preds.iloc[:,2].str.split("_").str[1]

        np.testing.assert_array_equal(so_preds.values, rf_preds.values)

        # Check the feature importances for each model align
        so_fi = self.so.get_feature_importances()[self.so.get_feature_importances().real_date==pd.Timestamp(year=2020, month=11, day=30)].dropna()
        rf_fi = self.rf.get_feature_importances().dropna()
        so_fi.iloc[:,1] = so_fi.iloc[:,1].str.split("_").str[1]
        rf_fi.iloc[:,1] = rf_fi.iloc[:,1].str.split("_").str[1]

        np.testing.assert_array_equal(so_fi.values, rf_fi.values)

        # Check the intercepts for each model align
        so_int = self.so.get_intercepts()[self.so.get_intercepts().real_date==pd.Timestamp(year=2020, month=11, day=30)].dropna()
        rf_int = self.rf.get_intercepts().dropna()
        so_int.iloc[:,1] = so_int.iloc[:,1].str.split("_").str[1]
        rf_int.iloc[:,1] = rf_int.iloc[:,1].str.split("_").str[1]

        np.testing.assert_array_equal(so_int.values, rf_int.values)

        # Check feature selection aligns
        so_fs = self.so.get_selected_features()[self.so.get_selected_features().real_date==pd.Timestamp(year=2020, month=11, day=30)]
        rf_fs = self.rf.get_selected_features()
        so_fs.iloc[:,1] = so_fs.iloc[:,1].str.split("_").str[1]
        rf_fs.iloc[:,1] = rf_fs.iloc[:,1].str.split("_").str[1]

        np.testing.assert_array_equal(so_fs.values, rf_fs.values)

        # Check that the feature correlations are appropriately calculated
        so_corr = self.so.get_feature_correlations()[self.so.get_feature_correlations().real_date==pd.Timestamp(year=2020, month=11, day=30)]
        rf_corr = self.rf.get_feature_correlations()
        so_corr.iloc[:,1] = so_corr.iloc[:,1].str.split("_").str[1]
        rf_corr.iloc[:,1] = rf_corr.iloc[:,1].str.split("_").str[1]

        np.testing.assert_array_equal(so_corr.values, rf_corr.values)

