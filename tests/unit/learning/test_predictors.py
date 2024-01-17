import numpy as np
import pandas as pd
from scipy.stats import expon 

import unittest

from macrosynergy.learning import (
    NaivePredictor,
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
    LassoSelector,
    panel_cv_scores,
    SignalOptimizer,
    RollingKFoldPanelSplit,
)

from sklearn.linear_model import (
    LinearRegression,
)

from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_absolute_error,
)


class TestSWLRegression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-06-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all eligible dates
            sdate = df_cids.loc[cid]["earliest"]
            edate = df_cids.loc[cid]["latest"]
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]
            for work_day in work_days:
                tuples.append((cid, work_day))

        n_samples = len(tuples)
        ftrs = np.random.normal(loc=0, scale=1, size=(n_samples, 3))
        labels = np.matmul(ftrs, [1, 2, -1]) + np.random.normal(0, 0.5, len(ftrs))
        df = pd.DataFrame(
            data=np.concatenate((np.reshape(labels, (-1, 1)), ftrs), axis=1),
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
            columns=xcats,
            dtype=np.float32,
        )

        self.X = df.drop(columns="XR")
        self.y = df["XR"]

    def test_valid_init(self):
        # Test that a sign weighted ols model is successfully instantiated
        try:
            model = SignWeightedLinearRegression()
        except Exception as e:
            self.fail(
                "SignWeightedLinearRegression constructor with a LinearRegression object raised an exception: {}".format(
                    e
                )
            )
        self.assertIsInstance(model, SignWeightedLinearRegression)
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.copy_X, True)
        self.assertTrue(model.n_jobs is None)
        self.assertEqual(model.positive, False)
        self.assertIsInstance(model.model, LinearRegression)
        self.assertEqual(model.model.fit_intercept, model.fit_intercept)
        self.assertEqual(model.model.copy_X, model.copy_X)
        self.assertTrue(model.model.n_jobs is None)
        self.assertEqual(model.model.positive, model.positive)

        # Test that a sign weighted ols model is successfully instantiated
        # when other arguments are passed
        try:
            model = SignWeightedLinearRegression(fit_intercept=False, positive=True)
        except Exception as e:
            self.fail(
                "SignWeightedLinearRegression constructor with a non-negative OLS object raised an exception: {}".format(
                    e
                )
            )
        self.assertIsInstance(model, SignWeightedLinearRegression)
        self.assertEqual(model.fit_intercept, False)
        self.assertEqual(model.copy_X, True)
        self.assertEqual(model.positive, True)
        self.assertTrue(model.n_jobs is None)
        self.assertIsInstance(model.model, LinearRegression)
        self.assertEqual(model.model.fit_intercept, model.fit_intercept)
        self.assertEqual(model.model.copy_X, model.copy_X)
        self.assertTrue(model.model.n_jobs is None)
        self.assertEqual(model.model.positive, model.positive)

    def test_types_init(self):
        # Check that incorrectly specified arguments raise exceptions
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression(fit_intercept="fit_intercept")
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression(copy_X="copy_X")
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression(n_jobs="n_jobs")
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression(n_jobs=4.3)
        with self.assertRaises(ValueError):
            model = SignWeightedLinearRegression(n_jobs=-2)
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression(positive="positive")

    def test_valid_weightsfunc(self):
        # The weights function is designed to return the same weights that a classifier with
        # balanced class weights would return in scikit-learn. Test that this is the case.
        model = SignWeightedLinearRegression()
        sample_weights, pos_weight, neg_weight = model._SignWeightedLinearRegression__calculate_sample_weights(
            self.y
        )
        correct_pos_weight = len(self.y) / (2 * np.sum(self.y >= 0))
        correct_neg_weight = len(self.y) / (2 * np.sum(self.y < 0))
        self.assertEqual(pos_weight, correct_pos_weight)
        self.assertEqual(neg_weight, correct_neg_weight)

    def test_valid_fit(self):
        # Check that the SignWeightedRegressor fit is equivalent to using LR with sample weights
        model = SignWeightedLinearRegression()
        model.fit(self.X, self.y)
        self.assertIsInstance(model.model, LinearRegression)
        model_check = LinearRegression()
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        self.assertTrue(np.all(model.model.coef_ == model_check.coef_))
        self.assertEqual(model.model.intercept_, model_check.intercept_)
        # check fit with dataframe targets
        model_check.fit(self.X, self.y.to_frame(), sample_weight=model.sample_weights)
        self.assertTrue(np.all(model.model.coef_ == model_check.coef_))
        self.assertEqual(model.model.intercept_, model_check.intercept_)
        # Check that the SignWeightedRegressor fit is equivalent to using Ridge with sample weights
        model = SignWeightedLinearRegression(fit_intercept=False, positive=True)
        model.fit(self.X, self.y)
        self.assertIsInstance(model.model, LinearRegression)
        model_check = LinearRegression(fit_intercept=False, positive=True)
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        self.assertTrue(np.all(model.model.coef_ == model_check.coef_))
        self.assertEqual(model.model.intercept_, model_check.intercept_)
        # check fit with dataframe targets
        model_check.fit(self.X, self.y.to_frame(), sample_weight=model.sample_weights)
        self.assertTrue(np.all(model.model.coef_ == model_check.coef_))
        self.assertEqual(model.model.intercept_, model_check.intercept_)

    def test_types_fit(self):
        # Test that non dataframe X returns a TypeError
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression()
            model.fit(self.X.values, self.y)
        # Test that non dataframe and non series y returns a TypeError
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression()
            model.fit(self.X, self.y.values)
        # Test that a ValueError is raised if y is a dataframe with more than one column
        with self.assertRaises(ValueError):
            model = SignWeightedLinearRegression()
            model.fit(self.X, self.X)
        # Test that a ValueError is raised if X and y are not multi-indexed
        with self.assertRaises(ValueError):
            model = SignWeightedLinearRegression()
            model.fit(self.X.reset_index(), self.y)
        with self.assertRaises(ValueError):
            model = SignWeightedLinearRegression()
            model.fit(self.X, self.y.reset_index())
        # Test that a ValueError is raised if the multi-indices of X and y don't match
        with self.assertRaises(ValueError):
            model = SignWeightedLinearRegression()
            model.fit(self.X, self.y.iloc[1:])

    def test_valid_predict(self):
        # Check that the SignWeightedLinearRegression predict is equivalent to using LR with sample weights
        model = SignWeightedLinearRegression()
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        model_check = LinearRegression()
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        y_pred_check = model_check.predict(self.X)
        self.assertTrue(np.all(y_pred == y_pred_check))
        # Check that the SignWeightedLinearRegression predict is equivalent to using restricted LR with sample weights
        model = SignWeightedLinearRegression(fit_intercept=False, positive=True)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        model_check = LinearRegression(fit_intercept=False, positive=True)
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        y_pred_check = model_check.predict(self.X)
        self.assertTrue(np.all(y_pred == y_pred_check))

    def test_types_predict(self):
        # Test that non dataframe X returns a TypeError
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression()
            model.fit(self.X, self.y)
            model.predict(self.X.values)
        # Test that a ValueError is raised if X is not multi-indexed
        with self.assertRaises(ValueError):
            model = SignWeightedLinearRegression()
            model.fit(self.X, self.y)
            model.predict(self.X.reset_index())

    def test_panelcvscores_compatible(self):
        # Test that the SignWeightedLinearRegression is compatible with panel_cv_scores
        splitter = RollingKFoldPanelSplit(n_splits=5)
        estimators = {"SWOLS": SignWeightedLinearRegression(), "OLS": LinearRegression()}
        scoring = {"NEG_RMSE": make_scorer(mean_squared_error, squared=False, greater_is_better=False), "NEG_MAE": make_scorer(mean_absolute_error, greater_is_better=False)}
        try:
            cv_df = panel_cv_scores(
                X = self.X,
                y = self.y,
                splitter = splitter,
                estimators = estimators,
                scoring = scoring,
                n_jobs = 1
            )
        except Exception as e:
            self.fail(f"panel_cv_scores raised an exception {e} when using SignWeightedLinearRegression as an estimator.")
        self.assertIsInstance(cv_df, pd.DataFrame)
        self.assertEqual(cv_df.shape[1], 2)
        self.assertEqual(sorted(cv_df.columns), sorted(estimators.keys()))
        self.assertEqual(sorted(cv_df.index)[:2], sorted(scoring.keys())[:2])

    def test_signaloptimizer_compatible(self):
        # Test that the SignWeightedLinearRegression is compatible with SignalOptimizer
        inner_splitter = RollingKFoldPanelSplit(n_splits=5)
        models = {
            "SWOLS": SignWeightedLinearRegression(),
            "OLS": LinearRegression(),
        }
        metric = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
        hparam_grid = {
            "SWOLS": {"fit_intercept": [True, False]},
            "OLS": {"fit_intercept": [True, False]},
        }
        so = SignalOptimizer(
            inner_splitter=inner_splitter,
            X = self.X,
            y = self.y,
            blacklist=None,
        )
        try:
            so.calculate_predictions(
                name="test",
                models=models,
                hparam_grid=hparam_grid,
                metric=metric,
            )
        except Exception as e:
            self.fail(f"SignalOptimizer raised an exception {e} when using SignWeightedLinearRegression as an estimator.")

        df_sigs = so.get_optimized_signals(name="test")
        df_models = so.get_optimal_models(name="test")

        self.assertIsInstance(df_sigs, pd.DataFrame)
        self.assertEqual(df_sigs.shape[1], 4)
        self.assertEqual(sorted(df_sigs.columns), ["cid", "real_date", "value", "xcat"])
        self.assertTrue(len(df_sigs.xcat.unique()) == 1)
        self.assertEqual(df_sigs.xcat.unique()[0], "test")

        self.assertIsInstance(df_sigs, pd.DataFrame)
        self.assertEqual(df_models.shape[1], 4)
        self.assertEqual(sorted(df_models.columns), ["hparams", "model_type", "name", "real_date"])
        self.assertTrue(len(df_models.name.unique()) == 1)
        self.assertEqual(df_models.name.unique()[0], "test")

class TestTWLRegression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-06-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all eligible dates
            sdate = df_cids.loc[cid]["earliest"]
            edate = df_cids.loc[cid]["latest"]
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]
            for work_day in work_days:
                tuples.append((cid, work_day))

        n_samples = len(tuples)
        ftrs = np.random.normal(loc=0, scale=1, size=(n_samples, 3))
        labels = np.matmul(ftrs, [1, 2, -1]) + np.random.normal(0, 0.5, len(ftrs))
        df = pd.DataFrame(
            data=np.concatenate((np.reshape(labels, (-1, 1)), ftrs), axis=1),
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
            columns=xcats,
            dtype=np.float32,
        )

        self.X = df.drop(columns="XR")
        self.y = df["XR"]

    def test_valid_init(self):
        # Test that a time weighted ols model is successfully instantiated
        try:
            model = TimeWeightedLinearRegression()
        except Exception as e:
            self.fail(
                "TimeWeightedLinearRegression constructor raised an exception: {}".format(
                    e
                )
            )
        self.assertIsInstance(model, TimeWeightedLinearRegression)
        self.assertEqual(model.half_life, 12*21)
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.copy_X, True)
        self.assertTrue(model.n_jobs is None)
        self.assertEqual(model.positive, False)
        self.assertIsInstance(model.model, LinearRegression)
        self.assertEqual(model.model.fit_intercept, model.fit_intercept)
        self.assertEqual(model.model.copy_X, model.copy_X)
        self.assertTrue(model.model.n_jobs is None)
        self.assertEqual(model.model.positive, model.positive)

        # Test that a time weighted ols model is successfully instantiated
        # when other arguments are passed
        try:
            model = TimeWeightedLinearRegression(fit_intercept=False, positive=True)
        except Exception as e:
            self.fail(
                "TimeWeightedLinearRegression constructor with a non-negative OLS object raised an exception: {}".format(
                    e
                )
            )
        self.assertIsInstance(model, TimeWeightedLinearRegression)
        self.assertEqual(model.half_life, 12*21)
        self.assertEqual(model.fit_intercept, False)
        self.assertEqual(model.copy_X, True)
        self.assertEqual(model.positive, True)
        self.assertTrue(model.n_jobs is None)
        self.assertIsInstance(model.model, LinearRegression)
        self.assertEqual(model.model.fit_intercept, model.fit_intercept)
        self.assertEqual(model.model.copy_X, model.copy_X)
        self.assertTrue(model.model.n_jobs is None)
        self.assertEqual(model.model.positive, model.positive)

    def test_types_init(self):
        # Check that incorrectly specified arguments raise exceptions
        with self.assertRaises(TypeError):
            model = TimeWeightedLinearRegression(half_life="fit_intercept")
        with self.assertRaises(ValueError):
            model = TimeWeightedLinearRegression(half_life=-2)
        with self.assertRaises(TypeError):
            model = TimeWeightedLinearRegression(fit_intercept="fit_intercept")
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression(copy_X="copy_X")
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression(n_jobs="n_jobs")
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression(n_jobs=4.3)
        with self.assertRaises(ValueError):
            model = SignWeightedLinearRegression(n_jobs=-2)
        with self.assertRaises(TypeError):
            model = SignWeightedLinearRegression(positive="positive")

    def test_valid_weightsfunc(self):
        model = TimeWeightedLinearRegression(half_life=21)
        sample_weights = model._TimeWeightedLinearRegression__calculate_sample_weights(self.y)
        # compute expected weights using scipy 
        unique_dates = sorted(self.y.index.get_level_values("real_date").unique(), reverse=True)
        num_dates = len(unique_dates)
        scale = 21/np.log(2) 
        expected_weights = expon.pdf(np.arange(num_dates), scale=scale)
        expected_weights /= np.sum(expected_weights)
        weight_map = dict(zip(unique_dates, expected_weights))
        expected_weights = self.y.index.get_level_values("real_date").map(weight_map).values
        # check that the scipy weights and our calculated weights are equal
        np.testing.assert_array_almost_equal(sample_weights, expected_weights)

    def test_valid_fit(self):
        # Check that the TimeWeightedLinearRegressor fit is equivalent to using LR with sample weights
        model = TimeWeightedLinearRegression()
        model.fit(self.X, self.y)
        self.assertIsInstance(model.model, LinearRegression)
        model_check = LinearRegression()
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        self.assertTrue(np.all(model.model.coef_ == model_check.coef_))
        self.assertEqual(model.model.intercept_, model_check.intercept_)
        # check fit with dataframe targets
        model_check.fit(self.X, self.y.to_frame(), sample_weight=model.sample_weights)
        self.assertTrue(np.all(model.model.coef_ == model_check.coef_))
        self.assertEqual(model.model.intercept_, model_check.intercept_)
        # Check that the TimeWeightedLinearRegression fit is equivalent to using LR with sample weights
        model = TimeWeightedLinearRegression(half_life = 21, fit_intercept=False, positive=True)
        model.fit(self.X, self.y)
        self.assertIsInstance(model.model, LinearRegression)
        model_check = LinearRegression(fit_intercept=False, positive=True)
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        self.assertTrue(np.all(model.model.coef_ == model_check.coef_))
        self.assertEqual(model.model.intercept_, model_check.intercept_)
        # check fit with dataframe targets
        model_check.fit(self.X, self.y.to_frame(), sample_weight=model.sample_weights)
        self.assertTrue(np.all(model.model.coef_ == model_check.coef_))
        self.assertEqual(model.model.intercept_, model_check.intercept_)

    def test_types_fit(self):
        # Test that non dataframe X returns a TypeError
        with self.assertRaises(TypeError):
            model = TimeWeightedLinearRegression()
            model.fit(self.X.values, self.y)
        # Test that non dataframe and non series y returns a TypeError
        with self.assertRaises(TypeError):
            model = TimeWeightedLinearRegression()
            model.fit(self.X, self.y.values)
        # Test that a ValueError is raised if y is a dataframe with more than one column
        with self.assertRaises(ValueError):
            model = TimeWeightedLinearRegression()
            model.fit(self.X, self.X)
        # Test that a ValueError is raised if X and y are not multi-indexed
        with self.assertRaises(ValueError):
            model = TimeWeightedLinearRegression()
            model.fit(self.X.reset_index(), self.y)
        with self.assertRaises(ValueError):
            model = TimeWeightedLinearRegression()
            model.fit(self.X, self.y.reset_index())
        # Test that a ValueError is raised if the multi-indices of X and y don't match
        with self.assertRaises(ValueError):
            model = TimeWeightedLinearRegression()
            model.fit(self.X, self.y.iloc[1:])

    def test_valid_predict(self):
        # Check that the TimeWeightedLinearRegression predict is equivalent to using LR with sample weights
        model = TimeWeightedLinearRegression()
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        model_check = LinearRegression()
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        y_pred_check = model_check.predict(self.X)
        self.assertTrue(np.all(y_pred == y_pred_check))
        # Check that the TimeWeightedLinearRegression predict is equivalent to using restricted LR with sample weights
        model = TimeWeightedLinearRegression(fit_intercept=False, positive=True)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        model_check = LinearRegression(fit_intercept=False, positive=True)
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        y_pred_check = model_check.predict(self.X)
        self.assertTrue(np.all(y_pred == y_pred_check))

    def test_types_predict(self):
        # Test that non dataframe X returns a TypeError
        with self.assertRaises(TypeError):
            model = TimeWeightedLinearRegression()
            model.fit(self.X, self.y)
            model.predict(self.X.values)
        # Test that a ValueError is raised if X is not multi-indexed
        with self.assertRaises(ValueError):
            model = TimeWeightedLinearRegression()
            model.fit(self.X, self.y)
            model.predict(self.X.reset_index())

    def test_panelcvscores_compatible(self):
        # Test that the TimeWeightedLinearRegression is compatible with panel_cv_scores
        splitter = RollingKFoldPanelSplit(n_splits=5)
        estimators = {"SWOLS": TimeWeightedLinearRegression(), "OLS": LinearRegression()}
        scoring = {"NEG_RMSE": make_scorer(mean_squared_error, squared=False, greater_is_better=False), "NEG_MAE": make_scorer(mean_absolute_error, greater_is_better=False)}
        try:
            cv_df = panel_cv_scores(
                X = self.X,
                y = self.y,
                splitter = splitter,
                estimators = estimators,
                scoring = scoring,
                n_jobs = 1
            )
        except Exception as e:
            self.fail(f"panel_cv_scores raised an exception {e} when using TimeWeightedLinearRegression as an estimator.")
        self.assertIsInstance(cv_df, pd.DataFrame)
        self.assertEqual(cv_df.shape[1], 2)
        self.assertEqual(sorted(cv_df.columns), sorted(estimators.keys()))
        self.assertEqual(sorted(cv_df.index)[:2], sorted(scoring.keys())[:2])

    def test_signaloptimizer_compatible(self):
        # Test that the TimeWeightedLinearRegression is compatible with SignalOptimizer
        inner_splitter = RollingKFoldPanelSplit(n_splits=5)
        models = {
            "SWOLS": TimeWeightedLinearRegression(),
            "OLS": LinearRegression(),
        }
        metric = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
        hparam_grid = {
            "SWOLS": {"fit_intercept": [True, False], "half_life": [21, 42, 100, 200]},
            "OLS": {"fit_intercept": [True, False]},
        }
        so = SignalOptimizer(
            inner_splitter=inner_splitter,
            X = self.X,
            y = self.y,
            blacklist=None,
        )
        try:
            so.calculate_predictions(
                name="test",
                models=models,
                hparam_grid=hparam_grid,
                metric=metric,
            )
        except Exception as e:
            self.fail(f"SignalOptimizer raised an exception {e} when using TimeWeightedLinearRegression as an estimator.")

        df_sigs = so.get_optimized_signals(name="test")
        df_models = so.get_optimal_models(name="test")

        self.assertIsInstance(df_sigs, pd.DataFrame)
        self.assertEqual(df_sigs.shape[1], 4)
        self.assertEqual(sorted(df_sigs.columns), ["cid", "real_date", "value", "xcat"])
        self.assertTrue(len(df_sigs.xcat.unique()) == 1)
        self.assertEqual(df_sigs.xcat.unique()[0], "test")

        self.assertIsInstance(df_sigs, pd.DataFrame)
        self.assertEqual(df_models.shape[1], 4)
        self.assertEqual(sorted(df_models.columns), ["hparams", "model_type", "name", "real_date"])
        self.assertTrue(len(df_models.name.unique()) == 1)
        self.assertEqual(df_models.name.unique()[0], "test") 