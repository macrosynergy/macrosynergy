import numpy as np
import pandas as pd
from scipy.stats import expon 

import unittest

from macrosynergy.learning import (
    NaivePredictor,
    SignWeightedRegressor,
    TimeWeightedRegressor,
    LassoSelector,
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    LogisticRegression,
)


class TestSWRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2013-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2010-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2010-01-01", "2020-12-31"]

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
            model = SignWeightedRegressor(LinearRegression)
        except Exception as e:
            self.fail(
                "SignWeightedRegressor constructor with a LinearRegression object raised an exception: {}".format(
                    e
                )
            )
        self.assertIsInstance(model, SignWeightedRegressor)
        self.assertIsInstance(model.model, LinearRegression)

        # Test that a sign weighted ridge model is successfully instantiated
        try:
            model = SignWeightedRegressor(Ridge, fit_intercept=False, alpha=0.1)
        except Exception as e:
            self.fail(
                "SignWeightedRegressor constructor with a Ridge object raised an exception: {}".format(
                    e
                )
            )
        self.assertIsInstance(model, SignWeightedRegressor)
        self.assertIsInstance(model.model, Ridge)
        self.assertEqual(model.model.fit_intercept, False)
        self.assertEqual(model.model.alpha, 0.1)

    def test_types_init(self):
        # Check that incorrectly specified arguments raise exceptions
        with self.assertRaises(TypeError):
            model = SignWeightedRegressor(LinearRegression())
        with self.assertRaises(ValueError):
            model = SignWeightedRegressor(NaivePredictor)
        # check that a TypeError is raised if regressor doesn't inherit from RegressorMixin
        with self.assertRaises(TypeError):
            model = SignWeightedRegressor(LassoSelector, alpha=1)

    def test_valid_weightsfunc(self):
        # The weights function is designed to return the same weights that a classifier with
        # balanced class weights would return in scikit-learn. Test that this is the case.
        model = SignWeightedRegressor(LinearRegression)
        sample_weights, pos_weight, neg_weight = model._SignWeightedRegressor__calculate_sample_weights(
            self.y
        )
        correct_pos_weight = len(self.y) / (2 * np.sum(self.y >= 0))
        correct_neg_weight = len(self.y) / (2 * np.sum(self.y < 0))
        self.assertEqual(pos_weight, correct_pos_weight)
        self.assertEqual(neg_weight, correct_neg_weight)

    def test_valid_fit(self):
        # Check that the SignWeightedRegressor fit is equivalent to using LR with sample weights
        model = SignWeightedRegressor(LinearRegression)
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
        model = SignWeightedRegressor(Ridge, fit_intercept=False, alpha=0.1)
        model.fit(self.X, self.y)
        self.assertIsInstance(model.model, Ridge)
        model_check = Ridge(fit_intercept=False, alpha=0.1)
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
            model = SignWeightedRegressor(LinearRegression)
            model.fit(self.X.values, self.y)
        # Test that non dataframe and non series y returns a TypeError
        with self.assertRaises(TypeError):
            model = SignWeightedRegressor(LinearRegression)
            model.fit(self.X, self.y.values)
        # Test that a ValueError is raised if y is a dataframe with more than one column
        with self.assertRaises(ValueError):
            model = SignWeightedRegressor(LinearRegression)
            model.fit(self.X, self.X)
        # Test that a ValueError is raised if X and y are not multi-indexed
        with self.assertRaises(ValueError):
            model = SignWeightedRegressor(LinearRegression)
            model.fit(self.X.reset_index(), self.y)
        with self.assertRaises(ValueError):
            model = SignWeightedRegressor(LinearRegression)
            model.fit(self.X, self.y.reset_index())
        # Test that a ValueError is raised if the multi-indices of X and y don't match
        with self.assertRaises(ValueError):
            model = SignWeightedRegressor(LinearRegression)
            model.fit(self.X, self.y.iloc[1:])

    def test_valid_predict(self):
        # Check that the SignWeightedRegressor predict is equivalent to using LR with sample weights
        model = SignWeightedRegressor(LinearRegression)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        model_check = LinearRegression()
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        y_pred_check = model_check.predict(self.X)
        self.assertTrue(np.all(y_pred == y_pred_check))
        # Check that the SignWeightedRegressor predict is equivalent to using Ridge with sample weights
        model = SignWeightedRegressor(Ridge, fit_intercept=False, alpha=0.1)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        model_check = Ridge(fit_intercept=False, alpha=0.1)
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        y_pred_check = model_check.predict(self.X)
        self.assertTrue(np.all(y_pred == y_pred_check))

    def test_types_predict(self):
        # Test that non dataframe X returns a TypeError
        with self.assertRaises(TypeError):
            model = SignWeightedRegressor(LinearRegression)
            model.fit(self.X, self.y)
            model.predict(self.X.values)
        # Test that a ValueError is raised if X is not multi-indexed
        with self.assertRaises(ValueError):
            model = SignWeightedRegressor(LinearRegression)
            model.fit(self.X, self.y)
            model.predict(self.X.reset_index())

class TestTWRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2013-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2010-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2010-01-01", "2020-12-31"]

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
            model = TimeWeightedRegressor(LinearRegression, half_life=21)
        except Exception as e:
            self.fail(
                "TimeWeightedRegressor constructor with a LinearRegression object raised an exception: {}".format(
                    e
                )
            )
        self.assertIsInstance(model, TimeWeightedRegressor)
        self.assertIsInstance(model.model, LinearRegression)
        self.assertEqual(model.half_life, 21)

        # Test that a time weighted ridge model is successfully instantiated
        try:
            model = TimeWeightedRegressor(Ridge, half_life=21*12, fit_intercept=False, alpha=0.1)
        except Exception as e:
            self.fail(
                "TimeWeightedRegressor constructor with a Ridge object raised an exception: {}".format(
                    e
                )
            )
        self.assertIsInstance(model, TimeWeightedRegressor)
        self.assertIsInstance(model.model, Ridge)
        self.assertEqual(model.model.fit_intercept, False)
        self.assertEqual(model.model.alpha, 0.1)
        self.assertEqual(model.half_life, 21*12)

    def test_types_init(self):
        with self.assertRaises(TypeError):
            model = TimeWeightedRegressor(LinearRegression(), half_life=21)
        with self.assertRaises(ValueError):
            model = TimeWeightedRegressor(NaivePredictor, half_life=21)
        with self.assertRaises(TypeError):
            model = TimeWeightedRegressor(LassoSelector, half_life=21, alpha=1)
        with self.assertRaises(TypeError):
            model = TimeWeightedRegressor(LinearRegression, half_life="str")
        with self.assertRaises(ValueError):
            model = TimeWeightedRegressor(LinearRegression, half_life=-1)

    def test_valid_weightsfunc(self):
        model = TimeWeightedRegressor(LinearRegression, half_life=21)
        sample_weights = model._TimeWeightedRegressor__calculate_sample_weights(self.y)
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
        # Check that the TimeWeightedRegressor fit is equivalent to using LR with sample weights
        model = TimeWeightedRegressor(LinearRegression, half_life=21)
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
        # Check that the TimeWeightedRegressor fit is equivalent to using Ridge with sample weights
        model = TimeWeightedRegressor(Ridge, half_life=21, fit_intercept=False, alpha=0.1)
        model.fit(self.X, self.y)
        self.assertIsInstance(model.model, Ridge)
        model_check = Ridge(fit_intercept=False, alpha=0.1)
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
            model = TimeWeightedRegressor(LinearRegression, half_life=21)
            model.fit(self.X.values, self.y)
        # Test that non dataframe and non series y returns a TypeError
        with self.assertRaises(TypeError):
            model = TimeWeightedRegressor(LinearRegression, half_life=21)
            model.fit(self.X, self.y.values)
        # Test that a ValueError is raised if y is a dataframe with more than one column
        with self.assertRaises(ValueError):
            model = TimeWeightedRegressor(LinearRegression, half_life=21)
            model.fit(self.X, self.X)
        # Test that a ValueError is raised if X and y are not multi-indexed
        with self.assertRaises(ValueError):
            model = TimeWeightedRegressor(LinearRegression, half_life=21)
            model.fit(self.X.reset_index(), self.y)
        with self.assertRaises(ValueError):
            model = TimeWeightedRegressor(LinearRegression, half_life=21)
            model.fit(self.X, self.y.reset_index())
        # Test that a ValueError is raised if the multi-indices of X and y don't match
        with self.assertRaises(ValueError):
            model = TimeWeightedRegressor(LinearRegression, half_life=21)
            model.fit(self.X, self.y.iloc[1:])

    def test_valid_predict(self):
        # Check that the TimeWeightedRegressor predict is equivalent to using LR with sample weights
        model = TimeWeightedRegressor(LinearRegression, half_life=21)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        model_check = LinearRegression()
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        y_pred_check = model_check.predict(self.X)
        self.assertTrue(np.all(y_pred == y_pred_check))
        # Check that the TimeWeightedRegressor predict is equivalent to using Ridge with sample weights
        model = TimeWeightedRegressor(Ridge, half_life=21, fit_intercept=False, alpha=0.1)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        model_check = Ridge(fit_intercept=False, alpha=0.1)
        model_check.fit(self.X, self.y, sample_weight=model.sample_weights)
        y_pred_check = model_check.predict(self.X)
        self.assertTrue(np.all(y_pred == y_pred_check))

    def test_types_predict(self):
        # Test that non dataframe X returns a TypeError
        with self.assertRaises(TypeError):
            model = TimeWeightedRegressor(LinearRegression,half_life=21)
            model.fit(self.X, self.y)
            model.predict(self.X.values)
        # Test that a ValueError is raised if X is not multi-indexed
        with self.assertRaises(ValueError):
            model = TimeWeightedRegressor(LinearRegression, half_life=21)
            model.fit(self.X, self.y)
            model.predict(self.X.reset_index())