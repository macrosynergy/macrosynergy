import os
import numbers
import numpy as np
import pandas as pd

import unittest

from macrosynergy.learning import (
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
)

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from parameterized import parameterized

class TestSignWeightedLinearRegression(unittest.TestCase):
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
        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y = df["XR"]
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

        self.sign_weights = SignWeightedLinearRegression()._calculate_sign_weights(self.y)

    def test_types_init(self):
        # fit_intercept
        self.assertRaises(TypeError, SignWeightedLinearRegression, fit_intercept="True")
        self.assertRaises(TypeError, SignWeightedLinearRegression, fit_intercept=1)
        # positive 
        self.assertRaises(TypeError, SignWeightedLinearRegression, positive="True")
        self.assertRaises(TypeError, SignWeightedLinearRegression, positive=1)
        # alpha
        self.assertRaises(TypeError, SignWeightedLinearRegression, alpha="1")
        self.assertRaises(TypeError, SignWeightedLinearRegression, alpha=True)
        self.assertRaises(ValueError, SignWeightedLinearRegression, alpha=-1)
        # shrinkage_type
        self.assertRaises(TypeError, SignWeightedLinearRegression, shrinkage_type=1)
        self.assertRaises(ValueError, SignWeightedLinearRegression, shrinkage_type="l3")

    def test_valid_init(self):
        # Check defaults set correctly
        ls = SignWeightedLinearRegression()
        self.assertEqual(ls.fit_intercept, True)
        self.assertEqual(ls.positive, False)
        self.assertEqual(ls.alpha, 0)
        self.assertEqual(ls.shrinkage_type, "l1")

        # Change defaults
        ls = SignWeightedLinearRegression(
            fit_intercept=False,
            positive=True,
            alpha=0.1,
            shrinkage_type="l2",
        )
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.1)
        self.assertEqual(ls.shrinkage_type, "l2")

    def test_types_fit(self):
        # X
        ls = SignWeightedLinearRegression()
        self.assertRaises(TypeError, ls.fit, X=1, y=self.y)
        self.assertRaises(TypeError, ls.fit, X="X", y=self.y)
        self.assertRaises(ValueError, ls.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, ls.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, ls.fit, X=self.X_nan.values, y=self.y)
        # y
        self.assertRaises(TypeError, ls.fit, X=self.X, y=1)
        self.assertRaises(TypeError, ls.fit, X=self.X, y="y")
        self.assertRaises(ValueError, ls.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, ls.fit, X=self.X, y=self.y_nan)
        self.assertRaises(ValueError, ls.fit, X=self.X, y=self.y_nan.values)

    def test_valid_fit(self):
        """Check default SignWeightedLinearRegression fit method works as expected"""
        ls = SignWeightedLinearRegression(fit_intercept = True)
        try:
            ls.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Default SignWeightedLinearRegression fit method failed with exception: {e}")
        self.assertTrue(isinstance(ls.coef_, np.ndarray))
        self.assertTrue(len(ls.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(ls.intercept_, numbers.Number))
        self.assertTrue(ls.intercept_ != 0)
        # Check the solution is close to LinearRegression from scikit-learn
        qr = LinearRegression(fit_intercept=True).fit(self.X, self.y, sample_weight=self.sign_weights)
        np.testing.assert_almost_equal(ls.coef_, qr.coef_,decimal=2)
        np.testing.assert_almost_equal(ls.intercept_, qr.intercept_,decimal=2)

        """Check default SignWeightedLinearRegression fit method works as expected with numpy arrays"""
        ls = SignWeightedLinearRegression(fit_intercept = True)
        try:
            ls.fit(X=self.X.values, y=self.y.values)
        except Exception as e:
            self.fail(f"Default SignWeightedLinearRegression fit method failed with exception: {e}")
        self.assertTrue(isinstance(ls.coef_, np.ndarray))
        self.assertTrue(len(ls.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(ls.intercept_, numbers.Number))
        self.assertTrue(ls.intercept_ != 0)
        # Check the solution is close to LinearRegression from scikit-learn
        qr = LinearRegression(fit_intercept=True).fit(self.X.values, self.y.values, sample_weight=self.sign_weights)
        np.testing.assert_almost_equal(ls.coef_, qr.coef_,decimal=2)
        np.testing.assert_almost_equal(ls.intercept_, qr.intercept_,decimal=2)

        """Check no intercept SignWeightedLinearRegression fit method works as expected"""
        ls = SignWeightedLinearRegression(fit_intercept = False)
        try:
            ls.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"No intercept SignWeightedLinearRegression fit method failed with exception: {e}")
        self.assertTrue(isinstance(ls.coef_, np.ndarray))
        self.assertTrue(len(ls.coef_) == self.X.shape[1])
        self.assertTrue(ls.intercept_ == 0)
        qr = LinearRegression(fit_intercept=False).fit(self.X, self.y, sample_weight=self.sign_weights)
        np.testing.assert_almost_equal(ls.coef_, qr.coef_,decimal=2)
        np.testing.assert_almost_equal(ls.intercept_, qr.intercept_,decimal=2)

        """Check L1 regularization SignWeightedLinearRegression fit method works as expected"""
        ls1 = SignWeightedLinearRegression(fit_intercept = True, alpha = 1, shrinkage_type = "l1")
        try:
            ls1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"L1 regularization SignWeightedLinearRegression, alpha = 1, fit method failed with exception: {e}")
        self.assertTrue(isinstance(ls1.coef_, np.ndarray))
        self.assertTrue(len(ls1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(ls1.intercept_, numbers.Number))
        self.assertTrue(ls1.intercept_ != 0)
        qr = Lasso(alpha = 1, fit_intercept=True).fit(self.X, self.y, sample_weight=self.sign_weights)
        np.testing.assert_almost_equal(ls1.coef_, qr.coef_,decimal=2)

        """Check L2 regularization SignWeightedLinearRegression fit method works as expected"""
        ls1 = SignWeightedLinearRegression(fit_intercept = True, alpha = 1, shrinkage_type = "l2")
        try:
            ls1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"L2 regularization SignWeightedLinearRegression, alpha = 1, fit method failed with exception: {e}")
        self.assertTrue(isinstance(ls1.coef_, np.ndarray))
        self.assertTrue(len(ls1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(ls1.intercept_, numbers.Number))
        self.assertTrue(ls1.intercept_ != 0)
        qr = Ridge(alpha = 1, fit_intercept=True).fit(self.X, self.y, sample_weight=self.sign_weights)
        np.testing.assert_almost_equal(ls1.coef_, qr.coef_,decimal=2)

    def test_types_predict(self):
        ls = SignWeightedLinearRegression().fit(self.X, self.y)
        self.assertRaises(TypeError, ls.predict, X=1)
        self.assertRaises(TypeError, ls.predict, X="X")
        self.assertRaises(ValueError, ls.predict, X=self.X.iloc[:,:-1])
        self.assertRaises(ValueError, ls.predict, X=self.X_nan)
        self.assertRaises(ValueError, ls.predict, X=self.X_nan.values)

    def test_valid_predict(self):
        """Check default SignWeightedLinearRegression predict method works as expected"""
        ls = SignWeightedLinearRegression().fit(self.X, self.y)
        try:
            pred = ls.predict(self.X)
        except Exception as e:
            self.fail(f"Default SignWeightedLinearRegression predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])
        ls = SignWeightedLinearRegression().fit(self.X.values, self.y.values)
        try:
            pred = ls.predict(self.X)
        except Exception as e:
            self.fail(f"Default SignWeightedLinearRegression predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L1 regularization SignWeightedLinearRegression predict method works as expected"""
        ls = SignWeightedLinearRegression(alpha = 1, shrinkage_type = "l1").fit(self.X, self.y)
        try:
            pred = ls.predict(self.X)
        except Exception as e:
            self.fail(f"L1 regularization SignWeightedLinearRegression predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])
        ls = SignWeightedLinearRegression(alpha = 1, shrinkage_type = "l1").fit(self.X.values, self.y.values)
        try:
            pred = ls.predict(self.X)
        except Exception as e:
            self.fail(f"L1 regularization SignWeightedLinearRegression predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L2 regularization SignWeightedLinearRegression predict method works as expected"""
        ls = SignWeightedLinearRegression(alpha = 1, shrinkage_type = "l2").fit(self.X, self.y)
        try:
            pred = ls.predict(self.X)
        except Exception as e:
            self.fail(f"L2 regularization SignWeightedLinearRegression predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])
        ls = SignWeightedLinearRegression(alpha = 1, shrinkage_type = "l2").fit(self.X.values, self.y.values)
        try:
            pred = ls.predict(self.X)
        except Exception as e:
            self.fail(f"L2 regularization SignWeightedLinearRegression predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

    def test_valid_set_params(self):
        ls = SignWeightedLinearRegression()
        ls.set_params(fit_intercept = False, positive = True, alpha = 0.1, shrinkage_type = "l2")
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.1)
        self.assertEqual(ls.shrinkage_type, "l2")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)
        self.assertEqual(ls.model.alpha, 0.1)

        ls = SignWeightedLinearRegression()
        ls.set_params(fit_intercept = False, positive = True, alpha = 0.2, shrinkage_type = "l1")
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.2)
        self.assertEqual(ls.shrinkage_type, "l1")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)
        self.assertEqual(ls.model.alpha, 0.2)

        ls = SignWeightedLinearRegression()
        ls.set_params(fit_intercept = False, positive = True, alpha = 0, shrinkage_type = "l2")
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0)
        self.assertEqual(ls.shrinkage_type, "l2")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)

class TestTimeWeightedLinearRegression(unittest.TestCase):
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
        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y = df["XR"]
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

        self.time_weights = TimeWeightedLinearRegression(half_life=24)._calculate_time_weights(self.y)

    def test_types_init(self):
        # fit_intercept
        self.assertRaises(TypeError, TimeWeightedLinearRegression, fit_intercept="True")
        self.assertRaises(TypeError, TimeWeightedLinearRegression, fit_intercept=1)
        # positive 
        self.assertRaises(TypeError, TimeWeightedLinearRegression, positive="True")
        self.assertRaises(TypeError, TimeWeightedLinearRegression, positive=1)
        # alpha
        self.assertRaises(TypeError, TimeWeightedLinearRegression, alpha="1")
        self.assertRaises(TypeError, TimeWeightedLinearRegression, alpha=True)
        self.assertRaises(ValueError, TimeWeightedLinearRegression, alpha=-1)
        # shrinkage_type
        self.assertRaises(TypeError, TimeWeightedLinearRegression, shrinkage_type=1)
        self.assertRaises(ValueError, TimeWeightedLinearRegression, shrinkage_type="l3")
        # half_life
        self.assertRaises(TypeError, TimeWeightedLinearRegression, half_life="1")
        self.assertRaises(TypeError, TimeWeightedLinearRegression, half_life=True)

    def test_valid_init(self):
        # Check defaults set correctly
        ls = TimeWeightedLinearRegression()
        self.assertEqual(ls.fit_intercept, True)
        self.assertEqual(ls.positive, False)
        self.assertEqual(ls.alpha, 0)
        self.assertEqual(ls.shrinkage_type, "l1")
        self.assertEqual(ls.half_life, 252)

        # Change defaults
        ls = TimeWeightedLinearRegression(
            fit_intercept=False,
            positive=True,
            alpha=0.1,
            shrinkage_type="l2",
            half_life=24
        )
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.1)
        self.assertEqual(ls.shrinkage_type, "l2")
        self.assertEqual(ls.half_life, 24)

    def test_types_fit(self):
        # X
        ls = TimeWeightedLinearRegression()
        self.assertRaises(TypeError, ls.fit, X=1, y=self.y)
        self.assertRaises(TypeError, ls.fit, X="X", y=self.y)
        self.assertRaises(TypeError, ls.fit, X=self.X.values, y=self.y)
        self.assertRaises(ValueError, ls.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, ls.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, ls.fit, X=self.X, y=1)
        self.assertRaises(TypeError, ls.fit, X=self.X, y="y")
        self.assertRaises(TypeError, ls.fit, X=self.X, y=self.y.values)
        self.assertRaises(ValueError, ls.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, ls.fit, X=self.X, y=self.y_nan)

    def test_valid_fit(self):
        """Check default TimeWeightedLinearRegression fit method works as expected"""
        ls = TimeWeightedLinearRegression(fit_intercept = True, half_life=24)
        try:
            ls.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Default TimeWeightedLinearRegression fit method failed with exception: {e}")
        self.assertTrue(isinstance(ls.coef_, np.ndarray))
        self.assertTrue(len(ls.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(ls.intercept_, numbers.Number))
        self.assertTrue(ls.intercept_ != 0)
        # Check the solution is close to LinearRegression from scikit-learn
        qr = LinearRegression(fit_intercept=True).fit(self.X, self.y, sample_weight=self.time_weights)
        np.testing.assert_almost_equal(ls.coef_, qr.coef_,decimal=2)
        np.testing.assert_almost_equal(ls.intercept_, qr.intercept_,decimal=2)

        """Check no intercept TimeWeightedLinearRegression fit method works as expected"""
        ls = TimeWeightedLinearRegression(fit_intercept = False, half_life = 24)
        try:
            ls.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"No intercept TimeWeightedLinearRegression fit method failed with exception: {e}")
        self.assertTrue(isinstance(ls.coef_, np.ndarray))
        self.assertTrue(len(ls.coef_) == self.X.shape[1])
        self.assertTrue(ls.intercept_ == 0)
        qr = LinearRegression(fit_intercept=False).fit(self.X, self.y, sample_weight=self.time_weights)
        np.testing.assert_almost_equal(ls.coef_, qr.coef_,decimal=2)
        np.testing.assert_almost_equal(ls.intercept_, qr.intercept_,decimal=2)

        """Check L1 regularization TimeWeightedLinearRegression fit method works as expected"""
        ls1 = TimeWeightedLinearRegression(fit_intercept = True, alpha = 1, shrinkage_type = "l1", half_life = 24)
        try:
            ls1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"L1 regularization TimeWeightedLinearRegression, alpha = 1, fit method failed with exception: {e}")
        self.assertTrue(isinstance(ls1.coef_, np.ndarray))
        self.assertTrue(len(ls1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(ls1.intercept_, numbers.Number))
        self.assertTrue(ls1.intercept_ != 0)
        qr = Lasso(alpha = 1, fit_intercept=True).fit(self.X, self.y, sample_weight=self.time_weights)
        np.testing.assert_almost_equal(ls1.coef_, qr.coef_,decimal=2)

        """Check L2 regularization TimeWeightedLinearRegression fit method works as expected"""
        ls1 = TimeWeightedLinearRegression(fit_intercept = True, alpha = 1, shrinkage_type = "l2", half_life = 24)
        try:
            ls1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"L2 regularization TimeWeightedLinearRegression, alpha = 1, fit method failed with exception: {e}")
        self.assertTrue(isinstance(ls1.coef_, np.ndarray))
        self.assertTrue(len(ls1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(ls1.intercept_, numbers.Number))
        self.assertTrue(ls1.intercept_ != 0)
        qr = Ridge(alpha = 1, fit_intercept=True).fit(self.X, self.y, sample_weight=self.time_weights)
        np.testing.assert_almost_equal(ls1.coef_, qr.coef_,decimal=2)

    def test_types_predict(self):
        ls = TimeWeightedLinearRegression().fit(self.X, self.y)
        self.assertRaises(TypeError, ls.predict, X=1)
        self.assertRaises(TypeError, ls.predict, X="X")
        self.assertRaises(TypeError, ls.predict, X=self.X.values)
        self.assertRaises(ValueError, ls.predict, X=self.X.iloc[:,:-1])
        self.assertRaises(ValueError, ls.predict, X=self.X_nan)

    def test_valid_predict(self):
        """Check default TimeWeightedLinearRegression predict method works as expected"""
        ls = TimeWeightedLinearRegression().fit(self.X, self.y)
        try:
            pred = ls.predict(self.X)
        except Exception as e:
            self.fail(f"Default TimeWeightedLinearRegression predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L1 regularization TimeWeightedLinearRegression predict method works as expected"""
        ls = TimeWeightedLinearRegression(alpha = 1, shrinkage_type = "l1").fit(self.X, self.y)
        try:
            pred = ls.predict(self.X)
        except Exception as e:
            self.fail(f"L1 regularization TimeWeightedLinearRegression predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L2 regularization TimeWeightedLinearRegression predict method works as expected"""
        ls = TimeWeightedLinearRegression(alpha = 1, shrinkage_type = "l2").fit(self.X, self.y)
        try:
            pred = ls.predict(self.X)
        except Exception as e:
            self.fail(f"L2 regularization TimeWeightedLinearRegression predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

    def test_valid_set_params(self):
        ls = TimeWeightedLinearRegression()
        ls.set_params(fit_intercept = False, positive = True, alpha = 0.1, half_life = 3, shrinkage_type = "l2")
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.1)
        self.assertEqual(ls.half_life, 3)
        self.assertEqual(ls.shrinkage_type, "l2")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)
        self.assertEqual(ls.model.alpha, 0.1)

        ls = TimeWeightedLinearRegression()
        ls.set_params(fit_intercept = False, positive = True, half_life = 6, alpha = 0.2, shrinkage_type = "l1")
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.2)
        self.assertEqual(ls.half_life, 6)
        self.assertEqual(ls.shrinkage_type, "l1")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)
        self.assertEqual(ls.model.alpha, 0.2)

        ls = TimeWeightedLinearRegression()
        ls.set_params(fit_intercept = False, positive = True, half_life = 12, alpha = 0, shrinkage_type = "l2")
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0)
        self.assertEqual(ls.half_life, 12)
        self.assertEqual(ls.shrinkage_type, "l2")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)