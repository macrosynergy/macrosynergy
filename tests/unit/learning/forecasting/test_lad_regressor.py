import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from macrosynergy import PYTHON_3_9_OR_LATER
from macrosynergy.learning import (
    LADRegressor,
    SignWeightedLADRegressor,
    TimeWeightedLADRegressor,
)


class TestLADRegressor(unittest.TestCase):
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
        self.X_numpy = self.X.values
        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y = df["XR"]
        self.y_numpy = self.y.values
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

    def test_types_init(self):
        # fit_intercept
        self.assertRaises(TypeError, LADRegressor, fit_intercept=1)
        self.assertRaises(TypeError, LADRegressor, fit_intercept="True")
        # positive
        self.assertRaises(TypeError, LADRegressor, positive=1)
        self.assertRaises(TypeError, LADRegressor, positive="True")
        # alpha
        self.assertRaises(TypeError, LADRegressor, alpha="1")
        self.assertRaises(TypeError, LADRegressor, alpha=True)
        self.assertRaises(ValueError, LADRegressor, alpha=-1)
        # shrinkage_type
        self.assertRaises(TypeError, LADRegressor, shrinkage_type=1)
        self.assertRaises(TypeError, LADRegressor, shrinkage_type=True)
        self.assertRaises(ValueError, LADRegressor, shrinkage_type="l3")
        self.assertRaises(ValueError, LADRegressor, shrinkage_type="string")
        # tol
        self.assertRaises(TypeError, LADRegressor, tol="1")
        self.assertRaises(TypeError, LADRegressor, tol=True)
        self.assertRaises(ValueError, LADRegressor, tol=-1)
        # max_iter
        self.assertRaises(TypeError, LADRegressor, maxiter="1")
        self.assertRaises(TypeError, LADRegressor, maxiter=True)
        self.assertRaises(ValueError, LADRegressor, maxiter=-1)

    def test_valid_init(self):
        # Check defaults set correctly
        lad = LADRegressor()
        self.assertEqual(lad.fit_intercept, True)
        self.assertEqual(lad.positive, False)
        self.assertEqual(lad.alpha, 0)
        self.assertEqual(lad.shrinkage_type, "l1")
        self.assertEqual(lad.tol, None)
        self.assertEqual(lad.maxiter, None)

        # Change defaults
        lad = LADRegressor(
            fit_intercept=False,
            positive=True,
            alpha=0.1,
            shrinkage_type="l2",
            tol=0.1,
            maxiter=100,
        )
        self.assertEqual(lad.fit_intercept, False)
        self.assertEqual(lad.positive, True)
        self.assertEqual(lad.alpha, 0.1)
        self.assertEqual(lad.shrinkage_type, "l2")
        self.assertEqual(lad.tol, 0.1)
        self.assertEqual(lad.maxiter, 100)

    def test_types_fit(self):
        # X - when a dataframe
        lad = LADRegressor()
        self.assertRaises(TypeError, lad.fit, X=1, y=self.y)
        self.assertRaises(TypeError, lad.fit, X="X", y=self.y)
        self.assertRaises(ValueError, lad.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, lad.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, lad.fit, X=self.X_nan.values, y=self.y)
        # X - when a numpy array
        self.assertRaises(ValueError, lad.fit, X=self.X.reset_index().values, y=self.y)
        self.assertRaises(
            ValueError, lad.fit, X=self.X_nan.reset_index(drop=True).values, y=self.y
        )
        # y - when a series
        self.assertRaises(TypeError, lad.fit, X=self.X, y=1)
        self.assertRaises(TypeError, lad.fit, X=self.X, y="y")
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y.reset_index()["cid"])
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y_nan)
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y_nan.values)
        # y - when a dataframe
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(
            ValueError, lad.fit, X=self.X, y=pd.DataFrame(self.y.reset_index()["cid"])
        )
        self.assertRaises(
            ValueError,
            lad.fit,
            X=self.X,
            y=pd.DataFrame(self.y_nan.reset_index(drop=True)),
        )
        # y - when a numpy array
        self.assertRaises(
            ValueError, lad.fit, X=self.X.values, y=np.zeros((len(self.X), 2))
        )
        self.assertRaises(
            ValueError, lad.fit, X=self.X.values, y=np.array(["hello"] * len(self.X))
        )
        self.assertRaises(
            ValueError, lad.fit, X=self.X.values, y=np.array([np.nan] * len(self.X))
        )

        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y[:-1])

        # sample_weight
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y, sample_weight=1)
        self.assertRaises(
            ValueError,
            lad.fit,
            X=self.X,
            y=self.y,
            sample_weight=np.ones((len(self.X), 2), dtype=int),
        )
        self.assertRaises(
            ValueError,
            lad.fit,
            X=self.X,
            y=self.y,
            sample_weight=np.array(["hello"] * len(self.X)),
        )
        self.assertRaises(
            ValueError,
            lad.fit,
            X=self.X,
            y=self.y,
            sample_weight=np.array([1] * (len(self.X) - 1)),
        )

    def test_valid_fit(self):
        """Check default LADRegressor fit method works as expected"""
        lad = LADRegressor(fit_intercept=True)
        try:
            lad.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Default LADRegressor fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad.intercept_, float))
        self.assertTrue(lad.intercept_ != 0)
        if PYTHON_3_9_OR_LATER:
            # Check the solution is close to QuantileRegressor from scikit-learn
            qr = QuantileRegressor(alpha=0, fit_intercept=True).fit(self.X, self.y)
            self.assertTrue(np.allclose(lad.coef_, qr.coef_, atol=0.01, rtol=0.01))
            np.testing.assert_almost_equal(lad.intercept_, qr.intercept_, decimal=2)

        """Check default LADRegressor fit method works as expected with numpy arrays"""
        lad = LADRegressor(fit_intercept=True)
        try:
            lad.fit(X=self.X.values, y=self.y.values)
        except Exception as e:
            self.fail(f"Default LADRegressor fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad.intercept_, float))
        self.assertTrue(lad.intercept_ != 0)
        if PYTHON_3_9_OR_LATER:
            # Check the solution is close to QuantileRegressor from scikit-learn
            qr = QuantileRegressor(alpha=0, fit_intercept=True).fit(
                self.X.values, self.y.values
            )
            self.assertTrue(np.allclose(lad.coef_, qr.coef_, atol=0.01, rtol=0.01))
            np.testing.assert_almost_equal(lad.intercept_, qr.intercept_, decimal=2)

        """Check no intercept LADRegressor fit method works as expected"""
        lad = LADRegressor(fit_intercept=False)
        try:
            lad.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"No intercept LADRegressor fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(lad.intercept_ == 0)
        if PYTHON_3_9_OR_LATER:
            qr = QuantileRegressor(alpha=0, fit_intercept=False).fit(self.X, self.y)
            self.assertTrue(np.allclose(lad.coef_, qr.coef_, atol=0.01, rtol=0.01))
            np.testing.assert_almost_equal(lad.intercept_, qr.intercept_, decimal=2)

        """Check positive restriction LADRegressor fit method works as expected"""
        lad1 = LADRegressor(fit_intercept=True, positive=True)
        lad2 = LADRegressor(fit_intercept=False, positive=True)
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"LADRegressor, fit_intercept True, positive True, fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(lad1.coef_.min() >= 0)
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)
        try:
            lad2.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"LADRegressor, fit_intercept False, positive True, fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad2.coef_, np.ndarray))
        self.assertTrue(len(lad2.coef_) == self.X.shape[1])
        self.assertTrue(lad2.coef_.min() >= 0)
        self.assertTrue(lad2.intercept_ == 0)

        """Check L1 regularization LADRegressor fit method works as expected"""
        lad1 = LADRegressor(fit_intercept=True, alpha=1, shrinkage_type="l1")
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"L1 regularization LADRegressor, alpha = 1, fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)
        if PYTHON_3_9_OR_LATER:
            qr = QuantileRegressor(alpha=2 * self.X.shape[0], fit_intercept=True).fit(
                self.X, self.y
            )
            self.assertTrue(np.allclose(lad1.coef_, qr.coef_, atol=0.01, rtol=0.01))

        """Check L2 regularization LADRegressor fit method works as expected"""
        lad1 = LADRegressor(fit_intercept=True, alpha=1, shrinkage_type="l2")
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"L2 regularization LADRegressor, alpha = 1, fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)

    def test_types_predict(self):
        lad = LADRegressor().fit(self.X, self.y)
        # X - when a dataframe
        self.assertRaises(TypeError, lad.predict, X=1)
        self.assertRaises(TypeError, lad.predict, X="X")
        self.assertRaises(ValueError, lad.predict, X=self.X.iloc[:, :-1])
        self.assertRaises(ValueError, lad.predict, X=self.X_nan.values)
        self.assertRaises(
            ValueError,
            lad.predict,
            X=pd.DataFrame(
                data = np.array([["hello"] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        self.assertRaises(
            ValueError,
            lad.predict,
            X=pd.DataFrame(
                data = np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        # X - when a numpy array
        self.assertRaises(ValueError, lad.predict, X=np.expand_dims(self.X_numpy, 0))
        self.assertRaises(
            ValueError,
            lad.predict,
            X=np.array([["hello"] * self.X.shape[1]] * self.X.shape[0])
        )
        self.assertRaises(
            ValueError,
            lad.predict,
            X=np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0])
        )

    def test_valid_predict(self):
        """Check default LADRegressor predict method works as expected"""
        lad = LADRegressor().fit(self.X, self.y)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(f"Default LADRegressor predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])
        lad = LADRegressor().fit(self.X.values, self.y.values)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(f"Default LADRegressor predict method failed with exception: {e}")
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L1 regularization LADRegressor predict method works as expected"""
        lad = LADRegressor(alpha=1, shrinkage_type="l1").fit(self.X, self.y)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L1 regularization LADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])
        lad = LADRegressor(alpha=1, shrinkage_type="l1").fit(
            self.X.values, self.y.values
        )
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L1 regularization LADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L2 regularization LADRegressor predict method works as expected"""
        lad = LADRegressor(alpha=1, shrinkage_type="l2").fit(self.X, self.y)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L2 regularization LADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])
        lad = LADRegressor(alpha=1, shrinkage_type="l2").fit(
            self.X.values, self.y.values
        )
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L2 regularization LADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])


class TestSignWeightedLADRegression(unittest.TestCase):
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

        self.sign_weights = SignWeightedLADRegressor()._calculate_sign_weights(self.y)

    def test_types_init(self):
        # fit_intercept
        self.assertRaises(TypeError, SignWeightedLADRegressor, fit_intercept=1)
        self.assertRaises(TypeError, SignWeightedLADRegressor, fit_intercept="True")
        # positive
        self.assertRaises(TypeError, SignWeightedLADRegressor, positive=1)
        self.assertRaises(TypeError, SignWeightedLADRegressor, positive="True")
        # alpha
        self.assertRaises(TypeError, SignWeightedLADRegressor, alpha="1")
        self.assertRaises(TypeError, SignWeightedLADRegressor, alpha=True)
        self.assertRaises(ValueError, SignWeightedLADRegressor, alpha=-1)
        # shrinkage_type
        self.assertRaises(TypeError, SignWeightedLADRegressor, shrinkage_type=1)
        self.assertRaises(TypeError, SignWeightedLADRegressor, shrinkage_type=True)
        self.assertRaises(ValueError, SignWeightedLADRegressor, shrinkage_type="l3")
        self.assertRaises(ValueError, SignWeightedLADRegressor, shrinkage_type="string")
        # tol
        self.assertRaises(TypeError, SignWeightedLADRegressor, tol="1")
        self.assertRaises(TypeError, SignWeightedLADRegressor, tol=True)
        self.assertRaises(ValueError, SignWeightedLADRegressor, tol=-1)
        # max_iter
        self.assertRaises(TypeError, SignWeightedLADRegressor, maxiter="1")
        self.assertRaises(TypeError, SignWeightedLADRegressor, maxiter=True)
        self.assertRaises(ValueError, SignWeightedLADRegressor, maxiter=-1)

    def test_valid_init(self):
        # Check defaults set correctly
        lad = SignWeightedLADRegressor()
        self.assertEqual(lad.fit_intercept, True)
        self.assertEqual(lad.positive, False)
        self.assertEqual(lad.alpha, 0)
        self.assertEqual(lad.shrinkage_type, "l1")
        self.assertEqual(lad.tol, None)
        self.assertEqual(lad.maxiter, None)

        # Change defaults
        lad = SignWeightedLADRegressor(
            fit_intercept=False,
            positive=True,
            alpha=0.1,
            shrinkage_type="l2",
            tol=0.1,
            maxiter=100,
        )
        self.assertEqual(lad.fit_intercept, False)
        self.assertEqual(lad.positive, True)
        self.assertEqual(lad.alpha, 0.1)
        self.assertEqual(lad.shrinkage_type, "l2")
        self.assertEqual(lad.tol, 0.1)
        self.assertEqual(lad.maxiter, 100)

    def test_types_fit(self):
        # X
        lad = SignWeightedLADRegressor()
        self.assertRaises(TypeError, lad.fit, X=1, y=self.y)
        self.assertRaises(TypeError, lad.fit, X="X", y=self.y)
        self.assertRaises(ValueError, lad.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, lad.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, lad.fit, X=self.X_nan.values, y=self.y)
        # y
        self.assertRaises(TypeError, lad.fit, X=self.X, y=1)
        self.assertRaises(TypeError, lad.fit, X=self.X, y="y")
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y_nan)
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y_nan.values)

    def test_valid_fit(self):
        """Check default SWLAD fit method works as expected"""
        lad = SignWeightedLADRegressor(fit_intercept=True)
        try:
            lad.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Default SWLAD fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad.intercept_, float))
        self.assertTrue(lad.intercept_ != 0)
        if PYTHON_3_9_OR_LATER:
            # Check the solution is close to QuantileRegressor from scikit-learn
            qr = QuantileRegressor(alpha=0, fit_intercept=True).fit(
                self.X, self.y, sample_weight=self.sign_weights
            )
            self.assertTrue(np.allclose(lad.coef_, qr.coef_, atol=0.01, rtol=0.01))
            np.testing.assert_almost_equal(lad.intercept_, qr.intercept_, decimal=2)

        """Check default SWLAD fit method works as expected with numpy arrays"""
        lad = SignWeightedLADRegressor(fit_intercept=True)
        try:
            lad.fit(X=self.X.values, y=self.y.values)
        except Exception as e:
            self.fail(f"Default SWLADRegressor fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad.intercept_, float))
        self.assertTrue(lad.intercept_ != 0)
        if PYTHON_3_9_OR_LATER:
            # Check the solution is close to QuantileRegressor from scikit-learn
            qr = QuantileRegressor(alpha=0, fit_intercept=True).fit(
                self.X.values, self.y.values, sample_weight=self.sign_weights
            )
            self.assertTrue(np.allclose(lad.coef_, qr.coef_, atol=0.01, rtol=0.01))
            np.testing.assert_almost_equal(lad.intercept_, qr.intercept_, decimal=2)

        """Check no intercept SWLAD fit method works as expected"""
        lad = SignWeightedLADRegressor(fit_intercept=False)
        try:
            lad.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"No intercept SignWeightedLADRegressor fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(lad.intercept_ == 0)
        if PYTHON_3_9_OR_LATER:
            qr = QuantileRegressor(alpha=0, fit_intercept=False).fit(
                self.X, self.y, sample_weight=self.sign_weights
            )
            self.assertTrue(np.allclose(lad.coef_, qr.coef_, atol=0.01, rtol=0.01))
            np.testing.assert_almost_equal(lad.intercept_, qr.intercept_, decimal=2)

        """Check L1 regularization SignWeightedLADRegressor fit method works as expected"""
        lad1 = SignWeightedLADRegressor(
            fit_intercept=True, alpha=1, shrinkage_type="l1"
        )
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"L1 regularization SignWeightedLADRegressor, alpha = 1, fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)
        if PYTHON_3_9_OR_LATER:
            qr = QuantileRegressor(alpha=2 * self.X.shape[0], fit_intercept=True).fit(
                self.X, self.y, sample_weight=self.sign_weights
            )
            self.assertTrue(np.allclose(lad1.coef_, qr.coef_, atol=0.01, rtol=0.01))

        """Check L2 regularization SignWeightedLADRegressor fit method works as expected"""
        lad1 = SignWeightedLADRegressor(
            fit_intercept=True, alpha=1, shrinkage_type="l2"
        )
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"L2 regularization SignWeightedLADRegressor, alpha = 1, fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)

    def test_types_predict(self):
        lad = SignWeightedLADRegressor().fit(self.X, self.y)
        self.assertRaises(TypeError, lad.predict, X=1)
        self.assertRaises(TypeError, lad.predict, X="X")
        self.assertRaises(ValueError, lad.predict, X=self.X.iloc[:, :-1])
        self.assertRaises(ValueError, lad.predict, X=self.X_nan)
        self.assertRaises(ValueError, lad.predict, X=self.X_nan.values)

    def test_valid_predict(self):
        """Check default SignWeightedLADRegressor predict method works as expected"""
        lad = SignWeightedLADRegressor().fit(self.X, self.y)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"Default SignWeightedLADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])
        lad = SignWeightedLADRegressor().fit(self.X.values, self.y.values)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"Default SignWeightedLADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L1 regularization SignWeightedLADRegressor predict method works as expected"""
        lad = SignWeightedLADRegressor(alpha=1, shrinkage_type="l1").fit(self.X, self.y)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L1 regularization SignWeightedLADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])
        lad = SignWeightedLADRegressor(alpha=1, shrinkage_type="l1").fit(
            self.X.values, self.y.values
        )
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L1 regularization SignWeightedLADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L2 regularization SignWeightedLADRegressor predict method works as expected"""
        lad = SignWeightedLADRegressor(alpha=1, shrinkage_type="l2").fit(self.X, self.y)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L2 regularization SignWeightedLADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])
        lad = SignWeightedLADRegressor(alpha=1, shrinkage_type="l2").fit(
            self.X.values, self.y.values
        )
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L2 regularization SignWeightedLADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

    def test_valid_set_params(self):
        ls = SignWeightedLADRegressor()
        ls.set_params(
            fit_intercept=False, positive=True, alpha=0.1, shrinkage_type="l2"
        )
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.1)
        self.assertEqual(ls.shrinkage_type, "l2")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)
        self.assertEqual(ls.model.alpha, 0.1)

        ls = SignWeightedLADRegressor()
        ls.set_params(
            fit_intercept=False, positive=True, alpha=0.2, shrinkage_type="l1"
        )
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.2)
        self.assertEqual(ls.shrinkage_type, "l1")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)
        self.assertEqual(ls.model.alpha, 0.2)

        ls = SignWeightedLADRegressor()
        ls.set_params(fit_intercept=False, positive=True, alpha=0, shrinkage_type="l2")
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0)
        self.assertEqual(ls.shrinkage_type, "l2")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)


class TestTimeWeightedLADRegression(unittest.TestCase):
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

        self.time_weights = TimeWeightedLADRegressor(
            half_life=24
        )._calculate_time_weights(self.y)

    def test_types_init(self):
        # fit_intercept
        self.assertRaises(TypeError, TimeWeightedLADRegressor, fit_intercept=1)
        self.assertRaises(TypeError, TimeWeightedLADRegressor, fit_intercept="True")
        # positive
        self.assertRaises(TypeError, TimeWeightedLADRegressor, positive=1)
        self.assertRaises(TypeError, TimeWeightedLADRegressor, positive="True")
        # half_life
        self.assertRaises(TypeError, TimeWeightedLADRegressor, half_life="1")
        self.assertRaises(TypeError, TimeWeightedLADRegressor, half_life=True)
        self.assertRaises(ValueError, TimeWeightedLADRegressor, half_life=-1)
        # alpha
        self.assertRaises(TypeError, TimeWeightedLADRegressor, alpha="1")
        self.assertRaises(TypeError, TimeWeightedLADRegressor, alpha=True)
        self.assertRaises(ValueError, TimeWeightedLADRegressor, alpha=-1)
        # shrinkage_type
        self.assertRaises(TypeError, TimeWeightedLADRegressor, shrinkage_type=1)
        self.assertRaises(TypeError, TimeWeightedLADRegressor, shrinkage_type=True)
        self.assertRaises(ValueError, TimeWeightedLADRegressor, shrinkage_type="l3")
        self.assertRaises(ValueError, TimeWeightedLADRegressor, shrinkage_type="string")
        # tol
        self.assertRaises(TypeError, TimeWeightedLADRegressor, tol="1")
        self.assertRaises(TypeError, TimeWeightedLADRegressor, tol=True)
        self.assertRaises(ValueError, TimeWeightedLADRegressor, tol=-1)
        # max_iter
        self.assertRaises(TypeError, TimeWeightedLADRegressor, maxiter="1")
        self.assertRaises(TypeError, TimeWeightedLADRegressor, maxiter=True)
        self.assertRaises(ValueError, TimeWeightedLADRegressor, maxiter=-1)

    def test_valid_init(self):
        # Check defaults set correctly
        lad = TimeWeightedLADRegressor()
        self.assertEqual(lad.fit_intercept, True)
        self.assertEqual(lad.positive, False)
        self.assertEqual(lad.half_life, 252)
        self.assertEqual(lad.alpha, 0)
        self.assertEqual(lad.shrinkage_type, "l1")
        self.assertEqual(lad.tol, None)
        self.assertEqual(lad.maxiter, None)

        # Change defaults
        lad = TimeWeightedLADRegressor(
            fit_intercept=False,
            positive=True,
            half_life=12,
            alpha=0.1,
            shrinkage_type="l2",
            tol=0.1,
            maxiter=100,
        )
        self.assertEqual(lad.fit_intercept, False)
        self.assertEqual(lad.positive, True)
        self.assertEqual(lad.half_life, 12)
        self.assertEqual(lad.alpha, 0.1)
        self.assertEqual(lad.shrinkage_type, "l2")
        self.assertEqual(lad.tol, 0.1)
        self.assertEqual(lad.maxiter, 100)

    def test_types_fit(self):
        # X
        lad = TimeWeightedLADRegressor()
        self.assertRaises(TypeError, lad.fit, X=1, y=self.y)
        self.assertRaises(TypeError, lad.fit, X="X", y=self.y)
        self.assertRaises(ValueError, lad.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, lad.fit, X=self.X_nan, y=self.y)
        self.assertRaises(TypeError, lad.fit, X=self.X_nan.values, y=self.y)
        self.assertRaises(TypeError, lad.fit, X=self.X.values, y=self.y)
        # y
        self.assertRaises(TypeError, lad.fit, X=self.X, y=1)
        self.assertRaises(TypeError, lad.fit, X=self.X, y="y")
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, lad.fit, X=self.X, y=self.y_nan)
        self.assertRaises(TypeError, lad.fit, X=self.X, y=self.y_nan.values)
        self.assertRaises(TypeError, lad.fit, X=self.X, y=self.y.values)

    def test_valid_fit(self):
        """Check default TWLAD fit method works as expected"""
        lad = TimeWeightedLADRegressor(fit_intercept=True, half_life=24)
        try:
            lad.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Default TWLAD fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad.intercept_, float))
        self.assertTrue(lad.intercept_ != 0)
        if PYTHON_3_9_OR_LATER:
            # Check the solution is close to QuantileRegressor from scikit-learn
            qr = QuantileRegressor(alpha=0, fit_intercept=True).fit(
                self.X, self.y, self.time_weights
            )
            self.assertTrue(np.allclose(lad.coef_, qr.coef_, atol=0.01, rtol=0.01))
            np.testing.assert_almost_equal(lad.intercept_, qr.intercept_, decimal=2)

        """Check no intercept TWLAD fit method works as expected"""
        lad = TimeWeightedLADRegressor(fit_intercept=False, half_life=24)
        try:
            lad.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"No intercept TimeWeightedLADRegressor fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(lad.intercept_ == 0)
        if PYTHON_3_9_OR_LATER:
            qr = QuantileRegressor(alpha=0, fit_intercept=False).fit(
                self.X, self.y, sample_weight=self.time_weights
            )
            self.assertTrue(np.allclose(lad.coef_, qr.coef_, atol=0.01, rtol=0.01))
            np.testing.assert_almost_equal(lad.intercept_, qr.intercept_, decimal=2)

        """Check L1 regularization TimeWeightedLADRegressor fit method works as expected"""
        lad1 = TimeWeightedLADRegressor(
            fit_intercept=True, alpha=1, shrinkage_type="l1", half_life=24
        )
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"L1 regularization TimeWeightedLADRegressor, alpha = 1, fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)
        if PYTHON_3_9_OR_LATER:
            qr = QuantileRegressor(alpha=2 * self.X.shape[0], fit_intercept=True).fit(
                self.X, self.y, sample_weight=self.time_weights
            )
            self.assertTrue(np.allclose(lad1.coef_, qr.coef_, atol=0.01, rtol=0.01))

        """Check L2 regularization TimeWeightedLADRegressor fit method works as expected"""
        lad1 = TimeWeightedLADRegressor(
            fit_intercept=True, alpha=1, shrinkage_type="l2", half_life=24
        )
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"L2 regularization TimeWeightedLADRegressor, alpha = 1, fit method failed with exception: {e}"
            )
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)

    def test_types_predict(self):
        lad = TimeWeightedLADRegressor(half_life=24).fit(self.X, self.y)
        self.assertRaises(TypeError, lad.predict, X=1)
        self.assertRaises(TypeError, lad.predict, X="X")
        self.assertRaises(ValueError, lad.predict, X=self.X.iloc[:, :-1])
        self.assertRaises(ValueError, lad.predict, X=self.X_nan)
        self.assertRaises(TypeError, lad.predict, X=self.X.values)

    def test_valid_predict(self):
        """Check default TimeWeightedLADRegressor predict method works as expected"""
        lad = TimeWeightedLADRegressor().fit(self.X, self.y)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"Default TimeWeightedLADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L1 regularization TimeWeightedLADRegressor predict method works as expected"""
        lad = TimeWeightedLADRegressor(alpha=1, shrinkage_type="l1").fit(self.X, self.y)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L1 regularization TimeWeightedLADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

        """Check L2 regularization SignWeightedLADRegressor predict method works as expected"""
        lad = TimeWeightedLADRegressor(alpha=1, shrinkage_type="l2").fit(self.X, self.y)
        try:
            pred = lad.predict(self.X)
        except Exception as e:
            self.fail(
                f"L2 regularization TimeWeightedLADRegressor predict method failed with exception: {e}"
            )
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertTrue(len(pred) == self.X.shape[0])

    def test_valid_set_params(self):
        ls = TimeWeightedLADRegressor()
        ls.set_params(
            fit_intercept=False,
            positive=True,
            alpha=0.1,
            half_life=3,
            shrinkage_type="l2",
        )
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.1)
        self.assertEqual(ls.half_life, 3)
        self.assertEqual(ls.shrinkage_type, "l2")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)
        self.assertEqual(ls.model.alpha, 0.1)

        ls = TimeWeightedLADRegressor()
        ls.set_params(
            fit_intercept=False,
            positive=True,
            half_life=6,
            alpha=0.2,
            shrinkage_type="l1",
        )
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0.2)
        self.assertEqual(ls.half_life, 6)
        self.assertEqual(ls.shrinkage_type, "l1")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)
        self.assertEqual(ls.model.alpha, 0.2)

        ls = TimeWeightedLADRegressor()
        ls.set_params(
            fit_intercept=False,
            positive=True,
            half_life=12,
            alpha=0,
            shrinkage_type="l2",
        )
        self.assertEqual(ls.fit_intercept, False)
        self.assertEqual(ls.positive, True)
        self.assertEqual(ls.alpha, 0)
        self.assertEqual(ls.half_life, 12)
        self.assertEqual(ls.shrinkage_type, "l2")
        self.assertEqual(ls.model.fit_intercept, False)
        self.assertEqual(ls.model.positive, True)


if __name__ == "__main__":
    Test = TestTimeWeightedLADRegression()
    Test.setUpClass()
    Test.test_valid_fit()
