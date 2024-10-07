import os
import numpy as np
import pandas as pd

import unittest

from macrosynergy.learning import (
    LADRegressor,
)
from sklearn.linear_model import QuantileRegressor

from parameterized import parameterized

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
        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y = df["XR"]
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
        # X
        lad = LADRegressor()
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
        """Check default LADRegressor fit method works as expected"""
        lad = LADRegressor(fit_intercept = True)
        try:
            lad.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Default LADRegressor fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad.intercept_, float))
        self.assertTrue(lad.intercept_ != 0)
        # Check the solution is close to QuantileRegressor from scikit-learn
        qr = QuantileRegressor(alpha = 0, fit_intercept=True).fit(self.X, self.y)
        np.testing.assert_almost_equal(lad.coef_, qr.coef_,decimal=2)
        np.testing.assert_almost_equal(lad.intercept_, qr.intercept_,decimal=2)

        """Check no intercept LADRegressor fit method works as expected"""
        lad = LADRegressor(fit_intercept = False)
        try:
            lad.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"No intercept LADRegressor fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad.coef_, np.ndarray))
        self.assertTrue(len(lad.coef_) == self.X.shape[1])
        self.assertTrue(lad.intercept_ == 0)
        qr = QuantileRegressor(alpha = 0, fit_intercept=False).fit(self.X, self.y)
        np.testing.assert_almost_equal(lad.coef_, qr.coef_,decimal=2)
        np.testing.assert_almost_equal(lad.intercept_, qr.intercept_,decimal=2)
        
        """Check positive restriction LADRegressor fit method works as expected"""
        lad1 = LADRegressor(fit_intercept = True, positive=True)
        lad2 = LADRegressor(fit_intercept = False, positive=True)
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"LADRegressor, fit_intercept True, positive True, fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(lad1.coef_.min() >= 0)
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)
        try:
            lad2.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"LADRegressor, fit_intercept False, positive True, fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad2.coef_, np.ndarray))
        self.assertTrue(len(lad2.coef_) == self.X.shape[1])
        self.assertTrue(lad2.coef_.min() >= 0)
        self.assertTrue(lad2.intercept_ == 0)

        """Check L1 regularization LADRegressor fit method works as expected"""
        lad1 = LADRegressor(fit_intercept = True, alpha = 1, shrinkage_type = "l1")
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"L1 regularization LADRegressor, alpha = 1, fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)
        qr = QuantileRegressor(alpha = 2 * self.X.shape[0], fit_intercept=True).fit(self.X, self.y)
        np.testing.assert_almost_equal(lad1.coef_, qr.coef_,decimal=2)

        """Check L2 regularization LADRegressor fit method works as expected"""
        lad1 = LADRegressor(fit_intercept = True, alpha = 1, shrinkage_type = "l2")
        try:
            lad1.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"L2 regularization LADRegressor, alpha = 1, fit method failed with exception: {e}")
        self.assertTrue(isinstance(lad1.coef_, np.ndarray))
        self.assertTrue(len(lad1.coef_) == self.X.shape[1])
        self.assertTrue(isinstance(lad1.intercept_, float))
        self.assertTrue(lad1.intercept_ != 0)

    def test_types_predict(self):
        lad = LADRegressor().fit(self.X, self.y)
        self.assertRaises(TypeError, lad.predict, X=1)
        self.assertRaises(TypeError, lad.predict, X="X")
        self.assertRaises(ValueError, lad.predict, X=self.X.iloc[:,:-1])
        self.assertRaises(ValueError, lad.predict, X=self.X_nan)
        self.assertRaises(ValueError, lad.predict, X=self.X_nan.values)

    def test_valid_predict(self):
        pass