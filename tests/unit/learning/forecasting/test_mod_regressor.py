import os
import numbers
import numpy as np
import pandas as pd

import unittest

from macrosynergy.learning import (
    ModifiedLinearRegression,
    ModifiedSignWeightedLinearRegression,
    ModifiedTimeWeightedLinearRegression,
)

from sklearn.linear_model import LinearRegression

from parameterized import parameterized

class TestModifiedLinearRegression(unittest.TestCase):
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
        self.X_ones = self.X.copy()
        self.X_ones["intercept"] = 1
        self.y = df["XR"]
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

    def test_types_init(self):
        # method
        self.assertRaises(TypeError, ModifiedLinearRegression, method=1)
        self.assertRaises(ValueError, ModifiedLinearRegression, method="string")
        self.assertRaises(ValueError, ModifiedLinearRegression, method="invalid_method")
        # fit_intercept
        self.assertRaises(TypeError, ModifiedLinearRegression, method="analytic", fit_intercept=1)
        self.assertRaises(TypeError, ModifiedLinearRegression, method="bootstrap", fit_intercept=1)
        # positive
        self.assertRaises(TypeError, ModifiedLinearRegression, method="analytic", positive=1)
        self.assertRaises(TypeError, ModifiedLinearRegression, method="bootstrap", positive=1)
        # error_offset
        self.assertRaises(TypeError, ModifiedLinearRegression, method="analytic", error_offset="hello")
        self.assertRaises(TypeError, ModifiedLinearRegression, method="bootstrap", error_offset="hello")
        self.assertRaises(ValueError, ModifiedLinearRegression, method="analytic", error_offset=-1)
        self.assertRaises(ValueError, ModifiedLinearRegression, method="bootstrap", error_offset=-1)
        # bootstrap_panel
        self.assertRaises(TypeError, ModifiedLinearRegression, method="bootstrap", bootstrap_method=1)
        self.assertRaises(ValueError, ModifiedLinearRegression, method="bootstrap", bootstrap_method="string")
        # bootstrap_iters
        self.assertRaises(TypeError, ModifiedLinearRegression, method="bootstrap", bootstrap_iters="string")
        self.assertRaises(ValueError, ModifiedLinearRegression, method="bootstrap", bootstrap_iters=-1)
        # resample_ratio
        self.assertRaises(TypeError, ModifiedLinearRegression, method="bootstrap", resample_ratio="string")
        self.assertRaises(ValueError, ModifiedLinearRegression, method="bootstrap", resample_ratio=-1)
        # analytic_method
        self.assertRaises(TypeError, ModifiedLinearRegression, method="analytic", analytic_method=1)

    def test_valid_init(self):
        # Check defaults set correctly
        lr = ModifiedLinearRegression(method="analytic")
        self.assertEqual(lr.method, "analytic")
        self.assertEqual(lr.fit_intercept, True)
        self.assertEqual(lr.positive, False)
        self.assertEqual(lr.error_offset, 1e-2)
        self.assertEqual(lr.bootstrap_method, "panel")
        self.assertEqual(lr.bootstrap_iters, 1000)
        self.assertEqual(lr.resample_ratio, 1)
        self.assertEqual(lr.analytic_method, None)
        lr = ModifiedLinearRegression(method="bootstrap")
        self.assertEqual(lr.method, "bootstrap")
        self.assertEqual(lr.fit_intercept, True)
        self.assertEqual(lr.positive, False)
        self.assertEqual(lr.error_offset, 1e-2)
        self.assertEqual(lr.bootstrap_method, "panel")
        self.assertEqual(lr.bootstrap_iters, 1000)
        self.assertEqual(lr.resample_ratio, 1)
        self.assertEqual(lr.analytic_method, None)

        # Change defaults
        lr = ModifiedLinearRegression(
            method="analytic",
            fit_intercept=False,
            positive=True,
            error_offset=1e-3,
            analytic_method="White",
        )
        self.assertEqual(lr.method, "analytic")
        self.assertEqual(lr.fit_intercept, False)
        self.assertEqual(lr.positive, True)
        self.assertEqual(lr.error_offset, 1e-3)
        self.assertEqual(lr.bootstrap_method, "panel")
        self.assertEqual(lr.bootstrap_iters, 1000)
        self.assertEqual(lr.resample_ratio, 1)
        self.assertEqual(lr.analytic_method, "White")

        lr = ModifiedLinearRegression(
            method="bootstrap",
            fit_intercept=False,
            positive=True,
            error_offset=1e-3,
            bootstrap_method="period",
            bootstrap_iters=100,
            resample_ratio=0.5,
        )
        self.assertEqual(lr.method, "bootstrap")
        self.assertEqual(lr.fit_intercept, False)
        self.assertEqual(lr.positive, True)
        self.assertEqual(lr.error_offset, 1e-3)
        self.assertEqual(lr.bootstrap_method, "period")
        self.assertEqual(lr.bootstrap_iters, 100)
        self.assertEqual(lr.resample_ratio, 0.5)
        self.assertEqual(lr.analytic_method, None)

    def test_types_fit(self):
        mlr = ModifiedLinearRegression(method="analytic")
        # X
        self.assertRaises(TypeError, mlr.fit, X=1, y=self.y)
        self.assertRaises(TypeError, mlr.fit, X="string", y=self.y)
        self.assertRaises(ValueError, mlr.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, mlr.fit, X=self.X, y=1)
        self.assertRaises(TypeError, mlr.fit, X=self.X, y="string")
        self.assertRaises(ValueError, mlr.fit, X=self.X, y=self.y_nan)

        # Repeat for the bootstrap
        mlr = ModifiedLinearRegression(method="bootstrap")
        # X
        self.assertRaises(TypeError, mlr.fit, X=1, y=self.y)
        self.assertRaises(TypeError, mlr.fit, X="string", y=self.y)
        self.assertRaises(ValueError, mlr.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, mlr.fit, X=self.X, y=1)
        self.assertRaises(TypeError, mlr.fit, X=self.X, y="string")
        self.assertRaises(ValueError, mlr.fit, X=self.X, y=self.y_nan)

    def test_valid_fit(self):
        # ModifiedLinearRegression with usual standard errors
        mlr1 = ModifiedLinearRegression(method="analytic", error_offset=1e-2)
        mlr1.fit(self.X, self.y)
        self.assertIsInstance(mlr1.coef_, np.ndarray)
        self.assertIsInstance(mlr1.intercept_, numbers.Number)

        lr = LinearRegression().fit(self.X, self.y)
        # determine standard errors
        n = self.X.shape[0]
        p = self.X.shape[1]
        y_hat = lr.predict(self.X)
        residuals = self.y - y_hat
        sigma2 = np.sum(residuals ** 2) / (n - p - 1)
        XTX = np.matmul(self.X_ones.T, self.X_ones)
        XTX_inv = np.linalg.inv(XTX)
        ses = sigma2 * np.diag(XTX_inv)
        intercept_adj = lr.intercept_ / (np.sqrt(ses[-1])+1e-2)
        coef_adj = lr.coef_ / (np.sqrt(ses[:-1])+1e-2)
        np.testing.assert_array_almost_equal(mlr1.coef_, coef_adj)
        np.testing.assert_almost_equal(mlr1.intercept_, intercept_adj)

        # ModifiedLinearRegression with HC3 errors
        mlr2 = ModifiedLinearRegression(method="analytic", error_offset=1e-2, analytic_method="White")
        mlr2.fit(self.X, self.y)
        self.assertIsInstance(mlr2.coef_, np.ndarray)
        self.assertIsInstance(mlr2.intercept_, numbers.Number)

        # ModifiedLinearRegression with panel bootstrap
        mlr3 = ModifiedLinearRegression(method="bootstrap", bootstrap_method="panel", bootstrap_iters=100)
        mlr3.fit(self.X, self.y)
        self.assertIsInstance(mlr3.coef_, np.ndarray)
        self.assertIsInstance(mlr3.intercept_, numbers.Number)
        
        # ModifiedLinearRegression with period bootstrap
        ml4 = ModifiedLinearRegression(method="bootstrap", bootstrap_method="period", bootstrap_iters=100)
        ml4.fit(self.X, self.y)
        self.assertIsInstance(ml4.coef_, np.ndarray)
        self.assertIsInstance(ml4.intercept_, numbers.Number)
        # ModifiedLinearRegression with cross-sectional bootstrap
        mlr5 = ModifiedLinearRegression(method="bootstrap", bootstrap_method="cross", bootstrap_iters=100)
        mlr5.fit(self.X, self.y)
        self.assertIsInstance(mlr5.coef_, np.ndarray)
        self.assertIsInstance(mlr5.intercept_, numbers.Number)
        # ModifiedLinearRegression with period_per_cross bootstrap
        mlr6 = ModifiedLinearRegression(method="bootstrap", bootstrap_method="period_per_cross", bootstrap_iters=100)
        mlr6.fit(self.X, self.y)
        self.assertIsInstance(mlr6.coef_, np.ndarray)
        self.assertIsInstance(mlr6.intercept_, numbers.Number)
        # ModifiedLinearRegression with cross_per_period bootstrap
        mlr7 = ModifiedLinearRegression(method="bootstrap", bootstrap_method="cross_per_period", bootstrap_iters=100)
        mlr7.fit(self.X, self.y)
        self.assertIsInstance(mlr7.coef_, np.ndarray)
        self.assertIsInstance(mlr7.intercept_, numbers.Number)

    def test_types_predict(self):
        mlr = ModifiedLinearRegression(method="analytic")
        mlr.fit(self.X, self.y)
        # X
        self.assertRaises(TypeError, mlr.predict, X=1)
        self.assertRaises(TypeError, mlr.predict, X="string")
        self.assertRaises(ValueError, mlr.predict, X=self.X_nan)
        self.assertRaises(ValueError, mlr.predict, X=self.X_nan.reset_index())

    def test_valid_predict(self):
        mlr = ModifiedLinearRegression(method="analytic")
        mlr.fit(self.X, self.y)
        y_pred = mlr.predict(self.X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(self.y))
        np.testing.assert_array_equal(y_pred, mlr.model.predict(self.X))

    def test_valid_create_signal(self):
        mlr = ModifiedLinearRegression(method="analytic")
        mlr.fit(self.X, self.y)
        y_pred = mlr.create_signal(self.X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(self.y))
        np.testing.assert_array_equal(y_pred, mlr.intercept_ + np.matmul(self.X, mlr.coef_))

    def test_types_create_signal(self):
        mlr = ModifiedLinearRegression(method="analytic")
        mlr.fit(self.X, self.y)
        # X
        self.assertRaises(TypeError, mlr.create_signal, X=1)
        self.assertRaises(TypeError, mlr.create_signal, X="string")
        self.assertRaises(ValueError, mlr.create_signal, X=self.X_nan)
        self.assertRaises(ValueError, mlr.create_signal, X=self.X_nan.reset_index())