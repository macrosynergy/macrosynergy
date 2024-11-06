import os
import numbers
import numpy as np
import pandas as pd

import unittest
import itertools

from macrosynergy.learning import (
    ModifiedLinearRegression,
    ModifiedSignWeightedLinearRegression,
    ModifiedTimeWeightedLinearRegression,
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
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
        self.assertRaises(
            TypeError, ModifiedLinearRegression, method="analytic", fit_intercept=1
        )
        self.assertRaises(
            TypeError, ModifiedLinearRegression, method="bootstrap", fit_intercept=1
        )
        # positive
        self.assertRaises(
            TypeError, ModifiedLinearRegression, method="analytic", positive=1
        )
        self.assertRaises(
            TypeError, ModifiedLinearRegression, method="bootstrap", positive=1
        )
        # error_offset
        self.assertRaises(
            TypeError, ModifiedLinearRegression, method="analytic", error_offset="hello"
        )
        self.assertRaises(
            TypeError,
            ModifiedLinearRegression,
            method="bootstrap",
            error_offset="hello",
        )
        self.assertRaises(
            ValueError, ModifiedLinearRegression, method="analytic", error_offset=-1
        )
        self.assertRaises(
            ValueError, ModifiedLinearRegression, method="bootstrap", error_offset=-1
        )
        # bootstrap_panel
        self.assertRaises(
            TypeError, ModifiedLinearRegression, method="bootstrap", bootstrap_method=1
        )
        self.assertRaises(
            ValueError,
            ModifiedLinearRegression,
            method="bootstrap",
            bootstrap_method="string",
        )
        # bootstrap_iters
        self.assertRaises(
            TypeError,
            ModifiedLinearRegression,
            method="bootstrap",
            bootstrap_iters="string",
        )
        self.assertRaises(
            ValueError, ModifiedLinearRegression, method="bootstrap", bootstrap_iters=-1
        )
        # resample_ratio
        self.assertRaises(
            TypeError,
            ModifiedLinearRegression,
            method="bootstrap",
            resample_ratio="string",
        )
        self.assertRaises(
            ValueError, ModifiedLinearRegression, method="bootstrap", resample_ratio=-1
        )
        # analytic_method
        self.assertRaises(
            TypeError, ModifiedLinearRegression, method="analytic", analytic_method=1
        )

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
        self.assertRaises(
            ValueError, mlr.fit, X=self.X.reset_index(drop=True), y=self.y
        )
        self.assertRaises(
            ValueError,
            mlr.fit,
            X=pd.DataFrame(
                data=np.zeros((self.X.shape[0], 2)),
                index=self.X.reset_index().set_index(["RIR", "real_date"]),
            ),
            y=self.y,
        )
        self.assertRaises(
            ValueError,
            mlr.fit,
            X=pd.DataFrame(
                data=np.zeros((self.X.shape[0], 2)),
                index=self.X.reset_index().set_index(["cid", "RIR"]),
            ),
            y=self.y,
        )
        # y
        self.assertRaises(TypeError, mlr.fit, X=self.X, y=1)
        self.assertRaises(TypeError, mlr.fit, X=self.X, y="string")
        self.assertRaises(ValueError, mlr.fit, X=self.X, y=self.y_nan)
        self.assertRaises(
            ValueError, mlr.fit, X=self.X, y=self.y.reset_index(drop=True)
        )
        self.assertRaises(
            ValueError,
            mlr.fit,
            X=self.X,
            y=pd.DataFrame(data=np.ones((self.y.shape[0], 2)), index=self.y.index),
        )

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
        """ModifiedLinearRegression with usual standard errors"""
        mlr1 = ModifiedLinearRegression(
            method="analytic", error_offset=1e-2, fit_intercept=False
        )
        mlr1.fit(self.X, self.y)
        self.assertIsInstance(mlr1.coef_, np.ndarray)
        self.assertIsInstance(mlr1.intercept_, numbers.Number)
        self.assertEqual(mlr1.intercept_, 0)
        # Check this aligns with theory
        lr = LinearRegression(fit_intercept=False).fit(self.X, self.y)
        n = self.X.shape[0]
        p = self.X.shape[1]
        y_hat = lr.predict(self.X)
        residuals = self.y - y_hat
        sigma2 = np.sum(residuals**2) / (n - p)
        XTX = np.matmul(self.X.T.values, self.X.values)
        XTX_inv = np.linalg.inv(XTX)
        ses = np.sqrt(sigma2 * np.diag(XTX_inv))
        coef_adj = lr.coef_ / (ses + 1e-2)
        np.testing.assert_array_almost_equal(mlr1.coef_, coef_adj, decimal=4)

        """ ModifiedLinearRegression with usual HC3 errors """
        mlr2 = ModifiedLinearRegression(
            method="analytic",
            error_offset=1e-2,
            analytic_method="White",
            fit_intercept=False,
        )
        mlr2.fit(self.X, self.y)
        self.assertIsInstance(mlr2.coef_, np.ndarray)
        self.assertIsInstance(mlr2.intercept_, numbers.Number)
        self.assertEqual(mlr2.intercept_, 0)
        # determine theoretical standard errors
        hat_matrix = self.X.values @ XTX_inv @ self.X.T.values
        ses = np.sqrt(
            np.diag(
                XTX_inv
                @ (
                    self.X.values.T
                    @ np.diag(residuals**2 / (1 - np.diag(hat_matrix)) ** 2)
                    @ self.X.values
                )
                @ XTX_inv
            )
        )
        coef_adj = lr.coef_ / (ses + 1e-2)
        np.testing.assert_array_almost_equal(mlr2.coef_, coef_adj, decimal=4)

        """ ModifiedLinearRegression with panel bootstrap errors """
        mlr3 = ModifiedLinearRegression(
            method="bootstrap",
            bootstrap_method="panel",
            bootstrap_iters=100,
            fit_intercept=False,
        )
        mlr3.fit(self.X, self.y)
        self.assertIsInstance(mlr3.coef_, np.ndarray)
        self.assertIsInstance(mlr3.intercept_, numbers.Number)
        self.assertEqual(mlr3.intercept_, 0)

        # ModifiedLinearRegression with period bootstrap
        ml4 = ModifiedLinearRegression(
            method="bootstrap",
            bootstrap_method="period",
            bootstrap_iters=100,
            fit_intercept=False,
        )
        ml4.fit(self.X, self.y)
        self.assertIsInstance(ml4.coef_, np.ndarray)
        self.assertIsInstance(ml4.intercept_, numbers.Number)
        self.assertEqual(ml4.intercept_, 0)

        # ModifiedLinearRegression with cross-sectional bootstrap
        mlr5 = ModifiedLinearRegression(
            method="bootstrap",
            bootstrap_method="cross",
            bootstrap_iters=100,
            fit_intercept=False,
        )
        mlr5.fit(self.X, self.y)
        self.assertIsInstance(mlr5.coef_, np.ndarray)
        self.assertIsInstance(mlr5.intercept_, numbers.Number)
        self.assertEqual(mlr5.intercept_, 0)

        # ModifiedLinearRegression with period_per_cross bootstrap
        mlr6 = ModifiedLinearRegression(
            method="bootstrap",
            bootstrap_method="period_per_cross",
            bootstrap_iters=100,
            fit_intercept=False,
        )
        mlr6.fit(self.X, self.y)
        self.assertIsInstance(mlr6.coef_, np.ndarray)
        self.assertIsInstance(mlr6.intercept_, numbers.Number)
        self.assertEqual(mlr6.intercept_, 0)

        # ModifiedLinearRegression with cross_per_period bootstrap
        mlr7 = ModifiedLinearRegression(
            method="bootstrap",
            bootstrap_method="cross_per_period",
            bootstrap_iters=100,
            fit_intercept=False,
        )
        mlr7.fit(self.X, self.y)
        self.assertIsInstance(mlr7.coef_, np.ndarray)
        self.assertIsInstance(mlr7.intercept_, numbers.Number)
        self.assertEqual(mlr7.intercept_, 0)

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
        np.testing.assert_array_equal(
            y_pred, mlr.intercept_ + np.matmul(self.X, mlr.coef_)
        )

    def test_types_create_signal(self):
        mlr = ModifiedLinearRegression(method="analytic")
        mlr.fit(self.X, self.y)
        # X
        self.assertRaises(TypeError, mlr.create_signal, X=1)
        self.assertRaises(TypeError, mlr.create_signal, X="string")
        self.assertRaises(ValueError, mlr.create_signal, X=self.X_nan)
        self.assertRaises(ValueError, mlr.create_signal, X=self.X_nan.reset_index())

    def test_valid_set_params(self):
        mlr = ModifiedLinearRegression(method="analytic")
        mlr.set_params(fit_intercept=False, positive=True, error_offset=1e-3)
        self.assertEqual(mlr.fit_intercept, False)
        self.assertEqual(mlr.positive, True)
        self.assertEqual(mlr.error_offset, 1e-3)
        self.assertEqual(mlr.model.fit_intercept, False)
        self.assertEqual(mlr.model.positive, True)
        mlr.set_params(
            method="bootstrap",
            bootstrap_method="period",
            bootstrap_iters=100,
            resample_ratio=0.5,
            fit_intercept=False,
            positive=True,
        )
        self.assertEqual(mlr.method, "bootstrap")
        self.assertEqual(mlr.bootstrap_method, "period")
        self.assertEqual(mlr.bootstrap_iters, 100)
        self.assertEqual(mlr.resample_ratio, 0.5)
        self.assertEqual(mlr.fit_intercept, False)
        self.assertEqual(mlr.positive, True)


class TestModifiedSignWeightedLinearRegression(unittest.TestCase):
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

        self.sign_weights = SignWeightedLinearRegression()._calculate_sign_weights(
            self.y
        )

    def test_types_init(self):
        # method
        self.assertRaises(TypeError, ModifiedSignWeightedLinearRegression, method=1)
        self.assertRaises(
            ValueError, ModifiedSignWeightedLinearRegression, method="string"
        )
        self.assertRaises(
            ValueError, ModifiedSignWeightedLinearRegression, method="invalid_method"
        )
        # fit_intercept
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="analytic",
            fit_intercept=1,
        )
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            fit_intercept=1,
        )
        # positive
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="analytic",
            positive=1,
        )
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            positive=1,
        )
        # error_offset
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="analytic",
            error_offset="hello",
        )
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            error_offset="hello",
        )
        self.assertRaises(
            ValueError,
            ModifiedSignWeightedLinearRegression,
            method="analytic",
            error_offset=-1,
        )
        self.assertRaises(
            ValueError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            error_offset=-1,
        )
        # bootstrap_panel
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            bootstrap_method=1,
        )
        self.assertRaises(
            ValueError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            bootstrap_method="string",
        )
        # bootstrap_iters
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            bootstrap_iters="string",
        )
        self.assertRaises(
            ValueError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            bootstrap_iters=-1,
        )
        # resample_ratio
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            resample_ratio="string",
        )
        self.assertRaises(
            ValueError,
            ModifiedSignWeightedLinearRegression,
            method="bootstrap",
            resample_ratio=-1,
        )
        # analytic_method
        self.assertRaises(
            TypeError,
            ModifiedSignWeightedLinearRegression,
            method="analytic",
            analytic_method=1,
        )

    def test_valid_init(self):
        # Check defaults set correctly
        lr = ModifiedSignWeightedLinearRegression(method="analytic")
        self.assertEqual(lr.method, "analytic")
        self.assertEqual(lr.fit_intercept, True)
        self.assertEqual(lr.positive, False)
        self.assertEqual(lr.error_offset, 1e-2)
        self.assertEqual(lr.bootstrap_method, "panel")
        self.assertEqual(lr.bootstrap_iters, 1000)
        self.assertEqual(lr.resample_ratio, 1)
        self.assertEqual(lr.analytic_method, None)
        lr = ModifiedSignWeightedLinearRegression(method="bootstrap")
        self.assertEqual(lr.method, "bootstrap")
        self.assertEqual(lr.fit_intercept, True)
        self.assertEqual(lr.positive, False)
        self.assertEqual(lr.error_offset, 1e-2)
        self.assertEqual(lr.bootstrap_method, "panel")
        self.assertEqual(lr.bootstrap_iters, 1000)
        self.assertEqual(lr.resample_ratio, 1)
        self.assertEqual(lr.analytic_method, None)

        # Change defaults
        lr = ModifiedSignWeightedLinearRegression(
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

        lr = ModifiedSignWeightedLinearRegression(
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
        mlr = ModifiedSignWeightedLinearRegression(method="analytic")
        # X
        self.assertRaises(TypeError, mlr.fit, X=1, y=self.y)
        self.assertRaises(TypeError, mlr.fit, X="string", y=self.y)
        self.assertRaises(ValueError, mlr.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, mlr.fit, X=self.X, y=1)
        self.assertRaises(TypeError, mlr.fit, X=self.X, y="string")
        self.assertRaises(ValueError, mlr.fit, X=self.X, y=self.y_nan)

        # Repeat for the bootstrap
        mlr = ModifiedSignWeightedLinearRegression(method="bootstrap")
        # X
        self.assertRaises(TypeError, mlr.fit, X=1, y=self.y)
        self.assertRaises(TypeError, mlr.fit, X="string", y=self.y)
        self.assertRaises(ValueError, mlr.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, mlr.fit, X=self.X, y=1)
        self.assertRaises(TypeError, mlr.fit, X=self.X, y="string")
        self.assertRaises(ValueError, mlr.fit, X=self.X, y=self.y_nan)

    def test_valid_fit(self):
        """ModifiedSignWeightedLinearRegression with usual standard errors"""
        mlr1 = ModifiedSignWeightedLinearRegression(
            method="analytic", error_offset=1e-2, fit_intercept=False
        )
        mlr1.fit(self.X, self.y)
        self.assertIsInstance(mlr1.coef_, np.ndarray)
        self.assertIsInstance(mlr1.intercept_, numbers.Number)
        self.assertEqual(mlr1.intercept_, 0)
        # Check this aligns with theory
        sqrt_weight_matrix = np.diag(np.sqrt(self.sign_weights))
        X_adj = np.matmul(sqrt_weight_matrix, self.X.values)
        y_adj = np.matmul(sqrt_weight_matrix, self.y.values)
        lr = LinearRegression(fit_intercept=False).fit(X_adj, y_adj)
        n = X_adj.shape[0]
        p = X_adj.shape[1]
        y_hat = lr.predict(X_adj)
        residuals = y_adj - y_hat
        sigma2 = np.sum(residuals**2) / (n - p)
        XTX = np.matmul(X_adj.T, X_adj)
        XTX_inv = np.linalg.inv(XTX)
        ses = sigma2 * np.diag(XTX_inv)
        coef_adj = lr.coef_ / (np.sqrt(ses) + 1e-2)
        np.testing.assert_array_almost_equal(mlr1.coef_, coef_adj, decimal=4)

        """ ModifiedSignWeightedLinearRegression with HC3 errors """
        mlr2 = ModifiedSignWeightedLinearRegression(
            method="analytic",
            error_offset=1e-2,
            analytic_method="White",
            fit_intercept=False,
        )
        mlr2.fit(self.X, self.y)
        self.assertIsInstance(mlr2.coef_, np.ndarray)
        self.assertIsInstance(mlr2.intercept_, numbers.Number)
        self.assertEqual(mlr2.intercept_, 0)
        # determine theoretical standard errors
        hat_matrix = X_adj @ XTX_inv @ X_adj.T
        Omega = X_adj.T @ np.diag(residuals**2 / (1 - np.diag(hat_matrix)) ** 2) @ X_adj
        ses = np.sqrt(np.diag(XTX_inv @ Omega @ XTX_inv))
        coef_adj = lr.coef_ / (ses + 1e-2)
        np.testing.assert_array_almost_equal(mlr2.coef_, coef_adj, decimal=4)

        # ModifiedLinearRegression with panel bootstrap
        mlr3 = ModifiedLinearRegression(
            method="bootstrap", bootstrap_method="panel", bootstrap_iters=100
        )
        mlr3.fit(self.X, self.y)
        self.assertIsInstance(mlr3.coef_, np.ndarray)
        self.assertIsInstance(mlr3.intercept_, numbers.Number)

        # ModifiedSignWeightedLinearRegression with period bootstrap
        ml4 = ModifiedSignWeightedLinearRegression(
            method="bootstrap",
            bootstrap_method="period",
            bootstrap_iters=100,
            fit_intercept=False,
        )
        ml4.fit(self.X, self.y)
        self.assertIsInstance(ml4.coef_, np.ndarray)
        self.assertIsInstance(ml4.intercept_, numbers.Number)
        self.assertEqual(ml4.intercept_, 0)

        # ModifiedSignWeightedLinearRegression with cross-sectional bootstrap
        mlr5 = ModifiedSignWeightedLinearRegression(
            method="bootstrap",
            bootstrap_method="cross",
            fit_intercept=False,
            bootstrap_iters=100,
        )
        mlr5.fit(self.X, self.y)
        self.assertIsInstance(mlr5.coef_, np.ndarray)
        self.assertIsInstance(mlr5.intercept_, numbers.Number)
        self.assertEqual(mlr5.intercept_, 0)

        # ModifiedSignWeightedLinearRegression with period_per_cross bootstrap
        mlr6 = ModifiedSignWeightedLinearRegression(
            method="bootstrap",
            bootstrap_method="period_per_cross",
            fit_intercept=False,
            bootstrap_iters=100,
        )
        mlr6.fit(self.X, self.y)
        self.assertIsInstance(mlr6.coef_, np.ndarray)
        self.assertIsInstance(mlr6.intercept_, numbers.Number)
        self.assertEqual(mlr6.intercept_, 0)

        # ModifiedSignWeightedLinearRegression with cross_per_period bootstrap
        mlr7 = ModifiedSignWeightedLinearRegression(
            method="bootstrap",
            fit_intercept=False,
            bootstrap_method="cross_per_period",
            bootstrap_iters=100,
        )
        mlr7.fit(self.X, self.y)
        self.assertIsInstance(mlr7.coef_, np.ndarray)
        self.assertIsInstance(mlr7.intercept_, numbers.Number)
        self.assertEqual(mlr7.intercept_, 0)

    @parameterized.expand(itertools.product([None, "White"], [True, False]))
    def test_types_predict(self, analytic_method, fit_intercept):
        mlr = ModifiedSignWeightedLinearRegression(
            method="analytic",
            fit_intercept=fit_intercept,
            analytic_method=analytic_method,
        )
        mlr.fit(self.X, self.y)
        # X
        self.assertRaises(TypeError, mlr.predict, X=1)
        self.assertRaises(TypeError, mlr.predict, X="string")
        self.assertRaises(ValueError, mlr.predict, X=self.X_nan)
        self.assertRaises(ValueError, mlr.predict, X=self.X_nan.reset_index())

    @parameterized.expand(itertools.product([None, "White"], [True, False]))
    def test_valid_predict(self, analytic_method, fit_intercept):
        mlr = ModifiedSignWeightedLinearRegression(
            method="analytic",
            fit_intercept=fit_intercept,
            analytic_method=analytic_method,
        )
        mlr.fit(self.X, self.y)
        y_pred = mlr.predict(self.X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(self.y))
        np.testing.assert_array_equal(y_pred, mlr.model.predict(self.X))

    @parameterized.expand(itertools.product([None, "White"], [True, False]))
    def test_valid_create_signal(self, analytic_method, fit_intercept):
        mlr = ModifiedSignWeightedLinearRegression(
            method="analytic",
            fit_intercept=fit_intercept,
            analytic_method=analytic_method,
        )
        mlr.fit(self.X, self.y)
        y_pred = mlr.create_signal(self.X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(self.y))
        np.testing.assert_array_equal(
            y_pred, mlr.intercept_ + np.matmul(self.X, mlr.coef_)
        )

    @parameterized.expand(itertools.product([None, "White"], [True, False]))
    def test_types_create_signal(self, analytic_method, fit_intercept):
        mlr = ModifiedSignWeightedLinearRegression(
            method="analytic",
            fit_intercept=fit_intercept,
            analytic_method=analytic_method,
        )
        mlr.fit(self.X, self.y)
        # X
        self.assertRaises(TypeError, mlr.create_signal, X=1)
        self.assertRaises(TypeError, mlr.create_signal, X="string")
        self.assertRaises(ValueError, mlr.create_signal, X=self.X_nan)
        self.assertRaises(ValueError, mlr.create_signal, X=self.X_nan.reset_index())

    def test_valid_set_params(self):
        mlr = ModifiedSignWeightedLinearRegression(method="analytic")
        mlr.set_params(fit_intercept=False, positive=True, error_offset=1e-3)
        self.assertEqual(mlr.fit_intercept, False)
        self.assertEqual(mlr.positive, True)
        self.assertEqual(mlr.error_offset, 1e-3)
        self.assertEqual(mlr.model.fit_intercept, False)
        self.assertEqual(mlr.model.positive, True)
        mlr.set_params(
            method="bootstrap",
            bootstrap_method="period",
            bootstrap_iters=100,
            resample_ratio=0.5,
            fit_intercept=False,
            positive=False,
        )
        self.assertEqual(mlr.method, "bootstrap")
        self.assertEqual(mlr.bootstrap_method, "period")
        self.assertEqual(mlr.bootstrap_iters, 100)
        self.assertEqual(mlr.resample_ratio, 0.5)
        self.assertEqual(mlr.fit_intercept, False)
        self.assertEqual(mlr.positive, False)
        self.assertEqual(mlr.model.fit_intercept, False)
        self.assertEqual(mlr.model.positive, False)


class TestModifiedTimeWeightedLinearRegression(unittest.TestCase):
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

        self.time_weights = TimeWeightedLinearRegression(
            half_life=12
        )._calculate_time_weights(self.y)

    def test_types_init(self):
        # method
        self.assertRaises(TypeError, ModifiedTimeWeightedLinearRegression, method=1)
        self.assertRaises(
            ValueError, ModifiedTimeWeightedLinearRegression, method="string"
        )
        self.assertRaises(
            ValueError, ModifiedTimeWeightedLinearRegression, method="invalid_method"
        )
        # fit_intercept
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="analytic",
            fit_intercept=1,
        )
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            fit_intercept=1,
        )
        # positive
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="analytic",
            positive=1,
        )
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            positive=1,
        )
        # half_life
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="analytic",
            half_life="string",
        )
        self.assertRaises(
            ValueError,
            ModifiedTimeWeightedLinearRegression,
            method="analytic",
            half_life=-1,
        )
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            half_life="string",
        )
        self.assertRaises(
            ValueError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            half_life=-1,
        )
        # error_offset
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="analytic",
            error_offset="hello",
        )
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            error_offset="hello",
        )
        self.assertRaises(
            ValueError,
            ModifiedTimeWeightedLinearRegression,
            method="analytic",
            error_offset=-1,
        )
        self.assertRaises(
            ValueError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            error_offset=-1,
        )
        # bootstrap_panel
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            bootstrap_method=1,
        )
        self.assertRaises(
            ValueError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            bootstrap_method="string",
        )
        # bootstrap_iters
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            bootstrap_iters="string",
        )
        self.assertRaises(
            ValueError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            bootstrap_iters=-1,
        )
        # resample_ratio
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            resample_ratio="string",
        )
        self.assertRaises(
            ValueError,
            ModifiedTimeWeightedLinearRegression,
            method="bootstrap",
            resample_ratio=-1,
        )
        # analytic_method
        self.assertRaises(
            TypeError,
            ModifiedTimeWeightedLinearRegression,
            method="analytic",
            analytic_method=1,
        )

    def test_valid_init(self):
        # Check defaults set correctly
        lr = ModifiedTimeWeightedLinearRegression(method="analytic")
        self.assertEqual(lr.method, "analytic")
        self.assertEqual(lr.fit_intercept, True)
        self.assertEqual(lr.positive, False)
        self.assertEqual(lr.half_life, 252)
        self.assertEqual(lr.error_offset, 1e-2)
        self.assertEqual(lr.bootstrap_method, "panel")
        self.assertEqual(lr.bootstrap_iters, 1000)
        self.assertEqual(lr.resample_ratio, 1)
        self.assertEqual(lr.analytic_method, None)
        lr = ModifiedTimeWeightedLinearRegression(method="bootstrap")
        self.assertEqual(lr.method, "bootstrap")
        self.assertEqual(lr.fit_intercept, True)
        self.assertEqual(lr.positive, False)
        self.assertEqual(lr.half_life, 252)
        self.assertEqual(lr.error_offset, 1e-2)
        self.assertEqual(lr.bootstrap_method, "panel")
        self.assertEqual(lr.bootstrap_iters, 1000)
        self.assertEqual(lr.resample_ratio, 1)
        self.assertEqual(lr.analytic_method, None)

        # Change defaults
        lr = ModifiedTimeWeightedLinearRegression(
            method="analytic",
            fit_intercept=False,
            positive=True,
            half_life=12,
            error_offset=1e-3,
            analytic_method="White",
        )
        self.assertEqual(lr.method, "analytic")
        self.assertEqual(lr.fit_intercept, False)
        self.assertEqual(lr.positive, True)
        self.assertEqual(lr.half_life, 12)
        self.assertEqual(lr.error_offset, 1e-3)
        self.assertEqual(lr.bootstrap_method, "panel")
        self.assertEqual(lr.bootstrap_iters, 1000)
        self.assertEqual(lr.resample_ratio, 1)
        self.assertEqual(lr.analytic_method, "White")

        lr = ModifiedTimeWeightedLinearRegression(
            method="bootstrap",
            fit_intercept=False,
            positive=True,
            half_life=12,
            error_offset=1e-3,
            bootstrap_method="period",
            bootstrap_iters=100,
            resample_ratio=0.5,
        )
        self.assertEqual(lr.method, "bootstrap")
        self.assertEqual(lr.fit_intercept, False)
        self.assertEqual(lr.positive, True)
        self.assertEqual(lr.half_life, 12)
        self.assertEqual(lr.error_offset, 1e-3)
        self.assertEqual(lr.bootstrap_method, "period")
        self.assertEqual(lr.bootstrap_iters, 100)
        self.assertEqual(lr.resample_ratio, 0.5)
        self.assertEqual(lr.analytic_method, None)

    def test_types_fit(self):
        mlr = ModifiedTimeWeightedLinearRegression(method="analytic")
        # X
        self.assertRaises(TypeError, mlr.fit, X=1, y=self.y)
        self.assertRaises(TypeError, mlr.fit, X="string", y=self.y)
        self.assertRaises(ValueError, mlr.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, mlr.fit, X=self.X, y=1)
        self.assertRaises(TypeError, mlr.fit, X=self.X, y="string")
        self.assertRaises(ValueError, mlr.fit, X=self.X, y=self.y_nan)

        # Repeat for the bootstrap
        mlr = ModifiedTimeWeightedLinearRegression(method="bootstrap")
        # X
        self.assertRaises(TypeError, mlr.fit, X=1, y=self.y)
        self.assertRaises(TypeError, mlr.fit, X="string", y=self.y)
        self.assertRaises(ValueError, mlr.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, mlr.fit, X=self.X, y=1)
        self.assertRaises(TypeError, mlr.fit, X=self.X, y="string")
        self.assertRaises(ValueError, mlr.fit, X=self.X, y=self.y_nan)

    def test_valid_fit(self):
        """ModifiedTimeWeightedLinearRegression with usual standard errors"""
        mlr1 = ModifiedTimeWeightedLinearRegression(
            method="analytic", half_life=12, error_offset=1e-2, fit_intercept=False
        )
        mlr1.fit(self.X, self.y)
        self.assertIsInstance(mlr1.coef_, np.ndarray)
        self.assertIsInstance(mlr1.intercept_, numbers.Number)
        self.assertEqual(mlr1.intercept_, 0)
        # Check this aligns with theory
        sqrt_weight_matrix = np.diag(np.sqrt(self.time_weights))
        X_adj = np.matmul(sqrt_weight_matrix, self.X.values)
        y_adj = np.matmul(sqrt_weight_matrix, self.y.values)
        lr = LinearRegression(fit_intercept=False).fit(X_adj, y_adj)
        n = X_adj.shape[0]
        p = X_adj.shape[1]
        y_hat = lr.predict(X_adj)
        residuals = y_adj - y_hat
        sigma2 = np.sum(residuals**2) / (n - p)
        XTX = np.matmul(X_adj.T, X_adj)
        XTX_inv = np.linalg.inv(XTX)
        ses = sigma2 * np.diag(XTX_inv)
        coef_adj = lr.coef_ / (np.sqrt(ses) + 1e-2)
        np.testing.assert_array_almost_equal(mlr1.coef_, coef_adj, decimal=4)

        """ ModifiedTimeWeightedLinearRegression with HC3 errors """
        mlr2 = ModifiedTimeWeightedLinearRegression(
            method="analytic",
            half_life=12,
            error_offset=1e-2,
            analytic_method="White",
            fit_intercept=False,
        )
        mlr2.fit(self.X, self.y)
        self.assertIsInstance(mlr2.coef_, np.ndarray)
        self.assertIsInstance(mlr2.intercept_, numbers.Number)
        self.assertEqual(mlr2.intercept_, 0)
        # determine theoretical standard errors
        hat_matrix = X_adj @ XTX_inv @ X_adj.T
        Omega = X_adj.T @ np.diag(residuals**2 / (1 - np.diag(hat_matrix)) ** 2) @ X_adj
        ses = np.sqrt(np.diag(XTX_inv @ Omega @ XTX_inv))
        coef_adj = lr.coef_ / (ses + 1e-2)
        np.testing.assert_array_almost_equal(mlr2.coef_, coef_adj, decimal=4)

        """ ModifiedTimeWeightedLinearRegression with panel bootstrap errors """
        mlr3 = ModifiedTimeWeightedLinearRegression(
            method="bootstrap",
            half_life=12,
            bootstrap_method="panel",
            bootstrap_iters=100,
        )
        mlr3.fit(self.X, self.y)
        self.assertIsInstance(mlr3.coef_, np.ndarray)
        self.assertIsInstance(mlr3.intercept_, numbers.Number)

        # ModifiedTimeWeightedLinearRegression with period bootstrap
        ml4 = ModifiedTimeWeightedLinearRegression(
            method="bootstrap",
            half_life=12,
            bootstrap_method="period",
            bootstrap_iters=100,
            fit_intercept=False,
        )
        ml4.fit(self.X, self.y)
        self.assertIsInstance(ml4.coef_, np.ndarray)
        self.assertIsInstance(ml4.intercept_, numbers.Number)
        self.assertEqual(ml4.intercept_, 0)

        # ModifiedTimeWeightedLinearRegression with cross-sectional bootstrap
        mlr5 = ModifiedTimeWeightedLinearRegression(
            method="bootstrap",
            half_life=12,
            bootstrap_method="cross",
            fit_intercept=False,
            bootstrap_iters=100,
        )
        mlr5.fit(self.X, self.y)
        self.assertIsInstance(mlr5.coef_, np.ndarray)
        self.assertIsInstance(mlr5.intercept_, numbers.Number)
        self.assertEqual(mlr5.intercept_, 0)

        # ModifiedTimeWeightedLinearRegression with period_per_cross bootstrap
        mlr6 = ModifiedTimeWeightedLinearRegression(
            method="bootstrap",
            half_life=12,
            bootstrap_method="period_per_cross",
            fit_intercept=False,
            bootstrap_iters=100,
        )
        mlr6.fit(self.X, self.y)
        self.assertIsInstance(mlr6.coef_, np.ndarray)
        self.assertIsInstance(mlr6.intercept_, numbers.Number)
        self.assertEqual(mlr6.intercept_, 0)

        # ModifiedTimeWeightedLinearRegression with cross_per_period bootstrap
        mlr7 = ModifiedTimeWeightedLinearRegression(
            method="bootstrap",
            half_life=12,
            fit_intercept=False,
            bootstrap_method="cross_per_period",
            bootstrap_iters=100,
        )
        mlr7.fit(self.X, self.y)
        self.assertIsInstance(mlr7.coef_, np.ndarray)
        self.assertIsInstance(mlr7.intercept_, numbers.Number)
        self.assertEqual(mlr7.intercept_, 0)

    @parameterized.expand(itertools.product([None, "White"], [True, False]))
    def test_types_predict(self, analytic_method, fit_intercept):
        mlr = ModifiedTimeWeightedLinearRegression(
            method="analytic",
            half_life=12,
            fit_intercept=fit_intercept,
            analytic_method=analytic_method,
        )
        mlr.fit(self.X, self.y)
        # X
        self.assertRaises(TypeError, mlr.predict, X=1)
        self.assertRaises(TypeError, mlr.predict, X="string")
        self.assertRaises(ValueError, mlr.predict, X=self.X_nan)
        self.assertRaises(ValueError, mlr.predict, X=self.X_nan.reset_index())

    @parameterized.expand(itertools.product([None, "White"], [True, False]))
    def test_valid_predict(self, analytic_method, fit_intercept):
        mlr = ModifiedTimeWeightedLinearRegression(
            method="analytic",
            half_life=12,
            fit_intercept=fit_intercept,
            analytic_method=analytic_method,
        )
        mlr.fit(self.X, self.y)
        y_pred = mlr.predict(self.X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(self.y))
        np.testing.assert_array_equal(y_pred, mlr.model.predict(self.X))

    @parameterized.expand(itertools.product([None, "White"], [True, False]))
    def test_valid_create_signal(self, analytic_method, fit_intercept):
        mlr = ModifiedTimeWeightedLinearRegression(
            method="analytic",
            half_life=12,
            fit_intercept=fit_intercept,
            analytic_method=analytic_method,
        )
        mlr.fit(self.X, self.y)
        y_pred = mlr.create_signal(self.X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_pred), len(self.y))
        np.testing.assert_array_equal(
            y_pred, mlr.intercept_ + np.matmul(self.X, mlr.coef_)
        )

    @parameterized.expand(itertools.product([None, "White"], [True, False]))
    def test_types_create_signal(self, analytic_method, fit_intercept):
        mlr = ModifiedTimeWeightedLinearRegression(
            method="analytic",
            half_life=12,
            fit_intercept=fit_intercept,
            analytic_method=analytic_method,
        )
        mlr.fit(self.X, self.y)
        # X
        self.assertRaises(TypeError, mlr.create_signal, X=1)
        self.assertRaises(TypeError, mlr.create_signal, X="string")
        self.assertRaises(ValueError, mlr.create_signal, X=self.X_nan)
        self.assertRaises(ValueError, mlr.create_signal, X=self.X_nan.reset_index())

    def test_valid_set_params(self):
        mlr = ModifiedTimeWeightedLinearRegression(method="analytic")
        mlr.set_params(
            fit_intercept=False, positive=True, half_life=12, error_offset=1e-3
        )
        self.assertEqual(mlr.fit_intercept, False)
        self.assertEqual(mlr.positive, True)
        self.assertEqual(mlr.half_life, 12)
        self.assertEqual(mlr.error_offset, 1e-3)
        self.assertEqual(mlr.model.fit_intercept, False)
        self.assertEqual(mlr.model.positive, True)
        mlr.set_params(
            method="bootstrap",
            bootstrap_method="period",
            bootstrap_iters=100,
            resample_ratio=0.5,
            fit_intercept=False,
            positive=False,
        )
        self.assertEqual(mlr.method, "bootstrap")
        self.assertEqual(mlr.bootstrap_method, "period")
        self.assertEqual(mlr.bootstrap_iters, 100)
        self.assertEqual(mlr.resample_ratio, 0.5)
        self.assertEqual(mlr.fit_intercept, False)
        self.assertEqual(mlr.positive, False)
        self.assertEqual(mlr.model.fit_intercept, False)
        self.assertEqual(mlr.model.positive, False)
        self.assertEqual(mlr.half_life, 12)
