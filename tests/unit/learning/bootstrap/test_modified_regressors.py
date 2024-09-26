import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVR

import itertools

import unittest
from unittest.mock import patch

from macrosynergy.learning.predictors.bootstrap import (
    BaseModifiedRegressor,
    ModifiedSignWeightedLinearRegression,
    ModifiedTimeWeightedLinearRegression,
)
from macrosynergy.learning.predictors import (
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
)
from parameterized import parameterized


class TestBaseModifiedRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2000-01-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all elidgible dates
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

    @parameterized.expand(
        itertools.product(
            [LinearRegression(), LinearSVR()],
            ["analytic", "bootstrap"],
            [1e-5, 1e-3, 1e-1],
            ["panel", "period", "cross", "period_per_cross", "cross_per_period"],
            [0.25, 0.5, 1.0],
            ["White"],
        )
    )
    def test_valid_init(
        self,
        model,
        method,
        error_offset,
        bootstrap_method,
        resample_ratio,
        analytic_method,
    ):
        # Check that the class initializes
        try:
            bmr = BaseModifiedRegressor(
                model=model,
                method=method,
                error_offset=error_offset,
                bootstrap_method=bootstrap_method,
                resample_ratio=resample_ratio,
                analytic_method=analytic_method,
            )
        except Exception as e:
            self.fail(
                f"Failed to initialize BaseModifiedRegressor with parameters: {e}"
            )

        # Check that the class attributes have been set correctly
        self.assertEqual(bmr.model, model)
        self.assertEqual(bmr.method, method)
        self.assertEqual(bmr.error_offset, error_offset)
        self.assertEqual(bmr.bootstrap_iters, 100)
        self.assertEqual(bmr.bootstrap_method, bootstrap_method)
        self.assertEqual(bmr.resample_ratio, resample_ratio)
        self.assertEqual(bmr.analytic_method, analytic_method)

    def test_types_init(self):
        # model
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model="invalid_model")
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model=LogisticRegression())

        lr = LinearRegression()
        # method
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model=lr, method=1)
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model=lr, method="invalid_method")

        # error_offset
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(
                model=lr,
                method="analytic",
                error_offset="invalid_offset",
            )
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model=lr, method="analytic", error_offset=-1)

        # bootstrap_method
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model=lr, method="bootstrap", bootstrap_method=1)
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(
                model=lr,
                method="bootstrap",
                bootstrap_method="invalid_method",
            )

        # bootstrap_iters
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(
                model=lr,
                method="bootstrap",
                bootstrap_iters="invalid_iters",
            )
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model=lr, method="bootstrap", bootstrap_iters=-1)

        # resample_ratio
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(
                model=lr,
                method="bootstrap",
                resample_ratio="invalid_ratio",
            )
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model=lr, method="bootstrap", resample_ratio=-1)
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model=lr, method="bootstrap", resample_ratio=1.1)

        # max_features
        with self.assertRaises(NotImplementedError):
            BaseModifiedRegressor(model=lr, method="bootstrap", max_features=1)

        # analytic_method
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model=lr, method="analytic", analytic_method=1)

    def test_types_fit(self):

        lr = LinearRegression()
        bmr = BaseModifiedRegressor(
            model=lr,
            method="analytic",
        )

        # X
        with self.assertRaises(TypeError):
            bmr.fit(X=1, y=self.y)
        with self.assertRaises(TypeError):
            bmr.fit(X="self.X", y=self.y)
        with self.assertRaises(ValueError):
            bmr.fit(X=self.X.iloc[:20], y=self.y)
        with self.assertRaises(ValueError):
            bmr.fit(X=self.X.reset_index(), y=self.y)

        # y
        with self.assertRaises(TypeError):
            bmr.fit(X=self.X, y=1)
        with self.assertRaises(TypeError):
            bmr.fit(X=self.X, y="self.y")
        with self.assertRaises(ValueError):
            bmr.fit(X=self.X, y=self.y.iloc[:20])
        with self.assertRaises(ValueError):
            bmr.fit(X=self.X, y=self.y.reset_index())


class TestModifiedSignWeightedLinearRegression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2000-01-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all elidgible dates
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

    @parameterized.expand(
        itertools.product(
            ["analytic", "bootstrap"],
            [True, False],
            [True, False],
            [1e-5, 1e-3, 1e-1],
            ["panel", "period", "cross", "period_per_cross", "cross_per_period"],
            [0.25, 0.5, 1.0],
            ["White"],
        )
    )
    def test_valid_init(
        self,
        method,
        fit_intercept,
        positive,
        error_offset,
        bootstrap_method,
        resample_ratio,
        analytic_method,
    ):
        # Check that the class initializes
        try:
            mswls = ModifiedSignWeightedLinearRegression(
                fit_intercept=fit_intercept,
                positive=positive,
                method=method,
                error_offset=error_offset,
                bootstrap_method=bootstrap_method,
                resample_ratio=resample_ratio,
                analytic_method=analytic_method,
            )
        except Exception as e:
            self.fail(
                f"Failed to initialize ModifiedSignWeightedLinearRegression with parameters: {e}"
            )

        # Check that the class attributes have been set correctly
        self.assertEqual(mswls.fit_intercept, fit_intercept)
        self.assertEqual(mswls.positive, positive)
        self.assertEqual(mswls.method, method)
        self.assertEqual(mswls.error_offset, error_offset)
        self.assertEqual(mswls.bootstrap_iters, 1000)
        self.assertEqual(mswls.bootstrap_method, bootstrap_method)
        self.assertEqual(mswls.resample_ratio, resample_ratio)
        self.assertEqual(mswls.analytic_method, analytic_method)

    def test_types_init(self):
        # method
        with self.assertRaises(TypeError):
            ModifiedSignWeightedLinearRegression(method=1)
        with self.assertRaises(ValueError):
            ModifiedSignWeightedLinearRegression(method="invalid_method")
        # error_offset
        with self.assertRaises(TypeError):
            ModifiedSignWeightedLinearRegression(
                method="analytic", error_offset="invalid_offset"
            )
        with self.assertRaises(ValueError):
            ModifiedSignWeightedLinearRegression(method="analytic", error_offset=-1)
        # bootstrap_method
        with self.assertRaises(TypeError):
            ModifiedSignWeightedLinearRegression(method="bootstrap", bootstrap_method=1)
        with self.assertRaises(ValueError):
            ModifiedSignWeightedLinearRegression(
                method="bootstrap", bootstrap_method="invalid_method"
            )
        # bootstrap_iters
        with self.assertRaises(TypeError):
            ModifiedSignWeightedLinearRegression(
                method="bootstrap", bootstrap_iters="invalid_iters"
            )
        with self.assertRaises(ValueError):
            ModifiedSignWeightedLinearRegression(method="bootstrap", bootstrap_iters=-1)
        # resample_ratio
        with self.assertRaises(TypeError):
            ModifiedSignWeightedLinearRegression(
                method="bootstrap", resample_ratio="invalid_ratio"
            )
        with self.assertRaises(ValueError):
            ModifiedSignWeightedLinearRegression(method="bootstrap", resample_ratio=-1)
        with self.assertRaises(ValueError):
            ModifiedSignWeightedLinearRegression(method="bootstrap", resample_ratio=1.1)
        # analytic_method
        with self.assertRaises(TypeError):
            ModifiedSignWeightedLinearRegression(method="analytic", analytic_method=1)

    def test_types_fit(self):
        mswls = ModifiedSignWeightedLinearRegression(
            method="analytic",
        )

        # X
        with self.assertRaises(TypeError):
            mswls.fit(X=1, y=self.y)
        with self.assertRaises(TypeError):
            mswls.fit(X="self.X", y=self.y)
        with self.assertRaises(ValueError):
            mswls.fit(X=self.X.iloc[:20], y=self.y)
        with self.assertRaises(ValueError):
            mswls.fit(X=self.X.reset_index(), y=self.y)

        # y
        with self.assertRaises(TypeError):
            mswls.fit(X=self.X, y=1)
        with self.assertRaises(TypeError):
            mswls.fit(X=self.X, y="self.y")
        with self.assertRaises(ValueError):
            mswls.fit(X=self.X, y=self.y.iloc[:20])
        with self.assertRaises(ValueError):
            mswls.fit(X=self.X, y=self.y.reset_index())

    @parameterized.expand(itertools.product([True, False], [True, False]))
    def test_valid_fit_analytic(self, fit_intercept, positive):
        # First check that setting no analytic method works as expected
        mswls = ModifiedSignWeightedLinearRegression(
            method="analytic",
            fit_intercept=fit_intercept,
            positive=positive,
        )
        try:
            mswls.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"Failed to fit ModifiedSignWeightedLinearRegression with parameters: fit_intercept={fit_intercept}, positive={positive}, method=analytic. Error: {e}."
            )

        # Determine analytical expression for standard errors and check that the two align
        model = SignWeightedLinearRegression(
            fit_intercept=fit_intercept, positive=positive
        ).fit(X=self.X, y=self.y)
        if fit_intercept:
            X_new = np.column_stack((np.ones(len(self.X)), self.X.values))
        else:
            X_new = self.X.values

        W = np.diag(model.sample_weights)

        # Calculate the standard errors
        predictions = model.predict(self.X)
        residuals = (self.y - predictions).to_numpy()
        XtWX_inv = np.linalg.inv(X_new.T @ W @ X_new)

        se = np.sqrt(
            np.diag(
                XtWX_inv
                * np.sum(np.square(residuals))
                / (X_new.shape[0] - X_new.shape[1])
            )
        )

        if fit_intercept:
            coef_se = se[1:]
            intercept_se = se[0]
        else:
            coef_se = se
            intercept_se = 0

        # Adjust the coefficients and intercepts by the standard errors
        coef = model.coef_ / (coef_se + 0.01)
        intercept = model.intercept_ / (intercept_se + 0.01)

        np.testing.assert_array_almost_equal(mswls.coef_, coef)
        self.assertAlmostEqual(mswls.intercept_, intercept)

    @parameterized.expand(itertools.product([True, False], [True, False]))
    def test_valid_fit_white(self, fit_intercept, positive):
        # First check that setting no analytic method works as expected
        mswls = ModifiedSignWeightedLinearRegression(
            method="analytic",
            fit_intercept=fit_intercept,
            positive=positive,
            analytic_method="White",
        )
        try:
            mswls.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"Failed to fit ModifiedSignWeightedLinearRegression with parameters: fit_intercept={fit_intercept}, positive={positive}, method=analytic. Error: {e}."
            )

        # Determine analytical expression for standard errors and check that the two align
        model = SignWeightedLinearRegression(
            fit_intercept=fit_intercept, positive=positive
        ).fit(X=self.X, y=self.y)
        if fit_intercept:
            X_new = np.column_stack((np.ones(len(self.X)), self.X.values))
        else:
            X_new = self.X.values

        W = np.diag(np.sqrt(model.sample_weights))

        # Rescale X
        X_new = W @ X_new

        predictions = model.predict(self.X)
        residuals = (self.y - predictions).to_numpy()
        XtX_inv = np.linalg.inv(X_new.T @ X_new)

        # Determine white standard errors
        leverages = np.sum((X_new @ XtX_inv) * X_new, axis=1)
        weights = 1 / (1 - leverages) ** 2
        residuals_squared = np.square(residuals)
        weighted_residuals_squared = weights * residuals_squared
        Omega = X_new.T * weighted_residuals_squared @ X_new
        cov_matrix = XtX_inv @ Omega @ XtX_inv
        se = np.sqrt(np.diag(cov_matrix))

        if fit_intercept:
            coef_se = se[1:]
            intercept_se = se[0]
        else:
            coef_se = se
            intercept_se = 0

        # Adjust the coefficients and intercepts by the standard errors
        coef = model.coef_ / (coef_se + 0.01)
        intercept = model.intercept_ / (intercept_se + 0.01)

        np.testing.assert_array_almost_equal(mswls.coef_, coef, decimal=3)
        self.assertAlmostEqual(mswls.intercept_, intercept, places=3)


class TestModifiedTimeWeightedLinearRegression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2000-01-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all elidgible dates
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

    @parameterized.expand(
        itertools.product(
            ["analytic", "bootstrap"],
            [True, False],
            [True, False],
            [12, 24, 36, 60],
            [1e-5, 1e-3, 1e-1],
            ["panel", "period", "cross", "period_per_cross", "cross_per_period"],
            [0.25, 0.5, 1.0],
            ["White"],
        )
    )
    def test_valid_init(
        self,
        method,
        fit_intercept,
        positive,
        half_life,
        error_offset,
        bootstrap_method,
        resample_ratio,
        analytic_method,
    ):
        # Check that the class initializes
        try:
            mswls = ModifiedTimeWeightedLinearRegression(
                fit_intercept=fit_intercept,
                positive=positive,
                half_life=half_life,
                method=method,
                error_offset=error_offset,
                bootstrap_method=bootstrap_method,
                resample_ratio=resample_ratio,
                analytic_method=analytic_method,
            )
        except Exception as e:
            self.fail(
                f"Failed to initialize ModifiedTimeWeightedLinearRegression with parameters: {e}"
            )

        # Check that the class attributes have been set correctly
        self.assertEqual(mswls.fit_intercept, fit_intercept)
        self.assertEqual(mswls.positive, positive)
        self.assertEqual(mswls.half_life, half_life)
        self.assertEqual(mswls.method, method)
        self.assertEqual(mswls.error_offset, error_offset)
        self.assertEqual(mswls.bootstrap_iters, 1000)
        self.assertEqual(mswls.bootstrap_method, bootstrap_method)
        self.assertEqual(mswls.resample_ratio, resample_ratio)
        self.assertEqual(mswls.analytic_method, analytic_method)

    def test_types_init(self):
        # fit_intercept
        with self.assertRaises(TypeError):
            ModifiedTimeWeightedLinearRegression(method="analytic", fit_intercept=1)
        # positive
        with self.assertRaises(TypeError):
            ModifiedTimeWeightedLinearRegression(method="analytic", positive=1)
        # half_life
        with self.assertRaises(TypeError):
            ModifiedTimeWeightedLinearRegression(
                method="analytic", half_life="invalid_half_life"
            )
        with self.assertRaises(ValueError):
            ModifiedTimeWeightedLinearRegression(method="analytic", half_life=-1)
        # method
        with self.assertRaises(TypeError):
            ModifiedTimeWeightedLinearRegression(method=1)
        with self.assertRaises(ValueError):
            ModifiedTimeWeightedLinearRegression(method="invalid_method")
        # error_offset
        with self.assertRaises(TypeError):
            ModifiedTimeWeightedLinearRegression(
                method="analytic", error_offset="invalid_offset"
            )
        with self.assertRaises(ValueError):
            ModifiedTimeWeightedLinearRegression(method="analytic", error_offset=-1)
        # bootstrap_method
        with self.assertRaises(TypeError):
            ModifiedTimeWeightedLinearRegression(method="bootstrap", bootstrap_method=1)
        with self.assertRaises(ValueError):
            ModifiedTimeWeightedLinearRegression(
                method="bootstrap", bootstrap_method="invalid_method"
            )
        # bootstrap_iters
        with self.assertRaises(TypeError):
            ModifiedTimeWeightedLinearRegression(
                method="bootstrap", bootstrap_iters="invalid_iters"
            )
        with self.assertRaises(ValueError):
            ModifiedTimeWeightedLinearRegression(method="bootstrap", bootstrap_iters=-1)
        # resample_ratio
        with self.assertRaises(TypeError):
            ModifiedTimeWeightedLinearRegression(
                method="bootstrap", resample_ratio="invalid_ratio"
            )
        with self.assertRaises(ValueError):
            ModifiedTimeWeightedLinearRegression(method="bootstrap", resample_ratio=-1)
        with self.assertRaises(ValueError):
            ModifiedTimeWeightedLinearRegression(method="bootstrap", resample_ratio=1.1)
        # analytic_method
        with self.assertRaises(TypeError):
            ModifiedTimeWeightedLinearRegression(method="analytic", analytic_method=1)

    def test_types_fit(self):
        mswls = ModifiedTimeWeightedLinearRegression(
            method="analytic",
        )

        # X
        with self.assertRaises(TypeError):
            mswls.fit(X=1, y=self.y)
        with self.assertRaises(TypeError):
            mswls.fit(X="self.X", y=self.y)
        with self.assertRaises(ValueError):
            mswls.fit(X=self.X.iloc[:20], y=self.y)
        with self.assertRaises(ValueError):
            mswls.fit(X=self.X.reset_index(), y=self.y)

        # y
        with self.assertRaises(TypeError):
            mswls.fit(X=self.X, y=1)
        with self.assertRaises(TypeError):
            mswls.fit(X=self.X, y="self.y")
        with self.assertRaises(ValueError):
            mswls.fit(X=self.X, y=self.y.iloc[:20])
        with self.assertRaises(ValueError):
            mswls.fit(X=self.X, y=self.y.reset_index())

    @parameterized.expand(
        itertools.product([True, False], [True, False], [12, 24, 36, 60])
    )
    def test_valid_fit_analytic(self, fit_intercept, positive, half_life):
        # First check that setting no analytic method works as expected
        mswls = ModifiedTimeWeightedLinearRegression(
            method="analytic",
            fit_intercept=fit_intercept,
            positive=positive,
            half_life=half_life,
        )
        try:
            mswls.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"Failed to fit ModifiedTimeWeightedLinearRegression with parameters: fit_intercept={fit_intercept}, positive={positive}, method=analytic. Error: {e}."
            )

        # Determine analytical expression for standard errors and check that the two align
        model = TimeWeightedLinearRegression(
            fit_intercept=fit_intercept, positive=positive, half_life=half_life
        ).fit(X=self.X, y=self.y)
        if fit_intercept:
            X_new = np.column_stack((np.ones(len(self.X)), self.X.values))
        else:
            X_new = self.X.values

        W = np.diag(model.sample_weights)

        # Calculate the standard errors
        predictions = model.predict(self.X)
        residuals = (self.y - predictions).to_numpy()
        XtWX_inv = np.linalg.inv(X_new.T @ W @ X_new)

        se = np.sqrt(
            np.diag(
                XtWX_inv
                * np.sum(np.square(residuals))
                / (X_new.shape[0] - X_new.shape[1])
            )
        )

        if fit_intercept:
            coef_se = se[1:]
            intercept_se = se[0]
        else:
            coef_se = se
            intercept_se = 0

        # Adjust the coefficients and intercepts by the standard errors
        coef = model.coef_ / (coef_se + 0.01)
        intercept = model.intercept_ / (intercept_se + 0.01)

        np.testing.assert_array_almost_equal(mswls.coef_, coef, decimal=3)
        self.assertAlmostEqual(mswls.intercept_, intercept, places=3)

    @parameterized.expand(
        itertools.product([True, False], [True, False], [12, 24, 36, 60])
    )
    def test_valid_fit_white(self, fit_intercept, positive, half_life):
        # First check that setting no analytic method works as expected
        mswls = ModifiedTimeWeightedLinearRegression(
            method="analytic",
            fit_intercept=fit_intercept,
            positive=positive,
            half_life=half_life,
            analytic_method="White",
        )
        try:
            mswls.fit(X=self.X, y=self.y)
        except Exception as e:
            self.fail(
                f"Failed to fit ModifiedTimeWeightedLinearRegression with parameters: fit_intercept={fit_intercept}, positive={positive}, method=analytic. Error: {e}."
            )

        # Determine analytical expression for standard errors and check that the two align
        model = TimeWeightedLinearRegression(
            fit_intercept=fit_intercept, positive=positive, half_life=half_life
        ).fit(X=self.X, y=self.y)
        if fit_intercept:
            X_new = np.column_stack((np.ones(len(self.X)), self.X.values))
        else:
            X_new = self.X.values

        W = np.diag(np.sqrt(model.sample_weights))

        # Rescale X
        X_new = W @ X_new

        predictions = model.predict(self.X)
        residuals = (self.y - predictions).to_numpy()
        XtX_inv = np.linalg.inv(X_new.T @ X_new)

        # Determine white standard errors
        leverages = np.sum((X_new @ XtX_inv) * X_new, axis=1)
        weights = 1 / (1 - leverages) ** 2
        residuals_squared = np.square(residuals)
        weighted_residuals_squared = weights * residuals_squared
        Omega = X_new.T * weighted_residuals_squared @ X_new
        cov_matrix = XtX_inv @ Omega @ XtX_inv
        se = np.sqrt(np.diag(cov_matrix))

        if fit_intercept:
            coef_se = se[1:]
            intercept_se = se[0]
        else:
            coef_se = se
            intercept_se = 0

        # Adjust the coefficients and intercepts by the standard errors
        coef = model.coef_ / (coef_se + 0.01)
        intercept = model.intercept_ / (intercept_se + 0.01)

        np.testing.assert_array_almost_equal(mswls.coef_, coef, decimal=3)
        self.assertAlmostEqual(mswls.intercept_, intercept, places=3)
