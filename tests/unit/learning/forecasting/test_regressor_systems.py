import numpy as np
import pandas as pd
from scipy.stats import expon
import itertools

import unittest

from macrosynergy.learning import (
    LADRegressor,
    LinearRegressionSystem,
    LADRegressionSystem,
    RidgeRegressionSystem,
    CorrelationVolatilitySystem,
)
from macrosynergy.learning.forecasting.model_systems.base_regression_system import (
    BaseRegressionSystem,
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)

from sklearn.base import BaseEstimator, RegressorMixin

from parameterized import parameterized


class TestLinearRegressionSystem(unittest.TestCase):
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

    def test_types_fit(self):
        """
        Test validity of parameters entered into the fit method.
        """
        model = LinearRegressionSystem()
        # X
        self.assertRaises(TypeError, model.fit, X=1, y=self.y)
        self.assertRaises(TypeError, model.fit, X="string", y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, model.fit, X=self.X, y=1)
        self.assertRaises(TypeError, model.fit, X=self.X, y="string")
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y_nan)

    @parameterized.expand([True])  # , False])
    def test_valid_fit(self, single_feature):
        """Check the model fits are as expected"""
        cross_sections = self.X.index.get_level_values(0).unique()
        param_names = [
            "roll",
            "fit_intercept",
            "positive",
            "data_freq",
            "min_xs_samples",
        ]
        param_values = list(
            itertools.product(
                [None, 21, 21 * 6, 21 * 12],  # roll
                [True, False],  # fit_intercept
                [True, False],  # positive
                [None, "unadjusted"],  # data_freq
                [2, 21 * 15],  # min_xs_samples
            )
        )
        for params in param_values:
            param_dict = {name: param for name, param in zip(param_names, params)}
            system_model = LinearRegressionSystem().set_params(**param_dict)
            if single_feature:
                system_model.fit(pd.DataFrame(self.X.iloc[:, 0]), self.y)
            # else:
            #     system_model.fit(pd.DataFrame(self.X), self.y)
            system_coefs = system_model.coefs_
            system_intercepts = system_model.intercepts_
            for cid in cross_sections:
                model = LinearRegression().set_params(
                    **{
                        key: value
                        for key, value in param_dict.items()
                        if key in ["fit_intercept", "positive"]
                    }
                )
                unique_dates = sorted(self.X.xs(cid).index.unique())
                if params[4] > len(unique_dates):
                    continue
                if params[0] is not None:
                    roll_dates = unique_dates[-params[0] :]
                    if len(roll_dates) >= len(unique_dates):
                        continue
                    if single_feature:
                        X = self.X.iloc[:, 0][
                            self.X.index.get_level_values(1).isin(roll_dates)
                        ]
                        y = self.y[self.y.index.get_level_values(1).isin(roll_dates)]
                    # else:
                    #     X = self.X[self.X.index.get_level_values(1).isin(roll_dates)]
                    #     y = self.y[self.y.index.get_level_values(1).isin(roll_dates)]
                else:
                    if single_feature:
                        X = self.X.iloc[:, 0]
                        y = self.y
                    # else:
                    #     X = self.X
                    #     y = self.y

                if single_feature:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # else:
                #     model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # Check that the model coefficients and intercepts are the same
                np.testing.assert_almost_equal(model.intercept_, system_intercepts[cid])
                np.testing.assert_almost_equal(model.coef_, system_coefs[cid])

        # Now test that weekly frequency works as expected
        param_values = list(
            itertools.product(
                [None, 5 * 3, 5 * 6, 5 * 12],  # roll
                [True, False],  # fit_intercept
                [True, False],  # positive
                ["W"],  # data_freq
                [2],  # min_xs_samples
            )
        )

        for params in param_values:
            param_dict = {name: param for name, param in zip(param_names, params)}
            system_model = LinearRegressionSystem().set_params(**param_dict)
            if single_feature:
                system_model.fit(pd.DataFrame(self.X.iloc[:, 0]), self.y)
            else:
                system_model.fit(pd.DataFrame(self.X), self.y)
            system_coefs = system_model.coefs_
            system_intercepts = system_model.intercepts_
            for cid in cross_sections:
                # First downsample the data to weekly
                X = self.X.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="W"),
                    ]
                ).sum()
                y = self.y.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="W"),
                    ]
                ).sum()
                model = LinearRegression().set_params(
                    **{
                        key: value
                        for key, value in param_dict.items()
                        if key in ["fit_intercept", "positive"]
                    }
                )
                unique_dates = sorted(X.xs(cid).index.unique())
                if params[4] > len(unique_dates):
                    continue
                if params[0] is not None:
                    roll_dates = unique_dates[-params[0] :]
                    if len(roll_dates) >= len(unique_dates):
                        continue
                    if single_feature:
                        X = X.iloc[:, 0][X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                    else:
                        X = X[X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                else:
                    if single_feature:
                        X = X.iloc[:, 0]

                if single_feature:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                else:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # Check that the model coefficients and intercepts are the same
                np.testing.assert_almost_equal(
                    model.intercept_, system_intercepts[cid], decimal=5
                )
                np.testing.assert_almost_equal(
                    model.coef_, system_coefs[cid], decimal=5
                )

        # Now test that monthly frequency works as expected
        param_values = list(
            itertools.product(
                [None, 6, 12],  # roll
                [True, False],  # fit_intercept
                [True, False],  # positive
                ["M"],  # data_freq
                [2],  # min_xs_samples
            )
        )

        for params in param_values:
            param_dict = {name: param for name, param in zip(param_names, params)}
            system_model = LinearRegressionSystem().set_params(**param_dict)
            if single_feature:
                system_model.fit(pd.DataFrame(self.X.iloc[:, 0]), self.y)
            else:
                system_model.fit(pd.DataFrame(self.X), self.y)
            system_coefs = system_model.coefs_
            system_intercepts = system_model.intercepts_
            for cid in cross_sections:
                # First downsample the data to weekly
                X = self.X.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="M"),
                    ]
                ).sum()
                y = self.y.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="M"),
                    ]
                ).sum()
                model = LinearRegression().set_params(
                    **{
                        key: value
                        for key, value in param_dict.items()
                        if key in ["fit_intercept", "positive"]
                    }
                )
                unique_dates = sorted(X.xs(cid).index.unique())
                if params[4] > len(unique_dates):
                    continue
                if params[0] is not None:
                    unique_dates = sorted(y.xs(cid).index.unique())
                    roll_dates = unique_dates[-params[0] :]
                    if len(roll_dates) >= len(unique_dates):
                        continue
                    if single_feature:
                        X = X.iloc[:, 0][X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                    else:
                        X = X[X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                else:
                    if single_feature:
                        X = X.iloc[:, 0]

                if single_feature:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                else:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # Check that the model coefficients and intercepts are the same
                np.testing.assert_almost_equal(
                    model.intercept_, system_intercepts[cid], decimal=5
                )
                np.testing.assert_almost_equal(
                    model.coef_, system_coefs[cid], decimal=5
                )

    def test_valid_predict(self):
        pass

    def test_check_init_params(self):
        # Test default params
        model = LinearRegressionSystem()
        self.assertEqual(model.roll, "full")
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.positive, False)
        self.assertEqual(model.data_freq, None)
        self.assertEqual(model.min_xs_samples, 2)

        # Test custom params
        model = LinearRegressionSystem(
            roll=5,
            fit_intercept=False,
            positive=True,
            data_freq="M",
            min_xs_samples=3,
        )
        self.assertEqual(model.roll, 5)
        self.assertEqual(model.fit_intercept, False)
        self.assertEqual(model.positive, True)
        self.assertEqual(model.data_freq, "M")
        self.assertEqual(model.min_xs_samples, 3)

    def test_invalid_params(self):
        with self.assertRaises(TypeError):
            LinearRegressionSystem(roll=5.5)
        with self.assertRaises(ValueError):
            LinearRegressionSystem(roll=-5)
        with self.assertRaises(TypeError):
            LinearRegressionSystem(roll=None)
        with self.assertRaises(TypeError):
            LinearRegressionSystem(fit_intercept="False")
        with self.assertRaises(TypeError):
            LinearRegressionSystem(positive="True")
        with self.assertRaises(TypeError):
            LinearRegressionSystem(data_freq=5)
        with self.assertRaises(ValueError):
            LinearRegressionSystem(data_freq="hello")
        with self.assertRaises(TypeError):
            LinearRegressionSystem(min_xs_samples="2")
        with self.assertRaises(ValueError):
            LinearRegressionSystem(min_xs_samples=-5)
        with self.assertRaises(ValueError):
            LinearRegressionSystem(min_xs_samples=0)
        with self.assertRaises(ValueError):
            LinearRegressionSystem(min_xs_samples=1)
        with self.assertRaises(TypeError):
            LinearRegressionSystem(min_xs_samples=3.7)

    @parameterized.expand(
        [
            (5, True, False, "unadjusted", 2),
            (5, False, True, "unadjusted", 2),
            (5, True, True, "M", 2),
        ]
    )
    def test_valid_init(self, roll, fit_intercept, positive, data_freq, min_xs_samples):
        # Test default params
        try:
            model = LinearRegressionSystem()
        except Exception as e:
            self.fail(
                "LinearRegressionSystem constructor raised an exception: {}".format(e)
            )
        self.assertEqual(model.roll, "full")
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.positive, False)
        self.assertEqual(model.data_freq, None)
        self.assertEqual(model.min_xs_samples, 2)
        self.assertIsInstance(model, LinearRegressionSystem)
        self.assertIsInstance(model, BaseRegressionSystem)
        self.assertIsInstance(model, RegressorMixin)
        self.assertIsInstance(model, BaseEstimator)

        try:
            model = LinearRegressionSystem(
                roll=roll,
                fit_intercept=fit_intercept,
                positive=positive,
                data_freq=data_freq,
                min_xs_samples=min_xs_samples,
            )
        except Exception as e:
            self.fail(
                "LinearRegressionSystem constructor raised an exception: {}".format(e)
            )
        self.assertEqual(model.roll, roll)
        self.assertEqual(model.fit_intercept, fit_intercept)
        self.assertEqual(model.positive, positive)
        self.assertEqual(model.data_freq, data_freq)
        self.assertEqual(model.min_xs_samples, min_xs_samples)
        self.assertIsInstance(model, LinearRegressionSystem)
        self.assertIsInstance(model, BaseRegressionSystem)
        self.assertIsInstance(model, RegressorMixin)
        self.assertIsInstance(model, BaseEstimator)

    @parameterized.expand(
        [
            ("full", True, False, "unadjusted", 2),
            ("full", False, False, "unadjusted", 2),
            ("full", False, True, "unadjusted", 2),
            ("full", True, True, "unadjusted", 2),
            (5, True, False, "unadjusted", 2),
            (5, False, False, "unadjusted", 2),
            (5, False, True, "unadjusted", 2),
            (5, True, True, "unadjusted", 2),
            (21, True, False, "unadjusted", 2),
            (21, False, False, "unadjusted", 2),
            (21, False, True, "unadjusted", 2),
            (21, True, True, "unadjusted", 2),
        ]
    )
    def test_create_model(
        self, roll, fit_intercept, positive, data_freq, min_xs_samples
    ):
        model = LinearRegressionSystem(
            roll=roll,
            fit_intercept=fit_intercept,
            positive=positive,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
        )
        self.assertIsInstance(model.create_model(), LinearRegression)
        self.assertEqual(model.create_model().fit_intercept, fit_intercept)
        self.assertEqual(model.create_model().positive, positive)

    @parameterized.expand(itertools.product(["full", 5], ["unadjusted", "W", "M", "Q"]))
    def test_fit(self, roll, data_freq):
        model = LinearRegressionSystem(roll=roll, data_freq=data_freq)
        model.fit(self.X, self.y)
        self.assertIsInstance(model.coefs_, dict)
        self.assertIsInstance(model.intercepts_, dict)

    def test_predict(self):
        model = LinearRegressionSystem()
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        self.assertIsInstance(y_pred, pd.Series)
        self.assertEqual(len(y_pred), len(self.y))

    def test_types_predict(self):
        model = LinearRegressionSystem()
        model.fit(self.X, self.y)
        self.assertRaises(TypeError, model.predict, X=1)
        self.assertRaises(ValueError, model.predict, X=pd.DataFrame())
        lad = LinearRegressionSystem().fit(self.X, self.y)
        self.assertRaises(TypeError, lad.predict, X=1)
        self.assertRaises(TypeError, lad.predict, X="X")
        self.assertRaises(ValueError, lad.predict, X=self.X.iloc[:, :-1])
        self.assertRaises(ValueError, lad.predict, X=self.X_nan)
        self.assertRaises(TypeError, lad.predict, X=self.X_nan.values)

    def test_valid_predict(self):
        lad = LinearRegressionSystem().fit(self.X, self.y)
        y_pred_system = lad.predict(self.X)
        self.assertIsInstance(y_pred_system, pd.Series)
        self.assertEqual(len(y_pred_system), len(self.y))

        # Loop through all cross sections and check that the predictions are correct
        for cid in self.X.index.get_level_values(0).unique():
            X = self.X.xs(cid, level=0)
            y = self.y.xs(cid, level=0)
            lad = LinearRegression().fit(X, y)
            y_pred_lad = lad.predict(X)
            np.testing.assert_almost_equal(
                y_pred_lad, y_pred_system.xs(cid, level=0).values, decimal=2
            )


class TestLADRegressionSystem(unittest.TestCase):
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

    def test_types_fit(self):
        """
        Test validity of parameters entered into the fit method.
        """
        model = LADRegressionSystem()
        # X
        self.assertRaises(TypeError, model.fit, X=1, y=self.y)
        self.assertRaises(TypeError, model.fit, X="string", y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, model.fit, X=self.X, y=1)
        self.assertRaises(TypeError, model.fit, X=self.X, y="string")
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y_nan)

    def test_types_fit(self):
        """
        Test validity of parameters entered into the fit method.
        """
        model = LADRegressionSystem()
        # X
        self.assertRaises(TypeError, model.fit, X=1, y=self.y)
        self.assertRaises(TypeError, model.fit, X="string", y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, model.fit, X=self.X, y=1)
        self.assertRaises(TypeError, model.fit, X=self.X, y="string")
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y_nan)

    @parameterized.expand([True])  # , False])
    def test_valid_fit(self, single_feature):
        """Check the model fits are as expected"""
        cross_sections = self.X.index.get_level_values(0).unique()
        param_names = [
            "roll",
            "fit_intercept",
            "positive",
            "data_freq",
            "min_xs_samples",
        ]
        param_values = list(
            itertools.product(
                [None, 21, 21 * 6, 21 * 12],  # roll
                [True, False],  # fit_intercept
                [True, False],  # positive
                [None, "unadjusted"],  # data_freq
                [2, 21 * 15],  # min_xs_samples
            )
        )
        for params in param_values:
            param_dict = {name: param for name, param in zip(param_names, params)}
            system_model = LADRegressionSystem().set_params(**param_dict)
            if single_feature:
                system_model.fit(pd.DataFrame(self.X.iloc[:, 0]), self.y)
            # else:
            #     system_model.fit(pd.DataFrame(self.X), self.y)
            system_coefs = system_model.coefs_
            system_intercepts = system_model.intercepts_
            for cid in cross_sections:
                model = LADRegressor().set_params(
                    **{
                        key: value
                        for key, value in param_dict.items()
                        if key in ["fit_intercept", "positive"]
                    }
                )
                unique_dates = sorted(self.X.xs(cid).index.unique())
                if params[4] > len(unique_dates):
                    continue
                if params[0] is not None:
                    roll_dates = unique_dates[-params[0] :]
                    if len(roll_dates) >= len(unique_dates):
                        continue
                    if single_feature:
                        X = self.X.iloc[:, 0][
                            self.X.index.get_level_values(1).isin(roll_dates)
                        ]
                        y = self.y[self.y.index.get_level_values(1).isin(roll_dates)]
                    # else:
                    #     X = self.X[self.X.index.get_level_values(1).isin(roll_dates)]
                    #     y = self.y[self.y.index.get_level_values(1).isin(roll_dates)]
                else:
                    if single_feature:
                        X = self.X.iloc[:, 0]
                        y = self.y
                    # else:
                    #     X = self.X
                    #     y = self.y

                if single_feature:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # else:
                #     model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # Check that the model coefficients and intercepts are the same
                np.testing.assert_almost_equal(model.intercept_, system_intercepts[cid])
                np.testing.assert_almost_equal(model.coef_, system_coefs[cid])

        # Now test that weekly frequency works as expected
        param_values = list(
            itertools.product(
                [None, 5 * 3, 5 * 6, 5 * 12],  # roll
                [True, False],  # fit_intercept
                [True, False],  # positive
                ["W"],  # data_freq
                [2],  # min_xs_samples
            )
        )

        for params in param_values:
            param_dict = {name: param for name, param in zip(param_names, params)}
            system_model = LADRegressionSystem().set_params(**param_dict)
            if single_feature:
                system_model.fit(pd.DataFrame(self.X.iloc[:, 0]), self.y)
            else:
                system_model.fit(pd.DataFrame(self.X), self.y)
            system_coefs = system_model.coefs_
            system_intercepts = system_model.intercepts_
            for cid in cross_sections:
                # First downsample the data to weekly
                X = self.X.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="W"),
                    ]
                ).sum()
                y = self.y.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="W"),
                    ]
                ).sum()
                model = LADRegressor().set_params(
                    **{
                        key: value
                        for key, value in param_dict.items()
                        if key in ["fit_intercept", "positive"]
                    }
                )
                unique_dates = sorted(X.xs(cid).index.unique())
                if params[4] > len(unique_dates):
                    continue
                if params[0] is not None:
                    roll_dates = unique_dates[-params[0] :]
                    if len(roll_dates) >= len(unique_dates):
                        continue
                    if single_feature:
                        X = X.iloc[:, 0][X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                    else:
                        X = X[X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                else:
                    if single_feature:
                        X = X.iloc[:, 0]

                if single_feature:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                else:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # Check that the model coefficients and intercepts are the same
                np.testing.assert_almost_equal(
                    model.intercept_, system_intercepts[cid], decimal=5
                )
                np.testing.assert_almost_equal(
                    model.coef_, system_coefs[cid], decimal=5
                )

        # Now test that monthly frequency works as expected
        param_values = list(
            itertools.product(
                [None, 6, 12],  # roll
                [True, False],  # fit_intercept
                [True, False],  # positive
                ["M"],  # data_freq
                [2],  # min_xs_samples
            )
        )

        for params in param_values:
            param_dict = {name: param for name, param in zip(param_names, params)}
            system_model = LADRegressionSystem().set_params(**param_dict)
            if single_feature:
                system_model.fit(pd.DataFrame(self.X.iloc[:, 0]), self.y)
            else:
                system_model.fit(pd.DataFrame(self.X), self.y)
            system_coefs = system_model.coefs_
            system_intercepts = system_model.intercepts_
            for cid in cross_sections:
                # First downsample the data to weekly
                X = self.X.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="M"),
                    ]
                ).sum()
                y = self.y.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="M"),
                    ]
                ).sum()
                model = LADRegressor().set_params(
                    **{
                        key: value
                        for key, value in param_dict.items()
                        if key in ["fit_intercept", "positive"]
                    }
                )
                unique_dates = sorted(X.xs(cid).index.unique())
                if params[4] > len(unique_dates):
                    continue
                if params[0] is not None:
                    unique_dates = sorted(y.xs(cid).index.unique())
                    roll_dates = unique_dates[-params[0] :]
                    if len(roll_dates) >= len(unique_dates):
                        continue
                    if single_feature:
                        X = X.iloc[:, 0][X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                    else:
                        X = X[X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                else:
                    if single_feature:
                        X = X.iloc[:, 0]

                if single_feature:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                else:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # Check that the model coefficients and intercepts are the same
                np.testing.assert_almost_equal(
                    model.intercept_, system_intercepts[cid], decimal=5
                )
                np.testing.assert_almost_equal(
                    model.coef_, system_coefs[cid], decimal=5
                )

    def test_types_predict(self):
        lad = LADRegressionSystem().fit(self.X, self.y)
        self.assertRaises(TypeError, lad.predict, X=1)
        self.assertRaises(TypeError, lad.predict, X="X")
        self.assertRaises(ValueError, lad.predict, X=self.X.iloc[:, :-1])
        self.assertRaises(ValueError, lad.predict, X=self.X_nan)
        self.assertRaises(TypeError, lad.predict, X=self.X_nan.values)

    def test_valid_predict(self):
        lad = LADRegressionSystem().fit(self.X, self.y)
        y_pred_system = lad.predict(self.X)
        self.assertIsInstance(y_pred_system, pd.Series)
        self.assertEqual(len(y_pred_system), len(self.y))

        # Loop through all cross sections and check that the predictions are correct
        for cid in self.X.index.get_level_values(0).unique():
            X = self.X.xs(cid, level=0)
            y = self.y.xs(cid, level=0)
            lad = LADRegressor().fit(X, y)
            y_pred_lad = lad.predict(X)
            np.testing.assert_almost_equal(
                y_pred_lad, y_pred_system.xs(cid, level=0).values, decimal=2
            )

    def test_check_init_params(self):
        # Test default params
        model = LADRegressionSystem()
        self.assertEqual(model.roll, "full")
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.positive, False)
        self.assertEqual(model.data_freq, None)
        self.assertEqual(model.min_xs_samples, 2)

        # Test custom params
        model = LADRegressionSystem(
            roll=5,
            fit_intercept=False,
            positive=True,
            data_freq="M",
            min_xs_samples=3,
        )
        self.assertEqual(model.roll, 5)
        self.assertEqual(model.fit_intercept, False)
        self.assertEqual(model.positive, True)
        self.assertEqual(model.data_freq, "M")
        self.assertEqual(model.min_xs_samples, 3)

    def test_invalid_params(self):
        with self.assertRaises(TypeError):
            LADRegressionSystem(roll=5.5)
        with self.assertRaises(ValueError):
            LADRegressionSystem(roll=-5)
        with self.assertRaises(TypeError):
            LADRegressionSystem(roll=None)
        with self.assertRaises(TypeError):
            LADRegressionSystem(fit_intercept="False")
        with self.assertRaises(TypeError):
            LADRegressionSystem(positive="True")
        with self.assertRaises(TypeError):
            LADRegressionSystem(data_freq=5)
        with self.assertRaises(ValueError):
            LADRegressionSystem(data_freq="hello")
        with self.assertRaises(TypeError):
            LADRegressionSystem(min_xs_samples="2")
        with self.assertRaises(ValueError):
            LADRegressionSystem(min_xs_samples=-5)
        with self.assertRaises(ValueError):
            LADRegressionSystem(min_xs_samples=0)
        with self.assertRaises(ValueError):
            LADRegressionSystem(min_xs_samples=1)
        with self.assertRaises(TypeError):
            LADRegressionSystem(min_xs_samples=3.7)

    @parameterized.expand(
        [
            (5, True, False, "unadjusted", 2),
            (5, False, True, "unadjusted", 2),
            (5, True, True, "M", 2),
        ]
    )
    def test_valid_init(self, roll, fit_intercept, positive, data_freq, min_xs_samples):
        # Test default params
        try:
            model = LADRegressionSystem()
        except Exception as e:
            self.fail(
                "LADRegressionSystem constructor raised an exception: {}".format(e)
            )
        self.assertEqual(model.roll, "full")
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.positive, False)
        self.assertEqual(model.data_freq, None)
        self.assertEqual(model.min_xs_samples, 2)
        self.assertIsInstance(model, LADRegressionSystem)
        self.assertIsInstance(model, BaseRegressionSystem)
        self.assertIsInstance(model, RegressorMixin)
        self.assertIsInstance(model, BaseEstimator)
        try:
            model = LADRegressionSystem(
                roll=roll,
                fit_intercept=fit_intercept,
                positive=positive,
                data_freq=data_freq,
                min_xs_samples=min_xs_samples,
            )
        except Exception as e:
            self.fail(
                "LADRegressionSystem constructor raised an exception: {}".format(e)
            )
        self.assertEqual(model.roll, roll)
        self.assertEqual(model.fit_intercept, fit_intercept)
        self.assertEqual(model.positive, positive)
        self.assertEqual(model.data_freq, data_freq)
        self.assertEqual(model.min_xs_samples, min_xs_samples)
        self.assertIsInstance(model, LADRegressionSystem)
        self.assertIsInstance(model, BaseRegressionSystem)
        self.assertIsInstance(model, RegressorMixin)
        self.assertIsInstance(model, BaseEstimator)

    @parameterized.expand(
        [
            ("full", True, False, "unadjusted", 2),
            ("full", False, False, "unadjusted", 2),
            ("full", False, True, "unadjusted", 2),
            ("full", True, True, "unadjusted", 2),
            (5, True, False, "unadjusted", 2),
            (5, False, False, "unadjusted", 2),
            (5, False, True, "unadjusted", 2),
            (5, True, True, "unadjusted", 2),
            (21, True, False, "unadjusted", 2),
            (21, False, False, "unadjusted", 2),
            (21, False, True, "unadjusted", 2),
            (21, True, True, "unadjusted", 2),
        ]
    )
    def test_create_model(
        self, roll, fit_intercept, positive, data_freq, min_xs_samples
    ):
        model = LADRegressionSystem(
            roll=roll,
            fit_intercept=fit_intercept,
            positive=positive,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
        )
        self.assertIsInstance(model.create_model(), LADRegressor)
        self.assertEqual(model.create_model().fit_intercept, fit_intercept)
        self.assertEqual(model.create_model().positive, positive)


class TestRidgeRegressionSystem(unittest.TestCase):
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

    def test_types_fit(self):
        """
        Test validity of parameters entered into the fit method.
        """
        model = RidgeRegressionSystem()
        # X
        self.assertRaises(TypeError, model.fit, X=1, y=self.y)
        self.assertRaises(TypeError, model.fit, X="string", y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, model.fit, X=self.X, y=1)
        self.assertRaises(TypeError, model.fit, X=self.X, y="string")
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y_nan)

    @parameterized.expand([True])  # , False])
    def test_valid_fit(self, single_feature):
        """Check the model fits are as expected"""
        cross_sections = self.X.index.get_level_values(0).unique()
        param_names = [
            "roll",
            "alpha",
            "fit_intercept",
            "positive",
            "data_freq",
            "min_xs_samples",
        ]
        param_values = list(
            itertools.product(
                [None, 21, 21 * 6, 21 * 12],  # roll
                [0.0, 0.5, 1.0, 2.0],  # alpha
                [True, False],  # fit_intercept
                [True, False],  # positive
                [None, "unadjusted"],  # data_freq
                [2, 21 * 15],  # min_xs_samples
            )
        )
        for params in param_values:
            param_dict = {name: param for name, param in zip(param_names, params)}
            if param_dict["positive"]:
                param_dict["solver"] = "lbfgs"
            else:
                param_dict["solver"] = "lsqr"
            system_model = RidgeRegressionSystem().set_params(**param_dict)
            if single_feature:
                system_model.fit(pd.DataFrame(self.X.iloc[:, 0]), self.y)
            else:
                system_model.fit(pd.DataFrame(self.X), self.y)
            system_coefs = system_model.coefs_
            system_intercepts = system_model.intercepts_
            for cid in cross_sections:
                model = Ridge().set_params(
                    **{
                        key: value
                        for key, value in param_dict.items()
                        if key in ["fit_intercept", "positive", "alpha"]
                    }
                )
                unique_dates = sorted(self.X.xs(cid).index.unique())
                if params[5] > len(unique_dates):
                    continue
                if params[0] is not None:
                    roll_dates = unique_dates[-params[0] :]
                    if len(roll_dates) >= len(unique_dates):
                        continue
                    if single_feature:
                        X = self.X.iloc[:, 0][
                            self.X.index.get_level_values(1).isin(roll_dates)
                        ]
                        y = self.y[self.y.index.get_level_values(1).isin(roll_dates)]
                    else:
                        X = self.X[self.X.index.get_level_values(1).isin(roll_dates)]
                        y = self.y[self.y.index.get_level_values(1).isin(roll_dates)]
                else:
                    if single_feature:
                        X = self.X.iloc[:, 0]
                        y = self.y
                    else:
                        X = self.X
                        y = self.y

                if single_feature:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                else:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # Check that the model coefficients and intercepts are the same
                np.testing.assert_almost_equal(
                    model.intercept_, system_intercepts[cid], decimal=3
                )
                np.testing.assert_almost_equal(
                    model.coef_, system_coefs[cid], decimal=3
                )

        # Now test that weekly frequency works as expected
        param_values = list(
            itertools.product(
                [None, 5 * 3, 5 * 6, 5 * 12],  # roll
                [0.0, 0.5, 1.0, 2.0],  # alpha
                [True, False],  # fit_intercept
                [True, False],  # positive
                ["W"],  # data_freq
                [2, 5 * 15],  # min_xs_samples
            )
        )

        for params in param_values:
            param_dict = {name: param for name, param in zip(param_names, params)}
            if param_dict["positive"]:
                param_dict["solver"] = "lbfgs"
            else:
                param_dict["solver"] = "lsqr"
            system_model = RidgeRegressionSystem().set_params(**param_dict)
            if single_feature:
                system_model.fit(pd.DataFrame(self.X.iloc[:, 0]), self.y)
            else:
                system_model.fit(pd.DataFrame(self.X), self.y)
            system_coefs = system_model.coefs_
            system_intercepts = system_model.intercepts_
            for cid in cross_sections:
                # First downsample the data to weekly
                X = self.X.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="W"),
                    ]
                ).sum()
                y = self.y.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="W"),
                    ]
                ).sum()
                model = Ridge().set_params(
                    **{
                        key: value
                        for key, value in param_dict.items()
                        if key in ["fit_intercept", "positive", "alpha"]
                    }
                )
                unique_dates = sorted(X.xs(cid).index.unique())
                if params[5] > len(unique_dates):
                    continue
                if params[0] is not None:
                    roll_dates = unique_dates[-params[0] :]
                    if len(roll_dates) >= len(unique_dates):
                        continue
                    if single_feature:
                        X = X.iloc[:, 0][X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                    else:
                        X = X[X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                else:
                    if single_feature:
                        X = X.iloc[:, 0]

                if single_feature:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                else:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # Check that the model coefficients and intercepts are the same
                np.testing.assert_almost_equal(
                    model.intercept_, system_intercepts[cid], decimal=3
                )
                np.testing.assert_almost_equal(
                    model.coef_, system_coefs[cid], decimal=3
                )

        # Now test that monthly frequency works as expected
        param_values = list(
            itertools.product(
                [None, 6, 12],  # roll
                [0.0, 0.5, 1.0, 2.0],  # alpha
                [True, False],  # fit_intercept
                [True, False],  # positive
                ["M"],  # data_freq
                [2, 15],  # min_xs_samples
            )
        )

        for params in param_values:
            param_dict = {name: param for name, param in zip(param_names, params)}
            if param_dict["positive"]:
                param_dict["solver"] = "lbfgs"
            else:
                param_dict["solver"] = "lsqr"
            system_model = RidgeRegressionSystem().set_params(**param_dict)
            if single_feature:
                system_model.fit(pd.DataFrame(self.X.iloc[:, 0]), self.y)
            else:
                system_model.fit(pd.DataFrame(self.X), self.y)
            system_coefs = system_model.coefs_
            system_intercepts = system_model.intercepts_
            for cid in cross_sections:
                # First downsample the data to weekly
                X = self.X.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="M"),
                    ]
                ).sum()
                y = self.y.groupby(
                    [
                        pd.Grouper(level="cid"),
                        pd.Grouper(level="real_date", freq="M"),
                    ]
                ).sum()
                model = Ridge().set_params(
                    **{
                        key: value
                        for key, value in param_dict.items()
                        if key in ["fit_intercept", "positive", "alpha"]
                    }
                )
                unique_dates = sorted(X.xs(cid).index.unique())
                if params[5] > len(unique_dates):
                    continue
                if params[0] is not None:
                    unique_dates = sorted(y.xs(cid).index.unique())
                    roll_dates = unique_dates[-params[0] :]
                    if len(roll_dates) >= len(unique_dates):
                        continue
                    if single_feature:
                        X = X.iloc[:, 0][X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                    else:
                        X = X[X.index.get_level_values(1).isin(roll_dates)]
                        y = y[y.index.get_level_values(1).isin(roll_dates)]
                else:
                    if single_feature:
                        X = X.iloc[:, 0]

                if single_feature:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                else:
                    model.fit(pd.DataFrame(X).xs(cid, level=0), y.xs(cid, level=0))
                # Check that the model coefficients and intercepts are the same
                np.testing.assert_almost_equal(
                    model.intercept_, system_intercepts[cid], decimal=3
                )
                np.testing.assert_almost_equal(
                    model.coef_, system_coefs[cid], decimal=3
                )

    def test_check_init_params(self):
        # Test default params
        model = RidgeRegressionSystem()
        self.assertEqual(model.roll, "full")
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.positive, False)
        self.assertEqual(model.data_freq, None)
        self.assertEqual(model.min_xs_samples, 2)
        self.assertEqual(model.alpha, 1.0)
        np.testing.assert_almost_equal(model.tol, 0.0001)
        self.assertEqual(model.solver, "lsqr")

        # Test custom params
        model = RidgeRegressionSystem(
            roll=5,
            alpha=0.5,
            fit_intercept=False,
            positive=True,
            data_freq="M",
            min_xs_samples=3,
            tol=0.0001,
            solver="auto",
        )
        self.assertEqual(model.roll, 5)
        self.assertEqual(model.fit_intercept, False)
        self.assertEqual(model.positive, True)
        self.assertEqual(model.data_freq, "M")
        self.assertEqual(model.min_xs_samples, 3)
        self.assertEqual(model.alpha, 0.5)
        self.assertEqual(model.tol, 0.0001)
        self.assertEqual(model.solver, "auto")

    def test_invalid_params(self):
        with self.assertRaises(TypeError):
            RidgeRegressionSystem(roll=5.5)
        with self.assertRaises(ValueError):
            RidgeRegressionSystem(roll=-5)
        with self.assertRaises(TypeError):
            RidgeRegressionSystem(roll=None)
        with self.assertRaises(TypeError):
            RidgeRegressionSystem(fit_intercept="False")
        with self.assertRaises(TypeError):
            RidgeRegressionSystem(positive="True")
        with self.assertRaises(TypeError):
            RidgeRegressionSystem(data_freq=5)
        with self.assertRaises(TypeError):
            RidgeRegressionSystem(min_xs_samples="2")
        with self.assertRaises(ValueError):
            RidgeRegressionSystem(roll=-5)
        with self.assertRaises(TypeError):
            RidgeRegressionSystem(alpha="0.5")
        with self.assertRaises(ValueError):
            RidgeRegressionSystem(alpha=-0.5)
        with self.assertRaises(TypeError):
            RidgeRegressionSystem(tol="0.0001")
        with self.assertRaises(ValueError):
            RidgeRegressionSystem(tol=-0.0001)
        with self.assertRaises(ValueError):
            RidgeRegressionSystem(solver="unknown")

    @parameterized.expand(
        [
            (5, True, False, "unadjusted", 2),
            (5, False, True, "unadjusted", 2),
            (5, True, True, "M", 2),
        ]
    )
    def test_valid_init(self, roll, fit_intercept, positive, data_freq, min_xs_samples):

        # Test default params
        try:
            model = RidgeRegressionSystem()
        except Exception as e:
            self.fail(
                "RidgeRegressionSystem constructor raised an exception: {}".format(e)
            )
        self.assertEqual(model.roll, "full")
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.positive, False)
        self.assertEqual(model.data_freq, None)
        self.assertEqual(model.min_xs_samples, 2)
        self.assertEqual(model.alpha, 1.0)
        self.assertEqual(model.tol, 0.0001)
        self.assertEqual(model.solver, "lsqr")
        self.assertIsInstance(model, RidgeRegressionSystem)
        self.assertIsInstance(model, BaseRegressionSystem)
        self.assertIsInstance(model, RegressorMixin)
        self.assertIsInstance(model, BaseEstimator)

        try:
            model = RidgeRegressionSystem(
                roll=roll,
                fit_intercept=fit_intercept,
                positive=positive,
                data_freq=data_freq,
                min_xs_samples=min_xs_samples,
            )
        except Exception as e:
            self.fail(
                "RidgeRegressionSystem constructor raised an exception: {}".format(e)
            )
        self.assertEqual(model.roll, roll)
        self.assertEqual(model.fit_intercept, fit_intercept)
        self.assertEqual(model.positive, positive)
        self.assertEqual(model.data_freq, data_freq)
        self.assertEqual(model.min_xs_samples, min_xs_samples)
        self.assertEqual(model.alpha, 1.0)
        self.assertEqual(model.tol, 0.0001)
        self.assertEqual(model.solver, "lsqr")
        self.assertIsInstance(model, RidgeRegressionSystem)
        self.assertIsInstance(model, BaseRegressionSystem)
        self.assertIsInstance(model, RegressorMixin)
        self.assertIsInstance(model, BaseEstimator)

    @parameterized.expand(
        [
            ("full", 0.5, True, False, "unadjusted", 2, "lsqr"),
            ("full", 0.5, False, False, "unadjusted", 2, "lsqr"),
            ("full", 0.5, False, True, "unadjusted", 2, "lbfgs"),
            ("full", 0.5, True, True, "unadjusted", 2, "lbfgs"),
            (5, 2.0, True, False, "unadjusted", 2, "lsqr"),
            (5, 2.0, False, False, "unadjusted", 2, "lsqr"),
            (5, 2.0, False, True, "unadjusted", 2, "lbfgs"),
            (5, 2.0, True, True, "unadjusted", 2, "lbfgs"),
            (21, 1.0, True, False, "unadjusted", 2, "lsqr"),
            (21, 1.0, False, False, "unadjusted", 2, "lsqr"),
            (21, 1.0, False, True, "unadjusted", 2, "lbfgs"),
            (21, 1.0, True, True, "unadjusted", 2, "lbfgs"),
        ]
    )
    def test_create_model(
        self, roll, alpha, fit_intercept, positive, data_freq, min_xs_samples, solver
    ):
        model = RidgeRegressionSystem(
            roll=roll,
            alpha=alpha,
            fit_intercept=fit_intercept,
            positive=positive,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
            solver=solver,
        )
        self.assertIsInstance(model.create_model(), Ridge)
        self.assertEqual(model.create_model().fit_intercept, fit_intercept)
        self.assertEqual(model.create_model().positive, positive)
        self.assertEqual(model.create_model().alpha, alpha)
        self.assertEqual(model.create_model().solver, solver)

    def test_types_predict(self):
        model = RidgeRegressionSystem()
        model.fit(self.X, self.y)
        # X
        self.assertRaises(TypeError, model.predict, X=1)
        self.assertRaises(TypeError, model.predict, X="string")
        self.assertRaises(ValueError, model.predict, X=self.X.reset_index())
        self.assertRaises(ValueError, model.predict, X=self.X_nan)

    def test_valid_predict(self):
        lad = RidgeRegressionSystem().fit(self.X, self.y)
        y_pred_system = lad.predict(self.X)
        self.assertIsInstance(y_pred_system, pd.Series)
        self.assertEqual(len(y_pred_system), len(self.y))

        # Loop through all cross sections and check that the predictions are correct
        for cid in self.X.index.get_level_values(0).unique():
            X = self.X.xs(cid, level=0)
            y = self.y.xs(cid, level=0)
            lad = Ridge().fit(X, y)
            y_pred_lad = lad.predict(X)
            np.testing.assert_almost_equal(
                y_pred_lad, y_pred_system.xs(cid, level=0).values, decimal=2
            )


class TestCorrelationVolatilitySystem(unittest.TestCase):
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

    def test_check_init_params(self):
        # Test default params
        model = CorrelationVolatilitySystem()
        self.assertEqual(model.correlation_lookback, "full")
        self.assertEqual(model.correlation_type, "pearson")
        self.assertEqual(model.volatility_lookback, "full")
        self.assertEqual(model.volatility_window_type, "rolling")
        self.assertEqual(model.data_freq, None)
        self.assertEqual(model.min_xs_samples, 2)

        # Test custom params
        model = CorrelationVolatilitySystem(
            correlation_lookback=10,
            correlation_type="spearman",
            volatility_lookback=10,
            volatility_window_type="exponential",
            data_freq="M",
            min_xs_samples=10,
        )
        self.assertEqual(model.correlation_lookback, 10)
        self.assertEqual(model.correlation_type, "spearman")
        self.assertEqual(model.volatility_lookback, 10)
        self.assertEqual(model.volatility_window_type, "exponential")
        self.assertEqual(model.data_freq, "M")
        self.assertEqual(model.min_xs_samples, 10)

    def test_invalid_params(self):
        with self.assertRaises(TypeError):
            CorrelationVolatilitySystem(correlation_lookback=5.5)
        with self.assertRaises(ValueError):
            CorrelationVolatilitySystem(correlation_lookback=-5)
        with self.assertRaises(TypeError):
            CorrelationVolatilitySystem(correlation_type=2)
        with self.assertRaises(ValueError):
            CorrelationVolatilitySystem(correlation_type="Invalid")
        with self.assertRaises(ValueError):
            CorrelationVolatilitySystem(volatility_lookback="5")
        with self.assertRaises(ValueError):
            CorrelationVolatilitySystem(volatility_lookback=-1)
        with self.assertRaises(TypeError):
            CorrelationVolatilitySystem(volatility_window_type=5)
        with self.assertRaises(ValueError):
            CorrelationVolatilitySystem(volatility_window_type="Invalid")
        with self.assertRaises(TypeError):
            CorrelationVolatilitySystem(data_freq=-0.5)
        with self.assertRaises(ValueError):
            CorrelationVolatilitySystem(data_freq="INVALID")
        with self.assertRaises(TypeError):
            CorrelationVolatilitySystem(min_xs_samples=3.5)
        with self.assertRaises(ValueError):
            CorrelationVolatilitySystem(min_xs_samples=-8)

    def test_types_fit(self):
        """
        Check the types of the parameters entered into the fit method.
        """
        model = CorrelationVolatilitySystem()
        # X
        self.assertRaises(TypeError, model.fit, X=1, y=self.y)
        self.assertRaises(TypeError, model.fit, X="string", y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan, y=self.y)
        # y
        self.assertRaises(TypeError, model.fit, X=self.X, y=1)
        self.assertRaises(TypeError, model.fit, X=self.X, y="string")
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y_nan)

    def test_valid_fit(self):
        """Check the model fits are as expected"""
        # First check the default parameters yield expected coefficients
        model = CorrelationVolatilitySystem()
        model.fit(self.X, self.y)
        self.assertIsInstance(model.coefs_, dict)
        self.assertEqual(len(model.coefs_), 4)
        self.assertEqual(set(model.coefs_.keys()), set(self.X.index.unique(level=0)))

        cross_sections = self.X.index.unique(level=0)
        for cid in cross_sections:
            X_cid = self.X.xs(cid)
            y_cid = self.y.xs(cid)
            correlation = X_cid.iloc[:, 0].corr(y_cid, method="pearson")
            X_volatility = X_cid.iloc[:, 0].std()
            y_volatility = y_cid.std()
            np.testing.assert_almost_equal(
                model.coefs_[cid],
                correlation * y_volatility / X_volatility,
                decimal=4,
            )
            np.testing.assert_almost_equal(
                model.coefs_[cid],
                LinearRegression()
                .fit(X_cid.iloc[:, 0].values.reshape(-1, 1), y_cid)
                .coef_[0],
                decimal=4,
            )

        # Repeat but with spearman correlation
        model = CorrelationVolatilitySystem(correlation_type="spearman")
        model.fit(self.X, self.y)
        self.assertIsInstance(model.coefs_, dict)
        self.assertEqual(len(model.coefs_), 4)
        self.assertEqual(set(model.coefs_.keys()), set(self.X.index.unique(level=0)))

        cross_sections = self.X.index.unique(level=0)
        for cid in cross_sections:
            X_cid = self.X.xs(cid)
            y_cid = self.y.xs(cid)
            correlation = X_cid.iloc[:, 0].corr(y_cid, method="spearman")
            X_volatility = X_cid.iloc[:, 0].std()
            y_volatility = y_cid.std()
            np.testing.assert_almost_equal(
                model.coefs_[cid],
                correlation * y_volatility / X_volatility,
                decimal=4,
            )

        # Repeat but with kendall correlation
        model = CorrelationVolatilitySystem(correlation_type="kendall")
        model.fit(self.X, self.y)
        self.assertIsInstance(model.coefs_, dict)
        self.assertEqual(len(model.coefs_), 4)
        self.assertEqual(set(model.coefs_.keys()), set(self.X.index.unique(level=0)))

        cross_sections = self.X.index.unique(level=0)
        for cid in cross_sections:
            X_cid = self.X.xs(cid)
            y_cid = self.y.xs(cid)
            correlation = X_cid.iloc[:, 0].corr(y_cid, method="kendall")
            X_volatility = X_cid.iloc[:, 0].std()
            y_volatility = y_cid.std()
            np.testing.assert_almost_equal(
                model.coefs_[cid],
                correlation * y_volatility / X_volatility,
                decimal=4,
            )

    @parameterized.expand(
        itertools.product(
            ["pearson", "spearman", "kendall"], ["rolling", "exponential"]
        )
    )
    def test_valid_fit_custom(self, correlation_type, volatility_window_type):
        """
        Test validity of fit method with custom parameters.
        These tests use non-full lookback windows.
        """
        model = CorrelationVolatilitySystem(
            correlation_type=correlation_type,
            correlation_lookback=21 * 12,
            volatility_lookback=21,
            volatility_window_type=volatility_window_type,
        )
        model.fit(self.X, self.y)
        self.assertIsInstance(model.coefs_, dict)
        self.assertEqual(len(model.coefs_), 4)
        self.assertEqual(set(model.coefs_.keys()), set(self.X.index.unique(level=0)))

        cross_sections = self.X.index.unique(level=0)
        for cid in cross_sections:
            X_cid = self.X.xs(cid)
            y_cid = self.y.xs(cid)
            correlation = (
                X_cid.tail(21 * 12)
                .iloc[:, 0]
                .corr(y_cid.tail(21 * 12), method=correlation_type)
            )
            if volatility_window_type == "rolling":
                X_volatility = X_cid.iloc[:, 0].tail(21).std()
                y_volatility = y_cid.tail(21).std()
            elif volatility_window_type == "exponential":
                X_volatility = X_cid.iloc[:, 0].ewm(span=21, adjust=True).std().iloc[-1]
                y_volatility = y_cid.ewm(span=21, adjust=True).std().iloc[-1]
            np.testing.assert_almost_equal(
                model.coefs_[cid],
                correlation * y_volatility / X_volatility,
                decimal=4,
            )

        # Test weekly frequency
        model = CorrelationVolatilitySystem(
            correlation_type=correlation_type,
            correlation_lookback=4 * 12,  # Assume roughly 4 weeks in a month
            volatility_lookback=4,  # Assume roughly 4 weeks in a month
            volatility_window_type=volatility_window_type,
            data_freq="W",
        )
        model.fit(self.X, self.y)
        self.assertIsInstance(model.coefs_, dict)
        self.assertEqual(len(model.coefs_), 4)
        self.assertEqual(set(model.coefs_.keys()), set(self.X.index.unique(level=0)))

        cross_sections = self.X.index.unique(level=0)
        for cid in cross_sections:
            # First downsample the data to weekly
            X = self.X.groupby(
                [
                    pd.Grouper(level="cid"),
                    pd.Grouper(level="real_date", freq="W"),
                ]
            ).sum()
            y = self.y.groupby(
                [
                    pd.Grouper(level="cid"),
                    pd.Grouper(level="real_date", freq="W"),
                ]
            ).sum()
            X_cid = X.xs(cid)
            y_cid = y.xs(cid)
            correlation = (
                X_cid.tail(4 * 12)
                .iloc[:, 0]
                .corr(y_cid.tail(4 * 12), method=correlation_type)
            )
            if volatility_window_type == "rolling":
                X_volatility = X_cid.iloc[:, 0].tail(4).std()
                y_volatility = y_cid.tail(4).std()
            elif volatility_window_type == "exponential":
                X_volatility = X_cid.iloc[:, 0].ewm(span=4, adjust=True).std().iloc[-1]
                y_volatility = y_cid.ewm(span=4, adjust=True).std().iloc[-1]
            np.testing.assert_almost_equal(
                model.coefs_[cid],
                correlation * y_volatility / X_volatility,
                decimal=4,
            )

        # Test monthly frequency
        model = CorrelationVolatilitySystem(
            correlation_type=correlation_type,
            correlation_lookback=6,  # Use 6-month correlation
            volatility_lookback=3,  # Use 3-month volatility
            volatility_window_type=volatility_window_type,
            data_freq="M",
        )
        model.fit(self.X, self.y)
        self.assertIsInstance(model.coefs_, dict)
        self.assertEqual(len(model.coefs_), 4)
        self.assertEqual(set(model.coefs_.keys()), set(self.X.index.unique(level=0)))

        cross_sections = self.X.index.unique(level=0)
        for cid in cross_sections:
            # First downsample the data to monthly
            X = self.X.groupby(
                [
                    pd.Grouper(level="cid"),
                    pd.Grouper(level="real_date", freq="M"),
                ]
            ).sum()
            y = self.y.groupby(
                [
                    pd.Grouper(level="cid"),
                    pd.Grouper(level="real_date", freq="M"),
                ]
            ).sum()
            X_cid = X.xs(cid)
            y_cid = y.xs(cid)
            correlation = (
                X_cid.tail(6).iloc[:, 0].corr(y_cid.tail(6), method=correlation_type)
            )
            if volatility_window_type == "rolling":
                X_volatility = X_cid.iloc[:, 0].tail(3).std()
                y_volatility = y_cid.tail(3).std()
            elif volatility_window_type == "exponential":
                X_volatility = X_cid.iloc[:, 0].ewm(span=3, adjust=True).std().iloc[-1]
                y_volatility = y_cid.ewm(span=3, adjust=True).std().iloc[-1]
            np.testing.assert_almost_equal(
                model.coefs_[cid],
                correlation * y_volatility / X_volatility,
                decimal=4,
            )

    def test_types_predict(self):
        model = CorrelationVolatilitySystem()
        model.fit(self.X, self.y)
        # X
        self.assertRaises(TypeError, model.predict, X=1)
        self.assertRaises(TypeError, model.predict, X="string")
        self.assertRaises(ValueError, model.predict, X=self.X.reset_index())
        self.assertRaises(ValueError, model.predict, X=self.X_nan)

    def test_valid_predict(self):
        # Check that the predict method returns the expected values (series of zeros)
        model = CorrelationVolatilitySystem()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertIsInstance(predictions, pd.Series)
        self.assertEqual(len(predictions), len(self.y))
        np.testing.assert_almost_equal(predictions.values, 0, decimal=4)
