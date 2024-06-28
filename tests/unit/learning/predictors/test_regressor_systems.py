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

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)

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
        self.y = df["XR"]

    def test_check_init_params(self):
        # Test default params
        model = LinearRegressionSystem()
        self.assertEqual(model.roll, None)
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.positive, False)
        self.assertEqual(model.data_freq, "unadjusted")
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
            LinearRegressionSystem(min_xs_samples=1)
        with self.assertRaises(ValueError):
            LinearRegressionSystem(min_xs_samples=1)
        with self.assertRaises(TypeError):
            LinearRegressionSystem(min_xs_samples=3.7)

    @parameterized.expand(
        [(5, True, False, "unadjusted", 2), (5, False, True, "unadjusted", 2), (5, True, True, "M", 2)]
    )
    def test_valid_init(self, roll, fit_intercept, positive, data_freq, min_xs_samples):

        # Test default params
        try:
            model = LinearRegressionSystem()
        except Exception as e:
            self.fail(
                "LinearRegressionSystem constructor raised an exception: {}".format(e)
            )
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

    def test_create_model(self):
        model = LinearRegressionSystem()
        self.assertIsInstance(model.create_model(), LinearRegression)


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
        self.y = df["XR"]

    def test_check_init_params(self):
        # Test default params
        model = LADRegressionSystem()
        self.assertEqual(model.roll, None)
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.positive, False)
        self.assertEqual(model.data_freq, "unadjusted")
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
            LADRegressionSystem(fit_intercept="False")
        with self.assertRaises(TypeError):
            LADRegressionSystem(positive="True")
        with self.assertRaises(TypeError):
            LADRegressionSystem(data_freq=5)
        with self.assertRaises(TypeError):
            LADRegressionSystem(min_xs_samples="2")
        with self.assertRaises(ValueError):
            LADRegressionSystem(roll=-5)

    @parameterized.expand(
        [(5, True, False, "unadjusted", 2), (5, False, True, "unadjusted", 2), (5, True, True, "M", 2)]
    )
    def test_valid_init(self, roll, fit_intercept, positive, data_freq, min_xs_samples):

        # Test default params
        try:
            model = LADRegressionSystem()
        except Exception as e:
            self.fail(
                "LADRegressionSystem constructor raised an exception: {}".format(e)
            )
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

    def test_create_model(self):
        model = LADRegressionSystem()
        self.assertIsInstance(model.create_model(), LADRegressor)


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
        self.y = df["XR"]

    def test_check_init_params(self):
        # Test default params
        model = RidgeRegressionSystem()
        self.assertEqual(model.roll, None)
        self.assertEqual(model.fit_intercept, True)
        self.assertEqual(model.positive, False)
        self.assertEqual(model.data_freq, "unadjusted")
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
        [(5, True, False, "unadjusted", 2), (5, False, True, "unadjusted", 2), (5, True, True, "M", 2)]
    )
    def test_valid_init(self, roll, fit_intercept, positive, data_freq, min_xs_samples):

        # Test default params
        try:
            RidgeRegressionSystem()
        except Exception as e:
            self.fail(
                "RidgeRegressionSystem constructor raised an exception: {}".format(e)
            )
        try:
            RidgeRegressionSystem(
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

    def test_create_model(self):
        model = RidgeRegressionSystem()
        self.assertIsInstance(model.create_model(), Ridge)


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
        self.y = df["XR"]

    def test_check_init_params(self):
        # Test default params
        model = CorrelationVolatilitySystem()
        self.assertEqual(model.correlation_lookback, None)
        self.assertEqual(model.correlation_type, "pearson")
        self.assertEqual(model.volatility_lookback, 21)
        self.assertEqual(model.volatility_window_type, "rolling")
        self.assertEqual(model.data_freq, "unadjusted")
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
        with self.assertRaises(TypeError):
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
