import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVR

import itertools

import unittest
from unittest.mock import patch

from macrosynergy.learning.predictors.bootstrap import BaseModifiedRegressor
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

    @parameterized.expand(itertools.product([LinearRegression(), LinearSVR()], ["analytic", "bootstrap"], [1e-5, 1e-3, 1e-1], ["panel", "period", "cross", "period_per_cross", "cross_per_period"], [0.25,0.5,1.0], ["White"]))
    def test_valid_init(self, model, method, error_offset, bootstrap_method, resample_ratio, analytic_method):
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
        
        # method
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model = LinearRegression(), method=1)
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model = LinearRegression(), method="invalid_method")

        # error_offset
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model = LinearRegression(), method="analytic", error_offset="invalid_offset")
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model = LinearRegression(), method="analytic", error_offset=-1)

        # bootstrap_method
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model = LinearRegression(), method="bootstrap", bootstrap_method=1)
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model = LinearRegression(), method="bootstrap", bootstrap_method="invalid_method")

        # bootstrap_iters
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model = LinearRegression(), method="bootstrap", bootstrap_iters="invalid_iters")
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model = LinearRegression(), method="bootstrap", bootstrap_iters=-1)

        # resample_ratio
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model = LinearRegression(), method="bootstrap", resample_ratio="invalid_ratio")
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model = LinearRegression(), method="bootstrap", resample_ratio=-1)
        with self.assertRaises(ValueError):
            BaseModifiedRegressor(model = LinearRegression(), method="bootstrap", resample_ratio=1.1)

        # max_features
        with self.assertRaises(NotImplementedError):
            BaseModifiedRegressor(model = LinearRegression(), method="bootstrap", max_features=1)

        # analytic_method
        with self.assertRaises(TypeError):
            BaseModifiedRegressor(model = LinearRegression(), method="analytic", analytic_method=1)

    def test_types_fit(self):
        bmr = BaseModifiedRegressor(
            model=LinearRegression(),
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