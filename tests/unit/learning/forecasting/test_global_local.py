import unittest

import numpy as np
import pandas as pd

from macrosynergy.learning import (
    GlobalLocalRegression,
)

class TestGlobalLocalRegression(unittest.TestCase):
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
        # local_lambda
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(local_lambda="1.0")
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(local_lambda=True)
        with self.assertRaises(ValueError):
            model = GlobalLocalRegression(local_lambda=-1.0)
        # global_lambda 
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(global_lambda="1.0")
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(global_lambda=True)
        with self.assertRaises(ValueError):
            model = GlobalLocalRegression(global_lambda=-1.0)
        # positive
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(positive="True")
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(positive=7)
        # fit_intercept
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(fit_intercept="True")
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(fit_intercept=7)
        # min_xs_samples
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(min_xs_samples=1.5)
        with self.assertRaises(TypeError):
            model = GlobalLocalRegression(min_xs_samples=True)
        with self.assertRaises(ValueError):
            model = GlobalLocalRegression(min_xs_samples=-7)

    def test_valid_init(self):
        pass