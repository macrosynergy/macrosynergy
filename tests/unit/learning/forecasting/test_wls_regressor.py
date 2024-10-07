import os
import numpy as np
import pandas as pd

import unittest

from macrosynergy.learning import (
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
)

from sklearn.linear_model import LinearRegression

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

