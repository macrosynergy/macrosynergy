import os
import numbers
import numpy as np
import pandas as pd

import unittest
import itertools

from macrosynergy.learning import (
    PLSTransformer
)

from sklearn.cross_decomposition import PLSRegression

from parameterized import parameterized


class TestPLSTransformer(unittest.TestCase):
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
        self.y = np.sign(df["XR"])
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

    def test_types_init(self):
        with self.assertRaises(TypeError):
            pls = PLSTransformer(n_components="2")
        with self.assertRaises(TypeError):
            pls = PLSTransformer(n_components=1.95)
        with self.assertRaises(ValueError):
            pls = PLSTransformer(n_components=0)
        with self.assertRaises(ValueError):
            pls = PLSTransformer(n_components=-4)
         
    def test_valid_init(self):
        model = PLSTransformer(n_components=2)
        self.assertIsInstance(model, PLSTransformer)
        self.assertEqual(model.n_components, 2)