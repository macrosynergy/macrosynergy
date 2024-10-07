import os
import numpy as np
import pandas as pd

import unittest

from macrosynergy.learning import (
    LADRegressor,
)

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
        self.y = df["XR"]

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
        pass 

    def test_valid_fit(self):
        pass

    def test_types_predict(self):
        pass 

    def test_valid_predict(self):
        pass