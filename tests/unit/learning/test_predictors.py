import numpy as np 
import pandas as pd

import unittest 

from macrosynergy.learning import (
    NaivePredictor,
    SignWeightedRegressor,
    TimeWeightedRegressor,
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)

class TestSWRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2013-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2010-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2010-01-01", "2020-12-31"]

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

    def test_valid_init(self):
        # Test that a sign weighted ols model is successfully instantiated
        try:
            model = SignWeightedRegressor(LinearRegression)
        except Exception as e:
            self.fail(
                "SignWeightedRegressor constructor with a LinearRegression object raised an exception: {}".format(e)
            )
        self.assertIsInstance(model, SignWeightedRegressor)
        self.assertIsInstance(model.model, LinearRegression)
        
        # Test that a sign weighted ridge model is successfully instantiated
        try:
            model = SignWeightedRegressor(Ridge, fit_intercept=False, alpha=0.1)
        except Exception as e:
            self.fail(
                "SignWeightedRegressor constructor with a Ridge object raised an exception: {}".format(e)
            )
        self.assertIsInstance(model, SignWeightedRegressor)
        self.assertIsInstance(model.model, Ridge)
        self.assertEqual(model.model.fit_intercept, False)
        self.assertEqual(model.model.alpha, 0.1)
        
    def test_types_init(self):
        # Check that incorrectly specified arguments raise exceptions
        with self.assertRaises(TypeError):
            model = SignWeightedRegressor(LinearRegression())
        with self.assertRaises(ValueError):
            model = SignWeightedRegressor(NaivePredictor)