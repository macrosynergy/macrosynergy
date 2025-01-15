import os
import numpy as np
import pandas as pd

import unittest

from macrosynergy.learning import (
    NaiveRegressor,
)

from parameterized import parameterized

class TestNaiveRegressor(unittest.TestCase):
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

    def test_types_fit(self):
        # X
        with self.assertRaises(TypeError):
            nr = NaiveRegressor().fit(1, self.y)
        with self.assertRaises(TypeError):
            nr = NaiveRegressor().fit("string", self.y)
        with self.assertRaises(ValueError):
            nr = NaiveRegressor().fit(np.expand_dims(self.X.values,0), self.y)
        with self.assertRaises(ValueError):
            nr = NaiveRegressor().fit(np.zeros_like(None), self.y)


    def test_valid_fit(self):
        # Check that a fit on simulated data was successful
        try:
            nr = NaiveRegressor().fit(self.X, self.y)
        except Exception as e:
            self.fail(f"NaiveRegressor fit failed on simulated data. Error: {e}")
        # Check a fit is successful when only a single column is passed 
        try:
            nr = NaiveRegressor().fit(self.X["CPI"], self.y)
        except Exception as e:
            self.fail(f"NaiveRegressor fit failed on simulated data for column CPI. Error: {e}")

    def test_valid_predict(self):
        # pandas Series
        nr = NaiveRegressor().fit(self.X["CPI"], self.y)
        try:
            preds = nr.predict(self.X["CPI"])
        except Exception as e:
            self.fail(f"NaiveRegressor predict failed on a single series. Error: {e}.")
        np.testing.assert_array_equal(preds, self.X["CPI"].values)
        # single column numpy array
        nr = NaiveRegressor().fit(self.X["CPI"].values, self.y)
        try:
            preds = nr.predict(self.X["CPI"].values)
        except Exception as e:
            self.fail(f"NaiveRegressor predict failed on a single column, as a numpy array. Error: {e}.")
        np.testing.assert_array_equal(preds, self.X["CPI"].values)
        # single column dataframe
        nr = NaiveRegressor().fit(pd.DataFrame(self.X["CPI"]), self.y)
        try:
            preds = nr.predict(pd.DataFrame(self.X["CPI"]))
        except Exception as e:
            self.fail(f"NaiveRegressor predict failed on a single column. Error: {e}.")
        np.testing.assert_array_equal(preds, self.X["CPI"].values)
        # whole dataframe
        nr = NaiveRegressor().fit(self.X, self.y)
        try:
            preds = nr.predict(self.X)
        except Exception as e:
            self.fail(f"NaiveRegressor predict failed on a whole dataframe. Error: {e}.")
        np.testing.assert_array_equal(preds, self.X.mean(axis=1).values)
        # whole dataframe as numpy array
        nr = NaiveRegressor().fit(self.X.to_numpy(), self.y)
        try:
            preds = nr.predict(self.X.to_numpy())
        except Exception as e:
            self.fail(f"NaiveRegressor predict failed on a whole dataframe, represented as a numpy array. Error: {e}.")
        np.testing.assert_array_equal(preds, np.mean(self.X.values, axis = 1))

    def test_types_predict(self):
        # pandas series
        nr = NaiveRegressor().fit(self.X["CPI"], self.y)
        with self.assertRaises(TypeError):
            preds = nr.predict(1)
        with self.assertRaises(TypeError):
            preds = nr.predict("string")
        with self.assertRaises(ValueError):
            preds = nr.predict(self.X["CPI"].values) # Prohibit data of a different type to that seen in 'fit'
        with self.assertRaises(ValueError):
            preds = nr.predict(pd.DataFrame(self.X["CPI"]))
        # numpy array - single column
        nr = NaiveRegressor().fit(self.X["CPI"].values, self.y)
        with self.assertRaises(TypeError):
            preds = nr.predict(1)
        with self.assertRaises(TypeError):
            preds = nr.predict("string")
        with self.assertRaises(ValueError):
            preds = nr.predict(self.X["CPI"])
        with self.assertRaises(ValueError):
            preds = nr.predict(pd.DataFrame(self.X["CPI"]))
        with self.assertRaises(ValueError):
            preds = nr.predict(self.X.values) 
        # pandas dataframe - single column
        nr = NaiveRegressor().fit(pd.DataFrame(self.X["CPI"]), self.y)
        with self.assertRaises(TypeError):
            preds = nr.predict(1)
        with self.assertRaises(TypeError):
            preds = nr.predict("string")
        with self.assertRaises(ValueError):
            preds = nr.predict(self.X["CPI"].values)
        with self.assertRaises(ValueError):
            preds = nr.predict(self.X["CPI"])
        with self.assertRaises(ValueError):
            preds = nr.predict(self.X)
        # pandas dataframe - multiple columns
        nr = NaiveRegressor().fit(self.X, self.y)
        with self.assertRaises(TypeError):
            preds = nr.predict(1)
        with self.assertRaises(TypeError):
            preds = nr.predict("string")
        with self.assertRaises(ValueError):
            preds = nr.predict(self.X.values)
        with self.assertRaises(ValueError):
            preds = nr.predict(pd.DataFrame(self.X["CPI"]))
        # numpy array - multiple columns
        nr = NaiveRegressor().fit(self.X.values, self.y)
        with self.assertRaises(TypeError):
            preds = nr.predict(1)
        with self.assertRaises(TypeError):
            preds = nr.predict("string")
        with self.assertRaises(ValueError):
            preds = nr.predict(self.X)
        with self.assertRaises(ValueError):
            preds = nr.predict(self.X["CPI"].values)