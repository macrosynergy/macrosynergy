import unittest

import numpy as np
import pandas as pd

from macrosynergy.learning import (
    VotingRegressor,
    VotingClassifier,
)

import sklearn.ensemble as skl
from sklearn.linear_model import LinearRegression

class TestVotingRegressor(unittest.TestCase):
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

        # Valid model fitting
        self.my_vr = VotingRegressor(
            estimators = [
                ("rf", skl.RandomForestRegressor(random_state=1, n_estimators=5)),
                ("gb", skl.GradientBoostingRegressor(random_state=2, n_estimators=5)),
            ],
        ).fit(self.X, self.y)
        self.skl_vr = skl.VotingRegressor(
            estimators = [
                ("rf", skl.RandomForestRegressor(random_state=1, n_estimators = 5)),
                ("gb", skl.GradientBoostingRegressor(random_state=2, n_estimators = 5)),
            ],
        ).fit(self.X, self.y)
        self.rf = skl.RandomForestRegressor(random_state=1, n_estimators = 5).fit(self.X, self.y)
        self.gb = skl.GradientBoostingRegressor(random_state=2, n_estimators = 5).fit(self.X, self.y)

    def test_valid_init(self):
        """Just test that a feature importance attribute is added"""
        model = VotingRegressor(estimators=[("rf", skl.RandomForestRegressor())])
        self.assertEqual(model.feature_importances_, None)

    def test_types_fit(self):
        model = VotingRegressor(estimators=[("lr", LinearRegression())])
        # X - when a dataframe
        self.assertRaises(TypeError, model.fit, X=1, y=self.y)
        self.assertRaises(TypeError, model.fit, X="X", y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan.values, y=self.y)
        # X - when a numpy array
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index().values, y=self.y)
        self.assertRaises(
            ValueError, model.fit, X=self.X_nan.reset_index(drop=True).values, y=self.y
        )
        # y - when a series
        self.assertRaises(TypeError, model.fit, X=self.X, y=1)
        self.assertRaises(TypeError, model.fit, X=self.X, y="y")
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y_nan)
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y_nan.values)
        # y - when a dataframe
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(
            ValueError,
            model.fit,
            X=self.X,
            y=pd.DataFrame(self.y_nan.reset_index(drop=True)),
        )
        # y - when a numpy array
        self.assertRaises(
            ValueError, model.fit, X=self.X.values, y=np.zeros((len(self.X), 2))
        )
        self.assertRaises(
            ValueError, model.fit, X=self.X.values, y=np.array([np.nan] * len(self.X))
        )

        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y[:-1])

    def test_valid_fit(self):
        """  Check that the underlying regressors are the same as the ones fitted """
        self.assertTrue(np.allclose(self.my_vr.predict(self.X), self.skl_vr.predict(self.X)))

        """ Check that feature importances are calculated correctly """
        fis = self.my_vr.feature_importances_
        rf_fis = self.rf.feature_importances_
        gb_fis = self.gb.feature_importances_
        self.assertTrue(np.allclose(fis, (rf_fis + gb_fis) / 2))