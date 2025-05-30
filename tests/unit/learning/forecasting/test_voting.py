import unittest

import numpy as np
import pandas as pd

from macrosynergy.learning import (
    VotingRegressor,
    VotingClassifier,
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
)

import sklearn.ensemble as skl
from sklearn.linear_model import LinearRegression, LogisticRegression

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

        # Weighted voting regressor

        self.vr_weights = VotingRegressor(
            estimators = [
                ("ols", LinearRegression()),
                ("swls", SignWeightedLinearRegression()),
                ("twls", TimeWeightedLinearRegression()),
            ],
            weights = [0.25, 0.25, 0.5],
        ).fit(self.X, self.y)

    def test_valid_init(self):
        """Just test that a feature importance attribute is added"""
        model = VotingRegressor(estimators=[("rf", skl.RandomForestRegressor())])
        self.assertEqual(model.feature_importances_, None)

        """Test that the other attributes are set correctly"""
        self.assertEqual(model.estimators_, None)
        self.assertEqual(model.named_estimators_, None)

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

        """ Check that feature importances are right when custom weights are used """
        vr_weights_fis = self.vr_weights.feature_importances_
        ols_coefs = LinearRegression().fit(self.X, self.y).coef_
        swls_coefs = SignWeightedLinearRegression().fit(self.X, self.y).coef_
        twls_coefs = TimeWeightedLinearRegression().fit(self.X, self.y).coef_
        ols_fis = np.abs(ols_coefs) / np.sum(np.abs(ols_coefs))
        swls_fis = np.abs(swls_coefs) / np.sum(np.abs(swls_coefs))
        twls_fis = np.abs(twls_coefs) / np.sum(np.abs(twls_coefs))

        self.assertTrue(np.allclose(vr_weights_fis, (0.25 * ols_fis + 0.25 * swls_fis + 0.5 * twls_fis)))

    def test_valid_predict(self):
        """
        Check that the predictions of the voting regressor are the same as the average of the rf and gb
        as determined by the class instance
        """
        both_preds = self.my_vr._predict(self.X)
        vr_preds = self.my_vr.predict(self.X)
        self.assertTrue(np.allclose(np.mean(both_preds, axis=1), vr_preds))

        """ Check that the predictions of the voting regressor are the same as the average of the rf and gb"""
        rf_preds = self.rf.predict(self.X)
        gb_preds = self.gb.predict(self.X)
        self.assertTrue(np.allclose(vr_preds, (rf_preds + gb_preds) / 2))

        """
        Check that the predictions of the voting regressor with custom weights are the same as the weighted average of the rf and gb
        as determined by the class instance
        """
        both_preds = self.vr_weights._predict(self.X)
        vr_preds = self.vr_weights.predict(self.X)
        self.assertTrue(np.allclose(np.average(both_preds, weights = [0.25, 0.25, 0.5], axis=1), vr_preds))

        """
        Check that the predictions of the voting regressor with custom weights are the same as the weighted average of the rf and gb
        """
        ols_preds = LinearRegression().fit(self.X, self.y).predict(self.X)
        swls_preds = SignWeightedLinearRegression().fit(self.X, self.y).predict(self.X)
        twls_preds = TimeWeightedLinearRegression().fit(self.X, self.y).predict(self.X)
        self.assertTrue(np.allclose(vr_preds, (0.25 * ols_preds + 0.25 * swls_preds + 0.5 * twls_preds)))
        """

class TestVotingClassifier(unittest.TestCase):
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
        self.y = np.sign(df["XR"])
        self.y_numpy = self.y.values
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

        # Valid model fitting
        self.my_vc = VotingClassifier(
            estimators = [
                ("rf", skl.RandomForestClassifier(random_state=1, n_estimators=5)),
                ("gb", skl.GradientBoostingClassifier(random_state=2, n_estimators=5)),
            ],
        ).fit(self.X, self.y)
        self.skl_vc = skl.VotingClassifier(
            estimators = [
                ("rf", skl.RandomForestClassifier(random_state=1, n_estimators = 5)),
                ("gb", skl.GradientBoostingClassifier(random_state=2, n_estimators = 5)),
            ],
        ).fit(self.X, self.y)
        self.rf = skl.RandomForestClassifier(random_state=1, n_estimators = 5).fit(self.X, self.y)
        self.gb = skl.GradientBoostingClassifier(random_state=2, n_estimators = 5).fit(self.X, self.y)

    def test_valid_init(self):
        model = VotingClassifier(estimators=[("rf", skl.RandomForestClassifier())])
        self.assertEqual(model.feature_importances_, None)

    def test_types_fit(self):
        model = VotingClassifier(estimators=[("lr", LogisticRegression())])
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
        self.assertTrue(np.allclose(self.my_vc.predict(self.X), self.skl_vc.predict(self.X)))

        fis = self.my_vc.feature_importances_
        rf_fis = self.rf.feature_importances_
        gb_fis = self.gb.feature_importances_
        self.assertTrue(np.allclose(fis, (rf_fis + gb_fis) / 2))
"""