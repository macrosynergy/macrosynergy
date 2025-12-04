import numpy as np
import pandas as pd
import unittest

from macrosynergy.learning import KendallSignificanceSelector, BasePanelSelector

from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator

from scipy.stats import kendalltau

class TestKendallSelector(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        self.cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=self.cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-01-01", "2020-12-31"]

        tuples = []

        for cid in self.cids:
            # get list of all eligible dates
            sdate = df_cids.loc[cid]["earliest"]
            edate = df_cids.loc[cid]["latest"]
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]
            for work_day in work_days:
                tuples.append((cid, work_day))

        n_samples = len(tuples)
        ftrs = np.random.normal(loc=0, scale=1, size=(n_samples, 3))
        labels = np.matmul(ftrs, [1, 0, -1]) + np.random.normal(0, 0.5, len(ftrs))
        df = pd.DataFrame(
            data=np.concatenate((np.reshape(labels, (-1, 1)), ftrs), axis=1),
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
            columns=xcats,
            dtype=np.float32,
        )

        self.X = df.drop(columns="XR")
        self.y = df["XR"]

        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

    def test_init_types(self):
        """
        Test inputs of the init method are checked for correctness.
        """
        # Test alpha is a number
        with self.assertRaises(TypeError):
            model = KendallSignificanceSelector(alpha = "alpha")

        # Test alpha is between 0 and 1
        with self.assertRaises(ValueError):
            model = KendallSignificanceSelector(alpha = -0.5)
        with self.assertRaises(ValueError):
            model = KendallSignificanceSelector(alpha = 1.5)
        with self.assertRaises(ValueError):
            model = KendallSignificanceSelector(alpha = 0)
        with self.assertRaises(ValueError):
            model = KendallSignificanceSelector(alpha = 1)

    def test_init_valid(self):
        """
        Test validity of the init method.
        """
        # Test with default
        model = KendallSignificanceSelector()

        self.assertIsInstance(model, BasePanelSelector)
        self.assertIsInstance(model, SelectorMixin)
        self.assertIsInstance(model, BaseEstimator)

        self.assertEqual(model.alpha, 0.05)


        model = KendallSignificanceSelector(alpha=0.1)

        self.assertIsInstance(model, BasePanelSelector)
        self.assertIsInstance(model, SelectorMixin)
        self.assertIsInstance(model, BaseEstimator)

        self.assertEqual(model.alpha, 0.1)

    def test_fit_types(self):
        """
        Test inputs of the fit method are checked for correctness.
        """
        scaler = KendallSignificanceSelector()
        # Test type of 'X' parameter
        self.assertRaises(TypeError, scaler.fit, X=1, y=self.y)
        self.assertRaises(TypeError, scaler.fit, X="X", y=self.y)
        self.assertRaises(TypeError, scaler.fit, X=self.X.values, y=self.y)
        self.assertRaises(ValueError, scaler.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, scaler.fit, X=self.X.reset_index(), y=self.y)
        # Test type of 'y' parameter
        self.assertRaises(TypeError, scaler.fit, X=self.X, y=1)
        self.assertRaises(TypeError, scaler.fit, X=self.X, y="y")
        self.assertRaises(TypeError, scaler.fit, X=self.X, y=self.y.values)
        self.assertRaises(ValueError, scaler.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, scaler.fit, X=self.X, y=self.y_nan)

    def test_fit_valid(self):
        """
        Test that the fit method works as expected.
        """
        scaler = KendallSignificanceSelector().fit(self.X, self.y)
        
        # Check that the features from kendalltau match those from kendall selector
        pvals = np.array([kendalltau(self.X.iloc[:,i], self.y).pvalue for i in range(self.X.shape[1])])
        scores = np.array([kendalltau(self.X.iloc[:,i], self.y).statistic for i in range(self.X.shape[1])])
        support = pvals < 0.05
        np.testing.assert_array_equal(support, scaler.support_)
        np.testing.assert_array_equal(scores, scaler.scores_)

    def test_transform_types(self):
        """
        Test inputs of the transform method are checked for correctness.
        """
        scaler = KendallSignificanceSelector().fit(self.X, self.y)

        # Test type of 'X' parameter
        self.assertRaises(TypeError, scaler.transform, X=1)
        self.assertRaises(TypeError, scaler.transform, X="X")
        self.assertRaises(TypeError, scaler.transform, X=self.X.values)

        # Test validity of X against what was seen in training
        self.assertRaises(ValueError, scaler.transform, X=self.X.iloc[:,:-1])
        self.assertRaises(ValueError, scaler.transform, X=self.X_nan)
        self.assertRaises(ValueError, scaler.transform, X=self.X.reset_index())

    def test_transform_valid(self):
        """
        Test that the transform method works as expected
        """
        # Test default initialization
        scaler = KendallSignificanceSelector().fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertTrue(X_transformed.shape[1] == 2)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertTrue(all([xcat in self.X.columns for xcat in X_transformed.columns]))