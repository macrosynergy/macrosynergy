import numpy as np
import pandas as pd
import unittest
import itertools
from parameterized import parameterized

from macrosynergy.learning import LarsSelector, BasePanelSelector

from sklearn.feature_selection import SelectorMixin

class TestLarsSelector(unittest.TestCase):
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
        labels = np.matmul(ftrs, [1, 2, -1]) + np.random.normal(0, 0.5, len(ftrs))
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

    def test_types_init(self):
        # Test n_factors validity
        with self.assertRaises(TypeError):
            LarsSelector(n_factors="a")
        with self.assertRaises(TypeError):
            LarsSelector(n_factors=1.0)
        with self.assertRaises(ValueError):
            LarsSelector(n_factors=-1)
        with self.assertRaises(ValueError):
            LarsSelector(n_factors=0)

        # Test fit_intercept validity
        with self.assertRaises(TypeError):
            LarsSelector(fit_intercept="a")
        with self.assertRaises(TypeError):
            LarsSelector(fit_intercept=1)

    def test_valid_init(self):
        # Check default initialization works
        lars = LarsSelector()
        self.assertEqual(lars.n_factors, 10)
        self.assertEqual(lars.fit_intercept, False)
        self.assertIsInstance(lars, SelectorMixin)
        self.assertIsInstance(lars, BasePanelSelector)
        self.assertTrue(lars.n_features_in_ is None)
        self.assertTrue(lars.feature_names_in_ is None)

        # Check custom initialization works
        lars = LarsSelector(n_factors=5, fit_intercept=True)
        self.assertEqual(lars.n_factors, 5)
        self.assertEqual(lars.fit_intercept, True)
        self.assertIsInstance(lars, SelectorMixin)
        self.assertIsInstance(lars, BasePanelSelector)
        self.assertTrue(lars.n_features_in_ is None)
        self.assertTrue(lars.feature_names_in_ is None)

    def test_types_fit(self):
        scaler = LarsSelector()
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

    def test_valid_fit(self):
        # Test default initialization
        scaler = LarsSelector().fit(self.X, self.y)
        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)
        self.assertTrue(scaler.mask is not None)
        np.testing.assert_array_equal(scaler.mask, [True, True, True])
        np.testing.assert_array_equal(scaler._get_support_mask(), [True, True, True])
        np.testing.assert_array_equal(scaler.get_feature_names_out(), self.X.columns)

        # Test default initialization with fit_intercept=True
        scaler = LarsSelector(fit_intercept=True).fit(self.X, self.y)
        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)
        self.assertTrue(scaler.mask is not None)
        np.testing.assert_array_equal(scaler.mask, [True, True, True])
        np.testing.assert_array_equal(scaler._get_support_mask(), [True, True, True])
        np.testing.assert_array_equal(scaler.get_feature_names_out(), self.X.columns)

        # Try a single feature
        scaler = LarsSelector(n_factors = 1).fit(self.X, self.y)
        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)
        self.assertTrue(scaler.mask is not None)
        np.testing.assert_array_equal(np.sum(scaler.mask), 1)
        np.testing.assert_array_equal(np.sum(scaler._get_support_mask()), 1)
        np.testing.assert_array_equal(len(scaler.get_feature_names_out()), 1)

        # Try a single feature
        scaler = LarsSelector(n_factors = 1, fit_intercept = True).fit(self.X, self.y)
        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)
        self.assertTrue(scaler.mask is not None)
        np.testing.assert_array_equal(np.sum(scaler.mask), 1)
        np.testing.assert_array_equal(np.sum(scaler._get_support_mask()), 1)
        np.testing.assert_array_equal(len(scaler.get_feature_names_out()), 1)


    def test_types_transform(self):
        scaler = LarsSelector().fit(self.X, self.y)
        # Test type of 'X' parameter
        self.assertRaises(TypeError, scaler.transform, X=1)
        self.assertRaises(TypeError, scaler.transform, X="X")
        self.assertRaises(TypeError, scaler.transform, X=self.X.values)
        self.assertRaises(ValueError, scaler.transform, X=self.X_nan)
        self.assertRaises(ValueError, scaler.transform, X=self.X.reset_index())
        self.assertRaises(ValueError, scaler.transform, X=self.X.drop(columns="CPI"))
        self.assertRaises(ValueError, scaler.transform, X=self.X.rename(columns={"CPI": "CPI2"}))

    def test_valid_transform(self):
        # Test default initialization
        scaler = LarsSelector().fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertTrue(X_transformed.shape[1] == 3)
        np.testing.assert_array_equal(X_transformed.columns, self.X.columns)
        np.testing.assert_array_equal(X_transformed.index, self.X.index)

        # Test default initialization with fit_intercept=True
        scaler = LarsSelector(fit_intercept=True).fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertTrue(X_transformed.shape[1] == 3)
        np.testing.assert_array_equal(X_transformed.columns, self.X.columns)
        np.testing.assert_array_equal(X_transformed.index, self.X.index)

        # Try a single feature
        scaler = LarsSelector(n_factors = 1).fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertTrue(X_transformed.shape[1] == 1)
        self.assertTrue(len(X_transformed.columns) == 1)
        np.testing.assert_array_equal(X_transformed.index, self.X.index)

        scaler = LarsSelector(n_factors = 1, fit_intercept = True).fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertTrue(X_transformed.shape[1] == 1)
        self.assertTrue(len(X_transformed.columns) == 1)
        np.testing.assert_array_equal(X_transformed.index, self.X.index)