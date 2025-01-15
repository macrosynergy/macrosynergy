import itertools
import unittest

import numpy as np
import pandas as pd
import scipy.stats as stats
from linearmodels.panel import RandomEffects
from linearmodels.panel import RandomEffects as lm_RandomEffects
from parameterized import parameterized
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectorMixin
from statsmodels.tools.tools import add_constant

from macrosynergy.learning import MapSelector


class TestMapSelector(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-01-01", "2020-12-31"]

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

        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

        unique_xss = sorted(self.X.index.get_level_values(0).unique())
        xs_codes = dict(zip(unique_xss, range(1, len(unique_xss) + 1)))

        X_map_test = self.X.copy().rename(xs_codes, level=0, inplace=False).copy()
        y_map_test = self.y.copy().rename(xs_codes, level=0, inplace=False).copy()

        # determine the 95% significant features according to the MAP test
        self.est = []
        self.pval = []
        for col in X_map_test.columns:
            ftr = X_map_test[col]
            ftr = add_constant(ftr)
            re = RandomEffects(y_map_test.swaplevel(), ftr.swaplevel()).fit()
            est = re.params[col]
            zstat = est / re.std_errors[col]
            pval = 2 * (1 - stats.norm.cdf(zstat))
            self.est.append(est)
            self.pval.append(pval)

        # get chosen features with and without the positive restriction
        self.restr_ftrs = [
            self.X.columns[idx]
            for idx, est in enumerate(self.est)
            if self.pval[idx] < 0.05 and est > 0
        ]
        self.unrestr_ftrs = [
            self.X.columns[idx]
            for idx, est in enumerate(self.est)
            if self.pval[idx] < 0.05
        ]

        significance_level = 0.05
        positive_selector = MapSelector(
            significance_level=significance_level, positive=True
        )
        positive_selector.fit(self.X, self.y)
        negative_selector = MapSelector(
            significance_level=significance_level, positive=False
        )
        negative_selector.fit(self.X, self.y)
        self.fitted_selectors = [positive_selector, negative_selector]

    @parameterized.expand(itertools.product([0.01, 0.05, 0.1, 1.0], [None, 1, 2, 3]))
    def test_valid_init(self, significance_level, n_factors):
        positive = bool(np.random.choice([True, False]))
        # Test that the MapSelector class can be instantiated
        selector = MapSelector(
            significance_level=significance_level,
            positive=positive,
            n_factors=n_factors,
        )
        self.assertIsInstance(selector, MapSelector)
        self.assertEqual(selector.significance_level, significance_level)
        self.assertEqual(selector.positive, positive)
        self.assertEqual(selector.n_factors, n_factors)

    def test_types_init(self):
        # Test that non float/int significance_levels raise TypeError
        with self.assertRaises(TypeError):
            selector = MapSelector(significance_level="1")
        # Test that negative significance_levels raise ValueError
        with self.assertRaises(ValueError):
            selector = MapSelector(significance_level=-1.0)
        # Test that an incorrect 'positive' type results in a TypeError
        with self.assertRaises(TypeError):
            selector = MapSelector(significance_level=0.05, positive="True")
        # Test that non int n_factors raises TypeError
        self.assertRaises(
            TypeError, MapSelector, significance_level=0.05, n_factors="1"
        )
        # Test that negative n_factors raises ValueError
        self.assertRaises(
            ValueError, MapSelector, significance_level=0.05, n_factors=-1
        )
        # Test that zero n_factors raises ValueError
        self.assertRaises(ValueError, MapSelector, significance_level=0.05, n_factors=0)

    def test_valid_fit(self):
        # Test default initialization
        selector = MapSelector().fit(self.X, self.y)
        self.assertTrue(selector.n_features_in_ == 3)
        np.testing.assert_array_equal(selector.feature_names_in_, self.X.columns)
        self.assertTrue(selector.mask is not None)
        np.testing.assert_array_equal(selector.mask, [True, True, True])
        np.testing.assert_array_equal(selector._get_support_mask(), [True, True, True])
        np.testing.assert_array_equal(selector.get_feature_names_out(), self.X.columns)

        # Test 0 significance level
        selector = MapSelector(significance_level=0).fit(self.X, self.y)
        self.assertTrue(selector.n_features_in_ == 3)
        np.testing.assert_array_equal(selector.feature_names_in_, self.X.columns)
        self.assertTrue(selector.mask is not None)
        np.testing.assert_array_equal(selector.mask, [False, False, False])
        np.testing.assert_array_equal(
            selector._get_support_mask(), [False, False, False]
        )
        np.testing.assert_array_equal(selector.get_feature_names_out(), [])

        # Test n_factors yields the correct number of features
        # Test it should
        for n_factors in [1, 2, 3]:
            selector = MapSelector(n_factors=n_factors).fit(self.X, self.y)
            self.assertTrue(selector.n_features_in_ == 3)
            np.testing.assert_array_equal(selector.feature_names_in_, self.X.columns)
            self.assertTrue(selector.mask is not None)
            self.assertEqual(np.sum(selector.mask), n_factors)
            self.assertEqual(len(selector.get_feature_names_out()), n_factors)

    def test_types_fit(self):
        scaler = MapSelector()
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

    def test_types_transform(self):
        scaler = MapSelector().fit(self.X, self.y)
        # Test type of 'X' parameter
        self.assertRaises(TypeError, scaler.transform, X=1)
        self.assertRaises(TypeError, scaler.transform, X="X")
        self.assertRaises(TypeError, scaler.transform, X=self.X.values)
        self.assertRaises(ValueError, scaler.transform, X=self.X_nan)
        self.assertRaises(ValueError, scaler.transform, X=self.X.reset_index())
        self.assertRaises(ValueError, scaler.transform, X=self.X.drop(columns="CPI"))
        self.assertRaises(
            ValueError, scaler.transform, X=self.X.rename(columns={"CPI": "CPI2"})
        )

    def test_valid_transform(self):
        # Test 0 significance level
        selector = MapSelector(significance_level=0).fit(self.X, self.y)
        X_transformed = selector.transform(self.X)
        self.assertTrue(X_transformed.shape[1] == 0)
        self.assertTrue(len(X_transformed.values) == self.X.shape[0])
        np.testing.assert_array_equal(X_transformed.index, self.X.index)

        # Test default initialization
        selector = MapSelector().fit(self.X, self.y)
        X_transformed = selector.transform(self.X)
        self.assertTrue(X_transformed.shape[1] == 3)
        np.testing.assert_array_equal(X_transformed.columns, self.X.columns)
        np.testing.assert_array_equal(X_transformed.index, self.X.index)

        # Test n_factors yields the correct number of features
        for n_factors in [1, 2, 3]:
            selector = MapSelector(n_factors=n_factors).fit(self.X, self.y)
            X_transformed = selector.transform(self.X)
            self.assertTrue(X_transformed.shape[1] == n_factors)


if __name__ == "__main__":
    Test = TestMapSelector()
    Test.setUpClass()
    Test.test_valid_fit()
