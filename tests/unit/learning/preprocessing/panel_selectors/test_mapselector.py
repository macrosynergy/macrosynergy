import numpy as np
import pandas as pd
import unittest
import itertools
import scipy.stats as stats
from parameterized import parameterized

from macrosynergy.learning import MapSelector

from sklearn.feature_selection import SelectorMixin
from sklearn.exceptions import NotFittedError

from statsmodels.tools.tools import add_constant
from linearmodels.panel import RandomEffects

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
        positive_selector = MapSelector(significance_level=significance_level, positive=True)
        positive_selector.fit(self.X, self.y)
        negative_selector = MapSelector(significance_level=significance_level, positive=False)
        negative_selector.fit(self.X, self.y)
        self.fitted_selectors = [positive_selector, negative_selector]

    @parameterized.expand([0.01, 0.05, 0.1, 1.0])
    def test_valid_init(self, significance_level):
        positive = bool(np.random.choice([True, False]))
        # Test that the MapSelector class can be instantiated
        selector = MapSelector(significance_level=significance_level, positive=positive)
        self.assertIsInstance(selector, MapSelector)
        self.assertEqual(selector.significance_level, significance_level)
        self.assertEqual(selector.positive, positive)

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
            
    def test_valid_fit(self):
        # Test default initialization
        selector = MapSelector().fit(self.X, self.y)
        self.assertTrue(selector.n_features_in_ == 3)
        np.testing.assert_array_equal(selector.feature_names_in_, self.X.columns)
        self.assertTrue(selector.mask is not None)
        np.testing.assert_array_equal(selector.mask, [True, True, True])
        np.testing.assert_array_equal(selector._get_support_mask(), [True, True, True])
        np.testing.assert_array_equal(selector.get_feature_names_out(), self.X.columns)

        # Test default initialization with fit_intercept=True
        selector = MapSelector(significance_level=0).fit(self.X, self.y)
        self.assertTrue(selector.n_features_in_ == 3)
        np.testing.assert_array_equal(selector.feature_names_in_, self.X.columns)
        self.assertTrue(selector.mask is not None)
        np.testing.assert_array_equal(selector.mask, [False, False, False])
        np.testing.assert_array_equal(selector._get_support_mask(), [False, False, False])
        np.testing.assert_array_equal(selector.get_feature_names_out(), [])

    # @parameterized.expand([True, False])
    # def test_valid_fit(self, positive):
    #     # Test that the fit() method works as expected
    #     significance_level = 0.05
    #     selector = MapSelector(significance_level=significance_level, positive=positive)
    #     # Test that fitting with a pandas target series works
    #     try:
    #         selector.fit(self.X, self.y)
    #     except Exception as e:
    #         self.fail(f"Fit method for the Map selector raised an exception: {e}")
    #     # Test that fitting with a pandas target dataframe works
    #     try:
    #         selector.fit(self.X, self.y.to_frame())
    #     except Exception as e:
    #         self.fail(f"Fit method for the Map selector raised an exception: {e}")
    #     # check that the self.ftrs attribute is a list
    #     self.assertIsInstance(selector.ftrs, list)
    #     # check that the self.ftrs attribute comprises the correct features
    #     self.cols = self.X.columns
    #     if positive:
    #         self.assertTrue(np.all(selector.ftrs == self.restr_ftrs))
    #     else:
    #         self.assertTrue(np.all(selector.ftrs == self.unrestr_ftrs))

    # def test_types_fit(self):
    #     # Test that non dataframe X raises TypeError
    #     with self.assertRaises(TypeError):
    #         selector = MapSelector(significance_level=0.05)
    #         selector.fit("X", self.y)
    #     # Test that non series y raises TypeError
    #     with self.assertRaises(TypeError):
    #         selector = MapSelector(significance_level=0.05)
    #         selector.fit(self.X, "y")
    #     # Test that a value error is raised if the X index isn't a multi-index
    #     with self.assertRaises(ValueError):
    #         selector = MapSelector(significance_level=0.05)
    #         selector.fit(self.X.reset_index(), self.y)
    #     # Test that a value error is raised if the y index isn't a multi-index
    #     with self.assertRaises(ValueError):
    #         selector = MapSelector(significance_level=0.05)
    #         selector.fit(self.y.reset_index(), self.y)
    #     # Test that a dataframe of targets with multiple columns raises ValueError
    #     with self.assertRaises(ValueError):
    #         selector = MapSelector(significance_level=0.05)
    #         selector.fit(self.X, self.X)

    # @parameterized.expand([True, False])
    # def test_valid_get_support(self, indices):
    #     # Test that the get_support() method works as expected
    #     selector = self.fitted_selectors[0]
    #     support = selector.get_support(indices=indices)
    #     # check that the get_support method returns the correct features
    #     mask = [col in self.restr_ftrs for col in self.X.columns]

    #     if indices:
    #         true_support = np.where(mask)[0]
    #     else:
    #         true_support = np.array(mask)

    #     self.assertTrue(np.all(support == true_support))

    # def test_types_get_support(self):
    #     significance_level = 0.05
    #     positive = np.random.choice([True, False])
    #     selector = MapSelector(significance_level=significance_level, positive=positive)
    #     # Raise a NotFittedError if get_support is called without fitting
    #     with self.assertRaises(NotFittedError):
    #         selector.get_support()
    #     # Test that a TypeError is raised if a non-boolean indices parameter is entered
    #     selector = self.fitted_selectors[0]
    #     with self.assertRaises(TypeError):
    #         selector.get_support(indices="indices")

    # @parameterized.expand([0, 1])
    # def test_valid_get_feature_names_out(self, selector_idx):
    #     # Test that the get_feature_names_out() method works as expected
    #     selector = self.fitted_selectors[selector_idx]
    #     feature_names_out = selector.get_feature_names_out()
    #     # check that the get_feature_names_out method returns the correct features
    #     if selector_idx == 0:
    #         self.assertTrue(np.all(feature_names_out == self.restr_ftrs))
    #     else:
    #         self.assertTrue(np.all(feature_names_out == self.unrestr_ftrs))

    # def test_valid_transform(self):
    #     # Test that the transform() method works as expected
    #     selector = self.fitted_selectors[1]
    #     # check that the transformed dataframe equals self.X[self.ftrs]
    #     X_transformed = selector.transform(self.X)
    #     self.assertTrue(np.all(X_transformed.columns == self.X[selector.ftrs].columns))

    # def test_types_transform(self):
    #     # Test that non dataframe X raises TypeError
    #     with self.assertRaises(TypeError):
    #         selector = MapSelector(significance_level=0.05)
    #         selector.fit(self.X, self.y)
    #         selector.transform("X")
    #     # Test that value error is raised if the X index isn't a multi-index
    #     with self.assertRaises(ValueError):
    #         selector.transform(self.X.reset_index())
    #     # Test that value error is raised if the columns of X don't match the columns of self.X
    #     with self.assertRaises(ValueError):
    #         selector.transform(self.X.drop(columns="CPI"))
    
if __name__ == "__main__":
    Test = TestMapSelector()
    Test.setUpClass()
    Test.test_valid_fit()