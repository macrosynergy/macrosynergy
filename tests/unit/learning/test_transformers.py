import numpy as np
import pandas as pd
import unittest
from parameterized import parameterized

from macrosynergy.learning import (
    LassoSelector,
    MapSelector,
)

from statsmodels.tools.tools import add_constant
from statsmodels.regression.mixed_linear_model import MixedLM

class TestLassoSelector(unittest.TestCase):
    def setUp(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2000-01-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all elidgible dates
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
        selector1 = LassoSelector(alpha=1, positive=True)
        selector2 = LassoSelector(alpha=1, positive=False)
        self.assertIsInstance(selector1, LassoSelector)
        self.assertIsInstance(selector2, LassoSelector)

    def test_types_init(self):
        # Test that non float/int alphas raise TypeError
        with self.assertRaises(TypeError):
            selector = LassoSelector(alpha="1", positive=True)
        with self.assertRaises(TypeError):
            selector = LassoSelector(alpha="1", positive=False)  #
        # Test that non bool positives raise TypeError
        with self.assertRaises(TypeError):
            selector = LassoSelector(alpha=1, positive="True")
        with self.assertRaises(TypeError):
            selector = LassoSelector(alpha=1, positive="False")
        # Test that negative alphas raise ValueError
        with self.assertRaises(ValueError):
            selector = LassoSelector(alpha=-1, positive=True)
        with self.assertRaises(ValueError):
            selector = LassoSelector(alpha=-1, positive=False)

    def test_valid_fit(self):
        # Test that the fit() method works as expected
        # positive = False
        selector = LassoSelector(alpha=0.1, positive=False)
        try:
            selector.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"Fit method for the Lasso selector raised an exception: {e}")

        # positive = True
        selector_restrict = LassoSelector(alpha=0.1, positive=True)
        try:
            selector_restrict.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"Fit method for the Lasso selector raised an exception: {e}")

        # Test that the selected_ftr_idxs attribute is a list
        self.assertIsInstance(selector.selected_ftr_idxs, list)
        self.assertIsInstance(selector_restrict.selected_ftr_idxs, list)

        # Test that the selected_ftr_idxs attribute is either empty or a list of integers
        selector_empty_or_ints = not selector.selected_ftr_idxs or all(
            isinstance(item, int) for item in selector.selected_ftr_idxs
        )
        restrict_empty_or_ints = not selector_restrict.selected_ftr_idxs or all(
            isinstance(item, int) for item in selector.selected_ftr_idxs
        )

        self.assertTrue(
            selector_empty_or_ints,
            "selected_ftr_idxs should be either empty or a list of integers",
        )
        self.assertTrue(
            restrict_empty_or_ints,
            "selected_ftr_idxs should be either empty or a list of integers",
        )

    def test_types_fit(self):
        # Test that non np.ndarray X or dataframe raises TypeError
        with self.assertRaises(TypeError):
            selector = LassoSelector(alpha=0.1, positive=False)
            selector.fit("X", self.y)
        # Test that non np.ndarray or series y raises TypeError
        with self.assertRaises(TypeError):
            selector = LassoSelector(alpha=0.1, positive=True)
            selector.fit(self.X, "y")

    @parameterized.expand([True, False])
    def test_valid_transform(self, positive):
        # sample a potential alpha value between zero and one
        alpha = np.random.uniform(low=0, high=1)
        # instantiate a selector
        selector = LassoSelector(alpha=alpha, positive=positive)
        # fit the selector
        selector.fit(self.X, self.y)
        # verify that the transform method only selects features that were selected by the fit method
        X_transformed = selector.transform(self.X)
        self.assertTrue(
            np.all(X_transformed.columns == self.X.columns[selector.selected_ftr_idxs])
        )

    def test_types_transform(self):
        # Test that non np.ndarray X or dataframe raises TypeError
        with self.assertRaises(TypeError):
            selector = LassoSelector(alpha=0.1, positive=False)
            selector.fit(self.X, self.y)
            selector.transform("X")
        # Test that if the input is a dataframe, then so is the output
        selector = LassoSelector(alpha=0.1, positive=False)
        selector.fit(self.X, self.y)
        X_transformed = selector.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        # Check that X_transformed has the same index as X
        self.assertTrue(np.all(X_transformed.index == self.X.index))

class TestMapSelector(unittest.TestCase):
    def setUp(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2016-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2016-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2016-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2016-01-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all elidgible dates
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

        # determine the 95% significant features according to the MAP test
        self.ftrs = []
        for col in self.X.columns:
            ftr = self.X[col]
            ftr = add_constant(ftr)
            groups = ftr.index.get_level_values(1)
            model = MixedLM(self.y,ftr,groups).fit(reml=False)
            est = model.params.iloc[1]
            pval = model.pvalues.iloc[1]
            if (pval < 0.05) & (est > 0):
                self.ftrs.append(col)

    @parameterized.expand([0.01, 0.05, 0.1, 1.0])
    def test_valid_init(self, threshold):
        # Test that the MapSelector class can be instantiated
        selector = MapSelector(threshold=threshold)
        self.assertIsInstance(selector, MapSelector)
        self.assertEqual(selector.threshold, threshold)

    def test_types_init(self):
        # Test that non float/int thresholds raise TypeError
        with self.assertRaises(TypeError):
            selector = MapSelector(threshold="1")
        # Test that negative thresholds raise ValueError
        with self.assertRaises(ValueError):
            selector = MapSelector(threshold=-1)

    def test_valid_fit(self):
        # Test that the fit() method works as expected
        threshold = 0.05
        selector = MapSelector(threshold=threshold)
        try:
            selector.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"Fit method for the Map selector raised an exception: {e}")
        # check that the self.ftrs attribute is a list
        self.assertIsInstance(selector.ftrs, list)
        # check that the self.ftrs attribute comprises the correct features
        self.cols = self.X.columns
        self.assertTrue(np.all(selector.ftrs == self.ftrs))

    def test_types_fit(self):
        # Test that non dataframe X raises TypeError
        with self.assertRaises(TypeError):
            selector = MapSelector(threshold=0.05)
            selector.fit("X", self.y)
        # Test that non series y raises TypeError
        with self.assertRaises(TypeError):
            selector = MapSelector(threshold=0.05)
            selector.fit(self.X, "y")
        # Test that a value error is raised if the X index isn't a multi-index
        with self.assertRaises(ValueError):
            selector = MapSelector(threshold=0.05)
            selector.fit(self.X.reset_index(), self.y)

    def test_valid_transform(self):
        # Test that the transform() method works as expected
        selector = MapSelector(threshold=0.05)