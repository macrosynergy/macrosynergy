import numpy as np
import pandas as pd
import unittest
from parameterized import parameterized

from macrosynergy.learning import (
    LassoSelector,
    MapSelector,
    FeatureAverager,
    PanelMinMaxScaler,
    PanelStandardScaler,
    ZnScoreAverager,
)

from statsmodels.tools.tools import add_constant
from statsmodels.regression.mixed_linear_model import MixedLM

from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
            selector = MapSelector(threshold=-1.0)

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
        selector.fit(self.X, self.y)
        # check that the transformed dataframe equals self.X[self.ftrs]
        X_transformed = selector.transform(self.X)
        self.assertTrue(np.all(X_transformed.columns == self.X[selector.ftrs].columns))

    def test_types_transform(self):
        # Test that non dataframe X raises TypeError
        with self.assertRaises(TypeError):
            selector = MapSelector(threshold=0.05)
            selector.fit(self.X, self.y)
            selector.transform("X")
        # Test that value error is raised if the X index isn't a multi-index
        with self.assertRaises(ValueError):
            selector.transform(self.X.reset_index())
        # Test that value error is raised if the columns of X don't match the columns of self.X
        with self.assertRaises(ValueError):
            selector.transform(self.X.drop(columns="CPI"))

class TestFeatureAverager(unittest.TestCase):
    def setUp(self):
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

    def test_valid_init(self):
        # Test that the FeatureAverager class can be instantiated
        selector = FeatureAverager()
        self.assertIsInstance(selector, FeatureAverager)
        # Test that the use_signs attribute is correctly set
        selector = FeatureAverager(use_signs=True)
        self.assertTrue(selector.use_signs)
        selector = FeatureAverager(use_signs=False)
        self.assertFalse(selector.use_signs)

    def test_types_init(self):
        # Test that non bool use_signs raises TypeError
        with self.assertRaises(TypeError):
            selector = FeatureAverager(use_signs="True")

    def test_valid_fit(self):
        # Test that the fit() method works as expected
        selector = FeatureAverager()
        try:
            selector.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"Fit method for the FeatureAverager raised an exception: {e}")

    @parameterized.expand([False, True])
    def test_valid_transform(self, use_signs):
        # Test that the transform() method works as expected
        selector = FeatureAverager(use_signs=use_signs)
        selector.fit(self.X, self.y)
        X_transformed = selector.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertTrue(np.all(X_transformed.columns == ["signal"]))
        self.assertIsInstance(X_transformed.index, pd.MultiIndex)
        if use_signs:
            # check that X_transformed is the sign of the mean across columns of X
            self.assertTrue(np.all(X_transformed.values.reshape(-1) == np.sign(self.X.mean(axis=1)).astype(int).values.reshape(-1)))
        else:
            # check that X_transformed is the mean across columns of X
            self.assertTrue(np.all(X_transformed.values.reshape(-1) == self.X.mean(axis=1).values.reshape(-1)))

    def test_types_transform(self):
        selector = FeatureAverager()
        selector.fit(self.X, self.y)
        # Test that non dataframe X raises TypeError
        with self.assertRaises(TypeError):
            selector.transform("X")
        # Test that value error is raised if the X index isn't a multi-index
        with self.assertRaises(ValueError):
            selector.transform(self.X.reset_index())

class TestPanelMinMaxScaler(unittest.TestCase):
    def setUp(self):
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

    def test_valid_fit(self):
        # Test that the fit() method works as expected
        scaler = PanelMinMaxScaler()
        try:
            scaler.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"Fit method for the PanelMinMaxScaler raised an exception: {e}")

        self.assertTrue(np.all(scaler.mins == self.X.min(axis=0)))
        self.assertTrue(np.all(scaler.maxs == self.X.max(axis=0)))

    def test_types_fit(self):
        # Test that non dataframe and non series X raises TypeError
        with self.assertRaises(TypeError):
            scaler = PanelMinMaxScaler()
            scaler.fit("X", self.y)

        # Test that value error is raised if the X index isn't a multi-index
        with self.assertRaises(ValueError):
            scaler = PanelMinMaxScaler()
            scaler.fit(self.X.reset_index(), self.y)

    def test_valid_transform(self):
        # Test that the transform() method works as expected
        scaler = PanelMinMaxScaler()
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        # check that X_transformed has the same columns as X
        self.assertTrue(np.all(X_transformed.columns == self.X.columns))
        # check that X_transformed has values between 0 and 1
        self.assertTrue(np.all(X_transformed.values >= 0))
        self.assertTrue(np.all(X_transformed.values <= 1))

    def test_types_transform(self):
        scaler = PanelMinMaxScaler()
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        # check that X_transformed is a dataframe
        self.assertIsInstance(X_transformed, pd.DataFrame)
        # check that X_transformed has the same index as X
        self.assertTrue(np.all(X_transformed.index == self.X.index))

class TestPanelStandardScaler(unittest.TestCase):
    def setUp(self):
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

    @parameterized.expand([[False, False], [False, True], [True, False], [True, True]])
    def test_valid_init(self, with_mean, with_std):
        # Test that the PanelStandardScaler class can be instantiated
        scaler = PanelStandardScaler(with_mean=with_mean, with_std=with_std)
        self.assertIsInstance(scaler, PanelStandardScaler)
        self.assertEqual(scaler.with_mean, with_mean)
        self.assertEqual(scaler.with_std, with_std)
        self.assertEqual(scaler.means, None)
        self.assertEqual(scaler.stds, None)

    @parameterized.expand([[False, "True"], ["False", True], ["True", "True"]])
    def test_types_init(self, with_mean, with_std):
        # Test that each parameter combination results in a TypeError
        with self.assertRaises(TypeError):
            scaler = PanelStandardScaler(with_mean=with_mean, with_std=with_std)

    @parameterized.expand([[False, False], [False, True], [True, False], [True, True]])
    def test_valid_fit(self, with_mean, with_std):
        # Test that the fit() method works as expected
        scaler = PanelStandardScaler(with_mean=with_mean, with_std=with_std)
        try:
            scaler.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"Fit method for the PanelMinMaxScaler raised an exception: {e}")

        if with_mean:
            self.assertTrue(np.all(scaler.means == self.X.mean(axis=0)))
        if with_std:
            self.assertTrue(np.all(scaler.stds == self.X.std(axis=0)))

    @parameterized.expand([[False, False], [False, True], [True, False], [True, True]])
    def test_types_fit(self, with_mean, with_std):
        # Test that non dataframe and non series X raises TypeError
        with self.assertRaises(TypeError):
            scaler = PanelStandardScaler(with_mean=with_mean, with_std=with_std)
            scaler.fit("X", self.y)

        # Test that value error is raised if the X index isn't a multi-index
        with self.assertRaises(ValueError):
            scaler = PanelStandardScaler(with_mean=with_mean, with_std=with_std)
            scaler.fit(self.X.reset_index(), self.y)

    @parameterized.expand([[False, False], [False, True], [True, False], [True, True]])
    def test_valid_transform(self, with_mean, with_std):
        # Test that the transform() method works as expected
        scaler = PanelStandardScaler(with_mean=with_mean, with_std=with_std)
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        # check that X_transformed has the same columns as X
        self.assertTrue(np.all(X_transformed.columns == self.X.columns))

    @parameterized.expand([[False, False], [False, True], [True, False], [True, True]])
    def test_types_transform(self, with_mean, with_std):
        scaler = PanelStandardScaler(with_mean=with_mean, with_std=with_std)
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        # check that X_transformed is a dataframe
        self.assertIsInstance(X_transformed, pd.DataFrame)
        # check that X_transformed has the same index as X
        self.assertTrue(np.all(X_transformed.index == self.X.index))


class TestZnScoreAverager(unittest.TestCase):
    def setUp(self):
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

    def test_constructor(self):
        # Testing the constructor for correct attribute initialization
        averager = ZnScoreAverager(neutral="mean", use_signs=True)
        self.assertEqual(averager.neutral, "mean")
        self.assertTrue(averager.use_signs)

        # Testing the constructor for handling invalid inputs
        with self.assertRaises(TypeError):
            ZnScoreAverager(neutral=123)
        with self.assertRaises(ValueError):
            ZnScoreAverager(neutral="invalid_value")
        with self.assertRaises(TypeError):
            ZnScoreAverager(use_signs="not_a_boolean")

    def test_fit_neutral_zero(self):
        averager = ZnScoreAverager(neutral="zero")
        averager.fit(self.X)
        self.assertIsNotNone(averager.training_mads)
        self.assertEqual(averager.training_n, len(self.X))

        with self.assertRaises(TypeError):
            averager.fit("not_a_dataframe")
            
        with self.assertRaises(ValueError):
            averager.fit(pd.DataFrame())

    def test_fit_neutral_mean(self):
        averager = ZnScoreAverager(neutral="mean")
        averager.fit(self.X)
        self.assertIsNotNone(averager.training_means)
        self.assertIsNotNone(averager.training_sum_squares)
        self.assertEqual(averager.training_n, len(self.X))

        with self.assertRaises(TypeError):
            averager.fit("not_a_dataframe")
            
        with self.assertRaises(ValueError):
            averager.fit(pd.DataFrame())

    def test_transform_types_neutral_zero(self):
        averager = ZnScoreAverager(neutral="zero")
        averager.fit(self.X)
        transformed = averager.transform(self.X)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(transformed.shape, (self.X.shape[0], 1))

        with self.assertRaises(TypeError):
            averager.transform("not_a_dataframe")

    def test_transform_types_neutral_mean(self):
        averager = ZnScoreAverager(neutral="mean")
        averager.fit(self.X)
        transformed = averager.transform(self.X)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(transformed.shape, (self.X.shape[0], 1))

        with self.assertRaises(TypeError):
            averager.transform("not_a_dataframe")

    @parameterized.expand([["zero"], ["mean"]])
    def test_transform_values_use_signs(self, neutral):
        averager = ZnScoreAverager(neutral=neutral, use_signs=True)
        averager.fit(self.X)
        transformed = averager.transform(self.X).abs()
        transformed_abs = transformed.abs()
        self.assertTrue(np.all(transformed_abs == 1))

    def test_get_expanding_count(self):
        averager = ZnScoreAverager(neutral="zero")
        self.assertTrue(np.all(averager._get_expanding_count(self.X)[-1] == np.array([len(self.X)]*self.X.columns.size)))


if __name__ == "__main__":
    unittest.main()
