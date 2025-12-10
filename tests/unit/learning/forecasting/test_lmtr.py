import numpy as np
import pandas as pd

from macrosynergy.learning import LinearMultiTargetRegression, LarsSelector

import unittest

class TestLMTR(unittest.TestCase):
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

    def test_init_types(self):
        """
        Test inputs of the init method are checked for correctness.
        """
        # fit_intercept must be boolean
        self.assertRaises(TypeError, LinearMultiTargetRegression, fit_intercept="True")
        # seemingly unrelated must be boolean
        self.assertRaises(TypeError, LinearMultiTargetRegression, seemingly_unrelated="False")
        # ewm_covariance must be boolean
        self.assertRaises(TypeError, LinearMultiTargetRegression, ewm_covariance="True")
        # span should be a positive integer when ewm_covariance is True
        self.assertRaises(TypeError, LinearMultiTargetRegression, span="5")
        self.assertRaises(ValueError, LinearMultiTargetRegression, span=0)
        self.assertRaises(ValueError, LinearMultiTargetRegression, span=-3)
        # feature_selection must inherit from SelectorMixin
        self.assertRaises(
            TypeError,
            LinearMultiTargetRegression,
            feature_selection="not_a_selector",
        )

    def test_init_valid(self):
        """"
        Test validity of the init method
        """
        # Test defaults set correctly
        model = LinearMultiTargetRegression()
        self.assertIsInstance(model, LinearMultiTargetRegression)
        self.assertTrue(model.fit_intercept)
        self.assertFalse(model.seemingly_unrelated)
        self.assertTrue(model.ewm_covariance)
        self.assertEqual(model.span, 60)
        self.assertIsNone(model.feature_selection)

        # Test with custom valid parameters
        model = LinearMultiTargetRegression(
            fit_intercept=False,
            seemingly_unrelated=True,
            ewm_covariance=False,
            span=32,
            feature_selection=LarsSelector(n_factors = 1),
        )
        self.assertIsInstance(model, LinearMultiTargetRegression)
        self.assertFalse(model.fit_intercept)
        self.assertTrue(model.seemingly_unrelated)
        self.assertFalse(model.ewm_covariance)
        self.assertEqual(model.span, 32)
        self.assertIsInstance(model.feature_selection, LarsSelector)

    def test_fit_types(self):
        pass

    def test_fit_valid(self):
        pass

    def test_predict_types(self):
        pass

    def test_predict_valid(self):
        pass