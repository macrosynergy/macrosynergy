import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from macrosynergy.learning import LinearMultiTargetRegression, LarsSelector

import unittest
from parameterized import parameterized

class TestLMTR(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "XR2", "GROWTH", "RIR"]

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
        ftrs = np.random.normal(loc=0, scale=1, size=(n_samples, 2))
        labels1 = np.matmul(ftrs, [1, 2]) + np.random.normal(0, 0.5, len(ftrs))
        labels2 = 0.1 - 0.75 * labels1 + np.random.normal(0, 0.5, len(ftrs))
        df = pd.DataFrame(
            data=np.concatenate((np.reshape(labels1, (-1, 1)), np.reshape(labels2, (-1, 1)), ftrs), axis=1),
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
            columns=xcats,
            dtype=np.float32,
        )

        self.X = df.drop(columns=["XR", "XR2"])
        self.X_numpy = self.X.values
        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y = df[["XR", "XR2"]]
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
        """
        Test inputs of the fit method are checked for correctness.
        """
        model = LinearMultiTargetRegression()
        # Test type of 'X' parameter
        self.assertRaises(TypeError, model.fit, X=1, y=self.y)
        self.assertRaises(TypeError, model.fit, X="X", y=self.y)
        self.assertRaises(TypeError, model.fit, X=self.X.values, y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index(), y=self.y)
        # Test type of 'y' parameter
        self.assertRaises(TypeError, model.fit, X=self.X, y=1)
        self.assertRaises(TypeError, model.fit, X=self.X, y="y")
        self.assertRaises(TypeError, model.fit, X=self.X, y=self.y.values)
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y_nan)
        # Test type of sample_weight
        self.assertRaises(TypeError, model.fit, X=self.X, y=self.y, sample_weight="weight")
        self.assertRaises(TypeError, model.fit, X=self.X, y=self.y, sample_weight=["weight"] * len(self.X))
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y, sample_weight=[1.0, 2.0])
        self.assertRaises(TypeError, model.fit, X=self.X, y=self.y, sample_weight=[1.0, "two", 3.0] * (len(self.X)//3))
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y, sample_weight=[1.0, -2.0, 3.0] * (len(self.X)//3))


    @parameterized.expand([True, False])
    def test_fit_valid(self, fit_intercept):
        """
        Test that the fit method works as expected
        """
        # If seemingly_unrelated is False, should reduce to multiple independent OLS
        lmtr = LinearMultiTargetRegression(
            fit_intercept=fit_intercept,
            seemingly_unrelated=False,
        ).fit(X=self.X, y=self.y)
        lr = LinearRegression(fit_intercept=fit_intercept).fit(X=self.X_numpy, y=self.y_numpy)
        np.testing.assert_array_almost_equal(lmtr.coefs_["XR"], lr.coef_[0,:])
        np.testing.assert_array_almost_equal(lmtr.coefs_["XR2"], lr.coef_[1,:])
        if fit_intercept:
            np.testing.assert_array_almost_equal(lmtr.intercepts_["XR"], lr.intercept_[0])
            np.testing.assert_array_almost_equal(lmtr.intercepts_["XR2"], lr.intercept_[1])
        else:
            np.testing.assert_array_almost_equal(lmtr.intercepts_["XR"], 0.0)
            np.testing.assert_array_almost_equal(lmtr.intercepts_["XR2"], 0.0)

        # If seemingly_unrelated is False and a selector is provided, should reduce to
        # independent OLS with feature selection
        lmtr_fs = LinearMultiTargetRegression(
            fit_intercept=fit_intercept,
            seemingly_unrelated=False,
            feature_selection=LarsSelector(n_factors=1),
        ).fit(X=self.X, y=self.y)
        for target in ["XR", "XR2"]:
            X_new = LarsSelector(n_factors=1).fit_transform(X=self.X, y=self.y[target])
            lr_fs = LinearRegression(fit_intercept=fit_intercept).fit(
                X=X_new,
                y=self.y[target],
            )
            np.testing.assert_array_almost_equal(
                lmtr_fs.coefs_[target],
                lr_fs.coef_,
            )
            if fit_intercept:
                np.testing.assert_array_almost_equal(
                    lmtr_fs.intercepts_[target],
                    lr_fs.intercept_,
                )
            else:
                np.testing.assert_array_almost_equal(
                    lmtr_fs.intercepts_[target],
                    0.0,
                )

        # If seemingly_unrelated is True and all independent variables are the same across equations,
        # should reduce to separate OLS on each equation.
        lmtr_su = LinearMultiTargetRegression(
            fit_intercept=fit_intercept,
            seemingly_unrelated=True,
        ).fit(X=self.X, y=self.y)
        lr = LinearRegression(fit_intercept=fit_intercept).fit(X=self.X_numpy, y=self.y_numpy)
        np.testing.assert_array_almost_equal(lmtr_su.coefs_["XR"], lr.coef_[0,:])
        np.testing.assert_array_almost_equal(lmtr_su.coefs_["XR2"], lr.coef_[1,:])
        if fit_intercept:
            np.testing.assert_array_almost_equal(lmtr_su.intercepts_["XR"], lr.intercept_[0])
            np.testing.assert_array_almost_equal(lmtr_su.intercepts_["XR2"], lr.intercept_[1])
        else:
            np.testing.assert_array_almost_equal(lmtr_su.intercepts_["XR"], 0.0)
            np.testing.assert_array_almost_equal(lmtr_su.intercepts_["XR2"], 0.0)

    @parameterized.expand([True, False])
    def test_predict_types(self, seemingly_unrelated):
        """
        Test inputs of the fit method are checked for correctness.
        """
        model = LinearMultiTargetRegression(seemingly_unrelated=seemingly_unrelated).fit(X=self.X, y=self.y)
        # Test type of 'X' parameter
        self.assertRaises(TypeError, model.predict, X=1)
        self.assertRaises(TypeError, model.predict, X="X")
        self.assertRaises(TypeError, model.predict, X=self.X.values)
        self.assertRaises(ValueError, model.predict, X=self.X_nan)
        self.assertRaises(ValueError, model.predict, X=self.X.reset_index())

    def test_predict_valid(self):
        """
        Test that the predict method works as expected
        """
        # Test it returns correct shape and numpy array
        lmtr = LinearMultiTargetRegression(
            fit_intercept=True,
            seemingly_unrelated=True,
            feature_selection=LarsSelector(n_factors=1),
        ).fit(X=self.X, y=self.y)
        y_pred = lmtr.predict(X=self.X)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape, (self.X.shape[0], self.y.shape[1]))