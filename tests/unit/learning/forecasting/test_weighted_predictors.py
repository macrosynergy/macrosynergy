import numpy as np
import pandas as pd 

from macrosynergy.learning import TimeWeightedWrapper

from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LinearRegression

import unittest 

class TestTWMetaEstimator(unittest.TestCase):
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

    def test_types_init(self):
        """
        Test constructor type and value checks
        """
        # model should inherit from BaseEstimator or RegressorMixin
        self.assertRaises(TypeError, TimeWeightedWrapper, model="string", half_life=5)
        self.assertRaises(TypeError, TimeWeightedWrapper, model=EmpiricalCovariance(), half_life=5)
        # half_life should be a positive number
        self.assertRaises(TypeError, TimeWeightedWrapper, model=LinearRegression(), half_life="string")
        self.assertRaises(ValueError, TimeWeightedWrapper, model=LinearRegression(), half_life=-1)
        self.assertRaises(ValueError, TimeWeightedWrapper, model=LinearRegression(), half_life=0)

    def test_valid_init(self):
        """
        Test constructor correctly initializes with valid parameters.
        """
        model = TimeWeightedWrapper(model=LinearRegression(), half_life=36)
        self.assertIsInstance(model, TimeWeightedWrapper)
        self.assertIsInstance(model.model, LinearRegression)
        self.assertEqual(model.half_life, 36)


    def test_types_fit(self):
        """
        Test inputs of the fit method are checked for correctness.
        """
        model = TimeWeightedWrapper(model=LinearRegression(), half_life=36)
        # Test type of 'X' parameter
        self.assertRaises(TypeError, model.fit, X=1, y=self.y)
        self.assertRaises(TypeError, model.fit, X="X", y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index(), y=self.y)
        # Test type of 'y' parameter
        self.assertRaises(TypeError, model.fit, X=self.X, y=1)
        self.assertRaises(TypeError, model.fit, X=self.X, y="y")
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y_nan)

    def test_valid_fit(self):
        """
        Test the fit method correctly trains a model with time weighting
        """ 
        # This should be equivalent to a linear regression with sample weights
        # passed into the fit method. Check coefficients are very similar. 
        model_tw = TimeWeightedWrapper(model=LinearRegression(), half_life=36).fit(self.X, self.y)
        model_sw = LinearRegression()
        dates = sorted(self.y.index.get_level_values(1).unique(), reverse=True)
        num_dates = len(dates)
        weights = np.power(2, -np.arange(num_dates) / 36)
        weight_map = dict(zip(dates, weights))
        sample_weights = self.y.index.get_level_values(1).map(weight_map).to_numpy()
        model_sw.fit(self.X, self.y, sample_weight=sample_weights)
        np.testing.assert_allclose(model_tw.model.coef_, model_sw.coef_, rtol=1e-5)


    def test_types_predict(self):
        """
        Test inputs of the transform method are checked for correctness.
        """
        model = TimeWeightedWrapper(
            model=LinearRegression(), half_life=36
        ).fit(self.X, self.y)

        # Test type of 'X' parameter
        self.assertRaises(TypeError, model.predict, X=1)
        self.assertRaises(TypeError, model.predict, X="X")

        # Test validity of X against what was seen in training
        self.assertRaises(ValueError, model.predict, X=self.X.iloc[:,:-1])
        self.assertRaises(ValueError, model.predict, X=self.X_nan)
        self.assertRaises(ValueError, model.predict, X=self.X.reset_index())

    def test_valid_predict(self):
        # Test correct shape and type of predictions
        model = TimeWeightedWrapper(model=LinearRegression(), half_life=36).fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertIsInstance(preds, np.ndarray)
        self.assertTrue(len(preds) == self.X.shape[0])
        # Check that predictions are the same as through the sklearn API
        preds_sklearn = model.model.predict(self.X)
        np.testing.assert_array_equal(preds, preds_sklearn)