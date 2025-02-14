import os
import numbers
import numpy as np
import pandas as pd

import unittest
import itertools

from macrosynergy.learning import (
    KNNClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from parameterized import parameterized


class TestKNeighborsClassifier(unittest.TestCase):
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
        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.X_ones = self.X.copy()
        self.X_ones["intercept"] = 1
        self.y = np.sign(df["XR"])
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

    def test_types_init(self):
        # n_neighbors
        self.assertRaises(TypeError, KNNClassifier, n_neighbors=KNNClassifier())
        self.assertRaises(TypeError, KNNClassifier, n_neighbors=[1])
        self.assertRaises(TypeError, KNNClassifier, n_neighbors={1: 1})
        self.assertRaises(TypeError, KNNClassifier, n_neighbors=(1,))
        self.assertRaises(ValueError, KNNClassifier, n_neighbors=-1) 
        self.assertRaises(ValueError, KNNClassifier, n_neighbors=0)
        self.assertRaises(ValueError, KNNClassifier, n_neighbors=1.0)
        self.assertRaises(ValueError, KNNClassifier, n_neighbors=1.5)
        self.assertRaises(ValueError, KNNClassifier, n_neighbors="1")
        # weights
        self.assertRaises(TypeError, KNNClassifier, weights=1)
        self.assertRaises(ValueError, KNNClassifier, weights="hello")
        
    def test_valid_init(self):
        # Test default values
        knn = KNNClassifier()
        self.assertEqual(knn.n_neighbors, "sqrt")
        self.assertEqual(knn.weights, "uniform")
        self.assertEqual(knn.knn_, None)
        # Integer n_neighbors
        knn = KNNClassifier(n_neighbors=1, weights="distance")
        self.assertEqual(knn.n_neighbors, 1)
        self.assertEqual(knn.weights, "distance")
        self.assertEqual(knn.knn_, None)
        knn = KNNClassifier(n_neighbors=1, weights="uniform")
        self.assertEqual(knn.n_neighbors, 1)
        self.assertEqual(knn.weights, "uniform")
        self.assertEqual(knn.knn_, None)
        # Float n_neighbors
        knn = KNNClassifier(n_neighbors=0.5, weights="distance")
        self.assertEqual(knn.n_neighbors, 0.5)
        self.assertEqual(knn.weights, "distance")
        self.assertEqual(knn.knn_, None)
        knn = KNNClassifier(n_neighbors=0.5, weights="uniform")
        self.assertEqual(knn.n_neighbors, 0.5)
        self.assertEqual(knn.weights, "uniform")
        self.assertEqual(knn.knn_, None)

    def test_types_fit(self):
        # X - when a dataframe
        knn = KNNClassifier()
        self.assertRaises(TypeError, knn.fit, X=1, y=self.y)
        self.assertRaises(TypeError, knn.fit, X="X", y=self.y)
        self.assertRaises(ValueError, knn.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, knn.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, knn.fit, X=self.X_nan.values, y=self.y)
        # X - when a numpy array
        self.assertRaises(ValueError, knn.fit, X=self.X.reset_index().values, y=self.y)
        self.assertRaises(
            ValueError, knn.fit, X=self.X_nan.reset_index(drop=True).values, y=self.y
        )
        # y - when a series
        self.assertRaises(TypeError, knn.fit, X=self.X, y=1)
        self.assertRaises(TypeError, knn.fit, X=self.X, y="y")
        self.assertRaises(TypeError, knn.fit, X=self.X, y = pd.DataFrame(self.y))
        self.assertRaises(ValueError, knn.fit, X=self.X, y=self.y.reset_index()["cid"])
        self.assertRaises(ValueError, knn.fit, X=self.X, y=self.y_nan)
        self.assertRaises(ValueError, knn.fit, X=self.X, y=self.y_nan.values)
        # y - when a numpy array
        self.assertRaises(
            ValueError, knn.fit, X=self.X.values, y=np.zeros((len(self.X), 2))
        )
        self.assertRaises(
            ValueError, knn.fit, X=self.X.values, y=np.array([np.nan] * len(self.X))
        )

        self.assertRaises(ValueError, knn.fit, X=self.X, y=self.y[:-1])

    @parameterized.expand(["uniform", "distance"])
    def test_valid_fit(self, weights):
        """
        Test that the fit method works under different scenarios. The neighbors will be checked
        against the sklearn implementation.
        """
        # Default initialization
        knn_my = KNNClassifier(weights=weights).fit(self.X, self.y)
        knn_theirs = KNeighborsClassifier(n_neighbors = int(np.sqrt(len(self.X))), weights = weights).fit(self.X, self.y)
        self.assertTrue(np.allclose(knn_my.knn_.kneighbors(self.X)[0], knn_theirs.kneighbors(self.X)[0]))

        # Integer n_neighbors
        knn_my = KNNClassifier(n_neighbors=5, weights=weights).fit(self.X, self.y)
        knn_theirs = KNeighborsClassifier(n_neighbors = 5, weights = weights).fit(self.X, self.y)
        self.assertTrue(np.allclose(knn_my.knn_.kneighbors(self.X)[0], knn_theirs.kneighbors(self.X)[0]))

        # Floating point n_neighbors
        knn_my = KNNClassifier(n_neighbors=0.5, weights=weights).fit(self.X, self.y)
        knn_theirs = KNeighborsClassifier(n_neighbors = int(0.5 * len(self.X)), weights = weights).fit(self.X, self.y)
        self.assertTrue(np.allclose(knn_my.knn_.kneighbors(self.X)[0], knn_theirs.kneighbors(self.X)[0]))

    def test_types_predict(self):
        knn = KNNClassifier().fit(self.X, self.y)
        # X - when a dataframe
        self.assertRaises(TypeError, knn.predict, X=1)
        self.assertRaises(TypeError, knn.predict, X="X")
        self.assertRaises(ValueError, knn.predict, X=self.X.iloc[:, :-1])
        self.assertRaises(ValueError, knn.predict, X=self.X_nan.values)
        self.assertRaises(
            ValueError,
            knn.predict,
            X=pd.DataFrame(
                data = np.array([["hello"] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        self.assertRaises(
            ValueError,
            knn.predict,
            X=pd.DataFrame(
                data = np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        # X - when a numpy array
        self.assertRaises(ValueError, knn.predict, X=np.expand_dims(np.array(self.X), 0))
        self.assertRaises(
            ValueError,
            knn.predict,
            X=np.array([["hello"] * self.X.shape[1]] * self.X.shape[0])
        )
        self.assertRaises(
            ValueError,
            knn.predict,
            X=np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0])
        )

    @parameterized.expand(["uniform", "distance"])
    def test_valid_predict(self, weights):
        """
        Test that the predict method works under different initialization cases. 
        The predictions will be checked against the sklearn implementation.
        """
        # Default initialization
        knn_my = KNNClassifier(weights = weights).fit(self.X, self.y)
        knn_theirs = KNeighborsClassifier(n_neighbors = int(np.sqrt(len(self.X))), weights = weights).fit(self.X, self.y)
        self.assertTrue(np.allclose(knn_my.predict(self.X), knn_theirs.predict(self.X)))
        # Integer n_neighbors
        knn_my = KNNClassifier(n_neighbors=5, weights=weights).fit(self.X, self.y)
        knn_theirs = KNeighborsClassifier(n_neighbors = 5, weights = weights).fit(self.X, self.y)
        self.assertTrue(np.allclose(knn_my.predict(self.X), knn_theirs.predict(self.X)))
        # Floating point n_neighbors
        knn_my = KNNClassifier(n_neighbors=0.5, weights=weights).fit(self.X, self.y)
        knn_theirs = KNeighborsClassifier(n_neighbors = int(0.5 * len(self.X)), weights = weights).fit(self.X, self.y)
        self.assertTrue(np.allclose(knn_my.predict(self.X), knn_theirs.predict(self.X)))

    def test_types_predict_proba(self):
        knn = KNNClassifier().fit(self.X, self.y)
        # X - when a dataframe
        self.assertRaises(TypeError, knn.predict_proba, X=1)
        self.assertRaises(TypeError, knn.predict_proba, X="X")
        self.assertRaises(ValueError, knn.predict_proba, X=self.X.iloc[:, :-1])
        self.assertRaises(ValueError, knn.predict_proba, X=self.X_nan.values)
        self.assertRaises(
            ValueError,
            knn.predict_proba,
            X=pd.DataFrame(
                data = np.array([["hello"] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        self.assertRaises(
            ValueError,
            knn.predict_proba,
            X=pd.DataFrame(
                data = np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        # X - when a numpy array
        self.assertRaises(ValueError, knn.predict_proba, X=np.expand_dims(np.array(self.X), 0))
        self.assertRaises(
            ValueError,
            knn.predict_proba,
            X=np.array([["hello"] * self.X.shape[1]] * self.X.shape[0])
        )
        self.assertRaises(
            ValueError,
            knn.predict_proba,
            X=np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0])
        )

    @parameterized.expand(["uniform", "distance"])
    def test_valid_predict_proba(self, weights):
        """
        Test that the predict method works under different initialization cases. 
        The predictions will be checked against the sklearn implementation.
        """
        # Default initialization
        knn_my = KNNClassifier(weights = weights).fit(self.X, self.y)
        knn_theirs = KNeighborsClassifier(n_neighbors = int(np.sqrt(len(self.X))), weights = weights).fit(self.X, self.y)
        self.assertTrue(np.allclose(knn_my.predict_proba(self.X), knn_theirs.predict_proba(self.X)))
        # Integer n_neighbors
        knn_my = KNNClassifier(n_neighbors=5, weights=weights).fit(self.X, self.y)
        knn_theirs = KNeighborsClassifier(n_neighbors = 5, weights = weights).fit(self.X, self.y)
        self.assertTrue(np.allclose(knn_my.predict_proba(self.X), knn_theirs.predict_proba(self.X)))
        # Floating point n_neighbors
        knn_my = KNNClassifier(n_neighbors=0.5, weights=weights).fit(self.X, self.y)
        knn_theirs = KNeighborsClassifier(n_neighbors = int(0.5 * len(self.X)), weights = weights).fit(self.X, self.y)
        self.assertTrue(np.allclose(knn_my.predict_proba(self.X), knn_theirs.predict_proba(self.X)))