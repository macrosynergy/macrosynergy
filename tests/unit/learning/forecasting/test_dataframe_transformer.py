import numpy as np
import pandas as pd

import unittest

from parameterized import parameterized

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning.forecasting.meta_estimators.dataframe_transformer import DataFrameTransformer

class TestDataFrameTransformer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

        cls.X = df.drop(columns="XR")
        cls.X_numpy = cls.X.values
        cls.X_nan = cls.X.copy()
        cls.X_nan.iloc[:, 0] = np.nan  # Introduce NaN in the first column

        cls.y = np.sign( df["XR"] )
        cls.y_numpy = cls.y.values
        cls.y_nan = cls.y.copy()
        cls.y_nan.iloc[0] = np.nan  # Introduce NaN in the first element

    def test_types_init(self):
        """
        Test that the constructor raises TypeError or ValueError when
        invalid types are provided for the transformer or column_names.
        """
        # transformer
        with self.assertRaises(TypeError):
            dt = DataFrameTransformer(transformer = 1)
        # column_names
        with self.assertRaises(TypeError):
            dt = DataFrameTransformer(
                    transformer = PCA(),
                    column_names = 1,
            )
        with self.assertRaises(ValueError):
            dt = DataFrameTransformer(
                    transformer = PCA(),
                    column_names = [],
            )
        with self.assertRaises(ValueError):
            dt = DataFrameTransformer(
                    transformer = PCA(),
                    column_names = ["a", "b", 2],
            )
        with self.assertRaises(ValueError):
            dt = DataFrameTransformer(
                    transformer = PCA(),
                    column_names = ["a", "b", "a"],
            )

    def test_valid_init(self):
        """
        Test that the constructor works as designed.
        """
        # Test when no column_names are provided
        dt = DataFrameTransformer(
            transformer=PCA(),
        )
        self.assertIsInstance(dt, DataFrameTransformer)
        self.assertIsInstance(dt.transformer, PCA)
        self.assertIsNone(dt.column_names)
        # Test when column_names are provided
        dt = DataFrameTransformer(
            transformer=PCA(),
            column_names=["PCA1", "PCA2", "PCA3"],
        )
        self.assertIsInstance(dt, DataFrameTransformer)
        self.assertIsInstance(dt.transformer, PCA)
        self.assertEqual(dt.column_names, ["PCA1", "PCA2", "PCA3"])

    @parameterized.expand([PCA(), StandardScaler()])
    def test_types_fit(self, transformer):
        """
        Test that the fit method raises TypeError or ValueError when
        invalid types are provided for X or y.
        """
        # X - when a dataframe
        dt = DataFrameTransformer(transformer=transformer)
        self.assertRaises(TypeError, dt.fit, X=1, y=self.y)
        self.assertRaises(TypeError, dt.fit, X="X", y=self.y)
        self.assertRaises(ValueError, dt.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, dt.fit, X=self.X_nan, y=self.y)
        # y - when a series
        self.assertRaises(TypeError, dt.fit, X=self.X, y=1)
        self.assertRaises(TypeError, dt.fit, X=self.X, y="y")
        self.assertRaises(ValueError, dt.fit, X=self.X, y=self.y_nan)
        # y - when a dataframe
        self.assertRaises(ValueError, dt.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(
            ValueError,
            dt.fit,
            X=self.X,
            y=pd.DataFrame(self.y_nan.reset_index(drop=True)),
        )
        # y - when a numpy array
        self.assertRaises(
            ValueError, dt.fit, X=self.X, y=np.zeros((len(self.X), 2))
        )
        self.assertRaises(
            ValueError, dt.fit, X=self.X, y=np.array([np.nan] * len(self.X))
        )

        self.assertRaises(ValueError, dt.fit, X=self.X, y=self.y[:-1])

        # Raise error when column_names do not match the number of features
        with self.assertRaises(ValueError):
            dt = DataFrameTransformer(
                transformer=transformer,
                column_names=["PCA1", "PCA2", "PCA3", "PCA4"],
            ).fit(X=self.X, y=self.y)

    @parameterized.expand([PCA(), StandardScaler()])
    def test_valid_fit(self, transformer):
        """
        Check that the underling transformer is fitted correctly
        when valid data is provided, both when no column_names are provided
        and when they are provided.
        """
        # without column names
        dt = DataFrameTransformer(
            transformer=transformer,
        )
        dt.fit(self.X, self.y)
        if isinstance(dt.transformer, PCA):
            self.assertIsNotNone(dt.transformer.components_)
        elif isinstance(dt.transformer, StandardScaler):
            self.assertIsNotNone(dt.transformer.mean_)
            self.assertIsNotNone(dt.transformer.scale_)

        # with column names
        dt = DataFrameTransformer(
            transformer=transformer,
            column_names=["FACTOR1", "FACTOR2", "FACTOR3"],
        )
        dt.fit(self.X, self.y)
        if isinstance(dt.transformer, PCA):
            self.assertIsNotNone(dt.transformer.components_)
        elif isinstance(dt.transformer, StandardScaler):
            self.assertIsNotNone(dt.transformer.mean_)
            self.assertIsNotNone(dt.transformer.scale_)

    def test_types_transform(self):
        """
        Test that the transform method raises TypeError or ValueError when
        invalid types are provided for X.
        """
        dt = DataFrameTransformer(transformer=PCA()).fit(X=self.X, y=self.y)
        with self.assertRaises(TypeError):
            dt.transform(X=1)
        with self.assertRaises(TypeError):
            dt.transform(X="X")
        with self.assertRaises(TypeError):
            dt.transform(X=self.X.values)
        with self.assertRaises(ValueError):
            dt.transform(X=self.X.reset_index())

    def test_valid_transform(self):
        """
        Test that the transform method works correctly
        when valid data is provided, both when no column_names are provided
        and when they are provided.
        """
        # PCA without column names
        dt = DataFrameTransformer(transformer=PCA(n_components=2)).fit(X=self.X, y=self.y)
        transformed_data = dt.transform(X=self.X)
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertEqual(transformed_data.shape[0], self.X.shape[0])
        self.assertEqual(transformed_data.shape[1], dt.transformer.n_components_)
        self.assertTrue(np.all(np.isfinite(transformed_data.values)))
        self.assertIsInstance(transformed_data.index, pd.MultiIndex)
        self.assertEqual(transformed_data.index.names, ["cid", "real_date"])
        self.assertEqual(transformed_data.columns.tolist(), ['Factor_0', 'Factor_1'])

        # PCA with column names
        dt = DataFrameTransformer(transformer=PCA(n_components=2), column_names=["PCA1", "PCA2", "PCA3"]).fit(X=self.X, y=self.y)
        transformed_data = dt.transform(X=self.X)
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertEqual(transformed_data.shape[0], self.X.shape[0])
        self.assertEqual(transformed_data.shape[1], dt.transformer.n_components_)
        self.assertTrue(np.all(np.isfinite(transformed_data.values)))
        self.assertIsInstance(transformed_data.index, pd.MultiIndex)
        self.assertEqual(transformed_data.index.names, ["cid", "real_date"])
        self.assertEqual(transformed_data.columns.tolist(), ['PCA1', 'PCA2'])

        # StandardScaler without column names
        dt = DataFrameTransformer(transformer=StandardScaler()).fit(X=self.X, y=self.y)
        transformed_data = dt.transform(X=self.X)
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertEqual(transformed_data.shape[0], self.X.shape[0])
        self.assertEqual(transformed_data.shape[1], self.X.shape[1])
        self.assertTrue(np.all(np.isfinite(transformed_data.values)))
        self.assertIsInstance(transformed_data.index, pd.MultiIndex)
        self.assertEqual(transformed_data.index.names, ["cid", "real_date"])
        self.assertEqual(transformed_data.columns.tolist(), ["Factor_0", "Factor_1", "Factor_2"])

        # StandardScaler with column names
        dt = DataFrameTransformer(
            transformer=StandardScaler(),
            column_names=["Scaled1", "Scaled2", "Scaled3"],
        ).fit(X=self.X, y=self.y)
        transformed_data = dt.transform(X=self.X)
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertEqual(transformed_data.shape[0], self.X.shape[0])
        self.assertEqual(transformed_data.shape[1], self.X.shape[1])
        self.assertTrue(np.all(np.isfinite(transformed_data.values)))
        self.assertIsInstance(transformed_data.index, pd.MultiIndex)
        self.assertEqual(transformed_data.index.names, ["cid", "real_date"])
        self.assertEqual(transformed_data.columns.tolist(), ["Scaled1", "Scaled2", "Scaled3"])
