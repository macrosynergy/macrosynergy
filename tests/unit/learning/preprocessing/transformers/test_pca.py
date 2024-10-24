import numpy as np
import pandas as pd
import unittest
import itertools
from parameterized import parameterized

from macrosynergy.learning import PanelPCA, PanelStandardScaler

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class TestPanelPCA(unittest.TestCase):
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

    def test_types_init(self):
        # n_components should be an int, float or None
        with self.assertRaises(TypeError):
            PanelPCA(n_components="a")
        with self.assertRaises(TypeError):
            PanelPCA(n_components=True)
        with self.assertRaises(ValueError):
            PanelPCA(n_components=-1)
        with self.assertRaises(ValueError):
            PanelPCA(n_components=0)
        with self.assertRaises(ValueError):
            PanelPCA(n_components=1.2)

        # kaiser_criterion should be a bool
        with self.assertRaises(TypeError):
            PanelPCA(kaiser_criterion="a")
        with self.assertRaises(TypeError):
            PanelPCA(kaiser_criterion=1)

        # adjust_signs should be a bool
        with self.assertRaises(TypeError):
            PanelPCA(adjust_signs="a")
        with self.assertRaises(TypeError):
            PanelPCA(adjust_signs=1)

    def test_valid_init(self):
        # Test default values
        pca = PanelPCA()
        self.assertEqual(pca.n_components, None)
        self.assertEqual(pca.kaiser_criterion, False)
        self.assertEqual(pca.adjust_signs, False)
        self.assertIsInstance(pca, TransformerMixin)
        self.assertIsInstance(pca, PanelPCA)
        self.assertIsInstance(pca, BaseEstimator)

        # Test alternative values
        pca = PanelPCA(
            n_components = 0.5,
            kaiser_criterion = True,
            adjust_signs = True,
        )

        self.assertEqual(pca.n_components, 0.5)
        self.assertEqual(pca.kaiser_criterion, True)
        self.assertEqual(pca.adjust_signs, True)
        self.assertIsInstance(pca, TransformerMixin)
        self.assertIsInstance(pca, PanelPCA)
        self.assertIsInstance(pca, BaseEstimator)

    def test_types_fit(self):
        scaler = PanelPCA()
        # Test type of 'X' parameter
        self.assertRaises(TypeError, scaler.fit, X=1, y=self.y)
        self.assertRaises(TypeError, scaler.fit, X="X", y=self.y)
        self.assertRaises(TypeError, scaler.fit, X=self.X.values, y=self.y)
        self.assertRaises(ValueError, scaler.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, scaler.fit, X=self.X.reset_index(), y=self.y)
        # Test type of 'y' parameter
        scaler = PanelPCA(adjust_signs=True)
        self.assertRaises(TypeError, scaler.fit, X=self.X, y=1)
        self.assertRaises(TypeError, scaler.fit, X=self.X, y="y")
        self.assertRaises(ValueError, scaler.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(ValueError, scaler.fit, X=self.X, y=self.y_nan)

    @parameterized.expand(["numpy", "series", "dataframe"])
    def test_valid_fit(self, y_type):
        if y_type == "numpy":
            y = self.y.values
        elif y_type == "series":
            y = self.y
        elif y_type == "dataframe":
            y = self.y.to_frame()

        # Test default initialization
        pca = PanelPCA()
        try:
            pca.fit(X=self.X, y=y)
        except Exception as e:
            self.fail(f"Unexpected exception {e}")

        self.assertIsInstance(pca.adjusted_evecs, np.ndarray)
        self.assertTrue(pca.adjusted_evecs.shape[0] == self.X.shape[1])
        self.assertTrue(pca.adjusted_evecs.shape[1] == self.X.shape[1])
        self.assertIsInstance(pca.adjusted_evals, np.ndarray)
        self.assertTrue(pca.adjusted_evals.shape[0] == self.X.shape[1])
        self.assertTrue(np.all(pca.adjusted_evals >= 0))
        self.assertTrue(np.all([isinstance(e, (float, np.floating)) for e in pca.adjusted_evals]))

        # Test setting n_components to an int
        pca = PanelPCA(n_components = 2)
        try:
            pca.fit(X=self.X, y=y)
        except Exception as e:
            self.fail(f"Unexpected exception {e}")

        self.assertIsInstance(pca.adjusted_evecs, np.ndarray)
        self.assertTrue(pca.adjusted_evecs.shape[0] == self.X.shape[1])
        self.assertTrue(pca.adjusted_evecs.shape[1] == 2)
        self.assertIsInstance(pca.adjusted_evals, np.ndarray)
        self.assertTrue(pca.adjusted_evals.shape[0] == 2)
        self.assertTrue(np.all(pca.adjusted_evals >= 0))
        self.assertTrue(np.all([isinstance(e, (float, np.floating)) for e in pca.adjusted_evals]))

        self.assertIsInstance(pca, TransformerMixin)
        self.assertIsInstance(pca, PanelPCA)
        self.assertIsInstance(pca, BaseEstimator)

        # Test setting n_components to a float

        pca = PanelPCA(n_components = 0.95)
        try:
            pca.fit(X=self.X, y=y)
        except Exception as e:
            self.fail(f"Unexpected exception {e}")

        self.assertIsInstance(pca.adjusted_evecs, np.ndarray)
        self.assertTrue(pca.adjusted_evecs.shape[0] == self.X.shape[1])
        self.assertIsInstance(pca.adjusted_evals, np.ndarray)
        self.assertTrue(pca.adjusted_evals.shape[0] < self.X.shape[1])
        self.assertTrue(np.all(pca.adjusted_evals >= 0))
        self.assertTrue(np.all([isinstance(e, (float, np.floating)) for e in pca.adjusted_evals]))

        self.assertIsInstance(pca, TransformerMixin)
        self.assertIsInstance(pca, PanelPCA)
        self.assertIsInstance(pca, BaseEstimator)

        # Test setting kaiser_criterion to True
        pca = PanelPCA(kaiser_criterion = True)
        try:
            pca.fit(X=self.X, y=y)
        except Exception as e:
            self.fail(f"Unexpected exception {e}")
        
        self.assertIsInstance(pca.adjusted_evecs, np.ndarray)
        self.assertTrue(pca.adjusted_evecs.shape[0] == self.X.shape[1])
        self.assertIsInstance(pca.adjusted_evals, np.ndarray)
        self.assertTrue(pca.adjusted_evals.shape[0] < self.X.shape[1])
        self.assertTrue(np.all(pca.adjusted_evals >= 0))
        self.assertTrue(np.all([isinstance(e, (float, np.floating)) for e in pca.adjusted_evals]))

        self.assertIsInstance(pca, TransformerMixin)
        self.assertIsInstance(pca, PanelPCA)
        self.assertIsInstance(pca, BaseEstimator)

        # Test setting adjust_signs to True
        pca = PanelPCA(adjust_signs = True)
        try:
            pca.fit(X=self.X, y=y)
        except Exception as e:
            self.fail(f"Unexpected exception {e}")
        
        self.assertIsInstance(pca.adjusted_evecs, np.ndarray)
        self.assertTrue(pca.adjusted_evecs.shape[0] == self.X.shape[1])
        self.assertIsInstance(pca.adjusted_evals, np.ndarray)
        self.assertTrue(pca.adjusted_evals.shape[0] == self.X.shape[1])
        self.assertTrue(np.all(pca.adjusted_evals >= 0))
        self.assertTrue(np.all([isinstance(e, (float, np.floating)) for e in pca.adjusted_evals]))

        # Loop through each eigenvector and check the signs are correct
        for i in range(self.X.shape[1]):
            self.assertTrue(
                np.corrcoef(self.X.values @ pca.adjusted_evecs[:, i], self.y)[0, 1] > 0
            )

        self.assertIsInstance(pca, TransformerMixin)
        self.assertIsInstance(pca, PanelPCA)
        self.assertIsInstance(pca, BaseEstimator)
        
    def test_types_transform(self):
        scaler = PanelPCA().fit(self.X, self.y)
        # Test type of 'X' parameter
        self.assertRaises(TypeError, scaler.transform, X=1)
        self.assertRaises(TypeError, scaler.transform, X="X")
        self.assertRaises(TypeError, scaler.transform, X=self.X.values)
        self.assertRaises(ValueError, scaler.transform, X=self.X_nan)
        self.assertRaises(ValueError, scaler.transform, X=self.X.reset_index())
        self.assertRaises(ValueError, scaler.transform, X=self.X.drop(columns="CPI"))
        self.assertRaises(ValueError, scaler.transform, X=self.X.rename(columns={"CPI": "CPI2"}))

    @parameterized.expand(itertools.product([True, False], [0.95, 2]))
    def test_valid_transform(self, adjust_signs, n_components):
        # Test equivalence of transormation with sklearn PCA
        my_pca = Pipeline([
            ("scaler", PanelStandardScaler()),
            ("pca", PanelPCA(n_components = n_components, adjust_signs = adjust_signs)),
        ]).fit(self.X, self.y)

        sklearn_pca = Pipeline([
            ("scaler", PanelStandardScaler()),
            ("pca", PCA(n_components = n_components)),
        ]).fit(self.X, self.y)

        X_transformed_my = my_pca.transform(self.X)
        X_transformed_sklearn = sklearn_pca.transform(self.X)

        # Check dimensions are correct
        if n_components != 2:
            self.assertTrue(X_transformed_my.shape[0] == X_transformed_sklearn.shape[0])
            self.assertTrue(X_transformed_my.shape[1] + 1 == X_transformed_sklearn.shape[1])
        else:
            self.assertTrue(X_transformed_my.shape[0] == X_transformed_sklearn.shape[0])
            self.assertTrue(X_transformed_my.shape[1] == X_transformed_sklearn.shape[1])

        # Ensure the absolute values of each column are almost equal
        for i in range(X_transformed_my.shape[1]):
            np.testing.assert_almost_equal(
                np.abs(X_transformed_my.values[:, i]),
                np.abs(X_transformed_sklearn[:, i]),
                decimal=3,
            )