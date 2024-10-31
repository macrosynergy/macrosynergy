import numpy as np
import pandas as pd
import unittest
import itertools
from parameterized import parameterized

from macrosynergy.learning import PanelStandardScaler

class TestPanelStandardScaler(unittest.TestCase):
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

    def test_valid_init(self):
        # Test default initialization
        scaler = PanelStandardScaler()
        self.assertEqual(scaler.type, "panel")
        self.assertTrue(scaler.with_mean)
        self.assertTrue(scaler.with_std)
        self.assertTrue(scaler.feature_names_in_ is None)
        self.assertTrue(scaler.n_features_in_ is None)
        self.assertIsInstance(scaler, PanelStandardScaler)

        # Test custom initialization
        scaler = PanelStandardScaler(type="cross_section", with_mean=False, with_std=False)
        self.assertEqual(scaler.type, "cross_section")
        self.assertFalse(scaler.with_mean)
        self.assertFalse(scaler.with_std)
        self.assertTrue(scaler.feature_names_in_ is None)
        self.assertTrue(scaler.n_features_in_ is None)
        self.assertIsInstance(scaler, PanelStandardScaler)

        # Test another custom initialization
        scaler = PanelStandardScaler(type="cross_section", with_mean=False)
        self.assertEqual(scaler.type, "cross_section")
        self.assertFalse(scaler.with_mean)
        self.assertTrue(scaler.with_std)
        self.assertTrue(scaler.feature_names_in_ is None)
        self.assertTrue(scaler.n_features_in_ is None)
        self.assertIsInstance(scaler, PanelStandardScaler)



    def test_types_init(self):
        # Test type of 'type' parameter
        self.assertRaises(TypeError, PanelStandardScaler, type=1)
        self.assertRaises(ValueError, PanelStandardScaler, type="invalid") 
        # Test type of 'with_mean' parameter
        self.assertRaises(TypeError, PanelStandardScaler, with_mean=1)
        # Test type of 'with_std' parameter
        self.assertRaises(TypeError, PanelStandardScaler, with_std=1)

    def test_valid_fit(self):
        # Test panel scaling
        scaler = PanelStandardScaler()
        scaler.fit(self.X, self.y)
        self.assertIsInstance(scaler.statistics, dict)
        self.assertIsInstance(scaler.statistics["panel"], dict)
        self.assertEqual(len(scaler.statistics["panel"]), 3)
        self.assertIsInstance(scaler.statistics["panel"]["CPI"], list)
        self.assertEqual(scaler.statistics["panel"]["CPI"][0], self.X["CPI"].mean())
        self.assertEqual(scaler.statistics["panel"]["CPI"][1], self.X["CPI"].std())
        self.assertEqual(len(scaler.statistics["panel"]["CPI"]), 2)
        self.assertIsInstance(scaler.statistics["panel"]["GROWTH"], list)
        self.assertEqual(scaler.statistics["panel"]["GROWTH"][0], self.X["GROWTH"].mean())
        self.assertEqual(scaler.statistics["panel"]["GROWTH"][1], self.X["GROWTH"].std())
        self.assertEqual(len(scaler.statistics["panel"]["GROWTH"]), 2)
        self.assertIsInstance(scaler.statistics["panel"]["RIR"], list)
        self.assertEqual(scaler.statistics["panel"]["RIR"][0], self.X["RIR"].mean())
        self.assertEqual(scaler.statistics["panel"]["RIR"][1], self.X["RIR"].std())
        self.assertEqual(len(scaler.statistics["panel"]["RIR"]), 2)

        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)

        # Test cross-section scaling
        scaler = PanelStandardScaler(type="cross_section")
        scaler.fit(self.X, self.y)
        self.assertIsInstance(scaler.statistics, dict)
        self.assertIsInstance(scaler.statistics["panel"], dict)
        
        for cid in self.cids:
            self.assertIsInstance(scaler.statistics[cid], dict)
            self.assertEqual(len(scaler.statistics[cid]), 3)
            self.assertIsInstance(scaler.statistics[cid]["CPI"], list)
            self.assertEqual(scaler.statistics[cid]["CPI"][0], self.X.loc[cid]["CPI"].mean())
            self.assertEqual(scaler.statistics[cid]["CPI"][1], self.X.loc[cid]["CPI"].std())
            self.assertEqual(len(scaler.statistics[cid]["CPI"]), 2)
            self.assertIsInstance(scaler.statistics[cid]["GROWTH"], list)
            self.assertEqual(scaler.statistics[cid]["GROWTH"][0], self.X.loc[cid]["GROWTH"].mean())
            self.assertEqual(scaler.statistics[cid]["GROWTH"][1], self.X.loc[cid]["GROWTH"].std())
            self.assertEqual(len(scaler.statistics[cid]["GROWTH"]), 2)
            self.assertIsInstance(scaler.statistics[cid]["RIR"], list)
            self.assertEqual(scaler.statistics[cid]["RIR"][0], self.X.loc[cid]["RIR"].mean())
            self.assertEqual(scaler.statistics[cid]["RIR"][1], self.X.loc[cid]["RIR"].std())
            self.assertEqual(len(scaler.statistics[cid]["RIR"]), 2)

        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)

        # Panel scaling with no mean 
        # We still calculate these just don't transform using the mean
        scaler = PanelStandardScaler(with_mean=False)
        scaler.fit(self.X, self.y)
        self.assertIsInstance(scaler.statistics, dict)
        self.assertIsInstance(scaler.statistics["panel"], dict)
        self.assertEqual(len(scaler.statistics["panel"]), 3)
        self.assertIsInstance(scaler.statistics["panel"]["CPI"], list)
        self.assertEqual(scaler.statistics["panel"]["CPI"][0], self.X["CPI"].mean())
        self.assertEqual(scaler.statistics["panel"]["CPI"][1], self.X["CPI"].std())
        self.assertEqual(len(scaler.statistics["panel"]["CPI"]), 2)
        self.assertIsInstance(scaler.statistics["panel"]["GROWTH"], list)
        self.assertEqual(scaler.statistics["panel"]["GROWTH"][0], self.X["GROWTH"].mean())
        self.assertEqual(scaler.statistics["panel"]["GROWTH"][1], self.X["GROWTH"].std())
        self.assertEqual(len(scaler.statistics["panel"]["GROWTH"]), 2)
        self.assertIsInstance(scaler.statistics["panel"]["RIR"], list)
        self.assertEqual(scaler.statistics["panel"]["RIR"][0], self.X["RIR"].mean())
        self.assertEqual(scaler.statistics["panel"]["RIR"][1], self.X["RIR"].std())
        self.assertEqual(len(scaler.statistics["panel"]["RIR"]), 2)

        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)

        # Panel scaling with no std
        # We still calculate these just don't transform using the std
        scaler = PanelStandardScaler(with_std=False)
        scaler.fit(self.X, self.y)
        self.assertIsInstance(scaler.statistics, dict)
        self.assertIsInstance(scaler.statistics["panel"], dict)
        self.assertEqual(len(scaler.statistics["panel"]), 3)
        self.assertIsInstance(scaler.statistics["panel"]["CPI"], list)
        self.assertEqual(scaler.statistics["panel"]["CPI"][0], self.X["CPI"].mean())
        self.assertEqual(scaler.statistics["panel"]["CPI"][1], self.X["CPI"].std())
        self.assertEqual(len(scaler.statistics["panel"]["CPI"]), 2)
        self.assertIsInstance(scaler.statistics["panel"]["GROWTH"], list)
        self.assertEqual(scaler.statistics["panel"]["GROWTH"][0], self.X["GROWTH"].mean())
        self.assertEqual(scaler.statistics["panel"]["GROWTH"][1], self.X["GROWTH"].std())
        self.assertEqual(len(scaler.statistics["panel"]["GROWTH"]), 2)
        self.assertIsInstance(scaler.statistics["panel"]["RIR"], list)
        self.assertEqual(scaler.statistics["panel"]["RIR"][0], self.X["RIR"].mean())
        self.assertEqual(scaler.statistics["panel"]["RIR"][1], self.X["RIR"].std())
        self.assertEqual(len(scaler.statistics["panel"]["RIR"]), 2)

        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)

        # Cross-section scaling with no mean
        # We still calculate these just don't transform using the mean
        scaler = PanelStandardScaler(type="cross_section", with_mean=False)
        scaler.fit(self.X, self.y)
        self.assertIsInstance(scaler.statistics, dict)
        self.assertIsInstance(scaler.statistics["panel"], dict)

        for cid in self.cids:
            self.assertIsInstance(scaler.statistics[cid], dict)
            self.assertEqual(len(scaler.statistics[cid]), 3)
            self.assertIsInstance(scaler.statistics[cid]["CPI"], list)
            self.assertEqual(scaler.statistics[cid]["CPI"][0], self.X.loc[cid]["CPI"].mean())
            self.assertEqual(scaler.statistics[cid]["CPI"][1], self.X.loc[cid]["CPI"].std())
            self.assertEqual(len(scaler.statistics[cid]["CPI"]), 2)
            self.assertIsInstance(scaler.statistics[cid]["GROWTH"], list)
            self.assertEqual(scaler.statistics[cid]["GROWTH"][0], self.X.loc[cid]["GROWTH"].mean())
            self.assertEqual(scaler.statistics[cid]["GROWTH"][1], self.X.loc[cid]["GROWTH"].std())
            self.assertEqual(len(scaler.statistics[cid]["GROWTH"]), 2)
            self.assertIsInstance(scaler.statistics[cid]["RIR"], list)
            self.assertEqual(scaler.statistics[cid]["RIR"][0], self.X.loc[cid]["RIR"].mean())
            self.assertEqual(scaler.statistics[cid]["RIR"][1], self.X.loc[cid]["RIR"].std())
            self.assertEqual(len(scaler.statistics[cid]["RIR"]), 2)

        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)

        # Cross-section scaling with no std
        # We still calculate these just don't transform using the std
        scaler = PanelStandardScaler(type="cross_section", with_std=False)
        scaler.fit(self.X, self.y)
        self.assertIsInstance(scaler.statistics, dict)
        self.assertIsInstance(scaler.statistics["panel"], dict)

        for cid in self.cids:
            self.assertIsInstance(scaler.statistics[cid], dict)
            self.assertEqual(len(scaler.statistics[cid]), 3)
            self.assertIsInstance(scaler.statistics[cid]["CPI"], list)
            self.assertEqual(scaler.statistics[cid]["CPI"][0], self.X.loc[cid]["CPI"].mean())
            self.assertEqual(scaler.statistics[cid]["CPI"][1], self.X.loc[cid]["CPI"].std())
            self.assertEqual(len(scaler.statistics[cid]["CPI"]), 2)
            self.assertIsInstance(scaler.statistics[cid]["GROWTH"], list)
            self.assertEqual(scaler.statistics[cid]["GROWTH"][0], self.X.loc[cid]["GROWTH"].mean())
            self.assertEqual(scaler.statistics[cid]["GROWTH"][1], self.X.loc[cid]["GROWTH"].std())
            self.assertEqual(len(scaler.statistics[cid]["GROWTH"]), 2)
            self.assertIsInstance(scaler.statistics[cid]["RIR"], list)
            self.assertEqual(scaler.statistics[cid]["RIR"][0], self.X.loc[cid]["RIR"].mean())
            self.assertEqual(scaler.statistics[cid]["RIR"][1], self.X.loc[cid]["RIR"].std())
            self.assertEqual(len(scaler.statistics[cid]["RIR"]), 2)

        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)

    def test_types_fit(self):
        scaler = PanelStandardScaler()
        # Test type of 'X' parameter
        self.assertRaises(TypeError, scaler.fit, X=1, y=self.y)
        self.assertRaises(TypeError, scaler.fit, X="X", y=self.y)
        self.assertRaises(TypeError, scaler.fit, X=self.X.values, y=self.y)
        self.assertRaises(ValueError, scaler.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, scaler.fit, X=self.X.reset_index(), y=self.y)

    def test_types_transform(self):
        scaler = PanelStandardScaler()
        scaler.fit(self.X, self.y)
        # Test type of 'X' parameter
        self.assertRaises(TypeError, scaler.transform, X=1)
        self.assertRaises(TypeError, scaler.transform, X="X")
        self.assertRaises(TypeError, scaler.transform, X=self.X.values)
        self.assertRaises(ValueError, scaler.transform, X=self.X_nan)
        self.assertRaises(ValueError, scaler.transform, X=self.X.reset_index())
        self.assertRaises(ValueError, scaler.transform, X=self.X.drop(columns="CPI"))
        self.assertRaises(ValueError, scaler.transform, X=self.X.rename(columns={"CPI": "CPI2"}))

    def test_valid_transform(self):
        # Test panel scaling
        scaler = PanelStandardScaler()
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertTrue(np.allclose(X_transformed["CPI"], (self.X["CPI"] - self.X["CPI"].mean()) / self.X["CPI"].std()))
        self.assertTrue(np.allclose(X_transformed["GROWTH"], (self.X["GROWTH"] - self.X["GROWTH"].mean()) / self.X["GROWTH"].std()))
        self.assertTrue(np.allclose(X_transformed["RIR"], (self.X["RIR"] - self.X["RIR"].mean()) / self.X["RIR"].std()))

        # Test cross-section scaling
        scaler = PanelStandardScaler(type="cross_section")
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape, self.X.shape)
        for cid in self.cids:
            self.assertTrue(np.allclose(X_transformed.xs(cid)["CPI"], (self.X.xs(cid)["CPI"] - self.X.xs(cid)["CPI"].mean()) / self.X.xs(cid)["CPI"].std()))
            self.assertTrue(np.allclose(X_transformed.xs(cid)["GROWTH"], (self.X.xs(cid)["GROWTH"] - self.X.xs(cid)["GROWTH"].mean()) / self.X.xs(cid)["GROWTH"].std()))
            self.assertTrue(np.allclose(X_transformed.xs(cid)["RIR"], (self.X.xs(cid)["RIR"] - self.X.xs(cid)["RIR"].mean()) / self.X.xs(cid)["RIR"].std()))

        # Panel scaling with no mean 
        # We still calculate these just don't transform using the mean
        scaler = PanelStandardScaler(with_mean=False)
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertTrue(np.allclose(X_transformed["CPI"], self.X["CPI"] / self.X["CPI"].std()))
        self.assertTrue(np.allclose(X_transformed["GROWTH"], self.X["GROWTH"] / self.X["GROWTH"].std()))
        self.assertTrue(np.allclose(X_transformed["RIR"], self.X["RIR"] / self.X["RIR"].std()))

        # Panel scaling with no std
        # We still calculate these just don't transform using the std
        scaler = PanelStandardScaler(with_std=False)
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertTrue(np.allclose(X_transformed["CPI"], self.X["CPI"] - self.X["CPI"].mean()))
        self.assertTrue(np.allclose(X_transformed["GROWTH"], self.X["GROWTH"] - self.X["GROWTH"].mean()))
        self.assertTrue(np.allclose(X_transformed["RIR"], self.X["RIR"] - self.X["RIR"].mean()))

        # Cross-section scaling with no mean
        # We still calculate these just don't transform using the mean
        scaler = PanelStandardScaler(type="cross_section", with_mean=False)
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape, self.X.shape)
        for cid in self.cids:
            self.assertTrue(np.allclose(X_transformed.xs(cid)["CPI"], self.X.xs(cid)["CPI"] / self.X.xs(cid)["CPI"].std()))
            self.assertTrue(np.allclose(X_transformed.xs(cid)["GROWTH"], self.X.xs(cid)["GROWTH"] / self.X.xs(cid)["GROWTH"].std()))
            self.assertTrue(np.allclose(X_transformed.xs(cid)["RIR"], self.X.xs(cid)["RIR"] / self.X.xs(cid)["RIR"].std()))

        # Cross-section scaling with no std
        # We still calculate these just don't transform using the std
        scaler = PanelStandardScaler(type="cross_section", with_std=False)
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape, self.X.shape)
        for cid in self.cids:
            self.assertTrue(np.allclose(X_transformed.xs(cid)["CPI"], self.X.xs(cid)["CPI"] - self.X.xs(cid)["CPI"].mean()))
            self.assertTrue(np.allclose(X_transformed.xs(cid)["GROWTH"], self.X.xs(cid)["GROWTH"] - self.X.xs(cid)["GROWTH"].mean()))
            self.assertTrue(np.allclose(X_transformed.xs(cid)["RIR"], self.X.xs(cid)["RIR"] - self.X.xs(cid)["RIR"].mean()))