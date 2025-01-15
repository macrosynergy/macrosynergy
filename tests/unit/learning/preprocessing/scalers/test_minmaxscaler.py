import numpy as np
import pandas as pd
import unittest
import itertools
from parameterized import parameterized

from macrosynergy.learning import PanelMinMaxScaler

class TestPanelMinMaxScaler(unittest.TestCase):
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
        scaler = PanelMinMaxScaler()
        self.assertEqual(scaler.type, "panel")
        self.assertTrue(scaler.feature_names_in_ is None)
        self.assertTrue(scaler.n_features_in_ is None)
        self.assertIsInstance(scaler, PanelMinMaxScaler)

        # Test custom initialization
        scaler = PanelMinMaxScaler(type="cross_section")
        self.assertEqual(scaler.type, "cross_section")
        self.assertTrue(scaler.feature_names_in_ is None)
        self.assertTrue(scaler.n_features_in_ is None)
        self.assertIsInstance(scaler, PanelMinMaxScaler)

    def test_types_init(self):
        # Test type of 'type' parameter
        self.assertRaises(TypeError, PanelMinMaxScaler, type=1)
        self.assertRaises(ValueError, PanelMinMaxScaler, type="invalid") 

    def test_valid_fit(self):
        # Test panel scaling
        scaler = PanelMinMaxScaler()
        scaler.fit(self.X, self.y)
        self.assertIsInstance(scaler.statistics, dict)
        self.assertIsInstance(scaler.statistics["panel"], dict)
        self.assertEqual(len(scaler.statistics["panel"]), 3)
        self.assertIsInstance(scaler.statistics["panel"]["CPI"], list)
        self.assertEqual(scaler.statistics["panel"]["CPI"][0], self.X["CPI"].min())
        self.assertEqual(scaler.statistics["panel"]["CPI"][1], self.X["CPI"].max())
        self.assertEqual(len(scaler.statistics["panel"]["CPI"]), 2)
        self.assertIsInstance(scaler.statistics["panel"]["GROWTH"], list)
        self.assertEqual(scaler.statistics["panel"]["GROWTH"][0], self.X["GROWTH"].min())
        self.assertEqual(scaler.statistics["panel"]["GROWTH"][1], self.X["GROWTH"].max())
        self.assertEqual(len(scaler.statistics["panel"]["GROWTH"]), 2)
        self.assertIsInstance(scaler.statistics["panel"]["RIR"], list)
        self.assertEqual(scaler.statistics["panel"]["RIR"][0], self.X["RIR"].min())
        self.assertEqual(scaler.statistics["panel"]["RIR"][1], self.X["RIR"].max())
        self.assertEqual(len(scaler.statistics["panel"]["RIR"]), 2)

        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)

        # Test cross-section scaling
        scaler = PanelMinMaxScaler(type="cross_section")
        scaler.fit(self.X, self.y)
        self.assertIsInstance(scaler.statistics, dict)
        self.assertIsInstance(scaler.statistics["panel"], dict)
        
        for cid in self.cids:
            self.assertIsInstance(scaler.statistics[cid], dict)
            self.assertEqual(len(scaler.statistics[cid]), 3)
            self.assertIsInstance(scaler.statistics[cid]["CPI"], list)
            self.assertEqual(scaler.statistics[cid]["CPI"][0], self.X.loc[cid]["CPI"].min())
            self.assertEqual(scaler.statistics[cid]["CPI"][1], self.X.loc[cid]["CPI"].max())
            self.assertEqual(len(scaler.statistics[cid]["CPI"]), 2)
            self.assertIsInstance(scaler.statistics[cid]["GROWTH"], list)
            self.assertEqual(scaler.statistics[cid]["GROWTH"][0], self.X.loc[cid]["GROWTH"].min())
            self.assertEqual(scaler.statistics[cid]["GROWTH"][1], self.X.loc[cid]["GROWTH"].max())
            self.assertEqual(len(scaler.statistics[cid]["GROWTH"]), 2)
            self.assertIsInstance(scaler.statistics[cid]["RIR"], list)
            self.assertEqual(scaler.statistics[cid]["RIR"][0], self.X.loc[cid]["RIR"].min())
            self.assertEqual(scaler.statistics[cid]["RIR"][1], self.X.loc[cid]["RIR"].max())
            self.assertEqual(len(scaler.statistics[cid]["RIR"]), 2)

        self.assertTrue(scaler.n_features_in_ == 3)
        np.testing.assert_array_equal(scaler.feature_names_in_, self.X.columns)

    def test_types_fit(self):
        scaler = PanelMinMaxScaler()
        # Test type of 'X' parameter
        self.assertRaises(TypeError, scaler.fit, X=1, y=self.y)
        self.assertRaises(TypeError, scaler.fit, X="X", y=self.y)
        self.assertRaises(TypeError, scaler.fit, X=self.X.values, y=self.y)
        self.assertRaises(ValueError, scaler.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, scaler.fit, X=self.X.reset_index(), y=self.y)

    def test_types_transform(self):
        scaler = PanelMinMaxScaler()
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
        scaler = PanelMinMaxScaler()
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertTrue(np.allclose(X_transformed["CPI"], (self.X["CPI"] - self.X["CPI"].min()) / (self.X["CPI"].max() - self.X["CPI"].min())))
        self.assertTrue(np.allclose(X_transformed["GROWTH"], (self.X["GROWTH"] - self.X["GROWTH"].min()) / (self.X["GROWTH"].max() - self.X["GROWTH"].min())))
        self.assertTrue(np.allclose(X_transformed["RIR"], (self.X["RIR"] - self.X["RIR"].min()) / (self.X["RIR"].max() - self.X["RIR"].min())))

        # Test cross-section scaling
        scaler = PanelMinMaxScaler(type="cross_section")
        scaler.fit(self.X, self.y)
        X_transformed = scaler.transform(self.X)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape, self.X.shape)
        for cid in self.cids:
            self.assertTrue(np.allclose(X_transformed.xs(cid)["CPI"], (self.X.xs(cid)["CPI"] - self.X.xs(cid)["CPI"].min()) / (self.X.xs(cid)["CPI"].max() - self.X.xs(cid)["CPI"].min())))
            self.assertTrue(np.allclose(X_transformed.xs(cid)["GROWTH"], (self.X.xs(cid)["GROWTH"] - self.X.xs(cid)["GROWTH"].min()) / (self.X.xs(cid)["GROWTH"].max() - self.X.xs(cid)["GROWTH"].min())))
            self.assertTrue(np.allclose(X_transformed.xs(cid)["RIR"], (self.X.xs(cid)["RIR"] - self.X.xs(cid)["RIR"].min()) / (self.X.xs(cid)["RIR"].max() - self.X.xs(cid)["RIR"].min())))