import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from macrosynergy.compat import PD_NEW_DATE_FREQ
from macrosynergy.learning import ModelAveragingRegressor


class TestModelAveragingFeatureImportances(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        rng = np.random.default_rng(0)
        cids = ["AUD", "CAD", "GBP", "USD"]
        freq = "BME" if PD_NEW_DATE_FREQ else "BM"
        dates = pd.date_range("2010-01-01", periods=60, freq=freq)
        index = pd.MultiIndex.from_product([cids, dates], names=["cid", "real_date"])

        self.features = ["CPI", "GROWTH", "RIR"]
        self.X = pd.DataFrame(
            rng.normal(size=(len(index), 3)), index=index, columns=self.features
        )

        signal = 2 * self.X["CPI"] - 1 * self.X["GROWTH"] + 0 * self.X["RIR"]
        self.y = (signal + 0.1 * rng.normal(size=len(index))).rename("XR")

    def test_all_models_have_coef_(self):
        estimators = [("lr", LinearRegression(), {}), ("ridge", Ridge(), {})]

        ma_reg = ModelAveragingRegressor(
            estimators=estimators,
            cv=KFold(n_splits=3),
            temperature=1.0,
        ).fit(self.X, self.y)

        result = ma_reg.coef_
        expected = [2, -1, 0]

        np.testing.assert_allclose(result, expected, atol=1e-2)
        self.assertTrue(not hasattr(ma_reg, "feature_importances_"))

    def test_all_models_have_feature_importances_(self):
        estimators = [
            ("rf", RandomForestRegressor(), {}),
            ("dt", DecisionTreeRegressor(), {}),
        ]

        ma_reg = ModelAveragingRegressor(
            estimators=estimators,
            cv=KFold(n_splits=3),
            temperature=1.0,
        ).fit(self.X, self.y)

        dt_fi = np.array(ma_reg.best_estimators_["dt"].feature_importances_)
        rf_fi = np.array(ma_reg.best_estimators_["rf"].feature_importances_)

        dt_wgt = ma_reg.weights_["dt"]
        rf_wgt = ma_reg.weights_["rf"]

        result = ma_reg.feature_importances_
        expected = dt_wgt * dt_fi + rf_wgt * rf_fi

        np.testing.assert_allclose(result, expected, atol=1e-5)
        self.assertTrue(not hasattr(ma_reg, "coef_"))

    def test_all_models_have_coef_or_feature_importances_(self):
        estimators = [
            ("lr", LinearRegression(), {}),
            ("dt", DecisionTreeRegressor(), {}),
        ]

        ma_reg = ModelAveragingRegressor(
            estimators=estimators,
            cv=KFold(n_splits=3),
            temperature=1.0,
        ).fit(self.X, self.y)

        dt_fi = np.abs(ma_reg.best_estimators_["dt"].feature_importances_)
        lr_fi = np.abs(ma_reg.best_estimators_["lr"].coef_)

        dt_fi /= dt_fi.sum()
        lr_fi /= lr_fi.sum()

        dt_wgt = ma_reg.weights_["dt"]
        rf_wgt = ma_reg.weights_["lr"]

        result = ma_reg.feature_importances_
        expected = dt_wgt * dt_fi + rf_wgt * lr_fi

        np.testing.assert_allclose(result, expected, atol=1e-5)
        self.assertTrue(not hasattr(ma_reg, "coef_"))

    def test_not_all_models_have_coef_or_feature_importances_(self):
        estimators = [
            ("lr", LinearRegression(), {}),
            ("dt", DecisionTreeRegressor(), {}),
            ("dt", KNeighborsRegressor(), {}),
        ]

        ma_reg = ModelAveragingRegressor(
            estimators=estimators,
            cv=KFold(n_splits=3),
            temperature=1.0,
        ).fit(self.X, self.y)

        self.assertTrue(not hasattr(ma_reg, "coef_"))
        self.assertTrue(not hasattr(ma_reg, "feature_importances_"))

    def test_pipelines(self):
        selector_pipe = Pipeline(
            [
                ("selector", SelectKBest(f_regression, k=2)),
                ("lr", LinearRegression()),
            ]
        )
        estimators = [("lr", LinearRegression(), {}), ("pipe", selector_pipe, {})]

        ma_reg = ModelAveragingRegressor(
            estimators=estimators,
            cv=KFold(n_splits=3),
            temperature=1.0,
        ).fit(self.X, self.y)

        # The selector drops RIR, which is assigned a zero coefficient
        pipe_coef = np.zeros(3)
        pipe_coef[:2] = ma_reg.best_estimators_["pipe"][-1].coef_
        lr_coef = ma_reg.best_estimators_["lr"].coef_

        result = ma_reg.coef_
        expected = ma_reg.weights_["lr"] * lr_coef + ma_reg.weights_["pipe"] * pipe_coef

        np.testing.assert_allclose(result, expected, atol=1e-5)
        self.assertTrue(not hasattr(ma_reg, "feature_importances_"))


if __name__ == "__main__":
    unittest.main()
