import numpy as np
import pandas as pd
import unittest
import matplotlib
import matplotlib.pyplot as plt

from macrosynergy.management import make_qdf
from unittest.mock import patch

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from macrosynergy.learning import PanelPCA, PanelStandardScaler, LassoSelector

class TestReturnForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        self.cids = ["AUD", "CAD", "GBP", "USD"]
        self.xcats = ["XR", "CPI", "GROWTH", "RIR"]
        cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD"] = ["2014-01-01", "2020-12-31", 0, 1]
        df_cids.loc["CAD"] = ["2015-01-01", "2020-12-31", 0, 1]
        df_cids.loc["GBP"] = ["2015-01-01", "2020-12-31", 0, 1]
        df_cids.loc["USD"] = ["2015-01-01", "2020-12-31", 0, 1]

        df_xcats = pd.DataFrame(index=self.xcats, columns=cols)
        df_xcats.loc["XR"] = ["2014-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CPI"] = ["2015-01-01", "2020-12-31", 1, 2, 0.95, 1]
        df_xcats.loc["GROWTH"] = ["2015-01-01", "2020-12-31", 1, 2, 0.9, 1]
        df_xcats.loc["RIR"] = ["2015-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

        self.df = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.df["value"] = self.df["value"].astype("float32")

        # Set up different experiments
        # (1) Ridge regression 
        # (2) Variable selection + Linear regression
        # (3) Random forest regression
        # (4) PCA + Ridge regression
        # In addition to specialised tests for each of these models, I need to test that
        # SignalOptimizer gives the same results as ReturnForecaster. 

        self.pipelines = [
            {
                "Ridge": Pipeline([
                    ("scaler", PanelStandardScaler()),
                    ("ridge", Ridge())
                ]),
            },
            {
                "Var+LR": Pipeline([
                    ("scaler", PanelStandardScaler()),
                    ("selector", LassoSelector(n_components = 2)),
                    ("lr", LinearRegression())
                ]),
            },
            {
                "RF": RandomForestRegressor(random_state = 1),
            },
            {
                "PCA+Ridge": Pipeline([
                    ("scaler", PanelStandardScaler()),
                    ("pca", PanelPCA(n_components = 2)),
                    ("ridge", Ridge())
                ]),
            }
        ]

        self.hyperparameters = [
            {
                "Ridge": {
                    "ridge__alpha": [0.1, 1, 10]
                }
            },
            {
                "Var+LR": {}
            },
            {
                "RF": {
                    "min_samples_leaf": [1, 32]
                }
            },
            {
                "PCA+Ridge": {
                    "ridge__alpha": [0.1, 1, 10]
                }
            }
        ]