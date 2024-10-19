import datetime
import itertools
import unittest
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from parameterized import parameterized
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from macrosynergy.learning import (
    ExpandingIncrementPanelSplit,
    ExpandingKFoldPanelSplit,
    LassoSelector,
    RollingKFoldPanelSplit,
    SignalOptimizer,
    regression_balanced_accuracy,
)
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils.df_utils import categories_df


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
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

        # create blacklist dictionary
        self.black_valid = {
            "AUD": (
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2020, month=4, day=1),
            ),
            "GBP": (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2020, month=2, day=1),
            ),
        }
        self.black_invalid1 = {
            "AUD": ["2018-09-01", "2018-10-01"],
            "GBP": ["2019-06-01", "2100-01-01"],
        }
        self.black_invalid2 = {
            "AUD": ("2018-09-01", "2018-10-01"),
            "GBP": ("2019-06-01", "2100-01-01"),
        }
        self.black_invalid3 = {
            "AUD": [
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2018, month=10, day=1),
            ],
            "GBP": [
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2100, month=1, day=1),
            ],
        }
        self.black_invalid4 = {
            "AUD": (pd.Timestamp(year=2018, month=9, day=1),),
            "GBP": (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2100, month=1, day=1),
            ),
        }
        self.black_invalid5 = {
            1: (
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2018, month=10, day=1),
            ),
            2: (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2100, month=1, day=1),
            ),
        }

        # models dictionary
        self.models = {
            "linreg": LinearRegression(),
            "ridge": Ridge(),
        }
        self.hyperparameters = {
            "linreg": {},
            "ridge": {"alpha": [0.1, 1.0]},
        }
        self.scorers = {
            "r2": make_scorer(r2_score),
            "bac": make_scorer(regression_balanced_accuracy),
        }
        # # instantiate some splitters
        # expandingkfold = ExpandingKFoldPanelSplit(n_splits=4)
        # rollingkfold = RollingKFoldPanelSplit(n_splits=4)
        # expandingincrement = ExpandingIncrementPanelSplit(min_cids=2, min_periods=100)
        # self.splitters = {
        #     "expanding_kfold": expandingkfold,
        #     "rolling": rollingkfold,
        #     "expanding": expandingincrement,
        # }
        self.inner_splitters = {
            "ExpandingKFold": ExpandingKFoldPanelSplit(n_splits=4),
            "RollingKFold": RollingKFoldPanelSplit(n_splits=4),
        }
        self.single_inner_splitter = {
            "ExpandingKFold": ExpandingKFoldPanelSplit(n_splits=4),
        }

        so = SignalOptimizer(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
        )
        # Now run calculate_predictions
        so.calculate_predictions(
            name="test",
            models=self.models,
            scorers=self.scorers,
            hyperparameters=self.hyperparameters,
            search_type="grid",
            n_jobs_outer=1,
            n_jobs_inner=1,
            inner_splitters=self.single_inner_splitter,
        )
        self.so_with_calculated_preds = so

        self.X, self.y, self.df_long = _get_X_y(so)

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(self.mpl_backend)

    @parameterized.expand(itertools.product([True, False], [True, False]))
    def test_valid_init(self, use_blacklist, use_cids):
        try:
            blacklist = self.black_valid if use_blacklist else None
            cids = self.cids if use_cids else None
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                cids=cids,
                blacklist=blacklist,
            )
        except Exception as e:
            self.fail(f"Instantiation of the SignalOptimizer raised an exception: {e}")
        self.assertIsInstance(so, SignalOptimizer)
        X, y, df_long = _get_X_y(so)
        pd.testing.assert_frame_equal(so.X, X)
        pd.testing.assert_series_equal(so.y, y)

        pd.testing.assert_frame_equal(
            so.chosen_models,
            pd.DataFrame(
                columns=[
                    "real_date",
                    "name",
                    "model_type",
                    "score",
                    "hparams",
                    "n_splits_used",
                ]
            ),
        )
        pd.testing.assert_frame_equal(
            so.preds,
            pd.DataFrame(columns=["cid", "real_date", "xcat", "value"]),
        )
        pd.testing.assert_frame_equal(
            so.ftr_coefficients,
            pd.DataFrame(columns=["real_date", "name"] + list(so.X.columns)),
        )
        pd.testing.assert_frame_equal(
            so.intercepts, pd.DataFrame(columns=["real_date", "name", "intercepts"])
        )

        min_date = min(so.unique_date_levels)
        max_date = max(so.unique_date_levels)
        forecast_date_levels = pd.date_range(start=min_date, end=max_date, freq="B")
        pd.testing.assert_index_equal(
            so.forecast_idxs,
            pd.MultiIndex.from_product(
                [so.unique_xs_levels, forecast_date_levels],
                names=["cid", "real_date"],
            ),
        )

    def test_types_init(self):
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=pd.Series([]),
                xcats=self.xcats,
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats="invalid",
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=[1, 2, 3],
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                cids="invalid",
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                cids=[1, 2, 3],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=pd.DataFrame([]),
                xcats=self.xcats,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=["invalid"],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats + ["invalid"],
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                cids=self.cids + ["invalid"],
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                start=1,
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                end=1,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                start="invalid",
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                end="invalid",
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                blacklist=list(),
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                blacklist={1: "invalid"},
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                blacklist={"CAD": "invalid"},
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                df=self.df,
                xcats=self.xcats,
                blacklist={"CAD": tuple()},
            )

def _get_X_y(so: SignalOptimizer):
    df_long = (
        categories_df(
            df=so.df,
            xcats=so.xcats,
            cids=so.cids,
            start=so.start,
            end=so.end,
            blacklist=so.blacklist,
            freq=so.freq,
            lag=so.lag,
            xcat_aggs=so.xcat_aggs,
        )
        .dropna()
        .sort_index()
    )
    X = df_long.iloc[:, :-1]
    y = df_long.iloc[:, -1]
    return X, y, df_long