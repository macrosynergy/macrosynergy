import numpy as np
import pandas as pd

import unittest

from macrosynergy.learning import (
    panel_cv_scores,
    ExpandingKFoldPanelSplit,
)
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error


class TestAll(unittest.TestCase):
    def setUp(self):
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2020-06-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-06-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-09-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2020-09-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all elidgible dates
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

        self.splitter = ExpandingKFoldPanelSplit(n_splits=2)
        self.estimators = {"est1": LinearRegression(), "est2": Ridge(alpha=1)}

    def test_param_checks(self):
        scoring = {
            "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
        }
        # X
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=1,
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=scoring,
            )
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X.reset_index(),
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=scoring,
            )
        # y
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=1,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=scoring,
            )
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y.reset_index(),
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=scoring,
            )
        # splitter
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=1,
                estimators=self.estimators,
                scoring=scoring,
            )
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=KFold(),
                estimators=self.estimators,
                scoring=scoring,
            )
        # estimators
        with self.assertRaises(ValueError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators={},
                scoring=scoring,
            )
        bad_est_dict = 1
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=bad_est_dict,
                scoring=scoring,
            )
        bad_est_dict = {"est1": 1}
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=bad_est_dict,
                scoring=scoring,
            )
        bad_est_dict = {1: LinearRegression()}
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=bad_est_dict,
                scoring=scoring,
            )
        # scoring
        with self.assertRaises(ValueError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring={},
            )
        bad_scoring = "scorer"
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=bad_scoring,
            )
        bad_scoring = {1: make_scorer(mean_absolute_error, greater_is_better=True)}
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=bad_scoring,
            )
        # show_longbias
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=scoring,
                show_longbias=1,
            )
        # show_std
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=scoring,
                show_std=1,
            )
        # verbose
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=scoring,
                verbose="hello",
            )
        with self.assertRaises(ValueError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=scoring,
                verbose=-50,
            )
        # n_jobs
        with self.assertRaises(TypeError):
            panel_cv_scores(
                X=self.X,
                y=self.y,
                splitter=self.splitter,
                estimators=self.estimators,
                scoring=scoring,
                n_jobs="hello",
            )

    def test_output(self):
        scoring = {
            "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
        }
        output = panel_cv_scores(
            X=self.X,
            y=self.y,
            splitter=self.splitter,
            estimators=self.estimators,
            scoring=scoring,
            n_jobs=1,
        )
        self.assertIsInstance(output, pd.DataFrame)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 2)
        scoring = {
            "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
        }
        output = panel_cv_scores(
            X=self.X,
            y=self.y,
            splitter=self.splitter,
            estimators=self.estimators,
            scoring=scoring,
            show_std=True,
            n_jobs=1,
        )
        self.assertIsInstance(output, pd.DataFrame)
        self.assertEqual(output.shape[0], 8)
        self.assertEqual(output.shape[1], 2)
        scoring = {
            "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
        }
        output = panel_cv_scores(
            X=self.X,
            y=self.y,
            splitter=self.splitter,
            estimators=self.estimators,
            scoring=scoring,
            show_longbias=False,
            n_jobs=1,
        )
        self.assertIsInstance(output, pd.DataFrame)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 2)
        scoring = {
            "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
        }
        output = panel_cv_scores(
            X=self.X,
            y=self.y,
            splitter=self.splitter,
            estimators=self.estimators,
            scoring=scoring,
            show_longbias=False,
            show_std=True,
            n_jobs=1,
        )
        self.assertIsInstance(output, pd.DataFrame)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 2)
