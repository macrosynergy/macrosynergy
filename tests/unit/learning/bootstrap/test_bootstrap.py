import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import itertools

import unittest
from unittest.mock import patch

from macrosynergy.learning.predictors.bootstrap import BasePanelBootstrap
from parameterized import parameterized

class TestBasePanelBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2000-01-01", "2020-12-31"]

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

    @parameterized.expand(itertools.product(["panel", "period", "cross", "cross_per_period", "period_per_cross"], [0.1, 0.25, 0.5, 0.75, 0.9, 1]))
    def test_valid_init(self, bootstrap_method, resample_ratio):
        # Check that the class initializes correctly
        try:
            bpb = BasePanelBootstrap(
                bootstrap_method=bootstrap_method,
                resample_ratio=resample_ratio,
            )
        except Exception as e:
            self.fail(
                f"Failed to initialize BasePanelBootstrap with parameters bootstrap_method"
                f"={bootstrap_method} and resample_ratio={resample_ratio}: {e}"
            )

        # Check the class attributes have been set correctly
        self.assertEqual(bpb.bootstrap_method, bootstrap_method)
        self.assertEqual(bpb.resample_ratio, resample_ratio)
        self.assertTrue(bpb.max_features is None)

    def test_types_init(self):
        # Bootstrap method 
        with self.assertRaises(TypeError):
            BasePanelBootstrap(bootstrap_method=1)
        with self.assertRaises(ValueError):
            BasePanelBootstrap(bootstrap_method="invalid_method")
        # resample_ratio 
        with self.assertRaises(TypeError):
            BasePanelBootstrap(resample_ratio="invalid_ratio")
        with self.assertRaises(ValueError):
            BasePanelBootstrap(resample_ratio=-1)
        with self.assertRaises(ValueError):
            BasePanelBootstrap(resample_ratio=1.1)
        with self.assertRaises(ValueError):
            BasePanelBootstrap(resample_ratio=0)
        # max_features
        with self.assertRaises(NotImplementedError):
            BasePanelBootstrap(max_features=1)

    @parameterized.expand(itertools.product(["panel", "period", "cross", "cross_per_period", "period_per_cross"], [0.1, 0.25, 0.5, 0.75, 0.9, 1]))
    def test_valid_dataset(self, bootstrap_method, resample_ratio):
        bpb = BasePanelBootstrap(
            bootstrap_method=bootstrap_method,
            resample_ratio=resample_ratio,
        )
        # Test that each returned object is of the correct format
        X, y = bpb.create_bootstrap_dataset(self.X, self.y)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(len(X) > 0)
        self.assertIsInstance(y, pd.Series)
        self.assertTrue(len(y) > 0)
        self.assertTrue(len(X) == len(y))
        self.assertTrue(all(X.index == y.index))

        
    @parameterized.expand(itertools.product(["panel", "period", "cross", "cross_per_period", "period_per_cross"], [0.1, 0.25, 0.5, 0.75, 0.9, 1]))
    def test_types_dataset(self, bootstrap_method, resample_ratio):
        bpb = BasePanelBootstrap(
            bootstrap_method=bootstrap_method,
            resample_ratio=resample_ratio,
        )
        # X
        with self.assertRaises(TypeError):
            bpb.create_bootstrap_dataset(X=1, y=self.y)
        with self.assertRaises(TypeError):
            bpb.create_bootstrap_dataset(X="self.X", y=self.y)
        with self.assertRaises(ValueError):
            bpb.create_bootstrap_dataset(X=self.X.iloc[:20], y=self.y)
        with self.assertRaises(ValueError):
            bpb.create_bootstrap_dataset(X=self.X.reset_index(), y=self.y)
        # y
        with self.assertRaises(TypeError):
            bpb.create_bootstrap_dataset(X=self.X, y=1)
        with self.assertRaises(TypeError):
            bpb.create_bootstrap_dataset(X=self.X, y="self.y")
        with self.assertRaises(ValueError):
            bpb.create_bootstrap_dataset(X=self.X, y=self.y.iloc[:20])
        with self.assertRaises(ValueError):
            bpb.create_bootstrap_dataset(X=self.X, y=self.y.reset_index())

    @parameterized.expand([0.1, 0.25, 0.5, 0.75, 0.9, 1])
    def test_valid_panel_bootstrap(self, resample_ratio):
        bpb = BasePanelBootstrap(
            bootstrap_method="panel",
            resample_ratio=resample_ratio,
        )
        # Test that a correctly resampled dataset is returned
        np.random.seed(42)
        X_resample, y_resample = bpb._panel_bootstrap(self.X, self.y)
        np.random.seed(42)
        idx = np.random.choice([i for i in range(len(self.X))], size=int(np.ceil(resample_ratio * self.X.shape[0])), replace=True)
        X_resample_true = self.X.iloc[idx]
        y_resample_true = self.y.iloc[idx]
        self.assertTrue(all(X_resample == X_resample_true))