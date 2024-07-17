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
