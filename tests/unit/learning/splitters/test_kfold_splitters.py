from typing import List
import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch
from parameterized import parameterized

from macrosynergy.learning import (
    ExpandingKFoldPanelSplit,
    RollingKFoldPanelSplit,
    RecencyKFoldPanelSplit,
)

class TestExpandingKFold(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2015-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2014-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2015-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2015-01-01", "2020-12-31"]

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

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(self.mpl_backend)

    def test_types_init(self):
        with self.assertRaises(TypeError):
            ExpandingKFoldPanelSplit(n_splits="a")
        with self.assertRaises(TypeError):
            ExpandingKFoldPanelSplit(n_splits=1.5)
        with self.assertRaises(ValueError):
            ExpandingKFoldPanelSplit(n_splits=0)
        with self.assertRaises(ValueError):
            ExpandingKFoldPanelSplit(n_splits=1)
        with self.assertRaises(ValueError):
            ExpandingKFoldPanelSplit(n_splits=-2)

    def test_valid_init(self):
        try:
            splitter = ExpandingKFoldPanelSplit(n_splits=5)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

        self.assertTrue(splitter.n_splits == 5)

    def test_types_split(self):
        splitter = ExpandingKFoldPanelSplit(n_splits=5)
        with self.assertRaises(TypeError):
            next(splitter.split(X="a", y=self.y))
        with self.assertRaises(TypeError):
            next(splitter.split(X=self.X, y="a"))

    def test_valid_splits(self):
        splitter = ExpandingKFoldPanelSplit(n_splits=5)
        splits = list(splitter.split(X=self.X, y=self.y))
        self.assertTrue(len(splits) == 5)
        

    def test_types_transform(self):
        pass 

    def test_valid_transform(self):
        pass
    
if __name__ == "__main__":
    Test = TestExpandingKFold()
    Test.setUpClass()
    Test.test_types_split()