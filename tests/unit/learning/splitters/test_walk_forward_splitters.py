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
    ExpandingIncrementPanelSplit,
)

class TestAll(unittest.TestCase):
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

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(self.mpl_backend)

    def test_expanding_increment_split_basic(self):
        periods1 = 6
        periods2 = 6
        X, y = self.make_simple_df(periods1=periods1, periods2=periods2)
        splitter = ExpandingIncrementPanelSplit(
            train_intervals=1, test_size=1, min_periods=1, min_cids=2, max_periods=None
        )
        for i, split in enumerate(splitter.split(X, y)):
            train_set, test_set = split

            # Training sets for each cid
            self.assertEqual(train_set[i], i)
            self.assertEqual(train_set[i * 2 + 1], i + periods1)

            # Test sets for each cid
            self.assertEqual(test_set[0], i + 1)
            self.assertEqual(test_set[1], i + periods1 + 1)

    def test_expanding_increment_split_min_cids(self):
        periods1 = 6
        periods2 = 5
        X, y = self.make_simple_df(
            start1="2020-01-01",
            start2="2020-01-02",
            periods1=periods1,
            periods2=periods2,
        )
        splitter = ExpandingIncrementPanelSplit(
            train_intervals=1, test_size=1, min_periods=1, min_cids=2, max_periods=None
        )
        for i, split in enumerate(splitter.split(X, y)):
            train_set, test_set = split

            # Training sets for each cid
            self.assertEqual(train_set[i], i)
            self.assertEqual(train_set[i * 2 + 2], i + periods1)
            if i == 0:
                self.assertEqual(train_set.shape[0], 3)
            # Test sets for each cid
            self.assertEqual(test_set[0], i + 2)
            self.assertEqual(test_set[1], i + periods1 + 1)

    def make_simple_df(
        self,
        start1="2020-01-01",
        start2="2020-01-01",
        periods1=10,
        periods2=10,
        freq1="D",
        freq2="D",
    ):
        dates_cid1 = pd.date_range(start=start1, periods=periods1, freq=freq1)
        dates_cid2 = pd.date_range(start=start2, periods=periods2, freq=freq2)

        # Create a MultiIndex for each cid with the respective dates
        multiindex_cid1 = pd.MultiIndex.from_product(
            [["cid1"], dates_cid1], names=["cid", "real_date"]
        )
        multiindex_cid2 = pd.MultiIndex.from_product(
            [["cid2"], dates_cid2], names=["cid", "real_date"]
        )

        # Concatenate the MultiIndexes
        multiindex = multiindex_cid1.append(multiindex_cid2)

        # Initialize a DataFrame with the MultiIndex and columns xcat1 and xcat2
        # and random data.
        df = pd.DataFrame(
            np.random.rand(len(multiindex), 2),
            index=multiindex,
            columns=["xcat1", "xcat2"],
        )

        X = df.drop(columns=["xcat2"])
        y = df["xcat2"]

        return X, y