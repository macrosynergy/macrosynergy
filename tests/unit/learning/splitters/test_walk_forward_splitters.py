from typing import List
import unittest
import itertools
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
    ExpandingFrequencyPanelSplit,
)

class TestExpandingIncrement(unittest.TestCase):
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
        # train_intervals 
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(train_intervals="1Y")
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(train_intervals=1.0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(train_intervals=0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(train_intervals=-1)
        # test_size
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(test_size="1Y")
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(test_size=1.0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(test_size=0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(test_size=-1)
        # min_cids
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(min_cids="1Y")
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(min_cids=1.0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(min_cids=0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(min_cids=-1)
        # min_periods 
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(min_periods="1Y")
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(min_periods=1.0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(min_periods=0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(min_periods=-1)
        # start_date 
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(start_date=1)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(start_date="2020-01d-01")
        # max_periods 
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(max_periods="1Y")
        with self.assertRaises(TypeError):
            splitter = ExpandingIncrementPanelSplit(max_periods=1.0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(max_periods=0)
        with self.assertRaises(ValueError):
            splitter = ExpandingIncrementPanelSplit(max_periods=-1)

    def test_valid_init(self):
        # Ensure default values set correctly
        splitter = ExpandingIncrementPanelSplit()
        self.assertEqual(splitter.train_intervals, 21)
        self.assertEqual(splitter.test_size, 21)
        self.assertEqual(splitter.min_cids, 4)
        self.assertEqual(splitter.min_periods, 500)
        self.assertEqual(splitter.start_date, None)
        self.max_periods = None

        # Change default values
        splitter = ExpandingIncrementPanelSplit(
            train_intervals=3,
            test_size=3,
            min_cids=2,
            min_periods=24,
            start_date="2015-01-01",
            max_periods=12*3
        )
        self.assertEqual(splitter.train_intervals, 3)
        self.assertEqual(splitter.test_size, 3)
        self.assertEqual(splitter.min_cids, 2)
        self.assertEqual(splitter.min_periods, 24)
        self.assertEqual(splitter.start_date, pd.Timestamp("2015-01-01"))
        self.assertEqual(splitter.max_periods, 12*3)

    def test_types_split(self):
        splitter = ExpandingIncrementPanelSplit()
        # Test invalid types for X
        with self.assertRaises(TypeError):
            next(splitter.split(X="a", y=self.y))
        with self.assertRaises(ValueError):
            next(splitter.split(X=self.X.reset_index(), y=self.y))

        # Test invalid types for y
        with self.assertRaises(TypeError):
            next(splitter.split(X=self.X, y="a"))
        with self.assertRaises(ValueError):
            next(splitter.split(X=self.X, y=self.y.reset_index(drop=True)))

    @parameterized.expand(itertools.product([1, 2, 3], [1, 2]))
    def test_valid_split(self, window_size, min_cids):
        # Test functionality on simple dataframe
        periods1 = 12
        periods2 = 9
        X, y = make_simple_df(periods1=periods1, periods2=periods2)
        splitter = ExpandingIncrementPanelSplit(
            train_intervals = window_size,
            test_size = window_size,
            min_cids = min_cids,
            min_periods = 2,
        )
        splits = list(splitter.split(X, y))
        if min_cids == 1:
            # The first split should have 2 samples
            self.assertEqual(len(splits[0][0]), 4)
            # There should be nine splits in total
            self.assertEqual(len(splits), 9)
        elif min_cids == 2:
            # The first split should have 7 samples
            self.assertEqual(len(splits[0][0]), 6)
            # There should be five splits in total
            self.assertEqual(len(splits), 5)
        

    def test_types_visualise_splits(self):
        pass 

    def test_types_visualise_splits(self):
        pass

class TestExpandingFrequency(unittest.TestCase):
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
        pass 

    def test_valid_init(self):
        pass 

    def test_types_split(self):
        pass

    def test_valid_split(self):
        pass

    def test_types_visualise_splits(self):
        pass 

    def test_types_visualise_splits(self):
        pass

def make_simple_df(
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