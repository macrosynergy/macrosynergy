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
    ExpandingKFoldPanelSplit,
    RollingKFoldPanelSplit,
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

    # def test_crossval_application(self):
    #     # Given a generated panel with a true linear relationship between features and target,
    #     # test that the cross validation procedure correctly identifies that a linear regression
    #     # is more suitable than a 1-nearest neighbor model.
    #     # self.setUp()

    #     # models
    #     lr = LinearRegression()
    #     knn = KNeighborsRegressor(n_neighbors=1)
    #     splitter = ExpandingIncrementPanelSplit(
    #         train_intervals=1, min_cids=2, min_periods=21 * 12, test_size=1
    #     )
    #     lrsplits = cross_val_score(
    #         lr,
    #         self.X,
    #         self.y,
    #         scoring="neg_root_mean_squared_error",
    #         cv=splitter,
    #         n_jobs=-1,
    #     )
    #     knnsplits = cross_val_score(
    #         knn,
    #         self.X,
    #         self.y,
    #         scoring="neg_root_mean_squared_error",
    #         cv=splitter,
    #         n_jobs=-1,
    #     )

    #     self.assertLess(np.mean(-lrsplits), np.mean(-knnsplits))

    @parameterized.expand([2, 4, 8])
    def test_expanding_kfold(self, n_splits):
        splitter = ExpandingKFoldPanelSplit(n_splits=n_splits)
        splits = list(splitter.split(self.X, self.y))
        self.assertEqual(len(splits), n_splits)

    @parameterized.expand([2, 4, 8])
    def test_rolling_kfold(self, n_splits):
        splitter = RollingKFoldPanelSplit(n_splits=n_splits)
        splits = list(splitter.split(self.X, self.y))
        self.assertEqual(len(splits), n_splits)

    def test_expanding_kfold_too_small(self):
        with self.assertRaises(ValueError):
            ExpandingKFoldPanelSplit(n_splits=1)

    def test_rolling_k_fold_too_small(self):
        with self.assertRaises(ValueError):
            RollingKFoldPanelSplit(n_splits=1)

    def test_expanding_kfold_split(self):
        periods1 = 6
        periods2 = 6
        n_splits = 5
        X, y = self.make_simple_df(periods1=periods1, periods2=periods2)
        splitter = ExpandingKFoldPanelSplit(n_splits=n_splits)
        for i, split in enumerate(splitter.split(X, y)):
            train_set, test_set = split

            # Training sets for each cid
            self.assertEqual(train_set[i], i)
            self.assertEqual(train_set[i * 2 + 1], i + periods1)

            # Test sets for each cid
            self.assertEqual(test_set[0], i + 1)
            self.assertEqual(test_set[1], i + periods1 + 1)

    def test_kfold_split(self):
        periods1 = 5
        periods2 = 5
        n_splits = 5
        X, y = self.make_simple_df(periods1=periods1, periods2=periods2)
        indices = np.arange(X.shape[0])
        splitter = RollingKFoldPanelSplit(n_splits=n_splits)
        for i, split in enumerate(splitter.split(X, y)):
            train_set, test_set = split

            # Test sets for each cid
            self.assertEqual(test_set[0], i)
            self.assertEqual(test_set[1], i + periods1)

            # assert train set is indices excluding test set
            self.assertTrue(np.array_equal(np.setdiff1d(indices, test_set), train_set))

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

    def test_valid_visualise_splits(self):
        # ExpandingIncrementPanelSplit
        splitter1 = ExpandingIncrementPanelSplit(
            train_intervals=21 * 12,
            test_size=1,
        )
        try:
            splitter1.visualise_splits(self.X, self.y)
        except Exception as e:
            self.fail(f"ExpandingIncrementPanelSplit.visualise_splits() raised {e}")

        # RollingKFoldPanelSplit
        splitter2 = RollingKFoldPanelSplit(n_splits=5)
        try:
            splitter2.visualise_splits(self.X, self.y)
        except Exception as e:
            self.fail(f"RollingKFoldPanelSplit.visualise_splits() raised {e}")
        # ExpandingKFoldPanelSplit
        splitter3 = ExpandingKFoldPanelSplit(n_splits=5)
        try:
            splitter3.visualise_splits(self.X, self.y)
        except Exception as e:
            self.fail(f"ExpandingKFoldPanelSplit.visualise_splits() raised {e}")

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


if __name__ == "__main__":
    unittest.main()

    # test = TestAll()
    # X, y = test.make_simple_df(periods1=5, periods2=5)
    # splitter = RollingKFoldPanelSplit(n_splits=5)
    # splits = list(splitter.split(X, y))
    # print(splits)
    # splitter.visualise_splits(X, y)
