import unittest
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch

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
        # Tests valid initialization
        try:
            splitter = ExpandingKFoldPanelSplit(n_splits=5)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
        self.assertTrue(splitter.n_splits == 5)

        try:
            splitter = ExpandingKFoldPanelSplit(n_splits=2)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
        self.assertTrue(splitter.n_splits == 2)

        try:
            splitter = ExpandingKFoldPanelSplit(n_splits=10)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
        self.assertTrue(splitter.n_splits == 10)
        
        # Test too small n_splits raises an error
        with self.assertRaises(ValueError):
            ExpandingKFoldPanelSplit(n_splits=1)

    def test_types_split(self):
        splitter = ExpandingKFoldPanelSplit(n_splits=5)
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

    def test_valid_split(self):
        splitter = ExpandingKFoldPanelSplit(n_splits=5)
        splits = list(splitter.split(X=self.X, y=self.y))
        self.assertTrue(len(splits) == 5)

        # Test functionality on a simple DataFrame
        periods1 = 6
        periods2 = 6
        n_splits = 5
        X, y = make_simple_df(periods1=periods1, periods2=periods2)
        splitter = ExpandingKFoldPanelSplit(n_splits=n_splits)
        for i, split in enumerate(splitter.split(X, y)):
            train_set, test_set = split

            # Training sets for each cid
            self.assertEqual(train_set[i], i)
            self.assertEqual(train_set[i * 2 + 1], i + periods1)

            # Test sets for each cid
            self.assertEqual(test_set[0], i + 1)
            self.assertEqual(test_set[1], i + periods1 + 1)

    def test_types_visualise_splits(self):
        splitter = ExpandingKFoldPanelSplit(n_splits=5)
        with self.assertRaises(TypeError):
            splitter.visualise_splits(X="a", y=self.y)
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X.reset_index(), y=self.y)

        with self.assertRaises(TypeError):
            splitter.visualise_splits(X=self.X, y="a")
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X, y=self.y.reset_index(drop=True))

    def test_valid_visualise_splits(self):
        splitter = ExpandingKFoldPanelSplit(n_splits=5)
        try:
            splitter.visualise_splits(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

class TestRollingKFold(unittest.TestCase):
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
            RollingKFoldPanelSplit(n_splits="a")
        with self.assertRaises(TypeError):
            RollingKFoldPanelSplit(n_splits=1.5)
        with self.assertRaises(ValueError):
            RollingKFoldPanelSplit(n_splits=0)
        with self.assertRaises(ValueError):
            RollingKFoldPanelSplit(n_splits=1)
        with self.assertRaises(ValueError):
            RollingKFoldPanelSplit(n_splits=-2)

    def test_valid_init(self):
        # Tests valid initialization
        try:
            splitter = RollingKFoldPanelSplit(n_splits=5)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
        self.assertTrue(splitter.n_splits == 5)

        try:
            splitter = RollingKFoldPanelSplit(n_splits=2)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
        self.assertTrue(splitter.n_splits == 2)

        try:
            splitter = RollingKFoldPanelSplit(n_splits=10)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
        self.assertTrue(splitter.n_splits == 10)
        
        # Test too small n_splits raises an error
        with self.assertRaises(ValueError):
            RollingKFoldPanelSplit(n_splits=1)

    def test_types_split(self):
        splitter = RollingKFoldPanelSplit(n_splits=5)
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

    def test_valid_split(self):
        splitter = RollingKFoldPanelSplit(n_splits=5)
        splits = list(splitter.split(X=self.X, y=self.y))
        self.assertTrue(len(splits) == 5)

        # Test functionality on a simple DataFrame
        periods1 = 6
        periods2 = 6
        n_splits = 5
        X, y = make_simple_df(periods1=periods1, periods2=periods2)
        true_test_set_dates = np.array_split(y.index.get_level_values(1).unique().sort_values(), 5)
        splitter = RollingKFoldPanelSplit(n_splits=n_splits)
        for i, split in enumerate(splitter.split(X, y)):
            train_set, test_set = split
            empirical_test_set_dates = y.index.get_level_values(1)[test_set].unique().sort_values()
            self.assertTrue(np.all(empirical_test_set_dates == true_test_set_dates[i]))
            # Check the training dates aren't in the test set
            self.assertTrue(
                np.all(
                    np.logical_not(
                        np.isin(
                            y.index.get_level_values(1)[train_set].unique().sort_values(),
                            empirical_test_set_dates,
                        )
                    )
                )
            )

    def test_types_visualise_splits(self):
        splitter = RollingKFoldPanelSplit(n_splits=5)
        with self.assertRaises(TypeError):
            splitter.visualise_splits(X="a", y=self.y)
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X.reset_index(), y=self.y)

        with self.assertRaises(TypeError):
            splitter.visualise_splits(X=self.X, y="a")
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X, y=self.y.reset_index(drop=True))

    def test_valid_visualise_splits(self):
        splitter = RollingKFoldPanelSplit(n_splits=5)
        try:
            splitter.visualise_splits(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

class TestRecencyKFold(unittest.TestCase):
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
        # Test invalid types for n_splits
        with self.assertRaises(TypeError):
            RecencyKFoldPanelSplit(n_splits="a")
        with self.assertRaises(TypeError):
            RecencyKFoldPanelSplit(n_splits=1.5)
        with self.assertRaises(ValueError):
            RecencyKFoldPanelSplit(n_splits=0)
        with self.assertRaises(ValueError):
            RecencyKFoldPanelSplit(n_splits=-2)
        
        # Test invalid types for n_periods
        with self.assertRaises(TypeError):
            RecencyKFoldPanelSplit(n_periods="a")
        with self.assertRaises(TypeError):
            RecencyKFoldPanelSplit(n_periods=1.5)
        with self.assertRaises(ValueError):
            RecencyKFoldPanelSplit(n_periods=0)
        with self.assertRaises(ValueError):
            RecencyKFoldPanelSplit(n_periods=-2)

    def test_valid_init(self):
        # Tests valid initialization
        try:
            splitter = RecencyKFoldPanelSplit(n_splits=5, n_periods = 1)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
        self.assertTrue(splitter.n_splits == 5)

        try:
            splitter = RecencyKFoldPanelSplit(n_splits=2, n_periods = 2)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
        self.assertTrue(splitter.n_splits == 2)

        try:
            splitter = RecencyKFoldPanelSplit(n_splits=10, n_periods = 1)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
        self.assertTrue(splitter.n_splits == 10)

    def test_types_split(self):
        splitter = RecencyKFoldPanelSplit(n_splits=5, n_periods = 1)
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

    def test_valid_split(self):
        splitter = RecencyKFoldPanelSplit(n_splits=5, n_periods = 1)
        splits = list(splitter.split(X=self.X, y=self.y))
        self.assertTrue(len(splits) == 5)

        # Test functionality on a simple DataFrame
        periods1 = 6
        periods2 = 6
        n_splits = 3
        X, y = make_simple_df(periods1=periods1, periods2=periods2)
        true_test_set_dates = [
            [date]
            for date in y.index.get_level_values(1).unique().sort_values()[3:]
        ]
        splitter = RecencyKFoldPanelSplit(n_splits=n_splits, n_periods = 1)
        for i, split in enumerate(splitter.split(X, y)):
            train_set, test_set = split
            empirical_test_set_dates = y.index.get_level_values(1)[test_set].unique().sort_values()
            self.assertTrue(np.all(empirical_test_set_dates == true_test_set_dates[i]))
            # Check the training dates aren't in the test set
            self.assertTrue(
                np.all(
                    np.logical_not(
                        np.isin(
                            y.index.get_level_values(1)[train_set].unique().sort_values(),
                            empirical_test_set_dates,
                        )
                    )
                )
            )

    def test_types_visualise_splits(self):
        splitter = RecencyKFoldPanelSplit(n_splits=5, n_periods = 1)
        with self.assertRaises(TypeError):
            splitter.visualise_splits(X="a", y=self.y)
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X.reset_index(), y=self.y)

        with self.assertRaises(TypeError):
            splitter.visualise_splits(X=self.X, y="a")
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X, y=self.y.reset_index(drop=True))

    def test_valid_visualise_splits(self):
        splitter = RecencyKFoldPanelSplit(n_splits=5, n_periods = 1)
        try:
            splitter.visualise_splits(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")      

        splitter = RecencyKFoldPanelSplit(n_splits=1, n_periods = 3)
        try:
            splitter.visualise_splits(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

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
    
if __name__ == "__main__":
    Test = TestExpandingKFold()
    Test.setUpClass()
    Test.test_types_split()