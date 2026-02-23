import unittest
import itertools
import pandas as pd
import numpy as np
import pytest
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
        xcats = ["RIR", "CPI", "GROWTH", "XR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2019-04-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2019-04-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-04-01", "2020-12-31"]

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

        # Create sample X and y dataframes resampled at monthly frequency
        self.X = df.drop(columns="XR")
        self.X = self.X.groupby(level=0).resample("M", level="real_date").mean()

        self.y = df["XR"]
        self.y = self.y.groupby(level=0).resample("M", level="real_date").last()

        # Create X and y dataframes with NaNs
        self.X_nan, self.y_nan = self.X.copy(), self.y.copy()

        num_nans_per_cid = {"AUD": 7, "CAD": 5, "GBP": 2, "USD": 1}
        nan_xcats = ["RIR", "GROWTH"]
        for cid, num_nans in num_nans_per_cid.items():
            idx = self.X_nan.loc[[cid], :,].head(num_nans).index
            self.X_nan.loc[idx, nan_xcats] = np.nan

        self.y_nan.loc[np.random.rand(len(self.y_nan)) < 0.15] = np.nan

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
        self.assertEqual(splitter.max_periods, None)

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

    @parameterized.expand(itertools.product([1, 2, 3], [1, 2], [1, 2, 3]))
    def test_valid_split_no_nans(self, window_size, min_cids, min_xcats):
        """
        Test splits when no NaNs in the data. min_xcats should not have any
        impact in this case
        """
        splitter = ExpandingIncrementPanelSplit(
            train_intervals = window_size,
            test_size = window_size,
            min_cids = min_cids,
            min_periods = 2,
            min_xcats=min_xcats, # should not have an impact because no nans in the data
        )
        splits = list(splitter.split(self.X, self.y))
        
        # Check first split is correct
        X_train_split1 = self.X.iloc[splits[0][0], :]
        X_test_split1 = self.X.iloc[splits[0][1], :]
        n_samples = len(X_train_split1)
        n_unique_dates = len(X_train_split1.index.get_level_values(1).unique())
        n_unique_test_dates = len(X_test_split1.index.get_level_values(1).unique())
        if min_cids == 1:
            # unique training dates should be two and number of samples should be two
            self.assertEqual(n_samples, 2)
            self.assertEqual(n_unique_dates, 2)
            self.assertEqual(n_unique_test_dates, window_size)
        else:
            # min_cids = 2
            # The first split should have (3 + 2) + 3*2 = 11 samples
            # Unique training dates should be 5
            self.assertEqual(n_samples, 11)
            self.assertEqual(n_unique_dates, 5)
            self.assertEqual(n_unique_test_dates, window_size)

        # Track the number of unique dates in each set
        current_n_unique_dates = n_unique_dates
        for split_idx in range(1, len(splits)):
            X_train_split = self.X.iloc[splits[split_idx][0], :]
            X_test_split = self.X.iloc[splits[split_idx][1], :]
            n_samples = len(X_train_split)
            n_unique_dates = len(X_train_split.index.get_level_values(1).unique())
            n_unique_test_dates = len(X_test_split.index.get_level_values(1).unique())
            self.assertEqual(n_unique_dates, current_n_unique_dates + window_size)
            current_n_unique_dates = n_unique_dates
            if split_idx != len(splits) - 1:
                self.assertEqual(n_unique_test_dates, window_size)
            else:
                self.assertLessEqual(n_unique_test_dates, window_size)
                self.assertGreater(n_unique_test_dates, 0)

    @parameterized.expand(itertools.product([1, 2, 3], [1, 2, 3, 4]))
    def test_valid_split_with_nans_and_min_xcat_is_one(self, window_size, min_cids):
        """
        This tests the scenario when min_xcats=1 and NaNs exist.

        CAD has 1 feature available from 2019-04-30
        AUD has 1 feature available from 2019-01-31
        GBP has 1 feature available from 2019-06-30
        USD has 1 feature available from 2019-04-30

        so,

        On 2019-01-31 we have 1 cid with at least 1 feature available
        On 2019-04-30 we have 4 cid with at least 1 feature available

        so,

        If min_cids=1, the first train set should contain data for [min_date, 2019-01-31 + (min_periods - 1)]
        If min_cids=2, the first train set should contain data for [min_date, 2019-04-30 + (min_periods - 1)]
        If min_cids=3, the first train set should contain data for [min_date, 2019-04-30 + (min_periods - 1)]
        If min_cids=4, the first train set should contain data for [min_date, 2019-04-30 + (min_periods - 1)]
        """
        splitter = ExpandingIncrementPanelSplit(
            train_intervals=window_size,
            test_size=window_size,
            min_cids=min_cids,
            min_periods=2,
            min_xcats=1,
        )

        splits = list(splitter.split(self.X_nan, self.y_nan))

        X_train_split1 = self.X_nan.iloc[splits[0][0], :]
        X_test_split1 = self.X_nan.iloc[splits[0][1], :]

        n_train_samples = len(X_train_split1)
        n_unique_train_dates = len(X_train_split1.index.get_level_values(1).unique())
        n_unique_test_dates = len(X_test_split1.index.get_level_values(1).unique())

        first_train_date = X_train_split1.index.get_level_values(1).min()
        first_test_date = X_test_split1.index.get_level_values(1).min()
        last_train_date = X_train_split1.index.get_level_values(1).max()

        assert first_train_date == pd.Timestamp("2019-01-31")
        assert n_unique_test_dates == window_size

        if min_cids==1:
            assert n_train_samples == 2
            assert n_unique_train_dates == 2
            assert last_train_date == pd.Timestamp("2019-02-28")
            assert first_test_date == pd.Timestamp("2019-03-31")
        elif min_cids==2:
            assert n_train_samples == 11
            assert n_unique_train_dates == 5
            assert last_train_date == pd.Timestamp("2019-05-31")
            assert first_test_date == pd.Timestamp("2019-06-30")
        elif min_cids==3:
            assert n_train_samples == 11
            assert n_unique_train_dates == 5
            assert last_train_date == pd.Timestamp("2019-05-31")
            assert first_test_date == pd.Timestamp("2019-06-30")
        elif min_cids==4:
            assert n_train_samples == 11
            assert n_unique_train_dates == 5
            assert last_train_date == pd.Timestamp("2019-05-31")
            assert first_test_date == pd.Timestamp("2019-06-30")
        else:
            pass

        # Track the number of unique dates in each set
        current_n_unique_dates = n_unique_train_dates
        for split_idx in range(1, len(splits)):
            X_train_split = self.X_nan.iloc[splits[split_idx][0], :]
            X_test_split = self.X_nan.iloc[splits[split_idx][1], :]
            n_unique_dates = len(X_train_split.index.get_level_values(1).unique())
            n_unique_test_dates = len(X_test_split.index.get_level_values(1).unique())
            self.assertEqual(n_unique_dates, current_n_unique_dates + window_size)
            current_n_unique_dates = n_unique_dates
            if split_idx != len(splits) - 1:
                self.assertEqual(n_unique_test_dates, window_size)
            else:
                self.assertLessEqual(n_unique_test_dates, window_size)
                self.assertGreater(n_unique_test_dates, 0)

    @parameterized.expand(itertools.product([1, 2, 3], [1, 2, 3, 4]))
    def test_valid_split_with_nans_and_largest_min_xcat(self, window_size, min_cids):
        """
        This tests the scenario when NaNs exist and min_xcats = number of features.

        CAD has all features available from 2019-09-30
        AUD has all features available from 2019-08-31
        GBP has all features available from 2019-06-30
        USD has all features available from 2019-05-31

        so,

        On 2019-05-31 we have 1 cid with all features available
        On 2019-06-30 we have 2 cids with all features available
        On 2019-08-31 we have 3 cids with all features available
        On 2019-09-30 we have all 4 cids with all features available

        so,

        If min_cids=1, the first train set should contain data for [min_date, 2019-05-31 + (min_periods - 1)]
        If min_cids=2, the first train set should contain data for [min_date, 2019-06-30 + (min_periods - 1)]
        If min_cids=3, the first train set should contain data for [min_date, 2019-08-31 + (min_periods - 1)]
        If min_cids=4, the first train set should contain data for [min_date, 2019-09-30 + (min_periods - 1)]
        """
        splitter = ExpandingIncrementPanelSplit(
            train_intervals=window_size,
            test_size=window_size,
            min_cids=min_cids,
            min_periods=2,
            min_xcats=len(self.X_nan.columns),
        )

        splits = list(splitter.split(self.X_nan, self.y_nan))

        X_train_split1 = self.X_nan.iloc[splits[0][0], :]
        X_test_split1 = self.X_nan.iloc[splits[0][1], :]

        n_train_samples = len(X_train_split1)
        n_unique_train_dates = len(X_train_split1.index.get_level_values(1).unique())
        n_unique_test_dates = len(X_test_split1.index.get_level_values(1).unique())

        first_train_date = X_train_split1.index.get_level_values(1).min()
        first_test_date = X_test_split1.index.get_level_values(1).min()
        last_train_date = X_train_split1.index.get_level_values(1).max()

        assert first_train_date == pd.Timestamp("2019-01-31")
        assert n_unique_test_dates == window_size

        if min_cids==1:
            assert n_train_samples == 15
            assert n_unique_train_dates == 6
            assert last_train_date == pd.Timestamp("2019-06-30")
            assert first_test_date == pd.Timestamp("2019-07-31")
        elif min_cids==2:
            assert n_train_samples == 19
            assert n_unique_train_dates == 7
            assert last_train_date == pd.Timestamp("2019-07-31")
            assert first_test_date == pd.Timestamp("2019-08-31")
        elif min_cids==3:
            assert n_train_samples == 27
            assert n_unique_train_dates == 9
            assert last_train_date == pd.Timestamp("2019-09-30")
            assert first_test_date == pd.Timestamp("2019-10-31")
        elif min_cids==4:
            assert n_train_samples == 31
            assert n_unique_train_dates == 10
            assert last_train_date == pd.Timestamp("2019-10-31")
            assert first_test_date == pd.Timestamp("2019-11-30")
        else:
            pass

        # Track the number of unique dates in each set
        current_n_unique_dates = n_unique_train_dates
        for split_idx in range(1, len(splits)):
            X_train_split = self.X_nan.iloc[splits[split_idx][0], :]
            X_test_split = self.X_nan.iloc[splits[split_idx][1], :]
            n_unique_dates = len(X_train_split.index.get_level_values(1).unique())
            n_unique_test_dates = len(X_test_split.index.get_level_values(1).unique())
            self.assertEqual(n_unique_dates, current_n_unique_dates + window_size)
            current_n_unique_dates = n_unique_dates
            if split_idx != len(splits) - 1:
                self.assertEqual(n_unique_test_dates, window_size)
            else:
                self.assertLessEqual(n_unique_test_dates, window_size)
                self.assertGreater(n_unique_test_dates, 0)

    def test_too_large_min_xcats(self):
        """This tests when min_xcats > len(features)"""
        splitter = ExpandingIncrementPanelSplit(
            train_intervals=1,
            test_size=1,
            min_cids=2,
            min_periods=2,
            min_xcats=1000,
        )

        with pytest.raises(ValueError):
            list(splitter.split(self.X_nan, self.y_nan))

    @parameterized.expand(itertools.product([1, 2, 3], [1, 2]))
    def test_types_visualise_splits(self, window_size, min_cids):
        splitter = ExpandingIncrementPanelSplit(
            train_intervals = window_size,
            test_size = window_size,
            min_cids = min_cids,
            min_periods = 2,
        )
        with self.assertRaises(TypeError):
            splitter.visualise_splits(X="a", y=self.y)
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X.reset_index(), y=self.y)

        with self.assertRaises(TypeError):
            splitter.visualise_splits(X=self.X, y="a")
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X, y=self.y.reset_index(drop=True))

    @parameterized.expand(itertools.product([1, 2, 3], [1, 2]))
    def test_valid_visualise_splits(self, window_size, min_cids):
        splitter = ExpandingIncrementPanelSplit(
            train_intervals = window_size,
            test_size = window_size,
            min_cids = min_cids,
            min_periods = 2,
        )
        try:
            splitter.visualise_splits(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

class TestExpandingFrequency(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["RIR", "CPI", "GROWTH", "XR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2019-04-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2019-04-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-04-01", "2020-12-31"]

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

        # Create sample X and y dataframes resampled at monthly frequency
        self.X = df.drop(columns="XR")
        self.X = self.X.groupby(level=0).resample("M", level="real_date").mean()

        self.y = df["XR"]
        self.y = self.y.groupby(level=0).resample("M", level="real_date").last()

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(self.mpl_backend)

    def test_types_init(self):
        # expansion_freq
        with self.assertRaises(TypeError):
            splitter = ExpandingFrequencyPanelSplit(expansion_freq=1)
        with self.assertRaises(ValueError):
            splitter = ExpandingFrequencyPanelSplit(expansion_freq="1")
        # test_freq
        with self.assertRaises(TypeError):
            splitter = ExpandingFrequencyPanelSplit(test_freq=1)
        with self.assertRaises(ValueError):
            splitter = ExpandingFrequencyPanelSplit(test_freq="1")
        # min_cids
        with self.assertRaises(TypeError):
            splitter = ExpandingFrequencyPanelSplit(min_cids="1Y")
        with self.assertRaises(TypeError):
            splitter = ExpandingFrequencyPanelSplit(min_cids=1.0)
        with self.assertRaises(ValueError):
            splitter = ExpandingFrequencyPanelSplit(min_cids=0)
        with self.assertRaises(ValueError):
            splitter = ExpandingFrequencyPanelSplit(min_cids=-1)
        # min_periods 
        with self.assertRaises(TypeError):
            splitter = ExpandingFrequencyPanelSplit(min_periods="1Y")
        with self.assertRaises(TypeError):
            splitter = ExpandingFrequencyPanelSplit(min_periods=1.0)
        with self.assertRaises(ValueError):
            splitter = ExpandingFrequencyPanelSplit(min_periods=0)
        with self.assertRaises(ValueError):
            splitter = ExpandingFrequencyPanelSplit(min_periods=-1)
        # start_date 
        with self.assertRaises(TypeError):
            splitter = ExpandingFrequencyPanelSplit(start_date=1)
        with self.assertRaises(ValueError):
            splitter = ExpandingFrequencyPanelSplit(start_date="2020-01d-01")
        # max_periods 
        with self.assertRaises(TypeError):
            splitter = ExpandingFrequencyPanelSplit(max_periods="1Y")
        with self.assertRaises(TypeError):
            splitter = ExpandingFrequencyPanelSplit(max_periods=1.0)
        with self.assertRaises(ValueError):
            splitter = ExpandingFrequencyPanelSplit(max_periods=0)
        with self.assertRaises(ValueError):
            splitter = ExpandingFrequencyPanelSplit(max_periods=-1)

    def test_valid_init(self):
        # Ensure default values set correctly
        splitter = ExpandingFrequencyPanelSplit()
        self.assertEqual(splitter.expansion_freq, "D")
        self.assertEqual(splitter.test_freq, "D")
        self.assertEqual(splitter.min_cids, 4)
        self.assertEqual(splitter.min_periods, 500)
        self.assertEqual(splitter.start_date, None)
        self.assertEqual(splitter.max_periods, None)

        # Change default values
        splitter = ExpandingFrequencyPanelSplit(
            expansion_freq="M",
            test_freq="Q",
            min_cids=2,
            min_periods=24,
            start_date="2015-01-01",
            max_periods=12*3
        )
        self.assertEqual(splitter.expansion_freq, "M")
        self.assertEqual(splitter.test_freq, "Q")
        self.assertEqual(splitter.min_cids, 2)
        self.assertEqual(splitter.min_periods, 24)
        self.assertEqual(splitter.start_date, pd.Timestamp("2015-01-01"))
        self.assertEqual(splitter.max_periods, 12*3)

    def test_types_split(self):
        splitter = ExpandingFrequencyPanelSplit()
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

    @parameterized.expand(itertools.product(["W", "M", "Q"], [1, 2]))
    def test_valid_split(self, expansion_freq, min_cids):
        # Test functionality on simple dataframe
        splitter = ExpandingFrequencyPanelSplit(
            expansion_freq = expansion_freq,
            test_freq = expansion_freq,
            min_cids = min_cids,
            min_periods = 2,
        )
        splits = list(splitter.split(self.X, self.y))
        
        # Check first split is correct
        X_train_split1 = self.X.iloc[splits[0][0], :]
        X_test_split1 = self.X.iloc[splits[0][1], :]
        n_samples = len(X_train_split1)
        n_unique_dates = len(X_train_split1.index.get_level_values(1).unique())
        n_unique_test_dates = len(X_test_split1.index.get_level_values(1).unique())
        if min_cids == 1:
            # unique training dates should be two and number of samples should be two
            self.assertEqual(n_samples, 2)
            self.assertEqual(n_unique_dates, 2)
            if expansion_freq == "W" or expansion_freq == "M":
                self.assertEqual(n_unique_test_dates, 1)
            elif expansion_freq == "Q":
                self.assertEqual(n_unique_test_dates, 3)
        else:
            # min_cids = 2
            # The first split should have (3 + 2) + 3*2 = 11 samples
            # Unique training dates should be 5
            self.assertEqual(n_samples, 11)
            self.assertEqual(n_unique_dates, 5)
            if expansion_freq == "W" or expansion_freq == "M":
                self.assertEqual(n_unique_test_dates, 1)
            elif expansion_freq == "Q":
                self.assertEqual(n_unique_test_dates, 3)

        # Track the number of unique dates in each set
        current_n_unique_dates = n_unique_dates
        for split_idx in range(1, len(splits)):
            X_train_split = self.X.iloc[splits[split_idx][0], :]
            X_test_split = self.X.iloc[splits[split_idx][1], :]
            n_samples = len(X_train_split)
            n_unique_dates = len(X_train_split.index.get_level_values(1).unique())
            n_unique_test_dates = len(X_test_split.index.get_level_values(1).unique())
            if expansion_freq == "W" or expansion_freq == "M":
                self.assertEqual(n_unique_dates, current_n_unique_dates + 1)
            elif expansion_freq == "Q":
                self.assertEqual(n_unique_dates, current_n_unique_dates + 3)

            current_n_unique_dates = n_unique_dates
            if split_idx != len(splits) - 1:
                if expansion_freq == "W" or expansion_freq == "M":
                    self.assertEqual(n_unique_test_dates, 1)
                elif expansion_freq == "Q":
                    self.assertEqual(n_unique_test_dates, 3)
            else:
                if expansion_freq == "W" or expansion_freq == "M":
                    self.assertEqual(n_unique_test_dates, 1)
                elif expansion_freq == "Q":
                    self.assertGreaterEqual(n_unique_test_dates, 3)
        

    def test_types_visualise_splits(self):
        splitter = ExpandingFrequencyPanelSplit()
        with self.assertRaises(TypeError):
            splitter.visualise_splits(X="a", y=self.y)
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X.reset_index(), y=self.y)

        with self.assertRaises(TypeError):
            splitter.visualise_splits(X=self.X, y="a")
        with self.assertRaises(ValueError):
            splitter.visualise_splits(X=self.X, y=self.y.reset_index(drop=True))

    @parameterized.expand(itertools.product(["W", "M", "Q"], [1, 2]))
    def test_valid_visualise_splits(self, expansion_freq, min_cids):
        splitter = ExpandingFrequencyPanelSplit(
            expansion_freq = expansion_freq,
            test_freq = expansion_freq,
            min_cids = min_cids,
            min_periods = 2,
        )
        try:
            splitter.visualise_splits(X=self.X, y=self.y)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")    