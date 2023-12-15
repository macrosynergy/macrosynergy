import numpy as np
import pandas as pd

import unittest 
import itertools

from parameterized import parameterized

from macrosynergy.learning import (
    SignalOptimizer,
    ExpandingKFoldPanelSplit,
    RollingKFoldPanelSplit,
    ExpandingIncrementPanelSplit,
)

class TestAll(unittest.TestCase):
    def setUp(self):
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2018-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2018-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2018-06-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2018-06-01", "2020-12-31"]

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

        # instantiate some splitters
        expandingkfold = ExpandingKFoldPanelSplit(n_splits=5)
        rollingkfold = RollingKFoldPanelSplit(n_splits=5)
        expandingincrement = ExpandingIncrementPanelSplit(min_cids=2, min_periods = 100)
        self.splitters = [expandingkfold, rollingkfold, expandingincrement]

        # create blacklist dictionary 
        self.black_valid = {
            "AUD": (pd.Timestamp(year=2018, month=9, day=1), pd.Timestamp(year=2018, month=10, day=1)),
            "GBP": (pd.Timestamp(year=2019, month=6, day=1), pd.Timestamp(year=2100, month=1, day=1))
        }
        self.black_invalid = {
            "AUD": ["2018-09-01", "2018-10-01"],
            "GBP": ["2019-06-01", "2100-01-01"],
        }

    @parameterized.expand(itertools.product([0,1,2],[True, False]))
    def test_valid_init(self, idx, use_blacklist):
        inner_splitter = self.splitters[idx]
        try:
            blacklist = self.black_valid if use_blacklist else None
            so = SignalOptimizer(inner_splitter=inner_splitter,X=self.X,y=self.y, blacklist=blacklist)
        except Exception as e:
            self.fail(f"Instantiation of the SignalOptimizer raised an exception: {e}")
        self.assertIsInstance(so, SignalOptimizer)
        self.assertEqual(so.inner_splitter, inner_splitter)
        pd.testing.assert_frame_equal(so.X, self.X)
        pd.testing.assert_series_equal(so.y, self.y)
        self.assertEqual(so.blacklist, blacklist)