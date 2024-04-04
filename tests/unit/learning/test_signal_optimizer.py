import numpy as np
import pandas as pd
import scipy.stats as stats

import unittest
import itertools
import datetime
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch

from parameterized import parameterized

from macrosynergy.learning import (
    SignalOptimizer,
    ExpandingKFoldPanelSplit,
    RollingKFoldPanelSplit,
    ExpandingIncrementPanelSplit,
    LassoSelector,
)

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        self.mock_show = patch("matplotlib.pyplot.show").start()

        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-06-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2020-06-01", "2020-12-31"]

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
        train = df[
            df.index.get_level_values(1) <= pd.Timestamp(day=1, month=10, year=2020)
        ]
        test = df[
            df.index.get_level_values(1) > pd.Timestamp(day=1, month=10, year=2020)
        ]
        self.X_train = train.drop(columns="XR")
        self.y_train = train["XR"]
        self.X_test = test.drop(columns="XR")
        self.y_test = test["XR"]

        # instantiate some splitters
        expandingkfold = ExpandingKFoldPanelSplit(n_splits=4)
        rollingkfold = RollingKFoldPanelSplit(n_splits=4)
        expandingincrement = ExpandingIncrementPanelSplit(min_cids=2, min_periods=100)
        self.splitters = [expandingkfold, rollingkfold, expandingincrement]

        # create blacklist dictionary
        self.black_valid = {
            "AUD": (
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2020, month=4, day=1),
            ),
            "GBP": (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2020, month=2, day=1),
            ),
        }
        self.black_invalid1 = {
            "AUD": ["2018-09-01", "2018-10-01"],
            "GBP": ["2019-06-01", "2100-01-01"],
        }
        self.black_invalid2 = {
            "AUD": ("2018-09-01", "2018-10-01"),
            "GBP": ("2019-06-01", "2100-01-01"),
        }
        self.black_invalid3 = {
            "AUD": [
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2018, month=10, day=1),
            ],
            "GBP": [
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2100, month=1, day=1),
            ],
        }
        self.black_invalid4 = {
            "AUD": (pd.Timestamp(year=2018, month=9, day=1),),
            "GBP": (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2100, month=1, day=1),
            ),
        }
        self.black_invalid5 = {
            1: (
                pd.Timestamp(year=2018, month=9, day=1),
                pd.Timestamp(year=2018, month=10, day=1),
            ),
            2: (
                pd.Timestamp(year=2019, month=6, day=1),
                pd.Timestamp(year=2100, month=1, day=1),
            ),
        }

        # models dictionary
        self.models = {
            "linreg": LinearRegression(),
            "ridge": Ridge(),
        }
        self.metric = make_scorer(mean_squared_error, greater_is_better=False)
        self.hparam_grid = {
            "linreg": {},
            "ridge": {"alpha": [0.1, 1.0]},
        }
        
        so = SignalOptimizer(
            inner_splitter=self.splitters[1],
            X=self.X_train,
            y=self.y_train,
        )
        # Now run calculate_predictions
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
            n_jobs=1,
        )
        self.so_with_calculated_preds = so

    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(self.mpl_backend)

    @parameterized.expand(itertools.product([0, 1], [True, False], [True, False]))
    def test_valid_init(self, idx, use_blacklist, change_n_splits):
        # Test standard instantiation of the SignalOptimizer works as expected
        inner_splitter = self.splitters[idx]
        try:
            blacklist = self.black_valid if use_blacklist else None
            if change_n_splits:
                # generate sample initial_nsplits and threshold_ndates
                initial_nsplits = np.random.choice([2, 5, 10])
                threshold_ndates = np.random.choice([1, 6, 12, 24])
                so = SignalOptimizer(
                    inner_splitter=inner_splitter,
                    X=self.X_train,
                    y=self.y_train,
                    blacklist=blacklist,
                    initial_nsplits=initial_nsplits,
                    threshold_ndates=threshold_ndates,
                )
            else:
                so = SignalOptimizer(
                    inner_splitter=inner_splitter,
                    X=self.X_train,
                    y=self.y_train,
                    blacklist=blacklist,
                )
        except Exception as e:
            self.fail(f"Instantiation of the SignalOptimizer raised an exception: {e}")
        self.assertIsInstance(so, SignalOptimizer)
        self.assertEqual(so.inner_splitter, inner_splitter)
        pd.testing.assert_frame_equal(so.X, self.X_train)
        pd.testing.assert_series_equal(so.y, self.y_train)
        self.assertEqual(so.blacklist, blacklist)
        if change_n_splits:
            self.assertEqual(so.initial_nsplits, initial_nsplits)
            self.assertEqual(so.threshold_ndates, threshold_ndates)
            # Run the signal optimizer again to check a warning is raised
            with self.assertWarns(RuntimeWarning):
                so = SignalOptimizer(
                    inner_splitter=inner_splitter,
                    X=self.X_train,
                    y=self.y_train,
                    blacklist=blacklist,
                    initial_nsplits=initial_nsplits,
                    threshold_ndates=threshold_ndates,
                )
                self.assertEqual(so.inner_splitter.n_splits, initial_nsplits)

        else:
            self.assertEqual(so.threshold_ndates, None)
            self.assertEqual(so.initial_nsplits, None)

        pd.testing.assert_frame_equal(
            so.preds, pd.DataFrame(columns=["cid", "real_date", "xcat", "value"])
        )
        pd.testing.assert_frame_equal(
            so.chosen_models,
            pd.DataFrame(
                columns=["real_date", "name", "model_type", "hparams", "n_splits_used"]
            ),
        )
        pd.testing.assert_frame_equal(
            so.ftr_coefficients,
            pd.DataFrame(columns=["real_date", "name"] + list(self.X_train.columns)),
        )
        pd.testing.assert_frame_equal(
            so.intercepts, pd.DataFrame(columns=["real_date", "name", "intercepts"])
        )

    def test_types_init(self):
        inner_splitter = KFold(n_splits=5, shuffle=False)
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=inner_splitter, X=self.X_train, y=self.y_train
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1], X="self.X", y=self.y_train
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1], X=self.X_train, y="self.y"
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train.reset_index(),
                y=self.y_train,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train.reset_index(),
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train.iloc[1:],
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                blacklist="blacklist",
            )
        # check that an incorrect blacklisting format is caught
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                blacklist=self.black_invalid1,
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                blacklist=self.black_invalid2,
            )
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                blacklist=self.black_invalid3,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                blacklist=self.black_invalid4,
            )

        # Check that wrong adaptive cv splits formats raise an error
        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits="initial_nsplits",
                threshold_ndates="threshold_ndates",
            )

        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits=5,
                threshold_ndates="threshold_ndates",
            )

        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits="initial_nsplits",
                threshold_ndates=6,
            )

        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[0],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits="initial_nsplits",
                threshold_ndates="threshold_ndates",
            )

        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[0],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits=5,
                threshold_ndates="threshold_ndates",
            )

        with self.assertRaises(TypeError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[0],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits="initial_nsplits",
                threshold_ndates=6,
            )

        # Check that initial_nsplits and threshold_ndates require each other
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits=10,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[1],
                X=self.X_train,
                y=self.y_train,
                threshold_ndates=6,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[0],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits=10,
            )
        with self.assertRaises(ValueError):
            so = SignalOptimizer(
                inner_splitter=self.splitters[0],
                X=self.X_train,
                y=self.y_train,
                threshold_ndates=6,
            )

    def test_valid_calculate_predictions(self):
        so1 = SignalOptimizer(
            inner_splitter=self.splitters[1], X=self.X_train, y=self.y_train
        )
        # check a correct optimisation runs
        try:
            so1.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=1,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df1 = so1.preds.copy()
        self.assertIsInstance(df1, pd.DataFrame)
        if len(df1.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df1.xcat.unique()[0], "test")
        # Test that blacklisting works as expected
        so2 = SignalOptimizer(
            inner_splitter=self.splitters[1],
            X=self.X_train,
            y=self.y_train,
            blacklist=self.black_valid,
        )
        try:
            so2.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="random",
                n_jobs=1,
                n_iter=1,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df2 = so2.preds.copy()
        self.assertIsInstance(df2, pd.DataFrame)
        for cross_section, periods in self.black_valid.items():
            cross_section_key = cross_section.split("_")[0]
            self.assertTrue(
                len(
                    df2[
                        (df2.cid == cross_section_key)
                        & (df2.real_date >= periods[0])
                        & (df2.real_date <= periods[1])
                    ].dropna()
                )
                == 0
            )
        # Test that rolling models work as expected
        so3 = SignalOptimizer(
            inner_splitter=self.splitters[1], X=self.X_train, y=self.y_train
        )
        try:
            so3.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                max_periods=21,  # monthly roll
                n_jobs=1,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df3 = so3.preds.copy()
        self.assertIsInstance(df3, pd.DataFrame)
        if len(df3.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df3.xcat.unique()[0], "test")
        # Test that an unreasonably large roll is equivalent to no roll
        so4 = SignalOptimizer(
            inner_splitter=self.splitters[1], X=self.X_train, y=self.y_train
        )
        try:
            so4.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                max_periods=int(1e6),  # million days roll
                n_jobs=1,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df4 = so4.preds.copy()
        self.assertIsInstance(df4, pd.DataFrame)
        if len(df4.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df4.xcat.unique()[0], "test")
        self.assertTrue(df1.equals(df4))
        self.assertFalse(df3.equals(df4))
        # Test that the signal optimiser works with dataframe targets as well as series targets
        so5 = SignalOptimizer(
            inner_splitter=self.splitters[1], X=self.X_train, y=self.y_train.to_frame()
        )
        # check a correct optimisation runs
        try:
            so5.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=1,
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df5 = so5.preds.copy()
        self.assertIsInstance(df5, pd.DataFrame)
        pd.testing.assert_frame_equal(df1, df5)

    @parameterized.expand(itertools.product([2, 5], [21, 21 * 3], [0, 1]))
    def test_valid_change_n_splits(
        self, initial_nsplits, threshold_ndates, splitter_idx
    ):
        # Test that the signals generated by the largest n_splits are correct
        # For each of the suggested initial number of splits, look at increasing
        # the number of splits by one after a) a month and b) six months
        if splitter_idx == 0:
            splitter = ExpandingKFoldPanelSplit(n_splits=5)
        else:
            splitter = RollingKFoldPanelSplit(n_splits=5)

        so = SignalOptimizer(
            inner_splitter=splitter,
            X=self.X_train,
            y=self.y_train,
            initial_nsplits=initial_nsplits,
            threshold_ndates=threshold_ndates,
        )
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
            n_jobs=1,
        )
        # Get the dates of the first and last times the model with largest n_splits was used.
        # Then filter the signals to only include those dates
        mods_df = so.get_optimal_models("test")
        filtered_mods_df = mods_df[mods_df.n_splits_used == mods_df.n_splits_used.max()]
        sigs_df = so.get_optimized_signals("test")

        earliest_date = filtered_mods_df.real_date.min()
        latest_date = filtered_mods_df.real_date.max()

        filtered_sigs_df = sigs_df[
            (sigs_df.real_date >= earliest_date) & (sigs_df.real_date <= latest_date)
        ]

        # Check that this is equivalent to the signals generated by the signal optimiser
        # with a fixed a n_splits equal to the largest n_splits used in the previous signal optimiser

        so2 = SignalOptimizer(
            inner_splitter=(
                ExpandingKFoldPanelSplit(n_splits=int(mods_df.n_splits_used.max()))
                if splitter_idx == 0
                else RollingKFoldPanelSplit(n_splits=int(mods_df.n_splits_used.max()))
            ),
            X=self.X_train,
            y=self.y_train,
        )
        so2.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
            n_jobs=1,
        )

        sigs_df2 = so2.get_optimized_signals("test")
        filtered_sigs_df2 = sigs_df2[
            (sigs_df2.real_date >= earliest_date) & (sigs_df2.real_date <= latest_date)
        ]

        pd.testing.assert_frame_equal(filtered_sigs_df, filtered_sigs_df2)

        # Now check that the largest number of splits is what's expected
        largest_nsplits = mods_df.n_splits_used.max()
        num_new_dates = len(
            pd.bdate_range(mods_df.real_date.min(), mods_df.real_date.max())
        )
        self.assertEqual(
            largest_nsplits, initial_nsplits + (num_new_dates // threshold_ndates)
        )

    def test_types_calculate_predictions(self):
        # Training set only
        so = SignalOptimizer(
            inner_splitter=self.splitters[1], X=self.X_train, y=self.y_train
        )
        # name
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name=1,
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=1,
            )
        # models
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models={},
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=1,
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=[LinearRegression()],
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=1,
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models={1: LinearRegression()},
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=1,
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models={"ols": 1},
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=1,
            )
        # metric
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric="metric",
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=1,
            )
        # hparam_grid - grid search
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=[1, 2],
                hparam_type="grid",
                n_jobs=1,
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"ols": [1, 2]},
                hparam_type="grid",
                n_jobs=1,
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={1: [1, 2]},
                hparam_type="grid",
                n_jobs=1,
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"ols": {"alpha": 1}},
                hparam_type="grid",
                n_jobs=1,
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"ols": {1: 1}, "ridge": {"alpha": [1, 2, 3]}},
                hparam_type="grid",
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"ols": {}, "ridge": {"alpha": []}},
                hparam_type="grid",
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"linreg": {}},
                hparam_type="grid",
                n_jobs=1,
            )
        # hparam_grid - random search
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=[1, 2],
                hparam_type="random",
                n_jobs=1,
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"ols": [1, 2]},
                hparam_type="random",
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"ols": {}, "ridge": {"alpha": []}},
                hparam_type="random",
                n_jobs=1,
                n_iter=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"ols": {"alpha": 1}},
                hparam_type="random",
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"ridge": {"alpha": stats.expon()}},
                hparam_type="random",
                n_jobs=1,
            )
        # hparam_type
        with self.assertRaises(NotImplementedError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="bayes",
                n_jobs=1,
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type=1,
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="somethingelse",
                n_jobs=1,
            )
        # min_cids
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                min_cids="min_cids",
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                min_cids=-1,
                n_jobs=1,
            )
        # min_periods
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                min_periods="min_periods",
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                min_periods=-1,
                n_jobs=1,
            )
        # max_periods
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                max_periods="max_periods",
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                max_periods=0,
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                max_periods=-1,
                n_jobs=1,
            )
        # n_iter
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="random",
                n_iter="n_iter",
                n_jobs=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="random",
                n_iter=-1,
                n_jobs=1,
            )
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="random",
                n_iter=0,
                n_jobs=1,
            )
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid={"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="random",
                n_iter=2.3,
                n_jobs=1,
            )
        # n_jobs
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs="n_jobs",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=0,
            )
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=1.3,
            )
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="grid",
                n_jobs=-3,
            )

    def test_types_get_optimized_signals(
        self,
    ):
        # Test that invalid names are caught
        so = self.so_with_calculated_preds
        with self.assertRaises(TypeError):
            so.get_optimized_signals(name=1)
        with self.assertRaises(TypeError):
            so.get_optimized_signals(name={})
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name=["test", "test2"])
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name="test2")
        # Test that if no signals have been calculated, an error is raised
        so = SignalOptimizer(
            inner_splitter=self.splitters[1], X=self.X_train, y=self.y_train
        )
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name="test2")

    @parameterized.expand(itertools.product([0, 1], [True, False]))
    def test_valid_get_optimized_signals(self, splitter_idx, change_n_splits):
        # Test that the output is a dataframe
        if change_n_splits:
            initial_nsplits = np.random.choice([2, 3, 5, 10])
            threshold_ndates = np.random.choice([21, 21 * 3, 21 * 6])
            so = SignalOptimizer(
                inner_splitter=self.splitters[splitter_idx],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits=initial_nsplits,
                threshold_ndates=threshold_ndates,
            )
        else:
            so = SignalOptimizer(
                inner_splitter=self.splitters[splitter_idx],
                X=self.X_train,
                y=self.y_train,
            )
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
            n_jobs=1,
        )
        df1 = so.get_optimized_signals(name="test")
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(df1.shape[1], 4)
        self.assertEqual(df1.columns[0], "cid")
        self.assertEqual(df1.columns[1], "real_date")
        self.assertEqual(df1.columns[2], "xcat")
        self.assertEqual(df1.columns[3], "value")
        self.assertEqual(df1.xcat.unique()[0], "test")
        # Add a second signal and check that the output is a dataframe
        so.calculate_predictions(
            name="test2",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
            n_jobs=1,
        )
        df2 = so.get_optimized_signals(name="test2")
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(df2.shape[1], 4)
        self.assertEqual(df2.columns[0], "cid")
        self.assertEqual(df2.columns[1], "real_date")
        self.assertEqual(df2.columns[2], "xcat")
        self.assertEqual(df2.columns[3], "value")
        self.assertEqual(df2.xcat.unique()[0], "test2")
        df3 = so.get_optimized_signals()
        self.assertIsInstance(df3, pd.DataFrame)
        self.assertEqual(df3.shape[1], 4)
        self.assertEqual(df3.columns[0], "cid")
        self.assertEqual(df3.columns[1], "real_date")
        self.assertEqual(df3.columns[2], "xcat")
        self.assertEqual(df3.columns[3], "value")
        self.assertEqual(len(df3.xcat.unique()), 2)
        df4 = so.get_optimized_signals(name=["test", "test2"])
        self.assertIsInstance(df4, pd.DataFrame)
        self.assertEqual(df4.shape[1], 4)
        self.assertEqual(df4.columns[0], "cid")
        self.assertEqual(df4.columns[1], "real_date")
        self.assertEqual(df4.columns[2], "xcat")
        self.assertEqual(df4.columns[3], "value")
        self.assertEqual(len(df4.xcat.unique()), 2)

    def test_types_get_optimal_models(self):
        
        so = self.so_with_calculated_preds
        with self.assertRaises(TypeError):
            so.get_optimal_models(name=1)
        with self.assertRaises(TypeError):
            so.get_optimal_models(name={})
        with self.assertRaises(ValueError):
            so.get_optimal_models(name=["test", "test2"])
        with self.assertRaises(ValueError):
            so.get_optimal_models(name="test2")
            
        splitter_idx = 0
        # Test that if no signals have been calculated, an error is raised
        so = SignalOptimizer(
            inner_splitter=self.splitters[splitter_idx], X=self.X_train, y=self.y_train
        )
        with self.assertRaises(ValueError):
            so.get_optimal_models(name="test2")

    @parameterized.expand(itertools.product([1], [True, False]))
    def test_valid_get_optimal_models(self, splitter_idx, change_n_splits):
        # Test that the output is a dataframe
        if change_n_splits:
            initial_nsplits = np.random.choice([2, 3, 5, 10])
            threshold_ndates = np.random.choice([21, 21 * 3, 21 * 6])
            so = SignalOptimizer(
                inner_splitter=self.splitters[splitter_idx],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits=initial_nsplits,
                threshold_ndates=threshold_ndates,
            )
        else:
            so = SignalOptimizer(
                inner_splitter=self.splitters[splitter_idx],
                X=self.X_train,
                y=self.y_train,
            )
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
            n_jobs=1,
        )
        df1 = so.get_optimal_models(name="test")
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(df1.shape[1], 5)
        self.assertEqual(df1.columns[0], "real_date")
        self.assertEqual(df1.columns[1], "name")
        self.assertEqual(df1.columns[2], "model_type")
        self.assertEqual(df1.columns[3], "hparams")
        self.assertEqual(df1.columns[4], "n_splits_used")
        if not change_n_splits:
            self.assertTrue(
                np.all(df1.iloc[:, 4] == self.splitters[splitter_idx].n_splits)
            )
        else:
            num_new_dates = len(
                pd.bdate_range(df1.real_date.min(), df1.real_date.max())
            )
            self.assertTrue(np.min(df1.iloc[:, 4]) == initial_nsplits)
            self.assertTrue(
                np.max(df1.iloc[:, 4])
                == initial_nsplits + (num_new_dates // threshold_ndates)
            )

        self.assertEqual(df1.name.unique()[0], "test")
        # Add a second signal and check that the output is a dataframe
        so.calculate_predictions(
            name="test2",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
            n_jobs=1,
        )
        df2 = so.get_optimal_models(name="test2")
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(df2.shape[1], 5)
        self.assertEqual(df2.columns[0], "real_date")
        self.assertEqual(df2.columns[1], "name")
        self.assertEqual(df2.columns[2], "model_type")
        self.assertEqual(df2.columns[3], "hparams")
        self.assertEqual(df2.columns[4], "n_splits_used")
        if not change_n_splits:
            self.assertTrue(
                np.all(df2.iloc[:, 4] == self.splitters[splitter_idx].n_splits)
            )
        else:
            num_new_dates = len(
                pd.bdate_range(df2.real_date.min(), df2.real_date.max())
            )
            self.assertTrue(np.min(df2.iloc[:, 4]) == initial_nsplits)
            self.assertTrue(
                np.max(df2.iloc[:, 4])
                == initial_nsplits + (num_new_dates // threshold_ndates)
            )
        self.assertEqual(df2.name.unique()[0], "test2")
        df3 = so.get_optimal_models()
        self.assertIsInstance(df3, pd.DataFrame)
        self.assertEqual(df3.shape[1], 5)
        self.assertEqual(df3.columns[0], "real_date")
        self.assertEqual(df3.columns[1], "name")
        self.assertEqual(df3.columns[2], "model_type")
        self.assertEqual(df3.columns[3], "hparams")
        self.assertEqual(df3.columns[4], "n_splits_used")
        if not change_n_splits:
            self.assertTrue(
                np.all(df3.iloc[:, 4] == self.splitters[splitter_idx].n_splits)
            )
        else:
            num_new_dates = len(
                pd.bdate_range(df3.real_date.min(), df3.real_date.max())
            )
            self.assertTrue(np.min(df3.iloc[:, 4]) == initial_nsplits)
            self.assertTrue(
                np.max(df3.iloc[:, 4])
                == initial_nsplits + (num_new_dates // threshold_ndates)
            )
        self.assertEqual(len(df3.name.unique()), 2)
        df4 = so.get_optimal_models(name=["test", "test2"])
        self.assertIsInstance(df4, pd.DataFrame)
        self.assertEqual(df4.shape[1], 5)
        self.assertEqual(df4.columns[0], "real_date")
        self.assertEqual(df4.columns[1], "name")
        self.assertEqual(df4.columns[2], "model_type")
        self.assertEqual(df4.columns[3], "hparams")
        self.assertEqual(df4.columns[4], "n_splits_used")
        if not change_n_splits:
            self.assertTrue(
                np.all(df4.iloc[:, 4] == self.splitters[splitter_idx].n_splits)
            )
        else:
            num_new_dates = len(
                pd.bdate_range(df4.real_date.min(), df4.real_date.max())
            )
            self.assertTrue(np.min(df4.iloc[:, 4]) == initial_nsplits)
            self.assertTrue(
                np.max(df4.iloc[:, 4])
                == initial_nsplits + (num_new_dates // threshold_ndates)
            )

        self.assertEqual(len(df4.name.unique()), 2)

    def test_types_feature_selection_heatmap(self):
        so = self.so_with_calculated_preds
        
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name=1)
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name=["test"])
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test2")
        # title
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", figsize="figsize")
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", figsize=1)
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(
                name="test", figsize=[1.5, 2]
            )  # needs to be a tuple!
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name="test", figsize=(1.5, "e"))
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test", figsize=(0,))
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test", figsize=(0, 1, 2))
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test", figsize=(2, -1))

    def test_valid_feature_selection_heatmap(self):
        so = self.so_with_calculated_preds
        try:
            so.feature_selection_heatmap(name="test")
        except Exception as e:
            self.fail(f"feature_selection_heatmap raised an exception: {e}")

    def test_types_models_heatmap(self):
        so = self.so_with_calculated_preds

        with self.assertRaises(TypeError):
            so.models_heatmap(name=1)
        with self.assertRaises(TypeError):
            so.models_heatmap(name=["test"])
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test2")
        # title
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", title=1)
        # cap
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", cap="cap")
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", cap=1.3)
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", cap=-1)
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", cap=0)
        # figsize
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize="figsize")
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize=1)
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize=[1.5, 2])  # needs to be a tuple!
        with self.assertRaises(TypeError):
            so.models_heatmap(name="test", figsize=(1.5, "e"))
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(0,))
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(0, 1, 2))
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(2, -1))

    def test_valid_models_heatmap(self):
        so = self.so_with_calculated_preds
        try:
            so.models_heatmap(name="test")
        except Exception as e:
            self.fail(f"models_heatmap raised an exception: {e}")
        # Repeat but for when cap > 20
        try:
            so.models_heatmap(name="test", cap=21)
        except Exception as e:
            self.fail(f"models_heatmap raised an exception: {e}")

    @parameterized.expand(itertools.product([0, 1], [True, False]))
    def test_valid_worker(self, splitter_idx, change_n_splits):
        # Check that the worker private method works as expected for a grid search
        outer_splitter = ExpandingIncrementPanelSplit(
            train_intervals=1,
            test_size=1,
            min_cids=4,
            min_periods=36,
            max_periods=None,
        )
        if change_n_splits:
            initial_nsplits = np.random.choice([2, 3, 5, 10])
            threshold_ndates = np.random.choice([21, 21 * 3, 21 * 6])
            so1 = SignalOptimizer(
                inner_splitter=self.splitters[splitter_idx],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits=initial_nsplits,
                threshold_ndates=threshold_ndates,
            )
        else:
            so1 = SignalOptimizer(
                inner_splitter=self.splitters[splitter_idx],
                X=self.X_train,
                y=self.y_train,
            )
        for idx, (train_idx, test_idx) in enumerate(
            outer_splitter.split(X=self.X_train, y=self.y_train)
        ):
            try:
                if not change_n_splits:
                    (
                        prediction_date,
                        modelchoice_data,
                        ftr_data,
                        inter_data,
                        ftr_selection_data,
                    ) = so1._worker(
                        train_idx=train_idx,
                        test_idx=test_idx,
                        name="test",
                        models=self.models,
                        metric=self.metric,
                        original_date_levels=sorted(
                            self.X_train.index.get_level_values(1).unique()
                        ),
                        hparam_grid=self.hparam_grid,
                        hparam_type="grid",
                    )
                else:
                    (
                        prediction_date,
                        modelchoice_data,
                        ftr_data,
                        inter_data,
                        ftr_selection_data,
                    ) = so1._worker(
                        train_idx=train_idx,
                        test_idx=test_idx,
                        name="test",
                        models=self.models,
                        metric=self.metric,
                        original_date_levels=sorted(
                            self.X_train.index.get_level_values(1).unique()
                        ),
                        hparam_grid=self.hparam_grid,
                        hparam_type="random",
                        n_iter=1,
                        nsplits_add=np.floor(idx / threshold_ndates),
                    )
            except Exception as e:
                self.fail(f"_worker raised an exception: {e}")

            self.assertIsInstance(prediction_date, list)
            self.assertTrue(prediction_date[0] == "test")
            self.assertIsInstance(prediction_date[1], pd.Index)
            self.assertIsInstance(prediction_date[2], pd.DatetimeIndex)
            self.assertIsInstance(prediction_date[3], np.ndarray)
            self.assertIsInstance(modelchoice_data, list)
            self.assertIsInstance(modelchoice_data[0], datetime.date)
            self.assertTrue(modelchoice_data[1] == "test")
            self.assertIsInstance(modelchoice_data[2], str)
            self.assertTrue(modelchoice_data[2] in ["linreg", "ridge"])
            self.assertIsInstance(modelchoice_data[3], dict)
            # feature coefficients
            self.assertIsInstance(ftr_data, list)
            self.assertTrue(len(ftr_data) == 2 + 3)  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_data[0], datetime.date)
            self.assertTrue(ftr_data[1] == "test")
            for i in range(2, len(ftr_data)):
                if ftr_data[i] != np.nan:
                    self.assertIsInstance(ftr_data[i], np.float32)
            # intercept data
            self.assertIsInstance(inter_data, list)
            self.assertTrue(len(inter_data) == 2 + 1)  # 1 intercept + 2 extra columns
            self.assertIsInstance(inter_data[0], datetime.date)
            self.assertTrue(inter_data[1] == "test")
            if inter_data[2] is not None:
                self.assertIsInstance(inter_data[2], np.float32)
            # feature selection data
            self.assertIsInstance(ftr_selection_data, list)
            self.assertTrue(
                len(ftr_selection_data) == 2 + 3
            )  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_selection_data[0], datetime.date)
            self.assertTrue(ftr_selection_data[1] == "test")
            for i in range(2, len(ftr_selection_data)):
                self.assertTrue(ftr_selection_data[i] in [0, 1])

        # Check that the worker private method works as expected for a random search
        if change_n_splits:
            initial_nsplits = np.random.choice([2, 3, 5, 10])
            threshold_ndates = np.random.choice([21, 21 * 3, 21 * 6])
            so2 = SignalOptimizer(
                inner_splitter=self.splitters[splitter_idx],
                X=self.X_train,
                y=self.y_train,
                initial_nsplits=initial_nsplits,
                threshold_ndates=threshold_ndates,
            )
        else:
            so2 = SignalOptimizer(
                inner_splitter=self.splitters[splitter_idx],
                X=self.X_train,
                y=self.y_train,
            )
        outer_splitter = ExpandingIncrementPanelSplit(
            train_intervals=1,
            test_size=1,
            min_cids=4,
            min_periods=36,
            max_periods=None,
        )
        for idx, (train_idx, test_idx) in enumerate(
            outer_splitter.split(X=self.X_train, y=self.y_train)
        ):
            try:
                if not change_n_splits:
                    (
                        prediction_date,
                        modelchoice_data,
                        ftr_data,
                        inter_data,
                        ftr_selection_data,
                    ) = so2._worker(
                        train_idx=train_idx,
                        test_idx=test_idx,
                        name="test",
                        models=self.models,
                        metric=self.metric,
                        original_date_levels=sorted(
                            self.X_train.index.get_level_values(1).unique()
                        ),
                        hparam_grid={"linreg": {}, "ridge": {"alpha": stats.expon()}},
                        hparam_type="random",
                        n_iter=1,
                    )
                else:
                    (
                        prediction_date,
                        modelchoice_data,
                        ftr_data,
                        inter_data,
                        ftr_selection_data,
                    ) = so2._worker(
                        train_idx=train_idx,
                        test_idx=test_idx,
                        name="test",
                        models=self.models,
                        metric=self.metric,
                        original_date_levels=sorted(
                            self.X_train.index.get_level_values(1).unique()
                        ),
                        hparam_grid={"linreg": {}, "ridge": {"alpha": stats.expon()}},
                        hparam_type="random",
                        n_iter=1,
                        nsplits_add=np.floor(idx / threshold_ndates),
                    )
            except Exception as e:
                self.fail(f"_worker raised an exception: {e}")
            self.assertIsInstance(prediction_date, list)
            self.assertTrue(prediction_date[0] == "test")
            self.assertIsInstance(prediction_date[1], pd.Index)
            self.assertIsInstance(prediction_date[2], pd.DatetimeIndex)
            self.assertIsInstance(prediction_date[3], np.ndarray)
            self.assertIsInstance(modelchoice_data, list)
            self.assertIsInstance(modelchoice_data[0], datetime.date)
            self.assertTrue(modelchoice_data[1] == "test")
            self.assertIsInstance(modelchoice_data[2], str)
            self.assertTrue(modelchoice_data[2] in ["linreg", "ridge"])
            self.assertIsInstance(modelchoice_data[3], dict)
            # feature coefficients
            self.assertIsInstance(ftr_data, list)
            self.assertTrue(len(ftr_data) == 2 + 3)  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_data[0], datetime.date)
            self.assertTrue(ftr_data[1] == "test")
            for i in range(2, len(ftr_data)):
                if ftr_data[i] != np.nan:
                    self.assertIsInstance(ftr_data[i], np.float32)
            # intercept data
            self.assertIsInstance(inter_data, list)
            self.assertTrue(len(inter_data) == 2 + 1)  # 1 intercept + 2 extra columns
            self.assertIsInstance(inter_data[0], datetime.date)
            self.assertTrue(inter_data[1] == "test")
            if inter_data[2] is not None:
                self.assertIsInstance(inter_data[2], np.float32)
            # feature selection data
            self.assertIsInstance(ftr_selection_data, list)
            self.assertTrue(
                len(ftr_selection_data) == 2 + 3
            )  # 3 ftrs + 2 extra columns
            self.assertIsInstance(ftr_selection_data[0], datetime.date)
            self.assertTrue(ftr_selection_data[1] == "test")
            for i in range(2, len(ftr_selection_data)):
                self.assertTrue(ftr_selection_data[i] in [0, 1])
        #     # Check that the worker function outputs as expected when no models are selected
        #     if change_n_splits:
        #         initial_nsplits = np.random.choice([2, 3, 5, 10])
        #         threshold_ndates = np.random.choice([21, 21 * 3, 21 * 6])
        #         so3 = SignalOptimizer(
        #             inner_splitter=self.splitters[splitter_idx],
        #             X=self.X_train,
        #             y=self.y_train,
        #             initial_nsplits=initial_nsplits,
        #             threshold_ndates=threshold_ndates,
        #         )
        #     else:
        #         so3 = SignalOptimizer(
        #             inner_splitter=self.splitters[splitter_idx],
        #             X=self.X_train,
        #             y=self.y_train,
        #         )
        #     outer_splitter = ExpandingIncrementPanelSplit(
        #         train_intervals=1,
        #         test_size=1,
        #         min_cids=4,
        #         min_periods=36,
        #         max_periods=None,
        #     )
        #     models = {
        #         "linreg": Pipeline(
        #             [
        #                 ("selector", LassoSelector(alpha=10000)),
        #                 ("linreg", LinearRegression()),
        #             ]
        #         )
        #     }
        #     for idx, (train_idx, test_idx) in enumerate(
        #         outer_splitter.split(X=self.X_train, y=self.y_train)
        #     ):
        #         try:
        #             if not change_n_splits:
        #                 (
        #                     prediction_date,
        #                     modelchoice_data,
        #                     ftr_data,
        #                     inter_data,
        #                     ftr_selection_data,
        #                 ) = so3._worker(
        #                     train_idx=train_idx,
        #                     test_idx=test_idx,
        #                     name="test",
        #                     models=models,
        #                     metric=self.metric,
        #                     original_date_levels=sorted(
        #                         self.X_train.index.get_level_values(1).unique()
        #                     ),
        #                     hparam_grid={"linreg": {}},
        #                     hparam_type="random",
        #                     n_iter=1,
        #                 )
        #             else:
        #                 (
        #                     prediction_date,
        #                     modelchoice_data,
        #                     ftr_data,
        #                     inter_data,
        #                     ftr_selection_data,
        #                 ) = so3._worker(
        #                     train_idx=train_idx,
        #                     test_idx=test_idx,
        #                     name="test",
        #                     models=models,
        #                     metric=self.metric,
        #                     original_date_levels=sorted(
        #                         self.X_train.index.get_level_values(1).unique()
        #                     ),
        #                     hparam_grid={"linreg": {}},
        #                     hparam_type="random",
        #                     n_iter=1,
        #                     nsplits_add=np.floor(idx / threshold_ndates),
        #                 )
        #         except Exception as e:
        #             self.fail(f"_worker raised an exception: {e}")
        #         self.assertIsInstance(prediction_date, list)
        #         self.assertTrue(prediction_date[0] == "test")
        #         self.assertIsInstance(prediction_date[1], pd.Index)
        #         self.assertIsInstance(prediction_date[2], pd.DatetimeIndex)
        #         self.assertIsInstance(prediction_date[3], np.ndarray)
        #         self.assertTrue(np.all(prediction_date[3] == 0))
        #         self.assertIsInstance(modelchoice_data, list)
        #         self.assertIsInstance(modelchoice_data[0], datetime.date)
        #         self.assertTrue(modelchoice_data[1] == "test")
        #         self.assertTrue(modelchoice_data[2] == "None")
        #         self.assertTrue(modelchoice_data[3] == {})
        #         # feature coefficients
        #         self.assertIsInstance(ftr_data, list)
        #         self.assertTrue(len(ftr_data) == 2 + 3)
        #         self.assertIsInstance(ftr_data[0], datetime.date)
        #         self.assertTrue(ftr_data[1] == "test")
        #         for ftr_coef in ftr_data[2:]:
        #             self.assertTrue(ftr_coef is np.nan)
        #         # intercept data
        #         self.assertIsInstance(inter_data, list)
        #         self.assertTrue(len(inter_data) == 2 + 1)
        #         self.assertIsInstance(inter_data[0], datetime.date)
        #         self.assertTrue(inter_data[1] == "test")
        #         self.assertTrue(inter_data[2] is np.nan)
        #         # feature selection data
        #         self.assertIsInstance(ftr_selection_data, list)
        #         self.assertTrue(len(ftr_selection_data) == 2 + 3)  # 3 ftrs + 2 extra columns
        #         self.assertIsInstance(ftr_selection_data[0], datetime.date)
        #         self.assertTrue(ftr_selection_data[1] == "test")
        #         for i in range(2, len(ftr_selection_data)):
        #             self.assertTrue(ftr_selection_data[i] in [0,1])

    def test_types_get_intercepts(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_intercepts(name="test2")
        # Test that the wrong dtype of a signal name raises an error
        with self.assertRaises(TypeError):
            so.get_intercepts(name=1)

    def test_valid_get_intercepts(self):
        so = self.so_with_calculated_preds
        # Test that running get_intercepts on pipeline "test" works
        try:
            intercepts = so.get_intercepts(name="test")
        except Exception as e:
            self.fail(f"get_intercepts raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(intercepts, pd.DataFrame)
        self.assertEqual(intercepts.shape[1], 3)
        self.assertEqual(intercepts.columns[0], "real_date")
        self.assertEqual(intercepts.columns[1], "name")
        self.assertEqual(intercepts.columns[2], "intercepts")
        self.assertTrue(intercepts.name.unique()[0] == "test")
        self.assertTrue(intercepts.isna().sum().sum() == 0)

    def test_types_get_selected_features(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_selected_features(name="test2")
        # Test that the wrong dtype of a signal name raises an error
        with self.assertRaises(TypeError):
            so.get_selected_features(name=1)

    def test_valid_get_selected_features(self):
        so = self.so_with_calculated_preds
        # Test that running get_selected_features on pipeline "test" works
        try:
            selected_ftrs = so.get_selected_features(name="test")
        except Exception as e:
            self.fail(f"get_selected_features raised an exception: {e}")
        # Test that the output is as expected
        # Test that the output is as expected
        self.assertIsInstance(selected_ftrs, pd.DataFrame)
        self.assertEqual(selected_ftrs.shape[1], 5)
        self.assertEqual(selected_ftrs.columns[0], "real_date")
        self.assertEqual(selected_ftrs.columns[1], "name")
        for i in range(2, 5):
            self.assertEqual(selected_ftrs.columns[i], self.X_train.columns[i - 2])
        self.assertTrue(selected_ftrs.name.unique()[0] == "test")
        self.assertTrue(selected_ftrs.isna().sum().sum() == 0)

    def test_types_get_ftr_coefficients(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_ftr_coefficients(name="test2")
        # Test that the wrong dtype of a signal name raises an error
        with self.assertRaises(TypeError):
            so.get_ftr_coefficients(name=1)

    def test_valid_get_ftr_coefficients(self):
        so = self.so_with_calculated_preds
        # Test that running get_ftr_coefficients on pipeline "test" works
        try:
            ftr_coefficients = so.get_ftr_coefficients(name="test")
        except Exception as e:
            self.fail(f"get_ftr_coefficients raised an exception: {e}")
        # Test that the output is as expected
        self.assertIsInstance(ftr_coefficients, pd.DataFrame)
        self.assertEqual(ftr_coefficients.shape[1], 5)
        self.assertEqual(ftr_coefficients.columns[0], "real_date")
        self.assertEqual(ftr_coefficients.columns[1], "name")
        for i in range(2, 5):
            self.assertEqual(ftr_coefficients.columns[i], self.X_train.columns[i - 2])
        self.assertTrue(ftr_coefficients.name.unique()[0] == "test")
        self.assertTrue(ftr_coefficients.isna().sum().sum() == 0)

    @parameterized.expand(itertools.product([True, False]))
    def test_types_get_parameter_stats(self, include_intercept):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.get_parameter_stats(name="test2", include_intercept=include_intercept)
        with self.assertRaises(TypeError):
            so.get_parameter_stats(name=1, include_intercept=include_intercept)
        with self.assertRaises(TypeError):
            so.get_parameter_stats(name="test", include_intercept=2)

    @parameterized.expand(itertools.product([True, False]))
    def test_valid_get_parameter_stats(self, include_intercept):
        so = self.so_with_calculated_preds
        # Test that running get_parameter_stats on pipeline "test" works
        try:
            parameter_stats = so.get_parameter_stats(
                name="test", include_intercept=include_intercept
            )
        except Exception as e:
            self.fail(f"get_parameter_stats raised an exception: {e}")
        # Test that the output is as expected
        if include_intercept:
            self.assertTrue(len(parameter_stats) == 4)
            ftr_coefs = so.get_ftr_coefficients(name="test")
            intercepts = so.get_intercepts(name="test")
            mean_coefs = ftr_coefs.iloc[:, 2:].mean(skipna=True)
            std_coefs = ftr_coefs.iloc[:, 2:].std(skipna=True)
            mean_intercept = intercepts.iloc[:, 2:].mean(skipna=True)
            std_intercept = intercepts.iloc[:, 2:].std(skipna=True)
            self.assertTrue(np.all(parameter_stats[0] == mean_coefs))
            self.assertTrue(np.all(parameter_stats[1] == std_coefs))
            self.assertTrue(np.all(parameter_stats[2] == mean_intercept))
            self.assertTrue(np.all(parameter_stats[3] == std_intercept))
        else:
            self.assertTrue(len(parameter_stats) == 2)
            ftr_coefs = so.get_ftr_coefficients(name="test")
            mean_coefs = ftr_coefs.iloc[:, 2:].mean(skipna=True)
            std_coefs = ftr_coefs.iloc[:, 2:].std(skipna=True)
            self.assertTrue(np.all(parameter_stats[0] == mean_coefs))
            self.assertTrue(np.all(parameter_stats[1] == std_coefs))

    def test_types_coefs_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.coefs_timeplot(name="test2")
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", figsize="figsize")
        with self.assertRaises(ValueError):
            so.coefs_timeplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", figsize=("hello", "hello"))
        # ftrs_renamed
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", ftrs_renamed=1)
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", ftrs_renamed={1: "ftr1"})
        with self.assertRaises(TypeError):
            so.coefs_timeplot(name="test", ftrs_renamed={"ftr1": 1})
        with self.assertRaises(ValueError):
            so.coefs_timeplot(name="test", ftrs_renamed={"ftr1": "ftr2"})

    def test_valid_coefs_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that running coefs_timeplot on pipeline "test" works
        try:
            so.coefs_timeplot(name="test")
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        # Check that the legend is correct
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(np.all(sorted(self.X_train.columns) == sorted(labels)))
        # Now rerun coefs_timeplot but with a feature renaming dictionary
        ftr_dict = {"CPI": "inflation"}
        try:
            so.coefs_timeplot(name="test", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            np.all(
                sorted(self.X_train.rename(columns=ftr_dict).columns) == sorted(labels)
            )
        )
        # Now rename two features
        ftr_dict = {"CPI": "inflation", "GROWTH": "growth"}
        try:
            so.coefs_timeplot(name="test", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            np.all(
                sorted(self.X_train.rename(columns=ftr_dict).columns) == sorted(labels)
            )
        )
        # Now rename all features
        ftr_dict = {ftr: f"ftr{i}" for i, ftr in enumerate(self.X_train.columns)}
        try:
            so.coefs_timeplot(name="test", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            np.all(
                sorted(self.X_train.rename(columns=ftr_dict).columns) == sorted(labels)
            )
        )
        # Finally, test that the title works
        title = ax.get_title()
        self.assertTrue(title == "Feature coefficients for pipeline: test")
        # Try changing the title
        try:
            so.coefs_timeplot(name="test", title="hello")
        except Exception as e:
            self.fail(f"coefs_timeplot raised an exception: {e}")
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "hello")

    def test_types_intercepts_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.intercepts_timeplot(name="test2")
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize="figsize")
        with self.assertRaises(ValueError):
            so.intercepts_timeplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.intercepts_timeplot(name="test", figsize=("hello", "hello"))

    def test_valid_intercepts_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that running intercepts_timeplot on pipeline "test" works
        try:
            so.intercepts_timeplot(name="test")
        except Exception as e:
            self.fail(f"intercepts_timeplot raised an exception: {e}")

    def test_types_coefs_stackedbarplot(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test2")
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize="figsize")
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", figsize=("hello", "hello"))
        # ftrs_renamed
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed=1)
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed={1: "ftr1"})
        with self.assertRaises(TypeError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed={"ftr1": 1})
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test", ftrs_renamed={"ftr1": "ftr2"})

    def test_valid_coefs_stackedbarplot(self):
        so = self.so_with_calculated_preds
        # Test that running coefs_stackedbarplot on pipeline "test" works
        try:
            so.coefs_stackedbarplot(name="test")
        except Exception as e:
            self.fail(f"coefs_stackedbarplot raised an exception: {e}")
        # Check that the title is correct
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "Stacked bar plot of model coefficients: test")
        # Change the title
        try:
            so.coefs_stackedbarplot(name="test", title="hello")
        except Exception as e:
            self.fail(f"coefs_stackedbarplot raised an exception: {e}")
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "hello")
        # Now rerun coefs_stackedbarplot but with a feature renaming dictionary
        ftr_dict = {"CPI": "inflation"}
        try:
            so.coefs_stackedbarplot(name="test", ftrs_renamed=ftr_dict)
        except Exception as e:
            self.fail(f"coefs_stackedbarplot raised an exception: {e}")
        ax = plt.gca()
        legend = ax.get_legend()
        labels = sorted([text.get_text() for text in legend.get_texts()])
        # Check that the legend is correct
        ftrcoef_df = so.get_ftr_coefficients(name="test")
        ftrcoef_df["year"] = ftrcoef_df["real_date"].dt.year
        ftrcoef_df = ftrcoef_df.drop(columns=["real_date", "name"])
        ftrcoef_df = ftrcoef_df.rename(columns=ftr_dict)
        avg_coefs = ftrcoef_df.groupby("year").mean()
        pos_coefs = avg_coefs.clip(lower=0)
        neg_coefs = avg_coefs.clip(upper=0)
        correct_labels = [
            col for col in list(pos_coefs.sum().index[pos_coefs.sum() > 0])
        ]
        correct_labels += [
            col for col in list(neg_coefs.sum().index[neg_coefs.sum() < 0])
        ]
        correct_labels = sorted(correct_labels)
        self.assertTrue(np.all(labels == correct_labels))

    def test_types_nsplits_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that a wrong signal name raises an error
        with self.assertRaises(ValueError):
            so.nsplits_timeplot(name="test2")
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name=1)
        # title
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name="test", title=1)
        # figsize
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name="test", figsize="figsize")
        with self.assertRaises(ValueError):
            so.nsplits_timeplot(name="test", figsize=(0, 1, 2))
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name="test", figsize=(10, "hello"))
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name="test", figsize=("hello", 6))
        with self.assertRaises(TypeError):
            so.nsplits_timeplot(name="test", figsize=("hello", "hello"))

    def test_valid_nsplits_timeplot(self):
        so = self.so_with_calculated_preds
        # Test that running nsplits_timeplot on pipeline "test" works
        try:
            so.nsplits_timeplot(name="test")
        except Exception as e:
            self.fail(f"nsplits_timeplot raised an exception: {e}")
        # Test that setting the title works
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "Number of CV splits for pipeline: test")
        # Try changing the title
        try:
            so.nsplits_timeplot(name="test", title="hello")
        except Exception as e:
            self.fail(f"nsplits_timeplot raised an exception: {e}")
        ax = plt.gca()
        title = ax.get_title()
        self.assertTrue(title == "hello")

    def test_invalid_plots(self):
        splitter_idx = 1
        so = SignalOptimizer(
            inner_splitter=self.splitters[splitter_idx],
            X=self.X_train,
            y=self.y_train,
        )
        # Test that an error is raised if calculate_predictions has not been run
        with self.assertRaises(ValueError):
            so.nsplits_timeplot(name="test")
        # Test that an error is raised if calculate_predictions has not been run
        with self.assertRaises(ValueError):
            so.coefs_stackedbarplot(name="test")
        # Test that an error is raised if calculate_predictions has not been run
        with self.assertRaises(ValueError):
            so.intercepts_timeplot(name="test")
        # Test that an error is raised if calculate_predictions has not been run
        with self.assertRaises(ValueError):
            so.coefs_timeplot(name="test")
        # Test that if no signals have been calculated, an error is raised
        with self.assertRaises(ValueError):
            so.get_parameter_stats(name="test", include_intercept=False)
        with self.assertRaises(ValueError):
            so.get_parameter_stats(name="test", include_intercept=True)
        # Test that if no signals have been calculated, an error is raised
        with self.assertRaises(ValueError):
            so.get_ftr_coefficients(name="test")
        # Test that if no signals have been calculated, an error is raised
        with self.assertRaises(ValueError):
            so.get_intercepts(name="test")
        # Test that if no signals have been calculated, an error is raised
        with self.assertRaises(ValueError):
            so.get_selected_features(name="test")
        # Test that invalid names are caught
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name=1)
        with self.assertRaises(TypeError):
            so.feature_selection_heatmap(name=[1, 2, 3])
        with self.assertRaises(ValueError):
            so.feature_selection_heatmap(name="test")
        # Test that invalid names are caught
        with self.assertRaises(TypeError):
            so.models_heatmap(name=1)
        with self.assertRaises(TypeError):
            so.models_heatmap(name=[1, 2, 3])
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test")
