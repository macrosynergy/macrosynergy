import numpy as np
import pandas as pd
import scipy.stats as stats

import unittest 
import itertools

from parameterized import parameterized

from macrosynergy.learning import (
    SignalOptimizer,
    ExpandingKFoldPanelSplit,
    RollingKFoldPanelSplit,
    ExpandingIncrementPanelSplit,
)

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error

class TestAll(unittest.TestCase):
    def setUp(self):
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
        train = df[df.index.get_level_values(1)<=pd.Timestamp(day=1,month=10,year=2020)]
        test = df[df.index.get_level_values(1)>pd.Timestamp(day=1,month=10,year=2020)]
        self.X_train = train.drop(columns="XR")
        self.y_train = train["XR"]
        self.X_test = test.drop(columns="XR")
        self.y_test = test["XR"]

        # instantiate some splitters
        expandingkfold = ExpandingKFoldPanelSplit(n_splits=5)
        rollingkfold = RollingKFoldPanelSplit(n_splits=5)
        expandingincrement = ExpandingIncrementPanelSplit(min_cids=2, min_periods = 100)
        self.splitters = [expandingkfold, rollingkfold, expandingincrement]

        # create blacklist dictionary 
        self.black_valid = {
            "AUD": (pd.Timestamp(year=2018, month=9, day=1), pd.Timestamp(year=2020, month=4, day=1)),
            "GBP": (pd.Timestamp(year=2019, month=6, day=1), pd.Timestamp(year=2020, month=2, day=1))
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
            "AUD": [pd.Timestamp(year=2018, month=9, day=1), pd.Timestamp(year=2018, month=10, day=1)],
            "GBP": [pd.Timestamp(year=2019, month=6, day=1), pd.Timestamp(year=2100, month=1, day=1)],
        }
        self.black_invalid4 = {
            "AUD": (pd.Timestamp(year=2018, month=9, day=1),),
            "GBP": (pd.Timestamp(year=2019, month=6, day=1), pd.Timestamp(year=2100, month=1, day=1)),
        }
        self.black_invalid5 = {
            1: (pd.Timestamp(year=2018, month=9, day=1), pd.Timestamp(year=2018, month=10, day=1)),
            2: (pd.Timestamp(year=2019, month=6, day=1), pd.Timestamp(year=2100, month=1, day=1))
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


    @parameterized.expand(itertools.product([0,1,2],[True, False]))
    def test_valid_init(self, idx, use_blacklist):
        # Test standard instantiation of the SignalOptimizer works as expected
        inner_splitter = self.splitters[idx]
        try:
            blacklist = self.black_valid if use_blacklist else None
            so = SignalOptimizer(inner_splitter=inner_splitter,X=self.X_train,y=self.y_train, blacklist=blacklist)
        except Exception as e:
            self.fail(f"Instantiation of the SignalOptimizer raised an exception: {e}")
        self.assertIsInstance(so, SignalOptimizer)
        self.assertEqual(so.inner_splitter, inner_splitter)
        pd.testing.assert_frame_equal(so.X, self.X_train)
        pd.testing.assert_series_equal(so.y, self.y_train)
        self.assertEqual(so.blacklist, blacklist)

    def test_types_init(self):
        inner_splitter = KFold(n_splits=5, shuffle=False)
        with self.assertRaises(TypeError):
            so = SignalOptimizer(inner_splitter=inner_splitter,X=self.X_train,y=self.y_train)  
        with self.assertRaises(TypeError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X="self.X",y=self.y_train)
        with self.assertRaises(TypeError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y="self.y")
        with self.assertRaises(ValueError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train.reset_index(),y=self.y_train)
        with self.assertRaises(ValueError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train.reset_index())
        with self.assertRaises(ValueError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train.iloc[1:])
        with self.assertRaises(TypeError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train, blacklist="blacklist")
        # check that an incorrect blacklisting format is caught
        with self.assertRaises(TypeError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train, blacklist = self.black_invalid1)
        with self.assertRaises(TypeError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train, blacklist = self.black_invalid2)
        with self.assertRaises(TypeError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train, blacklist = self.black_invalid3)
        with self.assertRaises(ValueError):
            so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train, blacklist = self.black_invalid4)


    def test_valid_calculate_predictions(self):
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        # check a correct optimisation runs
        try:
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        # checkout the output is a dataframe
        df1 = so.preds.copy()
        self.assertIsInstance(df1, pd.DataFrame)
        if len(df1.xcat.unique()) != 1:
            self.fail("The signal dataframe should only contain one xcat")
        self.assertEqual(df1.xcat.unique()[0], "test")
        # Test that blacklisting works as expected
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train, blacklist=self.black_valid)
        try:
            so.calculate_predictions(
                name="test",
                models=self.models,
                metric=self.metric,
                hparam_grid=self.hparam_grid,
                hparam_type="random",
            )
        except Exception as e:
            self.fail(f"calculate_predictions raised an exception: {e}")
        df3 = so.preds.copy()
        self.assertIsInstance(df3, pd.DataFrame)
        for cross_section, periods in self.black_valid.items():
            cross_section_key = cross_section.split("_")[0]
            self.assertTrue(len(df3[(df3.cid==cross_section_key) & (df3.real_date >= periods[0]) & (df3.real_date <= periods[1])].dropna())==0)

    def test_types_calculate_predictions(self):
        # Training set only
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        # name
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name=1,
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
            )
        # models 
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = {},
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = [LinearRegression()],
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = {1: LinearRegression()},
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = {"ols": 1},
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
            )
        # metric
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = "metric",
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
            )
        # hparam_grid - grid search
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = [1,2],
                hparam_type="grid",
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"ols": [1,2]},
                hparam_type="grid",
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {1: [1,2]},
                hparam_type="grid",
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"ols": {"alpha": 1}},
                hparam_type="grid",
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"ols": {1: 1}, "ridge": {"alpha": [1,2,3]}},
                hparam_type="grid",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"ols": {}, "ridge": {"alpha": []}},
                hparam_type="grid",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"linreg": {}},
                hparam_type="grid",
            )
        # hparam_grid - random search
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = [1,2],
                hparam_type="random",
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"ols": [1,2]},
                hparam_type="random",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"ols": {}, "ridge": {"alpha": []}},
                hparam_type="random",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"ols": {"alpha": 1}},
                hparam_type="random",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"ridge": {"alpha": stats.expon()}},
                hparam_type="random",
            )
        # hparam_type
        with self.assertRaises(NotImplementedError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="bayes",
            )
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type=1,
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="somethingelse",
            )
        # min_cids 
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
                min_cids="min_cids",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
                min_cids=-1,
            )
        # min_periods
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
                min_periods="min_periods",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
                min_periods=-1,
            )
        # n_iter 
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="random",
                n_iter="n_iter",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="random",
                n_iter=-1,
            )
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="random",
                n_iter=0,
            )
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = {"linreg": {}, "ridge": {"alpha": stats.expon()}},
                hparam_type="random",
                n_iter=2.3,
            )
        # n_jobs 
        with self.assertRaises(TypeError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
                n_jobs="n_jobs",
            )
        with self.assertRaises(ValueError):
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
                n_jobs=0,
            )
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
                n_jobs=1.3,
            )
            so.calculate_predictions(
                name="test",
                models = self.models,
                metric = self.metric,
                hparam_grid = self.hparam_grid,
                hparam_type="grid",
                n_jobs=-3,
            )
    def test_types_get_optimized_signals(
            self,
    ):
        # Test that invalid names are caught 
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
        )
        with self.assertRaises(TypeError):
            so.get_optimized_signals(name=1)
        with self.assertRaises(TypeError):
            so.get_optimized_signals(name={})
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name=["test", "test2"])
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name="test2")
        # Test that if no signals have been calculated, an error is raised
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        with self.assertRaises(ValueError):
            so.get_optimized_signals(name="test2")

    def test_valid_get_optimized_signals(self):
        # Test that the output is a dataframe
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
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
        # Test that invalid names are caught 
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
        )
        with self.assertRaises(TypeError):
            so.get_optimal_models(name=1)
        with self.assertRaises(TypeError):
            so.get_optimal_models(name={})
        with self.assertRaises(ValueError):
            so.get_optimal_models(name=["test", "test2"])
        with self.assertRaises(ValueError):
            so.get_optimal_models(name="test2")
        # Test that if no signals have been calculated, an error is raised
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        with self.assertRaises(ValueError):
            so.get_optimal_models(name="test2")

    def test_valid_get_optimal_models(self):
        # Test that the output is a dataframe
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
        )
        df1 = so.get_optimal_models(name="test")
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertEqual(df1.shape[1], 4)
        self.assertEqual(df1.columns[0], "real_date")
        self.assertEqual(df1.columns[1], "name")
        self.assertEqual(df1.columns[2], "model_type")
        self.assertEqual(df1.columns[3], "hparams")
        self.assertEqual(df1.name.unique()[0], "test")
        # Add a second signal and check that the output is a dataframe
        so.calculate_predictions(
            name="test2",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
        )
        df2 = so.get_optimal_models(name="test2")
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertEqual(df2.shape[1], 4)
        self.assertEqual(df2.columns[0], "real_date")
        self.assertEqual(df2.columns[1], "name")
        self.assertEqual(df2.columns[2], "model_type")
        self.assertEqual(df2.columns[3], "hparams")
        self.assertEqual(df2.name.unique()[0], "test2")
        df3 = so.get_optimal_models()
        self.assertIsInstance(df3, pd.DataFrame)
        self.assertEqual(df3.shape[1], 4)
        self.assertEqual(df3.columns[0], "real_date")
        self.assertEqual(df3.columns[1], "name")
        self.assertEqual(df3.columns[2], "model_type")
        self.assertEqual(df3.columns[3], "hparams")
        self.assertEqual(len(df3.name.unique()), 2)
        df4 = so.get_optimal_models(name=["test", "test2"])
        self.assertIsInstance(df4, pd.DataFrame)
        self.assertEqual(df4.shape[1], 4)
        self.assertEqual(df4.columns[0], "real_date")
        self.assertEqual(df4.columns[1], "name")
        self.assertEqual(df4.columns[2], "model_type")
        self.assertEqual(df4.columns[3], "hparams")
        self.assertEqual(len(df4.name.unique()), 2)

    def test_types_models_heatmap(self):
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        # Test that invalid names are caught
        with self.assertRaises(TypeError):
            so.models_heatmap(name=1)
        with self.assertRaises(TypeError):
            so.models_heatmap(name=[1,2,3])
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test")
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
        )
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
            so.models_heatmap(name="test", figsize=[1.5,2]) # needs to be a tuple!
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(0,))
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(0, 1, 2))
        with self.assertRaises(ValueError):
            so.models_heatmap(name="test", figsize=(2, -1))

    def test_valid_models_heatmap(self):
        so = SignalOptimizer(inner_splitter=self.splitters[1],X=self.X_train,y=self.y_train)
        so.calculate_predictions(
            name="test",
            models=self.models,
            metric=self.metric,
            hparam_grid=self.hparam_grid,
            hparam_type="grid",
        )
        try:
            so.models_heatmap(name="test")
        except Exception as e:
            self.fail(f"models_heatmap raised an exception: {e}")