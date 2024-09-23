import itertools
import unittest

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS
from linearmodels.panel import RandomEffects as lm_RandomEffects
from parameterized import parameterized
from scipy.stats import expon
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from statsmodels.tools.tools import add_constant

from macrosynergy.learning import (
    BaseWeightedRegressor,
    LADRegressor,
    LassoSelector,
    NaivePredictor,
    RollingKFoldPanelSplit,
    SignalOptimizer,
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
    WeightedLADRegressor,
    WeightedLinearRegression,
    panel_cv_scores,
    RandomEffects,
)

from statsmodels.regression.mixed_linear_model import MixedLM


class TestRandomEffects(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-06-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all eligible dates
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

        X = df.drop(columns="XR")
        y = df["XR"]

        unique_xss = sorted(X.index.get_level_values(0).unique())
        xs_codes = dict(zip(unique_xss, range(1, len(unique_xss) + 1)))

        self.X = X.rename(xs_codes, level=0, inplace=False)
        self.y = y.rename(xs_codes, level=0, inplace=False)

    def test_init(self):
        re = RandomEffects()
        self.assertIsInstance(re, BaseEstimator)
        self.assertIsInstance(re, RandomEffects)
        self.assertEqual(re.fit_intercept, True)
        # self.assertEqual(re.n_jobs, 1)
        # self.assertEqual(re.verbose, 0)
        # self.assertEqual(re.fit_intercept, True)
        # self.assertEqual(re.n_jobs, 1)
        # self.assertEqual(re.verbose, 0)
        # self.assertEqual(re._estimator_type, "regressor")
        # self.assertIsNone(re._estimator)
        # self.assertIsNone(re._fitted_estimator

    def test_fit_no_intercept(self):

        re = RandomEffects(fit_intercept=False, group_col="real_date")
        re.fit(self.X, self.y)
        lm_re = lm_RandomEffects(self.y.swaplevel(), self.X.swaplevel()).fit()

        self.assertTrue(np.allclose(re.coef_, lm_re.params.values, atol=1e-6))
        
    def test_fit_with_intercept(self):

        re = RandomEffects(fit_intercept=True, group_col="real_date")
        re.fit(self.X, self.y)
        
        groups = self.y.index.get_level_values(1)
        mlm_re = MixedLM(self.y, self.X, groups=groups).fit(reml=False)
        X = add_constant(self.X)
        lm_re = lm_RandomEffects(self.y.swaplevel(), X.swaplevel()).fit()

        self.assertTrue(np.allclose(re.coef_, lm_re.params.values, atol=1e-6))


if __name__ == "__main__":
    tests = TestRandomEffects()
    tests.setUpClass()
    tests.test_fit_with_intercept()
