import numpy as np
import pandas as pd
import unittest
import scipy.stats as stats

from sklearn.linear_model import LinearRegression

from macrosynergy.learning import (
    neg_mean_abs_corr,
)
from macrosynergy.learning.forecasting.model_systems import (
    BaseRegressionSystem,
    LinearRegressionSystem,
    LADRegressionSystem,
    RidgeRegressionSystem,
    CorrelationVolatilitySystem,
)

from parameterized import parameterized

from functools import partial

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils.df_utils import categories_df


class TestAbsCorrScorer(unittest.TestCase):
    def setUp(self):
        # Generate a panel of test data
        self.cids = ["AUD", "CAD", "GBP", "USD"]
        self.xcats = ["BENCH_XR", "XR"]
        cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD"] = ["2014-01-01", "2020-12-31", 0, 1]
        df_cids.loc["CAD"] = ["2015-01-01", "2020-12-31", 0, 1]
        df_cids.loc["GBP"] = ["2015-01-01", "2020-12-31", 0, 1]
        df_cids.loc["USD"] = ["2015-01-01", "2020-12-31", 0, 1]

        df_xcats = pd.DataFrame(index=self.xcats, columns=cols)
        df_xcats.loc["XR"] = ["2014-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["BENCH_XR"] = ["2015-01-01", "2020-12-31", 1, 2, 0.95, 1]

        self.df = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.df["value"] = self.df["value"].astype("float32")

        Xy = categories_df(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
            val="value",
            freq="W",
            lag=0,
            xcat_aggs=["sum", "sum"],
        ).dropna()

        # Create training set prior to 2020
        Xy_train = Xy[Xy.index.get_level_values("real_date") < "2020-01-01"]
        Xy_test = Xy[Xy.index.get_level_values("real_date") >= "2020-01-01"]

        self.X_train = pd.DataFrame(Xy_train.iloc[:, 0])
        self.y_train = Xy_train.iloc[:, 1]
        self.X_test = pd.DataFrame(Xy_test.iloc[:, 0])
        self.y_test = Xy_test.iloc[:, 1]
        self.X_test_nan = self.X_test.copy()
        self.X_test_nan.iloc[0] = np.nan
        self.y_test_nan = self.y_test.copy()
        self.y_test_nan.iloc[0] = np.nan

    def test_params(self):
        """
        Test that appropriate errors are raised when invalid parameters are entered
        """
        # estimator should inherit from BaseRegressionSystem and should already be trained
        self.assertRaises(
            TypeError,
            partial(neg_mean_abs_corr, X_test=self.X_test, y_test=self.y_test),
            estimator=1,
        )
        self.assertRaises(
            TypeError,
            partial(neg_mean_abs_corr, X_test=self.X_test, y_test=self.y_test),
            estimator="string",
        )
        self.assertRaises(
            TypeError,
            partial(neg_mean_abs_corr, X_test=self.X_test, y_test=self.y_test),
            estimator=LinearRegression(),
        )
        self.assertRaises(
            ValueError,
            partial(neg_mean_abs_corr, X_test=self.X_test, y_test=self.y_test),
            estimator=LinearRegressionSystem(),
        )
        self.assertRaises(
            ValueError,
            partial(neg_mean_abs_corr, X_test=self.X_test, y_test=self.y_test),
            estimator=LADRegressionSystem(),
        )
        self.assertRaises(
            ValueError,
            partial(neg_mean_abs_corr, X_test=self.X_test, y_test=self.y_test),
            estimator=RidgeRegressionSystem(),
        )

        # X_test should be a pandas dataframe with 1 column, multi-indexed by cross-section
        # and date
        model = LinearRegressionSystem().fit(X=self.X_train, y=self.y_train)
        self.assertRaises(
            TypeError, neg_mean_abs_corr, estimator=model, X_test=1, y_test=self.y_test
        )
        self.assertRaises(
            TypeError,
            neg_mean_abs_corr,
            estimator=model,
            X_test="string",
            y_test=self.y_test,
        )
        self.assertRaises(
            ValueError,
            neg_mean_abs_corr,
            estimator=model,
            X_test=self.X_test.reset_index(),
            y_test=self.y_test,
        )
        self.assertRaises(
            ValueError,
            neg_mean_abs_corr,
            estimator=model,
            X_test=self.X_test_nan,
            y_test=self.y_test,
        )

        # y_test should be a pandas series, multi-indexed by cross-section and date
        self.assertRaises(
            TypeError, neg_mean_abs_corr, estimator=model, X_test=self.X_test, y_test=1
        )
        self.assertRaises(
            TypeError,
            neg_mean_abs_corr,
            estimator=model,
            X_test=self.X_test,
            y_test="string",
        )
        self.assertRaises(
            TypeError,
            neg_mean_abs_corr,
            estimator=model,
            X_test=self.X_test,
            y_test=self.y_test.reset_index(),
        )
        self.assertRaises(
            ValueError,
            neg_mean_abs_corr,
            estimator=model,
            X_test=self.X_test,
            y_test=pd.Series(index=self.X_test.index, data="hello world"),
        )
        self.assertRaises(
            ValueError,
            neg_mean_abs_corr,
            estimator=model,
            X_test=self.X_test,
            y_test=self.y_test_nan,
        )

    @parameterized.expand(["pearson", "spearman", "kendall"])
    def test_valid(self, correlation_type):
        """
        For each correlation type, test the scorer calculates the negative mean
        absolute correlation between hedged returns and market returns
        over cross-sections correctly.
        """
        model = LinearRegressionSystem().fit(X=self.X_train, y=self.y_train)
        actual_results = neg_mean_abs_corr(
            model, self.X_test, self.y_test, correlation_type=correlation_type
        )

        # Produce results from scratch and compare
        cross_sections = self.X_train.index.unique(level=0)
        corr_sum = 0
        for cross_section in cross_sections:
            lr = LinearRegression(fit_intercept=True).fit(
                self.X_train.xs(cross_section), self.y_train.xs(cross_section)
            )
            hedged_returns = (
                self.y_test.xs(cross_section)
                - lr.coef_ * self.X_test.xs(cross_section).iloc[:, 0]
            )
            if correlation_type == "pearson":
                correlation = stats.pearsonr(
                    hedged_returns,
                    self.X_test.xs(cross_section).values.flatten(),
                )[0]
            elif correlation_type == "spearman":
                correlation = stats.spearmanr(
                    hedged_returns,
                    self.X_test.xs(cross_section).values.flatten(),
                )[0]
            elif correlation_type == "kendall":
                correlation = stats.kendalltau(
                    hedged_returns,
                    self.X_test.xs(cross_section).values.flatten(),
                )[0]
            corr_sum += np.abs(correlation)

        self.assertEqual(-corr_sum / len(cross_sections), actual_results)