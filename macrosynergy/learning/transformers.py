














































"""
Collection of custom scikit-learn transformer classes.
"""

import numpy as np
import pandas as pd

import datetime

import scipy.stats as stats

from sklearn.linear_model import Lasso, ElasticNet, Lars
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.exceptions import NotFittedError

# from statsmodels.tools.tools import add_constant

from linearmodels.panel import RandomEffects

from typing import Union, Any, Optional

import warnings

class FeatureAverager(BaseEstimator, TransformerMixin):
    def __init__(self, use_signs: Optional[bool] = False):
        """
        Transformer class to combine features into a benchmark signal by averaging.

        :param <Optional[bool]> use_signs: Boolean to specify whether or not to return the
            signs of the benchmark signal instead of the signal itself. Default is False.
        """
        if not isinstance(use_signs, bool):
            raise TypeError("'use_signs' must be a boolean.")

        self.use_signs = use_signs

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Fit method. Since this transformer is a simple averaging of features,
        no fitting is required.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Any> y: Placeholder for scikit-learn compatibility.
        """

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method to average features into a benchmark signal.
        If use_signs is True, the signs of the benchmark signal are returned.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <pd.DataFrame>: Pandas dataframe of benchmark signal.
        """
        # checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the FeatureAverager must be a pandas dataframe."
                " If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")

        # transform
        signal_df = X.mean(axis=1).to_frame(name="signal")
        if self.use_signs:
            return np.sign(signal_df).astype(int)

        return signal_df


class ZnScoreAverager(BaseEstimator, TransformerMixin):
    def __init__(self, neutral: str = "zero", use_signs: bool = False):
        """
        Transformer class to combine features into a benchmark signal
        according to a specified normalisation.

        :param <str> neutral: Specified neutral value for a normalisation. This can
            take either 'zero' or 'mean'. If the neutral level is zero, each feature
            is standardised by dividing by the mean absolute deviation. If the neutral
            level is mean, each feature is normalised by subtracting the mean and
            dividing by the standard deviation. Any statistics are computed according
            to a point-in-time principle. Default is 'zero'.
        :param <bool> use_signs: Boolean to specify whether or not to return the
            signs of the benchmark signal instead of the signal itself. Default is False.
        """
        if not isinstance(neutral, str):
            raise TypeError("'neutral' must be a string.")

        if neutral not in ["zero", "mean"]:
            raise ValueError("neutral must be either 'zero' or 'mean'.")

        if not isinstance(use_signs, bool):
            raise TypeError("'use_signs' must be a boolean.")

        self.neutral = neutral
        self.use_signs = use_signs

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Fit method to extract relevant standardisation/normalisation statistics from a
        training set so that PiT statistics can be computed in the transform method for
        a hold-out set.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Any> y: Placeholder for scikit-learn compatibility.
        """
        # checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the ZnScoreAverager must be a pandas "
                "dataframe. If used as part of an sklearn pipeline, ensure that previous "
                "steps return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")

        # fit
        self.training_n: int = len(X)

        if self.neutral == "mean":
            # calculate the mean and sum of squares of each feature
            means: pd.Series = X.mean(axis=0)
            sum_squares: pd.Series = np.sum(np.square(X), axis=0)
            self.training_means: pd.Series = means
            self.training_sum_squares: pd.Series = sum_squares
        else:
            # calculate the mean absolute deviation of each feature
            mads: pd.Series = np.mean(np.abs(X), axis=0)
            self.training_mads: pd.Series = mads

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform method to compute an out-of-sample benchmark signal for each unique
        date in the input test dataframe. At a given test time, the relevant statistics
        (implied by choice of neutral value) are calculated using all training information
        and test information until (and including) that test time, since the test time
        denotes the time at which the return was available and the features lag behind
        the returns.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        """
        # checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the ZnScoreAverager must be a pandas "
                "dataframe. If used as part of an sklearn pipeline, ensure that previous "
                "steps return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")

        X = X.sort_index(level="real_date")

        # transform
        signal_df = pd.DataFrame(index=X.index, columns=["signal"], dtype="float")

        if self.neutral == "zero":
            training_mads: pd.Series = self.training_mads
            training_n: int = self.training_n
            X_abs = X.abs()

            # We obtain a vector with the number of elements in each expanding window.
            # The count does not necessarily increase uniformly, since some cross-sections
            # may be missing for some dates.
            expanding_count = self._get_expanding_count(X)

            # We divide the expanding sum by the number of elements in each expanding window.
            X_test_expanding_mads = (
                X_abs.groupby(level="real_date").sum().expanding().sum()
                / expanding_count
            )
            n_total = training_n + expanding_count

            X_expanding_mads: pd.DataFrame = (
                (training_n) * training_mads + (expanding_count) * X_test_expanding_mads
            ) / n_total

            standardised_X: pd.DataFrame = (X / X_expanding_mads).fillna(0)

        elif self.neutral == "mean":
            training_means: pd.Series = self.training_means
            training_sum_squares: pd.Series = self.training_sum_squares
            training_n: int = self.training_n

            expanding_count = self._get_expanding_count(X)
            n_total = training_n + expanding_count

            X_square = X**2
            X_test_expanding_sum_squares = (
                X_square.groupby(level="real_date").sum().expanding().sum()
            )
            X_test_expanding_mean = (
                X.groupby(level="real_date").sum().expanding().sum() / expanding_count
            )

            X_expanding_means = (
                (training_n) * training_means
                + (expanding_count) * X_test_expanding_mean
            ) / n_total

            X_expanding_sum_squares = (
                training_sum_squares + X_test_expanding_sum_squares
            )
            comp1 = (X_expanding_sum_squares) / (n_total - 1)
            comp2 = 2 * np.square(X_expanding_means) * (n_total) / (n_total - 1)
            comp3 = (n_total) * np.square(X_expanding_means) / (n_total - 1)
            X_expanding_std: pd.Series = np.sqrt(comp1 - comp2 + comp3)
            standardised_X: pd.DataFrame = (
                (X - X_expanding_means) / X_expanding_std
            ).fillna(0)

        else:
            raise ValueError("neutral must be either 'zero' or 'mean'.")

        benchmark_signal: pd.DataFrame = pd.DataFrame(
            np.mean(standardised_X, axis=1), columns=["signal"], dtype="float"
        )
        signal_df.loc[benchmark_signal.index] = benchmark_signal

        if self.use_signs:
            return np.sign(signal_df).astype(int)

        return signal_df

    def _get_expanding_count(self, X):
        """
        Helper method to get the number of non-NaN values in each expanding window.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <np.ndarray>: Numpy array of expanding counts.
        """
        return X.groupby(level="real_date").count().expanding().sum().to_numpy()



if __name__ == "__main__":
    from macrosynergy.management import make_qdf
    import macrosynergy.management as msm

    np.random.seed(1)

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids2 = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids2.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["CAD"] = ["2013-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["GBP"] = ["2010-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["USD"] = ["2010-01-01", "2020-12-31", 0, 1]

    df_xcats2 = pd.DataFrame(index=xcats, columns=cols)
    df_xcats2.loc["XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats2.loc["CRY"] = ["2010-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats2.loc["GROWTH"] = ["2010-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2010-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)

    dfd2 = dfd2.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd2.drop(columns=["XR"])
    y = dfd2["XR"]

    # selector = LassoSelector(0.2)
    # selector.fit(X, y)
    # print(selector.transform(X).columns)

    # selector = MapSelector(1e-20)
    # selector.fit(X, y)
    # print(selector.transform(X).columns)

    # Split X and y into training and test sets
    X_train, X_test = (
        X[X.index.get_level_values(1) < pd.Timestamp(day=1, month=1, year=2018)],
        X[X.index.get_level_values(1) >= pd.Timestamp(day=1, month=1, year=2018)],
    )
    y_train, y_test = (
        y[y.index.get_level_values(1) < pd.Timestamp(day=1, month=1, year=2018)],
        y[y.index.get_level_values(1) >= pd.Timestamp(day=1, month=1, year=2018)],
    )

    selector = ZnScoreAverager(neutral="zero", use_signs=False)
    selector.fit(X_train, y_train)
    print(selector.transform(X_test))

    selector = ZnScoreAverager(neutral="zero")
    selector.fit(X_train, y_train)
    print(selector.transform(X_test))
