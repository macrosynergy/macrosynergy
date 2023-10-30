"""
Class to provide benchmark signals based on z-scores.

**NOTE: This module is under development, and is not yet ready for production use.**
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

from typing import List


class ZScoreRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn predictor class to determine a benchmark signal based on
    the rolling z-scores of the features.
    """

    def __update_metrics(
        self,
        current_mean: pd.Series,
        current_sum_squares: pd.Series,
        current_n: int,
        new_mean: pd.Series,
        new_sum_squares: pd.Series,
        new_n: int,
    ):
        """
        Helper method to update means and standard deviations in light of the mean,
        standard deviation and sample size of a new, previously unobserved, sample.
        Only the mean, sample size and sum of squares are needed to update the 
        mean and standard deviation.

        :param <pd.Series> current_mean: Pandas series of means of features.
        :param <pd.Series> current_sum_squares: Pandas series of
            the sum of squares of features.
        :param <int> current_n: Number of samples used to compute the current mean
        :param <pd.Series> new_mean: Pandas series of the new means of features.
        :param <pd.Series> new_sum_squares: Pandas series of
            the new sum of squares of features.
        :param <int> new_n: Number of samples in the new sample.

        :return <Tuple[pd.Series, pd.Series, pd.Series, int]>
            updated_mean: Pandas series of updated means of features.
            updated_std: Pandas series of updated standard deviations of features.
            updated_sum_squares: Pandas series of updated sum of squares of features.
            updated_n: Number of samples used to compute the updated mean.
        """
        # TODO: possibly write the updating rule formulae in a comment.
        updated_n: int = current_n + new_n

        # First update the means
        updated_mean = (current_n * current_mean + new_n * new_mean) / (updated_n)

        # Secondly, update the standard deviations.
        # Only the sample sizes, sums of squares and means are needed to do this.
        updated_sum_squares = current_sum_squares + new_sum_squares
        comp1 = (updated_sum_squares) / (updated_n - 1)
        comp2 = 2 * np.square(updated_mean) * (updated_n) / (updated_n - 1)
        comp3 = (updated_n) * np.square(updated_mean) / (updated_n - 1)
        updated_std = np.sqrt(comp1 - comp2 + comp3)

        # Return updated mean, sum of squares and sample size for ease of future 
        # computation. The standard deviation is returned for convenience in computing
        # the z-score in the predict method.
        return updated_mean, updated_std, updated_sum_squares, updated_n

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit method to learn the mean and sum of squares of the features in X, as well as the number of samples.

        :param <pd.DataFrame> X: Pandas dataframe of features multi-indexed by (cross-section, date).
            The dates must be in datetime format.
            The dataframe must be in wide format: each feature is a column.
        :param <pd.Series> y: Pandas series of target variable multi-indexed by (cross-section, date).
            The dates must be in datetime format.

        :return None
        """
        self.mean: pd.Series = np.mean(X, axis=0)
        self.sum_squares: pd.Series = np.sum(np.square(X), axis=0)
        self.n: int = len(X)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict method to compute an out-of-sample benchmark signal for each unique date in the input test dataframe.
        At a given test time, the means and standard deviations for each feature are calculated over all training dates
        and all test dates prior (and including) the concerned test date. This is dne in an online fashion.
        The benchmark signal for that test time is computed as the mean of the z-scores across features using the
        previously calculated means and standard deviations.

        :param <pd.DataFrame> X: Pandas dataframe of features multi-indexed by (cross-section, date).
            The dates must be in datetime format.
            The dataframe must be in wide format: each feature is a column.
            This only makes sense as a test set, as the benchmark signal is computed out-of-sample.

        :return <pd.Series> signal_df: Pandas series of benchmark signals multi-indexed by (cross-section, date).
        """
        # Create a series to store the benchmark signal
        signal_df = pd.Series(index=X.index, name="signal")

        # Set up all quantities needed to compute the z-scores sequentially
        unique_dates: List[pd.Timestamp] = sorted(X.index.get_level_values(1).unique())

        current_mean: pd.Series = self.mean
        current_sum_squares: pd.Series = self.sum_squares
        current_n: int = self.n

        for date in unique_dates:
            # get the subset of X corresponding to the current date
            X_date: pd.DataFrame = X.loc[(slice(None), date), :]
            new_mean: pd.Series = X_date.mean(axis=0)
            new_n: int = len(X_date)
            new_sum_squares: pd.Series = np.sum(np.square(X_date), axis=0)
            # get the updated mean, sum of squares and num_samples
            updated_mean: pd.Series
            updated_std: pd.Series
            updated_sum_squares: pd.Series
            updated_n: int
            (
                updated_mean,
                updated_std,
                updated_sum_squares,
                updated_n,
            ) = self.__update_metrics(
                current_mean,
                current_sum_squares,
                current_n,
                new_mean,
                new_sum_squares,
                new_n,
            )
            # normalise and take the mean across features
            normalised_X: pd.DataFrame = (X_date - updated_mean) / updated_std
            benchmark_signal: pd.Series = pd.Series(
                np.mean(normalised_X, axis=1), name="signal"
            )
            # store the signal
            signal_df.loc[benchmark_signal.index] = benchmark_signal
            # update metrics for the next iteration
            current_mean = updated_mean
            current_sum_squares = updated_sum_squares
            current_n = updated_n

        return signal_df


if __name__ == "__main__":
    from macrosynergy.management.simulate_quantamental_data import make_qdf
    import macrosynergy.management as msm
    from macrosynergy.learning.panel_timeseries_split import PanelTimeSeriesSplit
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example 1: Unbalanced panel """

    df_cids2 = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids2.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats2 = pd.DataFrame(index=xcats, columns=cols)
    df_xcats2.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats2.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats2.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)

    dfd2 = dfd2.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X2 = dfd2.drop(columns=["XR"])
    y2 = dfd2["XR"]

    # Demonstration of BenchmarkTransformer
    zsregressor = ZScoreRegressor()
    splitter = PanelTimeSeriesSplit(n_splits=4, n_split_method="expanding")
    splitter.visualise_splits(X2, y2)
    # Convert the regression problem into a classification problem
    acc_metric = make_scorer(
        lambda y_true, y_pred: np.mean(np.sign(y_true) == np.sign(y_pred))
    )
    print(cross_val_score(zsregressor, X2, y2, cv=splitter, scoring=acc_metric))
    print("Done")