"""
Panel K-Fold cross-validator classes. 
"""

import numpy as np
import pandas as pd

from macrosynergy.learning.splitters import BasePanelSplit, KFoldPanelSplit

class ExpandingKFoldPanelSplit(KFoldPanelSplit):
    """
    Time-respecting K-Fold cross-validator for panel data.

    Parameters
    ----------
    n_splits : int
        Number of folds i.e. (training set, test set) pairs. Default is 5.
        Must be at least 2.

    Notes
    -----
    This splitter can be considered to be a panel data analogue to the `TimeSeriesSplit`
    splitter provided by `scikit-learn`.
    
    Unique dates in the panel are divided into 'n_splits + 1' sequential and
    non-overlapping intervals, resulting in 'n_splits' pairs of training and test sets.
    The 'i'th training set is the union of the first 'i' intervals, and the 'i'th test set
    is the 'i+1'th interval.
    """
    def split(self, X, y, groups=None):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of features, multi-indexed by (cross-section, date). The
            dates must be in datetime format. Otherwise the dataframe must be in wide
            format: each feature is a column.

        y : Union[pd.DataFrame, pd.Series]
            Pandas dataframe or series of a target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format. If a dataframe
            is provided, the target variable must be the sole column.

        groups : None
            Ignored. Exists for compatibility with scikit-learn.

        Yields
        ------
        train : np.ndarray
            The training set indices for that split.

        test : np.ndarray
            The testing set indices for that split.
        """
        # Checks
        self._check_split_params(X, y, groups)

        # Store necessary quantities
        Xy = pd.concat([X, y], axis=1)
        Xy.dropna(subset=[Xy.columns[-1]], inplace=True)
        dates = Xy.index.get_level_values(1)
        unique_dates = dates.unique().sort_values()

        # Calculate splits
        splits = np.array_split(unique_dates, self.n_splits + 1)

        train_split = np.array([], dtype=np.datetime64)
        for i in range(0, self.n_splits):
            train_split = np.concatenate([train_split, splits[i]])
            train_indices = np.where(dates.isin(train_split))[0]
            test_indices = np.where(dates.isin(splits[i + 1]))[0]

            yield train_indices, test_indices

    def _determine_splits(self, unique_dates, n_splits):
        return np.array_split(unique_dates, n_splits + 1)

    def _get_split_indicies(self, idx, split, splits, Xy, dates, unique_dates):
        train_split = np.concatenate(splits[:idx+1])
        train_indices = np.where(dates.isin(train_split))[0]
        test_indices = np.where(dates.isin(split))[0]

        return train_indices, test_indices
    
class RollingKFoldPanelSplit(KFoldPanelSplit):
    """
    Unshuffled K-Fold cross-validator for panel data.

    Parameters
    ----------
    n_splits : int
        Number of folds. Default is 5. Must be at least 2.

    Notes
    -----
    This splitter can be considered to be a panel data analogue to the `KFold` splitter
    provided by `scikit-learn`, with `shuffle=False` and with splits determined on the 
    time dimension.

    Unique dates in the panel are divided into 'n_splits' sequential and non-overlapping
    intervals of equal size, resulting in 'n_splits' pairs of training and test sets.
    The 'i'th test set is the 'i'th interval, and the 'i'th training set is all other
    intervals.
    """

    def __init__(self, n_splits=5):
        super().__init__(n_splits=n_splits, min_nsplits=2)

    def _determine_splits(self, unique_dates, n_splits):
        """
        Determine panel splits based on the sorted collection of unique dates and the
        number of splits specified by the user.

        Parameters
        ----------
        unique_dates : pd.DatetimeIndex
            Sorted collection of unique dates in the panel.
        n_splits : int
            Number of splits to generate.

        Returns
        -------
        splits : list of np.ndarray
            List of numpy arrays denoting dates in each split.
        """
        splits = np.array_split(unique_dates, n_splits)
        return splits

    def _get_split_indicies(self, idx, split, splits, Xy, dates, unique_dates):
        """
        Determine the training and test set indices for a given split.

        Parameters
        ----------
        idx : int
            Index of the current split.
        split : np.ndarray
            Numpy array of dates to induce the split.
        splits : list of np.ndarray
            List of numpy arrays denoting dates in each split.
        Xy : pd.DataFrame
            Combined dataframe of the features and the target variable.
        dates : pd.DatetimeIndex
            DatetimeIndex of all dates in the panel.
        unique_dates : pd.DatetimeIndex
            Sorted collection of unique dates in the panel.

        Returns
        -------
        train : np.ndarray
            The training set indices for that split.
        test : np.ndarray
            The testing set indices for that split.
        """
        # Copy splits and remove the test split
        splits_copy = splits.copy()
        test_split = np.array(splits_copy.pop(idx), dtype=np.datetime64)
        train_split = np.concatenate(splits_copy, dtype=np.datetime64)

        train_indices = np.where(dates.isin(train_split))[0]
        test_indices = np.where(dates.isin(test_split))[0]

        return train_indices, test_indices
        
class RecencyKFoldPanelSplit(KFoldPanelSplit):
    """
    Time-respecting K-Fold panel cross-validator that creates training and test sets based
    on the most recent samples in the panel.
    
    Parameters
    ----------
    n_splits : int
        Number of folds i.e. (training set, test set) pairs. Default is 5.
        Must be at least 1.
    n_periods : int
        Number of time periods, in units of native dataset frequency, to comprise each 
        test set. Default is 252 (1 year for daily data).

    Notes
    -----
    This splitter is similar to the ExpandingKFoldPanelSplit, except that the sorted 
    unique timestamps are not divided into equal intervals. Instead, the last 
    `n_periods` * `n_splits` timestamps in the panel are divided into `n_splits`
    non-overlapping intervals, each of which is used as a test set. The corresponding
    training set is comprised of all samples with timestamps earlier than its test set.
    Consequently, this is a K-Fold walk-forward cross-validator, but with test folds
    concentrated on the most recent information. 
    """
    def __init__(self, n_splits=5, n_periods=252):
        # Checks
        if not isinstance(n_splits, int):
            raise TypeError(f"n_splits must be an integer. Got {type(n_splits)}.")
        if n_splits < 1:
            raise ValueError(
                f"Cannot have number of splits less than 1. Got {n_splits}."
            )
        if not isinstance(n_periods, int):
            raise TypeError(f"n_periods must be an integer. Got {type(n_periods)}.")
        if n_periods < 1:
            raise ValueError(
                f"Cannot have number of periods less than 1. Got {n_periods}."
            )

        # Attributes
        self.n_splits = n_splits
        self.n_periods = n_periods

    def split(self, X, y, groups=None):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of features, multi-indexed by (cross-section, date). The
            dates must be in datetime format. Otherwise the dataframe must be in wide
            format: each feature is a column.

        y : Union[pd.DataFrame, pd.Series]
            Pandas dataframe or series of a target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format. If a dataframe
            is provided, the target variable must be the sole column.

        groups : None
            Ignored. Exists for compatibility with scikit-learn.

        Yields
        ------
        train : np.ndarray
            The training set indices for that split.

        test : np.ndarray
            The testing set indices for that split.
        """
        # Checks
        self._check_split_params(X, y, groups)

        # Store necessary quantities
        Xy = pd.concat([X, y], axis=1)
        Xy.dropna(subset=[Xy.columns[-1]], inplace=True)
        dates = Xy.index.get_level_values(1)
        unique_dates = dates.unique().sort_values()

        # Calculate splits
        splits = np.array_split(unique_dates[-self.n_splits * self.n_periods:], self.n_splits)

        for i in range(self.n_splits):
            test_split = np.array(splits[i], dtype=np.datetime64)
            train_split = unique_dates[unique_dates < test_split[0]]

            train_indices = np.where(dates.isin(train_split))[0]
            test_indices = np.where(dates.isin(test_split))[0]

            yield train_indices, test_indices

    def _check_split_params(self, X, y, groups):
        """
        Splitter input checks.

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of features, multi-indexed by (cross-section, date). The
            dates must be in datetime format. Otherwise the dataframe must be in wide
            format: each feature is a column.

        y : Union[pd.DataFrame, pd.Series]
            Pandas dataframe or series of a target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format. If a dataframe
            is provided, the target variable must be the sole column.

        groups : None
            Ignored. Exists for compatibility with scikit-learn.
        """
        # X
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not pd.api.types.is_datetime64_any_dtype(X.index.get_level_values(1)):
            raise ValueError(
                f"The dates in X must be datetime objects. Got {X.index.get_level_values(1).dtype} instead."
            )
        # y
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not pd.api.types.is_datetime64_any_dtype(y.index.get_level_values(1)):
            raise ValueError(
                f"The dates in y must be datetime objects. Got {y.index.get_level_values(1).dtype} instead."
            )
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't"
                "match."
            )
        # groups
        if groups is not None:
            raise ValueError("groups is not supported by this splitter.")
        
    def _combine_Xy(self, X, y):
        """
        Combine the features and target variable into a single dataframe for subsequent
        processing.

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of features/quantamental indicators, multi-indexed by
            (cross-section, date). The dates must be in datetime format. The
            dataframe must be in wide format: each feature is a column.
        y : pd.DataFrame
            Pandas dataframe of target variable, multi-indexed by (cross-section, date).
            The dates must be in datetime format.

        Returns
        -------
        Xy : pd.DataFrame
            Combined dataframe of the features and the target variable.
        """
        # Drop samples with a missing target variable
        Xy = pd.concat([X, y], axis=1)
        Xy.dropna(inplace=True)

        return Xy
        
if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm

    # Example dataset
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd = msm.reduce_df(df=dfd, cids=cids, xcats=xcats, blacklist=black)

    dfd = dfd.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd.drop(columns=["XR"])
    y = dfd["XR"]

    # ExpandingKFoldPanelSplit
    splitter = ExpandingKFoldPanelSplit(n_splits=5)
    splitter.visualise_splits(X, y)

    # RollingKFoldPanelSplit
    splitter = RollingKFoldPanelSplit(n_splits=5)
    splitter.visualise_splits(X, y)

    # RecencyKFoldPanelSplit
    splitter = RecencyKFoldPanelSplit(n_splits=4, n_periods=21 * 3)
    splitter.visualise_splits(X, y)