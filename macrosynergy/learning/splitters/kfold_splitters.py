"""
Panel K-Fold cross-validator classes. 
"""

import numpy as np
import pandas as pd

from macrosynergy.learning.splitters import BasePanelSplit

from typing import List


class ExpandingKFoldPanelSplit(BasePanelSplit):
    """Time-respecting K-Fold cross-validator for panel data.

    Provides train/test indices to split a panel into train/test sets. The unique dates
    in the panel are divided into 'n_splits + 1' sequential and non-overlapping intervals,
    resulting in 'n_splits' pairs of training and test sets. The 'i'th training set is the
    union of the first 'i' intervals, and the 'i'th test set is the 'i+1'th interval.

    Parameters
    ----------
    n_splits : int
        Number of folds. Default is 5. Must be at least 2.

    Notes
    -----
    This splitter can be considered to be a panel data analogue to the `TimeSeriesSplit`
    splitter provided by `scikit-learn`.`
    """

    def __init__(self, n_splits=5):
        # Checks
        self._check_init_params(n_splits)

        # Attributes
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test sets.

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
        self.train_indices = []
        self.test_indices = []

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

    def _check_init_params(self, n_splits: int):
        # n_splits
        if not isinstance(n_splits, int):
            raise TypeError(f"n_splits must be an integer. Got {type(n_splits)}.")
        if n_splits < 2:
            raise ValueError(
                f"Cannot have number of splits less than 2. Got n_splits={n_splits}."
            )

    def _check_split_params(self, X, y, groups):
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


class RollingKFoldPanelSplit(BasePanelSplit):
    """
    Unshuffled K-Fold cross-validator for panel data.

    Provides train/test indices to split a panel into train/test sets. The unique dates
    in the panel are divided into 'n_splits' sequential and non-overlapping intervals of
    equal size, resulting in 'n_splits' pairs of training and test sets. The 'i'th
    test set is the 'i'th interval, and the 'i'th training set is all other intervals.

    Parameters
    ----------
    n_splits : int
        Number of folds. Default is 5. Must be at least 2.

    Notes
    -----
    The splits induced by this cross-validator give the effect of the
    test set "rolling" forward in time.
    """

    def __init__(self, n_splits=5):
        # Checks
        self._check_init_params(n_splits)

        # Attributes
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test sets.

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
        splits = np.array_split(self.unique_dates, self.n_splits)

        for i in range(self.n_splits):
            splits_copy = splits.copy()
            test_split = np.array(splits_copy.pop(i), dtype=np.datetime64)
            train_split = np.concatenate(splits_copy, dtype=np.datetime64)

            train_indices = np.where(dates.isin(train_split))[0]
            test_indices = np.where(dates.isin(test_split))[0]

            yield train_indices, test_indices

    def _check_init_params(self, n_splits: int):
        # n_splits
        if not isinstance(n_splits, int):
            raise TypeError(f"n_splits must be an integer. Got {type(n_splits)}.")
        if n_splits < 2:
            raise ValueError(
                f"Cannot have number of splits less than 2. Got {n_splits}."
            )

    def _check_split_params(self, X, y, groups):
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
        
class RecencyKFoldPanelSplit(BasePanelSplit):
    """
    K-Fold cross-validator for panel data that uses the last 'n_splits' collection of 
    'n_periods' panel dates as independent test sets, with all information prior to each of
    these the respective training sets. 
    """
    def __init__(self, n_splits=5, n_periods=252):
        # Checks
        self._check_init_params(n_splits, n_periods)

        # Attributes
        self.n_splits = n_splits
        self.n_periods = n_periods

    def split(self, X, y, groups=None):
        pass 

    def _check_init_params(self, n_splits: int, n_periods: int):
        # n_splits
        if not isinstance(n_splits, int):
            raise TypeError(f"n_splits must be an integer. Got {type(n_splits)}.")
        if n_splits < 2:
            raise ValueError(
                f"Cannot have number of splits less than 2. Got {n_splits}."
            )
        # n_periods
        if not isinstance(n_periods, int):
            raise TypeError(f"n_periods must be an integer. Got {type(n_periods)}.")
        if n_periods < 1:
            raise ValueError(
                f"Cannot have number of periods less than 1. Got {n_periods}."
            )
