"""
Classes for incremental expanding panel cross-validators. 
"""

import datetime
from typing import Optional, List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    cross_validate,
)
from sklearn.linear_model import Lasso, LinearRegression

from macrosynergy.learning.splitters import BasePanelSplit

class ExpandingIncrementPanelSplit(BasePanelSplit):
    """
    Walk-forward cross-validator over a panel. 

    Provides train/test indices to split data into train/test sets. The dataset is split
    so that subsequent training sets are expanded by a fixed number of time periods to
    incorporate the latest available information. Each training set is followed by a test
    set of fixed length. 

    Parameters
    ----------
    train_intervals : int
        The number of time periods by which the previous training set is expanded. 
        Default is 21.
    test_size : int
        The number of time periods forward of each training set to use in the associated
        test set. Default is 21. 
    min_cids : int
        The minimum number of cross-sections required for the first training set. Default
        is 4. Either start_date or (min_cids, min_periods, min_xcats) must be provided. If
        both are provided, start_date takes precedence.
    min_periods : int
        The minimum number of time periods required for the first training set. Default is
        500. Either start_date or (min_cids, min_periods, min_xcats) must be provided. If
        both are provided, start_date takes precedence.
    min_xcats : Union[int, str]
        The minimum number of features required in the first training set. Default is
        "ALL". Either start_date or (min_cids, min_periods, min_xcats) must be provided. If
        both are provided, start_date takes precedence.
    start_date : Optional[str]
        The first rebalancing date in ISO 8601 format. This is the first date of the first
        test set. Default is None. Either start_date or (min_cids, min_periods, min_xcats)
        must be provided. If both are provided, start_date takes precedence.
    max_periods : Optional[int]
        The maximum number of time periods in each training set. If the maximum is
        exceeded, the earliest periods are cut off. This effectively creates rolling
        training sets. Default is None.
    
    Notes
    -----
    The first training set is either determined by the specification of `start_date` or by
    the parameters `min_cids`, `min_periods` and `min_xcats` collectively. When
    `start_date` is provided, the initial training set comprises all available data prior
    to the `start_date`, unless `max_periods` is specified, in which case at most the last
    `max_periods` periods prior to the `start_date` are included. 

    If `start_date` is not provided, the first training set is determined by the parameters
    `min_cids`, `min_periods` and `min_xcats`. This set comprises at least `min_periods`
    time periods for at least `min_cids` cross-sections, subject to the availability of at
    least `min_xcats` features. This triple specification is critical in handling both
    cross-sectional and category panel imbalance, which are common in macroeconomic panels.
    """

    def __init__(
        self,
        train_intervals: Optional[int] = 21,
        min_cids: Optional[int] = 4,
        min_periods: Optional[int] = 500,
        test_size: int = 21,
        max_periods: Optional[int] = None,
    ):
        if type(train_intervals) != int:
            raise TypeError(
                f"train_intervals must be an integer. Got {type(train_intervals)}."
            )
        if type(min_cids) != int:
            raise TypeError(f"min_cids must be an integer. Got {type(min_cids)}.")
        if type(min_periods) != int:
            raise TypeError(f"min_periods must be an integer. Got {type(min_periods)}.")
        if type(test_size) != int:
            raise TypeError(f"test_size must be an integer. Got {type(test_size)}.")
        if max_periods is not None:
            if type(max_periods) != int:
                raise TypeError(
                    f"max_periods must be an integer. Got {type(max_periods)}."
                )
        
        if min_cids < 1:
            raise ValueError(
                f"min_cids must be an integer greater than 0. Got {min_cids}."
            )
        if min_periods < 1:
            raise ValueError(
                f"min_periods must be an integer greater than 0. Got {min_periods}."
            )
        if test_size < 1:
            raise ValueError(
                f"test_size must be an integer greater than 0. Got {test_size}."
            )
        if max_periods is not None:
            if max_periods < 1:
                raise ValueError(
                    f"max_periods must be an integer greater than 0. Got {max_periods}."
                )

        self.train_intervals: int = train_intervals
        self.min_cids: int = min_cids
        self.min_periods: int = min_periods
        self.test_size: int = test_size
        self.max_periods: int = max_periods

    def _determine_unique_time_splits(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[List[pd.DatetimeIndex], pd.DataFrame, int]:
        """
        Private helper method to determine the unique dates in each training split. This
        method is called by self.split(). It further returns other variables needed for
        ensuing components of the split method.

        :param X: Pandas dataframe of features
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            The dataframe must be in wide format: each feature is a column.
        :param y: Pandas dataframe of the target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.

        :return <Tuple[List[pd.DatetimeIndex],pd.DataFrame]>
            (train_splits_basic, Xy):
            Tuple comprising the unique dates in each training split and the concatenated
            dataframe of X and y.
        """

        self._validate_Xy(X, y)

        Xy: pd.DataFrame = pd.concat([X, y], axis=1)
        Xy = Xy.dropna()
        self.unique_dates: pd.DatetimeIndex = (
            Xy.index.get_level_values(1).unique().sort_values()
        )
        # First determine the dates for the first training set, determined by 'min_cids'
        # and 'min_periods'. (a) obtain a boolean mask of dates for which at least
        # 'min_cids' cross-sections are available over the panel
        init_mask: pd.Series = Xy.groupby(level=1).size().sort_index() >= self.min_cids
        # (b) obtain the earliest date for which the mask is true i.e. the earliest date
        # with 'min_cids' cross-sections available
        date_first_min_cids: pd.Timestamp = (
            init_mask[init_mask == True].reset_index().real_date.min()
        )
        # (c) find the last valid date in the first training set, which is the date
        # 'min_periods' - 1 after date_first_min_cids
        date_last_train: pd.Timestamp = self.unique_dates[
            self.unique_dates.get_loc(date_first_min_cids) + self.min_periods - 1
        ]
        # (d) determine the unique dates in the training sets after the first.
        # This is done by collecting all dates in the panel after the last date in the
        # first training set and before the last 'self.test_size' dates, calculating the
        # number of splits ('self.n_splits') required to split these dates into distinct
        # training intervals of length 'self.train_intervals' (where possible) and finally
        # splitting the mentioned dates into 'self.n_splits' splits (post the first split,
        # determined by 'self.min_cids' and 'self.min_periods').
        unique_dates_train: pd.arrays.DatetimeArray = self.unique_dates[
            self.unique_dates.get_loc(date_last_train) + 1 : -self.test_size
        ]
        self.n_splits: int = int(
            np.ceil(len(unique_dates_train) / self.train_intervals)
        )
        splits: List = np.array_split(unique_dates_train, self.n_splits)
        # (e) add the first training set to the list of training splits, so that the dates
        # that constitute each training split are together.
        splits.insert(
            0,
            Xy.index.get_level_values(1)[
                Xy.index.get_level_values(1) <= date_last_train
            ]
            .unique()
            .sort_values(),
        )

        self.n_splits += 1

        return splits, Xy

    def split(
        self, X: pd.DataFrame, y: pd.DataFrame, groups=None
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Method that produces pairs of training and test indices as intended by the
        ExpandingIncrementPanelSplit class. Wide format Pandas (panel) dataframes are
        expected, multi-indexed by cross-section and date. It is recommended for the
        features to lag behind the associated targets by a single native frequency unit.

        :param <pd.DataFrame> X: Pandas dataframe of features,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of the target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        :param <int> groups: Always ignored, exists for compatibility with scikit-learn.

        :return <Iterable[Tuple[np.ndarray[int],np.ndarray[int]]]> splits: Iterable of
            (train,test) indices.
        """
        train_indices: List[int] = []
        test_indices: List[int] = []

        splits, Xy = self._determine_unique_time_splits(X, y)

        train_splits: List[np.ndarray] = [
            splits[0] if not self.max_periods else splits[0][-self.max_periods :]
        ]
        for i in range(1, self.n_splits):
            train_splits.append(np.concatenate([train_splits[i - 1], splits[i]]))

            # Drop beginning of training set if it exceeds max_periods.
            if self.max_periods:
                train_splits[i] = train_splits[i][-self.max_periods :]

        for split in train_splits:
            train_indices: List[int] = np.where(
                Xy.index.get_level_values(1).isin(split)
            )[0]
            test_start: int = self.unique_dates.get_loc(split.max()) + 1
            test_indices: List[int] = np.where(
                Xy.index.get_level_values(1).isin(
                    self.unique_dates[test_start : test_start + self.test_size]
                )
            )[0]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Calculates and returns the number of splits.

        :param <pd.DataFrame> X: Pandas dataframe of features,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of the target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        :param <pd.DataFrame> groups: Always ignored, exists for compatibility.

        :return <int> n_splits: Returns the number of splits.
        """
        self._determine_unique_time_splits(X, y)

        return self.n_splits