"""
Classes for incremental expanding panel cross-validators. 
"""

import datetime
from typing import Optional, List

import numpy as np
import pandas as pd

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
        is 4. Either start_date or (min_cids, min_periods) must be provided. If
        both are provided, start_date takes precedence.
    min_periods : int
        The minimum number of time periods required for the first training set. Default is
        500. Either start_date or (min_cids, min_periods) must be provided. If
        both are provided, start_date takes precedence.
    min_xcats : Optional[int]
        The minimum number of features required in the first training set. Default is
        None, in which case only samples where all features are present are considered.
    start_date : Optional[str]
        The targeted final date in the initial training set in ISO 8601 format. When
        min_xcats is not None, the largest date out of start_date and the first date
        at which min_xcats features are available is used. Otherwise, start_date is used.
        Default is None. Either start_date or (min_cids, min_periods) must be provided.
        If both are provided, start_date takes precedence.
    max_periods : Optional[int]
        The maximum number of time periods in each training set. If the maximum is
        exceeded, the earliest periods are cut off. This effectively creates rolling
        training sets. Default is None.
    
    Notes
    -----
    The first training set is determined by, firstly, the specification of `start_date` or
    by the parameters `min_cids`, `min_periods`, and secondly, the value of `min_xcats`.
    When `start_date` is provided, the initial training set comprises all available data
    before and including the `start_date`, unless `max_periods` is specified, in which
    case at most the last `max_periods` periods prior to the `start_date` are included. 

    If `start_date` is not provided, the first training set is determined by the
    parameters `min_cids` and `min_periods`. This set comprises at least `min_periods`
    time periods for at least `min_cids` cross-sections.
    
    If `min_xcats` is not None, then the first training/test split is also subject to the
    availability of at least `min_xcats` features. If `min_xcats` is None, the analysis is
    restricted to samples for which all features are available. This specification
    is critical in handling both cross-sectional and category panel imbalance, which are
    common in macroeconomic panels.
    """

    def __init__(
        self,
        train_intervals: int = 21,
        test_size: int = 21,
        min_cids: int = 4,
        min_periods: int = 500,
        min_xcats: Optional[int] = None,
        start_date: Optional[str] = None,
        max_periods: Optional[int] = None,
    ):
        # Checks
        self._check_init_params(
            train_intervals = train_intervals,
            test_size = test_size,
            min_cids = min_cids,
            min_periods = min_periods,
            min_xcats = min_xcats,
            start_date = start_date,
            max_periods = max_periods,
        )

        # Attributes
        self.train_intervals = train_intervals
        self.test_size = test_size
        self.min_cids = min_cids
        self.min_periods = min_periods
        self.min_xcats = min_xcats
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.max_periods = max_periods

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

        # Determine the unique dates in each training split
        train_indices = []
        test_indices = []

        Xy = pd.concat([X, y], axis=1)
        Xy.dropna(subset=[Xy.columns[-1]], inplace=True)

        splits, Xy = self._determine_unique_training_times(Xy)

        # Determine the training and test indices for each split
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

            return train_indices, test_indices
    
    def _determine_unique_training_times(self, Xy):
        """
        Returns the unique dates in each training split.
        
        Parameters
        ----------
        Xy : pd.DataFrame
            Combined pandas dataframe of features and the target variable, multi-indexed
            by (cross-section, date).

        Notes
        -----
        This method is called by self.split(). It also returns other variables needed
        for ensuing components of the self.split() method.
        """
        self.unique_dates:  pd.DatetimeIndex = (
            Xy.index.get_level_values(1).unique().sort_values()
        )
        # First determine the dates for the first training set
        if self.min_xcats:
            # Get first date at which min_xcats features are available.
            X = Xy.drop(columns=[Xy.columns[-1]])
            xcat_first_dates = X.notna().apply(lambda col: X.index.get_level_values(1)[col].min())
            xcat_first_dates_counts = xcat_first_dates.value_counts().sort_index().cumsum()
            first_valid_date = xcat_first_dates_counts[xcat_first_dates_counts >= self.min_xcats].index.min()
            if self.start_date:
                date_last_train = max(self.start_date, first_valid_date)
            else:
                # then min_cids and min_periods are used to determine the first training set.
                init_mask = Xy.groupby(level=1).size().sort_index() >= self.min_cids
                date_first_min_cids: pd.Timestamp = (
                    init_mask[init_mask == True].reset_index().real_date.min()
                )
                date_last_train: pd.Timestamp = self.unique_dates[
                    self.unique_dates.get_loc(date_first_min_cids) + self.min_periods - 1
                ]
        else:
            Xy.dropna(inplace=True)
            if self.start_date:
                date_last_train = self.start_date
            else:
                init_mask = Xy.groupby(level=1).size().sort_index() >= self.min_cids
                date_first_min_cids: pd.Timestamp = (
                    init_mask[init_mask == True].reset_index().real_date.min()
                )
                date_last_train: pd.Timestamp = self.unique_dates[
                    self.unique_dates.get_loc(date_first_min_cids) + self.min_periods - 1
                ]

        unique_dates_train: pd.arrays.DatetimeArray = self.unique_dates[
            self.unique_dates.get_loc(date_last_train) + 1 : -self.test_size
        ]
        self.n_splits: int = int(
            np.ceil(len(unique_dates_train) / self.train_intervals)
        )
        splits: List = np.array_split(unique_dates_train, self.n_splits)
        # Add the first training set to the list of training splits, so that the dates
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

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Calculates and returns the number of splits.

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

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        Xy = pd.concat([X, y], axis=1)
        Xy.dropna(subset=[Xy.columns[-1]], inplace=True)
        self._determine_unique_time_splits(Xy)

        return self.n_splits
    
    def _check_init_params(
        self,
        train_intervals,
        test_size,
        min_cids,
        min_periods,
        min_xcats,
        start_date,
        max_periods,
    ):
        """
        Type and value checks for the class initialisation parameters. 

        Parameters
        ----------
        train_intervals : int
            The number of time periods by which the previous training set is expanded. 
        test_size : int
            The number of time periods forward of each training set to use in the associated
            test set.
        min_cids : int
            The minimum number of cross-sections required for the first training set.
        min_periods : int
            The minimum number of time periods required for the first training set.
        min_xcats : Optional[int]
            The minimum number of features required in the first training set.
        start_date : Optional[str]
            The targeted final date in the initial training set in ISO 8601 format.
        max_periods : Optional[int]
            The maximum number of time periods in each training set.
        """
        # train_intervals 
        if not isinstance(train_intervals, int):
            raise TypeError(
                f"train_intervals must be an integer. Got {type(train_intervals)}."
            )
        if train_intervals < 1:
            raise ValueError(
                f"train_intervals must be an integer greater than 0. Got {train_intervals}."
            )
        
        # test_size
        if not isinstance(test_size, int):
            raise TypeError(f"test_size must be an integer. Got {type(test_size)}.")
        if test_size < 1:
            raise ValueError(
                f"test_size must be an integer greater than 0. Got {test_size}."
            )
        
        # min_cids
        if not isinstance(min_cids, int):
            raise TypeError(f"min_cids must be an integer. Got {type(min_cids)}.")
        if min_cids < 1:
            raise ValueError(
                f"min_cids must be an integer greater than 0. Got {min_cids}."
            )
        
        # min_periods
        if not isinstance(min_periods, int):
            raise TypeError(f"min_periods must be an integer. Got {type(min_periods)}.")
        if min_periods < 1:
            raise ValueError(
                f"min_periods must be an integer greater than 0. Got {min_periods}."
            )
        
        # min_xcats
        if min_xcats is not None and not isinstance(min_xcats, int):
            raise TypeError(
                f"min_xcats must be an integer. Got {type(min_xcats)}."
            )
        if min_xcats is not None and min_xcats < 1:
            raise ValueError(
                f"min_xcats must be an integer greater than 0."
                f" Got {min_xcats}."
            )
        
        # start_date
        if start_date is not None and not isinstance(start_date, str):
            raise TypeError(
                f"start_date must be a string. Got {type(start_date)}."
            )
        if start_date is not None:
            try:
                datetime.datetime.fromisoformat(start_date)
            except ValueError:
                raise ValueError(
                    f"start_date must be in ISO 8601 format. Got {start_date}."
                )
            
        # max_periods
        if max_periods is not None and not isinstance(max_periods, int):
            raise TypeError(
                f"max_periods must be an integer. Got {type(max_periods)}."
            )
        if max_periods is not None and max_periods < 1:
            raise ValueError(
                f"max_periods must be an integer greater than 0. Got {max_periods}."
            )
        
    def _check_split_params(self, X, y, groups):
        """
        Type and value checks for the `split()` method parameters.
        
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
            raise ValueError(f"The dates in X must be datetime objects. Got {X.index.get_level_values(1).dtype} instead.")
        # y
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not pd.api.types.is_datetime64_any_dtype(y.index.get_level_values(1)):
            raise ValueError(f"The dates in y must be datetime objects. Got {y.index.get_level_values(1).dtype} instead.")
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't"
                "match."
            )
        # groups
        if groups is not None:
            raise ValueError("groups is not supported by this splitter.")

class ExpandingFrequencyPanelSplit(BasePanelSplit):
    """
    Walk-forward cross-validator over a panel. 

    Provides train/test indices to split data into train/test sets. The dataset is split
    so that subsequent training sets are expanded by a user-specified frequency to
    incorporate the latest available information. Each training set is followed by a test
    set spanning a user-defined frequency. 

    Parameters
    ----------
    expansion_freq : str
        Frequency of training set expansion. For a given native dataset frequency, the
        training sets expand by the smallest number of dates to cover this frequency.
        Default is "D". Accepted values are "D", "W", "M", "Q" and "Y".
    test_freq : str
        Frequency forward of each training set for the unique dates in each test set to
        cover. Default is "D". Accepted values are "D", "W", "M", "Q" and "Y".
    min_cids : int
        Minimum number of cross-sections required for the initial training set.
        Default is 4. Either start_date or (min_cids, min_periods, min_xcats) must be
        provided. If both are provided, start_date takes precedence.
    min_periods : int
        Minimum number of time periods required for the initial training set. Default is
        500. Either start_date or (min_cids, min_periods, min_xcats) must be provided. If
        both are provided, start_date takes precedence.
    min_xcats : Optional[int]
        Minimum number of features required in the initial training set. Default is None,
        in which case only samples where all features are present are considered. Either
        start_date or (min_cids, min_periods, min_xcats) must be provided. If both are
        provided, start_date takes precedence.
    start_date : Optional[str]
        First rebalancing date in ISO 8601 format. This is the last date of the first
        training set. Default is None. Either start_date or (min_cids, min_periods, min_xcats)
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
    time periods for at least `min_cids` cross-sections. If `min_xcats` is not None, then the
    first training/test split is also subject to the availability of at least `min_xcats`
    features. If `min_xcats` is None, the analysis is restricted to samples for which all 
    features are available. This triple specification is critical in handling both
    cross-sectional and category panel imbalance, which are common in macroeconomic panels.

    This initial training set is immediately adjusted depending on the specified training
    interval frequency. For instance, if the training frequency is "M", the initial
    training set is further expanding so that all samples prior to the end of the month 
    are included. 

    The associated test set immediately follows the adjusted initial training set and spans
    the specified test set frequency forward of its associated training set.
    For instance, if the test frequency is "Q", the available dates that cover the 
    subsequent quarter are grouped together to form the test set.

    Subsequent training sets are created by expanding the previous training set by the 
    smallest number of dates to cover the training frequency. As before, each test set immediately
    follows its associated training set and is determined in the same manner as the initial test set.
    """

    def __init__(
        self,
        expansion_freq: str = "D",
        test_freq: str = "D",
        min_cids: int = 4,
        min_periods: int = 500,
        min_xcats: Optional[int] = None,
        start_date: Optional[str] = None,
        max_periods: Optional[int] = None,
    ):
        # Input checks
        self._check_init_params(
            expansion_freq = expansion_freq,
            test_freq = test_freq,
            min_cids = min_cids,
            min_periods = min_periods,
            min_xcats = min_xcats,
            start_date = start_date,
            max_periods = max_periods,
        )

        # Attributes
        self.expansion_freq = expansion_freq
        self.test_freq = test_freq
        self.min_cids = min_cids
        self.min_periods = min_periods
        self.min_xcats = min_xcats
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.max_periods = max_periods

        self.freq_offsets = {
            "D" : pd.DateOffset(days = 1),
            "W" : pd.DateOffset(weeks = 1),
            "M" : pd.DateOffset(months = 1),
            "Q" : pd.DateOffset(months = 3),
            "Y" : pd.DateOffset(years = 1),
        }

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
        self._check_split_params(X, y)

        train_indices = []
        test_indices = []

        Xy = pd.concat([X, y], axis=1)
        Xy.dropna(subset=[Xy.columns[-1]], inplace=True)

        splits, Xy = self._determine_unique_training_times(Xy)

        train_splits: List[np.ndarray] = [
            splits[0] if not self.max_periods else splits[0][-self.max_periods :]
        ]
        for i in range(1, self.n_splits):
            train_splits.append(np.concatenate([train_splits[i - 1], splits[i]]))

            # Drop beginning of training set if it exceeds max_periods.
            if self.max_periods:
                train_splits[i] = train_splits[i][-self.max_periods :]

        test_offset = self.freq_offsets[self.test_freq]
        for split_idx, split in enumerate(train_splits):
            train_indices = np.where(Xy.index.get_level_values(1).isin(split))[0]
            test_start: pd.Timestamp = self.unique_dates[self.unique_dates.get_loc(split.max()) + 1]
            test_end: pd.Timestamp = test_start + test_offset
            if split_idx == len(train_splits) - 1:
                test_dates = sorted(self.unique_dates[self.unique_dates >= test_start])
            else:
                test_dates = sorted(self.unique_dates[(self.unique_dates >= test_start) & (self.unique_dates < test_end)])
            test_indices = np.where(Xy.index.get_level_values(1).isin(test_dates))[0]

            return train_indices, test_indices

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
    
    def _determine_unique_training_times(self, Xy):
        """
        Returns the unique dates in each training split. This method is called by
        self.split(). It further returns other variables needed for ensuing components of
        the split method.
        """
        self.unique_dates: pd.DatetimeIndex = (
            Xy.index.get_level_values(1).unique().sort_values()
        )
        # First determine the dates for the first training set
        if self.start_date:
            date_last_train = self.start_date
        elif self.min_xcats is None:
            # Then the splits are restricted to samples where all features are available.
            Xy.dropna(inplace=True)
            # Get all samples before the first date in the panel where `min_periods` time
            # periods are available for at least `min_cids` cross-sections. 
            init_mask: pd.Series = Xy.groupby(level=1).size().sort_index() >= self.min_cids
            date_first_min_cids: pd.Timestamp = (
                init_mask[init_mask == True].reset_index().real_date.min()
            )
            date_last_train: pd.Timestamp = self.unique_dates[
                self.unique_dates.get_loc(date_first_min_cids) + self.min_periods - 1
            ]
        else:
            # Then the first training set is dependent on min_cids, min_periods and min_xcats.
            # (1) Find first time at which min_xcats features are available and filter
            # the panel to include only dates after this time.
            X = Xy.drop(columns=[Xy.columns[-1]])
            xcat_first_dates = X.notna().apply(lambda col: X.index.get_level_values(1)[col].min())
            xcat_first_dates_counts = xcat_first_dates.value_counts().sort_index().cumsum()
            first_valid_date = xcat_first_dates_counts[xcat_first_dates_counts >= self.min_xcats].index.min()
            Xy = Xy.loc[Xy.index.get_level_values(1) >= first_valid_date]
            # Get samples prior to the first date in the panel where `min_periods` time
            # periods are available for at least `min_cids` cross-sections.
            init_mask = Xy.groupby(level=1).size().sort_index() >= self.min_cids
            date_first_min_cids: pd.Timestamp = (
                init_mask[init_mask == True].reset_index().real_date.min()
            )
            date_last_train: pd.Timestamp = self.unique_dates[
                self.unique_dates.get_loc(date_first_min_cids) + self.min_periods - 1
            ]

        # (2) Loop through the unique dates in the panel to create the training and test sets
        # We can loop until before the last "test_freq" dates in the panel
        end_date = self.unique_dates[-1] - self.freq_offsets[self.test_freq]
        unique_dates_train: pd.arrays.DatetimeArray = self.unique_dates[
            (self.unique_dates > date_last_train) & (self.unique_dates <= end_date)
        ].sort_values()
        current_date = unique_dates_train[0]
        splits = []

        while current_date < end_date:
            next_date = current_date + self.freq_offsets[self.expansion_freq]
            if next_date > end_date:
                mask = (unique_dates_train >= current_date) & (unique_dates_train <= end_date)
            else:
                mask = (unique_dates_train >= current_date) & (unique_dates_train < next_date)
            split_dates = unique_dates_train[mask]
            if not split_dates.empty:
                splits.append(split_dates)
            current_date = next_date

        self.n_splits = len(splits) + 1

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

        return splits, Xy
    
    def _check_init_params(
        self,    
        expansion_freq,
        test_freq,
        min_cids,
        min_periods,
        min_xcats,
        start_date,
        max_periods,
    ):
        # expansion_freq
        if not isinstance(expansion_freq, str):
            raise TypeError(
                f"expansion_freq must be a string. Got {type(expansion_freq)}."
            )
        if expansion_freq not in ["D", "W", "M", "Q", "Y"]:
            raise ValueError(
                f"expansion_freq must be one of 'D', 'W', 'M', 'Q' or 'Y'."
                f" Got {expansion_freq}."
            )
        
        # test_freq
        if not isinstance(test_freq, str):
            raise TypeError(f"test_freq must be a string. Got {type(test_freq)}.")
        if test_freq not in ["D", "W", "M", "Q", "Y"]:
            raise ValueError(
                f"test_freq must be one of 'D', 'W', 'M', 'Q' or 'Y'."
                f" Got {test_freq}."
            )
        
        # min_cids
        if not isinstance(min_cids, int):
            raise TypeError(f"min_cids must be an integer. Got {type(min_cids)}.")
        if min_cids < 1:
            raise ValueError(
                f"min_cids must be an integer greater than 0. Got {min_cids}."
            )
        
        # min_periods
        if not isinstance(min_periods, int):
            raise TypeError(f"min_periods must be an integer. Got {type(min_periods)}.")
        if min_periods < 1:
            raise ValueError(
                f"min_periods must be an integer greater than 0. Got {min_periods}."
            )
        
        # min_xcats
        if min_xcats is not None and not isinstance(min_xcats, int):
            raise TypeError(
                f"min_xcats must be an integer. Got {type(min_xcats)}."
            )
        if min_xcats is not None and min_xcats < 1:
            raise ValueError(
                f"min_xcats must be an integer greater than 0."
                f" Got {min_xcats}."
            )
        
        # start_date
        if start_date is not None and not isinstance(start_date, str):
            raise TypeError(
                f"start_date must be a string. Got {type(start_date)}."
            )
        if start_date is not None:
            try:
                datetime.datetime.fromisoformat(start_date)
            except ValueError:
                raise ValueError(
                    f"start_date must be in ISO 8601 format. Got {start_date}."
                )
            
        # max_periods
        if max_periods is not None and not isinstance(max_periods, int):
            raise TypeError(
                f"max_periods must be an integer. Got {type(max_periods)}."
            )
        if max_periods is not None and max_periods < 1:
            raise ValueError(
                f"max_periods must be an integer greater than 0. Got {max_periods}."
            )

    def _check_split_params(self, X, y):
        pass
        
if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    # """Example 1: Unbalanced panel """

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
    df_xcats2.loc["GROWTH"] = ["2001-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)

    dfd2 = dfd2.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X2 = dfd2.drop(columns=["XR"])
    y2 = dfd2["XR"]

    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12, test_size=1, start_date = "2010-01-01",
    )
    splitter.split(X2, y2)