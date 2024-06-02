"""
Tools to produce, visualise and use walk-forward validation splits across panels.
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


class BasePanelSplit(BaseCrossValidator):
    """
    Base class for the production of paired training and test splits for panel data.
    All children classes possess the following methods: get_n_splits and visualise_splits.
    The method 'get_n_splits' is required so that our panel splitters  can inherit from
    sklearn's BaseCrossValidator class, allowing for seamless integration with sklearn's
    API. The method 'visualise_splits' is a convenience method for visualising the splits
    produced by each child splitter, giving the user confidence in the splits produced for
    their use case.
    """

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splits in the cross-validator.

        :param <pd.DataFrame> X: Always ignored, exists for compatibility with
            scikit-learn.
        :param <pd.DataFrame> y: Always ignored, exists for compatibility with
            scikit-learn.
        :param <pd.DataFrame> groups: Always ignored, exists for compatibility with
            scikit-learn.

        :return <int> n_splits: Returns the number of splits.
        """
        return self.n_splits

    def _validate_Xy(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Private helper method to validate the input dataframes X and y.

        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        """
        # Check that X and y are multi-indexed
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        # Check the inner multi-index levels are datetime indices
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of y must be datetime.date.")
        # Check that X and y are indexed in the same order
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't"
                "match."
            )

    def _calculate_xranges(
        self,
        cs_dates: pd.DatetimeIndex,
        real_dates: pd.DatetimeIndex,
        freq_offset: pd.DateOffset,
    ) -> List[Tuple[pd.Timestamp, pd.Timedelta]]:
        """
        Private helper method to determine the ranges of contiguous dates in each training
        and test set, for use in visualisation.

        :param <pd.DatetimeIndex> cs_dates: DatetimeIndex of dates in a set for a given
            cross-section.
        :param <pd.DatetimeIndex> real_dates: DatetimeIndex of all dates in the panel.
        :param <pd.DateOffset> freq_offset: DateOffset object representing the frequency
            of the dates in the panel.

        :return <List[Tuple[pd.Timestamp,pd.Timedelta]]> xranges: list of tuples of the
            form (start date, length of contiguous dates).
        """

        xranges: List[Tuple[pd.Timestamp, pd.Timedelta]] = []
        if len(cs_dates) == 0:
            return xranges

        filtered_real_dates: pd.DatetimeIndex = real_dates[
            (real_dates >= cs_dates.min()) & (real_dates <= cs_dates.max())
        ]
        difference: pd.DatetimeIndex = filtered_real_dates.difference(cs_dates)

        # A single contiguous range of dates.
        if len(difference) == 0:
            xranges.append(
                (cs_dates.min(), cs_dates.max() + freq_offset - cs_dates.min())
            )
            return xranges

        # Multiple contiguous ranges of dates.
        else:
            while len(difference) > 0:
                xranges.append((cs_dates.min(), difference.min() - cs_dates.min()))
                cs_dates = cs_dates[(cs_dates >= difference.min())]
                difference = difference[(difference >= cs_dates.min())]

            xranges.append(
                (cs_dates.min(), cs_dates.max() + freq_offset - cs_dates.min())
            )
            return xranges

    def visualise_splits(
        self, X: pd.DataFrame, y: pd.DataFrame, figsize: Tuple[int, int] = (20, 5)
    ) -> None:
        """
        Method to visualise the splits created according to the parameters specified in
        the constructor.

        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        :param <Tuple[int,int]> figsize: tuple of integers specifying the splitter
            visualisation figure size.

        :return None
        """
        sns.set_theme(style="whitegrid", palette="colorblind")
        Xy: pd.DataFrame = pd.concat([X, y], axis=1).dropna()
        cross_sections: np.ndarray[str] = np.array(
            sorted(Xy.index.get_level_values(0).unique())
        )
        real_dates = Xy.index.get_level_values(1).unique().sort_values()

        freq_est = pd.infer_freq(real_dates)
        if freq_est is None:
            freq_est = real_dates.to_series().diff().min()
        freq_offset = pd.tseries.frequencies.to_offset(freq_est)

        splits: List[Tuple[np.ndarray[int], np.ndarray[int]]] = list(self.split(X, y))

        split_idxs: List[int] = (
            [0, len(splits) // 4, len(splits) // 2, 3 * len(splits) // 4, -1]
            if self.n_splits > 5
            else [i for i in range(self.n_splits)]
        )
        split_titles: List[str] = (
            [
                "Initial split",
                "Quarter progress",
                "Halfway progress",
                "Three quarters progress",
                "Final split",
            ]
            if self.n_splits > 5
            else [f"Split {i+1}" for i in range(self.n_splits)]
        )

        fig, ax = plt.subplots(
            nrows=len(Xy.index.get_level_values(0).unique()),
            ncols=min(self.n_splits, 5),
            figsize=figsize,
        )

        operations = []

        for cs_idx, cs in enumerate(cross_sections):
            for idx, split_idx in enumerate(split_idxs):
                # Get the dates in the training and test sets for the given cross-section.
                cs_train_dates: pd.DatetimeIndex = Xy.iloc[splits[split_idx][0]][
                    Xy.iloc[splits[split_idx][0]].index.get_level_values(0) == cs
                ].index.get_level_values(1)
                cs_test_dates: pd.DatetimeIndex = Xy.iloc[splits[split_idx][1]][
                    Xy.iloc[splits[split_idx][1]].index.get_level_values(0) == cs
                ].index.get_level_values(1)

                xranges_train: List[Tuple[pd.Timestamp, pd.Timedelta]] = (
                    self._calculate_xranges(cs_train_dates, real_dates, freq_offset)
                )
                xranges_test: List[Tuple[pd.Timestamp, pd.Timedelta]] = (
                    self._calculate_xranges(cs_test_dates, real_dates, freq_offset)
                )

                operations.append((cs_idx, idx, xranges_train, "royalblue", "Train"))
                operations.append((cs_idx, idx, xranges_test, "lightcoral", "Test"))

        # Calculate the difference between final two dates.
        # This will be added to the x-axis limits to ensure that the final split is
        # visible.
        last_date = real_dates[-1]  # Get the last date
        second_to_last_date = real_dates[-2]  # Get the second-to-last date
        difference = last_date - second_to_last_date

        # Finally, apply all your operations on the ax object.
        for cs_idx, idx, xranges, color, label in operations:
            if len(cross_sections) == 1:
                ax[idx].broken_barh(
                    xranges, (-0.4, 0.8), facecolors=color, label=label
                )
                ax[idx].set_xlim(real_dates.min(), real_dates.max() + difference)
                ax[idx].set_yticks([0])
                ax[idx].set_yticklabels([cross_sections[0]])
                ax[idx].tick_params(axis="x", rotation=90)
                ax[idx].set_title(f"{split_titles[idx]}")

            else:
                ax[cs_idx, idx].broken_barh(
                    xranges, (-0.4, 0.8), facecolors=color, label=label
                )
                ax[cs_idx, idx].set_xlim(real_dates.min(), real_dates.max() + difference)
                ax[cs_idx, idx].set_yticks([0])
                ax[cs_idx, idx].set_yticklabels([cross_sections[cs_idx]])
                ax[cs_idx, idx].tick_params(axis="x", rotation=90)

                # Ensure only the last row has x-axis labels.
                if cs_idx == len(ax) - 1:
                    ax[cs_idx, idx].tick_params(axis="x", rotation=90)
                else:
                    ax[cs_idx, idx].tick_params(axis="x", labelbottom=False)

                if cs_idx == 0:
                    ax[cs_idx, idx].set_title(f"{split_titles[idx]}")

        plt.suptitle(
            f"Training and test set pairs, number of training sets={self.n_splits}"
        )
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.show()


class ExpandingKFoldPanelSplit(BasePanelSplit):
    """
    Class for the production of paired training and test splits, created over a panel
    of countries. ExpandingKFoldPanelSplit operates similarly to sklearn's
    `TimeSeriesSplit` class, but is designed to handle panels of data, as opposed to
    single time series'. To create the splits, the sorted, unique dates in the panel are
    divided into 'n_splits + 1' sequential and non-overlapping intervals. This results in
    'n_splits' pairs of training and test sets, where the 'i'th training set is the union
    of the first 'i' intervals, and the 'i'th test set is the 'i+1'th interval.

    :param <int> n_splits: number of splits. Must be at least 2.
    """

    def __init__(self, n_splits: int = 5):
        if type(n_splits) != int:
            raise TypeError(f"n_splits must be an integer. Got {type(n_splits)}.")

        if n_splits < 2:
            raise ValueError(
                f"Cannot have number of splits less than 2. Got n_splits={n_splits}."
            )
        self.n_splits = n_splits

    def split(
        self, X: pd.DataFrame, y: pd.DataFrame, groups=None
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Method that produces pairs of training and test indices as intended by the
        ExpandingKFoldPanelSplit class. Wide format Pandas (panel) dataframes are
        expected, multi-indexed by cross-section and date. It is recommended for the
        features to lag behind the associated targets by a single native frequency unit.

        :param <pd.DataFrame> X: Pandas dataframe of features,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of the target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        :param <int> groups: Always ignored, exists for compatibility with scikit-learn.

        :return <Iterable[Tuple[np.ndarray[int],np.ndarray[int]]]> splits:
            Iterable of (train,test) indices.
        """
        self.train_indices: List[int] = []
        self.test_indices: List[int] = []

        self._validate_Xy(X, y)

        # drops row, corresponding with a country & period, if either a feature or the
        # target is missing. Resets index for efficiency later in the code.
        Xy: pd.DataFrame = pd.concat([X, y], axis=1)
        Xy = Xy.dropna()
        self.unique_dates: pd.DatetimeIndex = (
            Xy.index.get_level_values(1).unique().sort_values()
        )

        # split all unique dates into n_folds sub-arrays.
        splits: List[pd.DatetimeIndex] = np.array_split(
            self.unique_dates, self.n_splits + 1
        )

        train_split = np.array([], dtype=np.datetime64)
        for i in range(0, self.n_splits):
            train_split = np.concatenate([train_split, splits[i]])
            train_indices = np.where(Xy.index.get_level_values(1).isin(train_split))[0]

            test_indices = np.where(Xy.index.get_level_values(1).isin(splits[i + 1]))[0]

            yield train_indices, test_indices


class RollingKFoldPanelSplit(BasePanelSplit):
    """
    Class for the production of paired training and test splits, created over a panel
    of countries. RollingKFoldPanelSplit operates similarly to sklearn's `KFold` class
    (without shuffle enabled), but is designed to handle panels of data, as opposed to
    single time series'. To create the splits, the sorted, unique dates in the panel are
    divided into 'n_splits' sequential and non-overlapping intervals. This results in
    'n_splits' pairs of training and test sets, where the 'i'th training set is the 'i'th
    interval, and the 'i'th test set are all other intervals. This gives the effect of the
    test set "rolling" forward in time.

    :param <int> n_splits: number of splits. Must be at least 2.
    """

    def __init__(self, n_splits: int = 5):
        if type(n_splits) != int:
            raise TypeError(f"n_splits must be an integer. Got {type(n_splits)}.")
        if n_splits < 2:
            raise ValueError(
                f"Cannot have number of splits less than 2. Got {n_splits}."
            )
        self.n_splits = n_splits

    def split(
        self, X: pd.DataFrame, y: pd.DataFrame, groups=None
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Method that produces pairs of training and test indices as intended by the
        RollingKFoldPanelSplit class. Wide format Pandas (panel) dataframes are expected,
        multi-indexed by cross-section and date. It is recommended for the features to lag
        behind the associated targets by a single native frequency unit.

        :param <pd.DataFrame> X: Pandas dataframe of features,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of the target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        :param <int> groups: Always ignored, exists for compatibility with scikit-learn.

        :return <Iterable[Tuple[np.ndarray[int],np.ndarray[int]]]> splits: Iterable of
            (train,test) indices.
        """
        self._validate_Xy(X, y)

        # drops row, corresponding with a country & period, if either a feature or the
        # target is missing. Resets index for efficiency later in the code.
        Xy: pd.DataFrame = pd.concat([X, y], axis=1)
        Xy = Xy.dropna()
        self.unique_dates: pd.DatetimeIndex = (
            Xy.index.get_level_values(1).sort_values().unique()
        )

        # split all unique dates into n_splits sub-arrays.
        splits: List[pd.DatetimeIndex] = np.array_split(
            self.unique_dates, self.n_splits
        )

        for i in range(self.n_splits):
            splits_copy = splits.copy()

            # The i-th sub-array is the test set, the rest are the training sets.
            test_split = np.array(splits_copy.pop(i), dtype=np.datetime64)
            train_split = np.concatenate(splits_copy, dtype=np.datetime64)

            train_indices = np.where(Xy.index.get_level_values(1).isin(train_split))[0]
            test_indices = np.where(Xy.index.get_level_values(1).isin(test_split))[0]

            yield train_indices, test_indices

class ExpandingFrequencyPanelSplit(BasePanelSplit):
    """
    Class for the production of paired training and test splits, created over a panel 
    of countries. ExpandingFrequencyPanelSplit differs from ExpandingIncrementPanelSplit by
    specifying the frequency of training set expansion as well as the test set forward
    frequency. 

    As with ExpandingIncrementPanelSplit, the first training set is determined by the parameters
    'min_cids' and 'min_periods', defined below. This set comprises at least 'min_periods' time
    periods for at least 'min_cids' cross-sections. However, this set is subsequently adjusted 
    depending on the training interval frequency. The associated test set immediately follows the
    adjusted initial training set and is determined by grouping all dates within the specified
    test set frequency. For instance, if the test frequency is "Q", the available dates that 
    cover the subsequent quarter are grouped together to form the test set. 

    Subsequent training sets are created by expanding the previous training set by the 
    smallest number of dates to cover the training frequency. As before, each test set immediately
    follows its associated training set and is determined in the same manner as the initial test set.

    We also provide a parameter 'max_periods',
    which allows the user to roll the training set forward as opposed to expanding it.
    If the number of time periods in the training set exceeds 'max_periods', the earliest
    time periods are truncated.

    This splitter can be employed, in addition to standard use, to reflect a pipeline
    through time in a real-world setting. This is especially the case when
    the test set frequency matches the native data set frequencies. 

    .. note::

      This splitter is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle.

    :param <str> expansion_freq: frequency of training set expansion. For a given native
        dataset frequency, the training sets expand by the smallest number of dates to cover
        this frequency. Default is "D".
    :param <str> test_freq: frequency forward of each training set for the unique dates in
        each test set to cover. Default is "D".
    :param <int> min_cids: minimum number of cross-sections required for the initial
        training set. Default is 4.
    :param <int> min_periods: minimum number of time periods required for the initial
        training set. Default is 500.
    :param <int> max_periods: maximum length of each training set in interval training.
        If the maximum is exceeded, the earliest periods are cut off.
        Default is None.
    """

    def __init__(
        self,
        expansion_freq: Optional[str] = "D",
        test_freq: Optional[str] = "D",
        min_cids: Optional[int] = 4,
        min_periods: Optional[int] = 500,
        max_periods: Optional[int] = None,
    ):
        # Input checks
        self._check_init_params(expansion_freq, test_freq, min_cids, min_periods, max_periods)

        self.expansion_freq: str = expansion_freq
        self.test_freq: str = test_freq
        self.min_cids: int = min_cids
        self.min_periods: int = min_periods
        self.max_periods: int = max_periods

        self.freq_offsets = {
            "D" : pd.DateOffset(days = 1),
            "W" : pd.DateOffset(weeks = 1),
            "M" : pd.DateOffset(months = 1),
            "Q" : pd.DateOffset(months = 3),
            "Y" : pd.DateOffset(years = 1),
        }

    def _check_init_params(
        self,    
        expansion_freq,
        test_freq,
        min_cids,
        min_periods,
        max_periods
    ):
        pass

    def _check_split_params(self, X, y):
        pass

    def split(
        self, X: pd.DataFrame, y: pd.DataFrame, groups=None
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Method that produces pairs of training and test indices as intended by the 
        ExpandingFrequencyPanelSplit class. Wide format Pandas (panel) dataframes are
        expected, multi-indexed by cross-section and date. It is recommended for the
        features to lag behind the associated targets by a single native frequency unit.

        :param <pd.DataFrame> X: Pandas dataframe of features,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of the target variable, multi-indexed
            by (cross-section, date). The dates must be in datetime format.
        :param <int> groups: Always ignored, exists for compatibility with scikit-learn.

        :return <Iterable[Tuple[np.ndarray[int],np.ndarray[int]]]> splits: Iterable of
            (train,test) indices.
        """
        self._check_split_params(X, y)

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
    
    def _determine_unique_time_splits(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[List[pd.DatetimeIndex], pd.DataFrame]:
    
        self._validate_Xy(X, y)

        Xy: pd.DataFrame = pd.concat([X, y], axis=1)
        Xy = Xy.dropna()
        self.unique_dates: pd.DatetimeIndex = (
            Xy.index.get_level_values(1).unique().sort_values()
        )

        # (1) First determine the initial training set
        # This is fully defined by finding the last training date in the initial training set
        init_mask: pd.Series = Xy.groupby(level=1).size().sort_index() >= self.min_cids
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

    

class ExpandingIncrementPanelSplit(BasePanelSplit):
    """
    Class for the production of paired training and test splits, created over a panel of
    countries. ExpandingIncrementPanelSplit differs from ExpandingKFoldPanelSplit by
    specifying the structure of an initial training and test set,
    as well as the number of time periods to expand both the initial and subsequent
    training and test sets by. 
    
    This is a flexible alternative to defining the number of splits to make.

    The first training set is determined by the parameters ‘min_cids’ and ‘min_periods’,
    defined below. This set comprises at least ‘min_periods’ time periods for at least
    ‘min_cids’ cross-sections. Its associated test set immediately follows the training
    set, and is of length ‘test_size’. Subsequent training sets are created by expanding
    the previous training set by ‘train_intervals’ time periods, in the native frequency
    of the concerned datasets. As before, each test set immediately follows its associated
    training set, and is of length ‘test_size’. We also provide a parameter ‘max_periods’,
    which allows the user to roll the training set forward as opposed to expanding it.
    If the number of time periods in the training set exceeds ‘max_periods’, the earliest
    time periods are truncated.

    This splitter can be employed, in addition to standard use, to reflect a pipeline
    through time in a real-world setting. This is especially the case when
    'test_size' is set to 1.

    :param <int> train_intervals: training interval length in time periods for sequential
        training. This is the number of periods by which the training set is expanded at
        each subsequent split. Default is 21.
    :param <int> min_cids: minimum number of cross-sections required for the initial
        training set. Default is 4.
    :param <int> min_periods: minimum number of time periods required for the initial
        training set. Default is 500.
    :param <int> test_size: test set length for interval training. This is the number of
        periods to use for the test set subsequent to the training set. Default is 21.
    :param <int> max_periods: maximum length of each training set in interval training.
        If the maximum is exceeded, the earliest periods are cut off.
        Default is None.
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
    df_xcats2.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)

    dfd2 = dfd2.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X2 = dfd2.drop(columns=["XR"])
    y2 = dfd2["XR"]

    # 1) Demonstration of basic functionality

    # a) n_splits = 4, n_split_method = expanding
    """splitter = ExpandingKFoldPanelSplit(n_splits=4)
    splitter.split(X2, y2)
    cv_results = cross_validate(
        LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
    )
    splitter.visualise_splits(X2, y2)

    # b) n_splits = 4, n_split_method = expanding, Australia only visualisation
    splitter.visualise_splits(X2[X2.index.get_level_values(0)=="AUD"], y2[y2.index.get_level_values(0)=="AUD"])

    # c) n_splits = 4, n_split_method = rolling
    splitter = RollingKFoldPanelSplit(n_splits=4)
    splitter.split(X2, y2)
    cv_results = cross_validate(
        LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
    )
    splitter.visualise_splits(X2, y2)

    # d) n_splits = 4, n_split_method = rolling, Canada only visualisation
    splitter.visualise_splits(X2[X2.index.get_level_values(0)=="CAD"], y2[y2.index.get_level_values(0)=="CAD"]) 

    # e) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12, test_size=1, min_periods=21 * 12, min_cids=4
    )
    splitter.split(X2, y2)
    cv_results = cross_validate(
        LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
    )
    splitter.visualise_splits(X2, y2)"""

    splitter = ExpandingFrequencyPanelSplit(
        expansion_freq="W", test_freq="Y", min_cids=4, min_periods=21 * 12
    )
    splitter.visualise_splits(X2, y2)

    # f) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4, Britain only visualisation
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12, test_size=1, min_periods=21 * 12, min_cids=1
    )
    splitter.visualise_splits(X2, y2)

    # g) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4, max_periods=12*21
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12,
        test_size=21 * 12,
        min_periods=21 * 12,
        min_cids=4,
        max_periods=12 * 21,
    )
    splitter.split(X2, y2)
    cv_results = cross_validate(
        LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
    )
    splitter.visualise_splits(X2, y2)

    # h) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4, max_periods=12*21, USD only visualisation
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12,
        test_size=21 * 12,
        min_periods=21 * 12,
        min_cids=1,
        max_periods=12 * 21,
    )
    splitter.visualise_splits(X2[X2.index.get_level_values(0)=="USD"], y2[y2.index.get_level_values(0)=="USD"])

    # 2) Grid search capabilities
    lasso = Lasso()
    parameters = {"alpha": [0.1, 1, 10]}
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12, test_size=1, min_periods=21, min_cids=4
    )
    gs = GridSearchCV(
        lasso,
        parameters,
        cv=splitter,
        scoring="neg_root_mean_squared_error",
        refit=False,
        verbose=3,
    )
    gs.fit(X2, y2)
    print(gs.best_params_)
