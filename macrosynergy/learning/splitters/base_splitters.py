"""
Base classes for panel cross-validation splitters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import (
    BaseCrossValidator,
)

from abc import ABC, abstractmethod


class BasePanelSplit(BaseCrossValidator, ABC):
    """
    Generic cross-validation class for panel data.

    Notes
    -----
    This class is designed to provide a common interface for cross-validation on panel
    data. Much of the logic can be written in subclasses, but this base class contains
    the necessary code to visualise the splits for each cross-section in the panel.
    """

    @abstractmethod
    def __init__(self):
        """
        Constructor for the base class.
        """
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits in the cross-validator.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Pandas dataframe of features, multi-indexed by (cross-section, date). The
            dates must be in datetime format. Otherwise the dataframe must be in wide
            format: each feature is a column.
        y : Union[pd.Series, pd.DataFrame], optional
            Pandas dataframe or series of a target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format. If a dataframe
            is provided, the target variable must be the sole column.
        groups : None
            Always ignored, exists for compatibility with scikit-learn.

        Returns
        -------
        n_splits : int
            Number of splits in the cross-validator.
        """
        # Checks
        self._check_Xy(X, y)

        # Determine number of splits in the cross-validator
        if hasattr(self, "n_splits"):
            return self.n_splits
        else:
            return len(list(self.split(X, y)))

    def _calculate_xranges(
        self,
        cs_dates,
        real_dates,
        freq_offset,
    ):
        """
        Returns date ranges of contiguous blocks in each training and test set.

        Parameters
        ----------
        cs_dates : pd.DatetimeIndex
            DatetimeIndex of dates in a (training or test) set for a given cross-section.
        real_dates : pd.DatetimeIndex
            DatetimeIndex of all dates in the panel.
        freq_offset : pd.DateOffset
            DateOffset object representing the frequency of the dates in the panel.

        Returns
        -------
        xranges : tuple
            List of tuples of the form (start date, length of contiguous dates).
        """
        xranges = []
        if len(cs_dates) == 0:
            # No dates in the training or test set - return empty list.
            return xranges

        # Filter all dates in the panel spanning the dates in the considered set
        filtered_real_dates = real_dates[
            (real_dates >= cs_dates.min()) & (real_dates <= cs_dates.max())
        ]
        # Differences may arise due to blacklisting
        difference = filtered_real_dates.difference(cs_dates)

        if len(difference) == 0:
            # Only one contiguous range of dates.
            xranges.append(
                (cs_dates.min(), cs_dates.max() + freq_offset - cs_dates.min())
            )
        else:
            # Multiple contiguous blocks present. Iterate over them.
            while len(difference) > 0:
                xranges.append((cs_dates.min(), difference.min() - cs_dates.min()))
                cs_dates = cs_dates[(cs_dates >= difference.min())]
                difference = difference[(difference >= cs_dates.min())]

            xranges.append(
                (cs_dates.min(), cs_dates.max() + freq_offset - cs_dates.min())
            )

        return xranges

    def visualise_splits(
        self,
        X,
        y,
        figsize=(20, 5),
    ):
        """
        Visualise the cross-validation splits.

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of features/quantamental indicators, multi-indexed by
            (cross-section, date). The dates must be in datetime format. The
            dataframe must be in wide format: each feature is a column.
        y : pd.DataFrame
            Pandas dataframe of target variable, multi-indexed by (cross-section, date).
            The dates must be in datetime format.
        figsize : Tuple[int, int]
            Tuple of integers specifying the splitter visualisation figure size.
        """
        sns.set_theme(style="whitegrid", palette="colorblind")

        # Checks
        self._check_Xy(X, y)
        if not isinstance(figsize, tuple):
            raise TypeError("figsize must be a tuple.")
        for i in figsize:
            if not isinstance(i, int):
                raise TypeError("figsize must contain only integers.")
            if i <= 0:
                raise ValueError("figsize must contain only positive integers.")

        # Obtain relevant data
        Xy: pd.DataFrame = pd.concat([X, y], axis=1).dropna()
        cross_sections = np.array(sorted(Xy.index.get_level_values(0).unique()))
        real_dates = Xy.index.get_level_values(1).unique().sort_values()

        # Infer native dataset frequency
        freq_est = pd.infer_freq(real_dates)
        if freq_est is None:
            freq_est = real_dates.to_series().diff().min()
        freq_offset = pd.tseries.frequencies.to_offset(freq_est)  # Good approximation

        splits = list(self.split(X, y))

        # Set up plotting labels and figure
        split_idxs: list = (
            [0, len(splits) // 4, len(splits) // 2, 3 * len(splits) // 4, -1]
            if self.n_splits > 5
            else [i for i in range(self.n_splits)]
        )
        split_titles: list = (
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
            nrows=len(cross_sections),
            ncols=min(self.n_splits, 5),
            figsize=figsize,
        )

        # Determine ranges of contiguous dates in each training and test set, within each
        # cross-section and split.
        plot_components = []
        for cs_idx, cs in enumerate(cross_sections):
            for idx, split_idx in enumerate(split_idxs):
                # Get the dates in the training and test sets for the given cross-section.
                cs_train_dates: pd.DatetimeIndex = Xy.iloc[splits[split_idx][0]][
                    Xy.iloc[splits[split_idx][0]].index.get_level_values(0) == cs
                ].index.get_level_values(1)
                cs_test_dates: pd.DatetimeIndex = Xy.iloc[splits[split_idx][1]][
                    Xy.iloc[splits[split_idx][1]].index.get_level_values(0) == cs
                ].index.get_level_values(1)

                xranges_train: list = self._calculate_xranges(
                    cs_train_dates, real_dates, freq_offset
                )
                xranges_test: list = self._calculate_xranges(
                    cs_test_dates, real_dates, freq_offset
                )

                plot_components.append(
                    (cs_idx, idx, xranges_train, "royalblue", "Train")
                )
                plot_components.append(
                    (cs_idx, idx, xranges_test, "lightcoral", "Test")
                )

        # Calculate the difference between final two dates.
        # This will be added to the x-axis limits to ensure that the final split is visible.
        last_date = real_dates[-1]
        second_to_last_date = real_dates[-2]
        difference = last_date - second_to_last_date

        # Add all the broken bar plots to the figure.
        for cs_idx, idx, xranges, color, label in plot_components:
            if len(cross_sections) == 1:
                ax[idx].broken_barh(xranges, (-0.4, 0.8), facecolors=color, label=label)
                ax[idx].set_xlim(real_dates.min(), real_dates.max() + difference)
                ax[idx].set_yticks([0])
                ax[idx].set_yticklabels([cross_sections[0]])
                ax[idx].tick_params(axis="x", rotation=90)
                ax[idx].set_title(f"{split_titles[idx]}")
            elif len(split_idxs) == 1:
                ax[cs_idx].broken_barh(
                    xranges, (-0.4, 0.8), facecolors=color, label=label
                )
                ax[cs_idx].set_xlim(real_dates.min(), real_dates.max() + difference)
                ax[cs_idx].set_yticks([0])
                ax[cs_idx].set_yticklabels([cross_sections[cs_idx]])
                ax[cs_idx].tick_params(axis="x", rotation=90)
            else:
                ax[cs_idx, idx].broken_barh(
                    xranges, (-0.4, 0.8), facecolors=color, label=label
                )
                ax[cs_idx, idx].set_xlim(
                    real_dates.min(), real_dates.max() + difference
                )
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

    def _check_Xy(self, X, y):
        """
        Type and value checks of input feature and target panels.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Pandas dataframe of features/quantamental indicators, multi-indexed by
            (cross-section, date). The dates must be in datetime format. The
            dataframe must be in wide format: each feature is a column.
        y : pd.DataFrame, optional
            Pandas dataframe of target variable, multi-indexed by (cross-section, date).
            The dates must be in datetime format.
        """
        if X is None:
            # Check y is not provided
            if y is not None:
                raise ValueError("Either both X and y must be provided or neither.")
        else:
            # Check y is provided
            if y is None:
                raise ValueError("Either both X and y must be provided or neither.")

        if X is not None:
            # Check X and y are dataframes/series respectively
            if not isinstance(X, pd.DataFrame):
                raise TypeError("X must be a pandas dataframe.")
            if not isinstance(y, (pd.DataFrame, pd.Series)):
                raise TypeError("y must be a pandas dataframe or series.")
            if isinstance(y, pd.DataFrame) and len(y.columns) != 1:
                raise ValueError("If y is a dataframe, it must have only one column.")

            # Check indexing of X and y
            if not X.index.equals(y.index):
                raise ValueError("X and y must have the same index.")
            if not isinstance(X.index, pd.MultiIndex):
                raise ValueError("X and y must have a multi-index.")
            if X.index.get_level_values(0).dtype != "object":
                raise ValueError(
                    "The input data must have string outer index representing panel cross-sections."
                )
            if X.index.get_level_values(1).dtype != "datetime64[ns]":
                raise ValueError(
                    "The input data must have datetime inner index representing timestamps as datetime64[ns]."
                )


class WalkForwardPanelSplit(BasePanelSplit, ABC):
    """
    Generic walk-forward panel cross-validator.

    Parameters
    ----------
    min_cids : int
        Minimum number of cross-sections required for the first training set.
        Either start_date or (min_cids, min_periods) must be provided.
        If both are provided, start_date takes precedence.
    min_periods : int
        Minimum number of time periods required for the first training set. Either
        start_date or (min_cids, min_periods) must be provided. If both are
        provided, start_date takes precedence.
    start_date : str, optional
        The targeted final date in the initial training set in ISO 8601 format.
        Default is None. Either start_date or (min_cids, min_periods) must be provided.
        If both are provided, start_date takes precedence.
    max_periods : int, optional
        The maximum number of time periods in each training set. If the maximum is
        exceeded, the earliest periods are cut off. This effectively creates rolling
        training sets. Default is None.

    Notes
    -----
    Provides train/test indices to split a panel into train/test sets. Following an
    initial training set construction, a forward test set is created. The training and
    test set pair evolves over time by walking forward through the panel.
    """

    def __init__(
        self,
        min_cids,
        min_periods,
        start_date=None,
        max_periods=None,
    ):
        # Checks
        self._check_wf_params(
            min_cids=min_cids,
            min_periods=min_periods,
            start_date=start_date,
            max_periods=max_periods,
        )

        # Attributes
        self.min_cids = min_cids
        self.min_periods = min_periods
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.max_periods = max_periods

    def _check_wf_params(self, min_cids, min_periods, start_date, max_periods):
        """
        Type and value checks for the class initialisation parameters.

        Parameters
        ----------
        min_cids : int
            Minimum number of cross-sections required for the first training set.
        min_periods : int
            Minimum number of time periods required for the first training set.
        start_date : str
            The targeted final date in the initial training set in ISO 8601 format.
        max_periods : int
            The maximum number of time periods in each training set.
        """
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
        # start_date
        if start_date is not None and not isinstance(start_date, str):
            raise TypeError(f"start_date must be a string. Got {type(start_date)}.")
        if start_date is not None:
            try:
                datetime.datetime.fromisoformat(start_date)
            except ValueError:
                raise ValueError(
                    f"start_date must be in ISO 8601 format. Got {start_date}."
                )
        # max_periods
        if max_periods is not None and not isinstance(max_periods, int):
            raise TypeError(f"max_periods must be an integer. Got {type(max_periods)}.")
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

        y : pd.DataFrame or pd.Series
            Pandas dataframe or series of a target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format. If a dataframe
            is provided, the target variable must be the sole column.

        groups : None
            Ignored. Exists for compatibility with scikit-learn.
        """
        # X
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas dataframe.")
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not pd.api.types.is_datetime64_any_dtype(X.index.get_level_values(1)):
            raise ValueError(
                f"The dates in X must be datetime objects. Got {X.index.get_level_values(1).dtype} instead."
            )
        # y
        if not isinstance(y, (pd.DataFrame, pd.Series)):
            raise TypeError("y must be a pandas dataframe or series.")
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


class KFoldPanelSplit(BasePanelSplit, ABC):
    def __init__(self, n_splits=5, min_n_splits=2):
        """
        Generic K-Fold cross-validator for panel data.

        Parameters
        ----------
        n_splits : int
            Number of splits to generate.
        min_n_splits : int
            Minimum number of splits allowed.

        Notes
        -----
        Provides train/test indices to split a panel into train/test sets. The panel is
        divided into n_splits consecutive folds. Each fold is then used once as a
        validation fold whilst a proportion of the other folds are used as training data.
        """
        # Checks
        if not isinstance(n_splits, int):
            raise TypeError(f"n_splits must be an integer. Got {type(n_splits)}.")
        if n_splits < min_n_splits:
            raise ValueError(
                f"Cannot have number of splits less than {min_n_splits}. Got n_splits = {n_splits}."
            )

        # Attributes
        self.n_splits = n_splits

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
        Xy.dropna(inplace=True)
        dates = Xy.index.get_level_values(1)
        unique_dates = dates.unique().sort_values()

        # Calculate splits
        splits = self._determine_splits(unique_dates, self.n_splits)

        # Yield splits
        for n_split in range(self.n_splits):
            yield self._get_split_indicies(n_split, splits, Xy, dates, unique_dates)

    @abstractmethod
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
        pass

    @abstractmethod
    def _get_split_indicies(self, idx, split, splits, Xy, dates, unique_dates):
        """
        Determine the training and test set indices for a given split.

        Parameters
        ----------
        n_split : int
            Index of the current split.
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
        pass

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
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas dataframe.")
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not pd.api.types.is_datetime64_any_dtype(X.index.get_level_values(1)):
            raise ValueError(
                f"The dates in X must be datetime objects. Got {X.index.get_level_values(1).dtype} instead."
            )
        # y
        if not isinstance(y, (pd.DataFrame, pd.Series)):
            raise TypeError("y must be a pandas dataframe or series.")
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
