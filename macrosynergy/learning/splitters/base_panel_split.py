"""
Base class for a cross-validation splitter for panel data.
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
    Base class for a cross-validation splitter for panel data.

    BasePanelSplit defines the necessary methods for a panel cross-validator to contain
    in order to be compatible with sklearn's API. The class also contains a method for 
    visualising the splits produced by the splitter, which can be useful for both
    debugging and explainability. 
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