"""
Base class for a cross-validation splitter for panel data.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    BaseCrossValidator,
)

from abc import ABC, abstractmethod


class BasePanelSplit(BaseCrossValidator, ABC):
    """
    Abstract base class for a generic panel cross-validator.

    Provides the necessary visualisation methods for all panel splitters, for
    explainability and debugging purposes.
    """

    @abstractmethod
    def __init__(self):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits in the cross-validator.

        Parameters
        ----------
        X : pd.DataFrame
            Always ignored, exists for compatibility with scikit-learn.
        y : Union[pd.Series, pd.DataFrame]
            Always ignored, exists for compatibility with scikit-learn.
        groups : None
            Always ignored, exists for compatibility with scikit-learn.

        Returns
        -------
        n_splits : int
            Number of splits in the cross-validator.
        """
        if hasattr(self, "n_splits"):
            return self.n_splits
        else:
            return np.nan

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
        xranges : List[Tuple[pd.Timestamp, pd.Timedelta]]
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

        # Obtain relevant data
        Xy: pd.DataFrame = self._combine_Xy(X, y)
        cross_sections = np.array(sorted(Xy.index.get_level_values(0).unique()))
        real_dates = Xy.index.get_level_values(1).unique().sort_values()

        # Infer native dataset frequency
        freq_est = pd.infer_freq(real_dates)
        if freq_est is None:
            freq_est = real_dates.to_series().diff().min()
        freq_offset = pd.tseries.frequencies.to_offset(freq_est)  # Good approximation

        splits = list(self.split(X, y))

        # Set up plotting labels and figure
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
            nrows=len(cross_sections),
            ncols=min(self.n_splits, 5),
            figsize=figsize,
        )

        # Determine ranges of contiguous dates in each training and test set, within each
        # cross-section and split.
        operations = []
        for cs_idx, cs in enumerate(cross_sections):
            for idx, split_idx in enumerate(split_idxs):
                # Get the dates in the training and test sets for the given cross-section.
                cs_train_dates: pd.DatetimeIndex = X.iloc[splits[split_idx][0]][
                    X.iloc[splits[split_idx][0]].index.get_level_values(0) == cs
                ].index.get_level_values(1)
                cs_test_dates: pd.DatetimeIndex = X.iloc[splits[split_idx][1]][
                    X.iloc[splits[split_idx][1]].index.get_level_values(0) == cs
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
        # This will be added to the x-axis limits to ensure that the final split is visible.
        last_date = real_dates[-1]
        second_to_last_date = real_dates[-2]
        difference = last_date - second_to_last_date

        # Add all the broken bar plots to the figure.
        for cs_idx, idx, xranges, color, label in operations:
            if len(cross_sections) == 1:
                ax[idx].broken_barh(xranges, (-0.4, 0.8), facecolors=color, label=label)
                ax[idx].set_xlim(real_dates.min(), real_dates.max() + difference)
                ax[idx].set_yticks([0])
                ax[idx].set_yticklabels([cross_sections[0]])
                ax[idx].tick_params(axis="x", rotation=90)
                ax[idx].set_title(f"{split_titles[idx]}")

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

    @abstractmethod
    def _combine_Xy(self, X, y):
        """
        Combine the features and target variable into a single dataframe. This is
        dependent on the specific splitter implementation and, consequently, the
        implementation of the constructor.

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
        pass
