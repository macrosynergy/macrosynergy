"""
Tools to produce, visualise and use walk-forward validation splits across panels.

**NOTE: This module is under development, and is not yet ready for production use.**
"""

import logging
import datetime
from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    BaseCrossValidator,
    cross_validate,
    GridSearchCV,
)
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from macrosynergy.learning.panel_timeseries_split import PanelTimeSeriesSplit


class BasePanelTimeSeriesSplit(BaseCrossValidator):
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        :param <pd.DataFrame> X: Always ignored, exists for compatibility.
        :param <pd.DataFrame> y: Always ignored, exists for compatibility.
        :param <pd.DataFrame> groups: Always ignored, exists for compatibility.

        :return <int> n_splits: Returns the number of splits.
        """
        return self.n_splits
    
    def _validate(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
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
                "The indices of the input dataframe X and the output dataframe y don't match."
            )

    def __calculate_xranges(
        self, cs_dates: pd.DatetimeIndex, real_dates: pd.DatetimeIndex
    ):
        """
        Helper method to determine the ranges of contiguous dates in each training and test set, for use in visualisation.

        :param <pd.DatetimeIndex> cs_dates: DatetimeIndex of dates in a set for a given cross-section.
        :param <pd.DatetimeIndex> real_dates: DatetimeIndex of all dates in the panel.

        :return <List[Tuple[pd.Timestamp,pd.Timedelta]]> xranges: list of tuples of the form (start date, length of contiguous dates).
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
                (cs_dates.min(), cs_dates.max() - cs_dates.min() + pd.Timedelta(days=1))
            )
            return xranges

        # Multiple contiguous ranges of dates.
        else:
            while len(difference) > 0:
                xranges.append((cs_dates.min(), difference.min() - cs_dates.min()))
                cs_dates = cs_dates[(cs_dates >= difference.min())]
                difference = difference[(difference >= cs_dates.min())]

            xranges.append(
                (cs_dates.min(), cs_dates.max() - cs_dates.min() + pd.Timedelta(days=1))
            )
            return xranges

    def visualise_splits(
        self, X: pd.DataFrame, y: pd.DataFrame, figsize: Tuple[int, int] = (20, 5)
    ) -> None:
        """
        Method to visualise the splits created according to the parameters specified in the constructor.

        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        :param <Tuple[int,int]> figsize: tuple of integers specifying the splitter visualisation figure size.

        :return None
        """
        sns.set_theme(style="whitegrid", palette="colorblind")
        Xy: pd.DataFrame = pd.concat([X, y], axis=1).dropna()
        cross_sections: np.array[str] = np.array(
            sorted(Xy.index.get_level_values(0).unique())
        )
        real_dates = Xy.index.get_level_values(1).unique().sort_values()

        splits: List[Tuple[np.array[int], np.array[int]]] = self.split(X, y)

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

                xranges_train: List[
                    Tuple[pd.Timestamp, pd.Timedelta]
                ] = self.__calculate_xranges(cs_train_dates, real_dates)
                xranges_test: List[
                    Tuple[pd.Timestamp, pd.Timedelta]
                ] = self.__calculate_xranges(cs_test_dates, real_dates)

                if xranges_train:
                    operations.append(
                        (cs_idx, idx, xranges_train, "royalblue", "Train")
                    )
                if xranges_test:
                    operations.append((cs_idx, idx, xranges_test, "lightcoral", "Test"))

        # Calculate the difference between final two dates.
        # This will be added to the x-axis limits to ensure that the final split is visible.
        last_date = real_dates[-1]  # Get the last date
        second_to_last_date = real_dates[-2]  # Get the second-to-last date
        difference = last_date - second_to_last_date

        # Finally, apply all your operations on the ax object.
        for cs_idx, idx, xranges, color, label in operations:
            ax[cs_idx, idx].broken_barh(
                xranges, (-0.4, 0.8), facecolors=color, label=label
            )
            ax[cs_idx, idx].set_xlim(real_dates.min(), real_dates.max() + difference)
            ax[cs_idx, idx].set_yticks([0])
            ax[cs_idx, idx].set_yticklabels([cross_sections[cs_idx]])

            if cs_idx == 0:
                ax[cs_idx, idx].set_title(f"{split_titles[idx]}")

        plt.suptitle(
            f"Training and test set pairs, number of training sets={self.n_splits}"
        )
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.show()


class ExpandingPanelTimeSeriesSplit(BasePanelTimeSeriesSplit):
    """ """

    def __init__(self, n_splits: int = 5, test_size: int = None):
        """
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits. Must be at least 2.
        n_split_method : str, default="expanding"
            Method for splitting the data. Currently only "expanding" is supported.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing PanelTimeSeriesSplit")
        self.n_splits = n_splits
        self.n_folds = n_splits + 1

    def split(
        self, X: pd.DataFrame, y: pd.DataFrame, groups=None
    ) -> List[Tuple[np.array, np.array]]:
        """
        Method that determines pairs of training and test indices for a wide format Pandas (panel) dataframe, for use in
        sequential training, validation or walk-forward validation over a panel.

        :param <pd.DataFrame> X: Pandas dataframe of features,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of the target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        :param <int> groups: Always ignored, exists for compatibility with scikit-learn.

        :return <List[Tuple[np.array[int],np.array[int]]]> splits: list of (train,test) indices for walk-forward validation.
        """
        self.train_indices: List[int] = []
        self.test_indices: List[int] = []

        self._validate(X, y)

        # drops row, corresponding with a country & period, if either a feature or the target is missing. Resets index for efficiency later in the code.
        Xy: pd.DataFrame = pd.concat([X, y], axis=1)
        Xy = Xy.dropna()
        self.unique_times: pd.DatetimeIndex = (
            Xy.index.get_level_values(1).sort_values().unique()
        )

        train_splits_basic: List[pd.DatetimeIndex] = np.array_split(
            self.unique_times, self.n_folds
        )

        expanding_train_splits: List[pd.DatetimeIndex] = [train_splits_basic[0]]
        for i in range(1, self.n_folds):
            expanding_train_splits.append(
                np.concatenate([expanding_train_splits[i - 1], train_splits_basic[i]])
            )

        for split_idx in range(self.n_splits):
            self.train_indices.append(
                np.where(
                    Xy.index.get_level_values(1).isin(expanding_train_splits[split_idx])
                )[0]
            )
            self.test_indices.append(
                np.where(
                    Xy.index.get_level_values(1).isin(train_splits_basic[split_idx + 1])
                )[0]
            )
        iterator = list(zip(self.train_indices, self.test_indices))

        return iterator


class RollingPanelTimeSeriesSplit(BasePanelTimeSeriesSplit):
    """ """

    def __init__(self, n_splits: int = 5, test_size: int = None):
        """
        Parameters
        ----------
        n_splits : int, default=5
            Number of splits. Must be at least 2.
        n_split_method : str, default="expanding"
            Method for splitting the data. Currently only "expanding" is supported.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing PanelTimeSeriesSplit")
        self.n_splits = n_splits

    def split(
        self, X: pd.DataFrame, y: pd.DataFrame, groups=None
    ) -> List[Tuple[np.array, np.array]]:
        """
        Method that determines pairs of training and test indices for a wide format Pandas (panel) dataframe, for use in
        sequential training, validation or walk-forward validation over a panel.

        :param <pd.DataFrame> X: Pandas dataframe of features,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of the target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        :param <int> groups: Always ignored, exists for compatibility with scikit-learn.

        :return <List[Tuple[np.array[int],np.array[int]]]> splits: list of (train,test) indices for walk-forward validation.
        """
        self.train_indices: List[int] = []
        self.test_indices: List[int] = []

        self._validate(X, y)

        # drops row, corresponding with a country & period, if either a feature or the target is missing. Resets index for efficiency later in the code.
        Xy: pd.DataFrame = pd.concat([X, y], axis=1)
        Xy = Xy.dropna()
        self.unique_times: pd.DatetimeIndex = (
            Xy.index.get_level_values(1).sort_values().unique()
        )

        splits: List[pd.DatetimeIndex] = np.array_split(
            self.unique_times, self.n_splits
        )

        for i in range(self.n_splits):
            splits_copy = splits.copy()
            test_split = np.array(splits_copy.pop(i), dtype=np.datetime64)
            train_split = np.concatenate(splits_copy, dtype=np.datetime64)
            self.train_indices.append(
                np.where(Xy.index.get_level_values(1).isin(train_split))[0]
            )
            self.test_indices.append(
                np.where(Xy.index.get_level_values(1).isin(test_split))[0]
            )

        iterator = list(zip(self.train_indices, self.test_indices))

        return iterator


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm

    import pandas as pd

    # import numpy as np
    np.random.seed(0)

    # Define the multiindex using two cids and 10 dates.
    cids = ["cid1", "cid2"]
    dates = pd.date_range(start="2023-01-01", periods=6, freq="D")

    # Create a MultiIndex using the product of cids and dates.
    multiindex = pd.MultiIndex.from_product([cids, dates], names=["cid", "real_date"])

    # Initialize a DataFrame with the MultiIndex and columns A and B.
    # Fill in some example data using random numbers.
    df = pd.DataFrame(np.random.rand(12, 2), index=multiindex, columns=["A", "B"])

    df.head(20)  # Displaying the entire DataFrame since it has only 20 rows.
    X2 = df.drop(columns=["B"])
    y2 = df["B"]

    splitter = RollingPanelTimeSeriesSplit(n_splits=4)
    splits = splitter.split(X2, y2)
    print(splits)
    splitter.visualise_splits(X2, y2)

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

    # 1) Demonstration of basic functionality

    # a) n_splits = 4, n_split_method = expanding
    splitter = ExpandingPanelTimeSeriesSplit(n_splits=4)
    splitter.split(X2, y2)
    cv_results = cross_validate(
        LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
    )
    splitter.visualise_splits(X2, y2)
