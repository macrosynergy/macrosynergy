import numpy as np
import pandas as pd
import logging
import datetime
from sklearn.model_selection import BaseCrossValidator
from typing import Optional, List, Iterator, Tuple
import matplotlib.pyplot as plt


class PanelTimeSeriesSplit(BaseCrossValidator):
    """
    Class for the production of paired training and test splits for panel data. Thus, it
    can be used for sequential training, sequential validation and walk-forward validation
    over a panel.

    :param <int> train_intervals: training interval length in time periods for sequential
        training. This is the number of periods by which the training set is expanded at
        each subsequent split. Default is 21.
    :param <int> min_cids: minimum number of cross-sections required for the initial
        training set. Default is 4. Only used if train_intervals is specified.
    :param <int> min_periods: minimum number of time periods required for the initial
        training set. Default is 500. Only used if train_intervals is specified.
    :param <int> test_size: test set length for interval training. This is the number of
        periods to use for the test set subsequent to the training set. Default is 21.
    :param <int> max_periods: maximum length of each training set in interval training.
        If the maximum is exceeded, the earliest periods are cut off.
        Default is None.
    :param <int> n_splits: number of time splits to make. This is an alternative to
        defining sequential training intervals. If specified, the overall panel is split
        into n_splits blocks of adjacent dates. Then, either expanding or rolling windows
        of these blocks are used for training and testing. When `n_splits` is specified,
        `min_periods`, `test_size`, `min_cids`, `max periods` and `train_intervals` are ignored.
        Default is None.
    :param <int> n_split_method: either "expanding" (default) or "rolling". If "expanding"
        is chosen, the training sets are determined by sequential blocks of dates
        up to but including the last block. The test sets are always the block following
        the last date of training set, mimicking the sklearn class `TimeSeriesSplit` but
        splitting by observation date instead of index. If `rolling` is chosen, then the
        training sets are all combinations of adjacent blocks of dates, whereby the first
        and the last block count as adjacent. Thus in this case, the splits are not purely
        sequential.

    """

    def __init__(
        self,
        train_intervals: Optional[int] = 21,
        min_cids: Optional[int] = 4,
        min_periods: Optional[int] = 500,
        test_size: int = 21,
        max_periods: Optional[int] = None,
        n_splits: Optional[int] = None,
        n_split_method: Optional[str] = "expanding",
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing PanelTimeSeriesSplit")
        if n_splits is not None:
            if train_intervals is not None:
                self.logger.warning(
                    "'n_splits' overrides 'train_intervals'. Ignoring 'train_intervals'."
                )
            train_intervals = None
            min_periods = None
            test_size = None
            max_periods = None
            min_cids = None
            assert n_split_method in [
                "expanding",
                "rolling",
            ], "n_split_method must be either 'expanding' or 'rolling'."

        else:
            # Note for Ralph: the below is needed because the user could still (accidentally) set train_intervals, min_periods and min_cids to None even if n_splits is None.
            # This is despite the defaults set in the function definition.
            n_split_method = None
            assert (
                (train_intervals is not None)
                & (min_periods is not None)
                & (min_cids is not None)
                & (test_size is not None)
            ), "If 'n_splits' is not specified, then 'train_intervals', 'min_periods', 'min_cids' and 'test_size' must be specified."
            assert (min_cids > 0) & (
                type(min_cids) == int
            ), "min_cids must be an integer greater than 0."
            assert (min_periods > 0) & (
                type(min_periods) == int
            ), "min_periods must be an integer greater than 0."
            assert (test_size > 0) & (
                type(test_size) == int
            ), "test_size must be an integer greater than 0."

        self.train_intervals: int = train_intervals
        self.min_cids: int = min_cids
        self.min_periods: int = min_periods
        self.test_size: int = test_size
        self.max_periods: int = max_periods
        self.n_splits: int = n_splits
        self.n_split_method: str = n_split_method

    def get_n_splits(self, X: pd.DataFrame, y: pd.DataFrame) -> int:
        """
        Calculates number of splits. This method is implemented for compatibility with scikit-learn,
        in order to subclass BaseCrossValidator.

        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            The dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.

        :return <int> n_splits: number of splitting iterations in the cross-validator.
        """
        if self.train_intervals:
            _, _, self.n_splits = self.determine_unique_time_splits(X, y)

        return self.n_splits

    def determine_unique_time_splits(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[List[pd.DatetimeIndex], pd.DataFrame, int]:
        """
        Helper method to determine the unique dates in each training split. This method is called by self.split().
        It further returns other variables needed for ensuing components of the split method.
        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            The dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.

        :return <Tuple[List[pd.DatetimeIndex],pd.DataFrame,int]> (train_splits_basic, Xy, n_splits):
            Tuple comprising the unique dates in each training split, the concatenated dataframe of X and y, and the number of splits.
        """
        self.logger.info("Sanity checks")
        # Check that X and y are multi-indexed
        assert isinstance(X.index, pd.MultiIndex), "X must be multi-indexed."
        assert isinstance(y.index, pd.MultiIndex), "y must be multi-indexed."
        # Check the inner multi-index levels are datetime indices
        assert isinstance(
            X.index.get_level_values(1)[0], datetime.date
        ), "The inner index of X must be datetime.date."
        assert isinstance(
            y.index.get_level_values(1)[0], datetime.date
        ), "The inner index of y must be datetime.date."
        # Check that X and y are indexed in the same order
        assert X.index.equals(
            y.index
        ), "The indices of the input dataframe X and the output dataframe y don't match."
        # drops row, corresponding with a country & period, if either a feature or the target is missing. Resets index for efficiency later in the code.
        Xy = pd.concat([X, y], axis=1)
        Xy = Xy.dropna()
        self.unique_times: pd.DatetimeIndex = (
            Xy.index.get_level_values(1).sort_values().unique()
        )
        if self.min_periods is not None:
            assert self.min_periods <= len(
                self.unique_times
            ), """The minimum number of time periods for each cross-section in the first split cannot be larger than
            the number of unique dates in the entire dataframe X"""

        # (1) Determine the splits prior to aggregation
        # The approach is to calculate a numpy array for the unique dates in each training split.
        # The procedure is different depending on whether or not n_splits is specified.
        if self.train_intervals:
            # First determine the dates for the first training set, determined by 'min_cids' and 'min_periods'.
            # (a) obtain a boolean mask of dates for which at least 'min_cids' cross-sections are available over the panel
            init_mask: pd.Series = (
                Xy.groupby(level=1).size().sort_index() >= self.min_cids
            )
            # (b) obtain the earliest date for which the mask is true i.e. the earliest date with 'min_cids' cross-sections available
            date_first_min_cids: pd.Timestamp = (
                init_mask[init_mask == True].reset_index().real_date.min()
            )
            # (c) find the last valid date in the first training set, which is the date 'min_periods' - 1 after date_first_min_cids
            date_last_train: pd.Timestamp = self.unique_times[
                np.where(self.unique_times == date_first_min_cids)[0][0]
                + self.min_periods
                - 1
            ]
            # (d) determine the unique dates in the training sets after the first.
            # This is done by collecting all dates in the panel after the last date in the first training set and before the last 'self.test_size' dates,
            # calculating the number of splits ('self.n_splits') required to split these dates into distinct training intervals of length 'self.train_intervals' (where possible)
            # and finally splitting the mentioned dates into 'self.n_splits' splits (post the first split, determined by 'self.min_cids' and 'self.min_periods').
            unique_times_train: pd.arrays.DatetimeArray = self.unique_times[
                np.where(self.unique_times == date_last_train)[0][0]
                + 1 : -self.test_size
            ]
            self.adjusted_n_splits: int = int(
                np.ceil(len(unique_times_train) / self.train_intervals)
            )
            train_splits_basic: List = np.array_split(unique_times_train, self.adjusted_n_splits)
            # (e) add the first training set to the list of training splits, so that the dates that constitute each training split are together.
            train_splits_basic.insert(
                0,
                Xy.index.get_level_values(1)[
                    Xy.index.get_level_values(1) <= date_last_train
                ]
                .unique()
                .sort_values(),
            )
            # (f) increment n_splits by one to account for the first training set
            # This is needed in order to loop over the training dates later on to determine the dataframe indices that are returned.
            self.adjusted_n_splits += 1
        else:
            # when n_splits is specified, the unique dates in each training split are determined based on whether n_split_method is expanding or rolling.
            # expanding: the training sets are determined by sequential blocks of dates up to but not including the last block.
            train_splits_basic: List[pd.DatetimeIndex] = np.array_split(
                self.unique_times, self.adjusted_n_splits
            )
            if self.n_split_method == "expanding":
                del train_splits_basic[-1]
                self.adjusted_n_splits -= 1

        return train_splits_basic, Xy, self.adjusted_n_splits

    def adjust_time_splits(
        self, train_splits_basic: List[pd.DatetimeIndex]
    ) -> List[pd.DatetimeIndex]:
        """
        Helper method for adjusting the training dates in each split. If aggregation is specified, through either neglecting to set max_periods or by setting n_splits_method="expanding".
        then the training dates in each split are concatenated to the previous split.
        If 'max_periods' is specified for interval training, then the concatenated dates are cut off at the maximum number of periods specified.

        :param <List[pd.DatetimeIndex]> train_splits_basic: list of unique dates in each training split.

        :return <List[pd.DatetimeIndex]> train_splits: list of unique dates in each training split, adjusted for rolling or expanding windows.
        """
        if self.train_intervals:
            train_splits: List[np.array] = [
                train_splits_basic[0]
                if not self.max_periods
                else train_splits_basic[0][-self.max_periods :]
            ]
            for i in range(1, self.adjusted_n_splits):
                train_splits.append(
                    np.concatenate([train_splits[i - 1], train_splits_basic[i]])
                )
                if self.max_periods and self.train_intervals:
                    train_splits[i] = train_splits[i][-self.max_periods :]
        elif self.n_splits and self.n_split_method == "expanding":
            train_splits: List[pd.DatetimeIndex] = [train_splits_basic[0]]
            for i in range(1, self.adjusted_n_splits):
                train_splits.append(
                    np.concatenate([train_splits[i - 1], train_splits_basic[i]])
                )
        else:
            # n_splits specified and n_split_method is rolling.
            # This should ultimately work in the same way as KFold but preserving time.
            train_splits: List[np.array] = []
            for i in range(len(train_splits_basic)):
                concatenated_array = np.array([], dtype=np.datetime64)
                for j in range(len(train_splits_basic)):
                    if i != j:
                        concatenated_array = np.concatenate(
                            (concatenated_array, train_splits_basic[j])
                        )
                train_splits.append(concatenated_array)

        return train_splits

    def create_train_test_indices(
        self,
        train_splits: List[pd.DatetimeIndex],
        train_splits_basic: List[pd.DatetimeIndex],
        Xy: pd.DataFrame,
    ) -> List[Tuple[pd.MultiIndex, pd.MultiIndex]]:
        """
        Helper method for creating training and test indices from the unique dates in each training split.

        :param <List[pd.DatetimeIndex]> train_splits: list of unique dates in each training split, adjusted for rolling or expanding windows.
        :param <List[pd.DatetimeIndex]> train_splits_basic: list of unique dates in each training split, prior to adjustment.

        :return <List[Tuple[pd.MultiIndex,pd.MultiIndex]]>: list of (train,test) multi-indices, giving rise to the different splits.
        """
        if self.train_intervals:
            for split in train_splits:
                self.train_indices.append(
                    Xy.index[Xy.index.get_level_values(1).isin(split)]
                )
                self.test_indices.append(
                    Xy.index[
                        Xy.index.get_level_values(1).isin(
                            self.unique_times[
                                np.where(self.unique_times == np.max(split))[0][0]
                                + 1 : np.where(self.unique_times == np.max(split))[0][0]
                                + 1
                                + self.test_size
                            ]
                        )
                    ]
                )
        else:
            # then n_splits set
            if self.n_split_method == "expanding":
                for split_idx in range(self.adjusted_n_splits):
                    if split_idx != self.adjusted_n_splits - 1:
                        self.train_indices.append(
                            Xy.index[
                                Xy.index.get_level_values(1).isin(
                                    train_splits[split_idx]
                                )
                            ]
                        )
                        self.test_indices.append(
                            Xy.index[
                                Xy.index.get_level_values(1).isin(
                                    train_splits_basic[split_idx + 1]
                                )
                            ]
                        )
                    else:
                        self.train_indices.append(
                            Xy.index[
                                Xy.index.get_level_values(1).isin(
                                    train_splits[split_idx]
                                )
                            ]
                        )
                        if self.n_split_method == "expanding":
                            self.test_indices.append(
                                Xy.index[
                                    Xy.index.get_level_values(1).isin(
                                        self.unique_times[
                                            np.where(
                                                self.unique_times
                                                == np.max(train_splits[split_idx])
                                            )[0][0]
                                            + 1 :
                                        ]
                                    )
                                ]
                            )
            else:
                # rolling
                for split_idx in range(self.n_splits):
                    self.train_indices.append(
                        Xy.index[
                            Xy.index.get_level_values(1).isin(train_splits[split_idx])
                        ]
                    )
                    self.test_indices.append(
                        Xy.index[
                            ~Xy.index.get_level_values(1).isin(train_splits[split_idx])
                        ]
                    )

        return list(zip(self.train_indices, self.test_indices))

    def split(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> List[Tuple[pd.MultiIndex, pd.MultiIndex]]:
        """
        Method that determines pairs of training and test indices for a wide format Pandas (panel) dataframe, for use in
        sequential training, validation or walk-forward validation over a panel.

        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.

        :return <List[Tuple[pd.MultiIndex,pd.MultiIndex]]> splits: list of (train,test) multi-indices for walk-forward validation.
        """
        self.train_indices: List[pd.MultiIndex] = []
        self.test_indices: List[pd.MultiIndex] = []
        self.adjusted_n_splits: int = self.n_splits

        train_splits_basic, Xy, _ = self.determine_unique_time_splits(X, y)

        # (2) Aggregate or roll training dates in train_splits_basic depending on whether or not max_periods is specified
        # This is done by looping over the training date splits and concatenating the dates in each split to the previous split.
        # If max_periods is specified, then the concatenated dates are cut off at the maximum number of periods specified.
        train_splits = self.adjust_time_splits(train_splits_basic)

        # (3) Lastly, create the train and test indices.
        # train_splits now comprises the unique dates in each training split. Thus, to return indices, we loop through each date split,

        iterator = self.create_train_test_indices(train_splits, train_splits_basic, Xy)

        return iterator

    def calculate_xranges(self,cs_dates: pd.DatetimeIndex, real_dates: np.array):
        """
        Helper method to determine the ranges of contiguous dates in each training and test set, for use in visualisation.

        :param <pd.DatetimeIndex> cs_dates: DatetimeIndex of dates in a set for a given cross-section.
        :param <pd.DatetimeIndex> real_dates: DatetimeIndex of all dates in the panel.

        :return <List[Tuple[pd.Timestamp,pd.Timedelta]]> xranges: list of tuples of the form (start date, length of contiguous dates).
        """

        xranges: List[Tuple[pd.Timestamp,pd.Timedelta]] = []
        if len(cs_dates) == 0:
            return xranges
        
        lower_bound: pd.Timestamp = cs_dates.min()
        upper_bound: pd.Timestamp = cs_dates.max()

        upper_bound_idx = np.where(real_dates == upper_bound)[0]
        if upper_bound_idx and upper_bound_idx[0] + 1 < len(real_dates):
            upper_bound = real_dates[upper_bound_idx[0] + 1]
        
        in_contiguous: bool = True
        lower: pd.Timestamp = lower_bound
        upper: pd.Timestamp = upper_bound

        for real_date in real_dates[(real_dates >= lower_bound) & (real_dates <= upper_bound)]:
            if real_date in cs_dates:
                if not in_contiguous:
                    in_contiguous = True
                    lower = real_date

                upper = real_date if real_date == upper_bound else real_dates[np.where(real_dates == real_date)[0][0] + 1]
            else:
                if in_contiguous:
                    xranges.append((lower, upper - lower))
                    in_contiguous = False

        xranges.append((lower, upper - lower))
        return xranges
    
    def visualise_splits(self, X: pd.DataFrame, y: pd.DataFrame, figsize: Tuple[int, int] = (20,5)) -> None:
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
        plt.style.use("seaborn-whitegrid")
        Xy: pd.DataFrame = pd.concat(
            [X, y], axis=1
        ).dropna()  # remove dropna when splitter method fixed as per TODO #3
        cross_sections: np.array[str] = np.array(
            sorted(Xy.index.get_level_values(0).unique())
        )
        real_dates: np.array[pd.Timestamp] = np.array(
            sorted(Xy.index.get_level_values(1).unique())
        )
        splits: List[Tuple[pd.MultiIndex, pd.MultiIndex]] = self.split(X, y)

        n_splits: int = self.adjusted_n_splits if self.adjusted_n_splits <= 5 else 5
        split_idxs: List[int] = (
            [0, len(splits) // 4, len(splits) // 2, 3 * len(splits) // 4, -1]
            if self.adjusted_n_splits > 5
            else [i for i in range(self.adjusted_n_splits)]
        )
        split_titles: List[str] = (
            [
                "Initial split",
                "Quarter progress",
                "Halfway progress",
                "Three quarters progress",
                "Final split",
            ]
            if self.adjusted_n_splits > 5
            else [f"Split {i+1}" for i in range(self.n_splits)]
        )

        fig, ax = plt.subplots(
            nrows=len(Xy.index.get_level_values(0).unique()),
            ncols=n_splits,
            figsize=figsize,
        )


        operations = []

        for cs_idx, cs in enumerate(cross_sections):
            for idx, split_idx in enumerate(split_idxs):
                cs_train_dates: pd.DatetimeIndex = Xy.loc[splits[split_idx][0]][
                    Xy.loc[splits[split_idx][0]].index.get_level_values(0) == cs
                ].index.get_level_values(1)
                cs_test_dates: pd.DatetimeIndex = Xy.loc[splits[split_idx][1]][
                    Xy.loc[splits[split_idx][1]].index.get_level_values(0) == cs
                ].index.get_level_values(1)

                xranges_train: List[Tuple[pd.Timestamp,pd.Timedelta]] = self.calculate_xranges(cs_train_dates, real_dates)
                xranges_test: List[Tuple[pd.Timestamp,pd.Timedelta]] = self.calculate_xranges(cs_test_dates, real_dates)

                if xranges_train:
                    operations.append((cs_idx, idx, xranges_train, "royalblue", "Train"))
                if xranges_test:
                    operations.append((cs_idx, idx, xranges_test, "lightcoral", "Test"))

        # Finally, apply all your operations on the ax object.
        for cs_idx, idx, xranges, color, label in operations:
            ax[cs_idx, idx].broken_barh(xranges, (-0.4, 0.8), facecolors=color, label=label)
            ax[cs_idx, idx].set_xlim(real_dates.min(), real_dates.max())
            ax[cs_idx, idx].set_yticks([0])
            ax[cs_idx, idx].set_yticklabels([cross_sections[cs_idx]])

            if cs_idx == 0:
                ax[cs_idx, idx].set_title(f"{split_titles[idx]}")

        plt.suptitle(
            f"Training and test set pairs, number of training sets={self.adjusted_n_splits}"
        )
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from macrosynergy.management.simulate_quantamental_data import make_qdf
    import macrosynergy.management as msm

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

    # a) n_splits = 4, n_split_method = expanding
    splitter = PanelTimeSeriesSplit(n_splits=4, n_split_method="expanding")
    splitter.split(X2,y2)
    splitter.visualise_splits(X2, y2)
    # b) n_splits = 4, n_split_method = rolling
    splitter = PanelTimeSeriesSplit(n_splits=4, n_split_method="rolling")
    splitter.split(X2,y2)
    splitter.visualise_splits(X2, y2)
    # c) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4
    splitter = PanelTimeSeriesSplit(
        train_intervals=21 * 12, test_size=1, min_periods=21, min_cids=4
    )
    splitter.split(X2,y2)
    splitter.visualise_splits(X2, y2)
    # d) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4, max_periods=12*21
    splitter = PanelTimeSeriesSplit(
        train_intervals=21 * 12,
        test_size=21 * 12,
        min_periods=21,
        min_cids=4,
        max_periods=12 * 21,
    )
    splitter.split(X2,y2)
    splitter.visualise_splits(X2, y2)