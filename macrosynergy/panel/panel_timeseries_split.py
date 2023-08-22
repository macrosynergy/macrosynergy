import numpy as np
import pandas as pd
import logging
import datetime
from sklearn.model_selection import BaseCrossValidator
from typing import Optional, List, Iterator, Tuple

class PanelTimeSeriesSplit(BaseCrossValidator):

    """
    Class for the production of cross-section splits for panel data.

    :param <int> n_splits: number of time splits to make. If this is not specified, then
        the split will be governed by sequential training intervals. Otherwise,
        all arguments that are specific to sequential training intervals are ignored.
    :param <int> train_intervals: training interval length in time periods for sequential
        training. This is the number of periods by which the training set is expanded at
        each subsequent split. Default is 21.
    :param <int> min_cids: minimum number of cross-sections required for the initial
        training set. Default is 4. Only used if train_intervals is specified.
    :param <int> min_periods: minimum number of time periods required for the initial
        training set. Default is 500. Only used if train_intervals is specified.
    :param <int> test_size: test set length for interval training. This is the number of
        periods to use for the test set subsequent to the training set. Default is 21.
    :param <int> max_periods: maximum length of the training set in interval training.
        If the maximum is exceeded, the earliest periods are cut off.
        Default is None.

    N.B: The class provides training/validation indices to split the panel samples, observed
    at fixed dates for a number of cross-sections, in sequential training/validation sets.
    Unlike the sklearn class `TimeSeriesSplit`, this class makes splits based on the
    observation dates as opposed to the sample indices.

    The splitting occurs in one of two ways:
    (1) the number of splits are directly specified, or
    (2) the number of splits are determined by the number of forward time periods that are
    used to expand the training set at each iteration. Other parameters determine the
    configurations of the splits made and depend on the splitting method used.
    Default is defining an expanding training window, as opposed to a rolling window.
    """

    def __init__(
        self,
        n_splits: Optional[int] = None,
        train_intervals: Optional[int] = 21,
        test_size: int = 21,
        min_cids: Optional[int] = 4,
        min_periods: Optional[int] = 500,
        max_periods: Optional[int] = None,
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
            min_cids = None
        else:
            # Note for Ralph: the below is needed because the user could still (accidentally) set train_intervals, min_periods and min_cids to None even if n_splits is None.
            # This is despite the defaults set in the function definition.
            assert (
                (train_intervals is not None)
                & (min_periods is not None)
                & (min_cids is not None)
            ), "If 'n_splits' is not specified, then 'train_intervals', 'min_periods' and 'min_cids' must be specified."
            assert (min_cids > 0) & (
                type(min_cids) == int
            ), "min_cids must be an integer greater than 0."
            assert (min_periods > 0) & (
                type(min_periods) == int
            ), "min_periods must be an integer greater than 0."

        # Note for Ralph: as before, the below is needed because the user could still (accidentally) set test_size to None even if n_splits is not None.
        # Despite the default.
        assert test_size is not None, "test_size must be specified."
        assert (test_size > 0) & (
            type(test_size) == int
        ), "test_size must be an integer greater than 0."

        self.n_splits: int = n_splits
        self.train_intervals: int = train_intervals
        self.test_size: int = test_size
        self.min_cids: int = min_cids
        self.min_periods: int = min_periods
        self.max_periods: int = max_periods
        self.train_indices: List[pd.Index] = []
        self.test_indices: List[pd.Index] = []

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
        Xy = pd.concat([X, y], axis=1)

        Xy = (
            Xy.dropna().reset_index()
        )  # drops row, corresponding with a country & period, if either a feature or the target is missing.
        # All dates in the composite
        unique_times: pd.arrays.DatetimeArray = Xy["real_date"].sort_values().unique()

        if self.min_periods is not None:
            assert self.min_periods <= len(
                unique_times
            ), """The minimum number of time periods for each cross-section in the first split cannot be larger than
            the number of unique dates in the entire dataframe X"""

        if self.train_intervals:
            # Bools of dates with at least min_cids cross-sections
            init_mask: pd.Series = (
                Xy.groupby("real_date").size().sort_index() >= self.min_cids
            )
            # Oldest real date with sufficient cross-sections
            date_first_min_cids: pd.Timestamp = (
                init_mask[init_mask == True].reset_index().real_date.min()
            )
            # Oldest end date of training set
            date_last_train: pd.Timestamp = unique_times[
                np.where(unique_times == date_first_min_cids)[0][0]
                + self.min_periods
                - 1
            ]
            # Dates from oldest to youngest last training date
            unique_times_train: pd.arrays.DatetimeArray = unique_times[
                np.where(unique_times == date_last_train)[0][0] + 1 : -self.test_size
            ]
            # Just calulate the number of splits
            self.n_splits: int = (
                int(np.ceil(len(unique_times_train) / self.train_intervals)) + 1
            )

        return self.n_splits

    def split(self, X: pd.DataFrame, y: pd.DataFrame) -> Iterator[Tuple[int,int]]:
        """
        Method that splits a wide format Pandas (panel) dataframe into walk-forward validation sets. 

        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.

        :return <Iterator[Tuple[int,int]]> splits: iterator of (train,test) indices for walk-forward validation.
        """
        # TODO: most lines are the same as in get_n_splits. Can we combine them?
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
        Xy = Xy.dropna().reset_index()
        unique_times: pd.arrays.DatetimeArray = Xy["real_date"].sort_values().unique()
        if self.min_periods is not None:
            assert self.min_periods <= len(
                unique_times
            ), """The minimum number of time periods for each cross-section in the first split cannot be larger than
            the number of unique dates in the entire dataframe X"""

            # (1) Determine the splits prior to aggregation
            # The approach is to calculate a numpy array for the unique dates in each training split.
            # The procedure is different depending on whether or not n_splits is specified.
        if self.train_intervals:
            # First determine the dates for the first training set, determined by 'min_cids' and 'min_periods'.
            # (a) obtain a boolean mask of dates for which at least 'min_cids' cross-sections are available over the panel
            init_mask: pd.Series = (
                Xy.groupby("real_date").size().sort_index() >= self.min_cids
            )
            # (b) obtain the earliest date for which the mask is true i.e. the earliest date with 'min_cids' cross-sections available
            date_first_min_cids: pd.Timestamp = (
                init_mask[init_mask == True].reset_index().real_date.min()
            )
            # (c) find the last valid date in the first training set, which is the date 'min_periods' - 1 after date_first_min_cids
            date_last_train: pd.Timestamp = unique_times[
                np.where(unique_times == date_first_min_cids)[0][0]
                + self.min_periods
                - 1
            ]
            # (d) determine the unique dates in the training sets after the first. 
            # This is done by collecting all dates in the panel after the last date in the first training set and before the last 'self.test_size' dates,
            # calculating the number of splits ('self.n_splits') required to split these dates into distinct training intervals of length 'self.train_intervals' (where possible)
            # and finally splitting the mentioned dates into 'self.n_splits' splits (post the first split, determined by 'self.min_cids' and 'self.min_periods').
            unique_times_train: pd.arrays.DatetimeArray = unique_times[
                np.where(unique_times == date_last_train)[0][0] + 1 : -self.test_size
            ]
            self.n_splits: int = int(
                np.ceil(len(unique_times_train) / self.train_intervals)
            )
            train_splits_basic: List = np.array_split(unique_times_train, self.n_splits)
            # (e) add the first training set to the list of training splits, so that the dates that constitute each training split are together. 
            train_splits_basic.insert(
                0,
                pd.arrays.DatetimeArray(
                    np.array(
                        sorted(Xy["real_date"][Xy["real_date"] <= date_last_train].unique()),
                        dtype="datetime64[ns]",
                    )
                ),
            )
            # (f) increment n_splits by one to account for the first training set
            # This is needed in order to loop over the training dates later on to determine the dataframe indices that are returned.
            self.n_splits += 1
        else:
            # In the case where n_splits is specified, the logic is simpler. 
            # The unique dates until the latest date 'self.test_size' before the end of the panel are obtained and split into 'self.n_splits' equal (or as equal as possible) splits.
            unique_times_train: pd.arrays.DatetimeArray = unique_times[
                : -self.test_size
            ]
            train_splits_basic: List[pd.arrays.DatetimeArray] = np.array_split(
                unique_times_train, self.n_splits
            )

        # (2) Aggregate or roll training dates in train_splits_basic depending on whether or not max_periods is specified
        # This is done by looping over the training date splits and concatenating the dates in each split to the previous split.
        # If max_periods is specified, then the concatenated dates are cut off at the maximum number of periods specified.
        train_splits: List[np.array] = [train_splits_basic[0] if not self.max_periods else train_splits_basic[0][-self.min_periods:]]
        for i in range(1, self.n_splits):
            train_splits.append(
                np.concatenate([train_splits[i - 1], train_splits_basic[i]])
            )
            if self.max_periods:
                train_splits[i] = train_splits[i][-self.max_periods :]

        # (3) Lastly, create the train and test indices.
        # train_splits now comprises the unique dates in each training split. Thus, to return indices, we loop through each date split,
        for split in train_splits:
            self.train_indices.append(
                Xy.index[Xy["real_date"].isin(split)]
            )
            self.test_indices.append(
                Xy.index[Xy["real_date"].isin(unique_times[np.where(unique_times == np.max(split))[0][0] + 1:np.where(unique_times == np.max(split))[0][0] + 1 + self.test_size])]
            )

        return zip(self.train_indices, self.test_indices)


if __name__ == "__main__":
    from macrosynergy.management.simulate_quantamental_data import make_qdf

    """ Example 1: Balanced panel """

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    dfd = dfd.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd.drop(columns=["XR"])
    y = dfd["XR"]

    # a) n_splits = 4, test_size = default (21), aggregation
    print("--------------------")
    print("--------------------")
    print("Balanced panel: n_splits = 4, test_size = 21 days")
    print("--------------------")
    print("--------------------\n")
    splitter = PanelTimeSeriesSplit(n_splits=4)
    print(f"Number of splits: {splitter.get_n_splits(X,y)}")
    for idx, (train_idxs, test_idxs) in enumerate(splitter.split(X, y)):
        train_i = pd.concat([X.iloc[train_idxs], y.iloc[train_idxs]], axis=1)
        test_i = pd.concat([X.iloc[test_idxs], y.iloc[test_idxs]], axis=1)
        print("--------------------")
        print(f"Split {idx+1}:")
        print("--------------------")
        print(f"Concatenated training set at iteration {idx+1}")
        print(train_i)
        print(f"Concatenated test set at iteration {idx+1}")
        print(test_i)

    # # b) n_splits = 4, test_size = default (21), rolling window: most recent 21 days
    # print("--------------------")
    # print("--------------------")
    # print(
    #     "Balanced panel: n_splits = 4, test_size = 21 days, rolling window: most recent quarter (21 x 3 days roughly)"
    # )
    # print("--------------------")
    # print("--------------------\n")
    # splitter = PanelTimeSeriesSplit(n_splits=4, max_periods=21 * 3)
    # print(f"Number of splits: {splitter.get_n_splits(X,y)}")
    # for idx, (train_idxs, test_idxs) in enumerate(splitter.split(X, y)):
    #     train_i = pd.concat([X.iloc[train_idxs], y.iloc[train_idxs]], axis=1)
    #     test_i = pd.concat([X.iloc[test_idxs], y.iloc[test_idxs]], axis=1)
    #     # print("--------------------")
    #     print(f"Split {idx+1}:")
    #     # print("--------------------")
    #     print(f"Concatenated training set at iteration {idx+1}")
    #     print(train_i)
    #     print(f"Concatenated test set at iteration {idx+1}")
    #     print(test_i)

    # c) train_intervals = 1, test_size = 1, min_periods = 21 , min_cids = 4
    # This configuration means that on each iteration, the newest information state is added to the training set
    # and only the next date is in the test set.
    print("--------------------")
    print("--------------------")
    print(
        "Balanced panel: train_intervals = 1 day, test_size = 1 day, min_periods = 21, min_cids = 4"
    )
    print("--------------------")
    print("--------------------\n")
    # Since this is a balanced panel, the first set should be the first 21 dates in the whole dataframe
    X_reset = X.reset_index()
    unique_dates_X = np.unique(X_reset.real_date)

    splitter = PanelTimeSeriesSplit(train_intervals=1, test_size=1, min_periods = 21, min_cids=4)
    print(f"Number of splits: {splitter.get_n_splits(X,y)}")

    for idx, (train_idxs, test_idxs) in enumerate(splitter.split(X, y)):
        train_dates = np.unique(X_reset.real_date.iloc[train_idxs])
        test_dates = np.unique(X_reset.real_date.iloc[test_idxs])
        if idx == 0:
            assert (train_dates == unique_dates_X[:21]).all(), "The dates in the first training set aren't equal to the first 21 dates in X"

        # check that there is only a single date in the test set
        assert len(test_dates) == 1, "There are multiple test dates."

        # check that the last training date is immediately before the test date
        assert unique_dates_X[np.where(unique_dates_X == train_dates[-1])[0][0] + 1] == test_dates, "There is a split where the last training date does not precede the test date."

    print("The first training split comprises the first 21 days of the original input dataframe. Good.")
    print("The test splits all contain a single date and immediate proceed the last training date. Good.")

    # d) train_intervals = 21, test_size = 5, max_periods = 21*3, min_periods = 21 , min_cids = 3
    # This configuration means that on each iteration, the newest information state is added to the training set 
    # and only the next date is in the test set. 
    print("--------------------")
    print("--------------------")
    print(
        "Balanced panel: train_intervals = 1 month, test_size = 5 days, max_periods = 1 quarter, min_periods = 21, min_cids = 3"
    )
    print("--------------------")
    print("--------------------\n")
    splitter = PanelTimeSeriesSplit(train_intervals=21, test_size=5, max_periods=21*3, min_periods = 21, min_cids=3)
    print(f"Number of splits: {splitter.get_n_splits(X,y)}")

    for idx, (train_idxs, test_idxs) in enumerate(splitter.split(X, y)):
        train_dates = np.unique(X_reset.real_date.iloc[train_idxs])
        test_dates = np.unique(X_reset.real_date.iloc[test_idxs])
        if idx == 0:
            assert (train_dates == unique_dates_X[:21]).all(), "The dates in the first training set aren't equal to the first 21 dates in X"

        # check that there are only five unique dates in the test set
        assert len(test_dates) == 5, "There is a test set with unique dates different to five in it."

        # check that the last training date is immediately before the test date
        assert unique_dates_X[np.where(unique_dates_X == train_dates[-1])[0][0] + 1] == test_dates[0], "There is a split where the last training date does not immediately precede the test date."

        # check that the number of unique dates in each training split does not exceed a quarter. 
        assert len(train_dates) <= 21*3, "There exists a training split with samples from before the previous quarter."

    # TODO: add cases for an unbalanced panel now.