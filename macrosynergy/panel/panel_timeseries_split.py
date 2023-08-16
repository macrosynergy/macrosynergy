import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator
from typing import Union, Optional, List, Tuple, Iterable, Dict, Callable, Any


class PanelTimeSeriesSplit(BaseCrossValidator):
    """
    Class for the production of cross-section splits for panel data.

    :param <int> n_splits: number of time splits to make. If this is not specified, then
        the split will be governed by sequential training intervals. Either n_splits or train_intervals
        must be specified (not both).
    :param <int> train_intervals: training interval length in time periods for sequential
        training. This is the number of periods by which the training set is expanded at
        each subsequent split. Default is 21.
    :param <int> test_size: test set length for interval training. This is the number of
        periods to use for the test set subsequent to the training set. Default is 21.
    :param <int> max_periods: maximum length of the training set in interval training.
        If the maximum is exceeded then the earliest periods are cut off.
        Default is None.
    :param <int> min_periods: minimum number of time periods required for the initial
        training set. Default is 500. Only used if train_intervals is specified.
    :param <int> min_cids: minimum number of cross-sections required for the initial
        training set. Default is 4. Only used if train_intervals is specified.

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
        max_periods: Optional[int] = None,
        min_periods: Optional[int] = 500,
        min_cids: Optional[int] = 4,
    ):
        if n_splits is not None:
            train_intervals = None
            min_periods = None
            min_cids = None
        self.n_splits: int = n_splits
        self.train_intervals: int = train_intervals
        self.test_size: int = test_size
        self.max_periods: int = max_periods
        self.min_periods: int = min_periods
        self.min_cids: int = min_cids

        assert (self.n_splits is not None) ^ (
            self.train_intervals is not None
        ), "Either n_splits or train_intervals must be specified, but not both."

        if self.train_intervals is None:
            assert (
                self.min_periods is None
            ), "min_periods unnecessary if n_splits is specified."
            assert (
                self.min_cids is None
            ), "min_cids unnecessary if n_splits is specified."
        else:
            assert (
                self.min_periods is not None
            ), "min_periods must be specified when train_intervals are specified."
            assert (
                self.min_cids is not None
            ), "min_cids must be specified when train_intervals are specified."
            assert ((self.min_cids > 0) & (type(self.min_cids) == int)), "min_cids must be an integer greater than 0."
            assert ((self.min_periods > 0) & (type(self.min_periods) == int)), "min_periods must be an integer greater than 0."

        assert self.test_size is not None, "test_size must be specified."
        assert ((self.test_size > 0) & (type(self.test_size) == int)), "test_size must be an integer greater than 0."
        self.train_indices: List[pd.Index] = []
        self.test_indices: List[pd.Index] = []

    def get_n_splits(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Returns the number of splitting iterations in the cross-validator.

        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the frame must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        """
        X = pd.concat([X,y], axis=1)
        X = (
            X.dropna()
        )  # drops row, corresponding with a country & period, if either a feature or the target is missing
        unique_times: pd.arrays.DatetimeArray = (
            X.reset_index()["real_date"].sort_values().unique()
        )
        if self.train_intervals:
            init_mask: pd.Series = X.groupby(level=1).size() == self.min_cids
            date_first_min_cids: pd.Timestamp = (
                init_mask[init_mask == True].reset_index().real_date.min()
            )
            date_last_train: pd.Timestamp = unique_times[
                np.where(unique_times == date_first_min_cids)[0][0]
                + self.min_periods
                - 1
            ]
            # Determine the remaining splits based on train_intervals
            unique_times_train: pd.arrays.DatetimeArray = unique_times[
                np.where(unique_times == date_last_train)[0][0] + 1 : -self.test_size
            ]
            self.n_splits: int = (
                int(np.ceil(len(unique_times_train) / self.train_intervals)) + 1
            )

        return self.n_splits

    def split(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Splitter method.
        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators,
            multi-indexed by (cross-section, date). The dates must be in datetime format.
            Otherwise the dataframe must be in wide format: each feature is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by
            (cross-section, date). The dates must be in datetime format.
        """
        assert X.index.equals(y.index), "The indices of the input dataframe X and the output dataframe y don't match."
        X = pd.concat([X,y], axis=1)
        X = (
            X.dropna()
        )  # drops row, corresponding with a country & period, if either a feature or the target is missing
        unique_times: pd.arrays.DatetimeArray = (
            X.reset_index()["real_date"].sort_values().unique()
        )
        if self.min_periods is not None:
            assert self.min_periods <= len(
                unique_times
            ), "The minimum number of time periods for each cross-section in the first split cannot be larger than the number of unique dates in the entire dataframe X"            

        if self.train_intervals:
            # (1) Determine the splits prior to aggregation
            # Deal with the initial split determined by min_cids and min_periods
            init_mask: pd.Series = X.groupby(level=1).size() == self.min_cids
            date_first_min_cids: pd.Timestamp = (
                init_mask[init_mask == True].reset_index().real_date.min()
            )
            date_last_train: pd.Timestamp = unique_times[
                np.where(unique_times == date_first_min_cids)[0][0]
                + self.min_periods
                - 1
            ]
            train_idxs: pd.Index = X.reset_index().index[
                X.reset_index()["real_date"] <= date_last_train
            ]
            # Determine the remaining splits based on train_intervals
            unique_times_train: pd.arrays.DatetimeArray = unique_times[
                np.where(unique_times == date_last_train)[0][0] + 1 : -self.test_size
            ]
            self.n_splits: int = int(
                np.ceil(len(unique_times_train) / self.train_intervals)
            )
            train_splits_basic: List = np.array_split(unique_times_train, self.n_splits)
            # insert the unique dates for the already-created first split at the start of train_splits_basic
            train_splits_basic.insert(
                0,
                pd.arrays.DatetimeArray(
                    np.array(
                        sorted(X.reset_index().real_date.iloc[train_idxs].unique()),
                        dtype="datetime64[ns]",
                    )
                ),
            )
            # need to add one to n_splits in order to take into account the split already determined by min_cids and min_periods,
            # because n_splits was determined by the number of train_intervals starting from the second split
            self.n_splits += 1
        else:
            # (1) Determine the splits prior to agglomeration
            unique_times_train: pd.arrays.DatetimeArray = unique_times[
                : -self.test_size
            ]
            train_splits_basic: List[pd.arrays.DatetimeArray] = np.array_split(
                unique_times_train, self.n_splits
            )

        # (2) aggregate each split
        train_splits: List[np.array] = [
            np.concatenate(train_splits_basic[: i + 1]) for i in range(self.n_splits)
        ]

        # (3) If self.max_periods is specified, adjust each of the splits to only have 
        # he self.max_periods most recent times in each split
        if self.max_periods:
            for split_idx in range(len(train_splits)):
                train_splits[split_idx] = train_splits[split_idx][-self.max_periods :]

        # (4) Create the train and test indices
        for split in train_splits:
            smallest_date: np.datetime64 = min(split)
            largest_date: np.datetime64 = max(split)
            self.train_indices.append(
                X.reset_index().index[
                    (X.reset_index()["real_date"] >= smallest_date)
                    & (X.reset_index()["real_date"] <= largest_date)
                ]
            )
            self.test_indices.append(
                X.reset_index().index[
                    (X.reset_index()["real_date"] > largest_date)
                    & (
                        X.reset_index()["real_date"]
                        <= unique_times[
                            np.where(unique_times == largest_date)[0][0]
                            + self.test_size
                        ]
                    )
                ]
            )

        return zip(self.train_indices, self.test_indices)


"""
---------------
Testing
---------------
"""

if __name__ == "__main__":
    from macrosynergy.management.simulate_quantamental_data import make_qdf

    """ Example 1: Balanced panel """

    cids = ['AUD', 'CAD', 'GBP', 'USD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index=cids,
                           columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0, 1]
    df_cids.loc['CAD'] = ['2000-01-01', '2020-12-31', 0, 1]
    df_cids.loc['GBP'] = ['2000-01-01', '2020-12-31', 0, 1]
    df_cids.loc['USD'] = ['2000-01-01', '2020-12-31', 0, 1]

    cols = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef']
    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2000-01-01', '2020-12-31', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH'] = ['2000-01-01', '2020-12-31', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2000-01-01', '2020-12-31', 1, 2, 0.8, 0.5]

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
    for idx, (train_idxs, test_idxs) in enumerate(splitter.split(X, y)):
        train_i = pd.concat([X.iloc[train_idxs], y.iloc[train_idxs]],axis=1)
        test_i = pd.concat([X.iloc[test_idxs], y.iloc[test_idxs]],axis=1)
        #print("--------------------")
        print(f"Split {idx+1}:")
        #print("--------------------")
        print(f"Concatenated training set at iteration {idx+1}")
        print(train_i)
        print(f"Concatenated test set at iteration {idx+1}")
        print(test_i)
    # b) n_splits = 4, test_size = default (21), rolling window: most recent 21 days
    print("--------------------")
    print("--------------------")
    print("Balanced panel: n_splits = 4, test_size = 21 days, rolling window: most recent quarter (21 x 3 days roughly)")
    print("--------------------")
    print("--------------------\n")
    splitter = PanelTimeSeriesSplit(n_splits=4,max_periods=21*3)
    for idx, (train_idxs, test_idxs) in enumerate(splitter.split(X, y)):
        train_i = pd.concat([X.iloc[train_idxs], y.iloc[train_idxs]],axis=1)
        test_i = pd.concat([X.iloc[test_idxs], y.iloc[test_idxs]],axis=1)
        #print("--------------------")
        print(f"Split {idx+1}:")
        #print("--------------------")
        print(f"Concatenated training set at iteration {idx+1}")
        print(train_i)
        print(f"Concatenated test set at iteration {idx+1}")
        print(test_i)

"""if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats

    import macrosynergy.panel as msp
    import macrosynergy.management as msm

    from macrosynergy.download import JPMaQSDownload

    from timeit import default_timer as timer
    from datetime import timedelta, date, datetime

    import warnings

    # Imports

    cids_dm = [
        "AUD",
        "EUR",
        "GBP",
        "JPY",
        "NZD",
        "SEK",
        "USD",
    ]  # DM currency areas

    cids = cids_dm

    main = [
        "RYLDIRS05Y_NSA",
        "INTRGDPv5Y_NSA_P1M1ML12_3MMA",
        "CPIC_SJA_P6M6ML6AR",
        "INFTEFF_NSA",
        "PCREDITBN_SJA_P1M1ML12",
        "SPACSGDP_NSA_D1M1ML1",
        "RGDP_SA_P1Q1QL4_20QMA",
    ]
    econ = []
    mark = ["DU05YXR_NSA"]

    xcats = main + econ + mark

    # Download series from J.P. Morgan DataQuery by tickers

    start_date = "2000-01-01"
    end_date = "2022-12-31"

    tickers = [cid + "_" + xcat for cid in cids for xcat in xcats]
    print(f"Maximum number of tickers is {len(tickers)}")

    # Retrieve credentials

    oauth_id = os.getenv("DQ_CLIENT_ID")  # Replace with own client ID
    oauth_secret = os.getenv("DQ_CLIENT_SECRET")  # Replace with own secret

    # Download from DataQuery

    # with JPMaQSDownload(local_path=LOCAL_PATH) as downloader:
    with JPMaQSDownload(client_id=oauth_id, client_secret=oauth_secret) as downloader:
        start = timer()
        df = downloader.download(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            metrics=["value"],
            suppress_warning=True,
            show_progress=True,
        )
        end = timer()

    dfd = df

    print("Download time from DQ: " + str(timedelta(seconds=end - start)))

    xcatx = xcats
    cidx = cids

    train = msm.reduce_df(df=dfd, xcats=xcatx, cids=cidx, end="2020-12-31")
    valid = msm.reduce_df(
        df=dfd,
        xcats=xcatx,
        cids=cidx,
        start="2020-12-01",
        end="2021-12-31",
    )
    test = msm.reduce_df(
        df=dfd,
        xcats=xcatx,
        cids=cidx,
        start="2021-12-01",
        end="2022-12-31",
    )

    calcs = [
        "XCPIC_SJA_P6M6ML6AR = CPIC_SJA_P6M6ML6AR - INFTEFF_NSA",
        "XPCREDITBN_SJA_P1M1ML12 = PCREDITBN_SJA_P1M1ML12 - INFTEFF_NSA - RGDP_SA_P1Q1QL4_20QMA",
    ]

    dfa = msp.panel_calculator(train, calcs=calcs, cids=cidx)
    train = msm.update_df(train, dfa)

    train_wide = msm.categories_df(
        df=train, xcats=xcatx, cids=cidx, freq="M", lag=1, xcat_aggs=["mean", "sum"]
    )

    splitter = PanelTimeSeriesSplit(train_intervals=6, test_size=3, min_periods=12, min_cids=3)
    splitter.split(train_wide.iloc[:,:-1],train_wide.iloc[:,-1])

    # splitter = PanelTimeSeriesSplit(train_intervals=6, test_size=4, min_periods=12, min_cids=3, max_periods=3)
    # splitter.split(train_wide)

    # splitter = PanelTimeSeriesSplit(n_splits=5,test_size=6)
    # splitter.split(train_wide)

    # splitter = PanelTimeSeriesSplit(n_splits=10,test_size=1)
    # splitter.split(train_wide)

    #splitter = PanelTimeSeriesSplit(n_splits=10, test_size=1, max_periods=12)
    #print(splitter.get_n_splits(train_wide))
    ##splitter.split(train_wide)

    #splitter = PanelTimeSeriesSplit(
    #    train_intervals=1, test_size=1, max_periods=12, min_cids=3, min_periods=12
    #)
    #splitter.split(train_wide)
    #print(splitter.get_n_splits(train_wide))"""