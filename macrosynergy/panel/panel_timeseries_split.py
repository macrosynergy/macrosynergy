import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator
from typing import Union, Optional, List, Tuple, Iterable, Dict, Callable, Any 


class PanelTimeSeriesSplit(object):
    """
    This class provides a cross-validator for panel data. It could also be used for rolling model validation and training.

    It provides train/validation indices to split the panel samples, observed at fixed dates for a number of cross-sections,
    in sequential train/validation sets. Unlike the sklearn class `TimeSeriesSplit`, this class makes splits based on the observation dates
    as opposed to the sample indices.

    The splitting occurs in one of two ways: (1) the number of splits are directly specified, or (2) the number of splits are determined by the
    number of forward time periods to expand the training set at each iteration. Other parameters determine the configurations of the splits made 
    and depend on the splitting method used. Default is defining an expanding training window.

    :param <int> n_splits: number of time splits to make. If None, then train_intervals must be specified. Default is None.
    :param <int> train_intervals: number of forward time periods to expand the training set at each iteration. If None, then n_splits must be specified. Default is 21.
    :param <int> test_size: number of time periods for the subsequent validation set at each iteration. Default is 21. Must be specified.
    :param <int> max_periods: maximum number of time periods to include in the training set before earliest time periods are cut off. Default is None.
    :param <int> min_periods: minimum number of time periods to include in the initial training set. Default is 500. Only used if train_intervals is specified.
    :param <int> min_cids: minimum number of cross-sections to include in the initial training set. Default is 4. Only used if train_intervals is specified.
    """
    def __init__(self, n_splits: Optional[int] = None, train_intervals: Optional[int] = 21, test_size: int = 21, max_periods: Optional[int] = None, min_periods: Optional[int] = 500, min_cids: Optional[int] = 4):
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

        assert (self.n_splits is not None) ^ (self.train_intervals is not None), \
            "Either n_splits or train_intervals must be specified, but not both."

        if self.train_intervals is None:
            assert self.min_periods is None, "min_periods unnecessary if n_splits is specified."
            assert self.min_cids is None, "min_cids unnecessary if n_splits is specified."
        else:
            assert self.min_periods is not None, "min_periods must be specified when train_intervals are specified."
            assert self.min_cids is not None, "min_cids must be specified when train_intervals are specified."
            assert self.min_cids > 0, "min_cids must be greater than 0."
            assert self.min_periods > 0, "min_periods must be greater than 0."
            
        assert self.test_size is not None, "test_size must be specified."
        assert self.test_size > 0, "test_size must be greater than 0."
        self.train_indices: List = []
        self.test_indices: List = []
 
    def get_n_splits(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """
        Returns the number of splitting iterations in the cross-validator.
        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators, multi-indexed by (cross-section, date). The dates must be in datetime format.
                                 The dataframe must be in wide format, i.e. each feature/indicator is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by (cross-section, date). The dates must be in datetime format. This isn't used and 
                              is only included to be consistent with the sklearn API.
        """
        X = X.dropna() # Since the dataframe is in long-format, this will only drop rows where not all features are provided
        unique_times = X.reset_index()["real_date"].sort_values().unique()
        if self.train_intervals:
            init_mask = X.groupby(level=1).size() == self.min_cids
            date_first_min_cids = init_mask[init_mask == True].reset_index().real_date.min()
            date_last_train = unique_times[np.where(unique_times == date_first_min_cids)[0][0] + self.min_periods - 1]
            train_idxs = X.reset_index().index[X.reset_index()["real_date"] <= date_last_train]
            # Determine the remaining splits based on train_intervals
            unique_times_train = unique_times[np.where(unique_times == date_last_train)[0][0] + 1:-self.test_size]
            self.n_splits = int(np.ceil(len(unique_times_train) / self.train_intervals)) + 1

        return self.n_splits 

    def split(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """
        Splitter method.
        :param <pd.DataFrame> X: Pandas dataframe of features/quantamental indicators, multi-indexed by (cross-section, date). The dates must be in datetime format.
                                 The dataframe must be in wide format, i.e. each feature/indicator is a column.
        :param <pd.DataFrame> y: Pandas dataframe of target variable, multi-indexed by (cross-section, date). The dates must be in datetime format. This isn't used and 
                              is only included to be consistent with the sklearn API.
        """
        X = X.dropna() # Since the dataframe is in long-format, this will only drop rows where not all features are provided
        unique_times = X.reset_index()["real_date"].sort_values().unique()
        if self.min_periods is not None:
            assert self.min_periods <= len(unique_times), "The minimum number of time periods for the first split must be less than or equal to the number of time periods in the dataframe."

        if self.train_intervals:
            # (1) Determine the splits prior to aggregation
            # Deal with the initial split determined by min_cids and min_periods
            init_mask = X.groupby(level=1).size() == self.min_cids
            date_first_min_cids = init_mask[init_mask == True].reset_index().real_date.min()
            date_last_train = unique_times[np.where(unique_times == date_first_min_cids)[0][0] + self.min_periods - 1]
            train_idxs = X.reset_index().index[X.reset_index()["real_date"] <= date_last_train]
            # Determine the remaining splits based on train_intervals
            unique_times_train = unique_times[np.where(unique_times == date_last_train)[0][0] + 1:-self.test_size]
            self.n_splits = int(np.ceil(len(unique_times_train) / self.train_intervals))
            train_splits_basic = np.array_split(unique_times_train,self.n_splits)
            train_splits_basic.insert(0, pd.arrays.DatetimeArray(np.array(sorted(X.reset_index().real_date.iloc[train_idxs].unique()), dtype="datetime64[ns]")))
            # need to add one to n_splits because n_splits was determined by the number of train_intervals starting from the second split
            # need to take into account the split determined by min_cids and min_periods
            self.n_splits += 1
        else:
            # (1) Determine the splits prior to agglomeration
            unique_times_train = unique_times[:-self.test_size]
            train_splits_basic = np.array_split(unique_times_train,self.n_splits)
        
        # (2) aggregate each split 
        train_splits = [np.concatenate(train_splits_basic[:i+1]) for i in range(self.n_splits)]

        # (3) If self.max_periods is specified, adjust each of the splits to only have the self.max_periods most recent times in each split
        if self.max_periods:
            for split_idx in range(len(train_splits)):
                train_splits[split_idx] = train_splits[split_idx][-self.max_periods:]

        # (4) Create the train and test indices
        for split in train_splits:
            smallest_date = min(split) 
            largest_date = max(split)
            self.train_indices.append(X.reset_index().index[(X.reset_index()["real_date"] >= smallest_date) & (X.reset_index()["real_date"] <= largest_date)])
            self.test_indices.append(X.reset_index().index[(X.reset_index()["real_date"] > largest_date) & (X.reset_index()["real_date"] <= unique_times[np.where(unique_times==largest_date)[0][0] + self.test_size])])
            
        return zip(self.train_indices, self.test_indices)
        
"""
---------------
Test class
TODO: amend later to not download data but instead use make_qdf()
---------------
"""

if __name__ == "__main__":
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
    mark = []

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

    #splitter = PanelTimeSeriesSplit(train_intervals=6, test_size=3, min_periods=12, min_cids=3)
    #splitter.split(train_wide)
    
    #splitter = PanelTimeSeriesSplit(train_intervals=6, test_size=4, min_periods=12, min_cids=3, max_periods=3)
    #splitter.split(train_wide)

    #splitter = PanelTimeSeriesSplit(n_splits=5,test_size=6)
    #splitter.split(train_wide)

    #splitter = PanelTimeSeriesSplit(n_splits=10,test_size=1)
    #splitter.split(train_wide)

    splitter = PanelTimeSeriesSplit(n_splits=10,test_size=1, max_periods=12)
    splitter.split(train_wide)
    print(splitter.get_n_splits(train_wide))
    
    splitter = PanelTimeSeriesSplit(train_intervals=1,test_size=1, max_periods=12, min_cids=3, min_periods=12)
    splitter.split(train_wide)
    print(splitter.get_n_splits(train_wide))