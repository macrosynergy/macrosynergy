import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator
from typing import Union, Optional, List, Tuple, Iterable, Dict, Callable, Any 


class PanelTimeSeriesSplit(object):
    """
    Class for creating a collection of (train_idx, test_idx) pairs from a panel dataset for use in a machine learning development pipeline.
    The class enables: walk-forward validation on panel data, rolling model training and rolling model validation. The initial training split is determined 
    by specifying a minimum number of periods ('min_periods') for a specified minimum number of cross-sections ('min_cids'). Subsequent test and training splits 
    are governed by the 'test_size', 'train_intervals' and 'n_splits' parameters. Whether the splits aggregate or roll over time is determined by the 'max_periods' parameter.

    :param <int> n_splits: number of time splits to make. If None, then train_intervals must be specified. Default is None.
    :param <int> train_intervals: number of forward time periods to expand the training set at each iteration to use. If None, then n_splits must be specified. Default is 21.
    :param <int> test_size: number of time periods for the subsequent test set at each iteration. Default is 21.
    :param <int> max_periods: maximum number of time periods to include in the training set before earliest time periods are cut off. Default is None.
    :param <int> min_periods: minimum number of time periods to include in the initial training set. Default is 500.
    :param <int> min_cids: minimum number of cross-sections to include in the initial training set. Default is None.
    """
    def __init__(self, n_splits: Optional[int] = None, train_intervals: int = 21, test_size: int = 21, max_periods: Optional[int] = None, min_periods: Optional[int] = 500, min_cids: Optional[int] = None):
        self.n_splits: int = n_splits
        self.train_intervals: int = train_intervals
        self.test_size: int = test_size
        self.max_periods: int = max_periods
        self.min_periods: int = min_periods
        self.min_cids: int = min_cids

        assert (self.n_splits is not None) ^ (self.train_intervals) is not None, \
            "Either n_splits or train_intervals must be specified, but not both."
        assert self.min_periods is not None, "min_periods must be specified."
        assert self.min_periods > 0, "min_periods must be greater than 0."

        self.train_indices: List = []
        self.test_indices: List = []

    def split(self, X: pd.DataFrame, y=None):
        """
        Method to generate the indices that split the data into a collection of training and test sets.

        :param <pd.DataFrame> X: panel dataset of features. This should be multi-indexed by cross-section and time. This must be specified.
        :param <pd.Series> y: panel dataset of labels. Default is None. This should be multi-indexed by cross-section and time.
        """
        X = X.dropna()
        unique_times = X.reset_index()["real_date"].sort_values().unique()
    
        # (1) Get the first time at which all features are available for min_cids number of cross-sections
        init_mask = X.groupby(level=1).size() == self.min_cids
        date_first_min_cids = init_mask[init_mask == True].reset_index().real_date.min()
        # (2) In 'min_periods' number of time periods, at least 'min_periods' samples will be available for at least 'min_cids' number of cross-sections 
        # The below line works because indicators are forward filled, meaning that there will be an indicator value for each date post the starting date. 
        date_last_train = unique_times[np.where(unique_times == date_first_min_cids)[0] + self.min_periods - 1]
        date_last_test = unique_times[np.where(unique_times == date_last_train)[0] + self.test_size]
        # (3) Get indices of all samples in X with observation dates <= date_last_min_cids
        train1_idxs = X.reset_index().index[X.reset_index()["real_date"] <= date_last_train]
        # (4) Get indices of all test samples corresponding with the initial training set
        # TODO: now move onto the test indices for this and then start a loop until the end of the dataframe
        test1_idxs = X.reset_index().index[(X.reset_index()["real_date"] > date_last_train) & (X.reset_index()["real_date"] <= date_last_test)]
        # (5) Return the first set of training and test indices
        self.train_indices.append(train1_idxs)
        self.test_indices.append(test1_idxs)

        yield (train1_idxs, test1_idxs)
        # (6) Iteratively create the rest of the training and test indices for the panel walk-forward validation
        first_date = unique_times[np.where(unique_times == date_last_test)[0] + 1]
        last_date = unique_times[-self.test_size-1]

        # finaly yield 'True' to indicate that the generator is finished
        yield True

"""
---------------
Test class
---------------
"""
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname('.'))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats

    import macrosynergy.panel as msp
    import macrosynergy.management as msm

    from macrosynergy.download import JPMaQSDownload
    from macrosynergy.dev.local import LocalCache
    LOCAL_PATH = r"~\Macrosynergy\Macrosynergy - Documents\SharedData\JPMaQSTickers"

    from timeit import default_timer as timer
    from datetime import timedelta, date, datetime

    import warnings

    warnings.simplefilter("ignore")

    # Imports

    cids_dm = [
        "AUD",
        "CHF",
        "EUR",
        "GBP",
        "JPY",
        "NOK",
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

    with LocalCache(local_path=LOCAL_PATH) as downloader:
    # with JPMaQSDownload(client_id=oauth_id, client_secret=oauth_secret) as downloader:
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

    splitter: PanelTimeSeriesSplit = PanelTimeSeriesSplit(
        train_intervals=6, test_size=2, min_periods=12, min_cids=len(cids)
    )
    # splitter.bleh(train_wide)
    x = splitter.split(X=train_wide, y=None,)

    print(x)
