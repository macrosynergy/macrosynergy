# NaivePnL class
# [1] uses categories_df in same way as SignalReturnRelations
# [2] allows multiple signals
# [3] allows simple signal transformations (z-scoring, trimming, digital, vol-weight)
# [4] allows periodicity of signal to be set
# [5] allows ex-post vol-scaling of Pnl
# [6] allows equally weighted long-only benchmark
# [7] allows to set slippage in days
#
# Produces daily-frequency statistics:
# [1] chart of PnLs
# [2] table of key performance statistics
# Annualized return, ASD, Sharpe, Sortino, Calmar, positive years ratio
# [3] chart of cross_section PnLs
# [3] table of cross-section contributions
#
# Implementation
# [1] at initiation creates df of basic transformations
# [2] pnl_chart() method
# [3] pnl_table() method
# [4] cs_pnl_charts() method
# [5] cs_pnl_table method

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as skm
from scipy import stats

from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import categories_df


class NaivePnL:

    """Estimates and analyses naive illustrative PnLs with limited signal options and disregarding transaction costs

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value.
    :param <str> ret: return category.
    :param <List[str]> sigs: signal categories.
    :param <List[str]> cids: cross sections to be considered. Default is all in the data frame.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.

    """
    def __init__(self, df: pd.DataFrame, ret: str, sigs: str, cids: List[str] = None,
                 start: str = None, end: str = None, fwin: int = 1, blacklist: dict = None,
                 freq: str = 'M'):

        pass


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD',] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD',] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP',] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD',] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR',] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY',] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH',] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL',] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    black = {'AUD': ['2006-01-01', '2015-12-31'], 'GBP': ['2012-01-01', '2100-01-01']}

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)