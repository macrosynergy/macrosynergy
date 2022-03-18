import numpy as np
import pandas as pd
from typing import List, Union

def hedge_ratio(df: pd.DataFrame, xcats: List[str] = None, cids: List[str] = None,
                hedge_return: str = None,
                start: str = None, end: str = None,
                meth: str = 'ols', oos: bool = True, rfreq: str = 'm',
                minobs: int = 24,
                blacklist: dict = None):

    """
    Return dataframe of hedge ratios for one or more return categories
    
    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value. Will contain all of the data across all
        macroeconomic fields.
    :param <str> xcat:  extended category denoting the return series for which the
        hedge ratios are calculated.
    :param <List[str]> cids: cross sections for which hedge ratios are calculated;
        default is all available for the category.
    :param <str> hedge_return: ticker of return of the hedge asset or basket.
    :param <bool> oos: if True (deflault) hedge ratio are calculated out-of-sample,
        i.e. for the period subsequent to the estiamtion period at the given
        re-estimation frequency.
    :param <str> refreq: re-estimation frequency. Frequency at which hedge ratios are
        re-estimated. The re-estimation is conducted at the end of the period and
        fills all days of the subsequent period.
    :param <int> minobs: ... as in other functions
    :param <str> meth: method to estimate hedge ratio. At present the only method is
        OLS regression.
    
    N.B.: A hedge ratio is defined is the sensitivity of the main return in respect
    to the hedge return
    
    """


    pass