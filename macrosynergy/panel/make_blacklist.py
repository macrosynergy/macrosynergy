import numpy as np
import pandas as pd
from typing import List, Union, Tuple


def make_blacklist(df: pd.DataFrame, xcat: str, cids: List[str] = None,
                   start: str = None, end: str = None):

    """
    Converts binary category of standardized dataframe into a standardized dictionary
    that can serve as a blacklist for cross-sections in further analyses

    :param <pd.Dataframe> df: standardized DataFrame with following columns: 'cid',
        'xcats', 'real_date' and 'value'.
    :param <str> xcat: category with binary values, where 1 means blacklisted and 0 means
        not blacklisted.
    :param <str> cids: list of cross-sections which are considered in the formation of
        the blacklist. Per default, all available cross sections are considered.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the respective category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the respective category is available is used.

    :return <dict>: standardized dictionary with cross-sections as keys and tuples of
        start and end dates of the blacklist periods in ISO formats as values.
        If one cross section has multiple blacklist periods, numbers are added to the
        keys (i.e. TRY_1, TRY_2, etc.)
    """

    pass
