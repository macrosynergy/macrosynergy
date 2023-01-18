import numpy as np
import pandas as pd
from typing import List


def linear_composite(df: pd.DataFrame, xcats: List[str], weights=None, signs=None,
                     cids: List[str] = None, start: str = None, end: str = None,
                     complete_xcats: bool = True, new_xcat="NEW"):
    """
    Returns new category panel as linear combination of others as standard dataframe

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> xcats: all extended categories used for the linear combination.
    :param <List[float]> weights: weights of all categories in the linear combination.
        These must correspond to the order of xcats and the sum will be coerced to unity.
        Default is equal weights.
    :param <List[float]> signs: signs with which the categories are combined.
        These must be 1 or -1 for positive and negative and correspond to the order of
        xcats. Default is all positive.
    :param <List[str]> cids: cross-sections for which the linear combination is ti be
        calculated. Default is all cross-section available for the respective category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the respective category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date for
        which the respective category is available is used.
    :param <bool> complete_xcats: If True combinations are only calculated for
        observation dates on which all xcats are available. If False a combination of the
        available categories is used.
    :param <str> new_xcat: name of new composite xcat. Default is "NEW".

    """