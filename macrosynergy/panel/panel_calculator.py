import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def panel_calculator(df: pd.DataFrame, calcs: List[str], cids: List[str] = None,
                     start: str = None, end: str = None, blacklist: dict = None):
    """
    Calculates panels based on simple operations in input panels.

    :param <pd.Dataframe> df: standardized data frame with following necessary columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> calcs:  set of calculations in string format with category names
        as arguments.
    :param <List[str]> cids: cross sections for which the new panels are calculated.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the data frame. If one cross section has several blacklist periods append numbers
        to the cross-section code.

    :return <pd.Dataframe>: standardized dataframe with all new categories in standard
        format, i.e the columns 'cid', 'xcat', 'real_date' and 'value'.

    Notes:
    Panel calculation strings can use functions and unary mathematical operators
    in accordance with numpy convention to indicate the desired calculation, such as
    "NEWCAT = OLDCAT1 + 0.5 * OLDCAT2"
    "NEWCAT = np.log(OLDCAT1) - np.abs(OLDCAT2) ** 1/2"
    Panel calculation can also involve individual indicator series (to be applied
    to all series in the panel by using the @ as prefix, such as:
    "NEWCAT = OLDCAT1 - np.sqrt(@USD_OLDCAT2)"
    If more than one new category is calculated, the resulting panels can be used
    sequentially in the calculations, such as:
    ["NEWCAT1 = 1 + OLDCAT1/100", "NEWCAT2 = OLDCAT2 * NEWCAT1]

    """

    pass