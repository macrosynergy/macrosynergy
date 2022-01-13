import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
import re
import random

def separation(function: str):

    clean = lambda elem: elem.strip()

    split = function.split(' = ')
    if len(split) != 2:
        assert "Expected form of formula is y = f(x)."
    else:
        key_value = tuple(map(clean, split))

    return key_value

def involved_xcats(xcats: List[str], expression: str):

    indices_dict = {}
    for category in xcats:

        pattern = re.compile(category)
        indices = pattern.finditer(expression)

        groups = []
        for index in indices:
            groups.append(index.span())
        indices_dict[category] = groups

    return indices_dict

def expression_modify(df: pd.DataFrame, indices_dict: dict, expression: str,
                      main_category: str):

    assert main_category in indices_dict.keys(), "Error in defined function."

    for category, indices in indices_dict.items():

        dfx = df[df['xcat'] == category]
        dfw = dfx.pivot(index='real_date', columns='cid', values='value')
        # Iterate through the possible indices (where the expression is mentioned) and
        # substitute the corresponding dataframe.
        for tup in indices:
            first = tup[0]
            last = tup[1]
            expression = expression[:first] + "dfw" + expression[last:]


def panel_calculator(df: pd.DataFrame, calcs: List[str] = None, cids: List[str] = None,
                     xcats: List[str] = None, start: str = None, end: str = None,
                     blacklist: dict = None) -> object:
    """
    Calculates panels based on simple operations in input panels.

    :param <pd.Dataframe> df: standardized dataframe with following necessary columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> calcs:  List containing the functions applied to each respective
        category outlined in the xcats parameter. The function will be specified in the
        form of an equation. For instance, "XR = XR + 0.5".
    :param <List[str]> cids: cross sections for which the new panels are calculated.
    :param <List[str]> xcats: the categories the panel calculator is applied to.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the dataframe. If one cross section has several blacklist periods append numbers
        to the cross-section code.

    :return <pd.Dataframe>: standardized dataframe with all new categories in standard
        format, i.e the columns 'cid', 'xcat', 'real_date' and 'value'.

    Notes:
    Panel calculation strings can use functions and unary mathematical operators
    in accordance with numpy convention to indicate the desired calculation such as:
        "NEWCAT = OLDCAT1 + 0.5 * OLDCAT2"
        "NEWCAT = np.log(OLDCAT1) - np.abs(OLDCAT2) ** 1/2"
    Panel calculation can also involve individual indicator series (to be applied
    to all series in the panel by using the @ as prefix, such as:
        "NEWCAT = OLDCAT1 - np.sqrt(@USD_OLDCAT2)"
    If more than one new category is calculated, the resulting panels can be used
    sequentially in the calculations, such as:
        ["NEWCAT1 = 1 + OLDCAT1/100", "NEWCAT2 = OLDCAT2 * NEWCAT1]

    """
    assert isinstance(xcats, list), f"List of categories expected, and not object type:" \
                                    " {type(xcats)}."
    assert isinstance(cids, list), f"Cross-sections passed must be held in a List."
    assert isinstance(calcs, list), "List of functions expected."
    assert all([isinstance(elem, str) for elem in calcs]), "Elements must be strings."

    dfx = reduce_df(df, xcats=xcats, cids=cids, start=start,
                    end=end, blacklist=blacklist)

    dict_function = {}
    for calc in calcs:
        separate = separation(calc)
        dict_function[separate[0]] = separate[1]

    output_df = []
    unique_categories = dfx['xcat'].unique()
    # The function is applied to every cross-section uniformly and every date over the
    # time-period.
    for k, v in dict_function.items():

        indices_dict = involved_xcats(xcats=unique_categories, expression=v)


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    start = '2010-01-01'
    end = '2020-12-31'

    filt1 = (dfd['xcat'] == 'XR')
    dfdx = dfd[filt1]
    df_calc = panel_calculator(df=dfdx, calcs=["XR = (XR + 0.5) / XR"], cids=cids, xcats=['XR'],
                               start=start, end=end, blacklist=black)