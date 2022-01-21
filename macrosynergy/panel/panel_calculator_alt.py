import numpy as np
import pandas as pd
from typing import List
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
import re
import random

def involved_xcats(ops: dict):
    """
    Function used to understand the original categories involved in the panel
    calculations. To isolate the involved categories in the expression, spaces are
    required either side of the category. For instance, NEWCAT1 = np.abs( XR ).
    Further, each category will exclusively involve capital letters which aids
    determining the categories in the string expression.

    :param <dict> ops: dictionary containing the panel calculation where the key is the
        newly formed category and the value the calculation.
    """

    xcats_used = []

    new_xcats = list(ops.keys())
    for op in ops.values():
        op_list = op.split(' ')
        xcats_used += [x for x in op_list if re.match('^[A-Z]', x)
                       and x not in new_xcats]

    return set(xcats_used)

def panel_calculator(df: pd.DataFrame, calcs: List[str] = None, cids: List[str] = None,
                     start: str = None, end: str = None,
                     blacklist: dict = None):
    """
    Calculates new data panels through operations on existing panels.

    :param <pd.Dataframe> df: standardized dataframe with following necessary columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> calcs:  list of formulas denoting operations on panels of
        categories. Words in capital letters denote category panels.
        Otherwise the formulas can include numpy functions and standard binary operators.
        See notes below.
    :param <List[str]> cids: cross sections over which the panels are defined.
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
    Panel calculation strings can use numpy functions and unary/binary operators go
    category panels, whereby the category is indicated by capital letters, underscores
    and numbers.
    Calculated category and panel operations must be separated by '='.
        "NEWCAT = (OLDCAT1 + 0.5) * OLDCAT2"
        "NEWCAT = np.log(OLDCAT1) - np.abs(OLDCAT2) ** 1/2"
    Panel calculation can also involve individual indicator series (to be applied
    to all series in the panel by using the @ as prefix), such as:
        "NEWCAT = OLDCAT1 - np.sqrt(@USD_OLDCAT2)"
    If more than one new category is calculated, the resulting panels can be used
    sequentially in the calculations, such as:
    ["NEWCAT1 = 1 + OLDCAT1/100", "NEWCAT2 = OLDCAT2 * NEWCAT1"]

    """

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert set(cols).issubset(set(df.columns))
    assert isinstance(calcs, list), "List of functions expected."
    assert all([isinstance(elem, str) for elem in calcs]), \
        "Each formula in the panel calculation list must be a string."
    assert isinstance(cids, list), "List of cross-sections expected."

    ops = {}
    for calc in calcs:
        calc_parts = calc.split('=', maxsplit=1)
        ops[calc_parts[0].strip()] = calc_parts[1].strip()

    old_xcats_used = involved_xcats(ops=ops)

    available_xcats = set(df['xcat'].unique())
    missing = sorted(old_xcats_used - available_xcats)
    assert len(missing) == 0, f"Missing categories: {missing}"

    # Reduce the dataframe to the cross-sections available in all categories.
    dfx = reduce_df(df, xcats=list(old_xcats_used), cids=cids, start=start,
                    end=end, blacklist=blacklist, intersect=True)

    for xcat in old_xcats_used:
        dfxx = dfx[dfx['xcat'] == xcat]
        dfw = dfxx.pivot(index='real_date', columns='cid', values='value')
        exec(f'{xcat} = dfw')

    output_df = []
    for new_xcat, formula in ops.items():

        dfw_add = eval(formula)
        df_add = pd.melt(dfw_add.reset_index(), id_vars=['real_date'])
        df_add['xcat'] = new_xcat
        output_df.append(df_add)
        exec(f'{new_xcat} = dfw_add')  # we main need a df for subsequent calculations

    df_calc = pd.concat(output_df)[cols]
    df_calc.reset_index(drop=True)

    return df_calc


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2010-01-01', '2020-12-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2010-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    start = '2010-01-01'
    end = '2020-12-31'

    # Start of the testing. Various testcases included to understand the capabilities of
    # the designed function.

    formula_3 = "(GROWTH - np.sqrt( @USD_INFL ))"
    df_calc = panel_calculator(df=dfd, calcs=["NEW1 = np.abs( XR ) + 0.552 + 2 * CRY",
                                              "NEW2 = NEW1 * 2"],
                               cids=cids, start=start, end=end)
