import numpy as np
import pandas as pd
from typing import List
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
import re
import random


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
    to all series in the panel by using th 'i' as prefix), such as:
    "NEWCAT = OLDCAT1 - np.sqrt(iUSD_OLDCAT2)"
    If more than one new category is calculated, the resulting panels can be used
    sequentially in the calculations, such as:
    ["NEWCAT1 = 1 + OLDCAT1/100", "NEWCAT2 = OLDCAT2 * NEWCAT1"]
    """

    # A. Asserts

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert set(cols).issubset(set(df.columns))
    assert isinstance(calcs, list), "List of functions expected."
    assert all([isinstance(elem, str) for elem in calcs]),\
        "Each formula in the panel calculation list must be a string."
    assert isinstance(cids, list), "List of cross-sections expected."

    # B. Collect new category names and their formulas

    ops = {}
    for calc in calcs:
        calc_parts = calc.split('=', maxsplit=1)
        ops[calc_parts[0].strip()] = calc_parts[1].strip()

    # C. Check if all required categories are in the dataframe

    xcats_used = []
    singles_used = []
    for op in ops.values():
        op_list = op.split(' ')
        xcats_used += [x for x in op_list if re.match('^[A-Z]', x)]
        singles_used += [s for s in op_list if re.match('^i', s)]

    single_xcats = [x[5:] for x in singles_used]
    all_xcats_used = xcats_used + single_xcats
    old_xcats_used = list(set(all_xcats_used) - set([x for x in ops.keys()]))
    missing = sorted(set(old_xcats_used) - set(df['xcat'].unique()))
    assert len(missing) == 0, f"Missing categories: {missing}."

    # D. Reduce dataframe with intersection requirement

    dfx = reduce_df(df, xcats=old_xcats_used, cids=cids,
                    start=start, end=end, blacklist=blacklist,
                    intersect=True)
    cidx = np.sort(dfx['cid'].unique())

    # E. Create all required wide dataframes with category names

    for xcat in old_xcats_used:
        dfxx = dfx[dfx['xcat'] == xcat]
        dfw = dfxx.pivot(index='real_date', columns='cid', values='value')
        exec(f'{xcat} = dfw')

    for single in singles_used:
        dfxx = dfx[(dfx['cid'] + '_' + dfx['xcat']) == single[1:]]
        dfx1 = dfxx.set_index('real_date')['value'].to_frame()
        dfw = pd.concat([dfx1] * len(cidx), axis=1, ignore_index=True)
        dfw.columns = cidx
        exec(f'{single} = dfw')

    # F. Calculate the panels and collect

    for new_xcat, formula in ops.items():
        dfw_add = eval(formula)
        # Todo: check if this works for all operations and aligns indexes reliably
        # Todo: does it work with time series methods (.diff(), .lag() and so forth?
        df_add = pd.melt(dfw_add.reset_index(), id_vars=['real_date'])
        df_add['xcat'] = new_xcat
        if new_xcat == list(ops.keys())[0]:
            df_out = df_add[cols]
        else:
            df_out = df_out.append(df_add[cols])
        exec(f'{new_xcat} = dfw_add')  # we main need a df for subsequent calculations

    return df_out


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
    df_xcats.loc['INFL'] = ['2010-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    black = {'AUD': ['2000-01-01', '2003-12-31']}

    start = '2010-01-01'
    end = '2020-12-31'

    filt1 = (dfd['xcat'] == 'XR') | (dfd['xcat'] == 'CRY')
    dfdx = dfd[filt1]

    # Start of the testing. Various testcases included to understand the capabilities of
    # the designed function.
    df_calc = panel_calculator(df=dfd,
                               calcs=["NEW1 = np.abs( XR ) + 0.52 + 2 * CRY + iGBP_INFL",
                                      "NEW2 = NEW1 / XR"],
                               cids=cids, start=start, end=end)

    df_calc = panel_calculator(df=dfdx,
                               calcs=["NEW1 = np.abs( XR ) + 0.552 + 2 * CRY",
                                      "NEW2 = NEW1 / XR"],
                               cids=cids, start=start, end=end)

    df_calc.head()