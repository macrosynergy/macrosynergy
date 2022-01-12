import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def make_relative_value(df: pd.DataFrame, xcats: List[str], cids: List[str] = None,
                        start: str = None, end: str = None, blacklist: dict = None,
                        basket: List[str] = None, rel_meth: str = 'subtract',
                        rel_xcats: List[str] = None, postfix: str = 'R'):
    """
    Returns dataframe with values relative to an average for basket of cross sections
    through subtraction or division.

    :param <pd.DataFrame> df:  standardized data frame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> xcats: all extended categories for which relative values are to
        be calculated.
    :param <List[str]> cids: cross-sections for which relative values are calculated.
        Default is every cross-section available for each respective category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the respective category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date for
        which the respective category is available is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the output.
    :param <List[str]> basket: cross-sections to be used for the relative value
        benchmark. The default is every cross-section which is available in the dataframe
        over the respective time-period. If the basket is not complete, covering all
        cross-sections, the basket is required to be a valid subset of the available
        cross-sections.
    :param <str> rel_meth: method for calculating relative value. Default is 'subtract'.
        Alternative is 'divide'.
    :param <List[str]> rel_xcats: addendum to extended category name to indicate relative
        value used.
    :param <str> postfix: acronym to be appended to 'xcat' string to give the name for
        relative value category. Only applies if rel_xcats is None. Default is 'R'

    :return <pd.Dataframe>: standardized dataframe with the relative values, featuring
        the categories: 'cid', 'xcats', 'real_date' and 'value'.

    """

    assert rel_meth in ['subtract', 'divide'], "rel_meth must be 'subtract' or 'divide'," \
                                               "and not {rel_meth}."

    assert isinstance(xcats, list) or isinstance(xcats, str), "List of categories " \
                                                              "expected, or a single" \
                                                              "category passed as a " \
                                                              "string object."
    if isinstance(xcats, str):
        xcats = [xcats]

    if cids is None:
        cids = list(df['cid'].unique())

    if basket is not None:
        miss = set(basket) - set(cids)
        assert len(miss) == 0, f" The basket elements {miss} are not in specified or " \
                               f"are not available cross-sections."
    else:
        basket = cids  # Default basket is all available cross-sections.

    col_names = ['cid', 'xcat', 'real_date', 'value']
    # Host dataframe.
    df_out = pd.DataFrame(columns=col_names)

    available_xcats = df['xcat'].unique()

    if len(cids) == len(basket) == 1:
        return df_out

    # Implicit assumption that both categories are defined over the same cross-sections.
    for i, xcat in enumerate(xcats):

        assert xcat in available_xcats, f'category {xcat} is not in dataframe'

        dfx = reduce_df(df, [xcat], cids, start, end,
                        blacklist,
                        out_all=False)[['cid', 'real_date', 'value']]

        # Reduce the dataframe to the specified basket.
        dfb = dfx[dfx['cid'].isin(basket)]

        if len(basket) > 1:
            # Mean of (available) cross sections at each point in time.
            bm = dfb.groupby(by='real_date').mean()
        else:
            # Relative value is mapped against a single cross-section.
            bm = dfb.set_index('real_date')['value']
        dfw = dfx.pivot(index='real_date', columns='cid', values='value')

        # Taking an average is only justified if the number of cross-sections, for the
        # respective date, exceeds one.
        dfw = dfw[dfw.count(axis=1) > 1]
        dfa = pd.merge(dfw, bm, how='left', left_index=True, right_index=True)

        if rel_meth == 'subtract':
            dfo = dfa[dfw.columns].sub(dfa.loc[:, 'value'], axis=0)
        else:
            dfo = dfa[dfw.columns].div(dfa.loc[:, 'value'], axis=0)

        # Re-stack.
        df_new = dfo.stack().reset_index().rename({'level_1': 'cid', 0: 'value'},
                                                  axis=1)

        if rel_xcats is None:
            df_new['xcat'] = xcat + postfix
        else:
            df_new['xcat'] = rel_xcats[i]

        df_new = df_new.sort_values(['cid', 'real_date'])[col_names]
        df_out = df_out.append(df_new)

    return df_out.reset_index(drop=True)


if __name__ == "__main__":

    # Simulate dataframe.

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])
    df_cids.loc['AUD',] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD',] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP',] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD',] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR',] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY',] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH',] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL',] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Simulate blacklist

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    # Applications

    # dfd_1 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                # blacklist=None, rel_meth='subtract', rel_xcats=None,
                                # postfix='RV')

    # dfd_2 = make_relative_value(dfd, xcats=['XR', 'GROWTH', 'INFL'], cids=None,
                                # blacklist=None,  basket=['AUD', 'CAD', 'GBP'],
                                # rel_meth='subtract', rel_xcats=['XRvB3', 'GROWTHvB3',
                                # 'INFLvB3'])

    # dfd_3 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                # blacklist=None,  basket=['AUD'],
                                # rel_meth='subtract', rel_xcats=None, postfix='RV')
    # Contrived test example.
    dfd_4 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=['AUD'],
                                blacklist=None,  basket=['AUD'],
                                rel_meth='subtract', rel_xcats=None, postfix='RV')
