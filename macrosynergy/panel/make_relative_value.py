
import pandas as pd
from typing import List
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def make_relative_value(df: pd.DataFrame, xcats: List[str], cids: List[str] = None,
                        start: str = None, end: str = None, blacklist: dict = None,
                        basket: List[str] = None, complete_cross: bool = False,
                        rel_meth: str = 'subtract', rel_xcats: List[str] = None,
                        postfix: str = 'R'):
    """
    Returns panel of relative values versus an average of cross-sections.

    :param <pd.DataFrame> df:  standardized JPMaQS dataframe with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> xcats: all extended categories for which relative values are to
        be calculated.
    :param <List[str]> cids: cross-sections for which relative values are calculated.
        Default is all cross-section available for the respective category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the respective category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date for
        which the respective category is available is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the output.
    :param <List[str]> basket: cross-sections to be used for the relative value
        benchmark. The default is every cross-section in the chosen list that is
        available in the dataframe over the respective time-period.
        However, the basket can be reduced to a valid subset of the available
        cross-sections.
    :param <bool> complete_cross: Boolean parameter that outlines whether each category
        is required to have the full set of cross-sections held by the basket parameter.
        Default is False. If False, the mean, for the relative value, will use the subset
        that is available for that category. For instance, if basket = ['AUD', 'CAD',
        'GBP', 'NZD'] but available cids = ['GBP', 'NZD'], the basket will be implicitly
        updated to basket = ['GBP', 'NZD'] for that respective category.
    :param <str> rel_meth: method for calculating relative value. Default is 'subtract'.
        Alternative is 'divide'.
    :param <List[str]> rel_xcats: extended category name of the relative values.
    :param <str> postfix: acronym to be appended to 'xcat' string to give the name for
        relative value category. Only applies if rel_xcats is None. Default is 'R'

    :return <pd.Dataframe>: standardized dataframe with the relative values, featuring
        the categories: 'cid', 'xcats', 'real_date' and 'value'.

    """

    assert rel_meth in ['subtract', 'divide'], "rel_meth must be 'subtract' or 'divide'"

    xcat_error = "List of categories or single single category string expected "
    assert isinstance(xcats, (list, str)), xcat_error

    if isinstance(xcats, str):
        xcats = [xcats]

    if rel_xcats is not None:
        error_rel_xcat = "List of strings or single string expected for `rel_xcats`."
        assert isinstance(rel_xcats, (list, str)), error_rel_xcat

        if isinstance(rel_xcats, str):
            rel_xcats = [rel_xcats]

        error_length = "`rel_xcats` must have the same number of elements as `xcats`."
        assert len(xcats) == len(rel_xcats), error_length

    col_names = ['cid', 'xcat', 'real_date', 'value']
    # Host dataframe.
    df_out = pd.DataFrame(columns=col_names)

    dfx = reduce_df(df, xcats, cids, start, end, blacklist,
                    out_all=False)
    if cids is None:
        cids = list(dfx['cid'].unique())

    if basket is not None:
        miss = set(basket) - set(cids)
        assert len(miss) == 0, f"The basket elements {miss} are not specified or " \
                               f"are not available."
    else:
        basket = cids  # Default basket is all available cross-sections.

    available_xcats = dfx['xcat'].unique()

    if len(cids) == len(basket) == 1:
        run_error = "Computing the relative value on a single cross-section using a " \
                    "basket consisting exclusively of the aforementioned cross-section " \
                    "is an incorrect usage of the function."
        raise RuntimeError(run_error)

    intersection_function = lambda l_1, l_2: sorted(list(set(l_1) & set(l_2)))

    storage = [df_out]
    # Implicit assumption that both categories are defined over the same cross-sections.
    # Achieved by the reduce_df() subroutine will unify the cross-sections both
    # categories are defined over.
    for i, xcat in enumerate(available_xcats):

        df_xcat = dfx[dfx['xcat'] == xcat]
        available_cids = df_xcat['cid'].unique()

        # If True, all cross-sections defined in the "basket" data structure are
        # available for the respective category.

        intersection = intersection_function(basket, available_cids)
        clause = len(intersection)
        missing_cids = list(set(basket) - set(intersection))

        if clause != len(basket) and complete_cross:
            print(f"The category, {xcat}, is missing {missing_cids} which are included "
                  f"in the basket {basket}. Therefore, the category will be excluded "
                  f"from the returned DataFrame.")
            continue

        # Must be a valid subset of the available cross-sections.
        elif clause != len(basket):
            print(f"The category, {xcat}, is missing {missing_cids}. "
                  f"The new basket will be {intersection}.")

        dfx_xcat = df_xcat[['cid', 'real_date', 'value']]

        # Reduce the dataframe to the specified basket.
        dfb = dfx_xcat[dfx_xcat['cid'].isin(basket)]

        if len(basket) > 1:
            # Mean of (available) cross sections at each point in time. If all
            # cross-sections defined in the "basket" data structure are not available for
            # a specific date, compute the mean over the available subset.
            bm = dfb.groupby(by='real_date').mean()
        else:
            # Relative value is mapped against a single cross-section.
            bm = dfb.set_index('real_date')['value']

        dfw = dfx_xcat.pivot(index='real_date', columns='cid', values='value')

        # Taking an average and computing the relative value is only justified if the
        # number of cross-sections, for the respective date, exceeds one. Therefore, if
        # any rows have only a single cross-section, remove the dates from the DataFrame.
        dfw = dfw[dfw.count(axis=1) > 1]
        # The time-index will be delimited by the respective category.
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

        storage.append(df_new.sort_values(['cid', 'real_date'])[col_names])

    return pd.concat(storage).reset_index(drop=True)


if __name__ == "__main__":

    # Simulate dataframe.

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])
    df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Simulate blacklist

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    # Applications
    dfd_1 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                blacklist=None, rel_meth='subtract', rel_xcats=None,
                                postfix='RV')
    dfd_concatenate = pd.concat([dfd, dfd_1])
    dfd_concatenate = dfd_concatenate.reset_index(drop=True)

    dfd_1_black = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                      blacklist=black, rel_meth='subtract',
                                      rel_xcats=None, postfix='RV')

    dfd_2 = make_relative_value(dfd, xcats=['XR', 'GROWTH', 'INFL'], cids=None,
                                blacklist=None,  basket=['AUD', 'CAD', 'GBP'],
                                rel_meth='subtract', rel_xcats=['XRvB3', 'GROWTHvB3',
                                'INFLvB3'])

    dfd_3 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                blacklist=None,  basket=['AUD'],
                                rel_meth='subtract', rel_xcats=None, postfix='RV')

    # Contrived test examples.
    dfd_4 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=['AUD', 'CAD'],
                                blacklist=None,  basket=['AUD'],
                                rel_meth='subtract', rel_xcats=None, postfix='RV')

    dfd_5 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=['AUD', 'CAD'],
                                blacklist=None,  basket=['AUD'],
                                rel_meth='divide', rel_xcats=['GROWTH_D', 'INFL_D'],
                                postfix=None)
    print(dfd_5)

    # Testing for complete-cross parameter.
    xcats = ['XR', 'CRY']
    start = '2000-01-01'
    end = '2020-12-31'
    dfx = reduce_df(df=dfd, xcats=xcats, cids=cids, start=start,
                    end=end, blacklist=None, out_all=False)

    # On the reduced dataframe, remove a single cross-section from one of the
    # categories.
    filt1 = ~((dfx['cid'] == 'AUD') & (dfx['xcat'] == 'XR'))
    dfdx = dfx[filt1]
    # Pass in the filtered dataframe.
    dfd_rl = make_relative_value(dfdx, xcats=xcats, cids=cids, start=start,
                                 end=end, blacklist=None, basket=None,
                                 complete_cross=True, rel_meth='subtract',
                                 rel_xcats=None, postfix='RV')