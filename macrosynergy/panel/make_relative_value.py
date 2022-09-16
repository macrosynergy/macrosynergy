
import numpy as np
import pandas as pd
from typing import List
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df

def _prepare_basket(df: pd.DataFrame, xcat: str, basket: List[str],
                    cids_avl: List[str], complete_cross: bool):
    """
    Categories can be defined over different cross-sections. Will determine the
    respective basket given the available cross-sections for the respective category.

    :param <pd.DataFrame> df: long JPMaQS DataFrame of single category.
    :param <str> xcat: respective category for the relative value calculation.
    :param <pd.DataFrame> basket: cross-sections to be used for the relative value
        benchmark if available.
    :param <List[str] cids_avl: cross-sections available for the category.
    :param <bool> complete_cross: if True, the basket is only calculated if all cross-
        sections, held in the basket, are available for that respective category.
    """

    cids_used = sorted(set(basket) & set(cids_avl))
    cids_miss = [b for b in basket if b not in cids_used]

    # Not able to be greater than because of assertion on line 126. If the basket
    # references a cross-section not defined in the DataFrame, an error will be thrown.
    condition = len(cids_used) < len(basket)
    if condition and complete_cross:
        cids_used.clear()
        print(f"The category, {xcat}, is missing {cids_miss} which are included "
              f"in the basket {basket}. Therefore, the category will be excluded "
              f"from the returned DataFrame.")

    elif condition:
        print(f"The category, {xcat}, is missing {cids_miss} from the requested "
              f"basket. The new basket will be {cids_used}.")

    # Reduce the DataFrame to the specified basket given the available cross-sections.
    dfb = df[df['cid'].isin(cids_used)]

    return dfb, cids_used

def make_relative_value(df: pd.DataFrame, xcats: List[str], cids: List[str] = None,
                        start: str = None, end: str = None, blacklist: dict = None,
                        basket: List[str] = None, complete_cross: bool = False,
                        rel_meth: str = 'subtract', rel_xcats: List[str] = None,
                        postfix: str = 'R'):
    """
    Returns panel of relative values versus an average of cross-sections.

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
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
        available in the DataFrame over the respective time-period.
        However, the basket can be reduced to a valid subset of the available
        cross-sections.
    :param <bool> complete_cross: boolean parameter that outlines whether each category
        is required to have the full set of cross-sections held by the basket parameter
        for a relative value calculation to occur. If set to True, the category will be
        excluded from the output if cross-sections are missing.
        Default is False. If False, the mean, for the relative value, will use the subset
        that is available for that category. For instance, if basket = ['AUD', 'CAD',
        'GBP', 'NZD'] but available cids = ['GBP', 'NZD'], the basket will be implicitly
        updated to basket = ['GBP', 'NZD'] for that respective category.
    :param <str> rel_meth: method for calculating relative value. Default is 'subtract'.
        Alternative is 'divide'.
    :param <List[str]> rel_xcats: extended category name of the relative values. Will
        displace the original category names: xcat + postfix. The order should reflect
        the order of the passed categories.
    :param <str> postfix: acronym to be appended to 'xcat' string to give the name for
        relative value category. Only applies if rel_xcats is None. Default is 'R'

    :return <pd.DataFrame>: standardized DataFrame with the relative values, featuring
        the categories: 'cid', 'xcat', 'real_date' and 'value'.

    """

    expected_columns = ["cid", "xcat", "real_date", "value"]
    col_error = f"The DataFrame must contain the necessary columns: {expected_columns}."
    assert set(expected_columns).issubset(set(df.columns)), col_error

    df = df.loc[:, ["cid", "xcat", "real_date", "value"]]
    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

    assert rel_meth in ["subtract", "divide"], "rel_meth must be 'subtract' or 'divide'"

    xcat_error = f"List of categories or single single category string expected. " \
                 f"Received {type(xcats)}."
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

        rel_xcats_dict = dict(zip(xcats, rel_xcats))

    col_names = ['cid', 'xcat', 'real_date', 'value']
    # Host DataFrame.
    df_out = pd.DataFrame(columns=col_names)

    # Intersect parameter set to False. Therefore, cross-sections across the categories
    # can vary.
    dfx = reduce_df(
        df, xcats, cids, start, end, blacklist, out_all=False
    )

    if cids is None:
        # All cross-sections available - union across categories.
        cids = list(dfx['cid'].unique())

    if basket is not None:
        # Basket must be a subset of the available cross-sections.
        miss = set(basket) - set(cids)
        error_basket = f"The basket elements {miss} are not specified or " \
                       f"are not available."
        assert len(miss) == 0, error_basket
    else:
        # Default basket is all available cross-sections.
        basket = cids

    available_xcats = dfx['xcat'].unique()

    if len(cids) == len(basket) == 1:
        run_error = "Computing the relative value on a single cross-section using a " \
                    "basket consisting exclusively of the aforementioned cross-section " \
                    "is an incorrect usage of the function."
        raise RuntimeError(run_error)

    storage = [df_out]
    # Categories can be defined over a different set of cross-sections.
    for i, xcat in enumerate(available_xcats):

        df_xcat = dfx[dfx['xcat'] == xcat]
        available_cids = df_xcat['cid'].unique()

        dfx_xcat = df_xcat[['cid', 'real_date', 'value']]

        dfb, basket = _prepare_basket(
            df=dfx_xcat, xcat=xcat, basket=basket, cids_avl=available_cids,
            complete_cross=complete_cross
        )

        if len(basket) > 1:
            # Mean of (available) cross-sections at each point in time. If all
            # cross-sections defined in the "basket" data structure are not available for
            # a specific date, compute the mean over the available subset.
            bm = dfb.groupby(by='real_date').mean()
        elif len(basket) == 1:
            # Relative value is mapped against a single cross-section.
            bm = dfb.set_index('real_date')['value']
        else:
            # Category is not defined over all cross-sections in the basket and
            # 'complete_cross' equals True.
            continue

        dfw = dfx_xcat.pivot(index='real_date', columns='cid', values='value')

        # Computing the relative value is only justified if the number of cross-sections,
        # for the respective date, exceeds one. Therefore, if any rows have only a single
        # cross-section, remove the dates from the DataFrame.
        dfw = dfw[dfw.count(axis=1) > 1]
        # The time-index will be delimited by the respective category.
        dfa = pd.merge(dfw, bm, how='left', left_index=True, right_index=True)

        if rel_meth == 'subtract':
            dfo = dfa[dfw.columns].sub(dfa.loc[:, 'value'], axis=0)
        else:
            dfo = dfa[dfw.columns].div(dfa.loc[:, 'value'], axis=0)

        # Re-stack.
        df_new = dfo.stack().reset_index().rename(
            {'level_1': 'cid', 0: 'value'}, axis=1
        )

        if rel_xcats is None:
            df_new['xcat'] = xcat + postfix
        else:
            df_new['xcat'] = rel_xcats_dict[xcat]

        storage.append(df_new.sort_values(['cid', 'real_date'])[col_names])

    return pd.concat(storage).reset_index(drop=True)


if __name__ == "__main__":

    # Simulate DataFrame.

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
    dfd_1 = make_relative_value(
        dfd, xcats=["GROWTH", "INFL"], cids=None, blacklist=None, rel_meth='subtract',
        rel_xcats=None, postfix='RV'
    )

    rel_xcats = ["GROWTH_sRV", "INFL_sRV"]
    dfd_1_black = make_relative_value(
        dfd, xcats=["GROWTH", "INFL"], cids=None, blacklist=black, rel_meth='subtract',
        rel_xcats=rel_xcats, postfix='RV'
    )
