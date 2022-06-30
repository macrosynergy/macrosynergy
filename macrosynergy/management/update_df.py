
import pandas as pd
import numpy as np
from itertools import product
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.make_relative_value import make_relative_value

def update_df(df: pd.DataFrame, df_add: pd.DataFrame, categories: bool = False):
    """
    The purpose of the function is to combine two DataFrames: an aggregate DataFrame and
    a secondary DataFrame. If there are shared tickers across the two DataFrames, the
    tickers in the aggregate DataFrame will be replaced by the new series held in the
    secondary DataFrame. If an intersection does not exist between the two, the
    DataFrames will simply be concatenated.

    :param <pd.DataFrame> df: aggregate DataFrame used to store all tickers.
    :param <pd.DataFrame> df_add: DataFrame with the latest values. The tickers will
        either be appended to the aggregate DataFrame, or replace the previously defined
        series.
    :param <bool> categories: if the original DataFrame should be updated on the category
        level, an entire panel being changed, set the parameter to True. Default is False
        and updates will occur on the individual ticker level.

    :return <pd.DataFrame>: standardised DataFrame with the latest values of the modified
        or newly defined tickers added.
    """

    cols = ['cid', 'xcat', 'real_date', 'value']
    # Consider the other possible metrics that the DataFrame could be defined over.
    cols += ['grading', 'eop_lag', 'mop_lag']
    error_message = f"Expects a standardised DataFrame with possible columns: {cols}."

    df_cols = set(df.columns)
    df_add_cols = set(df_add.columns)

    assert df_cols.issubset(set(cols)), error_message

    additional_columns = filter(lambda c: c in df.columns, list(df_add.columns))
    df_error = f"The appended DataFrame must be defined over a subset of the columns " \
               f"in the returned DataFrame. The undefined column(s): " \
               f"{additional_columns}."
    assert df_add_cols.issubset(df_cols), df_error

    df_add = column_alignment(df_add, df_cols, df_add_cols)

    if categories:
        df = update_tickers(df, df_add)

    else:
        df = update_categories(df, df_add)

    return df.reset_index(drop=True)

def column_alignment(df_add: pd.DataFrame, df_cols: set, df_add_cols: set):
    """
    If the aggregate DataFrame is defined over additional metrics that are not present
    in the added DataFrame, add the missing columns but populate all dates with NaN
    values.

    :param <pd.DataFrame> df_add: new, added DataFrame.
    :param <set> df_cols: the columns of the aggregate DataFrame.
    :param <set> df_add_cols: the columns of the appended DataFrame.

    :return <pd.DataFrame> df_add: appended DataFrame but with additional metrics if
        required.
    """

    difference = list(df_cols.difference(df_add_cols))
    if difference:
        for c in difference:
            df_add[c] = np.empty(df_add.shape[0])

    return df_add

def df_tickers(df: pd.DataFrame):
    """
    Helper function used to delimit the tickers defined in a received DataFrame.

    :param <pd.DataFrame> df: standardised DataFrame.
    """
    cids_append = list(map(lambda c: c + '_', set(df['cid'])))
    tickers = list(product(cids_append, set(df['xcat'])))
    tickers = [c[0] + c[1] for c in tickers]

    return tickers

def update_tickers(df: pd.DataFrame, df_add: pd.DataFrame):
    """
    Method used to update aggregate DataFrame on a ticker level. The tickers in the
    secondary DataFrame will either replace an existing ticker or be appended to the
    returned DataFrame.

    :param <pd.DataFrame> df: aggregate dataframe used to store all tickers.
    :param <pd.DataFrame> df_add: dataframe with the latest values.

    """
    agg_df_tick = df_tickers(df)
    add_df_tick = df_tickers(df_add)

    # If the ticker is already defined in the DataFrame, replace with the new series
    # otherwise append the series to the aggregate DataFrame.
    for t in add_df_tick:

        if t in agg_df_tick:
            split = t.split('_')
            xcat = '_'.join(split[1:])
            filter_1 = ~((df['cid'] == split[0]) & (df['xcat'] == xcat))
            df = df[filter_1]
        else:
            continue

    df = pd.concat([df, df_add])
    return df.sort_values(['xcat', 'cid', 'real_date'])

def update_categories(df: pd.DataFrame, df_add):
    """
    Method used to update the DataFrame on the category level. The method aims to cover
    the most likely user cases from the sample space with computationally fast
    algorithms. Any residual instances will be covered by a more methodical helper
    function.

    :param <pd.DataFrame> df: aggregate DataFrame used to store all categories.
    :param <pd.DataFrame> df_add: DataFrame with the latest values. The categories will
        either be appended to the aggregate DataFrame, or replace the previously defined
        category(s).

    """

    incumbent_categories = list(df['xcat'].unique())
    new_categories = list(df_add['xcat'].unique())

    # Union of both category columns from the two DataFrames.
    append_condition = set(incumbent_categories) | set(new_categories)
    intersect = list(set(incumbent_categories).intersection(set(new_categories)))

    additional_category = list(set(new_categories).difference(set(intersect)))

    if len(append_condition) == len(incumbent_categories + new_categories):
        df = pd.concat([df, df_add])

    elif sorted(list(intersect)) == sorted(new_categories):
        retain = [c for c in incumbent_categories if c not in intersect]
        df = df[df['xcat'].isin(retain)]
        df = pd.concat([df, df_add])

    elif sorted(intersect + additional_category) == sorted(new_categories):
        temp_df = df_add[df_add['xcat'].isin(intersect)]
        new_df = df_add[df_add['xcat'].isin(additional_category)]
        df = pd.concat([update_categories(df, temp_df), new_df])

    else:
        df = update_categories_residual(df=df, df_add=df_add)

    return df

def update_categories_residual(df: pd.DataFrame, df_add):
    """
    Helper function.

    :param <pd.DataFrame> df: aggregate DataFrame used to store all categories.
    :param <pd.DataFrame> df_add: DataFrame with the latest values.

    :return <pd.DataFrame>: standardised DataFrame with the latest values of the modified
        or newly defined category.
    """

    incumbent_categories = list(df['xcat'].unique())
    new_categories = list(df_add['xcat'].unique())

    new_cats_copy = new_categories.copy()

    add = []
    for new_cat in new_categories:
        if new_cat not in incumbent_categories:
            temp_df = df_add[df_add['xcat'] == new_cat]
            add.append(temp_df)
            incumbent_categories.append(new_cat)
            new_cats_copy.remove(new_cat)
        else:
            continue

    df = pd.concat([df] + add)
    updated_categories = new_cats_copy
    if updated_categories:

        aggregate = []
        for xc in incumbent_categories:
            if xc not in updated_categories:
                temp_df = df[df['xcat'] == xc]
                aggregate.append(temp_df)

            else:
                temp_df = df_add[df_add['xcat'] == xc]
                aggregate.append(temp_df)

        df = pd.concat(aggregate)

    return df.reset_index(drop=True)


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
    tickers = df_tickers(dfd)

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    # Test the above method by using the in-built make_relative_value() method.
    dfd_1_rv = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                   blacklist=None, rel_meth='subtract', rel_xcats=None,
                                   postfix='RV')
    # First test will simply append the two DataFrames. The categories defined in the
    # secondary DataFrame will not be present in the original DataFrame. Therefore, the
    # append mechanism is sufficient.
    dfd_add = update_categories_residual(df=dfd, df_add=dfd_1_rv)

    dfd_1_rv_blacklist = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                             blacklist=None, rel_meth='divide',
                                             rel_xcats=None,
                                             postfix='RV')

    # Second test will be to replace updated categories. The second DataFrame's
    # categories will be a direct subset of the first. Therefore, replace the incumbent
    # categories with the latest values.
    dfd_add_2 = update_df(df=dfd_add, df_add=dfd_1_rv_blacklist, categories=True)
    print(dfd_add_2)