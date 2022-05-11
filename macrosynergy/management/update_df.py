
import pandas as pd
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.make_relative_value import make_relative_value

def update_df(df: pd.DataFrame, df_add):
    """
    Efficient version. Aims to cover the most likely user cases from the sample space
    with computationally fast algorithms. Any residual instances will be covered by a
    more methodical helper function.

    :param <pd.DataFrame> df: aggregate dataframe used to store all categories.
    :param <pd.DataFrame> df_add: dataframe with the latest values. The categories will
        either be appended to the aggregate dataframe, or replace the previously defined
        category.

    :return <pd.DataFrame>: standardised dataframe with the latest values of the modified
        or newly defined category.
    """

    cols = ['cid', 'xcat', 'real_date', 'value']
    error_message = f"Expects a standardised dataframe with columns: {cols}"
    assert list(df.columns) == cols, error_message
    assert list(df_add.columns) == cols, error_message

    incumbent_categories = list(df['xcat'].unique())
    new_categories = list(df_add['xcat'].unique())

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
        df = pd.concat([update_df(df, temp_df), new_df])

    else:
        df = update_df_residual(df=df, df_add=df_add)

    return df.reset_index(drop=True)

def update_df_residual(df: pd.DataFrame, df_add):
    """
    The method has two main purposes. Firstly, if the categories, in the second
    dataframe, are not present in the aggregated dataframe, append the new categories and
    return the combined version. Secondly, if the category is already present, the
    method will be used to replace the aforementioned category in the standardised
    dataframe with the new set of values. For instance, parameter values have been
    changed leading to a renewal of values on a specific category, and subsequently aim
    to update the aggregated dataframe with the latest values.

    :param <pd.DataFrame> df: aggregate dataframe used to store all categories.
    :param <pd.DataFrame> df_add: dataframe with the latest values. The categories will
        either be appended to the aggregate dataframe, or replace the previously defined
        category.

    :return <pd.DataFrame>: standardised dataframe with the latest values of the modified
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

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    # Test the above method by using the in-built make_relative_value() method.
    dfd_1_rv = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                   blacklist=None, rel_meth='subtract', rel_xcats=None,
                                   postfix='RV')
    # First test will simply append the two DataFrames. The categories defined in the
    # secondary DataFrame will not be present in the original DataFrame. Therefore, the
    # append mechanism is sufficient.
    dfd_add = update_df_residual(df=dfd, df_add=dfd_1_rv)

    dfd_1_rv_blacklist = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                             blacklist=None, rel_meth='divide',
                                             rel_xcats=None,
                                             postfix='RV')

    # Second test will be to replace updated categories. The second DataFrame's
    # categories will be a direct subset of the first. Therefore, replace the incumbent
    # categories with the latest values.
    dfd_add_2 = update_df(df=dfd_add, df_add=dfd_1_rv_blacklist)
    print(dfd_add_2)

    # Third example requires replacing existing categories but also appending a new
    # category to the aggregated, returned DataFrame.
    dfd_add_copy = dfd_add.copy()
    dfd_2_rv = make_relative_value(dfd, xcats=['CRY', 'GROWTH', 'INFL'], cids=None,
                                   blacklist=None, rel_meth='subtract', rel_xcats=None,
                                   postfix='RV')
    dfd_add_3 = update_df(df=dfd_add_copy, df_add=dfd_2_rv)