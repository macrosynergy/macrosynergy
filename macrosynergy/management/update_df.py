
import pandas as pd
from itertools import product
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.make_relative_value import make_relative_value

def update_df(df: pd.DataFrame, df_add: pd.DataFrame, xcat_replace: bool = False):
    """
    Append a standard DataFrame to a standard base DataFrame with ticker replacement on
    the intersection.

    :param <pd.DataFrame> df: standardised base JPMaQS DataFrame with the following
        necessary columns: 'cid', 'xcats', 'real_date' and 'value'.
    :param <pd.DataFrame> df_add: another standardised JPMaQS DataFrame, with the latest
        values, to be added with the necessary columns: 'cid', 'xcats', 'real_date', and
        'value'. Columns that are present in the base DataFrame but not in the appended
        DataFrame will be populated with NaN values.
    :param <bool> xcat_replace: all series belonging to the categories in the added
        DataFrame will be replaced, rather than just the added tickers.
        N.B.: tickers are combinations of cross-sections and categories.

    :return <pd.DataFrame>: standardised DataFrame with the latest values of the modified
        or newly defined tickers added.
    """

    cols = ['cid', 'xcat', 'real_date', 'value']
    # Consider the other possible metrics that the DataFrame could be defined over

    df_cols = set(df.columns)
    df_add_cols = set(df_add.columns)

    error_message = f"The base DataFrame must include the necessary columns: {cols}."
    assert set(cols).issubset(df_cols), error_message

    error_message = f"The added DataFrame must include the necessary columns: {cols}."
    assert set(cols).issubset(df_add_cols), error_message

    additional_columns = filter(lambda c: c in df.columns, list(df_add.columns))
    df_error = f"The appended DataFrame must be defined over a subset of the columns " \
               f"in the returned DataFrame. The undefined column(s): " \
               f"{additional_columns}."
    assert df_add_cols.issubset(df_cols), df_error

    if not xcat_replace:
        df = update_tickers(df, df_add)

    else:
        df = update_categories(df, df_add)

    return df.reset_index(drop=True)

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
    Method used to update aggregate DataFrame on a ticker level.

    :param <pd.DataFrame> df: aggregate DataFrame used to store all tickers.
    :param <pd.DataFrame> df_add: DataFrame with the latest values.

    """
    agg_df_tick = set(df_tickers(df))
    add_df_tick = set(df_tickers(df_add))

    df['ticker'] = df['cid'] + '_' + df['xcat']

    # If the ticker is already defined in the DataFrame, replace with the new series
    # otherwise append the series to the aggregate DataFrame.
    df = df[~df['ticker'].isin(list(agg_df_tick.intersection(add_df_tick)))]

    df = pd.concat([df, df_add], axis=0,
                   ignore_index=True)

    df = df.drop(['ticker'], axis=1)

    return df.sort_values(['xcat', 'cid', 'real_date'])

def update_categories(df: pd.DataFrame, df_add):
    """
    Method used to update the DataFrame on the category level.

    :param <pd.DataFrame> df: base DataFrame.
    :param <pd.DataFrame> df_add: appended DataFrame.

    """

    incumbent_categories = list(df['xcat'].unique())
    new_categories = list(df_add['xcat'].unique())

    # Union of both category columns from the two DataFrames.
    append_condition = set(incumbent_categories) | set(new_categories)
    intersect = list(set(incumbent_categories).intersection(set(new_categories)))

    if len(append_condition) == len(incumbent_categories + new_categories):
        df = pd.concat([df, df_add], axis=0,
                       ignore_index=True)

    # Shared categories plus any additional categories previously not defined in the base
    # DataFrame.
    else:
        df = df[~df['xcat'].isin(intersect)]
        df = pd.concat([df, df_add], axis=0,
                       ignore_index=True)

    return df


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

    dfd_add = update_categories(df=dfd, df_add=dfd_1_rv)

    dfd_1_rv_blacklist = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                             blacklist=None, rel_meth='divide',
                                             rel_xcats=None,
                                             postfix='RV')

    dfd_add_2 = update_df(
        df=dfd_add, df_add=dfd_1_rv_blacklist, xcat_replace=True
    )
