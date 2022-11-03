
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def missing_in_df(df: pd.DataFrame, xcats: List[str] = None, cids: List[str] = None):
    """
    Print missing cross-sections and categories

    :param <pd.DataFrame> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcats', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on. Default is all
        in the DataFrame.
    :param <List[str]> cids: cross sections to be checked on. Default is all in
        the DataFrame.

    """
    print("Missing xcats across df: ", set(xcats) - set(df['xcat'].unique()))

    cids = df["cid"].unique() if cids is None else cids
    xcats_used = sorted(list(set(xcats).intersection(set(df["xcat"].unique()))))

    for xcat in xcats_used:
        cids_xcat = df.loc[df["xcat"] == xcat, "cid"].unique()
        print(f"Missing cids for {xcat}: ", set(cids) - set(cids_xcat))


def check_startyears(df: pd.DataFrame):
    """
    DataFrame with starting years across all extended categories and cross-sections

    :param <pd.DataFrame> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcats', 'real_date'.

    """

    df = df.dropna(how='any')
    df_starts = df[['cid', 'xcat', 'real_date']].groupby(['cid', 'xcat']).min()
    df_starts['real_date'] = pd.DatetimeIndex(df_starts.loc[:, 'real_date']).year

    return df_starts.unstack().loc[:, 'real_date'].astype(int, errors='ignore')


def check_enddates(df: pd.DataFrame):
    """
    DataFrame with end dates across all extended categories and cross sections.

    :param <pd.DataFrame> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcats', 'real_date'.
    """

    df = df.dropna(how='any')
    df_ends = df[['cid', 'xcat', 'real_date']].groupby(['cid', 'xcat']).max()
    df_ends['real_date'] = df_ends['real_date'].dt.strftime('%Y-%m-%d')

    return df_ends.unstack().loc[:, 'real_date']

def business_day_dif(df: pd.DataFrame, maxdate: pd.Timestamp):
    """
    Number of business days between two respective business dates.

    :param <pd.DataFrame> df: DataFrame cross-sections rows and category columns. Each
        cell in the DataFrame will correspond to the start date of the respective series.
    :param <pd.Timestamp> maxdate: maximum release date found in the received DataFrame.
        In principle, all series should have values up until the respective business
        date. The difference will represent possible missing values.

    :return <pd.DataFrame>: DataFrame consisting of business day differences for all
        series.

    """
    year_df = (maxdate.year - df.apply(lambda x: x.dt.year))
    year_df *= 52

    week_max = maxdate.week
    week_df = week_max - df.apply(lambda x: x.dt.isocalendar().week)

    # Account for difference over a year.
    week_df += year_df

    # Account for weekends.
    week_df *= 2

    df = (maxdate - df).apply(lambda x: x.dt.days)
    return df - week_df

def visual_paneldates(df: pd.DataFrame, size: Tuple[float] = None):
    """
    Visualize panel dates with color codes.

    :param <pd.DataFrame> df: DataFrame cross sections rows and category columns.
    :param <Tuple[float]> size: tuple of floats with width/length of displayed heatmap.

    """

    # DataFrame of official timestamps.
    if all(df.dtypes == object):

        df = df.apply(pd.to_datetime)
        # All series, in principle, should be populated to the last active release date
        # in the DataFrame.
        maxdate = df.max().max()
        df = business_day_dif(df=df, maxdate=maxdate)

        df = df.astype(float)
        # Ideally the data type should be int, but Pandas cannot represent NaN as int.
        # -- https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#support-for-integer-na

        header = f"Missing days prior to {maxdate.strftime('%Y-%m-%d')}"

    else:

        header = "Start years of quantamental indicators."

    if size is None:
        size = (max(df.shape[0] / 2, 15), max(1, df.shape[1]/ 2))

    sns.set(rc={'figure.figsize': size})
    sns.heatmap(df.T, cmap='YlOrBr', center=df.stack().mean(), annot=True, fmt='.0f',
                linewidth=1, cbar=False)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(header, fontsize=18)
    plt.show()


def check_availability(df: pd.DataFrame, xcats: List[str] = None,
                       cids: List[str] = None, start: str = None,
                       start_size: Tuple[float] = None, end_size: Tuple[float] = None):
    """
    Wrapper for visualizing start and end dates of a filtered DataFrame.

    :param <pd.DataFrame> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcats', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on.
        Default is all in the DataFrame.
    :param <List[str]> cids: cross sections to be checked on.
        Default is all in the DataFrame.
    :param <str> start: string representing earliest considered date. Default is None.
    :param <Tuple[float]> start_size: tuple of floats with width / length of
        the start years heatmap. Default is None (format adjusted to data).
    :param <Tuple[float]> end_size: tuple of floats with width/length of
        the end dates heatmap. Default is None (format adjusted to data).

    """
    dfx = reduce_df(df, xcats=xcats, cids=cids, start=start)
    dfs = check_startyears(dfx)
    visual_paneldates(dfs, size=start_size)
    dfe = check_enddates(dfx)
    plt.figure()
    visual_paneldates(dfe, size=end_size)


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP']
    xcats = ['XR', 'CRY']

    cols_1 = ['earliest', 'latest', 'mean_add', 'sd_mult']
    df_cids = pd.DataFrame(index=cids, columns=cols_1)
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD', ] = ['2010-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

    cols_2 = cols_1 + ['ar_coef', 'back_coef']
    df_xcats = pd.DataFrame(index=xcats, columns=cols_2)
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    filt_na = (dfd['cid'] == 'CAD') & (dfd['real_date'] < '2011-01-01')
    dfd.loc[filt_na, 'value'] = np.nan

    xxcats = xcats + ['TREND']
    xxcids = cids + ['USD']

    check_availability(df=dfd, xcats=xcats, cids=cids)