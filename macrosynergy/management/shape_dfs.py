import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import random

from macrosynergy.management.simulate_quantamental_data import make_qdf


def reduce_df(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None,
              start: str = None, end: str = None, blacklist: dict = None,
              out_all: bool = False, intersect: bool = False):
    """
    Filter dataframe by xcats and cids and notify about missing xcats and cids.

    :param <pd.Dataframe> df: standardized dataframe with the necessary columns:
        'cid', 'xcats', 'real_date'.
    :param <List[str]> xcats: extended categories to be checked on. Default is all in the
        dataframe.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the
        dataframe.
    :param <str> start: string representing earliest date. Default is None.
    :param <str> end: string representing the latest date. Default is None.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the data frame. If one cross section has several blacklist periods append numbers
        to the cross-section code.
    :param <bool> out_all: if True the function returns reduced dataframe and selected/
        available xcats and cids.
        Default is False, i.e. only the dataframe is returned
    :param <bool> intersect: if True only retains cids that are available for all xcats.
        Default is False.

    :return <pd.Dataframe>: reduced dataframe that also removes duplicates or
        (for out_all True) dataframe and available and selected xcats and cids.
    """

    dfx = df[df['real_date'] >= pd.to_datetime(start)] if start is not None else df
    dfx = dfx[dfx['real_date'] <= pd.to_datetime(end)] if end is not None else dfx

    if blacklist is not None:
        for key, value in blacklist.items():
            filt1 = dfx['cid'] == key[:3]
            filt2 = dfx['real_date'] >= pd.to_datetime(value[0])
            filt3 = dfx['real_date'] <= pd.to_datetime(value[1])
            dfx = dfx[~(filt1 & filt2 & filt3)]

    xcats_in_df = dfx['xcat'].unique()
    if xcats is None:
        xcats = sorted(xcats_in_df)
    else:
        missing = sorted(set(xcats) - set(xcats_in_df))
        if len(missing) > 0:
            print(f"Missing categories: {missing}.")
            xcats.remove(missing)

    dfx = dfx[dfx['xcat'].isin(xcats)]

    if intersect:
        df_uns = dfx.groupby('xcat')['cid'].unique()
        cids_in_df = list(df_uns[0])
        for i in range(1, len(df_uns)):
            cids_in_df = [cid for cid in df_uns[i] if cid in cids_in_df]
    else:
        cids_in_df = dfx['cid'].unique()

    if cids is None:
        cids = sorted(cids_in_df)
    else:
        if not isinstance(cids, list):
           cids = [cids]
        missing = sorted(set(cids) - set(cids_in_df))
        if len(missing) > 0:
            print(f'Missing cross sections: {missing}')
        cids = sorted(list(set(cids).intersection(set(cids_in_df))))
        dfx = dfx[dfx['cid'].isin(cids)]

    if out_all:
        return dfx.drop_duplicates(), xcats, cids
    else:
        return dfx.drop_duplicates()


def reduce_df_by_ticker(df: pd.DataFrame, ticks: List[str] = None,  start: str = None,
                        end: str = None, blacklist: dict = None):
    """
    Filter dataframe by xcats and cids and notify about missing xcats and cids

    :param <pd.Dataframe> df: standardized dataframe with the following columns:
                              'cid', 'xcats', 'real_date'.
    :param <List[str]> ticks: tickers (cross sections + base categories)
    :param <str> start: string in ISO 8601 representing earliest date. Default is None.
    :param <str> end: string ISO 8601 representing the latest date. Default is None.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
                             the dataframe. If one cross section has several blacklist
                             periods append numbers to the cross section code.

    :return <pd.Dataframe>: reduced dataframe that also removes duplicates
    """

    dfx = df[df["real_date"] >= pd.to_datetime(start)] if start is not None else df
    dfx = dfx[dfx["real_date"] <= pd.to_datetime(end)] if end is not None else dfx

    if blacklist is not None:  # blacklisting by cross-section
        for key, value in blacklist.items():
            filt1 = dfx["cid"] == key[:3]
            filt2 = dfx["real_date"] >= pd.to_datetime(value[0])
            filt3 = dfx["real_date"] <= pd.to_datetime(value[1])
            dfx = dfx[~(filt1 & filt2 & filt3)]

    dfx["ticker"] = df["cid"] + '_' + df["xcat"]
    ticks_in_df = dfx["ticker"].unique()
    if ticks is None:
        ticks = sorted(ticks_in_df)
    else:
        missing = sorted(set(ticks) - set(ticks_in_df))
        if len(missing) > 0:
            print(f'Missing tickers: {missing}')
            ticks.remove(missing)

    dfx = dfx[dfx["ticker"].isin(ticks)]

    return dfx.drop_duplicates()


def categories_df(df: pd.DataFrame, xcats: List[str], cids: List[str] = None,
                  val: str = 'value', start: str = None, end: str = None,
                  blacklist: dict = None, years: int = None, freq: str = 'M',
                  lag: int = 0, fwin: int = 1, xcat_aggs: List[str] = ('mean', 'mean')):

    """Create custom two-categories dataframe with appropriate frequency and lags
       suitable for analysis.

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: exactly two extended categories whose relationship is to be
        analyzed.
    :param <List[str]> cids: cross sections to be included. Default is all in the
        dataframe.
    :param <str> start: earliest date in ISO 8601 format. Default is None, i.e. earliest
        date in data frame is used.
    :param <str> end: latest date in ISO 8601 format. Default is None, i.e. latest date
        in data frame is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the data frame. If one cross section has several blacklist periods append numbers
        to the cross section code.
    :param <int> years: Number of years over which data are aggregated. Supersedes freq
        and does not allow lags, Default is None, i.e. no multi-year aggregation.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
    :param <int> lag: Lag (delay of arrival) of first (explanatory) category in periods
        as set by freq. Default is 0.
    :param <int> fwin: Forward moving average window of first category. Default is 1,
        i.e no average.
        Note: This parameter is used mainly for target returns as dependent variables.
    :param <List[str]> xcat_aggs: Exactly two aggregation methods. Default is 'mean' for
        both.

    :return <pd.Dataframe>: custom data frame with two category columns
    """

    assert freq in ['D', 'W', 'M', 'Q', 'A']
    assert not (years is not None) & (lag != 0), 'Lags cannot be applied to year groups.'
    if years is not None:
        assert isinstance(start, str), 'Year aggregation requires a start date.'

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, blacklist, out_all=True)

    col_names = ['cid', 'xcat', 'real_date', val]

    df_output = []
    if years is None:
        for i in range(2):
            dfw = df[df['xcat'] == xcats[i]].pivot(index='real_date', columns='cid',
                                                   values=val)
            dfw = dfw.resample(freq).agg(xcat_aggs[i])
            if (i == 0) and (lag > 0):  # first category (explanatory) is shifted forward
                dfw = dfw.shift(lag)
            if (i == 1) and (fwin > 0):
                dfw = dfw.rolling(window=fwin).mean().shift(1 - fwin)
            dfx = pd.melt(dfw.reset_index(), id_vars=['real_date'],
                          value_vars=cids, value_name=val)
            dfx['xcat'] = xcats[i]
            df_output.append(dfx[col_names])
    else:
        s_year = pd.to_datetime(start).year
        start_year = s_year
        e_year = df['real_date'].max().year + 1

        grouping = int((e_year - s_year) / years)
        remainder = (e_year - s_year) % years

        year_groups = {}
        for group in range(grouping):
            value = [i for i in range(s_year, s_year + years)]
            key = f"{s_year} - {s_year + (years - 1)}"
            year_groups[key] = value

            s_year += years

        v = [i for i in range(s_year, s_year + (remainder + 1))]
        year_groups[f"{s_year} - now"] = v
        list_y_groups = list(year_groups.keys())

        translate_ = lambda year: list_y_groups[int((year % start_year) / years)]
        df['real_date'] = pd.to_datetime(df['real_date'], errors='coerce')
        df['custom_date'] = df['real_date'].dt.year.apply(translate_)
        for i in range(2):
            dfx = df[df['xcat'] == xcats[i]]
            dfx = dfx.groupby(['xcat', 'cid',
                               'custom_date']).agg(xcat_aggs[i]).reset_index()
            dfx = dfx.rename(columns={"custom_date": "real_date"})
            df_output.append(dfx[col_names])

    dfc = pd.concat(df_output)
    dfc = dfc.pivot(index=('cid', 'real_date'), columns='xcat',
                    values=val).dropna()[xcats]

    return dfc


if __name__ == "__main__":

    cids = ['NZD', 'AUD', 'GBP', 'CAD']
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

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfd_x1 = reduce_df(dfd, xcats=xcats[:-1], cids=cids[0],
                       start='2012-01-01', end='2018-01-31')
    print(dfd_x1['xcat'].unique())

    dfd_x2 = reduce_df(dfd, xcats=xcats, cids=cids, start='2012-01-01', end='2018-01-31')
    dfd_x3 = reduce_df(dfd, xcats=xcats, cids=cids, blacklist=black)

    tickers = [cid + "_XR" for cid in cids]
    dfd_xt = reduce_df_by_ticker(dfd, ticks=tickers, blacklist=black)

    # Testing categories_df().
    dfc1 = categories_df(dfd, xcats=['GROWTH', 'CRY'], cids=cids, freq='M', lag=1,
                         xcat_aggs=['mean', 'mean'], start='2000-01-01', blacklist=black)

    dfc2 = categories_df(dfd, xcats=['GROWTH', 'CRY'], cids=cids, freq='M', lag=0,
                         fwin=3, xcat_aggs=['mean', 'mean'],
                         start='2000-01-01', blacklist=black)

    dfc3 = categories_df(dfd, xcats=['GROWTH', 'CRY'], cids=cids, freq='M', lag=0,
                         xcat_aggs=['mean', 'mean'], start='2000-01-01', blacklist=black,
                         years=3)

    # Testing reduce_df()
    filt1 = ~((dfd['cid'] == 'AUD') & (dfd['xcat'] == 'XR'))
    filt2 = ~((dfd['cid'] == 'NZD') & (dfd['xcat'] == 'INFL'))
    dfdx = dfd[filt1 & filt2]  # simulate missing cross sections
    dfd_x1, xctx, cidx = reduce_df(dfdx, xcats=['XR', 'CRY', 'INFL'], cids=cids,
                                   intersect=True, out_all=True)

    dfc = categories_df(dfd, xcats=['XR', 'CRY'], cids=['CAD'],
                        freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                        start='2000-01-01', years=10)