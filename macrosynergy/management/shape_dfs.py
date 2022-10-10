import numpy as np
import pandas as pd
from typing import List
import random
from macrosynergy.management.simulate_quantamental_data import make_qdf
from itertools import product

def reduce_df(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None,
              start: str = None, end: str = None, blacklist: dict = None,
              out_all: bool = False, intersect: bool = False):
    """
    Filter DataFrame by xcats and cids and notify about missing xcats and cids.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> xcats: extended categories to be filtered on. Default is all in
        the DataFrame.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the
        dataframe.
    :param <str> start: string representing the earliest date. Default is None.
    :param <str> end: string representing the latest date. Default is None.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the data frame. If one cross-section has several blacklist periods append numbers
        to the cross-section code.
    :param <bool> out_all: if True the function returns reduced dataframe and selected/
        available xcats and cids.
        Default is False, i.e. only the DataFrame is returned
    :param <bool> intersect: if True only retains cids that are available for all xcats.
        Default is False.

    :return <pd.Dataframe>: reduced DataFrame that also removes duplicates or
        (for out_all True) DataFrame and available and selected xcats and cids.
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
        xcats = [xcat for xcat in xcats if xcat in xcats_in_df]

    dfx = dfx[dfx['xcat'].isin(xcats)]

    if intersect:
        df_uns = dict(dfx.groupby('xcat')['cid'].unique())
        df_uns = {k: set(v) for k, v in df_uns.items()}
        cids_in_df = list(set.intersection(*list(df_uns.values())))
    else:
        cids_in_df = dfx['cid'].unique()

    if cids is None:
        cids = sorted(cids_in_df)
    else:
        cids = [cids] if isinstance(cids, str) else cids
        cids = [cid for cid in cids if cid in cids_in_df]

        cids = set(cids).intersection(cids_in_df)
        dfx = dfx[dfx['cid'].isin(cids)]

    if out_all:
        return dfx.drop_duplicates(), xcats, sorted(list(cids))
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
        the dataframe. If one cross section has several blacklist periods append numbers
        to the cross section code.

    :return <pd.Dataframe>: reduced dataframe that also removes duplicates
    """

    dfx = df.copy()
    dfx = dfx[dfx["real_date"] >= pd.to_datetime(start)] if start is not None else dfx
    dfx = dfx[dfx["real_date"] <= pd.to_datetime(end)] if end is not None else dfx

    # Blacklisting by cross-section.
    if blacklist is not None:
        for key, value in blacklist.items():
            filt1 = dfx["cid"] == key[:3]
            filt2 = dfx["real_date"] >= pd.to_datetime(value[0])
            filt3 = dfx["real_date"] <= pd.to_datetime(value[1])
            dfx = dfx[~(filt1 & filt2 & filt3)]

    dfx["ticker"] = dfx["cid"] + '_' + dfx["xcat"]
    ticks_in_df = dfx["ticker"].unique()
    if ticks is None:
        ticks = sorted(ticks_in_df)
    else:
        ticks = [tick for tick in ticks if tick in ticks_in_df]

    dfx = dfx[dfx["ticker"].isin(ticks)]

    return dfx.drop_duplicates()

def aggregation_helper(dfx: pd.DataFrame, xcat_agg: str):
    """
    Helper method to down-sample each category in the DataFrame by aggregating over the
    intermediary dates according to a prescribed method.

    :param <List[str]> dfx: standardised DataFrame defined exclusively on a single
        category.
    :param <List[str]> xcat_agg: associated aggregation method for the respective
        category.

    """

    dfx = dfx.groupby(['xcat', 'cid', 'custom_date'])
    dfx = dfx.agg(xcat_agg).reset_index()

    if 'real_date' in dfx.columns:
        dfx = dfx.drop(['real_date'], axis=1)
    dfx = dfx.rename(columns={"custom_date": "real_date"})

    return dfx

def expln_df(df_w: pd.DataFrame, xpls: List[str], agg_meth: str, sum_condition: bool,
             lag: int):
    """
    Produces the explanatory column(s) for the custom DataFrame.

    :param <pd.DataFrame> df_w: group-by DataFrame which has been down-sampled. The
        respective aggregation method will be applied.
    :param <List[str]> xpls: list of explanatory category(s).
    :param <str> agg_meth: aggregation method used for all explanatory variables.
    :param <dict> sum_condition: required boolean to negate erroneous zeros if the
        aggregate method used, for the explanatory variable, is sum.
    :param <int> lag: lag of explanatory category(s). Applied uniformly to each
        category.
    """

    dfw_xpls = pd.DataFrame()
    for xpl in xpls:

        if not sum_condition:
            xpl_col = df_w[xpl].agg(agg_meth).astype(dtype=np.float32)
        else:
            xpl_col = df_w[xpl].sum(min_count=1)

        if lag > 0:
            xpl_col = xpl_col.groupby(level=0).shift(lag)

        dfw_xpls[xpl] = xpl_col

    return dfw_xpls

def categories_df(df: pd.DataFrame, xcats: List[str], cids: List[str] = None,
                  val: str = 'value', start: str = None, end: str = None,
                  blacklist: dict = None, years: int = None, freq: str = 'M',
                  lag: int = 0, fwin: int = 1, xcat_aggs: List[str] = ['mean', 'mean']):

    """
    In principle, create custom two-categories DataFrame with appropriate frequency and,
    if applicable, lags.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the following necessary
        columns: 'cid', 'xcats', 'real_date' and at least one column with values of
        interest.
    :param <List[str]> xcats: extended categories involved in the custom DataFrame. The
        last category in the list represents the dependent variable, and the (n - 1)
        preceding categories will be the explanatory variables(s).
    :param <List[str]> cids: cross-sections to be included. Default is all in the
        DataFrame.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> start: earliest date in ISO 8601 format. Default is None,
        i.e. earliest date in DataFrame is used.
    :param <str> end: latest date in ISO 8601 format. Default is None,
        i.e. latest date in DataFrame is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the DataFrame. If one cross section has several blacklist periods append numbers
        to the cross section code.
    :param <int> years: number of years over which data are aggregated. Supersedes the
        "freq" parameter and does not allow lags, Default is None, i.e. no multi-year
        aggregation.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'. Will always be the
        last business day of the respective frequency.
    :param <int> lag: lag (delay of arrival) of explanatory category(s) in periods
        as set by freq. Default is 0.
    :param <int> fwin: forward moving average window of first category. Default is 1,
        i.e no average.
        Note: This parameter is used mainly for target returns as dependent variable.
    :param <List[str]> xcat_aggs: exactly two aggregation methods. Default is 'mean' for
        both. The same aggregation method, the first method in the parameter, will be
        used for all explanatory variables.

    :return <pd.DataFrame>: custom DataFrame with category columns. All rows that contain
        NaNs will be excluded.

    N.B.:
    The number of explanatory categories that can be included is not restricted and will
    be appended column-wise to the returned DataFrame. The order of the DataFrame's
    columns will reflect the order of the categories list.
    """

    frq_options = ['D', 'W', 'M', 'Q', 'A']
    frq_error = f"Frequency parameter must be one of the stated options, {frq_options}."
    assert freq in frq_options, frq_error
    frq_dict = dict(zip(frq_options, ['B', 'W-Fri', 'BM', 'BQ', 'BA']))

    assert isinstance(xcats, list), f"<list> expected and not {type(xcats)}."
    assert all([isinstance(c, str) for c in xcats]), "List of categories expected."
    xcat_error = "The minimum requirement is that a single dependent and explanatory " \
                 "variable are included."
    assert len(xcats) >= 2, xcat_error

    aggs_error = "List of strings, outlining the aggregation methods, expected."
    assert isinstance(xcat_aggs, list), aggs_error
    assert all([isinstance(a, str) for a in xcat_aggs]), aggs_error
    aggs_len = "Only two aggregation methods required. The first will be used for all " \
               "explanatory category(s)."
    assert len(xcat_aggs) == 2, aggs_len

    assert not (years is not None) & (lag != 0), "Lags cannot be applied to year groups."
    if years is not None:
        assert isinstance(start, str), "Year aggregation requires a start date."

        no_xcats = "If the data is aggregated over a multi-year timeframe, only two " \
                   "categories are permitted."
        assert len(xcats) == 2, no_xcats

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, blacklist, out_all=True)

    metric = ["value", "grading", "mop_lag", "eop_lag"]
    val_error = "The column of interest must be one of the defined JPMaQS metrics, " \
                f"{metric}, but received {val}."
    assert val in metric, val_error
    avbl_cols = list(df.columns)
    assert val in avbl_cols, f"The passed column name, {val}, must be present in the " \
                             f"received DataFrame. DataFrame contains {avbl_cols}."

    # Reduce the columns in the DataFrame to the necessary columns:
    # ['cid', 'xcat', 'real_date'] + [val] (name of column that contains the
    # values of interest: "value", "grading", "mop_lag", "eop_lag").
    col_names = ['cid', 'xcat', 'real_date', val]

    df_output = []
    if years is None:

        df_w = df.pivot(index=('cid', 'real_date'), columns='xcat', values=val)

        dep = xcats[-1]
        # The possibility of multiple explanatory variables.
        xpls = xcats[:-1]

        df_w = df_w.groupby([pd.Grouper(level='cid'),
                             pd.Grouper(level='real_date', freq=frq_dict[freq])])

        dfw_xpls = expln_df(
            df_w=df_w, xpls=xpls, agg_meth=xcat_aggs[0],
            sum_condition=(xcat_aggs[0] == "sum"), lag=lag
        )

        # Handles for falsified zeros. Following the frequency conversion, if the
        # aggregation method is set to "sum", time periods that exclusively contain NaN
        # values will incorrectly be summed to the value zero which is misleading for
        # analysis.
        if not (xcat_aggs[-1] == "sum"):
            dep_col = df_w[dep].agg(xcat_aggs[1]).astype(dtype=np.float32)
        else:
            dep_col = df_w[dep].sum(min_count=1)

        if fwin > 1:
            s = 1 - fwin
            dep_col = dep_col.rolling(window=fwin).mean().shift(s)

        dfw_xpls[dep] = dep_col
        # Order such that the return category is the right-most column - will reflect the
        # order of the categories list.
        dfc = dfw_xpls[xpls + [dep]]

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

        dfx_list = [df[df['xcat'] == xcats[0]],
                    df[df['xcat'] == xcats[1]]]
        df_agg = list(map(aggregation_helper, dfx_list, xcat_aggs))
        df_output.extend([d[col_names] for d in df_agg])

        dfc = pd.concat(df_output)
        dfc = dfc.pivot(index=('cid', 'real_date'), columns='xcat',
                        values=val)

    # Adjusted to account for multiple signals requested. If the DataFrame is
    # two-dimensional, signal & a return, NaN values will be handled inside other
    # functionality, as categories_df() is simply a support function. If the parameter
    # how is set to "any", a potential unnecessary loss of data on certain categories
    # could arise.
    return dfc.dropna(axis=0, how='all')


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

    tickers = [cid + "_XR" for cid in cids]
    dfd_xt = reduce_df_by_ticker(dfd, ticks=tickers, blacklist=black)

    # Testing categories_df().
    dfc1 = categories_df(
        dfd, xcats=['GROWTH', 'CRY'], cids=cids, val="value", freq='W', lag=1,
        xcat_aggs=['mean', 'mean'], start='2000-01-01', blacklist=black
    )