import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def pan_neutral(df: pd.DataFrame, neutral: str = 'zero', sequential: bool = False):

    """
    Compute neutral values of return series based on a panel, i.e. all cross-sections in the dataFrame.

    :param <pd.Dataframe> df: "wide" dataframe with time index and cross section columns of returns.
    :param <str> neutral: method to determine neutral level. Default is 'zero'. Alternatives are 'mean' and "median".
    :param <bool> sequential: if True (default) score parameters (neutral level and standard deviations) are estimated
     sequentially with cumulative concurrently available information only. If False one neutral value will be calculated
     for the whole panel.

    :return <np.ndarray> arr_neutral: row-wise neutral statistic. A single value produced per row.
    Therefore, a one-dimensional array will be returned whose length is matching the dataframe df.
    """

    if neutral == 'mean':
        if sequential:
            ar_neutral = np.array([df.iloc[0:(i + 1), :].stack().mean() for i in range(df.shape[0])])
        else:  
            ar_neutral = np.repeat(df.stack().mean(), df.shape[0])
    elif neutral == 'median':  
        if sequential:
            ar_neutral = np.array([df.iloc[0:(i + 1), :].stack().median() for i in range(df.shape[0])])
        else:  
            ar_neutral = np.repeat(df.stack().median(), df.shape[0])
    else:
        ar_neutral = np.zeros(df.shape[0])

    return ar_neutral


def cross_neutral(df: pd.DataFrame, neutral: str = 'zero', sequential: bool = False):
    """
    Compute neutral values of return series individually for all cross sections.

    :param <pd.Dataframe> df: original DataFrame with the pivot function applied on the cross-sections. The DataFrame's columns
     will naturally consist of each cross-section's return series.
    :param <str> neutral: method to determine neutral level. Default is 'zero'. Alternatives are 'mean' and "median".
    :param <bool> sequential: if True (default) score parameters (neutral level and standard deviations) are estimated
     sequentially with cumulative concurrently available information only. If False one neutral value will be calculated
     for the whole panel.

    :return <np.ndarray> arr_neutral: column-wise neutral statistic. Same dimensions as the received DataFrame.
    """   
    cross_sections = df.columns
    no_dates = df.shape[0]
    arr_neutral = np.zeros((no_dates, len(cross_sections)))  # default is zeros only

    if neutral != 'zero':

        for i, cross in enumerate(cross_sections):
            column = df.iloc[:, i]

            if neutral == "mean":
                if sequential:
                    arr_neutral[:, i] = np.array([column[0:(j + 1)].mean() for j in range(no_dates)])
                else:
                    arr_neutral[:, i] = np.repeat(column.mean(), no_dates)
            else:  # median
                if sequential:
                    arr_neutral[:, i] = np.array([column[0:(j + 1)].median() for j in range(no_dates)])
                else:
                    arr_neutral[:, i] = np.repeat(column.median(), no_dates)
        
    return arr_neutral


def nan_insert(df: pd.DataFrame, min_obs: int = 252):

    """
    Adjust cross-sections individually for the minimum number of observations required by inserting NaN.

    :param <pd.Dataframe> df: original DataFrame with the pivot function applied on the cross-sections. The DataFrame's columns
     will naturally consist of each cross-section's return series.
    :param <int> min_obs:  the minimum number of observations required to calculate zn_scores. Default is 252.

    :return <pd.Dataframe> df: returns the same DataFrame received but with the insertion of NaN values.
    """
    
    active_dates = {}
    columns_ = df.columns
    index_dates = list(df.index)
    
    for i, col in enumerate(columns_):
        s = df.iloc[:, i]
        date = s.first_valid_index()
        active_dates[col] = index_dates.index(date)  # dictionary of indices of first non-NA value

    for k, v in active_dates.items():
        df[k][v: (v + min_obs)] = np.nan

    return df


def make_zn_scores(df: pd.DataFrame, xcat: str, cids: List[str] = None, start: str = None, end: str = None,
                   sequential: bool = False, min_obs: int = 252, neutral: str = 'zero', thresh: float = None,
                   pan_weight: float = 1, postfix: str = 'ZN'):

    """
    Computes z-scores for a panel around a neutral level ("zn scores").
    
    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcat', 'real_date' and 'value.
    :param <str> xcat:  extended category for which the zn_score is calculated.
    :param <List[str]> cids: cross sections for which zn_scores are calculated; default is all available for category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <bool> sequential: if True (default) score parameters (neutral level and standard deviations) are estimated
        sequentially with concurrently available information only.
    :param <int> min_obs: the minimum number of observations required to calculate zn_scores. Default is 252.
    :param <str> neutral: method to determine neutral level. Default is 'zero'. Alternatives are 'mean' and "median".
    :param <float> thresh: threshold value beyond which scores are winsori
    zed, i.e. contained at that threshold.
        The threshold is therefore the maximum absolute score value that the function is allowed to produce.
        The minimum threshold is 1 standard deviation
    :param <float> pan_weight: weight of panel (versus individual cross section) for calculating the z-score
        parameters, i.e. the neutral level and the standard deviation. Default is 1, i.e. panel data are the basis for
        the parameters. Lowest possible value is 0, i.e. parameters are all specific to cross section.
    :param <str> postfix: string appended to category name for output; default is "ZN".

    :return <pd.Dataframe>: standardized dataframe with the zn-scores of the chosen xcat:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    assert neutral in ['median', 'zero', 'mean']
    if thresh is not None:
        assert thresh > 1, "The 'thresh' parameter must be larger than 1"
    assert 0 <= pan_weight <= 1, "The 'pan_weight' parameter must be between 0 and 1"

    df = reduce_df(df, xcats=[xcat], cids=cids, start=start, end=end)
    dfw = df.pivot(index='real_date', columns='cid', values='value')

    no_dates = dfw.shape[0]
    cross_sections = dfw.columns

    if pan_weight > 0:

        ar_neutral = pan_neutral(dfw, neutral, sequential)

        dfx = dfw.sub(ar_neutral, axis='rows')  # df of excess values (minus neutrals)
        ar_sds = np.array([dfx.iloc[0:(i + 1), :].stack().abs().mean() for i in range(dfx.shape[0])])
        dfw_zns_pan = dfx.div(ar_sds, axis='rows')
    else:
        dfw_zns_pan = dfw * 0

    if pan_weight < 1:

        arr_neutral = cross_neutral(dfw, neutral, sequential)
        dfx = dfw.sub(arr_neutral, axis = 'rows')

        ar_sds = np.empty((no_dates, len(cross_sections)))
        for i in range(len(cross_sections)):  # produce cross-section specific deviations around neutral value
            column = dfx.iloc[:, i]
            ar_sds[:, i] = np.array([column[0:(j + 1)].abs().mean() for j in range(no_dates)])
        dfw_zns_css = dfx.div(ar_sds, axis='rows')
    else:
        dfw_zns_css = dfw * 0

    dfw_zns = (dfw_zns_pan * pan_weight) + (dfw_zns_css * (1 - pan_weight))
    dfw_zns = nan_insert(dfw_zns, min_obs)
    dfw_zns = dfw_zns.dropna(axis=0, how='all')
    
    if thresh is not None:
        dfw_zns.clip(lower=-thresh, upper=thresh, inplace=True)

    df_out = dfw_zns.unstack().reset_index().rename(mapper={0: 'value'}, axis=1)
    df_out['xcat'] = xcat + postfix

    return df_out[df.columns]


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add', 'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
    
    print("Uses Ralph's make_qdf() function.")
    dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)
    
    df_output = make_zn_scores(dfd, xcat='CRY', sequential=True, cids=cids, neutral='mean', pan_weight=0.65)
    df_pivot = df_output.pivot(index='real_date', columns='cid', values='value')
