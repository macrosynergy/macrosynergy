import numpy as np
import pandas as pd
from typing import List
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df

def func_executor(df: pd.DataFrame, neutral: str, n: int):
    """
    Function used to clean up the repetitive code in the below methods. Will produce the
    evolving neutral level.

    :param <pd.DataFrame> df:
    :param <str> neutral:
    :param <int> n: number of dates the neutral level is computed over.
    """

    if neutral == "mean":
        return np.array([df.iloc[0:(i + 1), :].stack().mean() for i in range(n)])
    else:
        return np.array([df.iloc[0:(i + 1), :].stack().median() for i in range(n)])

def pan_neutral(df: pd.DataFrame, neutral: str = 'zero', sequential: bool = False,
                min_obs: int = 261, iis: bool = False):

    """
    Compute neutral values of return series based on a panel, i.e. all cross-sections in
    the DataFrame. The two approaches to calculation are either a rolling neutral value
    or an entire in-sample value.
    Additionally, if the "iis" parameter is set to True, the number of days outlined by
    "min_obs" will be calculated in sample (single neutral value for the time period)
    whilst the remaining days are calculated on a rolling basis. However, if False, and
    "sequential" equals True, the rolling neutral value will be calculated from the start
    date.
    NB: It is worth noting that the evolving neutral level, if the "sequential" parameter
        is set to True, will be computed using the available cross-sections on the
        respective date. The code will not adjust for an incomplete set of cross-sections
        on each date. For instance, if the first 100 days only have 3 cross-sections with
        realised values out of the 4 defined, the rolling mean will be calculated using
        the available subset.

    :param <pd.Dataframe> df: "wide" dataframe with time index and cross section columns.
    :param <str> neutral: method to determine neutral level. Default is 'zero'.
        Alternatives are 'mean' and "median".
    :param <bool> sequential: if True (default) score parameters (neutral level and
        standard deviations) are estimated sequentially with cumulative concurrently
        available information only. If False one neutral value will be calculated for
        the whole panel.
    :param <int> min_obs:
    :param <bool> iis:

    :return <np.ndarray> arr_neutral: row-wise neutral statistic. A single value produced
        per row. Therefore, a one-dimensional array will be returned whose length matches
        the row dimension of the dataframe df.
    """
    no_rows = df.shape[0]

    if neutral == 'mean':
        if sequential and not iis:
            ar_neutral = func_executor(df=df, neutral=neutral,
                                       n=no_rows)
            ar_neutral[:min_obs] = np.nan

        elif sequential and iis:
            iis_neutral = np.repeat(df.iloc[0:min_obs].stack().mean(),
                                    min_obs)
            os_neutral = func_executor(df=df, neutral=neutral,
                                       n=no_rows)
            os_neutral = os_neutral[min_obs:]
            ar_neutral = np.concatenate([iis_neutral, os_neutral])
        else:  
            ar_neutral = np.repeat(df.stack().mean(), no_rows)

    elif neutral == 'median':  
        if sequential and not iis:
            ar_neutral = func_executor(df=df, neutral=neutral,
                                       n=no_rows)
            ar_neutral[:min_obs] = np.nan
        elif sequential and iis:
            iis_neutral = np.repeat(df.iloc[0:min_obs].stack().median(), min_obs)
            os_neutral = func_executor(df=df, neutral=neutral,
                                       n=no_rows)
            os_neutral = os_neutral[min_obs:]
            ar_neutral = np.concatenate([iis_neutral, os_neutral])
        else:  
            ar_neutral = np.repeat(df.stack().median(), df.shape[0])
    else:
        ar_neutral = np.zeros(df.shape[0])

    return ar_neutral

def first_value(column: pd.Series):
    """
    Returns the integer index at which the series' first realised value occurs.

    :param: < pd.Series > column:

    return <int> first_index:
    """

    index = column.index
    date = column.first_valid_index()
    date_index = next(iter(np.where(index == date)[0]))

    return date_index


def index_info(df_row_no: int, column: pd.Series, min_obs: int):
    """
    Method used to determine the first date where the cross-section has a realised value.
    Will vary across the panel.

    :param <int> df_row_no: the number of rows defined in the original pivoted dataframe.
        The number of rows the dataframe is defined over corresponds to the first and
        last date across the panel. Therefore, certain cross-sections will have NaN
        values if there series do not align.
    :param: <pd.Series> column: individual cross-section's data-series.
    :param: <int> min_obs:

    :return <int>:
    """

    date_index = first_value(column)
    df_row_no -= date_index
    first_date = date_index + min_obs

    return df_row_no, first_date, date_index

def in_sampling(column: pd.Series, neutral: str, active_days: int,
                date_index: int, min_obs: int):
    """
    To prevent loss of information, calculate the first minimum number of observation
    days using an in-sampling technique (time-series data is realised, and subsequently
    not computed on a rolling basis).

    :param <pd.Series> column: individual cross-section's time-series data.
    :param <str> neutral:
    :param <int> active_days: number of active days the cross-section is defined over.
    :param <int> date_index: index of the first active trading day.
    :param <int> min_obs:

    """

    iis_period = column.iloc[:min_obs]
    if neutral == "mean":
        neutral_iis = iis_period.mean()
        os_neutral = np.array([column.iloc[0:(i + 1)].mean()
                               for i in range(active_days)])
    else:
        neutral_iis = iis_period.median()
        os_neutral = np.array([column.iloc[0:(i + 1)].median()
                               for i in range(active_days)])

    iis_neutral = np.repeat(neutral_iis, min_obs)
    os_neutral = os_neutral[min_obs:]
    prior_to_first = np.empty(date_index)
    prior_to_first[:] = np.nan
    arr = np.concatenate([prior_to_first, iis_neutral, os_neutral])

    return arr

def cross_neutral(df: pd.DataFrame, neutral: str = 'zero', sequential: bool = False,
                  min_obs: int = 261, iis: bool = False):
    """
    Compute neutral values of return series individually for all cross-sections.

    :param <pd.Dataframe> df: original DataFrame with the pivot function applied on the
        cross-sections. The DataFrame's columns will naturally consist of each
        cross-section's return series.
    :param <str> neutral: method to determine neutral level. Default is 'zero'.
        Alternatives are 'mean' and "median".
    :param <bool> sequential: if True (default) score parameters (neutral level and
        standard deviations) are estimated sequentially using the preceding dates'
        realised returns. If False one neutral value will be calculated for the
        whole panel.
    :param <int> min_obs:
    :param <bool> iis:

    :return <np.ndarray> arr_neutral: column-wise neutral statistic. Same dimensions as
        the received DataFrame.
    """   
    cross_sections = df.columns
    no_dates = df.shape[0]
    no_cids = len(cross_sections)
    arr_neutral = np.zeros((no_dates, no_cids))

    if neutral != 'zero':
        for i, cross in enumerate(cross_sections):

            column = df.iloc[:, i]
            original_index_no = no_dates
            df_row_no, first_date, date_index = index_info(original_index_no, column,
                                                           min_obs=min_obs)
            column = column[date_index:]
            if neutral == "mean":
                if sequential and not iis:
                    mean_arr = np.array([column[0:(j + 1)].mean()
                                         for j in range(df_row_no)])

                    arr_neutral[date_index:, i] = mean_arr
                    arr_neutral[:date_index, i] = np.nan
                    arr_neutral[date_index:(date_index + min_obs), i] = np.nan

                elif sequential and iis:
                    arr_neutral[:, i] = in_sampling(column, neutral, df_row_no,
                                                    date_index, min_obs)
                else:
                    arr_neutral[date_index:, i] = np.repeat(column.mean(), df_row_no)
                    arr_neutral[:date_index, i] = np.nan
            else:
                if sequential and not iis:

                    median_arr = np.array([column[0:(j + 1)].median()
                                           for j in range(df_row_no)])

                    arr_neutral[date_index:, i] = median_arr
                    arr_neutral[:date_index, i] = np.nan
                    arr_neutral[date_index:(date_index + min_obs), i] = np.nan

                elif sequential and iis:
                    arr_neutral[:, i] = in_sampling(column, neutral, df_row_no,
                                                    date_index, min_obs)
                else:
                    arr_neutral[date_index:, i] = np.repeat(column.median(), df_row_no)
                    arr_neutral[:date_index, i] = np.nan
        
    return arr_neutral

def iis_std_panel(dfx: pd.DataFrame, min_obs: int, iis: bool = True):
    """
    Function designed to compute the standard deviations but accounts for in-sampling
    period. The in-sampling standard deviation will be a fixed value.

    :param <pd.DataFrame> dfx: dataFrame recording the differences from the neutral
        level.
    :param <int> min_obs:
    :param <bool> iis:

    :return <np.ndarray> ar_sds: an array of daily standard deviations.
    """
    no_dates = dfx.shape[0]

    ar_sds = np.array([dfx.iloc[0:(i + 1), :].stack().abs().mean()
                       for i in range(no_dates)])
    if iis:
        iis_dfx = dfx.iloc[0:min_obs, :]
        iis_sds = np.array(iis_dfx.stack().abs().mean())
        ar_sds[:min_obs] = iis_sds

    return ar_sds

def iis_std_cross(column: pd.Series, min_obs: int, date_index: int = 0,
                  iis: bool = True):
    """
    Standard deviation for cross-sectional zn_scores. Will account for the in-sampling
    period.

    :param <pd.Series> column: individual cross-section's deviation from the respective
        cross-section's neutral level.
    :param <int> min_obs:
    :param <int> date_index: index of the first realised value for each series. Only
        applicable for cross-sectional zn_scores.

    :return <np.ndarray> ar_sds:
    """

    no_dates = column.size
    ar_sds = np.array([column[0:(j + 1)].abs().mean() for j in range(no_dates)])

    if iis:
        iis_end_date = date_index + min_obs
        iis_column = column[date_index:iis_end_date]
        iis_sds = np.array(iis_column.abs().mean())
        ar_sds[date_index: iis_end_date] = iis_sds

    return ar_sds


def make_zn_scores(df: pd.DataFrame, xcat: str, cids: List[str] = None,
                   start: str = None, end: str = None, blacklist: dict = None,
                   sequential: bool = True, min_obs: int = 261,  iis: bool = True,
                   neutral: str = 'zero', thresh: float = None,
                   pan_weight: float = 1, postfix: str = 'ZN'):

    """
    Computes z-scores for a panel around a neutral level ("zn scores").
    
    :param <pd.Dataframe> df: standardized data frame with following necessary columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <str> xcat:  extended category for which the zn_score is calculated.
    :param <List[str]> cids: cross sections for which zn_scores are calculated; default
        is all available for category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the calculation of zn-scores. This means it is inputs that are blacklisted.
        This means that not only are there no zn-score values calculated for these
        periods, but also that they are not used for the coring of other periods.
        N.B.: If one cross section has several blacklist periods append numbers
        to the cross-section code.
    :param <bool> sequential: if True (default) score parameters (neutral level and
        standard deviations) are estimated sequentially with concurrently available
        information only.
    :param <int> min_obs: the minimum number of observations required to calculate
        zn_scores. Default is 261.
    :param <bool> iis: if True (default) zn-scores are also calculated for the initial
        sample period defined by min-obs on an in-sample basis to avoid losing history.
    :param <str> neutral: method to determine neutral level. Default is 'zero'.
        Alternatives are 'mean' and "median".
    :param <float> thresh: threshold value beyond which scores are winsorized,
        i.e. contained at that threshold. The threshold is the maximum absolute
        score value that the function is allowed to produce. The minimum threshold is 1
        standard deviation.
    :param <float> pan_weight: weight of panel (versus individual cross section) for
        calculating the z-score parameters, i.e. the neutral level and the standard
        deviation. Default is 1, i.e. panel data are the basis for the parameters.
        Lowest possible value is 0, i.e. parameters are all specific to cross section.
    :param <str> postfix: string appended to category name for output; default is "ZN".

    :return <pd.Dataframe>: standardized dataframe with the zn-scores of the chosen xcat:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    assert neutral in ['median', 'zero', 'mean']
    if thresh is not None:
        assert thresh > 1, "The 'thresh' parameter must be larger than 1."
    assert 0 <= pan_weight <= 1, "The 'pan_weight' parameter must be between 0 and 1."
    assert isinstance(iis, bool), "Boolean Object required."
    assert isinstance(min_obs, int) and min_obs >= 0, "Minimum observations must be a " \
                                                      "non-negative Integer value."

    df = df.loc[:, ['cid', 'xcat', 'real_date', 'value']]
    df = reduce_df(df, xcats=[xcat], cids=cids, start=start, end=end,
                   blacklist=blacklist)
    dfw = df.pivot(index='real_date', columns='cid', values='value')

    no_dates = dfw.shape[0]
    cross_sections = dfw.columns

    if pan_weight > 0:

        ar_neutral = pan_neutral(dfw, neutral, sequential, min_obs, iis)
        dfx = dfw.sub(ar_neutral, axis='rows')
        ar_sds = iis_std_panel(dfx, min_obs, iis)
        dfw_zns_pan = dfx.div(ar_sds, axis='rows')
    else:
        dfw_zns_pan = dfw * 0

    if pan_weight < 1:

        arr_neutral = cross_neutral(dfw, neutral, sequential, min_obs, iis)
        dfx = dfw.sub(arr_neutral, axis='rows')

        no_cids = len(cross_sections)
        ar_sds = np.empty((no_dates, no_cids))

        for i in range(no_cids):
            column = dfx.iloc[:, i]
            date_index = first_value(column)
            ar_sds[:, i] = iis_std_cross(column, min_obs, date_index, iis)
        dfw_zns_css = dfx.div(ar_sds, axis='rows')
    else:
        dfw_zns_css = dfw * 0

    dfw_zns = (dfw_zns_pan * pan_weight) + (dfw_zns_css * (1 - pan_weight))
    dfw_zns = dfw_zns.dropna(axis=0, how='all')
    
    if thresh is not None:
        dfw_zns.clip(lower=-thresh, upper=thresh, inplace=True)

    df_out = dfw_zns.stack().to_frame("value").reset_index()
    df_out['xcat'] = xcat + postfix

    col_names = ['cid', 'xcat', 'real_date', 'value']
    df_out = df_out.sort_values(['cid', 'real_date'])[col_names]

    return df_out[df.columns]


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
    
    print("Uses Ralph's make_qdf() function.")
    dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)

    filt1 = dfd['xcat'] == 'XR'
    dfd = dfd[filt1]
    dfw = dfd.pivot(index='real_date', columns='cid', values='value')
    min_obs = 251
    ar_mean = cross_neutral(dfw, neutral='mean', sequential=True,
                            min_obs=min_obs, iis=False)

    df_output = make_zn_scores(dfd, xcat='XR', sequential=False, cids=cids, iis=True,
                               neutral='mean', pan_weight=1.0, min_obs = 261)

    df_cross = make_zn_scores(dfd, 'XR', cids, start="2010-01-04",
                              sequential=True, min_obs=0, neutral='mean',
                              iis=False, thresh=None, pan_weight=0, postfix='ZN')
