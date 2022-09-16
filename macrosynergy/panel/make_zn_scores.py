
import numpy as np
import pandas as pd
from typing import List
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_qdf

def expanding_stat(df: pd.DataFrame, dates_iter: pd.DatetimeIndex,
                   stat: str = 'mean', sequential: bool = True,
                   min_obs: int = 261, iis: bool = True):

    """
    Compute statistic based on an expanding sample.

    :param <pd.Dataframe> df: Daily-frequency time series DataFrame.
    :param <pd.DatetimeIndex> dates_iter: controls the frequency of the neutral &
        standard deviation calculations.
    :param <str> stat: statistical method to be applied. This is typically 'mean',
        or 'median'.
    :param <bool> sequential: if True (default) the statistic is estimated sequentially.
        If this set to false a single value is calculated per time series, based on
        the full sample.
    :param <int> min_obs: minimum required observations for calculation of the
        statistic in days.
    :param <bool> iis: if set to True, the values of the initial interval determined
        by min_obs will be estimated in-sample, based on the full initial sample.

    :return: Time series dataframe of the chosen statistic across all columns
    """

    df_out = pd.DataFrame(np.nan, index=df.index, columns=['value'])
    # An adjustment for individual series' first realised value is not required given the
    # returned DataFrame will be subtracted from the original DataFrame. The original
    # DataFrame will implicitly host this information through NaN values such that when
    # the arithmetic operation is made, any falsified values will be displaced by NaN
    # values.

    first_observation = df.dropna(axis=0, how='all').index[0]
    # Adjust for individual cross-sections' series commencing at different dates.
    first_estimation = df.dropna(axis=0, how='all').index[min_obs]

    obs_index = next(iter(np.where(df.index == first_observation)))[0]
    est_index = next(iter(np.where(df.index == first_estimation)))[0]

    if stat == "zero":

        df_out["value"] = 0

    elif not sequential:
        # The entire series is treated as in-sample. Will automatically handle NaN
        # values.
        statval = df.stack().apply(stat)
        df_out["value"] = statval

    else:

        dates = dates_iter[dates_iter >= first_estimation]
        for date in dates:

            df_out.loc[date, "value"] = df.loc[first_observation:date].stack().apply(stat)

        df_out = df_out.fillna(method='ffill')

        if iis:
            df_out = df_out.fillna(method="bfill", limit=(est_index - obs_index))

    df_out.columns.name = 'cid'
    return df_out

def make_zn_scores(df: pd.DataFrame, xcat: str, cids: List[str] = None,
                   start: str = None, end: str = None, blacklist: dict = None,
                   sequential: bool = True, min_obs: int = 261, iis: bool = True,
                   neutral: str = 'zero', est_freq: str = 'd', thresh: float = None,
                   pan_weight: float = 1, postfix: str = 'ZN'):

    """
    Computes z-scores for a panel around a neutral level ("zn scores").
    
    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <str> xcat:  extended category for which the zn_score is calculated.
    :param <List[str]> cids: cross sections for which zn_scores are calculated; default
        is all available for category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the calculation of zn-scores.
        This means that not only are there no zn-score values calculated for these
        periods, but also that they are not used for the scoring of other periods.
        N.B.: The argument is a dictionary with cross-sections as keys and tuples of
        start and end dates of the blacklist periods in ISO formats as values.
        If one cross section has multiple blacklist periods, numbers are added to the
        keys (i.e. TRY_1, TRY_2, etc.)
    :param <bool> sequential: if True (default) score parameters (neutral level and
        standard deviations) are estimated sequentially with concurrently available
        information only.
    :param <int> min_obs: the minimum number of observations required to calculate
        zn_scores. Default is 261. The parameter is only applicable if the "sequential"
        parameter is set to True. Otherwise the neutral level and the standard deviation
        are both computed in-sample and will use the full sample.
    :param <bool> iis: if True (default) zn-scores are also calculated for the initial
        sample period defined by min-obs on an in-sample basis to avoid losing history.
        This is irrelevant if sequential is set to False.
    :param <str> neutral: method to determine neutral level. Default is 'zero'.
        Alternatives are 'mean' and "median".
    :param <str> est_freq: the frequency at which standard deviations or means are
        are re-estimated. The options are weekly, monthly & quarterly "w", "m", "q".
        Default is daily, "d". Re-estimation is performed at period end.
    :param <float> thresh: threshold value beyond which scores are winsorized,
        i.e. contained at that threshold. The threshold is the maximum absolute
        score value that the function is allowed to produce. The minimum threshold is 1
        standard deviation.
    :param <float> pan_weight: weight of panel (versus individual cross section) for
        calculating the z-score parameters, i.e. the neutral level and the standard
        deviation. Default is 1, i.e. panel data are the basis for the parameters.
        Lowest possible value is 0, i.e. parameters are all specific to cross section.
    :param <str> postfix: string appended to category name for output; default is "ZN".

    :return <pd.Dataframe>: standardized DataFrame with the zn-scores of the chosen xcat:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

    expected_columns = ["cid", "xcat", "real_date", "value"]
    col_error = f"The DataFrame must contain the necessary columns: {expected_columns}."
    assert set(expected_columns).issubset(set(df.columns)), col_error

    # --- Assertions

    assert neutral in ["mean", "median", "zero"]
    if thresh is not None:
        assert thresh > 1, "The 'thresh' parameter must be larger than 1."
    assert 0 <= pan_weight <= 1, "The 'pan_weight' parameter must be between 0 and 1."
    assert isinstance(iis, bool), "Boolean Object required."

    error_min = "Minimum observations must be a non-negative Integer value."
    assert isinstance(min_obs, int) and min_obs >= 0, error_min

    frequencies = ["d", "w", "m", "q"]
    error_freq = f"String Object required and must be one of the available frequencies: " \
                 f"{frequencies}."
    assert isinstance(est_freq, str) and est_freq in frequencies, error_freq
    pd_freq = dict(zip(frequencies, ['B', 'W-Fri', 'BM', 'BQ']))

    # --- Prepare re-estimation dates and time-series DataFrame.

    # Remove any additional metrics defined in the DataFrame.
    df = df.loc[:, expected_columns]
    df = reduce_df(
        df, xcats=[xcat], cids=cids, start=start, end=end, blacklist=blacklist
    )

    s_date = min(df['real_date'])
    e_date = max(df['real_date'])
    dates_iter = pd.date_range(start=s_date, end=e_date, freq=pd_freq[est_freq])

    dfw = df.pivot(index='real_date', columns='cid', values='value')
    cross_sections = dfw.columns

    # --- The actual scoring.

    dfw_zns_pan = dfw * 0
    dfw_zns_css = dfw * 0

    if pan_weight > 0:

        df_neutral = expanding_stat(dfw, dates_iter, stat=neutral,
                                    sequential=sequential, min_obs=min_obs,
                                    iis=iis)
        dfx = dfw.sub(df_neutral['value'], axis=0)
        df_mabs = expanding_stat(dfx.abs(), dates_iter, stat="mean",
                                 sequential=sequential,
                                 min_obs=min_obs, iis=iis)
        dfw_zns_pan = dfx.div(df_mabs['value'], axis='rows')

    if pan_weight < 1:

        for cid in cross_sections:
            dfi = dfw[cid]

            df_neutral = expanding_stat(dfi.to_frame(name=cid), dates_iter, stat=neutral,
                                        sequential=sequential,
                                        min_obs=min_obs, iis=iis)
            dfx = dfi - df_neutral['value']

            df_mabs = expanding_stat(dfx.abs().to_frame(name=cid), dates_iter,
                                     stat="mean", sequential=sequential,
                                     min_obs=min_obs, iis=iis)
            dfx = pd.DataFrame(data=dfx.to_numpy(),
                               index=dfx.index, columns=['value'])
            dfx = dfx.rename_axis('cid', axis=1)

            zns_css_df = dfx / df_mabs
            dfw_zns_css.loc[:, cid] = zns_css_df.to_numpy()

    dfw_zns = (dfw_zns_pan * pan_weight) + (dfw_zns_css * (1 - pan_weight))
    dfw_zns = dfw_zns.dropna(axis=0, how='all')
    
    if thresh is not None:
        dfw_zns.clip(lower=-thresh, upper=thresh, inplace=True)

    # --- Reformatting of output into standardised DataFrame.

    df_out = dfw_zns.stack().to_frame("value").reset_index()
    df_out['xcat'] = xcat + postfix

    col_names = ['cid', 'xcat', 'real_date', 'value']
    df_out = df_out.sort_values(['cid', 'real_date'])[col_names]

    return df_out[df.columns].reset_index(drop=True)


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2006-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2008-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2007-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2008-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    # Apply a blacklist period from series' start date.
    black = {'AUD': ['2010-01-01', '2013-12-31'],
             'GBP': ['2018-01-01', '2100-01-01']}

    dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)
    dfd["grading"] = np.ones(dfd.shape[0])

    # Monthly: panel + cross.
    dfzm = make_zn_scores(dfd, xcat='XR', sequential=True, cids=cids,
                          blacklist=black, iis=True, neutral='mean',
                          pan_weight=0.75, min_obs=261, est_freq="m")
    print(dfzm)

    # Weekly: panel + cross.
    dfzw = make_zn_scores(dfd, xcat='XR', sequential=True, cids=cids,
                          blacklist=black, iis=False, neutral='mean',
                          pan_weight=0.5, min_obs=261, est_freq="w")

    # Daily: panel. Neutral and standard deviation will be computed daily.
    dfzd = make_zn_scores(dfd, xcat='XR', sequential=True, cids=cids,
                          blacklist=black, iis=True, neutral='mean',
                          pan_weight=1.0, min_obs=261, est_freq="d")

    # Daily: cross.
    dfd['ticker'] = dfd["cid"] + "_" + dfd["xcat"]
    dfzd = make_zn_scores(dfd, xcat='XR', sequential=True, cids=cids,
                          blacklist=black, iis=True, neutral='mean',
                          pan_weight=0.0, min_obs=261, est_freq="d")

    panel_df = make_zn_scores(dfd, 'CRY', cids, start="2010-01-04", blacklist=black,
                              sequential=False, min_obs=0, neutral='mean',
                              iis=True, thresh=None, pan_weight=0.75, postfix='ZN')
