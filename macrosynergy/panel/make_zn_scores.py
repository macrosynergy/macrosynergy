import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def make_zn_scores(df: pd.DataFrame, xcat: str, cids: List[str] = None, start: str = None, end: str = None,
                   sequential: bool = False, min_obs: int = 252, neutral: str = 'zero', thresh: float = None,
                   pan_weight: float = 1, postfix: str = 'ZN'):

    """
    Computes z-scores for a panel around a neutral level ("zn scores".
    
    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value.
    :param <str> xcat:  extended category for which the zn_score is calculated.
    :param <List[str]> cids: cross sections for which zn_scores are calculated; default is all available for category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <bool> sequential: if True (default) score parameters (neutral level and standard deviations) are estimated
        sequentially with concurrently available information only.
    :param <int> min_obs: the minimum number of observations required to calculate zn_scores. Default is 252.
    :param <str> neutral: method to determine neutral level. Default is 'zero'. Alternatives are 'mean' and "median".
    :param <float> thresh: threshold value beyond which scores are winsorized, i.e. contained at that threshold.
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

    if pan_weight > 0:

        if neutral == 'mean':  # array of panel means
            if sequential:  # sequential estimation of neutral level
                ar_neutral = np.array([dfw.iloc[0:(i + 1), :].stack().mean() for i in range(dfw.shape[0])])
            else:  # full sample estimation of neutral level
                ar_neutral = np.repeat(dfw.stack().mean(), dfw.shape[0])
        elif neutral == 'median':  # array of sequential panel medians
            if sequential:  # sequential estimation of neutral level
                ar_neutral = np.array([dfw.iloc[0:(i + 1), :].stack().median() for i in range(dfw.shape[0])])
            else:   # full sample estimation of neutral level
                ar_neutral = np.repeat(dfw.stack().median(), dfw.shape[0])
        else:
            ar_neutral = np.zeros(dfw.shape[0])

        dfx = dfw.sub(ar_neutral, axis='rows')  # df of excess values (minus neutrals)
        ar_sds = np.array([dfx.iloc[0:(i + 1), :].stack().abs().mean() for i in range(dfx.shape[0])])
        dfw_zns_pan = dfx.div(ar_sds, axis='rows')

    else:

        dfw_zns_pan = dfw * 0

    if pan_weight < 1:
        cross_sections = dfw.columns
        no_dates = dfw.shape[0]
        arr_neutral = np.zeros((no_dates, len(cross_sections)))

        for i, cross in enumerate(cross_sections):
            column = dfw.iloc[:, i] ## Pandas Series, as opposed to a DataFrame.

            # Todo: df or array of neutral values as column specific means.
            # Todo: df or array of neutral values as column specific medians.
            # Todo: df or array of zeores.
            if neutral == "mean":
                if sequential:
                    arr_neutral[:, i] = np.array([column[0:(j + 1)].mean() for j in range(no_dates)])
                else:
                    arr_neutral[:, i] = np.repeat(column.mean(), no_dates)
            elif neutral == "median":
                if sequential:
                    arr_neutral[:, i] = np.array([column[0:(j + 1)].median() for j in range(no_dates)])
                else:
                    arr_neutral[:, i] = np.repeat(column.median(), no_dates)
            else:
                pass

        
        # Todo: df of excess values.
        print(arr_neutral)
        dfx = dfw.sub(arr_neutral, axis = 'rows')

        # Todo: df or array of standard deviations around neutral value.
        ar_sds = np.zeros((no_dates, len(cross_sections)))
        for i, cross in enumerate(cross_sections):
            column = dfx.iloc[:, i]
            ar_sds[:, i] = np.array([column[0:(j + 1)].abs().mean() for j in range(no_dates)])

        # Todo: df of cross-section specific zn-scores.
        dfw_zns_css = dfx.div(ar_sds, axis = 'rows')
    else:

        dfw_zns_css = dfw * 0

    dfw_zns = (dfw_zns_pan * pan_weight) + (dfw_zns_css * (1 - pan_weight))
    dfw_zns[0:min_obs] = np.nan  # disallow zn scores for early sample with less observations than required
    if thresh is not None:
        dfw_zns.clip(lower=-thresh, upper=thresh, inplace=True)  # winsorization based on threshold

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

    df = make_zn_scores(dfd, xcat='CRY', sequential=True, cids=cids, neutral='mean', pan_weight=0.5)
    df = make_zn_scores(dfd, xcat='CRY', cids=cids, neutral='median', sequential=False, pan_weight = 0.3)
    df = make_zn_scores(dfd, xcat='CRY', cids=cids, neutral='zero', pan_weight = 0.01)

