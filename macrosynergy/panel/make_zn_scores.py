import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df

def pn_neutral(dfw: pd.DataFrame, neutral: str = 'zero', sequential: bool = False):

    if neutral == 'mean':
        if sequential:
            ar_neutral = np.array([dfw.iloc[0:(i + 1), :].stack().mean() for i in range(dfw.shape[0])])
        else:  
            ar_neutral = np.repeat(dfw.stack().mean(), dfw.shape[0])
    elif neutral == 'median':  
        if sequential:
            ar_neutral = np.array([dfw.iloc[0:(i + 1), :].stack().median() for i in range(dfw.shape[0])])
        else:  
            ar_neutral = np.repeat(dfw.stack().median(), dfw.shape[0])
    else:
        ar_neutral = np.zeros(dfw.shape[0])

    return ar_neutral

def cross_neutral(dfw: pd.DataFrame, neutral: str = 'zero', sequential: bool = False):
    cross_sections = dfw.columns
    no_dates = dfw.shape[0]
    arr_neutral = np.zeros((no_dates, len(cross_sections)))

    for i, cross in enumerate(cross_sections):
        column = dfw.iloc[:, i] ## Pandas Series, as opposed to a DataFrame.


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
            continue
        
    return arr_neutral

def nan_insert(dfw_zns: pd.DataFrame, min_obs: int = 252):
    active_dates = {}
    columns_ = dfw_zns.columns
    index_dates = list(dfw_zns.index)
    
    for i, col in enumerate(columns_):
        s = dfw_zns.iloc[:, i]
        date = s.first_valid_index()
        active_dates[col] = index_dates.index(date)

    for k, v in active_dates.items():
        dfw_zns[k][v: (v + min_obs)] = np.nan

    ## Returning the DataFrame is not strictly necessary given it's defined in the parent scope but required for the Unit Testing.
    return dfw_zns

def make_zn_scores(df: pd.DataFrame, xcat: str, cids: List[str] = None, start: str = None, end: str = None,
                   sequential: bool = False, min_obs: int = 252, neutral: str = 'zero', thresh: float = None,
                   pan_weight: float = 1, postfix: str = 'ZN'):

    
    assert neutral in ['median', 'zero', 'mean']
    if thresh is not None:
        assert thresh > 1, "The 'thresh' parameter must be larger than 1"
    assert 0 <= pan_weight <= 1, "The 'pan_weight' parameter must be between 0 and 1"

    df = reduce_df(df, xcats=[xcat], cids=cids, start=start, end=end)
    dfw = df.pivot(index='real_date', columns='cid', values='value')
    no_dates = dfw.shape[0]
    cross_sections = dfw.columns
    
    if pan_weight > 0:

        ar_neutral = pn_neutral(dfw, neutral, sequential)

        dfx = dfw.sub(ar_neutral, axis='rows')  # df of excess values (minus neutrals)
        ar_sds = np.array([dfx.iloc[0:(i + 1), :].stack().abs().mean() for i in range(dfx.shape[0])])
        dfw_zns_pan = dfx.div(ar_sds, axis='rows')

    else:
        dfw_zns_pan = dfw * 0

    if pan_weight < 1:

        arr_neutral = cross_neutral(dfw, neutral, sequential)
        dfx = dfw.sub(arr_neutral, axis = 'rows')

        ar_sds = np.zeros((no_dates, len(cross_sections)))
        for i, cross in enumerate(cross_sections):
            column = dfx.iloc[:, i]
            ar_sds[:, i] = np.array([column[0:(j + 1)].abs().mean() for j in range(no_dates)])

        dfw_zns_css = dfx.div(ar_sds, axis = 'rows')
    else:

        dfw_zns_css = dfw * 0

    dfw_zns = (dfw_zns_pan * pan_weight) + (dfw_zns_css * (1 - pan_weight))

    
    dfw_zns = nan_insert(dfw_zns, min_obs)
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

    df = make_zn_scores(dfd, xcat='CRY', sequential=True, cids=cids, neutral='mean', pan_weight=0.5)
    df = make_zn_scores(dfd, xcat='CRY', cids=cids, neutral='median', sequential=False, pan_weight = 0.3)
    df = make_zn_scores(dfd, xcat='CRY', cids=cids, neutral='zero', pan_weight = 0.01)

