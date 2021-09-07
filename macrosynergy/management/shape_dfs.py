import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import random
import time

from simulate_quantamental_data import make_qdf_

def data_frame(df_fields, m_fields, str_):
    
    if m_fields is None:
        m_fields = sorted(df_fields)
        
    if str_ == 'cids' and isinstance(m_fields, str):
        m_fields = [m_fields]

    missing = list(set(m_fields) - set(df_fields))

    if missing and str_ == 'xcats':
        print(f"Missing cross sections: {missing}.")
        xcats = [e for e in m_fields if e not in missing]
        return xcats
    
    elif missing:
        print(f"Missing cross sections: {missing}.")
        cids = list(set(m_fields).intersection(set(df_fields)))
        return sorted(cids)

    return m_fields
        

## Truncating the time period assessed over, and the macroeconomic indicators.
def reduce_df(df: pd.DataFrame, xcats_in_df: List[str] = None, cids_in_df: List[str] = None, cids_cats: dict = None,
              xcats: List[str] = None, cids: List[str] = None, start: str = None, end: str = None,
              blacklist: dict = None, out_all: bool = False, intersect: bool = False):

    
    dfx = df[df['real_date'] >= pd.to_datetime(start)] if start is not None else df
    dfx = dfx[dfx['real_date'] <= pd.to_datetime(end)] if end is not None else dfx

    if blacklist is not None:
        for key, value in blacklist.items():
            filt1 = dfx['cid'] == key[:3]
            filt2 = dfx['real_date'] >= pd.to_datetime(value[0])
            filt3 = dfx['real_date'] <= pd.to_datetime(value[1])
            dfx = dfx[~(filt1 & filt2 & filt3)]
    
    xcats = data_frame(xcats_in_df, xcats, 'xcats')
    dfx = dfx[dfx['xcat'].isin(xcats)]

    if intersect:
        
        df_uns = dfx.groupby('xcat')['cid'].unique()
        cids_in_df = list(df_uns[0])
        for i in range(1, len(df_uns)):
            cids_in_df = [cid for cid in df_uns[i] if cid in cids_in_df]
    
    cids = data_frame(cids_in_df, cids, 'cids')
    dfx = dfx[dfx['cid'].isin(cids)]

    ## Would there ever be any duplicates ?
    if out_all: return dfx, xcats, cids
    else:
        return dfx

def dict_year(s_year, e_year, years):

    s_years = range(s_year, e_year, years) ## Intervals controlled by the number of years.
    dict_ = {}
        
    for y in s_years:

        ey = (y + years - 1)
        ey = ey if (ey) <= e_year else "end_date"
        y_key = f"{y} - {ey}"
        if ey == "end_date": years = (e_year - y)
        y_value = list(range(y, y + years))
        dict_[y_key] = y_value

    return dict_


## Create two custom category dataframes with appropriate frequency and lags suitable for analysis.
## The subroutine will receive a dataframe that has been reduced to the two respective macroeconomic indicators whose relationship is to be analysed over the respective countries.
def categories_df(df, xcats, fields_cats = [], fields_cids = [], cids_cats = {}, cids = [], val = 'value',
                  start = None, end = None, blacklist = None, years = None, freq = 'M', lag = 0,
                  fwin = 1, xcat_aggs = ('mean', 'mean')):

    assert freq in ['D', 'W', 'M', 'Q', 'A']

    df, xcats, cids = reduce_df(df, fields_cats, fields_cids, cids_cats, xcats, cids , start, end, blacklist, out_all = True)
    
    col_names = ['cid', 'xcat', 'real_date', val]
    dfc = pd.DataFrame(columns = col_names)

    ## Scope for improvement.
    if years is None:
        
        for i in range(2):
            ## Isolate the two dataframes for the respective fields.
            dfw = df[df['xcat'] == xcats[i]].pivot(index = 'real_date', columns = 'cid', values = val)
            ## Convenience method for frequency conversion and resampling of time series, and subsequently compute the mean over the period.
            ## Average return over the month. The data generating process is an AR(1).
            dfw = dfw.resample(freq).agg(xcat_aggs[i])

            ## Forward Moving Average Window of first category.
            ## Size of the Moving Window. This is the number of observations used for calculating the statistic.
            if (i == 0) & (fwin > 1):
                dfw = dfw.rolling(window = fwin).mean().shift(1 - fwin)
                
            elif (i == 1) & (lag > 0):
                ## Shift the timeseries forward by a period(s) - dependent on the lag variable.
                dfw = dfw.shift(lag)

            ## Two dataframes will be produced for each economic indicator.
            ## Reset the index of the DataFrame to the default one instead. Each row will be "identified" by the date, and unpivot on the countries.
            dfx = pd.melt(dfw.reset_index(), id_vars = ['real_date'], value_vars = cids, value_name = val)
            dfx['xcat'] = xcats[i]
            dfc = dfc.append(dfx[col_names])

    else:
        
        s_year = pd.to_datetime(start).year ## Conversion to a datetime object.
        e_year = df['real_date'].max().year + 1
        year_groups = dict_year(s_year, e_year, years)

        keys_ = np.array(list(year_groups.keys()))
        
        def translate(row):
            for k, v in year_groups.items():
                if row in v:
                    break
            return k
        df['custom_date'] = df['real_date'].dt.year.apply(translate)

        for i in range(2):
            dfx = df[df['xcat'] == xcats[i]]
            ## Calculate the average return series across the prescribed intervals for every country ID. 
            dfx = dfx.groupby(['xcat', 'cid', 'custom_date']).agg(xcat_aggs[i]).reset_index()
            dfx = dfx.rename(columns = {"custom_date": "real_date"})
            dfc = dfc.append(dfx[col_names])
    
    return dfc.pivot(index=('cid', 'real_date'), columns = 'xcat', values = val).dropna()[xcats]


if __name__ == "__main__":

    cids = ['NZD', 'AUD', 'GBP', 'CAD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    random.seed(2)

    ## Returns a collection of Autoregressive Series for each country ID on the outlined macroeconomic indicators.

    start = time.time()
    final_df, fields_cats, fields_cids, df_year, df_missing, cids_cats = make_qdf_(df_cids, df_xcats, back_ar = 0.75)
    
    dfc1 = categories_df(final_df, ['GROWTH', 'CRY'], fields_cats, fields_cids, cids_cats, cids, 'value',
                         start = '2000-01-01', years = 5, freq = 'M', lag = 0, xcat_aggs = (['mean'] * 2))
    print(f"Time Elapsed, test_file: {time.time() - start}.")
    

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}


    dfc2 = categories_df(final_df, ['GROWTH', 'CRY'], fields_cats, fields_cids, cids_cats, cids, 'value',
                         start = '2000-01-01', freq = 'M', lag = 0, fwin = 3, xcat_aggs = ['mean', 'mean'],
                         blacklist = black)

    black = {'AUD_1': ['2000-01-01', '2009-12-31'], 'AUD_2': ['2018-01-01', '2100-01-01']}

    filt1 = ~((final_df['cid'] == 'AUD') & (final_df['xcat'] == 'XR'))
    filt2 = ~((final_df['cid'] == 'NZD') & (final_df['xcat'] == 'INFL'))
    dfdx = final_df[filt1 & filt2] 

