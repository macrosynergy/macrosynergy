import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import datetime
import time
from collections import defaultdict

def simulate_ar(no_days: int, mean: float = 0, sd_mult: float = 1, ar_coef: float = 0.75):

    ar_params = np.r_[1, -ar_coef]
    ## AR(1) with coefficient 0.75 - high persistance from the first period.
    ar_proc = ArmaProcess(ar_params)
    ser = ar_proc.generate_sample(no_days)

    ## Standardise. The mean of AR(1) process should be close to zero.
    ser = (ser + mean) - np.mean(ser)
    
    return (sd_mult * ser) / np.std(ser)


def comp(row, begin, end):

    sdate = pd.to_datetime(max(row[0], begin))
    edate = pd.to_datetime(min(row[1], end))

    return [sdate, edate]

def date_range(df_se):
    
    df_dates = {}
    
    for index in df_se.index:
        
        dates = df_se.loc[index, :].to_numpy()
        all_days = pd.date_range(dates[0], dates[1])
        df_dates[index] = all_days[all_days.weekday < 5].to_numpy()

    return df_dates

def stack_df(dict_, string):
    list_ = []

    for k, v in dict_.items():

        index = v.index
        if string == 'startyears':

            v = v.to_frame()
            v_ = pd.DatetimeIndex(v.loc[:, 'Sdate']).year

            d_ = dict(zip(index.values, v_.values))
            v = pd.Series(data = d_, index = list(d_.keys()))

            d = {ind: v[ind] for ind in index}

        else:
            d = {ind: v[ind].strftime('%Y-%m-%d') for ind in index}
        
        df = pd.DataFrame(data = d, index = [k])
        list_.append(df)
        
    output = pd.concat(list_)
    output = output.reindex(sorted(df.columns), axis = 1)

    output.index.name = 'cid'
    output.columns.name = 'xcat'
    
    return output

## The colon represents a type annotation used by static analysis tools to check the Type.
## back_ar = 0.75.
## ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef']
def make_qdf_(df_cids: pd.DataFrame, df_xcats: pd.DataFrame, back_ar: float = 0):

    frames = []

    cids_cats = defaultdict(list)
    start_dates = {}
    end_dates = {}
    
    fields = df_xcats.index
    fields_cats = list(set(fields))

    fields = df_cids.index
    fields_cids = list(set(fields))
    
    qdf_cols = ['cid', 'xcat', 'real_date', 'value']
    df_out = pd.DataFrame(columns = qdf_cols)

    ## Back Coefficient with communal, Standard Normal, background factor is added to category values.
    ## Compute values across the two end points and apply to all fields.
    ## Background noise that has not been accounted for.
    if any(df_xcats['back_coef'] != 0):

        s_comb = pd.concat([df_cids['earliest'], df_xcats['earliest']])
        e_comb = pd.concat([df_cids['latest'], df_xcats['latest']])

        s_date = np.min(s_comb)
        e_date = np.max(e_comb)

        all_days = pd.date_range(s_date, e_date)
        
        work_days = all_days[all_days.weekday < 5]
        no_days = len(work_days)
        
        ser = simulate_ar(no_days, mean = 0, sd_mult = 1, ar_coef = back_ar)
        df_back = pd.DataFrame(index = work_days, columns = ['Value'])
        df_back['Value'] = ser

    ## Generate an Autoregressive process for each country on all macroeconomic indicators.  
    for cid in df_cids.index:

        df_se = pd.DataFrame(columns = ['Sdate', 'Edate'])

        cid_row = df_cids.loc[cid].to_numpy()
    
        df_se = df_xcats.apply(lambda row: comp(cid_row, row[0], row[1]), axis = 1, result_type = 'expand')
        df_se.columns = ['Sdate', 'Edate']
        start_dates[cid] = df_se['Sdate']
        end_dates[cid] = df_se['Edate']
        
        df_dates = date_range(df_se)

        cid_mean = df_cids.loc[cid, 'mean_add']
        xcats_mean = df_xcats['mean_add'].to_numpy()
        ser_mean = xcats_mean + cid_mean ## Combine the country / field - specific method of moments to the distribution.

        cid_sd = df_cids.loc[cid, 'sd_mult']
        xcats_sd = df_xcats['sd_mult'].to_numpy()
        ser_sd = (xcats_sd * cid_sd)

        i = 0
        for k, v in df_dates.items():
            cids_cats[cid].append(k)
    
            ser_arc = df_xcats.loc[k, 'ar_coef']
            output = simulate_ar(len(v), mean = ser_mean[i], sd_mult = ser_sd[i], ar_coef = ser_arc)
            back_coef = df_xcats.loc[k, 'back_coef']
        
            if back_coef != 0:

                ## Locate on the specific dates. Extract the AR(1) value generated.
                ## Disturbance from an unknown variable: add to the outstanding AR process. The extent to the disturbance will be determined by the correlation coefficient.
                output = output + (back_coef * df_back.loc[v, 'Value'].reset_index(drop = True))
                output = output.to_frame(name = "value")
                output['real_date'] = v
                
                output['cid'] = cid
                output['cxcat'] = k
                output = output.reindex(sorted(output.columns), axis = 1)
                output = output.rename(columns = {"cxcat": "xcat"})

            frames.append(output)  
            i += 1

    final_df = pd.concat(frames, ignore_index = True)
    
    df_year = stack_df(start_dates, 'startyears')
    df_missing = stack_df(end_dates, 'enddates')

    return final_df, fields_cats, fields_cids, df_year, df_missing, cids_cats
                

def MAIN():
    ## Country IDs.
    cids = ['AUD', 'CAD', 'GBP', 'USD']
    ## Macroeconomic Fields: GDP, Inflation
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    ## The script is tasked with generating a return series for each macroeconomic variable on every country being analysed.
    ## However, if the AR(1) process is applied uniformally it suggests that all countries have the same distribution for each indicator which is not an accurate reflection of financial markets.
    ## Therefore, include some idiosyncratic fields, Mean and Standard Deviation, to change the distribution of the error term to align with that country.
    ## For instance, Brazil's currency has a persistently high variance which would not be accounted for in a standard AR(1) process with an error term that is normally distributed, and explains the need for the idiosyncratic fields.
    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add', 'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-10-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
    
    ## Numpy Array.
    start = time.time()
    dfd, fields_cats, fields_cids, df_year, df_missing, cids_cats = make_qdf_(df_cids, df_xcats, back_ar = 0.75)
    print(f"Time Elapsed, test_file: {time.time() - start}.")

if __name__ == "__main__":
    
    MAIN()
