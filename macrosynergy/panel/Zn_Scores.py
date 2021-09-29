import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import time
from collections import defaultdict
from itertools import islice
from bisect import insort
from macrosynergy.management.simulate_quantamental_data import make_qdf

class RollingMedian(object):
    '''
    Applies the Bisection Algorithm to find the median value in O(log(n)) time instead of O(n) time.
    Will iterate through the Array, and calculate the evolving median as each day is realised. The median is returned through a Generator Object.

    :attribute <np.ndarray> iterable: The entire return series.

    :return <float>: the median value through a Generator.
    '''
    def __init__(self, iterable):
        self.it = iter(iterable)
        self.list = list(islice(self.it, 1))
        self.sortedlist = sorted(self.list)

    @staticmethod
    def isOdd(num):

        return (num & 1)

    def __iter__(self):
        
        list_ = self.list
        sortedlist = self.sortedlist
        length = len(list_)
        midpoint = length // 2
        yield sortedlist[midpoint]

        ## Iterate through the entire return series, and adjust for whether there are even or odd number of elements at the current time-period.
        for newelem in self.it:
            length += 1
            midpoint = length // 2

            insort(sortedlist, newelem)
            if not self.isOdd(length):
                sum_ = sortedlist[midpoint] + sortedlist[midpoint - 1]
                yield sum_ / 2 
            else:   
                yield sortedlist[midpoint]


def dimension(arr, n):
    """
    Used for broadcasting a one element Array to an Array of size n using Stride Tricks: element populated n times.

    :param <np.ndarray> arr: One dimensional Array.
    :param <int> n: Integer delimiting the desired length.

    :return <np.ndarray>: arr.shape = (n, ).
    """
    return np.lib.stride_tricks.as_strided(arr,
                                          shape = (n, ) + arr.shape,
                                          strides = (0, ) + arr.strides)

def rolling_std(v):
    """
    Receives the return series and instantiates it in the enclosing scope.
    
    :param <np.ndarray> v: One-dimensional Array of the return series.

    :return <func>: Returns a function to complete the execution.
    """
    def compute(mean, n):
        """
        Receives two values: the current mean and the number of days passed in the series.

        :param <float> mean: Mean up until the current time-period.
        :param <int> n: The number of days elapsed.

        :return <float>: Standard Deviation up until the current time period.
        """
        nonlocal v
        
        mean = np.array(mean)
        mean = dimension(mean, n)

        active_v = v[:n]
        numerator = np.sum((active_v - mean) ** 2)
        rational = numerator / n
        
        return np.sqrt(rational)

    return compute


def z_rolling(arr, neutral, min_obs):
    """
    Computes a rolling z_score for the entire time-series by calculating the evolving median / median / std.

    :param <np.ndarray> std: Two-dimensional Array of the dates and return series.
    :param <str> neutral: 'mean', 'median' or 'zero'.   
    :param <int> min_obs: the minimum number of observations required in the return series to activate the functionality. Default is 252.

    :return <np.ndarray>: Two-dimensional Array of the original return series and its computed z_score.
    """
    arr = arr[0][:, 1]
        
    length = len(arr)
    assert length > min_obs
        
    days = np.linspace(1, length, length, dtype = int)
            
    ret = np.cumsum(arr) ## Cumulative sum.
    rolling_mean = ret / days

    if neutral == 'median':
        ## Will calculate the rolling median for each day of the series, as it evolves. Alternative is to use pandas.rolling() and have an expanding window size.
        median_series = list(RollingMedian(arr))
        median_series = np.array(median_series)
        numerator = arr - median_series
    elif neutral == 'zero':
        zero_series = np.zeros(length)
        numerator = arr - zero_series
    else:
        numerator = arr - rolling_mean

    ## Will calculate the evolving Standard Deviation using Numpy's Vectorisation technique.
    func = rolling_std(arr)
    vfunc = np.vectorize(func)
    ## Vectorisation technique will pass down the two Arrays iteratively.
    std = vfunc(rolling_mean, days)

    z_scores = np.zeros(length)
    z_scores[1:] = np.divide(numerator[1:], std[1:])

    return np.column_stack((arr, z_scores))

def func(str_, int_):
    
    return [str_] * int_

def mult(xcats, list_):

    return list(map(func, xcats, list_))

class dfManipulation:
    """
    Receives a traditional dataframe and filters on the xcat received. Will iterate through the cross-sections,
    and subsequently create a dictionary where each Key equates to a cross-section and the Value will be the respective
    return series. Will also determine the number of dates each cross-section is defined over which is used to
    reconstruct the dataframe.
    
    """
    
    def __init__(self, df, cids, xcat):
        self.dfd = dfd = df
        self.cids = cids
        self.xcats = xcat
        self.d_xcat = defaultdict(list)
        self.cid_xcat = {}

    def cids_iter(self):
        
        for cid in self.cids:
            self.__dict__['cid'] = cid
            self.__dict__['df_temp'] = dfd[dfd['cid'] == cid]
            self.cid_xcat[cid] = list(map(self.filt_xcat, self.xcats))
                           
    def filt_xcat(self, x_cat):

        df_ = self.df_temp[self.df_temp['xcat'] == x_cat]
        data = df_[['real_date', 'value']].to_numpy()
        
        self.d_xcat[self.cid].append(len(data[:, 0]))
        return data


def standard_dev(mean, i = -1):
    """
    First Class Function used to compute the Standard Deviation for each row of data.
    Using a First Class Function, quasi-Class, to build out the scope to allow more parameters to be defined in the enclosing Scope.

    :param <np.ndarray> std: One-dimensional Array of the mean value for each row of the return matrix: std.shape = (dates, ).
    :param <int> i: Integer parameter used to track the index as the function iterates down the matrix.

    :return <function> compute: Returns the internal function.
    """
    def compute(row):
        """
        Compute the Standard Deviation for each row received.
            
        :param <np.ndarray> row: Row of cross-sectional data for a particuliar timestamp.

        :return <float> np.sqrt(rational): Returns the standard deviation of the row.
        """
        nonlocal mean, i ## Allows the parameters to be modified inside the internal scope.
        i += 1
        mean_row = mean[i] 
        
        data = row[np.nonzero(row)]
        countries = len(data)
        numerator = (data - mean_row) ** 2
        numerator = np.sum(numerator)
        rational = numerator / countries
    
        return np.sqrt(rational)

    return compute


def ordering(df, arr):
    """
    Function used to confirm the ordering of the cross_sections in the Numpy Array.

    :param <pd.Dataframe> df: DataFrame consisting of a single row of data with the column names.
    :param <np.ndarray> arr: The same row of data in the matrix. It is used to determine the order of the columns after the switch from the DataFrame to the Array.

    :return <List[str]> list_: The cross_sections the DataFrame is defined over in the exact order the data is held in the Array.
    """
    
    dict_ = df.to_dict()
    list_ = np.zeros(len(arr), dtype = object)
    
    for k, v in dict_.items():
        index_ = np.where(arr == v)
        index_ = int(index_[0][0])
        list_[index_] = k
        
    return list_


def z_formula(arr, neutral, std):
    """
    Computes the z-score for every cross-sections across the entire time-series using Numpy's Vectorisation.
    
    :param <np.ndarray> arr: Array of each cross-section's return series: arr.shape = (dates, cross_sections).
    :param <str> neutral: 'mean', 'median' or 'zero'.
    :param <np.ndarray> std: One-dimensional Array of standard deviations for each row of the return matrix: std.shape = (dates, ).

    :return <np.ndarray> z_data: Array, of the same dimensions as the input, with the computed z_scores.
    """
    neutral = neutral[:, np.newaxis]
    std = std[:, np.newaxis]
    
    diff = np.subtract(arr, neutral)
    ## Each row will be divided by its respective standard deviation: row-wise division.
    z_data = np.divide(diff, std)

    return z_data        

def cross_section(dfd: pd.DataFrame, cids: List[str] = None, min_obs: int = 252,
                  neutral: str = 'zero', xcat: str = None):
    """
    Compute the Z-Score, designed for all neutral approaches, for each Cross-Section using the Mean and Standard Deviation computed
    using the entire cross-section of data for each respective day.

    :param <pd.Dataframe> dfd: standardized data frame with the following necessary columns:
    'cid', 'xcats', 'real_date' and 'value.
    :param <List[str]> cids: cross sections for which volatility is calculated;
        default is all available for the category.
    :param <int> min_obs: the minimum number of observations required in the return series to activate the functionality. Default is 252.
    :param <string> neutral: the methodology used to compute zn_score ('mean', 'median' & 'zero')
    :param <str> xcat:  extended category denoting the return series for which volatility should be calculated.

    :return <pd.Dataframe> df_output: DataFrame with the calculated zn_scores, using the cross-sectional data, of the chosen xcat.
    'cid', 'xcat', 'real_date' and 'z_score'.
    """
    
    df_temp = dfd[dfd['xcat'] == xcat]
    df_pivot = df_temp.pivot(index = 'real_date', columns = 'cid', values = 'value')
    arr = df_pivot[cids].values
    dates = df_pivot.index.to_numpy()

    data = np.nan_to_num(arr, copy = True) ## Convert all np.nans to zero values.
    
    act_countr = (data != 0.0).sum(axis = 1) ## Understand the number of active cross-sections for each date by determining the number of zeros on a specific row.
    index = np.argmax(act_countr == len(df_pivot.columns)) ## Locate the first row where all cross-sections have a realised return. Used to determine the cross section of each column.
    
    columns_ = ordering(df_pivot.iloc[index], arr[index, :]) ## Will return the order of the cross-sections.
    columns_ax = columns_[:, np.newaxis]

    output_arr = arr.copy()
    flag = False
    ## The number of cross-sections required per row to initiate a calculation. In this instance, more than one cross-section is required.
    if not np.all(act_countr > 1):
        ## Isolate the indices where the clause is satisfied. By definition all values in "act_countr" will be positive, non-zero.
        index_ = np.where(act_countr != 1)[0]
        index_remove = np.where(act_countr == 1)[0]
        ## Filter the data on the isolated indices.
        data = np.take(data, index_, axis = 0)
        arr = np.take(arr, index_, axis = 0)
        act_countr = np.take(act_countr, index_)
        flag = True
    
    l_dates = len(dates)
    ## In the output DataFrame, each cross-section will be defined the same number of times either with a return or np.nan.
    ## Therefore, use np.repeat() to create the df_output['cid'] column.
    cids_arr = np.repeat(columns_ax, l_dates, axis = 1)
    cids_arr = cids_arr.reshape((l_dates * len(columns_), ))

    ## Sum along the row and divide by the number of active cross-sections to compute the cross-sectional mean for each date.
    mean_ = np.divide(np.sum(data, axis = 1), act_countr)
    func = standard_dev(mean_) ## Instantiate the First-Class Function.
    ## Compute the standard deviation adjusting for the number of active cross-sections held on each row (row corresponds to a timestamp).
    std = np.apply_along_axis(func, axis = 1, arr = data)

    if neutral == 'mean':
        z_data = z_formula(arr, mean_, std)
    elif neutral == 'median':
        ## Compute the median, across each timestamp, adjusting for potential np.nan.
        median = np.nanmedian(arr, axis = 1)
        ## The Standard Deviation, for each row, has already been computed.
        z_data = z_formula(arr, median, std)
    else:
        zero = np.zeros((l_dates, ))
        z_data = z_formula(arr, zero, std)

    if flag:
        first_index = index_[0]
        output_arr[first_index:z_data.shape[0]] = z_data
    else:
        output_arr = z_data
        
    pivot_df = pd.DataFrame(data = output_arr, columns = columns_, index = dates)
    df_out = pivot_df.stack().reset_index().rename(mapper={0: 'value'}, axis=1)
    
    df_out = df_out.sort_values('level_0')
    df_output = pd.DataFrame(data = None, columns = ['Timestamp', 'cid', 'xcat', 'z_score'])
    
    df_output['Timestamp'] = df_out['level_0']
    df_output['cid'] = df_out['level_1']
    df_output['xcat'] = xcat
    df_output['z_score'] = df_out['value']
    
    return df_output
            

def out_sample(dfd: pd.DataFrame, cids: List[str] = None, min_obs: int = 252,
               neutral: str = 'zero', xcat: str = None):

    """
    Computes the Z_Score at each timestamp by calculating the time-dependent mean and standard deviation for every cross-section upto the current time-period, t.
    
    :param <pd.Dataframe> dfd: standardized data frame with the following necessary columns:
    'cid', 'xcats', 'real_date' and 'value.
    :param <List[str]> cids: cross sections for which volatility is calculated;
        default is all available for the category.
    :param <int> min_obs: the minimum number of observations required in the return series to activate the functionality. Default is 252.
    :param <string> neutral: the methodology used to compute zn_score ('mean', 'median' & 'zero').
    :param <str> xcat:  extended category denoting the return series for which volatility should be calculated.

    :return <pd.Dataframe> final_df: DataFrame with the calculated zn_scores, using the time-series data for each cross-section, of the chosen xcat.
    'cid', 'xcat', 'real_date' 'value' and 'z_score'.
    """

    cid_xcat_cl = {}
    cid_z = {}

    ## Instantiate an instance of the Class, dfManipulation, which filters the DataFrame down into the desired xcat and isolates each cross-section.
    df_inst = dfManipulation(dfd, cids, xcat)
    df_inst.cids_iter()
    ## Returns a dictionary where each Key is the cross-section and each value is a two-dimensional Array consisting of the dates & return series.
    cid_xcat_cl = df_inst.cid_xcat
    d_xcat = df_inst.d_xcat ## Dictionary: Key = cross-section; Value = length of the respective return series for that cross-section.

    ## Compute the rolling zn_score for each cross section and store in a dictionary.
    for cid in cids:
        cid_z[cid] = z_rolling(cid_xcat_cl[cid], neutral, min_obs)

    stat = 'z_score_' + neutral
    qdf_cols = ['cid', 'xcat', 'real_date', 'value', stat]
    df_lists = []
    
    for cid in cids:
        df_out = pd.DataFrame(columns = qdf_cols)
        df_out['xcat'] = np.concatenate(mult(xcat, d_xcat[cid]))
        df_out['real_date'] = cid_xcat_cl[cid][0][:, 0]
        df_out[['value', stat]] = cid_z[cid]
        df_out['cid'] = cid
        df_lists.append(df_out)

    final_df = pd.concat(df_lists, ignore_index = True)
    
    final_df = final_df.sort_values('real_date')
    final_df = final_df.reset_index(drop = True)
    
    return final_df


def zn_score(dfd: pd.DataFrame, cids: List[str] = None, oos: bool = False,
             cross: bool = False, min_obs: int = 252, neutral: str = 'zero',
             abs_max: float = None, weighting: bool = False, c_weight: float = 0.5,
             xcat: str = None):

    """
    Computes the Weighted Z_Score between cross-sectional and time-series data for each date.
    
    :param <pd.Dataframe> dfd: standardized data frame with the following necessary columns:
    'cid', 'xcats', 'real_date' and 'value.
    :param <List[str]> cids: cross sections for which volatility is calculated;
        default is all available for the category.
    :param <bool> oos: Out-of-Sample (time-series computation).
    :param <bool> cross: Cross-sectional data.
    :param <int> min_obs: the minimum number of observations required in the return series to activate the functionality. Default is 252.
    :param <string> neutral: the methodology used to compute zn_score ('mean', 'median' & 'zero').
    :param <float> abs_max: Maximum absolute value permitted for Zn_Score.
    :param <bool> weighting: If True, calculates a weighted average between Out-of-Sample and cross-section.
    :param <float> c_weight: Weight applied to cross-sectional data. Default is 0.5.
    :param <str> xcat:  extended category denoting the return series for which volatility should be calculated.

    :return <pd.Dataframe> final_df: DataFrame with the calculated zn_scores with potential weighting.
    'cid', 'xcat', 'real_date' 'value' and 'z_score'.
    """
    
    assert neutral in ['median', 'zero', 'mean']
    
    if weighting:
        assert c_weight < 1.0 and c_weight > 0.0

    if not weighting:
        if oos:
            xcat = [xcat]
            final_df = out_sample(dfd, cids = cids,
                                  neutral = neutral, xcat = xcat)
        else:
            final_df = cross_section(dfd, cids = cids,
                                     neutral = neutral, xcat = xcat)
    else:
        final_df = pd.DataFrame(data = None, columns =
                                ['Timestamp', 'cid', 'xcat', 'z_score_time', 'z_cross', 'Weighted'])
        xcat_ = [xcat]
        time_df = out_sample(dfd, cids = cids, neutral = neutral, xcat = xcat_)
        cross_df = cross_section(dfd, cids = cids, neutral = neutral, xcat = xcat)

        final_df['Timestamp'] = time_df['real_date']
        final_df['cid'] = time_df['cid']
        final_df['xcat'] = xcat
        final_df['z_score_time'] = time_df['value']
        final_df['z_cross'] = cross_df['z_score']
        weight = (cross_df['z_score'] * c_weight) + ((1 - c_weight) * time_df['value'])
        final_df['Weighted'] = weight
        
    return final_df

        

if __name__ == "__main__":
    ## Country IDs.
    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    ## Macroeconomic Fields: GDP, Inflation
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

    start = time.time()
    df = zn_score(dfd, cids = cids, oos = True, cross = False,
                  neutral = 'mean', weighting = True, c_weight = 0.5, xcat = 'XR')
    print(f"Time Elapsed: {time.time() - start}.")
