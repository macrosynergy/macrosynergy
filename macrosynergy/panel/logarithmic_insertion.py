

import pandas as pd
import math
import numpy as np
import orderedstructs
from itertools import accumulate
from macrosynergy.management.simulate_quantamental_data import make_qdf

# Able to handle for NaN values which will invariably be present in the pivoted
# DataFrame.
def rolling_median_with_nan(dfw: pd.DataFrame):
    """
    Computes a rolling median of a vector of floats and returns the results. NaNs will be
    consumed.

    :param <pd.Dataframe> dfw: "wide" dataframe with time index and cross-sections as
        columns.

    :return <List[float] ret: a list containing the median values. The number of computed
        median values held inside the list will correspond to the number of timestamps
        the series is defined over.
    """

    data = dfw.to_numpy()
    no_rows = dfw.shape[0]
    no_columns = dfw.shape[1]
    no_elements = no_rows * no_columns

    # Flatten into a one-dimensional data structure. Use an auxiliary variables to
    # control for the number of cross-sections defined on a panel. The median is computed
    # across the panel daily.
    vector = data.reshape(no_elements)

    skip_list = orderedstructs.SkipList(float)
    ret = []

    count = 0
    non_nan = 0
    for i in range(no_elements):

        value = vector[i]
        # Counting the number of elements iterated through each window
        # (inclusive of NaN values).
        count += 1

        # NaN values are essentially excluded from the Data Structure when calculating a
        # rolling median.
        if not math.isnan(value):
            # Tracks the number of values held in the Skip List data structure. Required
            # to determine the median index.
            non_nan += 1

            # Inserting a value into a Skip List will occur in log(n) time. The data
            # structure will continuously store the data in sorted order which
            # subsequently avoids continuous re-ordering.
            skip_list.insert(float(value))

        if (count % no_columns) == 0:

            odd_index = non_nan // 2
            median_odd = skip_list.at(odd_index)

            if (non_nan % 2) == 0:

                even_index = odd_index - 1
                median_even = skip_list.at(even_index)
                median = (median_odd + median_even) / 2

            else:
                median = median_odd

            count = 0
            ret.append(median)

        else:
            continue

    return np.array(ret)

def rolling_mean_with_nan(dfw: pd.DataFrame):
    """
    Computes a rolling median of a vector of floats and returns the results. NaNs will be
    consumed.

    :param <pd.Dataframe> dfw: "wide" dataframe with time index and cross-sections as
        columns.

    :return <List[float] ret: a list containing the median values. The number of computed
        median values held inside the list will correspond to the number of timestamps
        the series is defined over.
    """

    data = dfw.to_numpy()

    no_rows = dfw.shape[0]
    no_columns = dfw.shape[1]
    no_elements = no_rows * no_columns

    one_dimension_arr = data.reshape(no_elements)
    rolling_summation = [np.nansum(one_dimension_arr[0:(no_columns * i)])
                         for i in range(1, no_rows + 1)]

    # Determine the number of active cross-sections per timestamp. Required for computing
    # the rolling mean.
    data_arr = data
    # Sum across the rows.
    active_cross = np.sum(~np.isnan(data_arr), axis=1)
    rolling_active_cross = list(accumulate(active_cross))

    mean_calc = lambda m, d: m / d
    ret = list(map(mean_calc, rolling_summation, rolling_active_cross))

    return np.array(ret)


if __name__ == "__main__":
    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

    # Define the cross-sections over different timestamps such that the pivoted DataFrame
    # includes NaN values: more realistic testcase.
    df_cids.loc['AUD'] = ['2022-01-01', '2022-02-01', 0.5, 2]
    df_cids.loc['CAD'] = ['2022-01-10', '2022-02-01', 0.5, 2]
    df_cids.loc['GBP'] = ['2022-01-20', '2022-02-01', -0.2, 0.5]
    df_cids.loc['USD'] = ['2022-01-01', '2022-02-01', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2022-01-05', '2022-02-01', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2022-02-01', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2022-02-01', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2022-02-01', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2022-02-01', 1, 2, 0.8, 0.5]

    print("Uses Ralph's make_qdf() function.")
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd_xr = dfd[dfd['xcat'] == 'XR']

    dfw = dfd_xr.pivot(index='real_date', columns='cid', values='value')
    no_rows = dfw.shape[0]

    ret_median = rolling_median_with_nan(dfw)

    ret_mean = rolling_mean_with_nan(dfw)