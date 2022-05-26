

import pandas as pd
import numpy as np
from itertools import accumulate
from macrosynergy.management.simulate_quantamental_data import make_qdf

def expanding_mean_with_nan(dfw: pd.DataFrame, absolute: bool = False):
    """
    Computes a rolling median of a vector of floats and returns the results. NaNs will be
    consumed.

    :param <pd.Dataframe> dfw: "wide" dataframe with time index and cross-sections as
        columns.
    :param <bool> absolute: if True, the rolling mean will be computed on the magnitude
        of each value. Default is False.

    :return <List[float] ret: a list containing the median values. The number of computed
        median values held inside the list will correspond to the number of timestamps
        the series is defined over.
    """

    assert isinstance(dfw, pd.DataFrame), "Method expects to receive a pd.DataFrame."
    error_index = "The index of the DataFrame must be timestamps."
    assert all([isinstance(d, pd.Timestamp) for d in dfw.index]), error_index
    assert isinstance(absolute, bool), "Boolean value expected."

    data = dfw.to_numpy()

    no_rows = dfw.shape[0]
    no_columns = dfw.shape[1]
    no_elements = no_rows * no_columns

    one_dimension_arr = data.reshape(no_elements)
    if absolute:
        one_dimension_arr = np.absolute(one_dimension_arr)

    rolling_summation = [np.nansum(one_dimension_arr[0:(no_columns * i)])
                         for i in range(1, no_rows + 1)]

    # Determine the number of active cross-sections per timestamp. Required for computing
    # the rolling mean.
    data_arr = data.astype(dtype=np.float32)
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

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd_xr = dfd[dfd['xcat'] == 'XR']

    dfw = dfd_xr.pivot(index='real_date', columns='cid', values='value')
    no_rows = dfw.shape[0]

    ret_mean = expanding_mean_with_nan(dfw)