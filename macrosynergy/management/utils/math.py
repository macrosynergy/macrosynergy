"""
Contains mathematical utility functions used across the package.
"""

import itertools
from numbers import Number
from typing import List

import numpy as np
import pandas as pd

from macrosynergy.management.simulate import make_qdf


def expanding_mean_with_nan(
    dfw: pd.DataFrame, absolute: bool = False
) -> List[np.float64]:
    """
    Computes a rolling median of a vector of floats and returns the results. NaNs will be
    consumed.

    :param <QuantamentalDataFrame> dfw: "wide" dataframe with time index and
        cross-sections as columns.
    :param <bool> absolute: if True, the rolling mean will be computed on the magnitude
        of each value. Default is False.

    :return <List[float]> ret: a list containing the median values. The number of computed
        median values held inside the list will correspond to the number of timestamps
        the series is defined over.
    """

    if not isinstance(dfw, pd.DataFrame):
        raise TypeError("Method expects to receive a pd.DataFrame.")

    # cast the index to pd.Timestamp, if error raise TypeError
    try:
        dfw.index = pd.to_datetime(dfw.index)
    except:
        raise TypeError("The index of the DataFrame must be of type `<pd.Timestamp>`.")

    if not isinstance(absolute, bool):
        raise TypeError("The parameter `absolute` must be of type `<bool>`.")

    data: np.ndarray = dfw.to_numpy()

    no_rows: int = dfw.shape[0]
    no_columns: int = dfw.shape[1]
    no_elements: int = no_rows * no_columns

    one_dimension_arr: np.ndarray = data.reshape(no_elements)
    if absolute:
        one_dimension_arr = np.absolute(one_dimension_arr)

    rolling_summation: List[float] = [
        np.nansum(one_dimension_arr[0 : (no_columns * i)])
        for i in range(1, no_rows + 1)
    ]

    # Determine the number of active cross-sections per timestamp. Required for computing
    # the rolling mean.
    data_arr = data.astype(dtype=np.float32)
    # Sum across the rows.
    active_cross = np.sum(~np.isnan(data_arr), axis=1)
    rolling_active_cross = list(itertools.accumulate(active_cross))

    mean_calc = lambda m, d: m / d
    ret = list(map(mean_calc, rolling_summation, rolling_active_cross))

    return np.array(ret)


def ewm_sum(df: pd.DataFrame, halflife: Number):
    """
    Compute the exponentially weighted moving sum of a DataFrame.
    
    :param <pd.DataFrame> df: DataFrame in the wide format for which to calculate weights.
    :param <Number> halflife: The halflife of the exponential decay.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Method expects to receive a pd.DataFrame.")
    if not isinstance(halflife, Number):
        raise TypeError("The parameter `halflife` must be of type `<Number>`.")
    
    weights = calculate_cumulative_weights(df, halflife)
    return df.ewm(halflife=halflife).mean().mul(weights, axis=0)
    
def calculate_cumulative_weights(df: pd.DataFrame, halflife: Number):
    """
    Calculate the cumulative moving exponential weights for a DataFrame. 
        
    :param <pd.DataFrame> df: DataFrame in the wide format for which to calculate weights.
    :param <Number> halflife: The halflife of the exponential decay.
    """
    n = len(df)
    raw_weights = [(1/2) ** (i / halflife) for i in range(n)]

    cumulative_weights = np.cumsum(raw_weights)
    
    return pd.Series(cumulative_weights, index=df.index)


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "USD", "NZD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )

    # Define the cross-sections over different timestamps such that the pivoted DataFrame
    # includes NaN values: more realistic testcase.
    df_cids.loc["AUD"] = ["2022-01-01", "2022-02-01", 0.5, 2]
    df_cids.loc["CAD"] = ["2022-01-10", "2022-02-01", 0.5, 2]
    df_cids.loc["GBP"] = ["2022-01-20", "2022-02-01", -0.2, 0.5]
    df_cids.loc["USD"] = ["2022-01-01", "2022-02-01", -0.2, 0.5]
    df_cids.loc["NZD"] = ["2022-01-05", "2022-02-01", -0.1, 2]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR"] = ["2010-01-01", "2022-02-01", 0, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2011-01-01", "2022-02-01", 1, 2, 0.9, 0.5]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2022-02-01", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2013-01-01", "2022-02-01", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd_xr = dfd[dfd["xcat"] == "XR"]

    dfw = dfd_xr.pivot(index="real_date", columns="cid", values="value")
    no_rows = dfw.shape[0]

    ret_mean = expanding_mean_with_nan(dfw)
