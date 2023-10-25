"""
Contains mathematical utility functions used across the package.
"""

from typing import List, Union, Tuple
from ..types import QuantamentalDataFrame, Numeric

import pandas as pd
import numpy as np
import itertools


def expanding_mean_with_nan(dfw: pd.DataFrame, absolute: bool = False):
    """
    Computes a rolling median of a vector of floats and returns the results. NaNs will be
    consumed.

    :param <QuantamentalDataFrame> dfw: "wide" dataframe with time index and cross-sections as
        columns.
    :param <bool> absolute: if True, the rolling mean will be computed on the magnitude
        of each value. Default is False.

    :return <List[float] ret: a list containing the median values. The number of computed
        median values held inside the list will correspond to the number of timestamps
        the series is defined over.
    """

    # assert all([isinstance(d, pd.Timestamp) for d in dfw.index]), error_index
    # assert isinstance(absolute, bool), "Boolean value expected."

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

    no_rows = dfw.shape[0]
    no_columns = dfw.shape[1]
    no_elements = no_rows * no_columns

    one_dimension_arr = data.reshape(no_elements)
    if absolute:
        one_dimension_arr = np.absolute(one_dimension_arr)

    rolling_summation = [
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
