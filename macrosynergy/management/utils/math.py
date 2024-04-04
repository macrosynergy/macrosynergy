"""
Contains mathematical utility functions used across the package.
"""

from typing import List, Union, Tuple, Callable

import pandas as pd
import numpy as np
import itertools
import functools
import logging

from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.historic_vol import expo_weights


cache = functools.lru_cache(maxsize=None)

logger = logging.getLogger(__name__)

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

    :return <List[float] ret: a list containing the median values. The number of computed
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




@cache
def flat_weights_arr(lback_periods: int, *args, **kwargs) -> np.ndarray:
    """Flat weights for the look-back period."""
    return np.ones(lback_periods) / lback_periods


@cache
def expo_weights_arr(lback_periods: int, half_life: int, *args, **kwargs) -> np.ndarray:
    """Exponential weights for the look-back period."""
    return expo_weights(lback_periods=lback_periods, half_life=half_life)


def _weighted_covariance(
    x: np.ndarray,
    y: np.ndarray,
    weights_func: Callable[[int, int], np.ndarray],
    lback_periods: int,
    *args,
    **kwargs,
) -> float:
    """
    Estimate covariance between two series after applying weights.

    """
    assert x.ndim == 1 or x.shape[1] == 1, "`x` must be a 1D array or a column vector"
    assert y.ndim == 1 or y.shape[1] == 1, "`y` must be a 1D array or a column vector"
    assert x.shape[0] == y.shape[0], "`x` and `y` must have same length"

    # if either of x or y is all NaNs, return NaN
    if np.isnan(x).all() or np.isnan(y).all():
        return np.nan

    xnans, ynans = np.isnan(x), np.isnan(y)
    wmask = xnans | ynans
    weightslen = min(sum(~wmask), lback_periods if lback_periods > 0 else len(x))

    # drop NaNs and only consider the most recent lback_periods
    x, y = x[~wmask][-weightslen:], y[~wmask][-weightslen:]

    # assert x.shape[0] == weightslen  # TODO what happens if it is less...
    for arr in [x, y]:
        if arr.shape[0] < weightslen:
            warnings.warn(
                f"Length of x is less than the weightslen: {arr.shape[0]} < {weightslen}"
            )
    w: np.ndarray = weights_func(
        lback_periods=weightslen,
        half_life=min(weightslen // 2, kwargs.get("half_life", 11)),
    )

    xmean, ymean = (w * x).sum(), (w * y).sum()

    rss = (x - xmean) * (y - ymean)

    return w.T.dot(rss)


def estimate_variance_covariance(
    piv_ret: pd.DataFrame,
    # weights_func: Callable[[int, int], np.ndarray],
    # lback_periods: int,
    # half_life: int,
    # remove_zeros: bool,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Estimation of the variance-covariance matrix needs to have the following configuration options

    1. Absolutely vs squared deviations,
    2. Flat weights (equal) vs. exponential weights,
    3. Frequency of estimation (daily, weekly, monthly, quarterly) and their weights.
    """

    cov_mat = np.zeros((len(piv_ret.columns), len(piv_ret.columns)))
    logger.info(f"Estimating variance-covariance matrix for {piv_ret.columns}")

    for i_b, c_b in enumerate(piv_ret.columns):
        for i_a, c_a in enumerate(piv_ret.columns[: i_b + 1]):
            logger.debug(f"Estimating covariance between {c_a} and {c_b}")
            est_vol = _weighted_covariance(
                x=piv_ret[c_a].values,
                y=piv_ret[c_b].values,
                *args,
                **kwargs,
            )
            cov_mat[i_a, i_b] = cov_mat[i_b, i_a] = est_vol

    assert np.all((cov_mat.T == cov_mat) ^ np.isnan(cov_mat))

    return pd.DataFrame(cov_mat, index=piv_ret.columns, columns=piv_ret.columns)


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