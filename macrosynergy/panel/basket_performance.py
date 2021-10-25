import numpy as np
import pandas as pd
from typing import List, Union
from ..panel.historic_vol import expo_weights, expo_std, flat_std
from ..management.shape_dfs import reduce_df_by_ticker
from ..panel.converge_row import ConvergeRow


def check_weights(weight: pd.DataFrame):
    check = weight.sum(axis=1)
    c = ~((abs(check - 1) < 1e-6) | (abs(check) < 1e-6))
    assert not any(c), f"weights must sum to one (or zero), not: {check[c]}"


def max_weight_func(weights: pd.DataFrame, max_weight: float):
    """max weight function

    Function designed to determine if all weights computed are within the maximum weight
    allowed per cross-section. If the maximum weight is less than the expected weight,
    replace the computed weight with the expected weight. For instance,
    [np.nan, 0.63, np.nan, np.nan, 0.27] becomes [np.nan, 0.5, np.nan, np.nan, 0.5].
    Otherwise, call the ConvergeRow Class to ensure all weights "converge" to a value
    within the upper-bound. Allow for a margin of error set to 0.001.

    :param <pd.DataFrame> weights: Corresponding weight matrix. Multidimensional.
    :param <float> max_weight: Upper-bound on the weight allowed for each cross-section.

    :return <pd.DataFrame>: Will return the modified weight DataFrame.
    """

    weights = weights.fillna(0.0)
    w_matrix = weights.to_numpy()

    for i, row in enumerate(w_matrix):
        inst, row = ConvergeRow.application(row, max_weight)
        weights.iloc[i, :] = row

    cols = weights.columns
    weights[cols] = weights[cols].replace({'0': np.nan, 0: np.nan})
    return weights


def equal_weight(df_ret: pd.DataFrame) -> pd.DataFrame:
    """Equal weight function

    Function will receive two Arrays: one consisting of the multi-dimensional return
    series and the other a one-dimensional Array outlining the number of active cross-
    sections per timestamp: use the second Array to delimit the weight distribution for
    each timestamp.

    :param <pd.DataFrame> df_ret: data-frame with returns.

    :return: Will return the generated weight <pd.DataFrame> of weights.
    """

    act_cross = (~df_ret.isnull())
    uniform = (1 / act_cross.sum(axis=1)).values
    uniform = uniform[:, np.newaxis]
    broadcast = np.repeat(uniform, df_ret.shape[1], axis=1)

    weight = act_cross.multiply(broadcast)
    check_weights(weight=weight)

    return weight


def fixed_weight(df_ret: pd.DataFrame, weights: List[float]):
    """
    The fixed weight method will receive a matrix of values, same dimensions as the
    DataFrame, and use the values for weights. For instance, GDP figures.
    The values will be normalised and account for NaNs to obtain the formal weight
    matrix.

    :param <pd.DataFrame> df_ret: Return series matrix. Multidimensional.
    :param <List[float]> weights: List of floats determining weight allocation.
                                  Example GDP.

    :return <pd.DataFrame>: Will return the generated weight Array
    """

    act_cross = (~df_ret.isnull())

    weights = np.array(weights, dtype = np.float32)
    rows = act_cross.shape[0]
    broadcast = np.tile(weights, (rows, 1))

    weight = act_cross.multiply(broadcast)
    weight_arr = weight.to_numpy()
    weight[weight.columns] = weight_arr / np.sum(weight_arr, axis = 1)[:, np.newaxis]

    check_weights(weight)

    return weight


def inverse_weight(
    dfw_ret: pd.DataFrame,
    lback_meth: str = "xma",
    lback_periods: int = 21,
    remove_zeros: bool = True):
    """
    The weights will be computed by taking the inverse of the rolling standard deviation
    of each return series. The rolling standard deviation will be calculated either
    using the standard Moving Average or the Exponential Moving Average.
    Both Moving Average's will require a window to be populated with returns before a
    weight can be computed, and subsequently the preceding timestamps will be set to NaN
    until the window has been filled. Therefore, modify the original Return Matrix and
    the active cross-section Array to reflect the additional NaN values.

    :param <pd.DataFrame> dfw_ret: DataFrame pivot on the cross-sections.
    :param <str> lback_meth: Lookback method for "invsd" weighting method.
                             Default is "xma".
    :param <int> lback_periods: Lookback periods. Default is 21.  Refers to half-time for
                                "xma" and full lookback period for "ma".
    :param <Bool> remove_zeros: Removes the zeros. Default is set to True.

    :return <pd.DataFrame>: Will return the generated weight DataFrame.
    """

    if lback_meth == "ma":
        dfwa = (
                dfw_ret.rolling(window=lback_periods).agg(flat_std, remove_zeros)
                * np.sqrt(252)
        )
    else:
        half_life = lback_periods
        weights = expo_weights(lback_periods * 2, half_life)
        dfwa = dfw_ret.rolling(window=lback_periods * 2).agg(
               expo_std, w=weights, remove_zeros=remove_zeros) * np.sqrt(252)

    w = 1 / dfwa
    weight = w / w.sum(axis=1).values[:, np.newaxis]
    check_weights(weight)

    return weight


def values_weight(weights: pd.DataFrame):
    """
    The values weight method will receive a matrix of values produced by another category
    and the values held in the matrix will be used as the weights. The weight matrix
    should contain floating point values.

    :param <pd.DataFrame> weights: Multidimensional Array of exogenously computed weights.

    :return <pd.DataFrame>: Will return the generated weight Array
    """

    weights = weights / weights.sum(axis=1)[:, np.newaxis]
    check_weights(weights)

    return weights


def basket_performance(
    df: pd.DataFrame,
    contracts: List[str],
    ret: str = "XR_NSA",
    cry: str = "CRY_NSA",
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    weight_meth: str = "equal",
    lback_meth: str = "xma",
    lback_periods: int = 21,
    remove_zeros: bool = True,
    weights: List[float] = None,
    weight_xcat: str = None,
    max_weight: float = 1.0,
    basket_tik: str = "GLB_ALL",
    return_weights: bool = False,
):

    """Basket performance

    Computes approximate return and carry series for a basket of underlying contracts.

    :param <pd.Dataframe> df: Standardized DataFrame with the following columns:
                              'cid', 'xcats', 'real_date' and 'value.
    :param <List[str]> contracts: Base Tickers (combinations of cross sections and base
                                  categories) denoting contracts that go into the basket.
    :param <str> ret: Return Category postfix; default is "XR_NSA".
    :param <str> cry: Return Category postfix; default is "CRY_NSA".
                      Choose None if carry not available.
    :param <str> start: String representing earliest date. Default is None.
    :param <str> end: String representing the latest date. Default is None.
    :param <dict> blacklist: Cross-sections with date ranges that should be excluded from
                             the DataFrame. If one cross-section has several blacklist
                             periods, append numbers to the cross-section code.
    :param <str> weight_meth: Method for weighting constituent returns and carry.
                              Default is "equal": all constituents have the same weight.
                              Alternatives are:
                              "fixed": proportionate to vector values supplied separately.
                              "invsd": inverse of past return standard deviations.
                              "values": proportionate to a panel of values of another
                                        exogenous category.
                              "inv_values": inversely proportionate to a panel of
                                            exogenous values supplied separately).
    :param <str> lback_meth: Lookback method for "invsd" weighting method.
                             Default is "xma".
    :param <int> lback_periods: Lookback periods. Default is 21.  Refers to half-time for
                                "xma" and full lookback period for "ma".
    :param <Bool> remove_zeros: Removes the zeros. Default is set to True.
    :param <List[float]> weights: List of weights corresponding to the tickers.
                                  Only relevant if weight_meth = "fixed", "values" or
                                  "inv_values" is chosen.
    :param <str> weight_xcat: Extended category name of values used for "values" and
                              "inv_values" methods.
    :param <float> max_weight: Maximum weight permitted for a single cross section.
                               Default is 1: a restriction is not applied.
    :param <str> basket_tik: Name of basket base ticker for which return and (possibly)
                             carry are calculated. Default is "GLB_ALL".
    :param <bool> return_weights: Add cross-section weights to output dataframe
                                  and uses ticks with 'WGT' postfix.
                                  Default is False.


    :return <pd.Dataframe>: Standardized DataFrame with the basket performance data
                            in standard form:
                            'cid', 'xcat', 'real_date' and 'value'.
    """

    assert isinstance(ret, str), (
        f"return category must be a <str> not {type(ret)}: {ret}"
    )

    assert isinstance(contracts, list) and all(isinstance(c, str) for c in contracts), (
        f"contracts must be a <list> of <str> not: {contracts}"
    )

    # Extract the relevant tickers from the input data-frame
    ticks_ret = [c + "_" + ret for c in contracts]

    cry_flag = cry is not None
    if cry_flag:
        ticks_cry = [c + "_" + cry for c in contracts]
    else:
        ticks_cry = []
    tickers = ticks_ret + ticks_cry

    dfx = reduce_df_by_ticker(
        df, start=start, end=end, ticks=tickers, blacklist=blacklist
    )

    dfx["ticker"] = dfx["cid"] + "_" + dfx["xcat"]
    dfw_ret = dfx[dfx["ticker"].isin(ticks_ret)].pivot(index="real_date", columns="cid",
                                                       values="value")

    if cry_flag:
        dfw_cry = dfx[dfx["ticker"].isin(ticks_cry)].pivot(index="real_date",
                                                           columns="cid", values="value")

    ret_arr = dfw_ret.to_numpy()
    act_cross = (~dfw_ret.isnull()).sum(axis=1)

    if weight_meth == "equal":
        w_matrix = equal_weight(df_ret=dfw_ret)

    elif weight_meth == "fixed":
        assert dfw_ret.shape[1] == len(weights)
        w_matrix = fixed_weight(df_ret=ret_arr, weights=weights)

    elif weight_meth == "invsd":

        w_matrix = inverse_weight(dfw_ret=dfw_ret, lback_meth=lback_meth, lback_periods
                                  =lback_periods, remove_zeros=remove_zeros)
    elif weight_meth == "values" or weight_meth == "inv_values":

        assert dfw_ret.shape[1] == len(weights)
        # w_matrix = values_weight()
        raise NotImplementedError(
            "Not yet implemented - need to extract the panel of weights from the input"
            " DataFrame")
    else:
        raise NotImplementedError(f"Weight method unknown {weight_meth}")

    assert 0.0 < max_weight <= 1.0
    if max_weight < 1.0:
        w_matrix = max_weight_func(weights=w_matrix, max_weight=max_weight)

    select = ["ticker", "real_date", "value"]
    dfxr = (dfw_ret.fillna(0).multiply(w_matrix)).sum(axis=1) / w_matrix.sum(axis=1)
    dfxr.index.name = "real_date"
    store = [dfxr.to_frame("value").reset_index()
             .assign(ticker=basket_tik + "_" + ret)[select]]
    if cry_flag:
        dfcry = (dfw_cry.fillna(0).multiply(w_matrix)).sum(axis=1) / w_matrix.sum(axis=1)
        dfcry.index.name = "real_date"
        store.append(dfcry.to_frame("value").reset_index().assign(ticker=basket_tik +
                                                                  "_" + cry)[select])

    if return_weights:
        w_matrix.index.name = "real_date"
        w_matrix.columns.name = "cid"
        w = w_matrix.stack().to_frame("value").reset_index()
        w["ticker"] = w["cid"] + "_WGTS"

        assert ((0 <= w["value"]) & (w["value"] <= 1.0)).all(axis=0)

        store.append(w.loc[w.value > 0, select])

    data = pd.concat(store, axis=0, ignore_index=True)

    return data
