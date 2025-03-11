"""
Implementation of adjust_weights.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Type, Set, Callable
import warnings
from numbers import Number
from packaging import version
from macrosynergy.management.utils import (
    reduce_df,
    is_valid_iso_date,
    get_cid,
    get_xcat,
)
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame


PD_FUTURE_STACK = (
    dict(future_stack=True)
    if version.parse(pd.__version__) > version.parse("2.1.0")
    else dict(dropna=False)
)


def check_missing_cids_xcats(weights, adj_zns, cids, r_xcats, r_cids):
    """
    Checks if there are missing cids or xcats in the input DataFrame.
    """
    missing_xcats = list(set([weights, adj_zns]) - set(r_xcats))
    if missing_xcats:
        raise ValueError(f"Missing xcats: {missing_xcats}")

    missing_cids = list(set(cids) - set(r_cids))
    if missing_cids:
        raise ValueError(f"Missing cids: {missing_cids}")


def check_types(
    weights: str, adj_zns: str, method: Callable, param: Number, cids: List[str]
):
    """
    Type checking for the input variables of adjust_weights.
    """
    for _var, _name, _type in [
        (weights, "weights", str),
        (adj_zns, "adj_zns", str),
        (method, "method", Callable),
        (param, "param", Number),
        (cids, "cids", list),
    ]:
        if not isinstance(_var, _type):
            raise TypeError(f"{_name} must be a {_type}, not {type(_var)}")

    if not all(isinstance(cid, str) for cid in cids):
        raise TypeError("`cids` must be a list of strings")


def adjust_weights_backend(
    df_weights_wide: pd.DataFrame,
    df_adj_zns_wide: pd.DataFrame,
    method: Callable,
    param: Number,
) -> pd.DataFrame:
    """
    Backend function for adjust_weights. Applies the `method` function to the weights and
    multiplies the result by the adjustment factors, and by the parameter `param`.

    Parameters
    ----------
    df_weights_wide : pd.DataFrame
        DataFrame with weights in wide format.

    df_adj_zns_wide : pd.DataFrame
        DataFrame with adjustment factors in wide format.

    method : Callable
        Function that will be applied to the weights to adjust them. This function must

    param : Number
        Parameter that will be passed to the method function.

    Returns
    -------
    pd.DataFrame
        DataFrame with the adjusted weights.
    """

    assert set(df_weights_wide.columns) == set(df_adj_zns_wide.columns)
    cids = sorted(df_weights_wide.columns)
    results = []
    dfw_result = pd.DataFrame(index=df_weights_wide.index)

    for cid in cids:
        cid_weight = df_weights_wide[cid]
        cid_adj_zns = df_adj_zns_wide[cid]

        result = cid_weight * cid_adj_zns.apply(method) * param
        dfw_result[cid] = result

    return dfw_result


def split_weights_adj_zns(
    df: QuantamentalDataFrame, weights: str, adj_zns: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input DataFrame into two DataFrames, one containing the weights and the
    other containing the adjustment factors.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame containing the weights and adjustment factors.

    weights : str
        Name of the xcat to be used as weights.

    adj_zns : str
        Name of the xcat to be used as adjustment factors.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing two wide DataFrames (one for weights and one for adjustment
        factors), with one column per cid.
    """

    df_weights = df[df["xcat"] == weights]
    df_adj_zns = df[df["xcat"] == adj_zns]

    df_weights_wide = QuantamentalDataFrame(df_weights).to_wide()
    df_adj_zns_wide = QuantamentalDataFrame(df_adj_zns).to_wide()

    df_weights_wide.columns = get_cid(df_weights_wide)
    df_adj_zns_wide.columns = get_cid(df_adj_zns_wide)

    miss_left = list(set(df_weights_wide.columns) - set(df_adj_zns_wide.columns))
    miss_right = list(set(df_adj_zns_wide.columns) - set(df_weights_wide.columns))
    if miss_left or miss_right:
        raise ValueError(f"Mismatched columns: {miss_left + miss_right}")

    return df_weights_wide, df_adj_zns_wide


def normalize_weights(df_weights_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Weights are normalized by dividing each row by the sum of the row. Function exists to
    allow easy modification of normalization method.

    Parameters
    ----------
    df_weights_wide : pd.DataFrame
        DataFrame with weights in wide format. (one column per cid)

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized weights (sum of each row is 1).
    """
    df_weights_wide = df_weights_wide.div(df_weights_wide.sum(axis=1), axis=0)
    return df_weights_wide


def adjust_weights(
    df: QuantamentalDataFrame,
    weights: str,
    adj_zns: str,
    method: Callable,
    param: Number,
    cids: List[str] = None,
    normalize: bool = True,
    adj_name: str = "ADJWGT",
):
    """
    Adjusts the weights of a given xcat by a given adjustment xcat using a given method.

    Parameters
    ----------
    df : QuantamentalDataFrame
        QuantamentalDataFrame with weights and adjustment xcats for all cids.
    weights : str
        Name of the xcat containing the weights.
    adj_zns : str
        Name of the xcat containing the adjustment factors.
    method : Callable
        Function that will be applied to the weights to adjust them. This function must
        take a single array-like argument and return an array-like object of the same
        shape.
    param : Number
        Parameter that will be passed to the method function.
    cids : List[str], optional
        List of cids to adjust. If None, all cids will be adjusted. Default is None.
    normalize : bool, optional
        If True, the weights will be normalized before being adjusted. Default is True.
    adj_name : str, optional
        Name of the resulting xcat. Default is "ADJWGT".
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("df must be a QuantamentalDataFrame")

    df = QuantamentalDataFrame(df)
    result_as_categorical = df.InitializedAsCategorical

    if cids is None:
        cids = df["cid"].unique().tolist()

    check_types(weights, adj_zns, method, param, cids)

    df, r_xcats, r_cids = reduce_df(
        df, cids=cids, xcats=[weights, adj_zns], intersect=True, out_all=True
    )

    check_missing_cids_xcats(weights, adj_zns, cids, r_xcats, r_cids)

    df_weights_wide, df_adj_zns_wide = split_weights_adj_zns(df, weights, adj_zns)

    if normalize:
        df_weights_wide = normalize_weights(df_weights_wide)

    dfw_res = adjust_weights_backend(df_weights_wide, df_adj_zns_wide, method, param)

    dfw_res.columns = list(map(lambda x: f"{x}_{adj_name}", dfw_res.columns))

    df_res = QuantamentalDataFrame.from_wide(dfw_res, categorical=result_as_categorical)

    return df_res


if __name__ == "__main__":
    df = make_test_df(xcats=["weights", "adj_zns"], cids=["cid1", "cid2", "cid3"])

    def sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))

    param = 3.14

    df_res = adjust_weights(df, "weights", "adj_zns", sigmoid, param)
    print(df_res)
