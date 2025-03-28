"""
Implementation of adjust_weights.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Dict, Any, Optional
import warnings
from numbers import Number
from macrosynergy.management.utils import reduce_df, get_cid
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.compat import PD_NEW_MAP

AVAILABLE_METHODS: List[str] = ["generic", "lincomb"]


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
    weights: str,
    adj_zns: str,
    method: str,
    adj_func: Callable,
    params: Dict[str, Any],
    cids: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """
    Type checking for the input variables of adjust_weights.
    """
    for _var, _name, _type in [
        (weights, "weights", str),
        (adj_zns, "adj_zns", str),
        (method, "method", str),
        (adj_func, "adj_func", (Callable, type(None))),
        (params, "param", dict),
        (cids, "cids", (list, type(None))),
        (start, "start", (str, type(None))),
        (end, "end", (str, type(None))),
    ]:
        if not isinstance(_var, _type):
            raise TypeError(f"{_name} must be a {_type}, not {type(_var)}")

    if cids is not None and (
        not all(isinstance(cid, str) for cid in cids) or len(cids) == 0
    ):
        raise TypeError("`cids` must be a None(default) or a non-empty list of strings")

    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Method {method} not available. Available methods: {AVAILABLE_METHODS}"
        )
    if method == "generic":
        if adj_func is None:
            raise ValueError("`adj_func` must be provided when method='generic'")


def lincomb_backend(
    df_adj_zns_wide: pd.DataFrame,
    df_weights_wide: pd.DataFrame,
    coeff_new: float,
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    """
    Linear combination of the parameters.

    Parameters
    ----------
    df_adj_zns_wide : pd.DataFrame
        DataFrame with adjustment factors in wide format.
    df_weights_wide : pd.DataFrame
        DataFrame with weights in wide format.
    coeff_new : float
        Coefficient (between 0 and 1) for the new weights. 1 means the result consists
        entirely of the new weights, 0 means the result consists entirely of the old
        weights.
    min_score : float, optional
        Minimum score for the adjustment factors. Default is None, where it is set to the
        minimum score discovered in the panel of `df_adj_zns_wide`.
    """

    assert set(df_weights_wide.columns) == set(df_adj_zns_wide.columns)
    assert set(df_weights_wide.index) == set(df_adj_zns_wide.index)

    if min_score is None:
        warnings.warn(
            "`min_score` not provided. Defaulting to minimum value from `df_adj_zns_wide`."
        )
        min_score = df_adj_zns_wide.min().min()

    err_str = "Parameter `coeff_new` must be provided as a floating point number between 0 and 1."
    if not isinstance(coeff_new, Number) or (
        isinstance(coeff_new, Number) and not 0 <= coeff_new <= 1
    ):
        raise ValueError(err_str)

    # Algorithm:
    # new_weight_basis[i, t] = max(adj_zns[i, t] - min_score, 0)
    # new_weight[i, t] = new_weight_basis[i, t] / sum(new_weight_basis[t])
    # output_raw_weight[i, t] = (1 - coeff_new) * old_weight[i, t] + coeff_new * new_weight[i, t]
    # output_weight[i, t] = output_raw_weight[i, t] / sum(output_raw_weight[i, t]))
    # where `i` is the cross-section and `t` is the date

    nwb = df_adj_zns_wide - min_score
    nwb[nwb < 0] = 0
    nw = nwb.div(nwb.sum(axis="columns"), axis="index")
    orw = (1 - coeff_new) * df_weights_wide + coeff_new * nw
    ow = orw.div(orw.sum(axis="columns"), axis="index")

    return ow


def generic_weights_backend(
    df_weights_wide: pd.DataFrame,
    df_adj_zns_wide: pd.DataFrame,
    adj_func: Callable,
    params: Dict[str, Any] = {},
) -> pd.DataFrame:
    """
    Backend function for adjust_weights. Applies the `method` function to the weights and
    multiplies the result by the adjustment factors, and by the parameter `param`.
    Expects the input DataFrames to be in wide format, with the same columns AND index
    (see macrosynergy.panel.adjust_weights.split_weights_adj_zns).

    Parameters
    ----------
    df_weights_wide : pd.DataFrame
        DataFrame with weights in wide format.

    df_adj_zns_wide : pd.DataFrame
        DataFrame with adjustment factors in wide format.

    method : Callable
        Function that will be applied to the weights to adjust them.

    params : Dict[str, Any], optional
        Parameters to be passed to the method function. Default is {}.

    Returns
    -------
    pd.DataFrame
        DataFrame with the adjusted weights.
    """

    assert set(df_weights_wide.columns) == set(df_adj_zns_wide.columns)
    assert set(df_weights_wide.index) == set(df_adj_zns_wide.index)

    if PD_NEW_MAP:
        dfw_result = df_weights_wide * df_adj_zns_wide.map(adj_func, **params)
    else:
        dfw_result = df_weights_wide * df_adj_zns_wide.applymap(adj_func, **params)

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
        Name of the z-n score xcat to be used as adjustment factors.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing two wide DataFrames (one for weights and one for adjustment
        factors), with one column per cid.
    """

    df_weights_wide = QuantamentalDataFrame(df.loc[df["xcat"] == weights]).to_wide()
    df_adj_zns_wide = QuantamentalDataFrame(df.loc[df["xcat"] == adj_zns]).to_wide()

    # cannot tolerate negative weights
    if any(df_weights_wide[~df_weights_wide.isna()].lt(0).any()):
        na_frame = QuantamentalDataFrame.from_wide(
            df_weights_wide[
                df_weights_wide[~df_weights_wide.isna()].lt(0).any(axis="columns")
            ]
        )

        na_frame = na_frame[na_frame["value"] < 0]
        raise ValueError(
            f"Negative weights found in the dataframe. Please check the following data:\n{na_frame}"
        )

    combined_index = df_weights_wide.index.union(df_adj_zns_wide.index)
    df_weights_wide = df_weights_wide.reindex(combined_index)
    df_adj_zns_wide = df_adj_zns_wide.reindex(combined_index)

    df_weights_wide.columns = get_cid(df_weights_wide.columns)
    df_adj_zns_wide.columns = get_cid(df_adj_zns_wide.columns)

    zns_missing_in_weights = set(df_adj_zns_wide.columns) - set(df_weights_wide.columns)
    weights_missing_in_zns = set(df_weights_wide.columns) - set(df_adj_zns_wide.columns)
    zns_missing_in_weights = [f"{c}_{adj_zns}" for c in zns_missing_in_weights]
    weights_missing_in_zns = [f"{c}_{weights}" for c in weights_missing_in_zns]
    all_missing = zns_missing_in_weights + weights_missing_in_zns
    if all_missing:
        raise ValueError(f"Missing tickers: {all_missing}")

    # get the corresponding rows in zns
    nan_zns_rows = df_adj_zns_wide.isna().all(axis="columns")
    all_zero_zns_rows = (df_adj_zns_wide.fillna(0) == 0).all(axis="columns")
    missing_zns_dates = df_adj_zns_wide.index[nan_zns_rows | all_zero_zns_rows]

    nan_weights_rows = df_weights_wide.isna().all(axis="columns")
    all_zero_weights_rows = (df_weights_wide.fillna(0) == 0).all(axis="columns")
    missing_weights_dates = df_weights_wide.index[
        nan_weights_rows | all_zero_weights_rows
    ]

    # if zn is missing, but weight is not missing, fill zn with 1
    missing_zns_dates = sorted(set(missing_zns_dates) - set(missing_weights_dates))
    if len(missing_zns_dates) > 0:
        estr = "Missing ZNs data (will be filled with 1 to preserve weights):"
        warnings.warn(f"{estr} {missing_zns_dates}")

        # replace missing zns data with standard weights
        df_adj_zns_wide.loc[missing_zns_dates] = 1

    return df_weights_wide, df_adj_zns_wide


def normalize_weights(
    out_weights: pd.DataFrame, normalize_to_pct: bool = False
) -> pd.DataFrame:
    """
    Output weights are normalized by dividing each row by the sum of the row. Function exists to
    allow easy modification of normalization method.

    Parameters
    ----------
    out_weights : pd.DataFrame
        DataFrame with weights in wide format. (one column per cid)

    normalize_to_pct : bool, optional
        If True, the resulting weights will be scaled to 100%. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized weights (sum of each row is 1).
    """
    out_weights = out_weights.div(out_weights.sum(axis="columns"), axis="index")

    norm_rows = out_weights.sum(axis="columns").apply(lambda x: np.isclose(x, 1))
    all_nan_rows = out_weights.index[out_weights.isnull().all(axis="columns")]

    # assert that all rows sum to 1 or are all NaN
    if not norm_rows.all() and all_nan_rows.size == 0:
        raise Exception("Normalization failed; weights do not sum to 1")

    if normalize_to_pct:
        out_weights = out_weights * 100

    return out_weights


def adjust_weights(
    df: QuantamentalDataFrame,
    weights_xcat: str,
    adj_zns_xcat: str,
    method: str = "generic",
    adj_func: Callable = None,
    params: Dict[str, Any] = {},
    cids: List[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Dict[str, Any] = None,
    normalize: bool = True,
    normalize_to_pct: bool = False,
    adj_name: str = "ADJWGT",
):
    """
    Adjusts the weights of a given xcat by a given adjustment xcat using a given method.
    The resulting weights will be scaled to sum to 100% for each date.

    Parameters
    ----------
    df : QuantamentalDataFrame
        QuantamentalDataFrame with weights and adjustment categories for all cross-sections.
    weights_xcat : str
        Name of the category containing the weights.
    adj_zns_xcat : str
        Name of the category containing the adjustment factors.
    method : Callable
        One of the available methods for adjusting weights. Default is "generic".
        See notes for available methods.
    adj_func : Callable, optional
        Function to be used for the adjustment when method is "generic". This function will
        be applied to the weights and multiplied by the adjustment factors. Default is None.
    params : Dict[str, Any], optional
        Parameters to be passed to the method function. Default is {}.
    cids : List[str], optional
        List of cross-sections to adjust. If None, all cross-sections will be adjusted. Default is None.
    start : str, optional
        Start date for the adjustment as YYYY-MM-DD. Default is None.
    end : str, optional
        End date for the adjustment as YYYY-MM-DD. Default is None.
    blacklist : Dict[str, Any], optional
        Blacklist dictionary passed to the reduce_df function. Default is None.
        See :meth:`macrosynergy.management.utils.df_utils.reduce_df` for more details.
    normalize : bool, optional
        If True, the resulting weights will be normalized to sum to one for each date for
        the entire list of cross-sections. Default is True.
    normalize_to_pct : bool, optional
        If True, the resulting weights will be scaled to 100%. Default is False.
        This only applies if `normalize` is True.
    adj_name : str, optional
        Name of the resulting xcat. Default is "ADJWGT".


    Returns
    -------
    QuantamentalDataFrame
        DataFrame with the adjusted weights.

    Notes
    -----
    Available methods:
    - "generic": Applies the method function to the weights and multiplies the result by the
        adjustment factors. The `method` function's signature must match:
        `method(weight: float, **params) -> float`.

    - "lincomb": Linear combination of the parameters. The method function must accept a single
        argument (the weight) and return a single value (the adjusted weight). The parameters
        `min_score` (minimum score for the adjustment factors) and `coeff_new` (coefficient for
        the new weights) must be provided in the `params` dictionary. See
        macrosynergy.panel.adjust_weights.lincomb_backend for more details.

    Examples
    --------

    >>> df = make_test_df(xcats=["weights", "adj_zns"], cids=["cid1", "cid2", "cid3"])

    >>>

    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("df must be a QuantamentalDataFrame")

    df: QuantamentalDataFrame = QuantamentalDataFrame(df)
    result_as_categorical: bool = df.InitializedAsCategorical

    check_types(
        weights=weights_xcat,
        adj_zns=adj_zns_xcat,
        method=method,
        adj_func=adj_func,
        params=params,
        cids=cids,
        start=start,
        end=end,
    )

    df, r_xcats, r_cids = reduce_df(
        df,
        cids=cids,
        xcats=[weights_xcat, adj_zns_xcat],
        start=start,
        end=end,
        blacklist=blacklist,
        intersect=True,
        out_all=True,
    )
    if cids is None:
        cids = df["cid"].unique().tolist()

    check_missing_cids_xcats(weights_xcat, adj_zns_xcat, cids, r_xcats, r_cids)

    df_weights_wide, df_adj_zns_wide = split_weights_adj_zns(
        df, weights_xcat, adj_zns_xcat
    )

    # no need to normalize weights before applying the adjustment

    if method == "lincomb":
        dfw_result = lincomb_backend(
            df_adj_zns_wide=df_adj_zns_wide,
            df_weights_wide=df_weights_wide,
            coeff_new=params.get("coeff_new", None),
            min_score=params.get("min_score", None),
        )

    elif method == "generic":
        dfw_result = generic_weights_backend(
            df_weights_wide=df_weights_wide,
            df_adj_zns_wide=df_adj_zns_wide,
            adj_func=adj_func,
            params=params,
        )
    else:
        # this condition is covered in a check above
        raise ValueError(f"Method {method} not available.")  # pragma: no cover

    all_nan_rows = dfw_result.index[dfw_result.isnull().all(axis="columns")]
    if all_nan_rows.size > 0:
        err = "The following dates have no data after applying the adjustment, and will be dropped:"
        warnings.warn(f"{err} {all_nan_rows}")
        dfw_result = dfw_result.dropna(how="all", axis="rows")
    if normalize:
        # normalize and scale to 100%
        dfw_result = normalize_weights(dfw_result, normalize_to_pct)

    if dfw_result.isna().all().all():
        raise ValueError(
            "The resulting DataFrame is empty. Please check the input data,"
            " the method function, and it's parameters."
        )

    dfw_result.columns += f"_{adj_name}"
    qdf = QuantamentalDataFrame.from_wide(dfw_result, categorical=result_as_categorical)
    qdf = qdf.dropna(how="any", axis=0).reset_index(drop=True)
    return qdf


if __name__ == "__main__":
    df = make_test_df(xcats=["weights", "adj_zns"], cids=["cid1", "cid2", "cid3"])
    dfb = make_test_df(xcats=["some_xcat", "other_xcat"], cids=["cid1", "cid2", "cid4"])

    # nan_mask = np.random.rand(len(df)) < 0.01
    # df.loc[nan_mask, "value"] = np.nan
    # nan_mask = np.random.rand(len(df)) < 0.1
    # df.loc[nan_mask, "value"] *= -1

    df = pd.concat([df, dfb], axis=0)

    # Using the lincomb method

    df_res = adjust_weights(
        df=df,
        weights_xcat="weights",
        adj_zns_xcat="adj_zns",
        method="lincomb",
        params={"min_score": None, "coeff_new": 0.5},
    )

    assert np.allclose(df_res.groupby("real_date")["value"].sum(), 1)

    # Using the generic method

    def sigmoid(x, amplitude=1.0, steepness=1.0, midpoint=0.0):
        """Sigmoid function with parameters for amplitude, steepness, and midpoint."""
        return amplitude / (1 + np.exp(-steepness * (x - midpoint)))

    params = {"amplitude": 1, "steepness": 4, "midpoint": 1}

    df_res = adjust_weights(
        df=df,
        weights_xcat="weights",
        adj_zns_xcat="adj_zns",
        method="generic",
        adj_func=sigmoid,
        params=params,
    )

    assert np.allclose(df_res.groupby("real_date")["value"].sum(), 1)

    print(df_res)
