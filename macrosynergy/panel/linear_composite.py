import numpy as np
import pandas as pd
from typing import *
import warnings

from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_qdf, make_test_df
from macrosynergy.management.utils import is_valid_iso_date


def linear_composite_on_cid(
    df: pd.DataFrame,
    xcats: Union[str, List[str]],
    cids: Optional[List[str]] = None,
    weights: Union[List[float], np.ndarray, pd.Series] = None,
    signs: Union[List[float], np.ndarray, pd.Series] = None,
    start: str = None,
    end: str = None,
    normalize_weights: bool = True,
    complete_xcats: bool = True,
    new_xcat="NEW",
):
    """Linear composite of various xcats across all cids and periods"""

    if not len(xcats) == len(weights) == len(signs):
        raise ValueError("xcats, weights, and signs must have same length")
    # TODO: weight not near 1 only a problem if normalize_weights is True
    # TODO:  hence this cannot be an or statement
    if not np.isclose(np.sum(weights), 1) or normalize_weights:
        if not normalize_weights:
            warnings.warn("`weights` does not sum to 1 and will be normalized. w←w/∑w")
        weights = weights / np.sum(weights)
        assert np.isclose(
            np.sum(weights), 1
        ), "Weights do not sum to 1. Normalization failed."

    if not np.all(np.isin(signs, [1, -1])):
        warnings.warn("signs must be 1 or -1. They will be coerced to 1 or -1.")
        signs = np.abs(signs) / signs  # should be faster?

    # main function is here and below.
    weights = pd.Series(weights * signs, index=xcats)

    dfc: pd.DataFrame = reduce_df(df=df, cids=cids, xcats=xcats, start=start, end=end)

    # dataframe with the xcats as columns and rows as cid-date combinations
    dfc_wide = dfc.set_index(["cid", "real_date", "xcat"])["value"].unstack(level=2)
    # dataframe for weights with same index as dfc_wide: each column will be a weight
    weights_wide = pd.DataFrame(
        data=[weights.sort_index()], index=dfc_wide.index, columns=dfc_wide.columns
    )

    # TODO: The below is all the normalization you need but should be conditional on
    #    normalize_weights == True.
    # boolean mask to help us work out the calcs
    mask = dfc_wide.isna()
    # series with an index of dfc_wide, and a value equal to the sum of the weights
    weights_sum = weights_wide[~mask].abs().sum(axis=1)
    # re-weighting the weights to sum to 1 considering the available xcats
    adj_weights_wide = weights_wide[~mask].div(weights_sum, axis=0)
    # final single series: the linear combination of the xcats and the weights

    out_df = (dfc_wide * adj_weights_wide).sum(axis=1)

    if complete_xcats:
        out_df[mask.any(axis=1)] = np.NaN
    else:
        out_df[mask.all(axis=1)] = np.NaN

    out_df = out_df.reset_index().rename(columns={0: "value"})
    out_df["xcat"] = new_xcat
    out_df = out_df[["cid", "xcat", "real_date", "value"]]

    return out_df


def linear_composite_on_xcat(
    df: pd.DataFrame,
    xcat: str,
    cids: List[str],
    weights: str,
    normalize_weights: bool = True,
    complete_cids: bool = False,
    new_cid="GLB",
):
    """Linear combination of one xcat across cids"""
    
    # TODO: The whole section only works for weights that are a category.
    #   but it should also work with fixed weights and signs as below.

    # sort and filter
    df = df.copy().sort_values(by=["cid", "xcat", "real_date"])
    cids_mask = df["cid"].isin(cids)

    # select the target and weights
    target_df: pd.DataFrame = df[(df["xcat"] == xcat) & cids_mask].copy()
    weights_df: pd.DataFrame = df[(df["xcat"] == weights) & cids_mask].copy()

    # set the targets and weights to wide indexing with cids as columns
    target_df = target_df.set_index(["real_date", "cid"])["value"].unstack(level=1)
    weights_df = weights_df.set_index(["real_date", "cid"])["value"].unstack(level=1)

    # TODO: Undocumented feature: if weights is the same as xcat there is no problem
    # Edge case where `weights` is the same as `xcat`, set weights to 1
    if weights is None or weights == "" or weights == xcat:
        weights_df = pd.DataFrame(
            data=np.ones(target_df.shape),
            index=target_df.index,
            columns=target_df.columns,
        )

    # TODO: For mormalization we need to multiply weights with signs of xcat values first
    # TODO: if weights are a category we need to assert that all values are positive
    # Normalize the weights to sum to 1 if specified
    if normalize_weights:
        weights_df = weights_df.div(weights_df.abs().sum(axis=1), axis=0)
        assert np.allclose(
            weights_df.abs().sum(axis=1), 1
        ), "Weights do not sum to 1. Normalization failed."

    # Form a mask to apply NaNs where the weight or the target is NaN
    nan_mask = target_df.isna() | weights_df.isna()

    # Apply the weights to the target
    out_df = target_df * weights_df

    # Drop NaN cids as specified
    if complete_cids:
        out_df = out_df[out_df.columns[~nan_mask.any(axis=0)]]
    else:
        out_df = out_df[out_df.columns[~nan_mask.all(axis=0)]]

    out_df = out_df.sum(axis=1).reset_index().rename(columns={0: "value"})

    out_df["cid"] = new_cid
    out_df["xcat"] = xcat
    out_df = out_df.reset_index()[["cid", "xcat", "real_date", "value"]].sort_values(
        by=["cid", "xcat", "real_date"]
    )

    return out_df


def linear_composite(
    df: pd.DataFrame,
    xcats: Union[str, List[str]],
    cids: Optional[List[str]] = None,
    weights: Optional[Union[List[float], np.ndarray, pd.Series, str]] = None,
    normalize_weights: bool = True,
    signs: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    complete_xcats: bool = True,
    complete_cids: bool = False,
    new_xcat="NEW",
    new_cid="GLB",
):
    """
    Weighted linear combinations of cross sections or categories

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <Union[str, List[str]> xcats: One or more categories to be combined.
        If a single category is given the linear combination is calculated across 
        sections. This results in a single series to which a new cross-sectional
        identifier is assigned.
        If more than pne category string is given the output will be a new category,
        i.e. a panel that is a linear combination of the categories specified.
    :param <List[str]> cids: cross-sections for which the linear combinations are
        calculated. Default is all cross-section available.
    :param <Union[List[float], str]> weights: This specifies how categories or cross 
        sections are combined. There are three principal options. 
        The first (default) is None, in which case equal weights are given to all 
        categories or cross sections that are available. 
        The second case is a set of fixed coefficients, in which case these very 
        coefficients are applied to all available categories of cross sections. 
        Per default the coefficients are normalized so that they add up to one for each 
        period. This can be changed with the argument `normalize_weights`. 
        The third case is the assignment of a weighting category. This only applies to 
        combinations of cross sections. In this care the weighting category is multiplied 
        for each period with the corresponding value of main category of the same cross 
        section. Per default the weight category values are normalized so that they add up 
        to one for each period. This can be changed with the argument `normalize_weights`.
    :param <bool> normalize_weights: If True (default) the weights are normalized to sum
        to 1. If False the weights are used as specified.
    :param <List[float]> signs: An array of consisting of +1s or -1s, of the same length
        as the number of categories in `xcats` to indicate whether the respective category
        should be added or subtracted from the linear combination. Not relevant when
        aggregating over cross-sections, i.e. when a single category is given in `xcats`.
        Default is None and all signs are set to +1.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the respective category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date for
        which the respective category is available is used.
    :param <bool> complete_xcats: If True (default) combinations are only calculated for
        observation dates on which all xcats are available. If False a combination of the
        available categories is used. Not relevant when aggregating over cross-sections,
        i.e. when a single category is given in `xcats`.
    :param <bool> complete_cids: If True (default) combinations are only calculated for
        observation dates on which all cids are available. If False a combination of the
        available cross-sections is used. Not relevant when aggregating over categories,
        i.e. when multiple categories are given in `xcats`.
    :param <str> new_xcat: Name of new composite xcat when aggregating over xcats for a
        given cid. Default is "NEW".
    :param <str> new_cid: Name of new composite cid when aggregating over cids for a given
        xcat. Default is "GLB".

    :return <pd.DataFrame>: standardized DataFrame with the relative values, featuring
        the categories: 'cid', 'xcat', 'real_date' and 'value'.
    """
    listtypes = (list, np.ndarray, pd.Series)

    # df check
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame")

    if not set(["cid", "xcat", "real_date", "value"]).issubset(set(df.columns)):
        raise ValueError(
            "`df` must be a standardized JPMaQS DataFrame with the necessary columns: "
            "'cid', 'xcat', 'real_date' and 'value'."
        )

    if df.empty:
        raise ValueError("`df` is empty")

    dfx: pd.DataFrame = df.copy()

    # dates check
    for varx, namex in zip([start, end], ["start", "end"]):
        if varx is not None:
            if not isinstance(varx, str):
                raise TypeError(f"`{namex}` must be a string")
            if not is_valid_iso_date(varx):
                raise ValueError(f"`{namex}` must be a valid ISO date")

    xcats = xcats.copy() if isinstance(xcats, list) else [xcats]
    cids = cids.copy() if cids is not None else dfx["cid"].unique().tolist()
    weights = weights.copy() if isinstance(weights, listtypes) else weights
    signs = signs.copy() if isinstance(signs, listtypes) else signs

    dfx["real_date"] = pd.to_datetime(dfx["real_date"])
    if start is None:
        start = dfx["real_date"].min()
    else:
        start = pd.to_datetime(start)

    if end is None:
        end = dfx["real_date"].max()
    else:
        end = pd.to_datetime(end)

    # crop the df
    dfx = dfx[(dfx["real_date"] >= start) & (dfx["real_date"] <= end)]

    # now make sure the specified cids are in the df
    if cids is not None:
        if (
            (not isinstance(cids, list))
            or (not len(cids))
            or (not all(isinstance(x, str) for x in cids))
        ):
            raise TypeError("`cids` must be a non-empty list of strings")
        if not all(x in dfx["cid"].unique() for x in cids):
            raise ValueError(f"not all cids in `cids` are available in DataFrame")
    else:
        cids: List[str] = sorted(list(dfx["cid"].unique()))

    # Branch off for the single category case
    if isinstance(xcats, str) or (isinstance(xcats, list) and len(xcats) == 1):
        xcats: str = xcats if isinstance(xcats, str) else xcats[0]
        if not xcats in dfx["xcat"].unique():
            raise ValueError(f"Category '{xcats}' not available in DataFrame")

        if isinstance(weights, str):
            if not weights in dfx["xcat"].unique():
                raise ValueError(f"Category '{weights}' not available in DataFrame")
        else:
            raise TypeError(
                "`weights` must be a string specifying a category to be used as a weight"
            )

        tdf = dfx[dfx["cid"].isin(cids)]
        for cid in tdf["cid"].unique():
            avail_xcats: Set[str] = set(tdf[tdf["cid"] == cid]["xcat"].unique())
            if not xcats in avail_xcats:
                warnings.warn(
                    f"Category '{xcats}' (target category) not available for cid '{cid}',"
                    f"dropping cid from `cids`"
                )
                cids.remove(cid)
            if not weights in avail_xcats:
                warnings.warn(
                    f"Category '{weights}' (used as weights) not available for cid '{cid}',"
                    f"dropping cid from `cids`"
                )
                cids.remove(cid)

        dfx = reduce_df(dfx, cids=cids, xcats=[xcats, weights], start=start, end=end)

        return linear_composite_on_xcat(
            df=dfx,
            xcat=xcats,
            cids=cids,
            weights=weights,
            normalize_weights=normalize_weights,
            complete_cids=complete_cids,
            new_cid=new_cid,
        )

    elif isinstance(xcats, list):
        if not all(isinstance(x, str) for x in xcats):
            raise TypeError("`xcats` must be a list of strings")
        if not all(x in dfx["xcat"].unique() for x in xcats):
            raise ValueError(f"Not all xcats in `xcats` are available in DataFrame")

        if weights is None:
            weights: np.ndarray = np.ones(len(xcats)) / len(xcats)
            # w←1/n

        if signs is None:
            signs: np.ndarray = np.ones(len(xcats))

        for varx, namex in zip([weights, signs], ["weights", "signs"]):
            if not isinstance(varx, listtypes):
                raise TypeError(
                    f"`{namex}` must be a list of floats, an np.ndarray or a pd.Series"
                )

        if not len(weights) == len(xcats) == len(signs):
            raise ValueError("`xcats`, `weights` and `signs` must have the same length")

        return linear_composite_on_cid(
            df=dfx,
            xcats=xcats,
            cids=cids,
            weights=weights,
            signs=signs,
            normalize_weights=normalize_weights,
            complete_xcats=complete_xcats,
            new_xcat=new_xcat,
        )


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP"]
    xcats = ["XR", "CRY", "INFL"]

    df: pd.DataFrame = pd.concat(
        [
            make_test_df(
                cids=cids,
                xcats=xcats[:-1],
                start_date="2000-01-01",
                end_date="2000-02-01",
                prefer="linear",
            ),
            make_test_df(
                cids=cids,
                xcats=["INFL"],
                start_date="2000-01-01",
                end_date="2000-02-01",
                prefer="decreasing-linear",
            ),
        ]
    )

    # all infls are now decreasing-linear, while everything else is increasing-linear

    lc_cid = linear_composite(
        df=df,
        xcats="CRY",
        weights="INFL",
    )

    lc_xcat = linear_composite(
        df=df,
        cids=["AUD", "CAD"],
        xcats=["XR", "CRY", "INFL"],
        weights=[1, 2, 1],
        signs=[1, -1, 1],
    )
