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
    normalize_weights: bool = False,
    complete_xcats: bool = True,
    new_xcat="NEW",
):

    if not len(xcats) == len(weights) == len(signs):
        raise ValueError("xcats, weights, and signs must have same length")
    if not np.isclose(np.sum(weights), 1) or normalize_weights:
        if not normalize_weights:
            warnings.warn("`weights` does not sum to 1 and will be normalized. w←w/∑w")
        weights = weights / np.sum(weights)
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
    normalize_weights: bool = False,
    complete_cids: bool = False,
    update_freq: str = "M",
    new_cid="GLB",
):
    # sort and filter
    df = df.copy().sort_values(by=["cid", "xcat", "real_date"])
    cids_mask = df["cid"].isin(cids)

    # seekect the target and weights
    target_df: pd.DataFrame = df[(df["xcat"] == xcat) & cids_mask].copy()
    weights_df: pd.DataFrame = df[(df["xcat"] == weights) & cids_mask].copy()

    # set the targets and weights to wide indexing with cids as columns
    target_df = target_df.set_index(["real_date", "cid"])["value"].unstack(level=1)
    weights_df = weights_df.set_index(["real_date", "cid"])["value"].unstack(level=1)

    # Edge case where `weights` is the same as `xcat`, set weights to 1
    if weights is None or weights == "" or weights == xcat:
        weights_df = pd.DataFrame(
            data=np.ones(target_df.shape),
            index=target_df.index,
            columns=target_df.columns,
        )

    # Normalize the weights to sum to 1 if specified
    if normalize_weights:
        weights_df = weights_df.div(weights_df.abs().sum(axis=0), axis=1)

    # Downsample the weights to the update frequency
    weights_df = (
        weights_df.resample(update_freq)
        .mean(numeric_only=True)
        .reindex(target_df.index, method="bfill")
    )

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
    update_freq: str = "M",
    signs: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    complete_xcats: bool = True,
    complete_cids: bool = False,
    new_xcat="NEW",
    new_cid="GLB",
):
    """
    Returns new category panel as linear combination of others as standard dataframe

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <Union[str, List[str]> xcats: If a single category
        is given, the linear combination is calculated for all cross-sections available
        for that category - aggregating over cross-sections specified in `cids`,
        resulting in a single output series labelled with the new cross section name
        specified by `new_cid`.
        If a list of categories is given, the linear combination is calculated for all
        cross-sections available for each category specified in `xcats`, resulting in a
        panel of series labelled with the new category name specified by `new_xcat` and
        the same cross-sections as the input data.
    :param <List[str]> cids: cross-sections for which the linear combination is to be
        calculated. Default is all cross-section available.
    :param <Union[List[float], np.ndarray, pd.Series, str]> weights: An array of weights
        of the same length as the number of categories in `xcats` to be used in the
        linear combination. If a single category is given in `xcats`, another category
        can be specified in `weights` to be used as weights.
    :param <str> update_freq: The sampling frequency of the output data. The output
        data will be downsampled to the specified frequency. Options are 'D', 'W', 'M',
        'Q', 'A'. Default is 'M' (monthly).
    :param <List[float]> signs: An array of [1, -1] of the same length as the number of
        categories in `xcats` to indicate whether the respective category should be
        added or subtracted from the linear combination. Not relevant when aggregating
        over cross-sections, i.e. when a single category is given in `xcats`.
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

    # udpate_freq check
    if not isinstance(update_freq, str):
        raise TypeError("`update_freq` must be a string")

    update_freq = update_freq.upper()
    if update_freq not in ["D", "W", "M", "Q", "A"]:
        raise ValueError("`update_freq` must be one of 'D', 'W', 'M', 'Q', 'A'")

    # dates check
    for varx, namex in zip([start, end], ["start", "end"]):
        if varx is not None:
            if not isinstance(varx, str):
                raise TypeError(f"`{namex}` must be a string")
            if not is_valid_iso_date(varx):
                raise ValueError(f"`{namex}` must be a valid ISO date")

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
    if isinstance(xcats, str):
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
            normalize_weights=False,
            complete_cids=complete_cids,
            new_cid=new_cid,
            update_freq=update_freq,
        )

    elif isinstance(xcats, list):
        if not all(isinstance(x, str) for x in xcats):
            raise TypeError("`xcats` must be a list of strings")
        if not all(x in dfx["xcat"].unique() for x in xcats):
            raise ValueError(f"Not all xcats in `xcats` are available in DataFrame")

        for varx, namex in zip([weights, signs], ["weights", "signs"]):
            if not isinstance(varx, listtypes):
                raise TypeError(
                    f"`{namex}` must be a list of floats, an np.ndarray or a pd.Series"
                )

        if weights is None:
            weights: np.ndarray = np.ones(len(xcats))

        if signs is None:
            signs: np.ndarray = np.ones(len(xcats))

        if not len(weights) == len(xcats) == len(signs):
            raise ValueError("`xcats`, `weights` and `signs` must have the same length")

        return linear_composite_on_cid(
            df=dfx,
            xcats=xcats,
            cids=cids,
            weights=weights,
            signs=signs,
            normalize_weights=True,
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
