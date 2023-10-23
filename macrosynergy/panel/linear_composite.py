"""
Implementation of linear_composite() function as a module.

::docs::linear_composite::sort_first::

"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Type, Set
import warnings

from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_test_df
from macrosynergy.management.utils import (
    is_valid_iso_date,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
    get_cid,
    get_xcat,
)
from macrosynergy.management.types import Numeric, QuantamentalDataFrame

listtypes: Tuple[Type, ...] = (list, np.ndarray, pd.Series, tuple)


def _linear_composite_basic(
    data_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    normalize_weights: bool = True,
    complete: bool = False,
    mode: str = "xcat_agg",
):
    """Main calculation function for linear_composite()"""

    # Create a boolean mask to help us work out the calcs
    nan_mask: pd.DataFrame = data_df.isna() | weights_df.isna()

    # Normalize weights (if requested)
    if normalize_weights:
        adj_weights_wide = weights_df[~nan_mask].div(
            weights_df[~nan_mask].abs().sum(axis=1), axis=0
        )
        adj_weights_wide[nan_mask] = np.NaN

        assert np.allclose(
            adj_weights_wide[~adj_weights_wide.isna().all(axis=1)].abs().sum(axis=1), 1
        ), "Weights do not sum to 1. Normalization failed."

        weights_df = adj_weights_wide.copy()

    # Multiply the weights by the target data
    out_df = data_df * weights_df

    # Sum across the columns
    out_df = out_df.sum(axis="columns")

    # NOTE: Using `axis` with strings, to make it more readable
    # Remove periods with missing data (if requested) (rows with any NaNs)
    if complete:
        out_df[nan_mask.any(axis="columns")] = np.NaN

    # put NaNs back in, as sum() removes them
    out_df[nan_mask.all(axis="columns")] = np.NaN

    # Reset index, rename columns and return
    out_df = out_df.reset_index().rename(columns={0: "value"})

    # TODO: out_df from cid_agg and xcat_agg are not in the same format...

    return out_df


def linear_composite_cid_agg(
    df: pd.DataFrame,
    xcat: str,
    weights: Union[str, List[float]],
    signs: List[float],
    normalize_weights: bool = True,
    complete_cids: bool = True,
    new_cid="GLB",
):
    """Linear composite of various cids for a given xcat across all periods."""

    if isinstance(weights, str):
        weights_df: pd.DataFrame = df[(df["xcat"] == weights)].copy()
        df = df[(df["xcat"] != weights)].copy()
        weights_df = weights_df.set_index(["real_date", "cid"])["value"].unstack(
            level=1
        )
        weights_df = weights_df.mul(signs, axis=1)

    else:
        weights_series: pd.Series = pd.Series(
            np.array(weights) * np.array(signs),
            index=df["cid"].unique().tolist(),
        )
        weights_df = pd.DataFrame(
            data=[weights_series.sort_index()],
            index=pd.to_datetime(df["real_date"].unique().tolist()),
            columns=df["cid"].unique(),
        )

        weights_df.index.names = ["real_date"]
        weights_df.columns.names = ["cid"]

    # create the data_df
    data_df: pd.DataFrame = (
        df[(df["xcat"] == xcat)]
        .set_index(["real_date", "cid"])["value"]
        .unstack(level=1)
    )
    # aligning the index of weights_df to the data one
    # so that we have the same set of dates and same set of CIDs -- thank you @mikiinterfiore
    weights_df = (
        weights_df.stack(dropna=False)
        .reindex(data_df.stack(dropna=False).index)
        .unstack(level=1)
    )

    # assert that data_df and weights_df have the same shape, index and columns
    assert (
        (data_df.shape == weights_df.shape)
        and (data_df.index.equals(weights_df.index))
        and (data_df.columns.equals(weights_df.columns))
    ), (
        "Unexpected shape of `data_df` and `weights_df`. "
        "Unable to shape data for calculation."
    )

    # Calculate the linear combination
    out_df: pd.DataFrame = _linear_composite_basic(
        data_df=data_df,
        weights_df=weights_df,
        normalize_weights=normalize_weights,
        complete=complete_cids,
        mode="cid_agg",
    )
    out_df["cid"] = new_cid
    out_df["xcat"] = xcat
    out_df = out_df[["cid", "xcat", "real_date", "value"]]
    return out_df


def linear_composite_xcat_agg(
    df: pd.DataFrame,
    weights: List[float],
    signs: List[float],
    normalize_weights: bool = True,
    complete_xcats: bool = True,
    new_xcat="NEW",
):
    """Linear composite of various xcats across all cids and periods"""

    # Create a weights series with the xcats as index
    weights_series: pd.Series = pd.Series(
        np.array(weights) * np.array(signs), index=df["xcat"].unique().tolist()
    )

    # Create wide dataframes for the data and weights
    data_df = df.set_index(["cid", "real_date", "xcat"])["value"].unstack(level=2)
    weights_df = pd.DataFrame(
        data=[weights_series.sort_index()],
        index=data_df.index,
        columns=data_df.columns,
    )

    # Calculate the linear combination
    out_df: pd.DataFrame = _linear_composite_basic(
        data_df=data_df,
        weights_df=weights_df,
        normalize_weights=normalize_weights,
        complete=complete_xcats,
        mode="xcat_agg",
    )
    out_df["xcat"] = new_xcat
    out_df = out_df[["cid", "xcat", "real_date", "value"]]
    return out_df


def linear_composite(
    df: pd.DataFrame,
    xcats: Union[str, List[str]],
    cids: Optional[List[str]] = None,
    weights: Optional[Union[List[float], str]] = None,
    normalize_weights: bool = True,
    signs: Optional[List[float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Dict[str, List[str]] = None,
    agg_cids: bool = True,
    agg_xcats: Optional[bool] = None,
    complete_xcats: bool = False,
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
        If more than one category string is given the output will be a new category,
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

    # df check
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("`df` must be a standardized Quantamental DataFrame.")

    # there must be a value column
    if "value" not in df.columns:
        raise ValueError("`df` must contain a `value` column.")

    if df["value"].isna().all():
        raise ValueError("`df` does not contain any valid values.")

    # copy df to avoid side effects
    # NOTE: The below arg validation contains code that "copy" the args.
    # Be careful when making changes.
    df: pd.DataFrame = df.copy()

    if start is None:
        start: str = pd.to_datetime(df["real_date"]).min().strftime("%Y-%m-%d")
    if end is None:
        end: str = pd.to_datetime(df["real_date"]).max().strftime("%Y-%m-%d")

    # dates check
    for varx, namex in zip([start, end], ["start", "end"]):
        if varx is not None:
            if not (isinstance(varx, str) and is_valid_iso_date(varx)):
                raise ValueError(f"`{namex}` must be a valid ISO date string.")

    # check xcats
    if xcats is None:
        xcats: List[str] = df["xcat"].unique().tolist()
    elif isinstance(xcats, str):
        xcats: List[str] = [xcats]
    elif isinstance(xcats, listtypes):
        xcats: List[str] = list(xcats)

    # check xcats in df
    if not set(xcats).issubset(set(df["xcat"].unique().tolist())):
        raise ValueError("Not all `xcats` are available in `df`.")

    # check cids
    if cids is None:
        cids: List[str] = df["cid"].unique().tolist()
    elif isinstance(cids, str):
        cids: List[str] = [cids]
    else:
        if not isinstance(cids, listtypes):
            raise TypeError("`cids` must be a string or list of strings.")

    # check cids in df
    if not set(cids).issubset(set(df["cid"].unique().tolist())):
        raise ValueError("Not all `cids` are available in `df`.")

    _xcat_agg: bool = len(xcats) > 1
    mode: str = "xcat_agg" if _xcat_agg else "cid_agg"

    ## Check weights
    expc_weights_len: int = len(xcats) if _xcat_agg else len(cids)
    if weights is None:
        weights: List[float] = [1 / expc_weights_len] * expc_weights_len
    elif isinstance(weights, str):
        weights: List[str] = [weights] * expc_weights_len

    if isinstance(weights, listtypes):
        if len(weights) != expc_weights_len:
            _temp: str = "cross-sections." if _xcat_agg else "categories."
            err_str: str = (
                f"`weights` must be a list of the same length as the number of "
                f"{_temp} ({expc_weights_len})."
            )

        if all(isinstance(w, str) for w in weights):
            if not _xcat_agg:
                raise ValueError(
                    "`weights` must be a list of floats when aggregating categories "
                    "for a given cross-section."
                )

    ## Validate weights
    category_weights: bool = False
    if isinstance(weights, list):
        if all(isinstance(w, Numeric) for w in weights):
            # normalize weights
            if normalize_weights:
                sw: float = sum(weights)
                weights: List[float] = [w / sw for w in weights]
        elif all(isinstance(w, str) for w in weights):
            # check weights in df
            if not set(weights).issubset(set(df["xcat"].unique().tolist())):
                raise ValueError("Not all `weights` are available in `df`.")
            category_weights: bool = True
        else:
            raise TypeError("`weights` must be a list of floats or strings.")

    ## Check signs
    if signs is None:
        signs: List[float] = [1] * expc_weights_len
    elif isinstance(signs, listtypes):
        if len(signs) != expc_weights_len:
            raise ValueError(
                f"`signs` must be a list of length {expc_weights_len} or a single "
                "float when aggregating a category across cross-sections."
            )
        if not all(isinstance(s, Numeric) for s in signs):
            raise TypeError("`signs` must be a list of floats.")
    else:
        raise TypeError("`signs` must be a list of floats.")

    ## Normalize signs
    if not all(s in [-1, 1] for s in signs):
        # are there any 0s?
        if any(s == 0 for s in signs):
            raise ValueError("`signs` must not contain any 0s.")
        warnings.warn(
            "`signs` must be a list of +1s or -1s. "
            "`signs` will be coerced to +1s/-1s. "
            "(i.e. signs ← abs(signs) / signs)"
        )
        signs: List[float] = [abs(s) / s for s in signs]

    if category_weights:
        for icid, cid in enumerate(cids.copy()):
            for xc, w in zip(xcats, weights):
                _fxc: pd.Series = (df["cid"] == cid) & (df["xcat"].isin([xc, w]))
                if not (set([xc, w]) == set(df[_fxc]["xcat"].unique().tolist())):
                    warnings.warn(
                        f"`{w}` not available for {cid} in {xc}. "
                        f"Removing {cid} from `cids`."
                    )
                    # remove the cid, weifht and sign
                    cids.pop(icid)
                    weights.pop(icid)
                    signs.pop(icid)

    ## Reduce dataframe
    _xcats: List[str] = xcats + (weights if category_weights else [])
    df, remaining_xcats, remaining_cids = reduce_df(
        df=df,
        xcats=_xcats,
        cids=cids,
        start=start,
        end=end,
        blacklist=blacklist,
        intersect=False,
        out_all=True,
    )
    if len(remaining_xcats) == 1 and len(remaining_cids) < len(cids) and not _xcat_agg:
        raise ValueError(
            "Not all `cids` have complete `xcat` data required for the calculation."
        )

    if df.empty:
        raise ValueError(
            "The arguments provided do not yield any data when filtered "
            "using reduce_df()."
        )

    ## Calculate linear combinations
    ...


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP"]
    xcats = ["XR", "CRY", "INFL"]

    df: pd.DataFrame = pd.concat(
        [
            make_test_df(
                cids=cids,
                xcats=xcats[:-1],
                start="2000-01-01",
                end="2000-02-01",
                style="linear",
            ),
            make_test_df(
                cids=cids,
                xcats=["INFL"],
                start="2000-01-01",
                end="2000-02-01",
                style="decreasing-linear",
            ),
        ]
    )

    # all infls are now decreasing-linear, while everything else is increasing-linear

    df.loc[
        (df["cid"] == "GBP")
        & (df["xcat"] == "INFL")
        & (df["real_date"] == "2000-01-17"),
        "value",
    ] = np.NaN

    df.loc[
        (df["cid"] == "AUD")
        & (df["xcat"] == "CRY")
        & (df["real_date"] == "2000-01-17"),
        "value",
    ] = np.NaN

    # there are now missing values for AUD-CRY and GBP-INFL on 2000-01-17

    lc_cid = linear_composite(
        df=df, xcats="XR", weights="INFL", normalize_weights=False
    )

    lc_xcat = linear_composite(
        df=df,
        cids=["AUD", "CAD"],
        xcats=["XR", "CRY", "INFL"],
        weights=[1, 2, 1],
        signs=[1, -1, 1],
    )
