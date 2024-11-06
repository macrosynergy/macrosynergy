"""
Implementation of linear_composite() function as a module.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Type, Set
import warnings
from packaging import version
from macrosynergy.management.utils import reduce_df, is_valid_iso_date
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame

listtypes: Tuple[Type, ...] = (list, np.ndarray, pd.Series, tuple)

PD_FUTURE_STACK = (
    dict(future_stack=True)
    if version.parse(pd.__version__) > version.parse("2.1.0")
    else dict(dropna=False)
)

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
    complete_xcats: bool = False,
    complete_cids: bool = False,
    new_xcat="NEW",
    new_cid="GLB",
):
    """
    Weighted linear combinations of cross sections or categories

    Parameters
    ----------
    df : ~pandas.DataFrame
        standardized JPMaQS DataFrame with the necessary columns: 'cid', 'xcat',
        'real_date' and 'value'.
    xcats : Union[str, List[str]
        One or more categories to be combined. If a single category is given the linear
        combination is calculated across cross-sections. This results in a single series
        to which a new cross-sectional identifier is assigned. If more than one category
        string is given the output will be a new category, i.e. a panel that is a linear
        combination of the categories specified.
    cids : List[str]
        cross-sections for which the linear combinations are calculated. Default is all
        cross-section available.
    weights : Union[List[float], str]
        This specifies how categories or cross sections are combined. There are three
        principal options. The first (default) is None, in which case equal weights are
        given to all categories or cross sections that are available. The second case is a
        set of fixed coefficients, in which case these very coefficients are applied to all
        available categories of cross sections. Per default the coefficients are normalized
        so that they add up to one for each period. This can be changed with the argument
        `normalize_weights`. The third case is the assignment of a weighting category. This
        only applies to combinations of cross sections. In this care the weighting category
        is multiplied for each period with the corresponding value of main category of the
        same cross section. Per default the weight category values are normalized so that
        they add up to one for each period. This can be changed with the argument
        `normalize_weights`.
    normalize_weights : bool
        If True (default) the weights are normalized to sum to 1. If False the weights
        are used as specified.
    signs : List[float]
        An array of consisting of +1s or -1s, of the same length as the number of
        categories in `xcats` to indicate whether the respective category should be added or
        subtracted from the linear combination. Not relevant when aggregating over cross-
        sections, i.e. when a single category is given in `xcats`. Default is None and all
        signs are set to +1.
    start : str
        earliest date in ISO format. Default is None and earliest date for which the
        respective category is available is used.
    end : str
        latest date in ISO format. Default is None and latest date for which the
        respective category is available is used.
    complete_xcats : bool
        If True (default) combinations are only calculated for observation dates on
        which all categories are available. If False a combination of the available
        categories is used. Not relevant when aggregating over cross-sections, i.e.
        when a single category is given in `xcats`.
    complete_cids : bool
        If True (default) combinations are only calculated for observation dates on
        which all cross-sections are available. If False a combination of the available
        cross-sections is used. Not relevant when aggregating over categories, i.e. when
        multiple categories are given in `xcats`.
    new_xcat : str
        Name of new composite category when aggregating over categories for a given
        cross-section. Default is "NEW".
    new_cid : str
        Name of new composite cross-section when aggregating over cross-sections for a 
        given category. Default is "GLB".

    Returns
    -------
    ~pandas.DataFrame
        standardized DataFrame with the composite values, with the columns:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    (
        df,
        xcats,
        cids,
        weights,
        normalize_weights,
        signs,
        start,
        end,
        blacklist,
        complete_xcats,
        complete_cids,
        new_xcat,
        new_cid,
        _xcat_agg,
        mode,
    ) = _check_args(
        df=df,
        xcats=xcats,
        cids=cids,
        weights=weights,
        normalize_weights=normalize_weights,
        signs=signs,
        start=start,
        end=end,
        blacklist=blacklist,
        complete_xcats=complete_xcats,
        complete_cids=complete_cids,
        new_xcat=new_xcat,
        new_cid=new_cid,
    )

    # update local variables

    _xcats: List[str] = xcats + ([weights] if isinstance(weights, str) else [])

    df = QuantamentalDataFrame(df)
    result_as_categorical = df.InitializedAsCategorical
    remaining_xcats: List[str]
    remaining_cids: List[str]
    # NOTE: the "remaining_*" variables will not be in the same order as the input
    # cids/xcats.
    # Do not used these for index based lookups/operations.
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

    if len(remaining_cids) < len(cids) and not _xcat_agg and complete_cids or len(remaining_cids) == 0:
        missing_cids_xcats_str = _missing_cids_xcats_str(df=df, cids=cids, xcats=xcats)
        raise ValueError(
            "Not all `cids` have complete `xcat` data required for the calculation.\n"
            f"{missing_cids_xcats_str}"
        )

    if _xcat_agg:
        df = _populate_missing_xcat_series(df)

        result_df: QuantamentalDataFrame =  linear_composite_xcat_agg(
            df=df,
            xcats=xcats,
            weights=weights,
            signs=signs,
            normalize_weights=normalize_weights,
            complete_xcats=complete_xcats,
            new_xcat=new_xcat,
        )

    else:  # mode == "cid_agg" -- single xcat
        df, cids, _xcat, weights, signs = _check_df_for_missing_cid_data(
            df=df, cids=cids, weights=weights, signs=signs
        )

        result_df: QuantamentalDataFrame =  linear_composite_cid_agg(
            df=df,
            xcat=_xcat,
            cids=cids,
            weights=weights,
            signs=signs,
            normalize_weights=normalize_weights,
            complete_cids=complete_cids,
            new_cid=new_cid,
        )
    
    return QuantamentalDataFrame(result_df, categorical=result_as_categorical)

def _missing_cids_xcats_str(
    df: QuantamentalDataFrame,
    cids: List[str],
    xcats: List[str],
) -> str:
    output_strs: List[str] = []

    found_cids = df["cid"].unique().tolist()
    found_xcats = df["xcat"].unique().tolist()

    if set(cids) != set(found_cids):
        missing_cids = list(set(cids) - set(found_cids))
    else:
        missing_cids = []

    if set(xcats) != set(found_xcats):
        missing_xcats = list(set(xcats) - set(found_xcats))
    else:
        missing_xcats = []

    xcat_dict: Dict[str, str] = {}
    for xc in sorted(xcats):
        miss_cids = list(
            set(cids) - set(df.loc[df["xcat"] == xc, "cid"].unique().tolist())
        )
        if miss_cids:
            xcat_dict[xc] = miss_cids

    if missing_cids:
        output_strs.append(f"Missing cids: {missing_cids}")
    if missing_xcats:
        output_strs.append(f"Missing xcats: {missing_xcats}")

    if xcat_dict:
        output_strs.append(
            "The following `cids` are missing for the respective `xcats`:"
        )
        longest_xc = max([len(xc) for xc in xcat_dict.keys()])
        for _xc, _cids in xcat_dict.items():
            msg = f"{_xc}: " + " " * (longest_xc - len(_xc)) + " " + str(sorted(_cids))
            output_strs.append(msg)

    return "\n".join(output_strs)


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
    df: QuantamentalDataFrame,
    xcat: str,
    cids: List[str],
    weights: Union[str, List[float]],
    signs: List[float],
    normalize_weights: bool = True,
    complete_cids: bool = True,
    new_cid="GLB",
):
    """Linear composite of various cids for a given category across all periods."""
    if isinstance(weights, str):
        weights_df: pd.DataFrame = df[(df["xcat"] == weights)]
        weights_df = weights_df.set_index(["real_date", "cid"])["value"].unstack(
            level=1
        )
        weights_df = weights_df[cids].mul(signs, axis=1)

    else:
        weights_series: pd.Series = pd.Series(
            np.array(weights) * np.array(signs),
            index=cids,
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
    # so that we have the same set of dates and same set of CIDs -- thank you
    # @mikiinterfiore
    weights_df = (
        weights_df.stack(**PD_FUTURE_STACK)
        .reindex(data_df.stack(**PD_FUTURE_STACK).index)
        .unstack(1)
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

    if df.is_categorical():
        out_df = QuantamentalDataFrame.from_timeseries(
            out_df.set_index("real_date")["value"], ticker=f"{new_cid}_{xcat}"
        )
    else:
        out_df["cid"] = new_cid
        out_df["xcat"] = xcat

    return out_df


def linear_composite_xcat_agg(
    df: QuantamentalDataFrame,
    xcats: List[str],
    weights: List[float],
    signs: List[float],
    normalize_weights: bool = True,
    complete_xcats: bool = True,
    new_xcat="NEW",
):
    """Linear composite of various xcats across all cids and periods"""

    # Create a weights series with the xcats as index
    weights_series: pd.Series = pd.Series(
        np.array(weights) * np.array(signs), index=xcats
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
    if df.is_categorical():
        # add a new column called xcat with the new_xcat value
        out_df["xcat"] = pd.Categorical.from_codes(
            codes=[0] * len(out_df), categories=[new_xcat]
        )
        out_df = QuantamentalDataFrame(out_df)
    else:
        out_df["xcat"] = new_xcat

    return out_df


def _populate_missing_xcat_series(
    df: QuantamentalDataFrame,
) -> QuantamentalDataFrame:
    """
    Populate missing xcat series with NaNs
    """
    found_cids: List[str] = df["cid"].unique().tolist()
    found_xcats: List[str] = df["xcat"].unique().tolist()
    found_xcats_set: Set[str] = set(found_xcats)
    dt_range: pd.DatetimeIndex = pd.to_datetime(df["real_date"].unique())
    wrn_msg: str = (
        "{cidx} does not have complete xcat data for {missing_xcats}."
        " These will be filled with NaNs for the calculation."
    )

    for cidx in found_cids:
        missing_xcats = list(
            found_xcats_set - set(df.loc[df["cid"] == cidx, "xcat"].unique())
        )
        if missing_xcats:
            warnings.warn(wrn_msg.format(cidx=cidx, missing_xcats=missing_xcats))
            for xc in missing_xcats:
                if df.is_categorical():
                    df.add_nan_series(
                        ticker=f"{cidx}_{xc}",
                        start=dt_range.min(),
                        end=dt_range.max(),
                    )
                else:
                    dct = {
                        "cid": cidx,
                        "xcat": xc,
                        "real_date": dt_range,
                        "value": np.NaN,
                    }
                    df = pd.concat([df, pd.DataFrame(data=dct)])

    return df


def _check_df_for_missing_cid_data(
    df: QuantamentalDataFrame,
    cids: List[str],
    weights: Union[str, List[float]],
    signs: List[float],
) -> Tuple[
    QuantamentalDataFrame, List[str], str, Union[str, List[float], None], List[float]
]:
    """
    Check the DataFrame for missing `cid` data and drop them if necessary and return the
    DataFrame with the missing `cid` data dropped.
    """

    found_cids: List[str] = df["cid"].unique().tolist()
    found_cids = [cid for cid in cids if cid in found_cids]
    found_xcats: List[str] = df["xcat"].unique().tolist()
    found_xcats_set: Set[str] = set(found_xcats)
    wrn_msg: str = (
        "`cid` {cidx} does not have complete `xcat` data for {missing_xcats}."
        " These will be dropped from the calculation."
    )
    if isinstance(weights, str):
        if not (
            (weights in found_xcats) and (len((set(found_xcats) - {weights})) == 1)
        ):
            raise ValueError(
                f"Weight category {weights} not found in `df`."
                f" Available categories are {found_xcats}."
            )

    if set(cids) - set(found_cids) != set():
        for cid in set(cids) - set(found_cids):
            # Cids has already been removed since it uses
            warnings.warn(f"cid {cid} not found in `df`. It will be ignored.")
            signs.pop(cids.index(cid))
            if isinstance(weights, list):
                weights.pop(cids.index(cid))


    ctr = 0
    for cidx in found_cids.copy():  # copy to allow modification of `cids`
        missing_xcats = list(
            found_xcats_set - set(df.loc[df["cid"] == cidx, "xcat"].unique())
        )
        if missing_xcats:
            found_cids.pop(ctr)
            signs.pop(ctr)
            if isinstance(weights, list):
                weights.pop(ctr)
            # drop from df
            df = df.loc[df["cid"] != cidx, :]
            warnings.warn(wrn_msg.format(cidx=cidx, missing_xcats=missing_xcats))
        else:
            ctr += 1

    if len(found_cids) == 0:
        raise ValueError(
            "No `cids` have complete `xcat` data required for the calculation."
        )

    _xcat: str = list(set(found_xcats) - {weights if isinstance(weights, str) else ""})[
        0
    ]

    rcids = [c for c in cids if c in found_cids]  # to preserve order
    return QuantamentalDataFrame(df), rcids, _xcat, weights, signs


def _check_args(
    df: QuantamentalDataFrame,
    xcats: Union[str, List[str]],
    cids: Optional[List[str]] = None,
    weights: Optional[Union[List[float], str]] = None,
    normalize_weights: bool = True,
    signs: Optional[List[float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Dict[str, List[str]] = None,
    complete_xcats: bool = False,
    complete_cids: bool = False,
    new_xcat="NEW",
    new_cid="GLB",
):
    """
    Check the arguments of linear_composite()
    """

    # df check
    if (
        (not isinstance(df, QuantamentalDataFrame))
        or ("value" not in df.columns)
        or (df["value"].isna().all())
    ):
        raise TypeError("`df` must be a standardized Quantamental DataFrame.")
    # copy df to avoid side effects
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
    else:
        raise TypeError("`xcats` must be a string or list of strings.")

    # check xcats in df
    if not set(xcats).issubset(set(df["xcat"].unique().tolist())):
        if complete_xcats:
            raise ValueError("Not all `xcats` are available in `df`.")
        else:
            missing_xcats = list(set(xcats) - set(df["xcat"].unique().tolist()))
            warnings.warn(
                f"Not all `xcats` are available in `df`: {missing_xcats} "
                "The calculation will be performed with the available xcats."
            )
            signs = [signs[i] for i, xc in enumerate(xcats) if xc not in missing_xcats]
            if isinstance(weights, list):
                weights = [weights[i] for i, xc in enumerate(xcats) if xc not in missing_xcats]
            xcats = [xc for xc in xcats if xc not in missing_xcats]

    # check cids
    if cids is None:
        cids: List[str] = df["cid"].unique().tolist()
    elif isinstance(cids, str):
        cids: List[str] = [cids]
    elif isinstance(cids, listtypes):
        cids: List[str] = list(cids)
    else:
        raise TypeError("`cids` must be a string or list of strings.")

    # check cids in df
    if not set(cids).issubset(set(df["cid"].unique().tolist())):
        if complete_cids:
            raise ValueError("Not all `cids` are available in `df`.")
        else:
            missing_cids = list(set(cids) - set(df["cid"].unique().tolist()))
            warnings.warn(
                f"Not all `cids` are available in `df`: {missing_cids} "
                "The calculation will be performed with the available cids."
            )
            if signs is not None:
                signs = [signs[i] for i, cid in enumerate(cids) if cid not in missing_cids]
            if isinstance(weights, list):
                weights = [weights[i] for i, cid in enumerate(cids) if cid not in missing_cids]
            cids = [cid for cid in cids if cid not in missing_cids]

    _xcat_agg: bool = len(xcats) > 1 or new_xcat != "NEW"
    mode: str = "xcat_agg" if _xcat_agg else "cid_agg"

    if _xcat_agg and isinstance(weights, str):
        raise ValueError(
            "When aggregating over xcats, `weights` "
            "must be a list of floats or integers."
        )

    # check weights
    expc_weights_len: int = len(xcats) if _xcat_agg else len(cids)

    if weights is None:
        weights: List[float] = list(np.ones(expc_weights_len) / expc_weights_len)
    elif isinstance(weights, listtypes):
        weights: List[float] = list(weights)
        if not all([isinstance(x, (float, int)) for x in weights]):
            raise TypeError("`weights` must be a list of floats or integers.")
        if len(weights) != expc_weights_len:
            raise ValueError(
                "`weights` must be a list of floats of the same length as `xcats`."
            )
        if any([x == 0.0 for x in weights]):
            raise ValueError("`weights` must not contain any 0s.")

    elif isinstance(weights, str):
        if weights not in df["xcat"].unique().tolist():
            raise ValueError(
                "When using a category-string as `weights`"
                " it must be present in `df`."
            )
    else:
        raise TypeError("`weights` must be a list of floats, a string or None.")

    # check signs
    if signs is None:
        signs: List[float] = [1.0] * (len(xcats) if _xcat_agg else len(cids))
    elif isinstance(signs, listtypes):
        signs: List[float] = list(signs)
        if len(signs) != expc_weights_len:
            raise ValueError(
                "`signs` must be a list of floats of the same length as `xcats`."
            )
        if not all([x in [-1.0, 1.0] for x in signs]):
            if any([x == 0.0 for x in signs]):
                raise ValueError("`signs` must not contain any 0s.")
            warnings.warn(
                "`signs` must be a list of +1s or -1s. "
                "`signs` will be coerced to +1s/-1s. "
                "(i.e. signs = abs(signs) / signs)"
            )

            signs: List[float] = [abs(x) / x for x in signs]

    else:
        raise TypeError("`signs` must be a list of floats/ints or None.")

    if not isinstance(normalize_weights, bool):
        raise TypeError("`normalize_weights` must be a boolean.")

    if not isinstance(complete_xcats, bool):
        raise TypeError("`complete_xcats` must be a boolean.")

    if not isinstance(complete_cids, bool):
        raise TypeError("`complete_cids` must be a boolean.")

    if not isinstance(new_xcat, str):
        raise TypeError("`new_xcat` must be a string.")

    if not isinstance(new_cid, str):
        raise TypeError("`new_cid` must be a string.")

    if blacklist is not None:
        if not isinstance(blacklist, dict):
            raise TypeError("`blacklist` must be a dictionary.")

    return (
        df,
        xcats,
        cids,
        weights,
        normalize_weights,
        signs,
        start,
        end,
        blacklist,
        complete_xcats,
        complete_cids,
        new_xcat,
        new_cid,
        _xcat_agg,
        mode,
    )


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
    df = QuantamentalDataFrame(df)
    lc_xcat = linear_composite(
        df=df,
        cids=["GBP", "AUD", "CAD"],
        xcats=["XR"],
        weights=[1, 2, 1],
        signs=[1, -1, 1],
        complete_xcats=True,
    )