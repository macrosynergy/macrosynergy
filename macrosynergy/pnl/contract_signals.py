"""
Module for calculating contract signals based on cross-section-specific signals, and
hedging them with a basket of contracts. Main function is `contract_signals`.
"""

import pandas as pd
import warnings
from numbers import Number
from typing import List, Union, Tuple, Optional, Set, Any, Dict

from macrosynergy.management.types import NoneType, QuantamentalDataFrame
from macrosynergy.panel import make_relative_value
from macrosynergy.management.utils import (
    is_valid_iso_date,
    update_df,
    reduce_df,
    estimate_release_frequency,
)
import logging

logger = logging.getLogger(__name__)


def _check_scaling_args(
    ctypes: List[str],
    cscales: Optional[List[Union[Number, str]]] = None,
    csigns: Optional[List[int]] = None,
    hbasket: Optional[List[str]] = None,
    hscales: Optional[List[Union[Number, str]]] = None,
    hratios: Optional[str] = None,
) -> Tuple[Any, Any, Any, Any, Any]:

    ## Check cscales and csigns
    if cscales is not None:
        # check that the number of scales is the same as the number of ctypes
        if len(cscales) != len(ctypes):
            raise ValueError("`cscales` must be of the same length as `ctypes`")
        if not all([isinstance(x, (str, Number)) for x in cscales]):
            raise TypeError("`cscales` must be a List of strings or numerical values")
    else:
        cscales: List[Number] = [1.0] * len(ctypes)

    if csigns is not None:
        # check that the number of signs is the same as the number of ctypes
        if len(csigns) != len(ctypes):
            raise ValueError("`csigns` must be of the same length as `ctypes`")
        if not all([isinstance(x, int) for x in csigns]):
            raise TypeError("`csigns` must be a List of integers")
        if not all([x in [-1, 1] for x in csigns]):
            warnings.warn(
                "`csigns` must be a List of -1 or 1, coercing to -1 or 1",
                UserWarning,
            )
            csigns: List[int] = [x / abs(x) for x in csigns]
    else:
        csigns: List[int] = [1] * len(ctypes)

    ## Check hbasket and hscales
    if hbasket is not None:
        if not (bool(hratios) and bool(hscales)):
            raise ValueError(
                "`hratios` and `hscales` must be provided if `hbasket` is provided"
            )
        else:
            # check that the number of scales is the same as the number of hbasket
            if len(hscales) != len(hbasket):
                raise ValueError("`hscales` must be of the same length as `hbasket`")
            if not all([isinstance(x, (str, Number)) for x in hscales]):
                raise TypeError(
                    "`hscales` must be a List of strings or numerical values"
                )

    return cscales, csigns, hbasket, hscales, hratios


def _check_estimation_frequency(df_wide: pd.DataFrame, rebal_freq: str) -> pd.DataFrame:
    """
    Check the timeseries to see if the estimated frequency matches the actual frequency.
    Raises a warning if the estimated frequency does not match the rebalancing frequency.

    Parameters
    ----------
    df_wide : pd.DataFrame
        dataframe in wide format with the contract signals.
    est_freq : str
        the estimated frequency of the contract signals.

    Returns
    -------
    pd.DataFrame
        dataframe with the estimated frequency.
    """

    estimated_freq: pd.Series = estimate_release_frequency(df_wide=df_wide)

    # for each series in the dataframe, check if the estimated frequency matches the rebal_freq
    for _col in df_wide.columns:
        if estimated_freq[_col] is None:
            warnings.warn(
                f"Unable to estimate frequency for `{_col}`",
            )
        elif estimated_freq[_col] != rebal_freq:
            warnings.warn(
                f"Estimated frequency for `{_col}` does not match "
                f"the rebalancing frequency`{rebal_freq}`",
            )

    return df_wide


def _make_relative_value(
    df: pd.DataFrame,
    sig: str,
    *args,
    **kwargs,
) -> pd.DataFrame:
    assert isinstance(df, QuantamentalDataFrame), "`df` must be a QuantamentalDataFrame"
    return make_relative_value(
        df=df[df["xcat"] == sig], xcats=[sig], postfix="", *args, **kwargs
    )


def _gen_contract_signals(
    df_wide: pd.DataFrame,
    cids: List[str],
    sig: str,
    ctypes: List[str],
    cscales: List[Union[Number, str]],
    csigns: List[int],
) -> pd.DataFrame:
    """
    Generate contract signals from cross-section-specific signals.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe in quantamental format with the contract signals and potentially
        categories required for translation into contract signals.
    cids : List[str]
        list of cross-sections whose signals are to be translated into contract signals.
    sig : str
        the category ticker of the cross-section-specific signal that is translated into
        contract signals.
    ctypes : List[str]
        list of identifiers for the contract types that are to be traded. They typically
        correspond to the contract type acronyms that are used in JPMaQS for generic
        returns, carry and volatility, such as "FX" for FX forwards or "EQ" for equity index
        futures. N.B. Overall a contract is identified by the combination of its cross-
        section and its contract type "<cid>_<ctype>".
    cscales : List[Union[Number, str]]
        list of scaling factors for the contract signals. These can be either a list of
        floats or a list of category tickers that serve as basis of translation. The former
        are fixed across time, the latter variable.
    csigns : List[int]
        list of signs for the contract signals. These must be either 1 for long position
        or -1 for short position.

    Returns
    -------
    pd.DataFrame
        dataframe with scaling applied.
    """

    expected_contract_signals: List[str] = [f"{cx}_{sig}" for cx in cids]

    # Check that all the contract signals are in the dataframe
    if not set(expected_contract_signals).issubset(set(df_wide.columns)):
        raise ValueError(
            "Some `cids` are missing the `sig` in the provided dataframe."
            f"\nMissing: {set(expected_contract_signals) - set(df_wide.columns)}"
        )

    # Multiply each cid_ctype by the corresponding scale and sign to produce the contract signal
    new_conts: List[str] = []
    for _cid in cids:
        sig_col: str = _cid + "_" + sig
        for ix, ctx in enumerate(ctypes):
            scale_var: Union[Number, pd.Series]
            # If the scale is a string, it must be a category ticker
            # Otherwise it is a fixed numeric value
            if isinstance(cscales[ix], str):
                scale_var: pd.Series = df_wide[_cid + "_" + cscales[ix]]
            else:
                scale_var: Number = cscales[ix]

            new_cont_name: str = _cid + "_" + ctx + "_CSIG"
            df_wide[new_cont_name] = df_wide[sig_col] * csigns[ix] * scale_var
            # For convenience, keep track of the new contract signals
            new_conts.append(new_cont_name)

    # Only return the new contract signals
    df_wide = df_wide.loc[:, new_conts]

    return df_wide


def _apply_hedge_ratios(
    df_wide: pd.DataFrame,
    cids: List[str],
    sig: str,
    hbasket: List[str],
    hscales: List[Union[Number, str]],
    hratios: str,
) -> pd.DataFrame:

    # check if the CID_SIG is in the dataframe
    expc_cid_sigs: List[str] = [f"{cx}_{sig}" for cx in cids]
    expc_cid_hr: List[str] = [f"{cx}_{hratios}" for cx in cids]
    err_str: str = (
        "Some `cids` are missing the `{sig_type}` in the provided dataframe."
        "\nMissing: {missing_items}"
    )
    for sig_type, expc_sigs in zip(["sig", "hratio"], [expc_cid_sigs, expc_cid_hr]):
        if not set(expc_sigs).issubset(set(df_wide.columns)):
            raise ValueError(
                err_str.format(
                    sig_type=sig_type,
                    missing_items=set(expc_sigs) - set(df_wide.columns),
                )
            )
    hedged_assets_list: List[str] = []
    for hb_ix, _hb in enumerate(hbasket):
        basket_pos: str = _hb + "_CSIG"
        df_wide[basket_pos] = 0.0  # initialise the basket position to zero
        for _cid in cids:
            # HBASKETx_CSIG = CIDx_SIG * CIDx_HRATIO * HBASKETx_SCALE
            # e.g.:
            # USD_EQ_CSIG = AUD_SIG * AUD_HRATIO * USD_EQ_HSCALE

            cid_sig: str = _cid + "_" + sig
            cid_hr: str = _cid + "_" + hratios

            hb_hratio: Union[Number, pd.Series]
            # If the scale is a string, it must be a category ticker
            # Otherwise it is a fixed numeric value
            if isinstance(hscales[hb_ix], str):
                hb_hratio: pd.Series = df_wide[_cid + "_" + hscales[hb_ix]]
            else:
                hb_hratio: Number = hscales[hb_ix]

            # Add the basket position to the exisitng basket_pos column in the df
            _posx: pd.Series = df_wide[cid_sig] * df_wide[cid_hr] * hb_hratio
            # set nans to zero, to avoid nans in the final result
            _posx = _posx.fillna(0.0)
            df_wide[basket_pos] += _posx

            hedged_assets_list.append(basket_pos)

    # List(Set(hedged_assets_list)) to remove duplicates
    hedged_assets_list: List[str] = list(set(hedged_assets_list))

    df_wide = df_wide.loc[:, hedged_assets_list]

    return df_wide


def _add_hedged_signals(
    df_wide_cs: pd.DataFrame,
    df_wide_hs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df_wide_hs is None:
        return df_wide_cs

    for _col in set(df_wide_hs.columns):
        if _col not in df_wide_cs.columns:
            df_wide_cs[_col] = 0.0
        df_wide_cs[_col] += df_wide_hs[_col]

    return df_wide_cs


def contract_signals(
    df: pd.DataFrame,
    sig: str,
    cids: List[str],
    ctypes: List[str],
    cscales: Optional[List[Union[Number, str]]] = None,
    csigns: Optional[List[int]] = None,
    hbasket: Optional[List[str]] = None,
    hscales: Optional[List[Union[Number, str]]] = None,
    hratios: Optional[str] = None,
    relative_value: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    rebal_freq: str = "M",
    blacklist: Optional[dict] = None,
    sname: str = "STRAT",
    *args,
    **kwargs,
) -> QuantamentalDataFrame:
    """
    Calculate contract-specific signals based on cross section-specific signals.
    Each contract signal a product of the cross-section-specific signal, a scaling factor
    or category, and a sign (+-1). Optionally, the contract signals can be hedged with a
    basket of contracts. The hedge ratios are determined by a cross-section-specific
    category. When hedging, the hedged signals are added to the contract signals; thus
    yielding a single contract signal per financial contract.

    Parameters
    ----------
    df : pd.DataFrame
        standardized JPMaQS DataFrame with the necessary columns: 'cid', 'xcat',
        'real_date' and 'value'. This dataframe must contain the cross-section-specific
        signals and possibly
            [1] categories for changing scale factors of the main contracts,
            [2] contracts of a hedging basket, and
            [3] cross-section specific hedge ratios
    sig : str
        the cross section-specific signal that serves as the basis of contract signals.
    cids : List[str]
        list of cross-sections whose signal is to be used.
    ctypes : List[str]
        list of identifiers for the contract types that are to be traded. They typically
        correspond to the contract type acronyms that are used in JPMaQS for generic
        returns, carry and volatility, such as "FX" for FX forwards or "EQ" for equity index
        futures. If n contracts are traded per cross sections `ctypes` must contain n
        arguments. N.B. Overall a contract is identified by the combination of its cross-
        section and its contract type "<cid>_<ctype>".
    cscales : List[str|float]
        list of scaling factors for the contract signals. These can be either a list of
        floats or a list of category tickers that serve as basis of translation. The former
        are fixed across time, the latter variable. The list `cscales` must be of the same
        length as the list `ctypes`.
    csigns : List[int]
        list of signs that determine the direction of the contract signals. Contract
        signal is sign x cross section signal. The signs must be either 1 or -1. The list
        `csigns` must be of the same length as `ctypes` and `cscales`.
    hbasket : List[str]
        list of contract identifiers in the format "<cid>_<ctype>" that serve as
        constituents of a hedging basket, if one is used.    
    hscales : List[str|float]
        list of scaling factors (weights) for the basket. These can be either a list of
        floats or a list of category tickers that serve as basis of translation. The former
        are fixed across time, the latter variable.
    hratios : str
        category name for cross-section-specific hedge ratios. The values of this
        category determine direction and size of the hedge basket per unit of the cross
        section-specific signal.
    relative_value : bool
        If False (default), no relative value is calculated. If True boolean, relative
        value is calculated for all cids in the strategy. # TODO split above
        `relative_value` argument into two: `relative_value` and `relative_value_cids`?
    start : str
        earliest date in ISO format. Default is None and earliest date in df is used.
    end : str
        latest date in ISO format. Default is None and latest date in df is used.
    blacklist : dict
        cross-sections with date ranges that should be excluded from the calculation of
        contract signals.
    sname : str
        name of the strategy. Default is "STRAT".

    Returns
    -------
    pd.DataFrame
        with the contract signals for all traded contracts and the specified strategy.
        It has the standard JPMaQS DataFrame. The contract signals have the following format
        "<cid>_<ctype>_<sname>_CSIG".
    """

    ## basic type and value checks
    for varx, namex, typex in [
        (sig, "sig", str),
        (cids, "cids", list),
        (ctypes, "ctypes", list),
        (cscales, "cscales", (list, NoneType)),
        (csigns, "csigns", (list, NoneType)),
        (hbasket, "hbasket", (list, NoneType)),
        (hscales, "hscales", (list, NoneType)),
        (hratios, "hratios", (str, NoneType)),
        (relative_value, "relative_value", bool),
        (start, "start", (str, NoneType)),
        (end, "end", (str, NoneType)),
        (blacklist, "blacklist", (dict, NoneType)),
        (sname, "sname", str),
    ]:
        if not isinstance(varx, typex):
            raise TypeError(f"`{namex}` must be <{typex}> not <{type(varx)}>")

        if typex in [list, str, dict] and len(varx) == 0:
            raise ValueError(f"`{namex}` must not be an empty {str(typex)}")

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("`df` must be a standardised quantamental dataframe")

    ## Standardise and copy the dataframe
    df: pd.DataFrame = QuantamentalDataFrame(df)
    _initialized_as_categorical: bool = df.InitializedAsCategorical

    ## Check the dates
    if start is None:
        start: str = pd.Timestamp(df["real_date"].min()).strftime("%Y-%m-%d")
    if end is None:
        end: str = pd.Timestamp(df["real_date"].max()).strftime("%Y-%m-%d")

    for dx, nx in [(start, "start"), (end, "end")]:
        if not is_valid_iso_date(dx):
            raise ValueError(f"`{nx}` must be a valid ISO-8601 date string")

    ## Reduce the dataframe
    df: pd.DataFrame = reduce_df(df=df, start=start, end=end, blacklist=blacklist)

    ## Check that all cid_ctype are in the dataframe
    expected_base_signals: Set[str] = set([f"{cx}_{sig}" for cx in cids])
    found_base_signals: Set[str] = set(QuantamentalDataFrame(df).list_tickers())
    if not (expected_base_signals).issubset(found_base_signals):
        raise ValueError(
            "Some `cids` are missing the `sig` in the provided dataframe."
            f"\nMissing: {expected_base_signals - found_base_signals}"
        )

    ## Check the scaling and hedging arguments
    cscales, csigns, hbasket, hscales, hratios = _check_scaling_args(
        ctypes=ctypes,
        cscales=cscales,
        csigns=csigns,
        hbasket=hbasket,
        hscales=hscales,
        hratios=hratios,
    )

    # Actual calculation

    ## Calculate relative value if requested
    if relative_value:
        rel_signals = _make_relative_value(
            df=df,
            sig=sig,
            blacklist=blacklist,
            cids=cids,
            start=start,
            end=end,
            *args,
            **kwargs,
        )
        df = update_df(df=df, df_add=rel_signals)

    ## Cast the dataframe to wide format
    df_wide: pd.DataFrame = QuantamentalDataFrame(df=df).to_wide()

    ## Check rebal_freq or downsample the dataframe
    # df_wide: pd.DataFrame = _check_estimation_frequency(
    #     df_wide=df_wide, rebal_freq=rebal_freq
    # )

    ## Generate primary contract signals
    df_contract_signals: pd.DataFrame = _gen_contract_signals(
        df_wide=df_wide,
        cids=cids,
        sig=sig,
        ctypes=ctypes,
        cscales=cscales,
        csigns=csigns,
    )

    ## Generate hedge contract signals
    df_hedge_signals: Optional[pd.DataFrame] = None
    if hbasket is not None:
        df_hedge_signals: pd.DataFrame = _apply_hedge_ratios(
            df_wide=df_wide,
            cids=cids,
            sig=sig,
            hbasket=hbasket,
            hscales=hscales,
            hratios=hratios,
        )

    # Add the hedge signals to the contract signals
    df_out: pd.DataFrame = _add_hedged_signals(
        df_wide_cs=df_contract_signals,
        df_wide_hs=df_hedge_signals,
    )

    ## Wide to quantamental
    df_out: pd.DataFrame = QuantamentalDataFrame.from_wide(df=df_out)

    # Append the strategy name to all the xcats
    df_out = df_out.rename_xcats(postfix=f"_{sname}")

    assert isinstance(df_out, QuantamentalDataFrame)
    return QuantamentalDataFrame(
        df_out, _initialized_as_categorical=_initialized_as_categorical
    ).to_original_dtypes()


def multi_signal_contract_signals(
    df: QuantamentalDataFrame,
    contract_dicts: Dict[str, Dict[str, Any]],
    common_args: Dict[str, Any],
):
    """
    Calculate contract signals for multiple signal specifications. Each signal
    specification is a dictionary with arguments for a call to `contract_signals()`. The
    common arguments are passed to all calls. The dataframe is passed to all calls, and
    should not be included in the dictionaries, or common arguments. Please see the
    documentation for `contract_signals()` for more information on the arguments. Note:
    when two (or more) of the arguments produce a series with the same name, each new
    series will overwrite the previous one.

    Parameters
    ----------
    df : pd.DataFrame
        standardized JPMaQS DataFrame with the necessary columns: 'cid', 'xcat',
        'real_date' and 'value'.
    contract_dicts : dict[str, dict[str, Any]]
        dictionary of dictionaries with arguments for each call to `contract_signals()`.
        The keys are the names of the signal specifications.
    common_args : dict[str, Any]
        dictionary with common arguments for all calls to `contract_signals()`. These
        are passed to all calls, and should not be included in the dictionaries in
        `contract_dicts`.

    Returns
    -------
    QuantamentalDataFrame
        dataframe with the contract signals for all traded contracts and the specified
        strategies. It has the standard JPMaQS DataFrame. The contract signals have the
        following format "<cid>_<ctype>_<sname>_CSIG".
    """

    # Check that the types are correct
    types_dict: dict = {
        "sig": str,
        "cids": list,
        "ctypes": list,
        "cscales": (list, NoneType),
        "csigns": (list, NoneType),
        "hbasket": (list, NoneType),
        "hscales": (list, NoneType),
        "hratios": (str, NoneType),
        "relative_value": bool,
        "start": (str, NoneType),
        "end": (str, NoneType),
        "blacklist": (dict, NoneType),
        "sname": str,
    }

    df = QuantamentalDataFrame(df)
    _initialized_as_categorical: bool = df.InitializedAsCategorical

    type_error_msg: str = "{} must be <{}> not <{}>"
    value_error_msg: str = "{} must not be an empty {}"
    for k, v in common_args.items():
        if not isinstance(v, types_dict[k]):
            raise TypeError(type_error_msg.format(k, types_dict[k], type(v)))
        if types_dict[k] in [list, str, dict] and len(v) == 0:
            raise ValueError(value_error_msg.format(k, str(types_dict[k])))

    type_error_msg = type_error_msg + " ; in arguments passed for {}"
    value_error_msg = value_error_msg + " ; in arguments passed for {}"

    for dict_name, dict_args in contract_dicts.items():
        for k, v in dict_args.items():
            if not isinstance(v, types_dict[k]):
                raise TypeError(
                    type_error_msg.format(k, types_dict[k], type(v), dict_name)
                )
            if types_dict[k] in [list, str, dict] and len(v) == 0:
                raise ValueError(
                    value_error_msg.format(k, str(types_dict[k]), dict_name)
                )

    # Calculate the contract signals for each signal specification
    # Keep updating the dataframe with the new signals to allow signals to be used to
    # generate other signals
    # the awkward syntax **{**a, **b} is used to update left dict values with right dict values
    # to avoid conflicts in the keys of the two dictionaries

    for dict_args in contract_dicts.values():
        # TODO: use update_df?
        df = QuantamentalDataFrame.from_qdf_list(
            [df, contract_signals(df=df, **{**common_args, **dict_args})]
        )

    return QuantamentalDataFrame(
        df, _initialized_as_categorical=_initialized_as_categorical
    ).to_original_dtypes()


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_test_df

    cids: List[str] = ["USD", "EUR", "GBP", "AUD", "CAD"]
    xcats: List[str] = ["SIG", "HR"]

    start: str = "2000-01-01"
    end: str = "2020-12-31"

    df: pd.DataFrame = make_test_df(
        cids=cids,
        xcats=xcats,
        start=start,
        end=end,
    )

    df.loc[(df["cid"] == "USD") & (df["xcat"] == "SIG"), "value"] = 1.0
    ctypes = ["FX", "IRS", "CDS"]
    cscales = [1.0, 0.5, 0.1]
    csigns = [1, -1, 1]

    hbasket = ["USD_EQ", "EUR_EQ"]
    hscales = [0.7, 0.3]

    df_cs: pd.DataFrame = contract_signals(
        df=df,
        sig="SIG",
        cids=cids,
        ctypes=ctypes,
        cscales=cscales,
        csigns=csigns,
        hbasket=hbasket,
        hscales=hscales,
        hratios="HR",
    )
    df_cs["xcat"].unique()
