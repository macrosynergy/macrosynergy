"""
Module for calculating contract signals based on cross-section-specific signals,
and hedging them with a basket of contracts. Main function is `contract_signals`.

::docs::contract_signals::sort_first::
"""

import numpy as np
import pandas as pd
import warnings

from typing import List, Union, Tuple, Optional, Set

import os, sys

sys.path.append(os.getcwd())

from macrosynergy.pnl import Numeric, NoneType
from macrosynergy.management.utils import (
    is_valid_iso_date,
    standardise_dataframe,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
    get_cid,
    get_xcat,
)
from macrosynergy.management.shape_dfs import reduce_df


def _check_arg_types(
    df: Optional[pd.DataFrame] = None,
    sig: Optional[str] = None,
    cids: Optional[List[str]] = None,
    ctypes: Optional[List[str]] = None,
    cscales: Optional[List[Union[Numeric, str]]] = None,
    csigns: Optional[List[int]] = None,
    hbasket: Optional[List[str]] = None,
    hscales: Optional[List[Union[Numeric, str]]] = None,
    hratios: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    sname: Optional[str] = None,
) -> bool:
    """
    Auxiliary function to check the types of the arguments passed to the main function.
    Accepts all arguments from `contract_signals()` as optional keyword arguments, and
    checks that they are of the correct type.

    :params: see `contract_signals()`
    :return <bool>: True if all arguments are of the correct type, False otherwise.
    """
    correct_types: bool = all(
        [
            isinstance(df, (pd.DataFrame, NoneType)),
            isinstance(sig, (str, NoneType)),
            isinstance(cids, (list, NoneType)),
            isinstance(ctypes, (list, NoneType)),
            isinstance(cscales, (list, NoneType)),
            isinstance(csigns, (list, NoneType)),
            isinstance(hbasket, (list, NoneType)),
            isinstance(hscales, (list, NoneType)),
            isinstance(hratios, (str, NoneType)),
            isinstance(start, (str, NoneType)),
            isinstance(end, (str, NoneType)),
            isinstance(blacklist, (dict, NoneType)),
            isinstance(sname, (str, NoneType)),
        ]
    )

    non_empty_iterables: bool = all(
        [
            df is None or len(df) > 0,
            cids is None or len(cids) > 0,
            ctypes is None or len(ctypes) > 0,
            cscales is None or len(cscales) > 0,
            csigns is None or len(csigns) > 0,
            hbasket is None or len(hbasket) > 0,
            hscales is None or len(hscales) > 0,
        ]
    )

    correct_nested_types: bool = (
        correct_types
        and non_empty_iterables
        and all(
            [
                cids is None or all([isinstance(x, str) for x in cids]),
                ctypes is None or all([isinstance(x, str) for x in ctypes]),
                cscales is None
                or all([isinstance(x, (str, Numeric)) for x in cscales]),
                csigns is None or all([isinstance(x, int) for x in csigns]),
                hbasket is None or all([isinstance(x, str) for x in hbasket]),
                hscales is None
                or all([isinstance(x, (str, Numeric)) for x in hscales]),
            ]
        )
    )

    return correct_nested_types


def _gen_contract_signals(
    df: pd.DataFrame,
    cids: List[str],
    sig: str,
    ctypes: List[str],
    cscales: List[Union[Numeric, str]],
    csigns: List[int],
    sname: str,
) -> pd.DataFrame:
    """
    Generate contract signals from cross-section-specific signals.

    :param <pd.DataFrame> df: dataframe in quantamental format with the contract signals
        and potentially categories required for translation into contract signals.
    :param <List[str]> cids: list of cross-sections whose signals are to be translated
        into contract signals.
    :param <str> sig: the category ticker of the cross-section-specific signal that
        is translated into contract signals.
    :param <List[str]> ctypes: list of identifiers for the contract types that are
        to be traded. They typically correspond to the contract type acronyms
        that are used in JPMaQS for generic returns, carry and volatility, such as
        "FX" for FX forwards or "EQ" for equity index futures.
        N.B. Overall a contract is identified by the combination of its cross-section
        and its contract type "<cid>_<ctype>".
    :param <List[Union[Numeric, str]]> cscales: list of scaling factors for the
        contract signals. These can be either a list of floats or a list of category
        tickers that serve as basis of translation. The former are fixed across time,
        the latter variable.
    :param <List[int]> csigns: list of signs for the contract signals. These must be
        either 1 for long position or -1 for short position.
    :param <str> sname: name of the strategy. Default is "STRAT".

    :return <pd.DataFrame>: dataframe with scaling applied.
    """
    # Type checks
    if not _check_arg_types(
        df=df,
        cids=cids,
        sig=sig,
        ctypes=ctypes,
        cscales=cscales,
        csigns=csigns,
        sname=sname,
    ):
        raise TypeError("Invalid arguments passed to `_gen_contract_tickers()`")

    expected_contract_signals: List[str] = [f"{cx}_{sig}" for cx in cids]

    # Pivot from quantamental to wide format
    df_wide: pd.DataFrame = qdf_to_ticker_df(df=df)

    # Check that all the contract signals are in the dataframe
    if not set(expected_contract_signals).issubset(set(df_wide.columns)):
        raise ValueError(
            "Some `cids` are missing the `sig` in the provided dataframe."
            f"\nMissing: {set(expected_contract_signals) - set(df_wide.columns)}"
        )

    # Multiply each cid_ctype by the corresponding scale and sign to produce the contract signal
    new_conts: List[str] = []
    for _cid in cids:
        for ix, ctx in enumerate(ctypes):
            sig_col: str = _cid + "_" + sig
            scale_var: Union[Numeric, pd.Series]
            # If the scale is a string, it must be a category ticker
            # Otherwise it is a fixed numeric value
            if isinstance(cscales[ix], str):
                scale_var: pd.Series = df_wide[_cid + "_" + cscales[ix]]
            else:
                scale_var: Numeric = cscales[ix]

            new_cont_name: str = _cid + "_" + ctx + "_CSIG" + "_" + sname
            df_wide[new_cont_name] = df_wide[sig_col] * csigns[ix] * scale_var
            new_conts.append(new_cont_name)

    # Only return the new contract signals
    df_wide = df_wide[new_conts]

    return ticker_df_to_qdf(df=df_wide)


def _apply_hedge_ratios(
    df: pd.DataFrame,
    hbasket: List[str],
    hscales: List[Union[Numeric, str]],
    hratios: Optional[str] = None,
    sname: str = "STRAT",
) -> pd.DataFrame:
    # Type checks
    if not _check_arg_types(
        df=df,
        hbasket=hbasket,
        hscales=hscales,
        hratios=hratios,
        sname=sname,
    ):
        raise TypeError("Invalid arguments passed to `apply_hedge_ratios()`")

    # Pivot the DF to wide ticker format
    df_wide: pd.DataFrame = qdf_to_ticker_df(df=df)
    # check if the hedge basket is in the dataframe
    if not set(hbasket).issubset(set(df_wide.columns)):
        raise ValueError(
            "Some `hbasket` are missing in the provided dataframe."
            f"\nMissing: {set(hbasket) - set(df_wide.columns)}"
        )

    expected_hratios: List[str] = [f"{cx}_{hratios}" for cx in get_cid(hbasket)]
    found_contract_types = lambda x: any([ct.startswith(x) for ct in df_wide.columns])
    if not all([found_contract_types(x) for x in expected_hratios]):
        missing_hratios: List[str] = [
            x for x in expected_hratios if not found_contract_types(x)
        ]

        warnings.warn(
            "Some `hbasket` are missing the `hratios` in the provided dataframe."
            f"\nMissing: {missing_hratios}",
            UserWarning,
        )

        df_wide[missing_hratios] = pd.NA

    # multiply each ticker by the corresponding scale and ratio
    for ix, contractx in enumerate(hbasket):
        scale_var: Union[Numeric, pd.Series]
        # If the scale is a string, it must be a category ticker, else
        # the scale is a fixed numeric value
        if isinstance(hscales[ix], str):
            scale_var: pd.Series = df_wide[contractx + "_" + hscales[ix]]
        else:
            scale_var: Numeric = hscales[ix]

        ratio_str: pd.Series = contractx + "_" + hratios[ix]

        df_wide[contractx] = df_wide[contractx] * df_wide[ratio_str] * scale_var

    # Only return the hedge basket
    df_wide = df_wide[hbasket]

    return ticker_df_to_qdf(df=df_wide)


def contract_signals(
    df: pd.DataFrame,
    sig: str,
    cids: List[str],
    ctypes: List[str],
    cscales: Optional[List[Union[Numeric, str]]] = None,
    csigns: Optional[List[int]] = None,
    hbasket: Optional[List[str]] = None,
    hscales: Optional[List[Union[Numeric, str]]] = None,
    hratios: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    sname: str = "STRAT",
) -> pd.DataFrame:
    """
    Calculate contract specific signals based on cross-section-specific signals.

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain
        [1] the cross-section-specific signals and possibly
        [2] categories for variable scale factors of the main contracts,
        [3] contracts of a hedging basket, and
        [4] cross-section specific hedge ratios
    :param <str> sig: the cross-section-specific signal that serves as the basis of
        contract signals.
    :param <List[str]> cids: list of cross-sections whose signal is to be used.
    :param <List[str]> ctypes: list of identifiers for the contract types that are
        to be traded. They typically correspond to the contract type acronyms
        that are used in JPMaQS for generic returns, carry and volatility, such as
        "FX" for FX forwards or "EQ" for equity index futures.
        If n contracts are traded per cross sections `ctypes` must contain n arguments.
        N.B. Overall a contract is identified by the combination of its cross-section
        and its contract type "<cid>_<ctype>".
    :param <List[str|float]> cscales: list of scaling factors for the contract signals.
        These can be either a list of floats or a list of category tickers that serve
        as basis of translation. The former are fixed across time, the latter variable.
        The list `cscales` must be of the same length as `ctypes`.
    :param <List[float]> csigns: list of signs for the contract signals. These must be
        either 1 for long position or -1 for short position.
        The list `csigns` must be of the same length as `ctypes` and `cscales`.
    :param <List[str]> hbasket: list of contract identifiers in the format "<cid>_<ctype>"
        that serve as constituents of the hedging basket.
    param <List[str|float]> hscales: list of scaling factors (weights) for the basket.
        These can be either a list of floats or a list of category tickers that serve
        as basis of translation. The former are fixed across time, the latter variable.
    :param <str> hratios: category name for cross-section-specific hedge ratios.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded
        from the calculation of contract signals.
    :param <str> sname: name of the strategy. Default is "STRAT".

    :return <pd.DataFrame>: with the contract signals for all traded contracts and the
        specified strategy. It has the standard JPMaQS DataFrame. The contract signals
        have the following format "<cid>_<ctype>_<sname>_CSIG".
    """
    ## basic type and value checks
    for varx, namex, typex in [
        (df, "df", pd.DataFrame),
        (sig, "sig", str),
        (cids, "cids", list),
        (ctypes, "ctypes", list),
        (cscales, "cscales", (list, NoneType)),
        (csigns, "csigns", (list, NoneType)),
        (hbasket, "hbasket", (list, NoneType)),
        (hscales, "hscales", (list, NoneType)),
        (hratios, "hratios", (str, NoneType)),
        (start, "start", (str, NoneType)),
        (end, "end", (str, NoneType)),
        (blacklist, "blacklist", (dict, NoneType)),
        (sname, "sname", str),
    ]:
        if not isinstance(varx, typex):
            raise TypeError(f"`{namex}` must be <{typex}> not <{type(varx)}>")

        if typex in [list, str, dict] and len(varx) == 0:
            raise ValueError(f"`{namex}` must not be an empty {str(typex)}")

    ## Standardise and copy the dataframe
    df: pd.DataFrame = standardise_dataframe(df.copy())

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
    expected_base_signals: List[str] = [f"{cx}_{sig}" for cx in cids]
    found_base_signals: Set[str] = set(df["cid"] + "_" + df["xcat"])
    if not set(expected_base_signals).issubset(found_base_signals):
        raise ValueError(
            "Some `cids` are missing the `sig` in the provided dataframe."
            f"\nMissing: {set(expected_base_signals) - found_base_signals}"
        )

    ## Check cscales and csigns
    if cscales is not None:
        # check that the number of scales is the same as the number of ctypes
        if len(cscales) != len(ctypes):
            raise ValueError("`cscales` must be of the same length as `ctypes`")
        if not all([isinstance(x, (str, Numeric)) for x in cscales]):
            raise TypeError("`cscales` must be a List of strings or numerical values")
    else:
        cscales: List[Numeric] = [1.0] * len(ctypes)

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
            if not all([isinstance(x, (str, Numeric)) for x in hscales]):
                raise TypeError(
                    "`hscales` must be a List of strings or numerical values"
                )

    ## Apply the cross-section-specific signal to the dataframe
    # df: pd.DataFrame = _apply_sig_conversion(df=df, sig=sig, cids=cids)

    ## Generate contract signals
    df_contract_signals: pd.DataFrame = _gen_contract_signals(
        df=df,
        cids=cids,
        sig=sig,
        ctypes=ctypes,
        cscales=cscales,
        csigns=csigns,
        sname=sname,
    )

    ## Generate hedge contract signals

    df_hedge_signals: Optional[pd.DataFrame] = None
    if hbasket is not None:
        df_hedge_signals: pd.DataFrame = _apply_hedge_ratios(
            df=df,
            hbasket=hbasket,
            hscales=hscales,
            hratios=hratios,
        )

    ## Consolidate contract signals and hedge signals
    df_out: pd.DataFrame = _consolidate_contract_signals(
        df_contract_signals=df_contract_signals,
        df_hedge_signals=df_hedge_signals,
    )

    return df_out


if __name__ == "__main__":
    from macrosynergy.management.simulate_quantamental_data import make_test_df

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
    ctypes = ["FX", "EQ", "IRS", "CDS"]
    cscales = [1.0, 2.0, 0.5, 0.1]
    csigns = [1, -1, 1, 1]

    hbasket = ["USD_EQ", "EUR_EQ"]
    hscales = [0.7, 0.3]

    rDF: pd.DataFrame = contract_signals(
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
