"""
Module for calculating contract signals based on cross-section-specific signals,
and hedging them with a basket of contracts. Main function is `contract_signals`.

::docs::contract_signals::sort_first::
"""

import numpy as np
import pandas as pd
import warnings

from typing import List, Union, Tuple, Optional, Set, Dict

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


def _apply_cscales(
    df: pd.DataFrame,
    cids: List[str],
    ctypes: List[str],
    cscales: List[Union[Numeric, str]],
    csigns: List[int],
) -> pd.DataFrame:
    """
    Apply the contract scales to the dataframe.

    :param <pd.DataFrame> df: QDF with the contract signals and scaling XCATs.
    :param <List[str]> cids: list of cross-sections whose signal is to be used.
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

    :return <pd.DataFrame>: dataframe with scaling applied.
    """
    # Type checks
    assert all(
        [
            isinstance(df, pd.DataFrame),
            isinstance(ctypes, list),
            all([isinstance(x, str) for x in ctypes]),
            isinstance(cscales, list),
            (
                all([isinstance(x, str) for x in cscales])
                ^ all([isinstance(x, Numeric) for x in cscales])
            ),  # must be Numeric XOR str, not both or neither
            isinstance(csigns, list),
            all([isinstance(x, int) for x in csigns]),
            len(ctypes) == len(cscales) == len(csigns),
        ]
    ), "Invalid arguments passed to `_apply_cscales()`"

    # TODO: ?
    # assert _arg_validation(
    #     raise_errors=True, df=df, ctypes=ctypes, cscales=cscales, csigns=csigns
    # )

    # Arg checks
    _ctypes: List[str] = df[df["xcat"].isin(ctypes)]["xcat"].unique().tolist()
    if not set(_ctypes).issubset(set(ctypes)):
        raise ValueError(
            "Some `ctypes` are missing the `cscales` in the provided dataframe."
            f"\nMissing: {set(_ctypes) - set(ctypes)}"
        )

    # Convert the dataframe to ticker format
    dfW: pd.DataFrame = qdf_to_ticker_df(df=df)

    # Multiply each cid_ctype by the corresponding scale and sign
    for _cid in cids:
        for ix, ctx in enumerate(ctypes):
            ctype_col: str = _cid + "_" + ctx
            scale_var: Union[Numeric, pd.Series]
            # If the scale is a string, it must be a category ticker
            # Otherwise it is a fixed numeric value
            if isinstance(cscales[ix], str):
                scale_var: pd.Series = dfW[_cid + "_" + cscales[ix]]
            else:
                scale_var: Numeric = cscales[ix]

            # get the of tickers that start with ctype_col
            ctype_cols: List[str] = [x for x in dfW.columns if x.startswith(ctype_col)]
            for _col in ctype_cols:
                dfW[_col] = dfW[_col] * csigns[ix] * scale_var

    return ticker_df_to_qdf(df=dfW)


def _apply_hscales(
    df: pd.DataFrame,
    hbasket: List[str],
    hscales: List[Union[Numeric, str]],
    hratios: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply hedging logic to the basket of contracts.

    :param <pd.DataFrame> df: QDF with the contract signals and scaling XCATs.
    :param <List[str]> hbasket: list of contract identifiers in the format "<cid>_<ctype>"
        that serve as constituents of the hedging basket.
    param <List[str|float]> hscales: list of scaling factors (weights) for the basket.
        These can be either a list of floats or a list of category tickers that serve
        as basis of translation. The former are fixed across time, the latter variable.
    :param <str> hratios: category names for cross-section-specific hedge ratios.
    :return <pd.DataFrame>: dataframe with the contracts in the same currency/units.
    """
    # Type checks
    assert all(
        [
            isinstance(df, pd.DataFrame),
            isinstance(hbasket, list),
            all([isinstance(x, str) for x in hbasket]),
            isinstance(hscales, list),
            (
                all([isinstance(x, Numeric) for x in hscales])
                ^ all([isinstance(x, str) for x in hscales])
            ),  # must be Numeric XOR str, not both or neither
            isinstance(hratios, (str, NoneType)),
            # hbasket and hscales must be of the same length
            len(hbasket) == len(hscales),
        ]
    ), "Invalid arguments passed to `_apply_hscales()`"

    # Pivot the DF to ticker format
    dfW: pd.DataFrame = qdf_to_ticker_df(df=df)

    # Check that all tickers in hbasket are in the dataframe
    if not set(hbasket).issubset(set(dfW.columns)):
        raise ValueError(
            "Some `hbasket` are missing in the provided dataframe."
            f"\nMissing: {set(hbasket) - set(dfW.columns)}"
        )

    if hratios is not None:
        expected_hratios: List[str] = [f"{cx}_{hratios}" for cx in get_cid(hbasket)]
        if not set(expected_hratios).issubset(set(dfW.columns)):
            raise ValueError(
                "Some `hratios` are missing in the provided dataframe."
                f"\nMissing: {set(expected_hratios) - set(dfW.columns)}"
            )

    # Calculations

    for ix, tickerx in enumerate(hbasket):
        scale_var: Union[Numeric, pd.Series]
        # If string uses the series from DF, else uses the numeric value
        if isinstance(hscales[ix], str):
            scale_var: pd.Series = dfW[tickerx + "_" + hscales[ix]]
        else:
            scale_var: Numeric = hscales[ix]

        # If hratios is not None, use the series from DF, else use 1.0
        ratio_var: Union[Numeric, pd.Series]
        if hratios is not None:
            ratio_var: pd.Series = dfW[tickerx + "_" + hratios[ix]]
        else:
            ratio_var: int = 1

        dfW[tickerx] = dfW[tickerx] * scale_var * ratio_var


def _apply_sig_conversion(
    df: pd.DataFrame,
    sig: str,
    cids: List[str],
) -> pd.DataFrame:
    """
    Get the wide dataframe with all assets reported in the same currency/units.

    :param <pd.DataFrame> df: QDF with the contract signals.
    :param <str> sig: the cross-section-specific signal that serves as the basis of
        contract signals.
    :param <List[str]> cids: list of cross-sections whose signal is to be used.
    :return <pd.DataFrame>: dataframe with the contracts in the same currency/units.
    """
    # Type checks
    assert all(
        [
            isinstance(df, pd.DataFrame),
            isinstance(sig, str),
            isinstance(cids, list),
            all([isinstance(x, str) for x in cids]),
        ]
    ), "Invalid arguments passed to `_apply_sig_conversion()`"

    # Arg checks
    expected_sigs: List[str] = [f"{cx}_{sig}" for cx in cids]

    if not set(expected_sigs).issubset(set(df["cid"] + "_" + df["xcat"])):
        raise ValueError(
            "Some `cids` are missing the `sig` in the provided dataframe."
            f"\nMissing: {set(expected_sigs) - set(df.columns)}"
        )

    # Pivot the DF to ticker format
    dfW: pd.DataFrame = qdf_to_ticker_df(df=df)

    # Multiply each ticker by the corresponding scale
    for ix, tickerx in enumerate(expected_sigs):
        if get_xcat(tickerx) != sig:
            sig_col: str = get_cid(tickerx) + "_" + sig
            dfW[tickerx] = dfW[tickerx] * dfW[sig_col]

    return ticker_df_to_qdf(df=dfW)


def _calculate_contract_signals(
    df: pd.DataFrame,
    cids: List[str],
    ctypes: List[str],
    sname: str = "STRAT",
) -> pd.DataFrame:
    # Type checks
    assert all(
        [
            isinstance(df, pd.DataFrame),
            isinstance(cids, list),
            all([isinstance(x, str) for x in cids]),
            isinstance(ctypes, list),
            all([isinstance(x, str) for x in ctypes]),
        ]
    ), "Invalid arguments passed to `_calculate_contract_signals()`"

    # Create a new DF with cols for each contract type, and dates from `df` as index

    cid_ctypes: List[str] = [f"{cx}_{ctx}" for cx in cids for ctx in ctypes]
    out_df: pd.DataFrame = pd.DataFrame(index=df["real_date"].unique().tolist())
    for _ctx in cid_ctypes:
        out_df[_ctx] = 0.0

    # Pivot the DF to ticker format
    dfW: pd.DataFrame = qdf_to_ticker_df(df=df)

    # Sum up the contract signals for each contract type
    for _cont in cid_ctypes:
        sel_cols: List[str] = [x for x in dfW.columns if x.startswith(_cont)]
        out_df[_cont] = dfW[sel_cols].sum(axis=1)

    # rename all columns to include the strategy name
    out_df.columns = [f"{cx}_{sname}_CSIG" for cx in out_df.columns]

    return ticker_df_to_qdf(df=out_df)


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
        This dataframe must contain the cross-section-specific signals and possibly
        categories for variable scale factors of the main contracts and the contracts
        in the hedging basket, as well as cross-section specific hedge ratios
    :param <str> sig: the cross-section-specific signal that serves as the basis of
        contract signals.
    :param <List[str]> cids: list of cross-sections whose signal is to be used.
    :param <List[str]> ctypes: list of identifiers for the contract types that are
        to be traded. They typically correspond to the contract type acronyms
        that are used in JPMaQS for generic returns, carry and volatility, such as
        "FX" for FX forwards or "EQ" for equity index futures.
        N.B. Overall a contract is identified by the combination of its cross-section
        and its contract type "<cid>_<ctype>".
    :param <List[str|float]> cscales: list of scaling factors for the contract signals.
        These can be either a list of floats or a list of category tickers that serve
        as basis of translation. The former are fixed across time, the latter variable.
    :param <List[float]> csigns: list of signs for the contract signals. These must be
        either 1 for long position or -1 for short position.
    :param <List[str]> hbasket: list of contract identifiers in the format "<cid>_<ctype>"
        that serve as constituents of the hedging basket.
    param <List[str|float]> hscales: list of scaling factors (weights) for the basket.
        These can be either a list of floats or a list of category tickers that serve
        as basis of translation. The former are fixed across time, the latter variable.
    :param <str> hratios: category names for cross-section-specific hedge ratios.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded
        from the dataframe.
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
        (hratios, "hratios", (list, NoneType)),
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

    ## Check that all the contracts in the hedging basket are in the dataframe as tickers
    _found_tickers: List[str] = (df["cid"] + "_" + df["xcat"]).unique().tolist()
    if hbasket is not None:
        if not set(hbasket).issubset(set(_found_tickers)):
            e_msg: str = (
                f"Some of the contracts/tickers in `hbasket` are not in `df`."
                f"\nMissing contracts: {set(hbasket) - set(df['cid'].unique())}"
            )
            raise ValueError(e_msg)
        if hscales is None:
            hscales: List[Numeric] = [1] * len(hbasket)
        else:
            if len(hbasket) != len(hscales):
                raise ValueError("`hbasket` and `hscales` must be of the same length.")

        if hratios is not None:
            if not set([f"{cx}_{hratios}" for cx in get_cid(hbasket)]).issubset(
                set(_found_tickers)
            ):
                e_msg: str = (
                    f"Some of the contracts/tickers in `hratios` are not in `df`."
                    f"\nMissing contracts: {set(hratios) - set(df['cid'].unique())}"
                )
                raise ValueError(e_msg)

    ## If contract scales are not specified, set them to 1.0
    if cscales is None:
        cscales: List[Union[Numeric, str]] = [1.0] * len(ctypes)

    ## If contract signs are not specified, set them to 1.0
    ## if specified, check that they are either 1 or -1
    if csigns is None:
        csigns: List[int] = [1] * len(ctypes)
    else:
        if not all(isinstance(x, Numeric) for x in csigns):
            raise TypeError("`csigns` must be a list of integers")

        if not all(isinstance(x, int) for x in csigns):
            warnings.warn("`csigns` is being coerced to integers")
            csigns: List[int] = [int(x) for x in csigns]
            if not all(x in [-1, 1] for x in csigns):
                warnings.warn("`csigns` is being coerced to [-1, 1]")
                csigns: List[int] = [int(x / abs(x)) for x in csigns]

    ## Check that the scales are of the same length as the contract types
    ## Also assert that they are either all strings or all floats
    for _scn, _sc, _sctn, _sct in [
        ("cscales", cscales, "ctypes", ctypes),
        ("hscales", hscales, "hbasket", hbasket),
    ]:
        if _sc is not None or _sct is not None:
            if len(_sc) != len(_sct):
                raise ValueError(f"`{_scn}` and `{_sctn}` must be of the same length")

            if not all([isinstance(x, (str, Numeric)) for x in _sc]):
                raise TypeError(f"`{_scn}` must be a list of strings or floats")

    ## Check that all the contract types are in the dataframe as xcats
    for cidx in cids:
        for _ctx in ctypes:
            f_xcats: List[str] = (
                df[df["cid"] == cidx]["xcat"].drop_duplicates().tolist()
            )
            if not any([_ctx in x for x in f_xcats]):
                raise ValueError(f"`{cidx}_{_ctx}` is not a contract type in `df`")

    ## Check that all the cross-sections are in the dataframe as cids
    if not set(cids).issubset(set(df["cid"].unique())):
        e_msg: str = f"Some of the cross-sections in `cids` are not in `df`"
        e_msg += f"\nMissing cross-sections: {set(cids) - set(df['cid'].unique())}"
        raise ValueError(e_msg)

    ## Apply the cross-section-specific signal to the dataframe
    df: pd.DataFrame = _apply_sig_conversion(df=df, sig=sig, cids=cids)

    ## Apply contract scales
    df: pd.DataFrame = _apply_cscales(
        df=df,
        cids=cids,
        ctypes=ctypes,
        cscales=cscales,
        csigns=csigns,
    )

    ## Apply hedging logic
    if hbasket is not None:
        df: pd.DataFrame = _apply_hscales(
            df=df,
            hbasket=hbasket,
            hscales=hscales,
            hratios=hratios,
        )

    ## Calculate the contract signals

    df: pd.DataFrame = _calculate_contract_signals(
        df=df, cids=cids, ctypes=ctypes, sname=sname
    )

    return df


if __name__ == "__main__":
    from macrosynergy.management.simulate_quantamental_data import make_test_df

    cids: List[str] = ["USD", "EUR", "GBP", "AUD", "CAD"]
    xcats: List[str] = ["FXXR_NSA", "EQXR_NSA", "IRXR_NSA", "CDS_NSA", "TOUSD"]

    start: str = "2000-01-01"
    end: str = "2020-12-31"

    df: pd.DataFrame = make_test_df(
        cids=cids,
        xcats=xcats,
        start=start,
        end=end,
    )

    df.loc[(df["cid"] == "USD") & (df["xcat"] == "TOUSD"), "value"] = 1.0

    rDF: pd.DataFrame = contract_signals(
        df=df,
        sig="TOUSD",
        cids=cids,
        ctypes=["FXXR", "EQXR", "IRXR", "CDS"],
    )
