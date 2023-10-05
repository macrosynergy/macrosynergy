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

from macrosynergy.pnl import Numeric
from macrosynergy.management.utils import is_valid_iso_date, standardise_dataframe
from macrosynergy.management.shape_dfs import reduce_df

def _apply_cscales(
    df: pd.DataFrame,
    ctypes: List[str],
    cscales: List[Union[Numeric, str]],
    csigns: List[int],
) -> pd.DataFrame:
    """
    Apply the contract scales to the dataframe.

    :param <pd.DataFrame> df: QDF with the contract signals and scaling XCATs.
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
            all([isinstance(x, (str, Numeric)) for x in cscales]),
            isinstance(csigns, list),
            all([isinstance(x, int) for x in csigns]),
            len(ctypes) == len(cscales) == len(csigns),
        ]
    ), "Invalid arguments passed to `_apply_cscales()`"

    # Arg checks
    _ctypes: List[str] = df[df["xcat"].isin(ctypes)]["xcat"].unique().tolist()
    if not set(_ctypes).issubset(set(ctypes)):
        raise ValueError(
            "Some `ctypes` are missing the `cscales` in the provided dataframe."
            f"\nMissing: {set(_ctypes) - set(ctypes)}"
        )

    # DF with scales

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
    _sigs: List[str] = [f"{cx}_{sig}" for cx in cids]
    _found_sigs: List[str] = df.columns.tolist()
    if not set(_sigs).issubset(set(_found_sigs)):
        raise ValueError(
            "Some `cids` are missing the `sig` in the provided dataframe."
            f"\nMissing: {set(_sigs) - set(_found_sigs)}"
        )

    # DF with signals for the cids
    sigs_df: pd.DataFrame = df[df["xcat"] == sig].pivot(
        index="real_date", columns="cid", values="value"
    )

    # Set index to real_date to allow for easy multiplication
    df = df.set_index("real_date")

    # Multiply the signals by the cross-section-specific signals
    for _cid in cids:
        fxcats: List[str] = df[df["cid"] == _cid]["xcat"].unique().tolist()
        fxcats: List[str] = list(set(fxcats) - set([sig]))
        for _fxcat in fxcats:
            sel_bools: pd.Series = (df["cid"] == _cid) & (df["xcat"] == _fxcat)
            df.loc[sel_bools, "value"] = (
                df.loc[sel_bools, "value"] * sigs_df.loc[:, _cid]
            )

    return df.reset_index()


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
        (cscales, "cscales", (list, type(None))),
        (csigns, "csigns", (list, type(None))),
        (hbasket, "hbasket", (list, type(None))),
        (hscales, "hscales", (list, type(None))),
        (hratios, "hratios", (list, type(None))),
        (start, "start", (str, type(None))),
        (end, "end", (str, type(None))),
        (blacklist, "blacklist", (dict, type(None))),
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
    if hbasket is not None:
        if not set(hbasket).issubset(
            set(
                df[["cid", "xcat"]]
                .drop_duplicates()
                .apply(lambda x: f"{x['cid']}_{x['xcat']}", axis=1)
                .unique()
                .tolist()
            )
        ):
            e_msg: str = f"Some of the contracts/tickers in `hbasket` are not in `df`"
            e_msg += f"\nMissing contracts: {set(hbasket) - set(df['cid'].unique())}"
            raise ValueError(e_msg)

    ## If hedging scales are not specified, set them to 1.0
    if hscales is None and hbasket is not None:
        hscales: List[Union[Numeric, str]] = [1.0] * len(hbasket)

    ## If hratios are specified, check that they are in the dataframe for the cids
    if hratios is not None and hbasket is not None:
        if len(hratios) != len(cids):
            raise ValueError("`hratios` and `cids` must be of the same length")

        _hratios: List[str] = [f"{cx}_{hr}" for cx in cids for hr in hratios]
        _found_hratios: List[str] = (
            (df["cid"] + "_" + df["xcat"]).drop_duplicates().tolist()
        )
        if not set(_hratios).issubset(set(_found_hratios)):
            raise ValueError(
                "Some `cids` are missing the `hratios` in the provided dataframe."
                f"\nMissing: {set(_hratios) - set(_found_hratios)}"
            )

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

            is_strs: bool = all([isinstance(x, str) for x in _sc])
            is_nums: bool = all([isinstance(x, Numeric) for x in _sc])
            if not (is_strs or is_nums):
                raise ValueError(
                    f"`{_scn}` must be a list of strings or floats/numeric types"
                )

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

    # TODO: Apply contract scales

    # TODO: Apply hscales if specified

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
