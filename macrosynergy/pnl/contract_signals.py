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

from macrosynergy.pnl import Numeric, _short_xcat
from macrosynergy.management.utils import is_valid_iso_date, standardise_dataframe
from macrosynergy.management.shape_dfs import reduce_df


def _apply_contract_scales_to_signal(
    df_signals: pd.DataFrame,
    df_scales: Union[pd.Series, pd.DataFrame],
    df_signs: Union[pd.Series, pd.DataFrame],
) -> pd.DataFrame:
    """Apply the contract scales to the signal.

    Match the contract types with their scales and apply the scales and signs to
    the dataframe.

    :param <pd.DataFrame> df: dataframe with the contract signals.
    :param <List[str]> ctypes: list of contract types.
    :param <List[Union[Numeric, str]]> cscales: list of scales for the contract types.
        These can be either floats or category tickers.
    :param <List[int]> csigns: list of signs for the contract types. These must be
        either 1 for long position or -1 for short position.
    :return: dataframe with the contract signals scaled and signed.
    """
    return df_signs * df_scales * df_signals


def _apply_cscales(
    df_signals: pd.DataFrame,
    df_wide: pd.DataFrame,
    ctypes: List[str],
    cscales: List[Union[Numeric, str]],
    csigns: List[int],
):
    """
    Match the contract types with their scales and apply the scales and signs to
    the dataframe.

    :param <pd.DataFrame> df: dataframe with the contract signals.
    :param <List[str]> ctypes: list of contract types.
    :param <List[Union[Numeric, str]]> cscales: list of scales for the contract types.
        These can be either floats or category tickers.
    :param <List[int]> csigns: list of signs for the contract types. These must be
        either 1 for long position or -1 for short position.
    """


    assert len(ctypes) == len(
        cscales
    ), "`ctypes` and `cscales` must be of the same length"
    assert len(ctypes) == len(
        csigns
    ), "`ctypes` and `csigns` must be of the same length"

    cids_found: List[str] = [x.split("_")[0] for x in df_wide.columns.tolist()]
    short_xcats: List[str] = _short_xcat(ticker=df_wide.columns.tolist())

    # If the scales are floats, apply them directly
    if all([isinstance(x, Numeric) for x in cscales]):
        df_cscales: pd.DataFrame = pd.Series(
            data=cscales, index=ctypes, name="cscales"
        )

    ## If the scales are category tickers, apply them by matching the dates
    elif all([isinstance(x, str) for x in cscales]):
        df_cscales: pd.DataFrame = df_wide.loc[:, cscales]
    else:
        raise ValueError("`cscales` must be a list of floats or category tickers")

    # TODO same for csigns...
    _apply_contract_scales_to_signal(
        df_signals: pd.DataFrame,
        df_scales: Union[pd.Series, pd.DataFrame],
        df_signs: Union[pd.Series, pd.DataFrame],
    )
        for _cidx in cids_found:
            sel_cids_bools: pd.Series = [
                x.startswith(_cidx + "_") for x in df_wide.columns
            ]
            for _xcat, _xscale, _xsign in zip(ctypes, cscales, csigns):
                sel_xcats_bools: pd.Series = [
                    _short_xcat(xcat=x) == _xcat for x in df_wide.columns.tolist()
                ]
                # and the two together
                sel_bools: pd.Series = sel_cids_bools & sel_xcats_bools

                _scale_series: pd.Series = (
                    df_wide.loc[:, sel_bools].set_index("real_date") * _xscale * _xsign
                )
                # multiply scale series by every column from df[selected_bools] with _scale_series

                df_wide.loc[:, sel_bools] = df_wide.loc[:, sel_bools].apply(
                    lambda x: x * _scale_series
                )

    return df


def _apply_hscales(
    df: pd.DataFrame,
    hbasket: List[str],
    hscales: List[Union[Numeric, str]],
):
    """
    Match the hedging basket with its scales and apply the scales to the dataframe.

    :param <pd.DataFrame> df: dataframe with the contract signals.
    :param <List[str]> hbasket: list of contracts in the hedging basket.
    :param <List[Union[Numeric, str]]> hscales: list of scales for the hedging basket.
        These can be either floats or category tickers.
    """
    assert len(hbasket) == len(
        hscales
    ), "`hbasket` and `hscales` must be of the same length"

    og_cols: List[str] = df.columns.tolist()
    # add a column called tickers
    df["ticker"] = df.apply(lambda x: f"{x['cid']}_{x['xcat']}", axis=1)

    ## If the scales are floats, apply them directly
    if all([isinstance(x, Numeric) for x in hscales]):
        df["value"] = df.apply(
            lambda x: x["value"] * hscales[hbasket.index(x["ticker"])], axis=1
        )
    ## If the scales are category tickers, apply them by matching the dates
    elif all([isinstance(x, str) for x in hscales]):
        out_dfs: List[pd.DataFrame] = []
        for _tickerx, _hscale in zip(hbasket, hscales):
            ticker_df: pd.DataFrame = df[df["ticker"] == _tickerx].copy()

            data_df: pd.DataFrame = ticker_df.set_index("real_date")
            w_df: pd.DataFrame = df[
                (df["xcat"] == _hscale) & (df["ticker"] == _tickerx)
            ].set_index("real_date")

            # match by date, and multiply the values
            data_df["value"] = data_df["value"] * w_df["value"]

            # append to the output dataframes
            out_dfs.append(data_df.reset_index())

        # concatenate the output dataframes
        df: pd.DataFrame = pd.concat(out_dfs, axis=0, ignore_index=True)
        out_dfs = None

    else:
        raise ValueError("`hscales` must be a list of floats or category tickers")

    # drop the ticker column
    df: pd.DataFrame = df.drop(columns=["ticker"])
    assert df.columns.tolist() == og_cols, "Columns have been changed"
    return df


def apply_hedge_ratio(
    df: pd.DataFrame,
    hratios: List[str],
    cids: List[str],
) -> pd.DataFrame:
    """
    Applies the hedge ratio to the dataframe.

    :param <pd.DataFrame> df: dataframe with the contract signals.
    :param <List[str]> hratios: list of hedge ratios.
    :param <List[str]> cids: list of cross-sections.
    """
    assert len(hratios) == len(cids), "`hratios` and `cids` must be of the same length"

    for _cid, _hratio in zip(cids, hratios):
        _ddf: pd.DataFrame = df[df["cid"] == _cid].set_index("real_date")
        _rdf: pd.DataFrame = _ddf[_ddf["xcat"] == _hratio].set_index("real_date")

        # match by date, and multiply the values
        for _xc in _ddf["xcat"].unique().tolist():
            _ddf.loc[_ddf["xcat"] == _xc, "value"] = (
                _ddf.loc[_ddf["xcat"] == _xc, "value"] * _rdf["value"]
            )

        # repopulate the dataframe
        for _xc in _ddf["xcat"].unique().tolist():
            df.loc[df["cid"] == _cid, "value"] = _ddf["value"].values

    return df


def _signal_to_contract(
    df_wide: pd.DataFrame,
    sig: str,
    cids: List[str],
    ctypes: List[str],
) -> pd.DataFrame:
    ## Check that all the CID_SIG pairs are in the dataframe
    _sigs: List[str] = [f"{cx}_{sig}" for cx in cids]
    _found_sigs: List[str] = df_wide.columns.tolist()
    assert set(_sigs).issubset(set(_found_sigs)), (
        "Some `cids` are missing the `sig` in the provided dataframe."
        f"\nMissing: {set(_sigs) - set(_found_sigs)}"
    )

    # check that all cid_ctypes are in the dataframe
    _cids_ctypes: List[str] = [f"{cx}_{ctx}" for cx in cids for ctx in ctypes]
    found_cid_ctypes: List[str] = [
        f"{_cid}_{_ctx}"
        for _cid in cids
        for _ctx in _short_xcat(ticker=df_wide.columns.tolist())
    ]
    assert set(_cids_ctypes).issubset(set(found_cid_ctypes)), (
        "Some contracts are missing the `ctypes` in the provided dataframe."
        f"\nMissing: {set(_cids_ctypes) - set(_found_sigs)}"
    )

    ## DF with signals for the cids
    # sigs_df: pd.DataFrame = df_wide[df_wide.columns.intersection(_sigs)]

    for _cidx in cids:
        sig_series: pd.Series = df_wide[_cidx + "_" + sig]
        col_selection: pd.Series = df_wide.columns.str.startswith(_cidx + "_")
        df_wide.loc[:, col_selection] = df_wide.loc[:, col_selection].apply(
            lambda x: x * sig_series
        )

    return df_wide


def contract_signals(
    df: pd.DataFrame,
    sig: str,  # string
    cids: List[str],  # cross sections
    ctypes: List[str],  # ctypes = ["FX", "EQ"]
    cscales: Optional[List[Union[Numeric, str]]] = None,
    csigns: Optional[List[int]] = None,
    hbasket: Optional[List[str]] = None,
    hscales: Optional[List[Union[Numeric, str]]] = None,
    hratios: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    sname: str = "STRAT",
) -> pd.DataFrame:
    """
    Calculate contract specific signals based on cross-section-specific signals.

    # TODO rewrite:
    inputs: `df` pivoted data-frame of (raw) signals with contracts to be traded on the columns and real dates as the index.


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
    :param <List[str]> hratios: category names for cross-section-specific hedge ratios.
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

    # Extracting matrix of signals, single signal per cross section
    df_signals: pd.DataFrame = df.loc[df.cid.isin(cids) & (df.xkat == sig), :].pivot(
        index="real_date", columns="cid", values="value"
    )

    if len(ctypes) > 1:
        df_signals = pd.concat(
            (
                df_signals.rename(columns={col: f"{col}_{cc}" for col in df_signals.columns})
                for cc for ctypes
            ),
            axis=1,
            ignore_index=False
        )

    # Extract matrix of returns, could be multiple asset types per cross section, apply same signal.
    df_returns = pd.concat(
        (
            df.loc[
                df.cid.isin(cids) & df.xkat.str.startswith(cc),
                :
            ].pivot(
                index="real_date", columns="cid", values="value"
            ).rename(columns={cc: f"{cid}_{cc}" for cid in cids})
            for cc in ctypes
        ),
        axis=1,
        ignore_index=False
    )

    # df_scales
    # df_signs

    # Hedging basket...

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
    _s_ctypes: List[str] = _short_xcat(xcat=df["xcat"].unique().tolist())
    if not set(ctypes).issubset(set(_s_ctypes)):
        e_msg: str = f"Some of the contract types in `ctypes` are not in `df`"
        e_msg += f"\nMissing contract types: {set(ctypes) - set(_s_ctypes)}"
        raise ValueError(e_msg)

    ## Check that all the cross-sections are in the dataframe as cids
    if not set(cids).issubset(set(df["cid"].unique())):
        e_msg: str = f"Some of the cross-sections in `cids` are not in `df`"
        e_msg += f"\nMissing cross-sections: {set(cids) - set(df['cid'].unique())}"
        raise ValueError(e_msg)

    ## Pivot the df on ticker
    df["ticker"] = df.apply(lambda x: f"{x['cid']}_{x['xcat']}", axis=1)
    # pivot such that the columns are individual tickers and the index is the real_date
    df_wide: pd.DataFrame = df.pivot(
        index="real_date", columns="ticker", values="value"
    )

    ## Calculate the contract signals

    df: pd.DataFrame = _signal_to_contract(
        df_wide=df_wide, sig=sig, cids=cids, ctypes=ctypes
    )

    # in order, multiple the contract signals by the contract scales
    df: pd.DataFrame = _apply_cscales(
        df=df, ctypes=ctypes, cscales=cscales, csigns=csigns
    )

    # now apply the hscales
    if hbasket is not None:
        df: pd.DataFrame = _apply_hscales(df=df, hbasket=hbasket, hscales=hscales)

    # TODO need cids as well...
    df["scat"] = df.apply(lambda x: _short_xcat(xcat=x["xcat"]), axis=1)
    # TODO non-position assets...

    # group by scat and sum the values
    df: pd.DataFrame = (
        df.groupby(["real_date", "cid", "scat"])["value"].sum().reset_index()
    )
    # drop the xcat column and rename the scat column to xcat
    df: pd.DataFrame = df.drop(columns=["xcat"]).rename(columns={"scat": "xcat"})

    # add the strategy name to the xcat
    df["xcat"] = df.apply(lambda x: f"{x['xcat']}_{sname}", axis=1)

    # apply hedge ratio if specified
    if hratios is not None and hbasket is not None:
        df: pd.DataFrame = apply_hedge_ratio(df=df, hratios=hratios, cids=cids)

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
