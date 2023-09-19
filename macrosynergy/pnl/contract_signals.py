"""
Module for calculating contract signals based on cross-section-specific signals,
and hedging them with a basket of contracts. Main function is `contract_signals`.

::docs::contract_signals::sort_first::
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from typing import List, Union, Tuple, Optional, Set, Dict

from macrosynergy.pnl import Numeric
from macrosynergy.management.utils import is_valid_iso_date, standardise_dataframe
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df





def _apply_cscales(
    df: pd.DataFrame,
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

    ## If the scales are floats, apply them directly
    if all([isinstance(x, Numeric) for x in cscales]):
        cscales: List[float] = [x * y for x, y in zip(cscales, csigns)]
        _cs: Dict[str, float] = dict(zip(ctypes, cscales))
        df["value"] = df.apply(
            lambda x: x["value"] * _cs[_short_xcat(xcat=x["xcat"])],
            axis=1,
        )
    ## If the scales are category tickers, apply them by matching the dates
    elif all([isinstance(x, str) for x in cscales]):
        out_dfs: List[pd.DataFrame] = []
        for _xcat, _xscale, _xsign in zip(ctypes, cscales, csigns):
            xcat_df: pd.DataFrame = df[_short_xcat(xcat=df["xcat"]) == _xcat].copy()
            cids_list: List[str] = xcat_df["cid"].unique().tolist()
            for cidx in cids_list:
                data_df: pd.DataFrame = xcat_df[xcat_df["cid"] == cidx].set_index(
                    "real_date"
                )
                w_df: pd.DataFrame = df[
                    (_short_xcat(xcat=df["xcat"]) == _xscale) & (df["cid"] == cidx)
                ].set_index("real_date")

                # match by date, and multiply the values
                data_df["value"] = data_df["value"] * w_df["value"] * _xsign

                # append to the output dataframes
                out_dfs.append(data_df.reset_index())

        # concatenate the output dataframes
        df: pd.DataFrame = pd.concat(out_dfs, axis=0, ignore_index=True)
        out_dfs = None
    else:
        raise ValueError("`cscales` must be a list of floats or category tickers")

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
        df.loc[df["cid"] == _cid, "value"] = _ddf["value"].values

    return df


def contract_signals(
    df: pd.DataFrame,
    sig: str,
    cids: List[str],
    ctypes: List[str],
    cscales: Optional[List[Union[Numeric, str]]] = None,
    csigns: Optional[List[int]] = None,
    hbasket: Optional[List[str]] = None,
    hscales: Optional[List[Union[Numeric, str]]] = None,
    hratio: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    sname: str = "STRAT",
) -> pd.DataFrame:
    """
    Caclulate contract specific signals based on cross-section-specific signals.

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
        that are used in JPMaQS for  generic returns, carry and volatility, such as
        "FX" for FX forwards or "EQ" for equity index futures.
        N.B. Overall a contract is identified by the combination of its cross-section
        and its contract type "<cid>_<ctype>".
    :param <List[str|float]> cscales: list of scaling factors for the contract signals.
        These can be eigher a list of floats or a list of category tickers that serve
        as basis of translation. The former are fixed across time, the latter variable.
    :param <List[float]> csigns: list of signs for the contract signals. These must be
        either 1 for long position or -1 for short position.
    :param <List[str]> hbasket: list of contract identifiers in the format "<cid>_<ctype>"
        that serve as constituents of the hedging basket.
    param <List[str|float]> hscales: list of scaling factors (weights) for the basket.
        These can be eigher a list of floats or a list of category tickers that serve
        as basis of translation. The former are fixed across time, the latter variable.
    :param <str> hratio: category names for cross-section-specific hedge ratios.
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
        # (sig, "sig", str),
        (cids, "cids", list),
        (ctypes, "ctypes", list),
        (cscales, "cscales", (list, type(None))),
        (csigns, "csigns", (list, type(None))),
        (hbasket, "hbasket", (list, type(None))),
        (hscales, "hscales", (list, type(None))),
        (hratio, "hratio", (str, type(None))),
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

    ## Add all contracts to hedging basket if none specified
    if hbasket is None:
        hbasket: List[str] = (
            df[["cid", "xcat"]]
            .drop_duplicates()
            .apply(lambda x: f"{x['cid']}_{x['xcat']}", axis=1)
            .unique()
            .tolist()
        )

    ## If hedging scales are not specified, set them to 1.0
    if hscales is None:
        hscales: List[Union[Numeric, str]] = [1.0] * len(hbasket)

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
        if _sc is not None:
            if len(_sc) != len(_sct):
                raise ValueError(f"`{_scn}` and `{_sctn}` must be of the same length")

            is_strs: bool = all([isinstance(x, str) for x in _sc])
            is_nums: bool = all([isinstance(x, Numeric) for x in _sc])
            if not (is_strs or is_nums):
                raise ValueError(
                    f"`{_scn}` must be a list of strings or floats/numeric types"
                )

    ## Check that all the contract types are in the dataframe as xcats
    _s_ctypes: List[str] = [_short_xcat(xcat=xc) for xc in df["xcat"].unique().tolist()]
    if not set(ctypes).issubset(set(_s_ctypes)):
        e_msg: str = f"Some of the contract types in `ctypes` are not in `df`"
        e_msg += f"\nMissing contract types: {set(ctypes) - set(_s_ctypes)}"
        raise ValueError(e_msg)

    ## Check that all the cross-sections are in the dataframe as cids
    if not set(cids).issubset(set(df["cid"].unique())):
        e_msg: str = f"Some of the cross-sections in `cids` are not in `df`"
        e_msg += f"\nMissing cross-sections: {set(cids) - set(df['cid'].unique())}"
        raise ValueError(e_msg)

    ## Check that all the contracts in the hedging basket are in the dataframe as tickers
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

    ## Calculate the contract signals

    # in order, multiple the contract signals by the contract scales
    df: pd.DataFrame = _apply_cscales(
        df=df, ctypes=ctypes, cscales=cscales, csigns=csigns
    )

    # now apply the hscales
    df: pd.DataFrame = _apply_hscales(df=df, hbasket=hbasket, hscales=hscales)

    df["scat"] = df.apply(lambda x: _short_xcat(xcat=x["xcat"]), axis=1)

    # group by scat and sum the values
    df: pd.DataFrame = df.groupby(["real_date", "scat"])["value"].sum().reset_index()
    # drop the xcat column and rename the scat column to xcat
    df: pd.DataFrame = df.drop(columns=["xcat"]).rename(columns={"scat": "xcat"})

    # add the strategy name to the xcat
    df["xcat"] = df.apply(lambda x: f"{x['xcat']}_{sname}", axis=1)

    # apply hedge ratio if specified
    if hratio is not None:
        df: pd.DataFrame = apply_hedge_ratio(df=df, hratios=[hratio], cids=cids)

    return df
