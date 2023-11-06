"""
Module for calculating notional positions based on contract signals, assets-under-management, 
and other relevant parameters.

::docs::notional_positions::sort_first::
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple, Optional

from macrosynergy.management.utils import (
    standardise_dataframe,
    reduce_df,
    is_valid_iso_date,
    apply_slip,
    reduce_df,
    qdf_to_ticker_df,
    ticker_df_to_qdf,
)

from macrosynergy.management.types import Numeric, NoneType, QuantamentalDataFrame


def _apply_slip(
    df: pd.DataFrame,
    slip: int,
    cids: List[str],
    xcats: List[str],
    metrics: List[str] = ["value"],
) -> pd.DataFrame:
    """
    Applies a slip using the function `apply_slip()` to a dataframe with contract
    signals and returns.

    :param <pd.DataFrame> df: Quantamental dataframe with contract signals and returns.
    :param <int> slip: the number of days to wait before applying the signal.
    :param <List[str]> cids: list of contract identifiers.
    :param <List[str]> xcats: list of contract categories.
    :param <List[str]> metrics: list of metrics to apply the slip to.
    """
    if slip == 0:
        return df
    else:
        return apply_slip(
            df=df,
            slip=slip,
            cids=cids,
            xcats=xcats,
            metrics=metrics,
            raise_error=False,
        )


def _get_csigs_for_contract(
    df: pd.DataFrame,
    contid: str,
    sname: str,
) -> List[str]:
    """
    Returns the contract signals for a given contract identifier and strategy name.

    :param <pd.DataFrame> df: Quantamental dataframe with contract signals and returns.
    :param <str> contid: the contract identifier.
    :param <str> sname: the strategy name.

    :return: <List[str]> list of contract signals.
    """
    # Get all tickers in the df
    tickers: List[str] = (df["cid"] + "_" + df["xcat"]).unique().tolist()

    # Identify the contract signals for the contract identifier and strategy name
    sig_ident: str = f"{sname}_CSIG"
    find_contid = lambda x: str(x).startswith(contid) and str(x).endswith(sig_ident)

    # filter and return
    tickers: List[str] = [tx for tx in tickers if find_contid(tx)]
    return tickers


def _leverage_positions(
    df: pd.DataFrame,
    sname: str,
    contids: List[str],
    aum: Numeric = 100,
    leverage: Numeric = 1.0,
    pname: str = "POS",
) -> QuantamentalDataFrame:
    """"""
    df_wide: pd.DataFrame = qdf_to_ticker_df(df=df)
    sig_ident: str = f"{sname}_CSIG"
    find_contid = lambda x: any(
        [str(x).startswith(c) and str(x).endswith(sig_ident) for c in contids]
    )
    sel_cols: pd.Series = [find_contid(x) for x in df_wide.columns]
    df_wide: pd.DataFrame = df_wide.loc[:, sel_cols]

    for ic, contx in enumerate(contids):
        pos_col: str = contx + "_" + pname

        # Get all signals for that contract
        sig_cols: List[str] = _get_csigs_for_contract(
            df=df_wide, contid=contx, sname=sname
        )
        # sum of all assets for that contract; if zero, set to NaN to avoid div by zero
        df_wide[pos_col] = df_wide[sig_cols].sum(axis=1)  # sum(row) all signals
        df_wide.loc[df_wide[pos_col] == 0, pos_col] = np.nan
        # USD position(asset) = AUM * leverage / (sum of signals * dollar per signal)
        df_wide[pos_col] = aum * leverage / (df_wide[pos_col])
        # TODO: this should be dfw_pos = dfw_sigs * aum * leverage / rowsums(dfw_sigs)

    generated_positions: List[str] = [f"{contx}_{pname}" for contx in contids]

    df_wide = df_wide.loc[:, generated_positions]

    return ticker_df_to_qdf(df=df_wide)


def notional_positions(
    df: pd.DataFrame,
    sname: str,
    contids: List[str],
    aum: Numeric = 100,
    dollar_per_signal: Numeric = 1.0,
    leverage: Optional[Numeric] = None,
    vol_target: Optional[Numeric] = None,
    rebal_freq: str = "m",
    slip: int = 1,
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life=11,
    rstring: str = "XR",
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    pname: str = "POS",
):
    """
    Calculates contract positions based on contract signals, assets under management (AUM)
    and other specifications.

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain the contract-specific signals and possibly
        related return series (for vol-targeting).
    :param <str> sname: the name of the strategy. It must correspond to contract
        signals in the dataframe, which have the format "<cid>_<ctype>_<sname>_CSIG", and
        which are typically calculated by the function contract_signals().
    :param <List[str]> contids: list of contract identifiers in the format
        "<cid>_<ctype>". It must correspond to contract signals in the dataframe.
    :param <float> aum: the assets under management in USD million (for consistency).
        This is basis for all position sizes. Default is 100.
    :param <float> dollar_per_signal: the amount of notional (e.g. USD) per
        contract signal value. Default is 1. The default scale is arbitrary
        and is merely a basis for tryouts.
    :param <float> leverage: the ratio of the sum of notional positions to AUM.
        This is the main basis for leveraged-based positioning. Since different
        contracts have different expected volatility and correlations this method
        does not control expected volatility. Default is None, i.e. the method is not
        applied.
    :param <float> vol_target: the target volatility of the portfolio in % of AUM.
        This is the main parameter for volatility-targeted positioning. That method
        estimates the annualized standard deviation of the signal-based portfolio
        for a 1 USD per signal portfolio based on past variances and covariances of
        the contract returns. The estimation is managed by the function
        `historic_portfolio_vol()`.
        Default is None, i.e. the volatility-targeting is not applied.
    :param <str> rebal_freq: the rebalancing frequency. Default is 'm' for monthly.
        Alternatives are 'w' for business weekly, 'd' for daily, and 'q' for quarterly.
        Contract signals are taken from the end of the holding period and applied to
        positions at the beginning of the next period, subject to slippage.
    :param <int> slip: the number of days to wait before applying the signal. Default is 1.
        This means that new positions are taken at the very end of the first trading day
        of the holding period and are the basis of PnL calculation from the second
        Trading day onward.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21. This passed through to
        the function `historic_portfolio_vol()`.
    :param <str> lback_meth: the method to use for the lookback period of the
        volatility-targeting method. Default is 'ma' for moving average. Alternative is
        "xma", for exponential moving average. Again this is passed through to
        the function `historic_portfolio_vol()`.
    :param <int> half_life: the half-life of the exponential moving average for
        volatility-targeting if the exponential moving average "xma" method has been
        chosen Default is 11. This is passed through to
        the function `historic_portfolio_vol()`.
    :param <str> rstring: a general string of the return category. This identifies
        the contract returns that are required for the volatility-targeting method, based
        on the category identifier format <cid>_<ctype><rstring>_<rstring> in accordance
        with JPMaQS conventions. Default is 'XR'.
    :param <str> start: the start date of the data. Default is None, which means that
        the start date is taken from the dataframe.
    :param <str> end: the end date of the data. Default is None, which means that
        the end date is taken from the dataframe.
    :param <dict> blacklist: a dictionary of contract identifiers to exclude from
        the calculation. Default is None, which means that no contracts are excluded.
    :param <str> pname: the name of the position. Default is 'POS'.

    :return: <pd.DataFrame> with the positions for all traded contracts and the
        specified strategy in USD million. It has the standard JPMaQS DataFrame.
        The contract signals have the following format "<cid>_<ctype>_<sname>_<pname>".

    """
    for varx, namex, typex in [
        (df, "df", pd.DataFrame),
        (sname, "sname", str),
        (contids, "contids", list),
        (aum, "aum", Numeric),
        (dollar_per_signal, "dollar_per_signal", Numeric),
        (leverage, "leverage", (Numeric, NoneType)),
        (vol_target, "vol_target", (Numeric, NoneType)),
        (rebal_freq, "rebal_freq", str),
        (slip, "slip", int),
        (lback_periods, "lback_periods", int),
        (lback_meth, "lback_meth", str),
        (half_life, "half_life", int),
        (rstring, "rstring", str),
        (start, "start", (str, NoneType)),
        (end, "end", (str, NoneType)),
        (blacklist, "blacklist", (dict, NoneType)),
        (pname, "pname", str),
    ]:
        if not isinstance(varx, typex):
            raise ValueError(f"`{namex}` must be {typex}.")

        if isinstance(varx, (str, list, dict)) and len(varx) == 0:
            raise ValueError(f"`{namex}` must not be an empty {str(typex)}.")

    ## Volatility targeting and leverage cannot be applied at the same time
    if not (bool(leverage) ^ bool(vol_target)):
        e_msg: str = "Either `leverage` or `vol_target` must be specified"
        # TODO: No it needs not. You can define notional positions simply on a
        #       dollar_per_signal basis. This is useful for testing.
        e_msg += (", but not both.") if (bool(leverage) and bool(vol_target)) else (".")
        raise ValueError(e_msg)

    if not isinstance(df, QuantamentalDataFrame):
        raise ValueError("`df` must be a QuantamentalDataFrame.")

    if "value" not in df.columns:
        raise ValueError("`df` must have a `value` column.")

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

    ## Check the contract identifiers and contract signals

    df["tickers"]: str = df["cid"] + "_" + df["xcat"]

    # There must be atleast one contract signal with the strategy name
    if not any(df["tickers"].str.endswith(f"_{sname}_CSIG")):
        raise ValueError(f"No contract signals for strategy `{sname}` in dataframe.")

    # Check that all contract identifiers have at least one signal
    u_tickers: List[str] = list(df["tickers"].unique())
    for contx in contids:
        if not any(
            [tx.startswith(contx) and tx.endswith(f"_{sname}_CSIG") for tx in u_tickers]
        ):
            raise ValueError(f"Contract identifier `{contx}` not in dataframe.")

    ## Apply the slip
    df: pd.DataFrame = _apply_slip(
        df=df,
        slip=slip,
        cids=contids,
        xcats=[],
        metrics=["value"],
    )

    if leverage:
        leveraged_positions: QuantamentalDataFrame = _leverage_positions(
            df=df,
            sname=sname,
            contids=contids,
            aum=aum,
            leverage=leverage,
            pname=pname,
        )

        return leveraged_positions

    else:
        raise NotImplementedError("Volatility targeting not implemented yet.")


def historic_portfolio_vol(
    df: pd.DataFrame,
    sname: str,
    contids: List[str],
    est_freq: str = "m",
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life=11,
    rstring: str = "XR",
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
):
    """
    Estimates the annualized standard deviations of portfolio, based on historic
    variances and co-variances.

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain the contract-specific signals and return series.
    :param <str> sname: the name of the strategy. It must correspond to contract
        signals in the dataframe, which have the format "<cid>_<ctype>_<sname>_CSIG", and
        which are typically calculated by the function contract_signals().
    :param <list[str]> contids: list of contract identifiers in the format
        "<cid>_<ctype>". It must correspond to contract signals in the dataframe.
    :param <str> est_freq: the frequency of the volatility estimation. Default is 'm'
        for monthly. Alternatives are 'w' for business weekly, 'd' for daily, and 'q'
        for quarterly. Estimations are conducted for the end of the period.
    :param <float> dollar_per_signal: the amount of notional currency (e.g. USD) per
        contract signal value. Default is 1. The default scale has no specific meaning
        and is merely a basis for tryouts.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21. This passed through to
        the function `historic_portfolio_vol()`.
    :param <str> lback_meth: the method to use for the lookback period of the
        volatility-targeting method. Default is 'ma' for moving average. Alternative is
        "xma", for exponential moving average. Again this is passed through to
        the function `historic_portfolio_vol()`.
    :param <str> rstring: a general string of the return category. This identifies
        the contract returns that are required for the volatility-targeting method, based
        on the category identifier format <cid>_<ctype><rstring>_<rstring> in accordance
        with JPMaQS conventions. Default is 'XR'.
    :param <str> start: the start date of the data. Default is None, which means that
        the start date is taken from the dataframe.
    :param <str> end: the end date of the data. Default is None, which means that
        the end date is taken from the dataframe.
    :param <dict> blacklist: a dictionary of contract identifiers to exclude from
        the calculation. Default is None, which means that no contracts are excluded.


    :return: <pd.DataFrame> with the annualized standard deviations of the portfolios.
        The values are in % annualized. Values between estimation points are forward
        filled.

    N.B.: If returns in the lookback window are not available the function will replace
    them with the average of the available returns of the same contract type. If no
    returns are available for a contract type the function will reduce the lookback window
    up to a minimum of 11 days. If no returns are available for a contract type for
    at least 11 days the function returns an NaN for that date and sends a warning of all
    the dates for which this happened.


    """
    ## Check inputs
    for varx, namex, typex in [
        (df, "df", pd.DataFrame),
        # (sname, "sname", str),
        (contids, "contids", list),
        (est_freq, "est_freq", str),
        (lback_periods, "lback_periods", int),
        (lback_meth, "lback_meth", str),
        (half_life, "half_life", int),
        (rstring, "rstring", str),
        (start, "start", (str, NoneType)),
        (end, "end", (str, NoneType)),
        (blacklist, "blacklist", (dict, NoneType)),
    ]:
        if not isinstance(varx, typex):
            raise ValueError(f"`{namex}` must be {typex}.")
        if typex in [str, list, dict] and len(varx) == 0:
            raise ValueError(f"`{namex}` must not be an empty {str(typex)}.")

    ## Standardize and copy DF
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


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_test_df
    from contract_signals import contract_signals

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
