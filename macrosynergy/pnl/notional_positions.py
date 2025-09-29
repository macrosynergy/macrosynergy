"""
Module for calculating notional positions based on contract signals, assets-under-
management, and other relevant parameters.
"""

import numpy as np
import pandas as pd
from numbers import Number
from typing import List, Union, Tuple, Optional, Set

from macrosynergy.management.utils import (
    standardise_dataframe,
    reduce_df,
    is_valid_iso_date,
    apply_slip as apply_slip_util,
    ticker_df_to_qdf,
)

from macrosynergy.management.types import NoneType, QuantamentalDataFrame
from macrosynergy.pnl.historic_portfolio_volatility import historic_portfolio_vol


def _apply_slip(
    df: pd.DataFrame,
    slip: int,
    fids: List[str],
) -> pd.DataFrame:
    """
    Applies a slip using the function `apply_slip()` to a dataframe with contract
    signals and returns.

    Parameters
    ----------
    df : pd.DataFrame
        Quantamental dataframe with contract signals and returns.
    slip : int
        the number of days to wait before applying the signal.
    fids : List[str]
        list of contract identifiers to apply the slip to.
    metrics : List[str]
        list of metrics to apply the slip to.
    """

    assert isinstance(df, QuantamentalDataFrame)
    assert isinstance(slip, int)
    assert (
        isinstance(fids, list)
        and len(fids) > 0
        and all([isinstance(x, str) for x in fids])
    )

    if slip == 0:
        return df
    else:
        cdf = QuantamentalDataFrame(df)
        filter_tickers = [tk for tk in cdf.list_tickers() if tk.startswith(tuple(fids))]
        cdf = cdf.reduce_df_by_ticker(tickers=filter_tickers)
        return apply_slip_util(
            df=cdf,
            tickers=filter_tickers,
            slip=slip,
            raise_error=False,
            metrics=["value"],
            extend_dates=True,
        )


def _check_df_for_contract_signals(
    df_wide: pd.DataFrame,
    sname: str,
    fids: List[str],
) -> None:
    """
    Checks if the dataframe contains contract signals for the specified strategy and the
    specified contract identifiers.

    Parameters
    ----------
    df : pd.DataFrame
        Wide dataframe with contract signals and returns.
    sname : str
        the name of the strategy.
    fids : List[str]
        list of contract identifiers to apply the slip to.
    """

    assert isinstance(sname, str)
    assert (
        isinstance(fids, list)
        and len(fids) > 0
        and all([isinstance(x, str) for x in fids])
    )

    sig_ident: str = f"_CSIG_{sname}"
    _check_conts: Set = set([f"{contx}{sig_ident}" for contx in fids])
    _found_conts: Set = set(df_wide.columns)
    if not _check_conts.issubset(_found_conts):
        raise ValueError(
            f"Contract signals for all contracts not in dataframe. \n"
            f"Missing: {_check_conts - _found_conts}"
        )

    return


def _vol_target_positions(
    df_wide: pd.DataFrame,
    sname: str,
    fids: List[str],
    aum: Number,
    vol_target: Number,
    rebal_freq: str,
    est_freqs: Union[str, List[str]],
    est_weights: Union[Number, List[Number]],
    lback_periods: Union[int, List[int]],
    half_life: Union[int, List[int]],
    nan_tolerance: float,
    remove_zeros: bool,
    lback_meth: str,
    rstring: str,
    pname: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Uses historic portfolio volatility to calculate notional positions based on contract
    signals, volatility targeting and other relevant parameters.
    """

    _check_df_for_contract_signals(df_wide=df_wide, sname=sname, fids=fids)

    sig_ident: str = f"_CSIG_{sname}"

    # TODO what is the units of histpvol?
    histpvol: QuantamentalDataFrame
    vcv_df: pd.DataFrame
    histpvol, vcv_df = historic_portfolio_vol(
        df=ticker_df_to_qdf(df_wide),
        sname=sname,
        fids=fids,
        rstring=rstring,
        lback_meth=lback_meth,
        est_freqs=est_freqs,
        est_weights=est_weights,
        lback_periods=lback_periods,
        half_life=half_life,
        nan_tolerance=nan_tolerance,
        rebal_freq=rebal_freq,
        remove_zeros=remove_zeros,
        return_variance_covariance=True,
    )
    # histpvol: only on rebalance dates...
    histpvol["scale"] = ((vol_target / histpvol["value"]) * aum).replace(np.inf, np.nan)
    # TODO check inf => convert to NaN
    histpvol.set_index("real_date", inplace=True)

    out_df = pd.DataFrame(index=df_wide.index)

    signal_columns: List[str] = [f"{contx:s}{sig_ident:s}" for contx in fids]
    df_signals: pd.DataFrame = df_wide.loc[histpvol.index, signal_columns]

    out_df: pd.DataFrame = (
        histpvol[["scale"]].dot(np.ones(shape=(1, df_signals.shape[1]))).values
        * df_signals
    )
    # TODO how to deal with unbalanced panel

    # drop rows with all na
    # TODO add log statement of how many N/A values are dropped
    out_df = out_df.reindex(df_wide.index)
    rebal_dates = sorted(histpvol.index.tolist())

    for num, rb in enumerate(rebal_dates[:-1]):
        mask = (out_df.index >= rb) & (out_df.index < rebal_dates[num + 1])
        out_df.loc[mask, :] = out_df.loc[mask, :].ffill()
    mask = out_df.index >= rebal_dates[-1]
    out_df.loc[mask, :] = out_df.loc[mask, :].ffill()

    # get na values per column
    na_per_col = out_df.isna().sum()
    na_per_col = na_per_col[na_per_col > 0]
    out_df = out_df.rename(
        columns={
            col: col.replace(sig_ident, f"_{sname}_{pname}")
            for col in out_df.columns.tolist()
        },
    ).dropna(how="all")

    return (
        out_df,
        standardise_dataframe(histpvol[["cid", "xcat", "value"]]),
        vcv_df,
    )


def _leverage_positions(
    df_wide: pd.DataFrame,
    sname: str,
    fids: List[str],
    aum: Number = 100,
    leverage: Number = 1.0,
    pname: str = "POS",
) -> pd.DataFrame:
    """"""
    _check_df_for_contract_signals(df_wide=df_wide, sname=sname, fids=fids)

    sig_ident: str = f"_CSIG_{sname}"

    _contracts: List[str] = [f"{contx}{sig_ident}" for contx in fids]

    rowsums: pd.Series = df_wide.loc[:, _contracts].sum(axis=1)
    # if any of the rowsums are zero, set to NaN to avoid div by zero
    rowsums[rowsums == 0] = np.nan

    for ic, contx in enumerate(fids):
        pos_col: str = f"{contx}_{sname}_{pname}"
        cont_name: str = contx + sig_ident
        # NOTE: this should be
        # dfw_pos = dfw_sigs * aum * leverage / rowsums(dfw_sigs)
        df_wide[pos_col] = df_wide[cont_name] * aum * leverage / rowsums

    # filter df to only contain position columns
    df_wide = df_wide.loc[:, [f"{contx}_{sname}_{pname}" for contx in fids]]

    return df_wide


def notional_positions(
    df: QuantamentalDataFrame,
    sname: str,
    fids: List[str],
    aum: Number = 100,
    dollar_per_signal: Number = 1.0,
    slip: int = 1,
    leverage: Optional[Number] = None,
    vol_target: Optional[Number] = None,
    nan_tolerance: float = 0.25,
    remove_zeros: bool = True,
    rebal_freq: str = "m",
    lback_meth: str = "ma",
    est_freqs: Union[str, List[str]] = ["D", "W", "M"],
    est_weights: Union[Number, List[Number]] = [1, 1, 1],
    lback_periods: Union[int, List[int]] = [-1, -1, -1],
    half_life: Union[int, List[int]] = [11, 5, 6],
    rstring: str = "XR",
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[dict] = None,
    pname: str = "POS",
    return_pvol: bool = False,
    return_vcv: bool = False,
) -> Union[
    QuantamentalDataFrame,
    Tuple[QuantamentalDataFrame, QuantamentalDataFrame],
    Tuple[QuantamentalDataFrame, pd.DataFrame],
    Tuple[QuantamentalDataFrame, QuantamentalDataFrame, pd.DataFrame],
]:
    """
    Calculates contract positions based on contract signals, assets under management
    (AUM) and other specifications.

    Parameters
    ----------
    df : QuantamentalDataFrame
        standardized JPMaQS DataFrame with the necessary columns: 'cid', 'xcat',
        'real_date' and 'value'. This dataframe must contain the contract-specific signals
        and possibly related return series (for vol-targeting).
    sname : str
        the name of the strategy. It must correspond to contract signals in the
        dataframe, which have the format "<cid>_<ctype>_CSIG_<sname>", and which are
        typically calculated by the function contract_signals().
    fids : List[str]
        list of financial contract identifiers in the format "<cid>_<ctype>". It must
        correspond to contract signals in the dataframe.
    aum : float
        the assets under management in USD million (for consistency). This is basis for
        all position sizes. Default is 100.
    dollar_per_signal : float
        the amount of notional (e.g. USD) per contract signal value. Default is 1. The
        default scale is arbitrary and is merely a basis for tryouts.
    leverage : float
        the ratio of the sum of notional positions to AUM. This is the main basis for
        leveraged-based positioning. Since different contracts have different expected
        volatility and correlations this method does not control expected volatility.
        Default is None, i.e. the method is not applied.
    vol_target : float
        the target volatility of the portfolio in % of AUM (For clarity, `vol_target=10`
        means 10%). This is the main parameter for volatility-targeted positioning. That
        method estimates the annualized standard deviation of the signal-based portfolio for
        a 1 USD per signal portfolio based on past variances and covariances of the contract
        returns. The estimation is managed by the function 
        :func:`macrosynergy.pnl.historic_portfolio_vol`. Default is None, i.e. the
        volatility-targeting is not applied.
    rebal_freq : str
        the rebalancing frequency. Default is 'm' for monthly. Alternatives are 'w' for
        business weekly, 'd' for daily, and 'q' for quarterly. Contract signals are taken
        from the end of the holding period and applied to positions at the beginning of the
        next period, subject to slippage.
    slip : int
        the number of days to wait before applying the signal. Default is 1. This means
        that new positions are taken at the very end of the first trading day of the holding
        period and are the basis of PnL calculation from the second Trading day onward.
    lback_meth : str
        the method to use for the lookback period of the volatility-targeting method.
        Default is 'ma' for moving average. Alternative is "xma", for exponential moving
        average. Again this is passed through to the function 
        :func:`macrosynergy.pnl.historic_portfolio_vol`.
    est_freqs : str or list of str
        the frequencies to use for the estimation of the variance-covariance matrix.
        Default is to use ["D", "W", "M"]. This is passed through to the function
        :func:`macrosynergy.pnl.historic_portfolio_vol`.
    est_weights : float or list of float
        the weights to use for the estimation of the variance-covariance matrix. The
        weights are normalized and are used to calculate the weighted average of the
        estimated variances and covariances. Default is [1, 1, 1]. This is passed
        through to the function :func:`macrosynergy.pnl.historic_portfolio_vol`.
    lback_periods : int or List[int]
        the number of periods to use for the lookback period of the volatility-targeting
        method. Each element corresponds to the the same index in `est_freqs`. Passing a
        single element will apply the same value to all frequencies. Default is [-1], which
        means that the lookback period is the full available data for all specified
    half_life : List[int]
        number of periods in the half-life of the exponential moving average. Each
        element corresponds to the same index in `est_freqs`. Default is [11, 5, 6] (for
        daily, weekly and monthly frequencies respectively). This is passed through to the
        function :func:`macrosynergy.pnl.historic_portfolio_vol`.
    rstring : str
        a general string of the return category. This identifies the contract returns
        that are required for the volatility-targeting method, based on the category
        identifier format <cid>_<ctype><rstring>_<rstring> in accordance with JPMaQS
        conventions. Default is 'XR'.
    start : str
        the start date of the data. Default is None, which means that the start date is
        taken from the dataframe.
    end : str
        the end date of the data. Default is None, which means that the end date is
        taken from the dataframe.
    blacklist : dict
        a dictionary of contract identifiers to exclude from the calculation. Default is
        None, which means that no contracts are excluded.
    pname : str
        the name of the position. Default is 'POS'.
    return_pvol : bool
        whether to return the historic portfolio volatility. Default is False.
    return_vcv : bool
        whether to return the variance-covariance matrix. Default is False.

    Returns
    -------
    pd.DataFrame
        a standard Quantamental DataFrame with the positions for all traded contracts
        and the specified strategy in USD million. The contract signals have the following
        format "<cid>_<ctype>_<sname>_<pname>".
    """
    if not isinstance(df, QuantamentalDataFrame):
        raise ValueError("`df` must be a QuantamentalDataFrame.")

    for varx, namex, typex in [
        (sname, "sname", str),
        (fids, "fids", list),
        (aum, "aum", Number),
        (dollar_per_signal, "dollar_per_signal", Number),
        (leverage, "leverage", (Number, NoneType)),
        (vol_target, "vol_target", (Number, NoneType)),
        (rebal_freq, "rebal_freq", str),
        (slip, "slip", int),
        (lback_meth, "lback_meth", str),
        (est_freqs, "est_freqs", (str, list)),
        (est_weights, "est_weights", (Number, list)),
        (lback_periods, "lback_periods", (int, list)),
        (half_life, "half_life", (int, list)),
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

    ## Convert df to QDF
    df: QuantamentalDataFrame = QuantamentalDataFrame(df)
    _initialized_as_categorical: bool = df.InitializedAsCategorical

    ## Volatility targeting and leverage cannot be applied at the same time
    if bool(leverage) and bool(vol_target):
        raise ValueError(
            "Either `leverage` or `vol_target` must be specified, but not both."
        )

    ## Check the dates
    if start is None:
        start: str = pd.Timestamp(df["real_date"].min()).strftime("%Y-%m-%d")
    if end is None:
        end: str = pd.Timestamp(df["real_date"].max()).strftime("%Y-%m-%d")

    for dx, nx in [(start, "start"), (end, "end")]:
        if not is_valid_iso_date(dx):
            raise ValueError(f"`{nx}` must be a valid ISO-8601 date string")

    ## Reduce the dataframe
    df: QuantamentalDataFrame = reduce_df(
        df=df, start=start, end=end, blacklist=blacklist
    )

    ## Check the contract identifiers and contract signals

    # df["ticker"] = df["cid"] + "_" + df["xcat"]
    df = QuantamentalDataFrame(df).add_ticker_column()

    # There must be atleast one contract signal with the strategy name
    if not any(df["ticker"].str.endswith(f"_CSIG_{sname}")):
        raise ValueError(f"No contract signals for strategy `{sname}` in dataframe.")

    # Check that all contract identifiers have at least one signal
    u_tickers: List[str] = list(df["ticker"].unique())
    for contx in fids:
        if not any(
            [tx.startswith(contx) and tx.endswith(f"_CSIG_{sname}") for tx in u_tickers]
        ):
            raise ValueError(f"Contract identifier `{contx}` not in dataframe.")

    ## Apply the slip
    df: pd.DataFrame = _apply_slip(
        df=df,
        slip=slip,
        fids=fids,
    )

    # TODO why pivot it out to a wide format?
    # df_wide = qdf_to_ticker_df(df)
    df_wide = QuantamentalDataFrame(df=df).to_wide()
    return_df = None
    if leverage:
        return_df: pd.DataFrame = _leverage_positions(
            df_wide=df_wide,
            sname=sname,
            fids=fids,
            aum=aum,
            leverage=leverage,
            pname=pname,
        )

    else:
        return_df, pvol, vcv_df = _vol_target_positions(
            df_wide=df_wide,
            sname=sname,
            fids=fids,
            aum=aum,
            vol_target=vol_target,
            rebal_freq=rebal_freq,
            lback_meth=lback_meth,
            est_freqs=est_freqs,
            est_weights=est_weights,
            lback_periods=lback_periods,
            half_life=half_life,
            rstring=rstring,
            pname=pname,
            nan_tolerance=nan_tolerance,
            remove_zeros=remove_zeros,
        )

    # return ticker_df_to_qdf(df=return_df).dropna()
    return_pvol = return_pvol and (locals().get("pvol") is not None)
    return_vcv = return_vcv and (locals().get("vcv_df") is not None)

    # return_df = ticker_df_to_qdf(df=return_df).dropna()
    return_df = QuantamentalDataFrame.from_wide(
        df=return_df, categorical=_initialized_as_categorical
    )
    if return_pvol and return_vcv:
        return (return_df, pvol, vcv_df)
    elif return_pvol:
        return (return_df, pvol)
    elif return_vcv:
        return (return_df, vcv_df)
    else:
        return return_df


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

    fids: List[str] = [f"{cid}_{ctype}" for cid in cids for ctype in ctypes]

    df_notional: pd.DataFrame = notional_positions(
        df=df_cs,
        fids=fids,
        leverage=1.1,
        sname="STRAT",
    )

    df_xr = make_test_df(
        cids=cids,
        xcats=[f"{_}XR" for _ in ctypes],
        start=start,
        end=end,
    )

    df_notional: pd.DataFrame = notional_positions(
        df=pd.concat([df_cs, df_xr], axis=0),
        fids=fids,
        sname="STRAT",
        vol_target=0.1,
        lback_meth="xma",
        lback_periods=-1,
        half_life=20,
        return_pvol=True,
        return_vcv=True,
    )
