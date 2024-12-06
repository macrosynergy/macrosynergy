"""
Module with functionality for generating mock quantamental data for testing purposes.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from typing import List, Dict, Union, Optional
from collections import defaultdict
import datetime
import warnings
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (
    ticker_df_to_qdf,
    is_valid_iso_date,
    get_cid,
    get_xcat,
)


def simulate_ar(nobs: int, mean: float = 0, sd_mult: float = 1, ar_coef: float = 0.75):
    """
    Create an auto-correlated data-series as numpy array.

    Parameters
    ----------
    nobs : int
        number of observations.
    mean : float
        mean of values, default is zero.
    sd_mult : float
        standard deviation multipliers of values, default is 1. This affects non-zero
        means.
    ar_coef : float
        autoregression coefficient (between 0 and 1): default is 0.75.

    Returns
    -------
    np.ndarray
        autocorrelated data series.
    """

    # Define relative parameters for creating an AR process.
    ar_params = np.r_[1, -ar_coef]
    ar_proc = ArmaProcess(ar_params)  # define ARMA process
    ser = ar_proc.generate_sample(nobs)
    ser = ser + mean - np.mean(ser)
    return sd_mult * ser / np.std(ser)


def dataframe_generator(
    df_cids: pd.DataFrame, df_xcats: pd.DataFrame, cid: str, xcat: str
):
    """
    Adjacent method used to construct the quantamental DataFrame.

    Parameters
    ----------
    df_cids : pd.DataFrame

    df_xcats : pd.DataFrame

    cid : str
        individual cross-section.
    xcat : str
        individual category.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DatetimeIndex]
        Tuple containing the quantamental DataFrame and a DatetimeIndex of the business
        days.
    """

    qdf_cols = ["cid", "xcat", "real_date", "value"]

    sdate = pd.to_datetime(
        max(df_cids.loc[cid, "earliest"], df_xcats.loc[xcat, "earliest"])
    )
    edate = pd.to_datetime(
        min(df_cids.loc[cid, "latest"], df_xcats.loc[xcat, "latest"])
    )
    all_days = pd.bdate_range(sdate, edate)
    work_days = all_days[all_days.weekday < 5]

    df_add = pd.DataFrame(columns=qdf_cols)
    df_add["real_date"] = work_days
    df_add["cid"] = cid
    df_add["xcat"] = xcat

    return df_add, work_days


def make_qdf(df_cids: pd.DataFrame, df_xcats: pd.DataFrame, back_ar: float = 0):
    """
    Make quantamental DataFrame with basic columns: 'cid', 'xcat', 'real_date', 'value'.

    Parameters
    ----------
    df_cids : pd.DataFrame
        DataFrame with parameters by cid. Row indices are cross-sections. Columns are:
        'earliest': string of earliest date (ISO) for which country values are available;
        'latest': string of latest date (ISO) for which country values are available;
        'mean_add': float of country-specific addition to any category's mean; 'sd_mult':
        float of country-specific multiplier of an category's standard deviation.
    df_xcats : pd.DataFrame
        dataframe with parameters by xcat. Row indices are cross-sections. Columns are:
        'earliest': string of earliest date (ISO) for which category values are available;
        'latest': string of latest date (ISO) for which category values are available;
        'mean_add': float of category-specific addition; 'sd_mult': float of country-
        specific multiplier of an category's standard deviation; 'ar_coef': float between 0
        and 1 denoting set auto-correlation of the category; 'back_coef': float, coefficient
        with which communal (mean 0, SD 1) background factor is added to category values.
    back_ar : float
        float between 0 and 1 denoting set auto-correlation of the background factor.
        Default is zero.

    Returns
    -------
    pd.DataFrame
        basic quantamental DataFrame according to specifications.
    """

    df_list = []

    if any(df_xcats["back_coef"] != 0):
        sdate = min(min(df_cids.loc[:, "earliest"]), min(df_xcats.loc[:, "earliest"]))
        edate = max(max(df_cids.loc[:, "latest"]), max(df_xcats.loc[:, "latest"]))
        all_days = pd.bdate_range(sdate, edate)
        work_days = all_days[all_days.weekday < 5]
        ser = simulate_ar(len(work_days), mean=0, sd_mult=1, ar_coef=back_ar)
        df_back = pd.DataFrame(index=work_days, columns=["value"])
        df_back["value"] = ser

    for cid in df_cids.index:
        for xcat in df_xcats.index:
            df_add, work_days = dataframe_generator(
                df_cids=df_cids, df_xcats=df_xcats, cid=cid, xcat=xcat
            )

            ser_mean = df_cids.loc[cid, "mean_add"] + df_xcats.loc[xcat, "mean_add"]
            ser_sd = df_cids.loc[cid, "sd_mult"] * df_xcats.loc[xcat, "sd_mult"]
            ser_arc = df_xcats.loc[xcat, "ar_coef"]
            df_add["value"] = simulate_ar(
                len(work_days), mean=ser_mean, sd_mult=ser_sd, ar_coef=ser_arc
            )

            back_coef = df_xcats.loc[xcat, "back_coef"]
            # Add the influence of communal background series.
            if back_coef != 0:
                dates = df_add["real_date"]
                df_add["value"] = df_add["value"] + back_coef * df_back.loc[
                    dates, "value"
                ].reset_index(drop=True)

            df_list.append(df_add)

    return pd.concat(df_list).reset_index(drop=True)


def make_qdf_black(df_cids: pd.DataFrame, df_xcats: pd.DataFrame, blackout: dict):
    """
    Make quantamental DataFrame with basic columns: 'cid', 'xcat', 'real_date', 'value'.
    In this DataFrame the column, 'value', will consist of Binary Values denoting
    whether the cross-section is active for the corresponding dates.

    Parameters
    ----------
    df_cids : pd.DataFrame
        dataframe with parameters by cid. Row indices are cross-sections. Columns are:
        'earliest': string of earliest date (ISO) for which country values are available;
        'latest': string of latest date (ISO) for which country values are available;
        'mean_add': float of country-specific addition to any category's mean; 'sd_mult':
        float of country-specific multiplier of an category's standard deviation.
    df_xcats : pd.DataFrame
        dataframe with parameters by xcat. Row indices are cross-sections. Columns are:
        'earliest': string of earliest date (ISO) for which category values are available;
        'latest': string of latest date (ISO) for which category values are available;
        'mean_add': float of category-specific addition; 'sd_mult': float of country-
        specific multiplier of an category's standard deviation; 'ar_coef': float between 0
        and 1 denoting set autocorrelation of the category; 'back_coef': float, coefficient
        with which communal (mean 0, SD 1) background factor is added to categoy values.
    blackout : dict
        Dictionary defining the blackout periods for each cross- section. The expected
        form of the dictionary is: {'AUD': (Timestamp('2000-01-13 00:00:00'),
        Timestamp('2000-01-13 00:00:00')), 'USD_1': (Timestamp('2000-01-03 00:00:00'),
        Timestamp('2000-01-05 00:00:00')), 'USD_2': (Timestamp('2000-01-09 00:00:00'),
        Timestamp('2000-01-10 00:00:00')), 'USD_3': (Timestamp('2000-01-12 00:00:00'),
        Timestamp('2000-01-12 00:00:00'))} The values of the dictionary are tuples
        consisting of the start & end-date of the respective blackout period. Each cross-
        section could have potentially more than one blackout period on a single category,
        and subsequently each key will be indexed to indicate the number of periods.

    Returns
    -------
    pd.DataFrame
        basic quantamental DataFrame according to specifications with binary values.
    """

    df_list = []

    conversion = lambda t: (pd.Timestamp(t[0]), pd.Timestamp(t[1]))
    dates_dict = defaultdict(list)
    for k, v in blackout.items():
        v = conversion(v)
        dates_dict[k[:3]].append(v)

    # At the moment the blackout period is being applied uniformally to each category:
    # each category will experience the same blackout periods.
    for cid in df_cids.index:
        for xcat in df_xcats.index:
            df_add, work_days = dataframe_generator(
                df_cids=df_cids, df_xcats=df_xcats, cid=cid, xcat=xcat
            )
            arr = np.repeat(0, df_add.shape[0])
            dates = df_add["real_date"].to_numpy()

            list_tuple = dates_dict[cid]
            for tup in list_tuple:
                start = tup[0]
                end = tup[1]

                condition_start = start.weekday() - 4
                condition_end = end.weekday() - 4

                # Will skip the associated blackout period because of the received
                # invalid date, if it is not within the respective data series' range,
                # but will continue to populate the dataframe according to the other keys
                # in the dictionary.
                # Naturally compare against the data-series' formal start & end date.
                if start < dates[0] or end > dates[-1]:
                    warnings.warn("Blackout period date not within data series range.")
                    break
                # If the date falls on a weekend, change to the following Monday.
                elif condition_start > 0:
                    while start.weekday() > 4:
                        start += datetime.timedelta(days=1)
                elif condition_end > 0:
                    while end.weekday() > 4:
                        end += datetime.timedelta(days=1)

                index_start = next(iter(np.where(dates == start)[0]))
                count = 0
                while start != tup[1]:
                    if start.weekday() < 5:
                        count += 1
                    start += datetime.timedelta(days=1)

                arr[index_start : (index_start + count + 1)] = 1

            df_add["value"] = arr

            df_list.append(df_add)

    return pd.concat(df_list).reset_index(drop=True)


def generate_lines(sig_len: int, style: str = "linear") -> Union[np.ndarray, List[str]]:
    """
    Returns a numpy array of a line with a given length.

    Parameters
    ----------
    sig_len : int
        The number of elements in the returned array.
    style : str
        The style of the line. Default `'linear'`. Current choices are: `linear`,
        `decreasing-linear`, `sharp-hill`, `four-bit-sine`, `sine`, `cosine`, `sawtooth`.
        Adding `"inv"` or "inverted" to the style will return the inverted version of that
        line. For example, `'inv-sawtooth'` or `'inverted sawtooth'` will return the
        inverted sawtooth line. `'any'` will return a random line. `'all'` will return a
        list of all the available styles.

    Returns
    -------
    Union[np.ndarray, List[str]]
        A numpy array of the line. If `style` is `'all'`, then a list (of strings) of
        all the available styles is returned.  NOTE: It is indeed request an `"inverted
        linear"` or `"inverted decreasing-linear"` line. They're just there for completeness
        and readability.
    """

    def _sawtooth(sig_len: int) -> np.ndarray:
        max_cycles = 4
        _tmp = sig_len // max_cycles
        up = np.tile(np.arange(-100, 100, 200 / _tmp), max_cycles * 2)
        return np.abs(up)[:sig_len]

    def _sharp_hill(sig_len: int) -> np.ndarray:
        return np.concatenate(
            [
                np.arange(1, sig_len // 4 + 1) * 100 / sig_len,
                np.arange(sig_len // 4, sig_len // 4 * 3 + 1)[::-1] * 100 / sig_len,
                np.arange(sig_len // 4 * 3, sig_len + 1) * 100 / sig_len,
            ]
        )

    def _four_bit_sine(sig_len: int) -> np.ndarray:
        return np.concatenate(
            [
                np.arange(1, sig_len // 4 + 1) * 100 / sig_len,
                np.arange(sig_len // 4, sig_len // 4 * 3 + 1)[::-1] * 100 / sig_len,
                np.arange(sig_len // 4 * 3, sig_len + 1) * 100 / sig_len,
            ]
        )

    def _sine(sig_len: int) -> np.ndarray:
        return np.sin(np.arange(1, sig_len + 1) * np.pi / (sig_len / 2)) * 50 + 50

    def _cosine(sig_len: int) -> np.ndarray:
        return np.cos(np.arange(1, sig_len + 1) * np.pi / (sig_len / 2)) * 50 + 50

    def _linear(sig_len: int) -> np.ndarray:
        return np.arange(1, sig_len + 1) * 100 / sig_len

    def _decreasing_linear(sig_len: int) -> np.ndarray:
        return _linear(sig_len)[::-1]

    if not isinstance(sig_len, int):
        raise TypeError("`sig_len` must be an integer.")
    elif sig_len < 1:
        raise ValueError("`sig_len` must be greater than 0.")

    if not isinstance(style, str):
        raise TypeError("`style` must be a string.")

    style: str = "-".join(style.strip().lower().split())

    lines: Dict[str, np.ndarray] = {
        "linear": _linear,
        "decreasing-linear": _decreasing_linear,
        "sharp-hill": _sharp_hill,
        "four-bit-sine": _four_bit_sine,
        "sine": _sine,
        "cosine": _cosine,
        "sawtooth": _sawtooth,
    }

    if "inv" in style:
        style = "-".join([s for s in style.split("-") if "inv" not in s])

    if style in lines:
        return lines[style](sig_len)[:sig_len]
    elif style == "any":
        r_choice: str = np.random.choice(list(lines.keys()))
        return lines[r_choice](sig_len)[:sig_len]
    elif style == "all":
        # return the list of choices
        opns: List[str] = list(lines.keys())
        inv_opns: List[str] = [f"inverted-{opn}" for opn in opns]
        return opns + inv_opns
    else:
        raise ValueError(f"Invalid style: {style}. Use one of: {list(lines.keys())}.")


def make_test_df(
    cids: Optional[List[str]] = ["AUD", "CAD", "GBP"],
    xcats: Optional[List[str]] = ["XR", "CRY"],
    tickers: Optional[List[str]] = None,
    metrics: List[str] = ["value"],
    start: str = "2010-01-01",
    end: str = "2020-12-31",
    style: str = "any",
) -> QuantamentalDataFrame:
    """
    Generates a test dataframe with pre-defined values. These values are meant to be
    used for testing purposes only. The functions generates a standard quantamental
    dataframe with where the value column is populated with pre-defined values. These
    values are simple lines, or waves that are easy to identify and differentiate in a
    plot.

    Parameters
    ----------
    cids : List[str]
        A list of strings for cids.
    xcats : List[str]
        A list of strings for xcats.
    tickers : List[str]
        A list of strings for tickers. If provided, `cids` and `xcats` will be ignored.
    metrics : List[str]
        A list of strings for metrics.
    start : str
        An ISO-formatted date string.
    end : str
        An ISO-formatted date string.
    style : str
        A string that specifies the type of line to generate. Current choices are:
        'linear', 'decreasing-linear', 'sharp-hill', 'four-bit-sine', 'sine', 'cosine',
        'sawtooth', 'any'. See
        `macrosynergy.management.simulate.simulate_quantamental_data.generate_lines()`.
    """

    ## Check the inputs
    if isinstance(cids, str):
        cids = [cids]
    if isinstance(xcats, str):
        xcats = [xcats]
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(metrics, list):
        raise TypeError("`metrics` must be a list of strings.")

    if "all" in metrics:
        metrics = ["value", "grading", "eop_lag", "mop_lag"]

    if (cids is None) != (xcats is None):
        raise ValueError("Please provide both `cids` and `xcats` or neither.")

    if tickers is None or len(tickers) == 0:
        if cids is None:
            raise ValueError("Please provide a list of tickers or `cids` & `xcats`.")

    if tickers is not None:
        cids = None
        xcats = None

    for varx, namex in zip(
        [cids, xcats, metrics, tickers], ["cids", "xcats", "metrics", "tickers"]
    ):
        if varx is not None:
            if not isinstance(varx, list):
                raise TypeError(f"`{namex}` must be a list.")
            if len(varx) == 0:
                raise ValueError(f"`{namex}` cannot be empty.")
            if not all(isinstance(x, str) for x in varx):
                raise TypeError(f"All elements in `{namex}` must be strings.")

    for varx, namex in zip([start, end], ["start", "end"]):
        if not is_valid_iso_date(varx):
            raise ValueError(f"`{namex}` must be a valid ISO date string.")

    ## Generate the dataframe

    dates: pd.DatetimeIndex = pd.bdate_range(start, end)
    all_tickers: List[str] = tickers if tickers is not None else []
    if cids is not None:
        all_tickers += [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
    all_tickers = sorted(set(all_tickers))
    df_list: List[pd.DataFrame] = []

    for ticker in all_tickers:
        df_add: pd.DataFrame = pd.DataFrame(
            index=dates, columns=["real_date", "cid", "xcat", *metrics]
        )
        df_add["cid"] = get_cid(ticker)
        df_add["xcat"] = get_xcat(ticker)
        df_add["real_date"] = dates
        for metric in metrics:
            df_add[metric] = generate_lines(len(dates), style=style)
        df_list.append(df_add)

    return pd.concat(df_list).reset_index(drop=True)


def simulate_returns_and_signals(
    # n_cids: int = 4,
    cids=["AUD", "CAD", "GBP", "USD"],
    xcat="EQ",
    return_suffix: str = "XR",
    signal_suffix: str = "_CSIG_STRAT",
    years: int = 20,
    sigma_eta: float = 0.01,
    sigma_0: float = 0.1,
    start: str = None,
    end: str = None,
):
    """Simulate returns and signals

    Equations for return and signal generation:
    1. r(t+1,i) = sigma(t+1,i)*(alpha(t+1,i) + beta(t+1,i)*rb(t+1) + epsilon(t+1,i))

    epsilon(t+1,i) ~ N(0, 1)

    2. ln(sigma(t+1,i)) = ln(sigma(t,i)) + eta(t+1,i), eta(t+1,i) ~ N(0, sigma_eta^2)

    3. alpha(t+1,i) = signal(t,i) + eta_alpha(t+1,i), eta_alpha(t+1,i) ~ N(0, sigma_alpha^2)

    4. beta(t+1,i) = beta(t,i) + eta_beta(t+1,i), eta_beta(t+1,i) ~ N(0, sigma_beta^2)

    5. rb(t+1) = mu + eta_rb(t+1), eta_rb(t+1) ~ N(0, sigma_rb^2)

    6. signal(t, i) =  ...  mean zero, but persistence....

    """

    n_cids = len(cids)
    periods = 252 * years
    assert (periods > 0) and (n_cids > 0)

    def simulate_volatility(
        periods: int = 252 * 20, sigma_eta: float = 0.01, sigma_0: float = 0.1
    ):
        sigma = np.empty(shape=(periods + 1))
        sigma[0] = sigma_0  # Daily volatility (10 percent ASD)
        eta_sigma = np.random.normal(0, sigma_eta, periods)
        for ii, ee in enumerate(eta_sigma):
            sigma[ii + 1] = np.exp(np.log(sigma[ii]) + ee)
        return sigma[1:]

    dates = pd.bdate_range(
        end=pd.Timestamp.today() + pd.offsets.BDay(n=0), periods=periods
    )
    # Generate volatility
    # print("Generate volatility (shared???)")
    volatility = np.empty(shape=(periods, n_cids))
    for nn in range(n_cids):
        volatility[:, nn] = simulate_volatility(
            periods=periods, sigma_eta=sigma_eta, sigma_0=sigma_0
        )

    # Generate signals: persistent?
    rho_signal = 0.9
    mean_signal = 0
    signals = np.empty(shape=(periods + 1, n_cids))
    signals[0, :] = mean_signal
    for tt in range(periods):
        signals[tt + 1, :] = (
            (1 - rho_signal) * mean_signal
            + rho_signal * signals[tt, :]
            + np.random.normal(0, 0.01, n_cids)
        )
    # signals = np.random.randn(periods, n_cids)  # Unit variance, zero mean
    signals = signals[1:, :]

    # Generate alpha
    # TODO alpha needs to be a function of lagged signal and not necessarily continous?
    # TODO signal and alpha can't be concurrent!
    # TODO signal proxy/captures a slow moving trend in the alpha (risk-premium)
    for ii in range(int(periods / years)):
        signals[ii * years : ii * years + years, :] = signals[ii * years, :]
    alpha = signals + np.random.randn(periods, n_cids)  # Unit variance, zero mean

    # Generate benchmark return
    rb = 0.4 / 252 + np.random.randn(periods, 1)  # 4% annual returns, unit variance

    # Generate beta
    beta = np.empty(shape=(1, n_cids, periods + 1))
    beta[:, :, 0] = 0.6  # Initial beta value

    for ii in range(periods):
        beta[:, :, ii + 1] = beta[:, :, ii] + 0.005 * np.random.randn(1, n_cids)
    beta = beta[:, :, 1:]
    # print("Final values of beta")
    # print(pd.Series(beta[0, :, -1]).describe())

    # TODO get with kron-product?
    rb_factor = np.array([rb[tt] * beta[0, :, tt] for tt in range(periods)])

    # Calculate returns
    rtn = volatility * (alpha + rb_factor + np.random.randn(periods, 1))

    # TODO test simulated returns matches random walk hypothesis on the face of it

    assert bool(start) ^ bool(end), "Only one of `start` or `end` is allowed."
    dtx = pd.Timestamp(start) if start else pd.Timestamp(end)
    dtx = pd.Timestamp(start) if start else pd.Timestamp(end) + pd.offsets.BDay(0)
    if start:
        dates = pd.bdate_range(start=dtx, periods=periods)
    else:
        dates = pd.bdate_range(end=dtx, periods=periods)

    df_rtn = pd.DataFrame(index=dates, data=rtn)
    df_signals = pd.DataFrame(index=dates, data=signals)
    # TODO change dates to be previous month...

    # TODO stack into quantamental dataframe
    # return df_rtn, df_signals
    xr_tickers = [f"{cid}_{xcat}{return_suffix}" for cid in cids]
    csig_tickers = [f"{cid}_{xcat}_{signal_suffix}" for cid in cids]
    dfR = pd.concat([df_rtn, df_signals], axis=1)
    dfR.columns = xr_tickers + csig_tickers
    dfR.index.name = "real_date"
    return ticker_df_to_qdf(dfR)


if __name__ == "__main__":
    ser_ar = simulate_ar(100, mean=0, sd_mult=1, ar_coef=0.75)

    cids = ["AUD", "CAD", "GBP"]
    xcats = ["XR", "CRY"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD",] = ["2010-01-01", "2020-12-31", 0.5, 2]
    df_cids.loc["CAD",] = ["2011-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP",] = ["2011-01-01", "2020-11-30", -0.2, 0.5]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
    df_xcats.loc["CRY",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
