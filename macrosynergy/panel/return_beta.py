"""
Functions for calculating the hedge ratios of a panel of returns with respect to a
single return. 
"""

import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import (
    reduce_df,
    _map_to_business_day_frequency,
    standardise_dataframe,
)
import matplotlib.pyplot as plt


def date_alignment(
    unhedged_return: pd.Series, benchmark_return: pd.Series
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Method used to align the two Series over the same timestamps: the sample data for the
    endogenous & exogenous variables must match throughout the re-estimation calculation.

    :param <pd.DataFrame> unhedged_return: the return series of the asset that is being
        hedged.
    :param <pd.Series> benchmark_return: the return series of the asset being used to
        hedge against the main asset.

    :return <pd.Timestamp, pd.Timestamp>: the shared start and end date across the two
        series.
    """
    ma_dates = unhedged_return.index
    ha_dates = benchmark_return.index

    if ma_dates[0] > ha_dates[0]:
        start_date = ma_dates[0]
    else:
        start_date = ha_dates[0]

    if ma_dates[-1] > ha_dates[-1]:
        end_date = ha_dates[-1]
    else:
        end_date = ma_dates[-1]

    return start_date, end_date


def hedge_calculator(
    unhedged_return: pd.Series,
    benchmark_return: pd.Series,
    rdates: List[pd.Timestamp],
    cross_section: str,
    meth: str = "ols",
    min_obs: int = 24,
    max_obs: int = 1000,
) -> pd.DataFrame:
    """
    Calculate the hedge ratios for each cross-section in the panel being hedged. It is
    worth noting that the sample of data used for calculating the hedge ratio will
    increase according to the dates parameter: each date represents an additional number
    of timestamps where the numeracy of dates added to the sample is instructed by the
    "refreq" parameter.

    :param <pd.Series> unhedged_return: the return series of the asset that is being
        hedged.
    :param <pd.Series> benchmark_return: the return series of the asset being used to
        hedge against the main asset.
    :param <List[pd.Timestamp]> rdates: the dates controlling the frequency of
        re-estimation.
    :param <str> cross_section: cross-section responsible for the "benchmark_return"
        series.
    :param <str> meth: method to estimate hedge ratio. At present the only method is
        OLS regression ('ols').
    :param <int> min_obs: a hedge ratio will only be computed if the number of days has
        surpassed the integer held by the parameter.
    :param <int> max_obs: the maximum number of latest observations allowed in order to
        estimate a hedge ratio. The default value is 1000.

    :return <pd.DataFrame>: returns a dataframe of the hedge ratios for the respective
        cross-section.
    """

    benchmark_return = benchmark_return.astype(dtype=np.float32)
    unhedged_return = unhedged_return.astype(dtype=np.float32)

    if benchmark_return.name is None:
        benchmark_return.name = "hedge"

    benchmark_return = benchmark_return[
        benchmark_return.first_valid_index() : benchmark_return.last_valid_index()
    ]
    unhedged_return = unhedged_return[
        unhedged_return.first_valid_index() : unhedged_return.last_valid_index()
    ]

    s_date, e_date = date_alignment(
        unhedged_return=unhedged_return, benchmark_return=benchmark_return
    )

    unhedged_return = unhedged_return.truncate(before=s_date, after=e_date)
    benchmark_return = benchmark_return.truncate(before=s_date, after=e_date)

    # The date series will be adjusted to each cross-section. Daily dates each return
    # series is defined over.
    date_series = unhedged_return.index
    df_ur = unhedged_return.to_frame(name="returns")
    df_ur = df_ur.reset_index()

    # Access the minimum date from the adjusted series: having aligned the unhedged asset
    # and the benchmark return. Both series will be defined over the same timestamps.
    min_obs_date = date_series[min_obs]

    # Storage dataframe defined over the re-balancing dates.
    data_column = np.empty(len(rdates))
    data_column[:] = np.nan
    df_hrat = pd.DataFrame(data=data_column, index=rdates, columns=["value"])

    for d in rdates:
        if d > min_obs_date:
            curr_start_date: pd.Timestamp = rdates[max(0, rdates.index(d) - max_obs)]
            # Inclusive of the re-estimation date.
            yvar = unhedged_return.loc[curr_start_date:d]
            xvar = benchmark_return.loc[curr_start_date:d]
            # Condition currently redundant but will become relevant.
            if meth == "ols":
                xvar = sm.add_constant(xvar)
                results: RegressionResults = sm.OLS(yvar, xvar).fit()
                results_params: pd.Series = results.params

            df_hrat.loc[d] = results_params.loc[benchmark_return.name]

    # Any dates prior to the minimum observation which would be classified by NaN values
    # remove from the DataFrame.
    df_hrat = df_hrat.dropna(axis=0, how="all")
    df_hrat.index.name = "real_date"
    df_hrat = df_hrat.reset_index(level=0)

    # Merge to convert to the re-estimation frequency. The intermediary dates, daily
    # business days between re-estimation dates, will be populated with np.NaN values.
    df_hr = df_ur.merge(df_hrat, on="real_date", how="left")

    df_hr = df_hr.drop("returns", axis=1)
    df_hr = df_hr.ffill()
    # Accounts for the application of the minimum number of observations required and
    # merging the two DataFrames. Drop the np.nan values prior to the application of the
    # shift (able to validate the logic).
    df_hr = df_hr.dropna(axis=0, how="any")

    df_hr = df_hr.set_index("real_date", drop=True).shift(1).reset_index(level=0)

    df_hr["cid"] = cross_section

    return df_hr


def adjusted_returns(
    benchmark_return: pd.Series, df_hedge: pd.DataFrame, dfw: pd.DataFrame
) -> pd.DataFrame:
    """
    Method used to compute the hedge ratio returns on the hedging asset which will
    subsequently be subtracted from the returns of the position contracts to calculate
    the adjusted returns (adjusted for the hedged position) across all cross-sections in
    the panel. For instance, if using US Equity to hedge Australia FX:
    AUD_FXXR_NSA_H = AUD_FXXR_NSA - HR_AUD * USD_EQXR_NSA.

    :param <pd.Series> benchmark_return: the return series of the asset being used to
        hedge against the main asset.
    :param <pd.DataFrame> df_hedge: standardised dataframe with the hedge ratios.
    :param <pd.DataFrame> dfw: pivoted dataframe of the relevant returns.

    :return <pd.DataFrame>: standardised dataframe of adjusted returns.
    """

    hedge_pivot = df_hedge.pivot(index="real_date", columns="cid", values="value")
    index = benchmark_return.index

    # Matching the dimensions to the number of assets being hedged.
    benchmark_return = np.tile(
        benchmark_return.to_numpy(), (len(hedge_pivot.columns), 1)
    ).T
    br_df = pd.DataFrame(
        data=benchmark_return, columns=hedge_pivot.columns, index=index
    )

    hedged_returns = hedge_pivot.multiply(br_df)
    adj_rets = dfw - hedged_returns

    df_stack = adj_rets.stack().reset_index(name="value")
    df_stack.columns = ["real_date", "cid", "value"]

    return df_stack


def return_beta(
    df: QuantamentalDataFrame,
    xcat: str = None,
    cids: List[str] = None,
    benchmark_return: str = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    meth: str = "ols",
    oos: bool = True,
    refreq: str = "M",
    min_obs: int = 24,
    max_obs: int = 1000,
    hedged_returns: bool = False,
    ratio_name: str = "_HR",
    hr_name: str = "H",
) -> QuantamentalDataFrame:
    """
    Estimate sensitivities (betas) of return category with respect to single return.

    :param <QuantamentalDataFrame> df: standardized DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and 'value.
    :param <str> xcat:  return category based on the type of positions that are
        to be hedged.
        N.B.: Each cross-section of this category uses the same hedge asset/basket.
    :param <List[str]> cids: cross-sections of the returns for which hedge ratios are
        to be calculated. Default is all that are available in the dataframe.
    :param <str> benchmark_return: ticker of return of the hedge asset or basket.
        This is a single series, e.g. U.S. equity index returns ("USD_EQXR_NSA").
    :param <str> start: earliest date in ISO format. Default is None: earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None: latest date in df is
        used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the sample of data used for estimating hedge ratios. The estimated ratios
        during blacklist periods will be set equal to the last valid estimate.
    :param <bool> oos: if True (default) hedge ratios are calculated out-of-sample,
        i.e. for the period following the estimation period at the given
        re-estimation frequency.
    :param <str> refreq: re-estimation frequency. This is period after which hedge ratios
        are re-estimated. The re-estimation is conducted at the end of the period and
        used as hedge ratio for all days of the following period. Re-estimation can have
        weekly, monthly, and quarterly frequency with the notations 'W', 'M', and 'Q'
        respectively. The default frequency is monthly.
    :param <int> min_obs: the minimum number of observations required in order to
        estimate a hedge ratio. The default value is 24 days.
        The permissible minimum is 10.
    :param <int> max_obs: the maximum number of latest observations allowed in order to
        estimate a hedge ratio. The default value is 1000.
    :param <str> meth: method used to estimate hedge ratio. At present the only method is
        OLS regression ('ols').
    :param <bool> hedged_returns: If True the function appends the hedged returns to the
        dataframe of hedge ratios. Default is False.
    :param <str> ratio_name: hedge ratio label that will be appended to the category
        name. The default is "_HR". For instance, 'xcat' + "_HR".
    :param <str> hr_name: label used to distinguish the hedged returns in the DataFrame.
        The label is appended to the category being hedged. The default is "H".

    :return <QuantamentalDataFrame>: DataFrame with hedge ratio estimates that update at the
        chosen re-estimation frequency.
        Additionally, the dataframe can include the hedged returns if the parameter
        `benchmark_return` has been set to True.

    N.B.: A return beta is the estimated sensitivity of the main return with respect to
    the asset used for hedging. The ratio is recorded for the period after the estimation
    sample up until the next re-estimation date.

    """

    # Value checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame but received <{type(df)}>.")
    df: QuantamentalDataFrame = standardise_dataframe(df)

    all_tix = np.unique(df["cid"] + "_" + df["xcat"])
    bm_error = f"Benchmark return ticker {benchmark_return} is not in the DataFrame."
    if not benchmark_return in all_tix:
        raise ValueError(bm_error)

    error_xcat = (
        f"The field, xcat, must be a string but received <{type(xcat)}>. Only"
        f" a single category is used to hedge against the main asset."
    )
    if not isinstance(xcat, str):
        raise ValueError(error_xcat)

    available_categories = df["xcat"].unique()
    error_hedging = (
        f"The return category used to be hedged, {xcat}, is "
        f"not defined in the dataframe."
    )
    if not xcat in list(available_categories):
        raise ValueError(error_hedging)

    min_obs_error = (
        "The number of minimum observations required to compute a hedge "
        "ratio is 10 business days, or two weeks. Please provide an integer "
        "value greater than 10."
    )
    if not isinstance(min_obs, int) or min_obs < 10:
        raise ValueError(min_obs_error)

    if not isinstance(max_obs, int) or max_obs < min_obs:
        raise ValueError(f"`max_obs` must be an integer ≫ `min_obs`.")

    # Information on hedge return and potential panel adjustment.

    post_fix = benchmark_return.split("_")
    xcat_hedge = "_".join(post_fix[1:])
    cid_hedge = post_fix[0]
    if xcat_hedge == xcat:
        if cid_hedge in cids:
            cids.remove(cid_hedge)
        warnings.warn(
            f"Return to be hedged for cross section {cid_hedge} is the hedge "
            f"return and has been removed from the panel."
        )

    # Wide time series DataFrame of unhedged and benchmark returns.

    # --- Time series DataFrame of unhedged returns.

    dfp = reduce_df(
        df, xcats=[xcat], cids=cids, start=start, end=end, blacklist=blacklist
    )
    dfp_w = dfp.pivot(index="real_date", columns="cid", values="value")
    dfp_w = dfp_w.dropna(axis=0, how="all")

    # --- Time series DataFrame of benchmark return for relevant dates.

    # The asset being used as the hedge could only be defined over a shorter time-period.
    dfh = reduce_df(
        df,
        xcats=[xcat_hedge],
        cids=cid_hedge,
        start=dfp_w.index[0],
        end=dfp_w.index[-1],
    )
    dfh_w = dfh.pivot(index="real_date", columns="cid", values="value")
    dfh_w.columns = ["hedge"]

    # --- Merge time series and calculate re-balancing dates.

    dfw = pd.merge(dfp_w, dfh_w, how="inner", on="real_date")
    br = dfw["hedge"]

    rf = _map_to_business_day_frequency(freq=refreq, valid_freqs=["W", "M", "Q"])
    dates_re = dfw.asfreq(rf).index

    if isinstance(dates_re, pd.DatetimeIndex):
        dates_re: List[str] = dates_re.to_list()

    # Cross-section-wise hedge ratio estimation.

    aggregate = []
    for c in cids:
        df_hr = hedge_calculator(
            unhedged_return=dfw[c],
            benchmark_return=br,
            rdates=dates_re,
            cross_section=c,
            meth=meth,
            min_obs=min_obs,
            max_obs=max_obs,
        )
        aggregate.append(df_hr)

    df_hedge = pd.concat(aggregate, ignore_index=True)

    df_hedge["xcat"] = xcat + ratio_name

    if hedged_returns:
        df_hreturn = adjusted_returns(df_hedge=df_hedge, dfw=dfw, benchmark_return=br)
        df_hreturn = df_hreturn.sort_values(["cid", "real_date"])
        df_hreturn["xcat"] = xcat + "_" + hr_name
        df_hedge = pd.concat([df_hedge, df_hreturn], ignore_index=True)

    return standardise_dataframe(df_hedge)


def beta_display(df_hedge: pd.DataFrame, subplots: bool = False, hr_name: str = "H"):
    """
    Method used to visualise the hedging ratios across the panel: assumes a single
    category is used to hedge the primary asset.

    :param <pd.DataFrame> df_hedge: DataFrame with hedge ratios.
    :param <bool> subplots: matplotlib parameter to determine if each hedging series is
        displayed on separate subplots.
    :param <str> hr_name: label used to distinguish the hedged returns in the DataFrame.
        Comparable to return_beta() method, the default is "H".

    """

    condition = lambda c: c.split("_")[-1] != hr_name

    apply = list(map(condition, df_hedge["xcat"]))
    df_hedge = df_hedge[apply]

    dfw_ratios = df_hedge.pivot(index="real_date", columns="cid", values="value")

    dfw_ratios.plot(subplots=subplots, title="Hedging Ratios.", legend=True)
    plt.xlabel("real_date, years")
    plt.show()


if __name__ == "__main__":
    # Emerging Market Asian countries.
    cids = ["IDR", "INR", "KRW", "MYR", "PHP"]
    # Add the US - used as the hedging asset.
    cids += ["USD"]
    xcats = ["FXXR_NSA", "GROWTHXR_NSA", "INFLXR_NSA", "EQXR_NSA"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )

    df_cids.loc["IDR"] = ["2010-01-01", "2020-12-31", 0.5, 2]
    df_cids.loc["INR"] = ["2011-01-01", "2020-11-30", 0, 1]
    df_cids.loc["KRW"] = ["2012-01-01", "2020-11-30", -0.2, 0.5]
    df_cids.loc["MYR"] = ["2013-01-01", "2020-09-30", -0.2, 0.5]
    df_cids.loc["PHP"] = ["2002-01-01", "2020-09-30", -0.1, 2]
    df_cids.loc["USD"] = ["2000-01-01", "2022-03-14", 0, 1.25]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )

    df_xcats.loc["FXXR_NSA"] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["GROWTHXR_NSA"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFLXR_NSA"] = ["2013-01-01", "2020-10-30", 1, 2, 0.8, 0.5]
    df_xcats.loc["EQXR_NSA"] = ["2010-01-01", "2022-03-14", 0.5, 2, 0, 0.2]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {"IDR": ["2010-01-01", "2014-01-04"], "INR": ["2010-01-01", "2013-12-31"]}

    xcat_hedge = "EQXR_NSA"
    # S&P500.
    benchmark_return = "USD_EQXR_NSA"
    df_hedge = return_beta(
        df=dfd,
        xcat=xcat_hedge,
        cids=["IDR", "INR", "KRW", "MYR", "PHP"],
        benchmark_return=benchmark_return,
        start="2010-01-01",
        end="2020-10-30",
        blacklist=black,
        meth="ols",
        oos=True,
        refreq="w",
        min_obs=24,
        hedged_returns=True,
    )

    print(df_hedge)
    beta_display(df_hedge=df_hedge, subplots=False)

    # Long position in S&P500 or the Nasdaq, and subsequently using US FX to hedge the
    # long position.
    xcats = "FXXR_NSA"
    cids = ["USD"]
    benchmark_return = "USD_EQXR_NSA"
    xcat_hedge_two = return_beta(
        df=dfd,
        xcat=xcats,
        cids=cids,
        benchmark_return=benchmark_return,
        start="2010-01-01",
        end="2020-10-30",
        blacklist=black,
        meth="ols",
        oos=True,
        refreq="m",
        min_obs=24,
    )
    print(xcat_hedge_two)
