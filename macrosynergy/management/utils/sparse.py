from typing import Dict, List, Any, Union, Optional, Callable
from numbers import Number

import pandas as pd
import numpy as np
from macrosynergy.management.utils import (
    qdf_to_ticker_df,
    ticker_df_to_qdf,
    concat_single_metric_qdfs,
    get_cid,
    get_xcat,
)
from macrosynergy.management.types import QuantamentalDataFrame

import warnings


def _get_diff_data(
    ticker: str,
    p_value: pd.DataFrame,
    p_eop: pd.DataFrame,
    p_grading: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get the diff data for a given ticker from wide/pivoted dataframes (`ticker_df`) of
    metrics `value`, `eop_lag` and `grading`.

    :param <str> ticker: The ticker to get the diff data for.
    :param <pd.DataFrame> p_value: The pivoted DataFrame of the `value` metric.
    :param <pd.DataFrame> p_eop: The pivoted DataFrame of the `eop_lag` metric.
    :param <pd.DataFrame> p_grading: The pivoted DataFrame of the `grading` metric.
    """

    # calculate basic density stats
    diff_mask = p_value.diff(axis=0).abs() > 0.0
    diff_density = 100 * diff_mask[ticker].sum() / (~p_value[ticker].isna()).sum()
    fvi = p_value[ticker].first_valid_index().strftime("%Y-%m-%d")
    lvi = p_value[ticker].last_valid_index().strftime("%Y-%m-%d")
    dtrange_str = f"{fvi} : {lvi}"
    ddict = {
        "diff_density": diff_density,
        "date_range": dtrange_str,
    }

    dates = p_value[ticker].index[diff_mask[ticker]]

    # create the diff dataframe
    df_temp = pd.concat(
        (
            p_value.loc[dates, ticker].to_frame("value"),
            p_eop.loc[dates, ticker].to_frame("eop_lag"),
            p_grading.loc[dates, ticker].to_frame("grading"),
        ),
        axis=1,
    )
    df_temp["eop"] = df_temp.index - pd.to_timedelta(df_temp["eop_lag"], unit="D")
    df_temp["release"] = df_temp["eop_lag"].diff(periods=1) < 0

    df_temp = df_temp.sort_index().reset_index()
    df_temp["count"] = df_temp.index

    df_temp = pd.merge(
        left=df_temp,
        right=df_temp.groupby(["eop"], as_index=False)["count"].min(),
        on=["eop"],
        how="outer",
        suffixes=(None, "_min"),
    )
    df_temp["version"] = df_temp["count"] - df_temp["count_min"]
    df_temp["diff"] = df_temp["value"].diff(periods=1)
    df_temp = df_temp.set_index("real_date")[
        ["value", "eop", "version", "grading", "diff"]
    ]

    return df_temp, ddict


def create_delta_data(
    df: QuantamentalDataFrame, return_density_stats: bool = False
) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Creates a dictionary of dataframes with the changes in the information state for each
    ticker in the QuantamentalDataFrame. Optionally, returns a DataFrame with the statistics
    for change frequency, density and date range for each ticker.
    :param <QuantamentalDataFrame> df: The QuantamentalDataFrame to calculate the changes for.
    :param <bool> return_density_stats: If True, returns a DataFrame with the density stats for each ticker.

    :return: A dictionary of DataFrames with the changes in the information state for each ticker.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise ValueError("`df` must be a QuantamentalDataFrame")

    # split into value, eop and grading
    p_value = qdf_to_ticker_df(df, value_column="value")
    p_eop = qdf_to_ticker_df(df, value_column="eop_lag")
    if "grading" not in df.columns:
        df["grading"] = np.nan
    p_grading = qdf_to_ticker_df(df, value_column="grading")
    assert set(p_value.columns) == set(p_eop.columns) == set(p_grading.columns)

    # create dicts to store the dataframes and density stats
    isc_dict: Dict[str, Any] = {}
    density_stats: Dict[str, Dict[str, Any]] = {}
    for ticker in p_value.columns:
        df_temp, ddict = _get_diff_data(
            ticker=ticker,
            p_value=p_value,
            p_eop=p_eop,
            p_grading=p_grading,
        )
        isc_dict[ticker], density_stats[ticker] = df_temp, ddict

    # flatten the density stats
    _dstats_flat = [
        (k, v["diff_density"], v["date_range"]) for k, v in density_stats.items()
    ]

    density_stats_df = (
        pd.DataFrame(_dstats_flat, columns=["ticker", "changes_density", "date_range"])
        .sort_values(by="changes_density", ascending=False)
        .reset_index(drop=True)
    )
    if return_density_stats:
        return isc_dict, density_stats_df
    else:
        return isc_dict


class SubscriptableMeta(type):
    """
    Convenience metaclass to allow subscripting of methods on a class.
    """

    def __getitem__(cls, item):
        if hasattr(cls, item) and callable(getattr(cls, item)):
            return getattr(cls, item)
        else:
            raise KeyError(f"{item} is not a valid method name")


class VolatilityEstimationMethods(metaclass=SubscriptableMeta):
    """
    Class to hold methods for calculating standard deviations.
    Each method must comply to the following signature:
        `func(s: pd.Series, **kwargs) -> pd.Series`

    Currently supported methods are:

    - `std`: Standard deviation
    - `abs`: Mean absolute deviation
    - `exp`: Exponentially weighted standard deviation
    - `exp_abs`: Exponentially weighted mean absolute deviation
    """

    @staticmethod
    def std(s: pd.Series, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the expanding standard deviation of a Series.

        :param <pd.Series> s: The Series to calculate the standard deviation for.
        :param <int> min_periods: The minimum number of periods required for the calculation.
        :return: The standard deviation of the Series.
        """
        return s.expanding(min_periods=min_periods).std()

    @staticmethod
    def abs(s: pd.Series, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the expanding mean absolute deviation of a Series.

        :param <pd.Series> s: The Series to calculate the absolute standard deviation for.
        :param <int> min_periods: The minimum number of periods required for the calculation.
        :return: The absolute standard deviation of the Series.
        """
        mean = s.expanding(min_periods=min_periods).mean()
        return (s - mean.bfill()).abs().expanding(min_periods=min_periods).mean()

    @staticmethod
    def exp(s: pd.Series, halflife: int, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the exponentially weighted standard deviation of a Series.

        :param <pd.Series> s: The Series to calculate the exponentially weighted standard
            deviation for.
        :param <int> halflife: The halflife of the exponential weighting.
        :param <int> min_periods: The minimum number of periods required for the
            calculation.
        :return: The exponentially weighted standard deviation of the Series.
        """
        return s.ewm(halflife=halflife, min_periods=min_periods).std()

    @staticmethod
    def exp_abs(s: pd.Series, halflife: int, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the exponentially weighted mean absolute deviation of a Series.

        :param <pd.Series> s: The Series to calculate the exponentially weighted absolute
            standard deviation for.
        :param <int> halflife: The halflife of the exponential weighting.
        :param <int> min_periods: The minimum number of periods required for the calculation.
        :return: The exponentially weighted absolute standard deviation of the Series.
        """
        mean = s.ewm(halflife=halflife, min_periods=min_periods).mean()
        sd = (s - mean.bfill()).abs().ewm(halflife=halflife, min_periods=min_periods).mean()
        return sd


CALC_SCORE_CUSTOM_METHOD_ERR_MSG = (
    "Method {std} not supported. "
    f"Supported methods are: {dir(VolatilityEstimationMethods)}. \n"
    "Alternatively, provide a custom method with signature "
    "`custom_method(s: pd.Series, **kwargs) -> pd.Series` "
    "using the `custom_method` and `custom_method_kwargs` arguments."
)


def calculate_score_on_sparse_indicator(
    isc: Dict[str, pd.DataFrame],
    std: str = "std",
    halflife: int = None,
    min_periods: int = 10,
    isc_version: int = 0,
    iis: bool = False,
    custom_method: Optional[Callable] = None,
    custom_method_kwargs: Dict = {},
) -> Dict[str, pd.DataFrame]:
    """Calculate score on sparse indicator

    :param <Dict[str, pd.DataFrame]> isc: A dictionary of DataFrames with the changes in
        the information state for each ticker.

    :param <str> std: The method to use for calculating the standard deviation.
        Supported methods are `std`, `abs`, `exp` and `exp_abs`. See the documentation for
        `VolatilityEstimationMethods` for more information.
    :param <int> halflife: The halflife of the exponential weighting. Only used with `exp`
        and `exp_abs` methods. Default is None.
    :param <int> min_periods: The minimum number of periods required for the calculation.
        Default is 10.
    :param <int> isc_version: The version of the information state changes to use. If set
        to 0 (default), only the first version is used. If set to any other positive integer,
        all versions are used.
    :param <bool> iis: if True (default) zn-scores are also calculated for the initial
        sample period defined by `min_periods`, on an in-sample basis, to avoid losing history.

    :param <Callable> custom_method: A custom method to use for calculating the standard
        deviation. Must have the signature `custom_method(s: pd.Series, **kwargs) -> pd.Series`.

    :param <Dict> custom_method_kwargs: Keyword arguments to pass to the custom method.

    :return: A dictionary of DataFrames with the changes in the information state for each ticker.
    """
    # TODO make into a method on InformationStateChanges?
    # TODO adjust score by eop_lag (business days?) to get a native frequency...
    # TODO convert below operation into a function call?
    # Operations on a per key in data dictionary

    curr_method: Callable[[pd.Series, Optional[Dict[str, Any]]], pd.Series]
    if custom_method is not None:
        if not callable(custom_method):
            raise TypeError("`custom_method` must be a callable")
        if not isinstance(custom_method_kwargs, dict):
            raise TypeError("`custom_method_kwargs` must be a dictionary")
        curr_method = custom_method
    else:
        if not hasattr(VolatilityEstimationMethods, std):
            raise ValueError(CALC_SCORE_CUSTOM_METHOD_ERR_MSG.format(std=std))
        # curr_method = getattr(StandardDeviationMethod, std)
        curr_method = VolatilityEstimationMethods[std]

    method_kwargs: Dict[str, Any] = dict(
        min_periods=min_periods, halflife=halflife, **custom_method_kwargs
    )
    # if not 0, then use all versions
    for key, v in isc.items():
        mask_rel = (v["version"] == 0) if isc_version == 0 else (v["version"] >= 0)
        s = v.loc[mask_rel, "diff"]
        # TODO exponential weights (requires knowledge of frequency...)

        result: pd.Series = curr_method(s, **method_kwargs)

        columns = [kk for kk in v.columns if kk != "std"]
        v = pd.merge(
            left=v[columns],
            right=result.to_frame("std"),
            how="left",
            left_index=True,
            right_index=True,
        )
        v["std"] = v["std"].ffill()
        if iis:
            v["std"] = v["std"].bfill()
        v["zscore"] = v["diff"] / v["std"]

        isc[key] = v

    # TODO return [1] change, and [2] volatility estimate (mainly estimation of volatility for changes...)
    # TODO clearer exposition
    return isc


def _infer_frequency_timeseries(eop_series: pd.Series) -> Optional[str]:
    """
    Infer the frequency of a time series based on the most common difference between
    consecutive end-of-period dates.
    :param <pd.Series> eop_series: A Series of end-of-period dates.
    :return: The inferred frequency of the time series. One of "D", "W", "M", "Q" or "A".
    """
    diff = eop_series.diff().dropna()
    most_common = diff.mode().values[0]
    frequency_mapping = {1: "D", 5: "W"}

    if most_common in frequency_mapping:
        return frequency_mapping[most_common]

    # Define ranges for other frequencies
    frequency_ranges = {
        "M": range(20, 24),  # (20-23 bdays)
        "Q": range(60, 67),  # (60-66 bdays)
        "A": range(250, 253),  # (250-252 bdays)
    }
    for freq, freq_range in frequency_ranges.items():
        if most_common in freq_range:
            return freq

    return "D"


def infer_frequency(df: QuantamentalDataFrame) -> pd.Series:
    """
    Infer the frequency of a QuantamentalDataFrame based on the most common difference
    between consecutive end-of-period dates.

    :param <QuantamentalDataFrame> df: The QuantamentalDataFrame to infer the frequency for.
    :return: A Series with the inferred frequency for each ticker in the QuantamentalDataFrame.
    """
    if not isinstance(df, QuantamentalDataFrame):
        raise ValueError("`df` must be a QuantamentalDataFrame")
    if not "eop_lag" in df.columns:
        raise ValueError("`df` must contain an `eop_lag` column")

    ticker_df = qdf_to_ticker_df(df, value_column="eop_lag")
    freq_dict = {
        ticker: _infer_frequency_timeseries(ticker_df[ticker])
        for ticker in ticker_df.columns
    }
    return pd.Series(freq_dict)


def weight_from_frequency(freq: str, base: float = 252):
    """Weight from frequency"""
    # TODO apply on multiple tickers
    freq_map = {"D": 1, "W": 5, "M": 21, "Q": 93, "A": 252}
    assert freq in freq_map, f"Frequency {freq} not supported"
    return freq_map[freq] / base


def _remove_insignificant_values(
    df: pd.DataFrame, threshold: float = 1e-12
) -> pd.DataFrame:
    """
    Convenience function to remove insignificant values from a DataFrame.

    :param <pd.DataFrame> df: The DataFrame to remove insignificant values from.
    :param <float> threshold: The threshold below which values are considered insignificant.
    :return: The DataFrame with insignificant values removed.
    """
    return df / (df.cumsum(axis=0).abs() > threshold).astype(int)


def _isc_dict_to_frames(
    isc: Dict[str, pd.DataFrame], metric: str = "value"
) -> List[pd.DataFrame]:
    """
    Convert a dictionary of DataFrames to a list of DataFrames with a specific metric.
    """
    frames = []
    for k, v in isc.items():
        assert isinstance(v, pd.DataFrame)
        assert metric in v.columns
        _fr = v[metric].to_frame(k)
        if _fr.empty:
            warnings.warn(f"Empty frame for {k}")
            continue

        frames.append(_fr)

    return frames


def _get_metric_df_from_isc(
    isc: Dict[str, pd.DataFrame],
    metric: str,
    date_range: pd.DatetimeIndex,
    fill: Union[str, Number] = 0,
) -> pd.DataFrame:
    """
    Get a DataFrame with a specific metric from a dictionary of DataFrames.

    :param <Dict[str, pd.DataFrame]> isc: A dictionary of DataFrames with the changes in
        the information state for each ticker.
    :param <str> metric: The name of the metric to extract.
    :param <pd.DatetimeIndex> date_range: The date range to reindex the DataFrame to.
    :param <Union[str, Number]> fill: The value to fill NaNs with. If 'ffill', forward fill
        NaNs. Default is 0.
    """
    fill_err: str = "`fill` must be a number to replace NaNs or 'ffill' to forward fill"
    if not isinstance(fill, (str, Number)):
        raise TypeError(fill_err)
    if isinstance(fill, str):
        if not fill == "ffill":
            raise ValueError(fill_err)

    df_frames: List[pd.DataFrame] = _isc_dict_to_frames(isc, metric=metric)
    tdf: pd.DataFrame = pd.concat(df_frames, axis=1).reindex(date_range)
    if fill == "ffill":
        tdf = tdf.ffill()
    else:
        tdf = tdf.fillna(fill)
    tdf.columns.name, tdf.index.name = "ticker", "real_date"
    return tdf


def sparse_to_dense(
    isc: Dict[str, pd.DataFrame],
    value_column: str,
    min_period: pd.Timestamp,
    max_period: pd.Timestamp,
    postfix: str = None,
    metrics: List[str] = ["eop", "grading"],
) -> pd.DataFrame:
    """
    Convert a dictionary of DataFrames with changes in the information state to a dense
    DataFrame (QuantamentalDataFrame).

    :param <Dict[str, pd.DataFrame]> isc: A dictionary of DataFrames with the changes in
        the information state for each ticker.
    :param <str> value_column: The name of the column to use as the value.
    :param <pd.Timestamp> min_period: The minimum period to include in the DataFrame.
    :param <pd.Timestamp> max_period: The maximum period to include in the DataFrame.
    :param <str> postfix: A postfix to append to the xcat column. Default is None.
    :param <List[str]> metrics: A list of metrics to include in the DataFrame. Default is
        ["eop", "grading"].
    :return: A dense DataFrame with the changes in the information state.
    """

    # TODO store real_date min and max in object...
    dtrange = pd.date_range(
        start=min_period,
        end=max_period,
        freq="B",
        inclusive="both",
    )

    tdf = _get_metric_df_from_isc(isc=isc, metric=value_column, date_range=dtrange)
    tdf = _remove_insignificant_values(tdf, threshold=1e-12)

    sm_qdfs: List[QuantamentalDataFrame] = [ticker_df_to_qdf(tdf)]
    for metric_name in metrics:
        wdf = _get_metric_df_from_isc(
            isc=isc, metric=metric_name, date_range=dtrange, fill="ffill"
        )
        # if wdf.empty or wdf.isna().all().all()
        if wdf.empty or wdf.isna().all().all():
            dfs = [
                pd.DataFrame(index=dtrange)
                .assign(
                    cid=get_cid(tickerx),
                    xcat=get_xcat(tickerx),
                    obs=np.nan,
                )
                .rename(columns={"obs": metric_name})
                .reset_index()
                for tickerx in wdf.columns
            ]
            m_qdf = pd.concat(dfs, axis=0).reset_index(drop=True)
        else:
            m_qdf = ticker_df_to_qdf(wdf, metric=metric_name)

        sm_qdfs.append(m_qdf)

    qdf: QuantamentalDataFrame = concat_single_metric_qdfs(sm_qdfs)
    if "eop" in metrics:
        qdf["eop_lag"] = (qdf["real_date"] - qdf["eop"]).dt.days

    if postfix:
        qdf["xcat"] += postfix

    tickers = (qdf["cid"] + "_" + qdf["xcat"]).unique().tolist()
    for cid, xcat in zip(get_cid(tickers), get_xcat(tickers)):
        mask = (qdf["cid"] == cid) & (qdf["xcat"] == xcat)
        min_date = qdf.loc[mask, "real_date"].min()
        # assert qdf.loc[mask & (qdf["real_date"] == min_date), "value"].isna().all()
        if qdf.loc[mask & (qdf["real_date"] == min_date)].isna().any().any():
            qdf = qdf[~mask | (qdf["real_date"] != min_date)]

    return qdf


def temporal_aggregator_exponential(
    df: QuantamentalDataFrame,
    halflife: int = 5,
    winsorise: float = None,
) -> pd.DataFrame:
    """
    Temporal aggregator using exponential moving average.

    :param <QuantamentalDataFrame> df: The QuantamentalDataFrame to aggregate.
    :param <int> halflife: The halflife of the exponential moving average.
    :param <float> winsorise: The value to winsorise the data to. Default is None.
    :return: A DataFrame with the aggregated values.
    """
    tdf: pd.DataFrame = qdf_to_ticker_df(df)
    if winsorise:
        tdf = tdf.clip(lower=-winsorise, upper=winsorise)
    # Exponential moving average weights
    tdf = tdf.ewm(halflife=halflife).mean()
    qdf: QuantamentalDataFrame = ticker_df_to_qdf(tdf)
    qdf["xcat"] += f"EWM{halflife:d}D"
    return qdf


def temporal_aggregator_mean(
    df: QuantamentalDataFrame,
    window: int = 21,
    winsorise: float = None,
) -> pd.DataFrame:
    """
    Temporal aggregator using a rolling mean.

    :param <QuantamentalDataFrame> df: The QuantamentalDataFrame to aggregate.
    :param <int> window: The window size for the rolling mean.
    :param <float> winsorise: The value to winsorise the data to. Default is None.
    :return: A DataFrame with the aggregated values.
    """
    tdf: pd.DataFrame = qdf_to_ticker_df(df)
    if winsorise:
        tdf = tdf.clip(lower=-winsorise, upper=winsorise)

    tdf = tdf.rolling(window=window).mean()
    qdf: QuantamentalDataFrame = ticker_df_to_qdf(tdf)
    qdf["xcat"] += f"MA{window:d}D"
    return qdf


def temporal_aggregator_period(
    isc: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    winsorise: int = 10,
    postfix: str = "_NCSUM",
) -> pd.DataFrame:
    """Temporal aggregator over periods of changes in the information state.

    :param <Dict[str, pd.DataFrame]> isc: A dictionary of DataFrames with the changes in
        the information state for each ticker.
    :param <pd.Timestamp> start: The start date of the period to aggregate.
    :param <pd.Timestamp> end: The end date of the period to aggregate.
    :param <int> winsorise: The value to winsorise the data to. Default is 10.
    :param <str> postfix: A postfix to append to the xcat column. Default is "_NCSUM".
    :return: A DataFrame with the aggregated values.
    """
    dt_range = pd.date_range(start=start, end=end, freq="B", inclusive="both")
    tdf: pd.DataFrame = _get_metric_df_from_isc(
        isc=isc,
        metric="zscore_norm_squared",
        date_range=dt_range,
        fill=0,
    )
    # Winsorise and remove insignificant values
    tdf = _remove_insignificant_values(tdf, threshold=1e-12)
    tdf = tdf.clip(lower=-winsorise, upper=winsorise)

    # Map out the eop dates
    p_eop: pd.DataFrame = _get_metric_df_from_isc(
        isc=isc,
        metric="eop",
        date_range=dt_range,
        fill="ffill",
    )

    qdf: QuantamentalDataFrame = concat_single_metric_qdfs(
        [
            ticker_df_to_qdf(tdf),
            ticker_df_to_qdf(p_eop, metric="eop"),
        ]
    )

    # group by cid, xcat, eop
    qdf["value"] = (
        qdf.groupby(["cid", "xcat", "eop"])["value"].cumsum().reset_index(drop=True)
    )
    if postfix:
        qdf["xcat"] += postfix
    return qdf


def _calculate_score_on_sparse_indicator_for_class(
    cls: "InformationStateChanges",
    std: str = "std",
    halflife: int = None,
    min_periods: int = 10,
    isc_version: int = 0,
    iis: bool = False,
    custom_method: Optional[Callable] = None,
    custom_method_kwargs: Dict = {},
):
    """
    Calculate score on sparse indicator for a class.
    Effectively a re-implementation of the function `calculate_score_on_sparse_indicator`
    that specifically operates on an `InformationStateChanges` object.
    """
    assert isinstance(
        cls, InformationStateChanges
    ), "cls must be an InformationStateChanges object"
    assert hasattr(cls, "isc_dict") and isinstance(
        cls.isc_dict, dict
    ), "InformationStateChanges object not initialized"

    curr_method: Callable[[pd.Series, Optional[Dict[str, Any]]], pd.Series]
    if custom_method is not None:
        if not callable(custom_method):
            raise TypeError("`custom_method` must be a callable")
        if not isinstance(custom_method_kwargs, dict):
            raise TypeError("`custom_method_kwargs` must be a dictionary")
        curr_method = custom_method
    else:
        if not hasattr(VolatilityEstimationMethods, std):
            raise ValueError(CALC_SCORE_CUSTOM_METHOD_ERR_MSG.format(std=std))
        # curr_method = getattr(StandardDeviationMethod, std)
        curr_method = VolatilityEstimationMethods[std]

    method_kwargs: Dict[str, Any] = dict(
        min_periods=min_periods, halflife=halflife, **custom_method_kwargs
    )
    # if not 0, then use all versions
    for key, v in cls.isc_dict.items():
        mask_rel = (v["version"] == 0) if isc_version == 0 else (v["version"] >= 0)
        s = v.loc[mask_rel, "diff"]

        result: pd.Series = curr_method(s, **method_kwargs)

        columns = [kk for kk in v.columns if kk != "std"]
        v = pd.merge(
            left=v[columns],
            right=result.to_frame("std"),
            how="left",
            left_index=True,
            right_index=True,
        )
        v["std"] = v["std"].ffill()
        if iis:
            v["std"] = v["std"].bfill()
        v["zscore"] = v["diff"] / v["std"]

        cls.isc_dict[key] = v


class InformationStateChanges(object):
    """
    Class to hold information state changes for a set of tickers.

    Initialize using the `from_qdf` class method to create an `InformationStateChanges`
    object from a `QuantamentalDataFrame`. The `calculate_score` method can be used to
    calculate scores for the information state changes.
    """

    def __init__(
        self,
        min_period: pd.Timestamp = None,
        max_period: pd.Timestamp = None,
    ):
        self.isc_dict: Dict[str, pd.DataFrame] = dict()
        self.density_stats_df: pd.DataFrame = None
        self._min_period: pd.Timestamp = min_period
        self._max_period: pd.Timestamp = max_period

    def __getitem__(self, item):
        return self.isc_dict[item]

    def __setitem__(self, key, value):
        self.isc_dict[key] = value

    def __str__(self):
        return str(self.isc_dict)

    def __repr__(self):
        return repr(self.isc_dict)

    def keys(self):
        """
        A list of tickers in the InformationStateChanges object.
        """
        return self.isc_dict.keys()

    def values(self):
        """
        Extract the DataFrames from the InformationStateChanges object.
        """
        return self.isc_dict.values()

    def items(self):
        """
        Iterate through (ticker, DataFrame) pairs in the InformationStateChanges object.
        """
        return self.isc_dict.items()

    @classmethod
    def from_qdf(
        cls, qdf: QuantamentalDataFrame, norm: bool = True, **kwargs
    ) -> "InformationStateChanges":
        """
        Create an InformationStateChanges object from a QuantamentalDataFrame.

        :param <QuantamentalDataFrame> qdf: The QuantamentalDataFrame to create the
            InformationStateChanges object from.
        :param <bool> norm: If True, calculate the score for the information state changes.
        :param <**kwargs>: Additional keyword arguments to pass to the `calculate_score`
            method.
        :return: An InformationStateChanges object.
        """

        isc = cls(min_period=qdf["real_date"].min(), max_period=qdf["real_date"].max())
        isc_dict, density_stats_df = create_delta_data(qdf, return_density_stats=True)

        isc.isc_dict = isc_dict
        isc.density_stats_df = density_stats_df

        if norm:
            isc.calculate_score(**kwargs)

        return isc

    def to_qdf(
        self,
        value_column: str = "value",
        postfix: str = None,
        metrics: List[str] = ["eop", "grading"],
    ) -> pd.DataFrame:
        """
        Convert the InformationStateChanges object to a QuantamentalDataFrame.

        :param <str> value_column: The name of the column to use as the value.
        :param <str> postfix: A postfix to append to the xcat column. Default is None.
        :param <List[str]> metrics: A list of metrics to include in the DataFrame. Default is
            ["eop", "grading"].
        :return: A QuantamentalDataFrame with the information state changes.
        """
        return sparse_to_dense(
            isc=self.isc_dict,
            value_column=value_column,
            min_period=self._min_period,
            max_period=self._max_period,
            postfix=postfix,
            metrics=metrics,
        )

    def temporal_aggregator_period(
        self,
        winsorise: int = 10,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Temporal aggregator over periods of changes in the information state.

        :param <int> winsorise: The value to winsorise the data to. Default is 10.
        :param <pd.Timestamp> start: The start date of the period to aggregate.
        :param <pd.Timestamp> end: The end date of the period to aggregate.
        :return: A DataFrame with the aggregated values.



        """
        return temporal_aggregator_period(
            isc=self.isc_dict,
            start=start or self._min_period,
            end=end or self._max_period,
            winsorise=winsorise,
        )

    def calculate_score(
        self,
        std: str = "std",
        halflife: int = None,
        min_periods: int = 10,
        isc_version: int = 0,
        iis: bool = False,
        custom_method: Optional[Callable] = None,
        custom_method_kwargs: Dict = {},
    ):
        """
        Calculate score on sparse indicator for the InformationStateChanges object.

        :param <str> std: The method to use for calculating the standard deviation.
            Supported methods are `std`, `abs`, `exp` and `exp_abs`. See the documentation for
            `StandardDeviationMethods` for more information.

        :param <int> halflife: The halflife of the exponential weighting. Only used with `exp`
            and `exp_abs` methods. Default is None.
        :param <int> min_periods: The minimum number of periods required for the calculation.
            Default is 10.
        :param <int> isc_version: The version of the information state changes to use. If set
            to 0 (default), only the first version is used. If set to any other positive integer,
            all versions are used.
        :param <bool> iis: if True (default) zn-scores are also calculated for the initial
            sample period defined by `min_periods`, on an in-sample basis, to avoid losing history.
        :param <Callable> custom_method: A custom method to use for calculating the standard
            deviation. Must have the signature `custom_method(s: pd.Series, **kwargs) -> pd.Series`.
        :param <Dict> custom_method_kwargs: Keyword arguments to pass to the custom method.


        """

        _calculate_score_on_sparse_indicator_for_class(
            cls=self,
            std=std,
            halflife=halflife,
            min_periods=min_periods,
            isc_version=isc_version,
            iis=iis,
            custom_method=custom_method,
            custom_method_kwargs=custom_method_kwargs,
        )
