from typing import Dict, List, Any, Union, Optional
from numbers import Number

import pandas as pd
import numpy as np
from macrosynergy.management.utils import (
    qdf_to_ticker_df,
    ticker_df_to_qdf,
    concat_single_metric_qdfs,
)
from macrosynergy.management.types import QuantamentalDataFrame

import warnings


# class InformationStateChanges(object):
#     """
#     # Functions for operations on sparse data
#     # class SparseIndicators(Dict) ...:
#     # """
#     def __init__(self, values: Dict[str, pd.DataFrame] = dict()):
#         # TODO store start and end dates per ticker...
#         self.values: Dict[str, pd.DataFrame] = values

#     @classmethod
#     def from_tickers(cls, tickers: List[str]) -> "InformationStateChanges":
#         return cls({ticker: pd.DataFrame for ticker in tickers})

#     def __setitem__(self, item: str, value: pd.DataFrame):
#         self.values[item] = value

#     def __getitem__(self, item: str) -> pd.DataFrame:
#         return self.values[item]

#     def __setattr__(self, key: str, value: pd.DataFrame):
#         self.values[key] = value

#     def __getattr__(self, item) -> pd.DataFrame:
#         return self.values[item]

#     def __getstate__(self):
#         return self.values

#     def __setstate__(self, values: Dict[str, pd.DataFrame]):
#         self.values = values

#     def keys(self):
#         return self.values.keys()

#     def items(self):
#         return self.values.items()

#     def to_dense(self) -> pd.DataFrame:
#         # TODO convert to QuantamentalDataFrame
#         pass


def _get_diff_data(
    ticker: str,
    p_value: pd.DataFrame,
    p_eop: pd.DataFrame,
    p_grading: pd.DataFrame,
) -> pd.DataFrame:
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
    df: pd.DataFrame, return_density_stats: bool = False
) -> pd.DataFrame:
    # split into value, eop and grading
    p_value = qdf_to_ticker_df(df, value_column="value")
    p_eop = qdf_to_ticker_df(df, value_column="eop_lag")
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


def calculate_score_on_sparse_indicator(
        isc: Dict[str, pd.DataFrame],
        std: str = "std",
        halflife: int = None,
        min_periods: int = 10
):
    """Calculate score on sparse indicator

    :param isc: InformationStateChanges
    :param std: str default "std" (quadratic loss function i.e. standard deviations)
        alternatives are "abs" for linex loss function (mean absolute deviations),
        exp for exponentially weighted quadratic loss function (requires halflife to be specified),
        and exp_abs for Linex loss function (mean absolute deviations exponentially weighted).
    :param halflife: int default None


    """
    # TODO make into a method on InformationStateChanges?
    # TODO adjust score by eop_lag (business days?) to get a native frequency...
    # TODO convert below operation into a function call?
    # Operations on a per key in data dictionary
    for key, v in isc.items():
        mask_rel = v["version"] == 0
        s = v.loc[mask_rel, "diff"]
        # TODO exponential weights (requires knowledge of frequency...)
        if std == "std":
            std = s.expanding(min_periods=10).std()
        elif std == "abs":
            std = s.abs().expanding(min_periods=10).mean()
        elif std == "exp":
            assert halflife is not None and halflife > 0, "halflife must be defined"
            std = s.ewm(halflife=halflife).std()
        elif std == "exp_abs":
            assert halflife is not None and halflife > 0, "halflife must be defined"
            std = s.abs().ewm(halflife=halflife).mean()
        else:
            raise ValueError(f"std {std} not supported")
        
        columns = [kk for kk in v.columns if kk != "std"]
        v = pd.merge(
            left=v[columns],
            right=std.to_frame("std"),
            how="left",
            left_index=True,
            right_index=True,
        )
        v["std"] = v["std"].ffill()
        v["zscore"] = v["diff"] / v["std"]

        isc[key] = v

    # TODO return [1] change, and [2] volatility estimate (mainly estimation of volatility for changes...)
    # TODO clearer exposition
    return isc


def _infer_frequency_timeseries(eop_series: pd.Series) -> Optional[str]:
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
    return df / (df.cumsum(axis=0).abs() > threshold).astype(int)


def _isc_dict_to_frames(
    isc: Dict[str, pd.DataFrame], metric: str = "value"
) -> List[pd.DataFrame]:
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
    if not isinstance(fill, (str, Number)):
        raise ValueError("fill must be a string or a number")
    if isinstance(fill, str):
        assert fill == "ffill", "Only ffill is a valid operation for time series fill"

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
    add_eop: bool = True,
    add_grading: bool = True,
) -> pd.DataFrame:
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
    for add_bool, metric_name in zip(
        [add_eop, add_grading],
        ["eop", "grading"],
    ):
        if add_bool:
            wdf = _get_metric_df_from_isc(
                isc=isc, metric=metric_name, date_range=dtrange, fill="ffill"
            )
            sm_qdfs.append(ticker_df_to_qdf(wdf, metric=metric_name))

    qdf: QuantamentalDataFrame = concat_single_metric_qdfs(sm_qdfs)
    if add_eop:
        qdf["eop_lag"] = (qdf["real_date"] - qdf["eop"]).dt.days

    if postfix:
        qdf["xcat"] += postfix
    # TODO add eop! (and grading!)
    return qdf


def temporal_aggregator_exponential(
    df: pd.DataFrame,
    halflife: int = 5,
    winsorise: float = None,
) -> pd.DataFrame:
    tdf: pd.DataFrame = qdf_to_ticker_df(df)
    if winsorise:
        tdf = tdf.clip(lower=-winsorise, upper=winsorise)
    # Exponential moving average weights
    # (check implementation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)

    tdf = tdf.ewm(halflife=halflife).mean()

    qdf: QuantamentalDataFrame = ticker_df_to_qdf(tdf)
    qdf["xcat"] += f"EWM{halflife:d}D"
    return qdf


def temporal_aggregator_mean(
    df: QuantamentalDataFrame,
    window: int = 21,
    winsorise: float = None,
) -> pd.DataFrame:
    tdf: pd.DataFrame = qdf_to_ticker_df(df)
    if winsorise:
        tdf = tdf.clip(lower=-winsorise, upper=winsorise)

    tdf = tdf.rolling(window=window).mean()
    qdf: QuantamentalDataFrame = ticker_df_to_qdf(tdf)
    qdf["xcat"] += f"MA{window:d}D"
    return qdf


# Aggreagte per period (temporal aggregator)
# xcats = [cc + "_N" for cc in growth + inflation + labour + sentiment + financial]
# dfx = msm.reduce_df(df, xcats=xcats, cids=["USD"])


def temporal_aggregator_period(
    isc: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    winsorise: int = 10,
) -> pd.DataFrame:
    """Temporal aggregator over periods of changes

    TODO add argument to choose how many periods to aggregate over.
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
    qdf["xcat"] += "_NCSUM"
    return qdf  # TODO: Check ?

    # rename c

    # # TODO use zscore instead fo zscore_norm_linear!
    # dfa = (
    #     pd.merge(
    #         left=tdf.stack().to_frame("value").reset_index(),
    #         right=p_eop.stack().to_frame("eop").reset_index(),
    #         how="left",
    #         on=["real_date", "ticker"],
    #     )
    #     .sort_values(by=["ticker", "real_date"])
    #     .rename(columns={"ticker": "xcat"})
    # )
    # dfa["cumsum"] = dfa.groupby(["xcat", "eop"])["value"].cumsum()
    # dfa["cid"] = "USD"

    # # Aggregate (cumulatively) on a per period basis
    # dfa["csum"] = dfa.groupby(["xcat", "eop"])["value"].cumsum()
    # dfa["xcat"] += "_NCSUM"

    # return dfa[["real_date", "cid", "xcat", "csum"]].rename(columns={"csum": "value"})


if __name__ == "__main__":
    ...
