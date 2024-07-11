import unittest
import pandas as pd
import warnings
import datetime
import numpy as np

from typing import List, Tuple, Dict, Union, Set, Any

from macrosynergy.management.utils import (
    get_cid,
    get_xcat,
    concat_single_metric_qdfs,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
)
from macrosynergy.management.utils.sparse import (
    _get_diff_data,
    create_delta_data,
    StandardDeviationMethods,
    calculate_score_on_sparse_indicator,
    _infer_frequency_timeseries,
    infer_frequency,
    weight_from_frequency,
    _remove_insignificant_values,
    _isc_dict_to_frames,
    _get_metric_df_from_isc,
    sparse_to_dense,
    temporal_aggregator_exponential,
    temporal_aggregator_mean,
    temporal_aggregator_period,
    _calculate_score_on_sparse_indicator_for_class,
    InformationStateChanges,
)
import random
import string


def random_string(
    length: int = 4,
) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=length))


def get_end_of_period_for_date(
    dt: pd.Timestamp,
    freq: str,
) -> pd.Timestamp:
    # Get the end of the period for a given date and frequency
    if freq == "B":
        return dt
    elif freq == "W-FRI":
        return dt + pd.DateOffset(days=4)
    elif freq == "BME":
        return dt + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    elif freq == "BQE":
        return dt + pd.DateOffset(months=3) - pd.DateOffset(days=1)
    elif freq == "BYE":
        return dt + pd.DateOffset(years=1) - pd.DateOffset(days=1)
    else:
        raise ValueError(f"Unknown frequency: {freq}")


def get_long_format_data(
    cids: List[str] = ["USD", "EUR", "JPY", "GBP"],
    start: str = "2010-01-01",
    end: str = "2020-01-01",
    xcats: List[str] = ["GDP", "CPI", "UNEMP", "RATE"],
    # num_xcats: int = 4,
    num_freqs: int = 2,
) -> pd.DataFrame:
    # Map of frequency codes to their descriptive names
    freq_map = {
        "B": "daily",
        "W-FRI": "weekly",
        "BME": "monthly",
        "BQE": "quarterly",
        "BYE": "yearly",
    }

    full_date_range = pd.bdate_range(start=start, end=end)

    get_random_freq = lambda: random.choice(list(freq_map.keys()))

    # Generate ticker symbols by combining currency ids and category names
    tickers = [f"{cid}_{xc}" for cid in cids for xc in xcats]
    ticker_freq_tuples = [
        (ticker, get_random_freq())
        for ticker in tickers
        for num_freq in range(num_freqs)
    ]

    # Generate time series data for each (ticker, frequency) tuple
    values_ts_list: List[pd.Series] = []
    for ticker, freq in ticker_freq_tuples:
        dts = pd.bdate_range(start=start, end=end, freq=freq)
        namex = f"{ticker}_{freq_map[freq].upper()}"
        ts = pd.Series(np.random.random(len(dts)), index=dts, name=namex)
        ts = ts.reindex(full_date_range).ffill()
        values_ts_list.append(ts)

    eop_ts_list: List[pd.Series] = []
    for ticker, freq in ticker_freq_tuples:
        dts = pd.bdate_range(start=start, end=end, freq=freq)
        namex = f"{ticker}_{freq_map[freq].upper()}"
        tpls = [(dt, get_end_of_period_for_date(dt, freq)) for dt in dts]
        ts = pd.Series(dict(tpls), name=namex)
        ts = ts.reindex(full_date_range).ffill()
        eop_ts_list.append(ts)

    eoplag_ts_list: List[pd.Series] = []
    for _ts in eop_ts_list:
        eop_lag = pd.Series(
            (_ts.index.to_series() - _ts).dt.days, index=_ts.index, name=_ts.name
        )
        eoplag_ts_list.append(eop_lag)

    def concat_ts_list(
        ts_list: List[pd.Series],
        metric: str,
    ) -> pd.DataFrame:
        df = pd.concat(ts_list, axis=1)
        df.index.name, df.columns.name = "real_date", "ticker"
        return ticker_df_to_qdf(df, metric=metric)

    return concat_single_metric_qdfs(
        [
            concat_ts_list(ts_list=values_ts_list, metric="value"),  # good
            concat_ts_list(ts_list=eop_ts_list, metric="eop"),
            concat_ts_list(ts_list=eoplag_ts_list, metric="eop_lag"),
        ]
    )


class TestFunctions(unittest.TestCase): ...


if __name__ == "__main__":
    df = get_long_format_data()
    df
