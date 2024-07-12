import unittest
import pandas as pd
import warnings
import datetime
import numpy as np

from typing import List, Tuple, Dict, Union, Set, Any
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import concat_single_metric_qdfs, ticker_df_to_qdf
from macrosynergy.management.utils.sparse import InformationStateChanges
import random
import string

FREQ_STR_MAP = {
    "B": "daily",
    "W-FRI": "weekly",
    "BME": "monthly",
    "BQE": "quarterly",
    "BYE": "yearly",
}


def random_string(
    length: int = 4,
) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=length))


def get_end_of_period_for_date(
    date: pd.Timestamp,
    freq: str,
) -> pd.Timestamp:
    prev_date = pd.bdate_range(start=date, periods=1, freq=freq)[0]
    return prev_date


def get_long_format_data(
    cids: List[str] = ["USD", "EUR", "JPY", "GBP"],
    start: str = "2010-01-01",
    end: str = "2020-01-21",
    xcats: List[str] = ["GDP", "CPI", "UNEMP", "RATE"],
    num_freqs: int = 2,
) -> pd.DataFrame:
    # Map of frequency codes to their descriptive names
    freq_map = FREQ_STR_MAP.copy()
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
    eop_ts_list: List[pd.Series] = []
    eoplag_ts_list: List[pd.Series] = []

    for ticker, freq in ticker_freq_tuples:
        dts = pd.bdate_range(start=start, end=end, freq=freq)
        namex = f"{ticker}_{freq_map[freq].upper()}"
        ts = pd.Series(np.random.random(len(dts)), index=dts, name=namex)
        ts = ts.reindex(full_date_range).ffill()
        ts.loc[ts.isna()] = np.random.random(ts.isna().sum())
        values_ts_list.append(ts)

        tseop = pd.Series(
            dict([(dt, get_end_of_period_for_date(dt, freq)) for dt in dts]), name=namex
        )
        tseop = tseop.reindex(full_date_range).ffill()
        tseop.loc[tseop.isna()] = min(full_date_range)
        eop_ts_list.append(tseop)

        _tdf = tseop.to_frame().reset_index()
        _tdf["eop_lag"] = _tdf.apply(
            lambda x: abs((x["index"].date() - x[tseop.name].date()).days),
            axis=1,
        )
        tseop = pd.Series(_tdf["eop_lag"].values, index=_tdf["index"], name=tseop.name)
        eoplag_ts_list.append(tseop)

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


def _get_helper_col(df: QuantamentalDataFrame) -> QuantamentalDataFrame:
    assert isinstance(df, QuantamentalDataFrame)
    return df["cid"] + "-" + df["xcat"] + "-" + df["real_date"].dt.strftime("%Y-%m-%d")


class TestInformationStateChanges(unittest.TestCase):
    def test_isc_object_round_trip(self) -> None:

        qdfidx = QuantamentalDataFrame.IndexCols

        df = get_long_format_data()
        dfc = df.copy()
        tdf = InformationStateChanges.from_qdf(df).to_qdf()

        diff_mask = abs(df["value"].diff())
        diff_mask: pd.Series = diff_mask > 0.0  # type: ignore
        diff_df: pd.DataFrame = tdf.loc[diff_mask, :].reset_index(drop=True)

        # create helper column which is cid-xcat-real_date
        diff_df.loc[:, "helper"] = _get_helper_col(diff_df)
        dfc.loc[:, "helper"] = _get_helper_col(dfc)

        # reduce the diff_df to only the diffs (helpers)
        dfc = dfc[dfc["helper"].isin(diff_df["helper"])].reset_index(drop=True)

        diff_df = diff_df.drop(columns=qdfidx).set_index("helper")
        dfc = dfc.drop(columns=qdfidx).set_index("helper")

        # keep only the users columns (no grading columns)
        diff_df = diff_df.loc[:, dfc.columns]

        self.assertTrue(diff_df.equals(dfc))

    def test_isc_to_qdf(self) -> None:
        df = get_long_format_data()
        ## Test that the grading is not output when not asked for
        tdf = InformationStateChanges.from_qdf(df).to_qdf(metrics=["eop"])
        self.assertTrue("value" in tdf.columns)
        self.assertTrue("eop" in tdf.columns)
        self.assertTrue("eop_lag" in tdf.columns)
        self.assertTrue("grading" not in tdf.columns)


if __name__ == "__main__":
    unittest.main()
