"""
Module with functionality for generating mock transaction costs data.



"""

from typing import Optional, List, Dict, Any, Union
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (
    get_cid,
    get_xcat,
    standardise_dataframe,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
)
from macrosynergy.management.constants import MARKET_AREAS
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
import numpy as np
import random
from macrosynergy.download import download_transaction_costs

BIDOFFER_MEDIAN: str = "BIDOFFER_MEDIAN"
ROLLCOST_MEDIAN: str = "ROLLCOST_MEDIAN"
SIZE_MEDIAN: str = "SIZE_MEDIAN"
BIDOFFER_90PCTL: str = "BIDOFFER_90PCTL"
ROLLCOST_90PCTL: str = "ROLLCOST_90PCTL"
SIZE_90PCTL: str = "SIZE_90PCTL"
CTYPES: List[str] = ["FX", "IRS", "CDS"]


def get_market_area(cid: str) -> str:
    for area, countries in MARKET_AREAS.items():
        if cid in countries:
            return area
    return "OTHER"


def sample_real_data_frame() -> QuantamentalDataFrame:

    df = download_transaction_costs(verbose=True)

    def earliest_date_in_group(group: pd.DataFrame) -> pd.DataFrame:
        return group.loc[group["real_date"].idxmin()]

    df = standardise_dataframe(
        df[
            df["real_date"].isin(
                (
                    df.groupby(["cid", "xcat", "value"])
                    .apply(earliest_date_in_group, include_groups=False)
                    .drop_duplicates()
                    .reset_index(drop=True)
                )["real_date"]
            )
        ].copy()
    )

    # df["cid"] = df["cid"].apply(get_market_area)

    # first group the dataframe by cid,xcat,real_date. keep only the first value for each group
    # df = df.groupby(["cid", "xcat", "real_date"]).mean().reset_index()

    tdf = qdf_to_ticker_df(df)
    id_cols = {}
    for ic, col in enumerate(tdf.columns):
        for colx in list(tdf.columns)[len(tdf.columns) - ic :]:
            if np.isclose(tdf[col], tdf[colx]).all():
                id_cols[col] = id_cols.get(col, []) + [colx]

    print(id_cols)

    return tdf


sample_real_data_frame()
