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
import os
import hashlib
from pandas.core.groupby import DataFrameGroupBy
import numpy as np
import random
import io
import json
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

    tdf = qdf_to_ticker_df(df)

    index_dates = pd.Series(tdf.index).apply(lambda x: x.strftime("%Y-%m-%d"))
    assert index_dates.nunique() == len(index_dates)

    col_hashes: Dict[str, str] = {
        col: str(hashlib.md5(str(tdf[col].values).encode()).hexdigest())
        for col in tdf.columns
    }
    hash_values: Dict[str, pd.Series] = {
        _hash: tdf[col] for col, _hash in col_hashes.items()
    }
    hash_values_df = pd.DataFrame(hash_values).T
    hash_values_df.columns = list(hash_values_df.columns)
    hash_values_df = hash_values_df.rename(
        columns={
            colname: pd.Timestamp(colname).strftime("%Y-%m-%d")
            for colname in hash_values_df.columns
        }
    )
    # change to list of float
    hash_values = {_hash: hash_values[_hash].tolist() for _hash in hash_values}
    hash_values_df_csv_str = hash_values_df.to_csv()

    hash_cols: Dict[str, List[str]] = {}
    for col, _hash in col_hashes.items():
        hash_cols[_hash] = hash_cols.get(_hash, []) + [col]

    hash_dict: Dict[str, Dict[str, List[str]]] = {}
    for _hash, cols in hash_cols.items():
        cidgroup = ",".join(sorted(set(get_cid(cols))))
        _xc = set(get_xcat(cols))
        assert len(_xc) == 1
        xcat = list(_xc)[0]
        if cidgroup not in hash_dict:
            hash_dict[cidgroup] = {"xcats": [xcat], "hashes": [_hash]}
        else:
            hash_dict[cidgroup]["xcats"].append(xcat)
            hash_dict[cidgroup]["hashes"].append(_hash)

    hash_dict_json = json.dumps(hash_dict)

    return {
        "hash_values": hash_values,
        "hash_values_df_csv_str": hash_values_df_csv_str,
        "hash_dict_json": hash_dict_json,
    }


sample_real_data_frame()
