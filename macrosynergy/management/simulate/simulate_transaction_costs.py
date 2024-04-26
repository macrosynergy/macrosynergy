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

    index_dates = pd.Series(tdf.index).apply(
        lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")
    )
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

    # for each set of columns that share a hash, select one
    sel_dict: Dict[str, Dict[str, Union[str, List[str]]]] = {}
    for _hash, cols in hash_cols.items():
        cols = sorted(cols)
        sel_dict[_hash] = {
            "xcat": list(set(get_xcat(cols))),
            "cids": list(set(get_cid(cols))),
        }

    # further squeeze the data: find xcats that have the same cids
    new_dict = {}
    for _hash, val in sel_dict.items():
        cids = val["cids"]
        xcats = val["xcat"]
        isnew = True
        for nidx, ditem in new_dict.items():
            # if cids are the same, add xcats to the existing xcats
            if set(cids) == set(ditem["cids"]):
                ditem["xcat"] = list(set(ditem["xcat"] + xcats))
                isnew = False
                break
        if isnew:
            new_dict[len(new_dict)] = {"cids": cids, "xcat": xcats}

    # now, where the cids are the same, group the xcats into one list
    new_dict2 = {}
    for nidx, ditem in new_dict.items():
        cids = ditem["cids"]
        xcats = ditem["xcat"]
        for nidx2, ditem2 in new_dict.items():
            if set(cids) == set(ditem2["cids"]):
                xcats = list(set(xcats + ditem2["xcat"]))
        new_dict2[len(new_dict2)] = {"cids": cids, "xcat": set(get_cid(xcats))}


sample_real_data_frame()
