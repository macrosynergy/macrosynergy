"""
Module hosting custom types and meta-classes for use across the package.
"""

from typing import List, Optional, Any, Dict, Iterable, Mapping, Union
import pandas as pd
import warnings

from .base import QuantamentalDataFrameBase


def change_column_format(
    df: QuantamentalDataFrameBase,
    cols: List[str],
    dtype: Any,
) -> QuantamentalDataFrameBase:
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a pandas DataFrame.")

    if not isinstance(cols, list) or not all([isinstance(col, str) for col in cols]):
        raise TypeError("`cols` must be a list of strings.")

    for col in cols:
        curr_type = df[col].dtype
        try:
            df[col] = df[col].astype(dtype)
        except:
            warnings.warn(
                f"Could not convert column {col} to {dtype} from {curr_type}."
            )

    return df


def apply_blacklist(
    df: QuantamentalDataFrameBase,
    blacklist: Mapping[str, Iterable[Union[str, pd.Timestamp]]],
) -> List[str]:
    """
    Apply a blacklist to a list of `cids` and `xcats`.
    """
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a pandas DataFrame.")

    if not isinstance(blacklist, dict):
        raise TypeError("`blacklist` must be a dictionary.")

    if not all([isinstance(k, str) for k in blacklist.keys()]):
        raise TypeError("Keys of `blacklist` must be strings.")

    if not all([isinstance(v, Iterable) for v in blacklist.values()]):
        raise TypeError("Values of `blacklist` must be iterables.")

    if not all([isinstance(v, (str, pd.Timestamp)) for v in blacklist.values()]):
        raise TypeError("Values of `blacklist` must be strings or pandas Timestamps.")

    blist = blacklist.copy()

    for key, value in blist.items():
        df = df[
            ~(
                (df["cid"] == key[:3])
                & (df["real_date"] >= value[0])
                & (df["real_date"] <= value[1])
            )
        ]

    return df.reset_index(drop=True)


def reduce_df(
    df: QuantamentalDataFrameBase,
    cids: Optional[List[str]] = None,
    xcats: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: dict = None,
    out_all: bool = False,
    intersect: bool = False,
) -> QuantamentalDataFrameBase:
    # Filter DataFrame by xcats and cids and notify about missing xcats and cids.
    """
    Filter DataFrame by `cids`, `xcats`, `tickers`, and `start` & `end` dates.
    """
    if isinstance(cids, str):
        cids = [cids]
    if isinstance(xcats, str):
        xcats = [xcats]

    if start is not None:
        df = df.loc[df["real_date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df["real_date"] <= pd.to_datetime(end)]

    if blacklist is not None:
        df = apply_blacklist(df, blacklist)

    if cids is None:
        cids = sorted(df["cid"].unique())
    else:
        cids_in_df = df["cid"].unique()
        cids = sorted(c for c in cids if c in cids_in_df)

    if xcats is None:
        xcats = sorted(df["xcat"].unique())
    else:
        xcats_in_df = df["xcat"].unique()
        xcats = sorted(x for x in xcats if x in xcats_in_df)

    if intersect:
        cids_in_df = set.intersection(
            *(set(df[df["xcat"] == xcat]["cid"].unique()) for xcat in xcats)
        )
    else:
        cids_in_df = df["cid"].unique()
        cids = sorted(c for c in cids if c in cids_in_df)

    df = df[df["xcat"].isin(xcats)]
    df = df[df["cid"].isin(cids)]

    xcats_found = sorted(set(df["xcat"].unique()))
    cids_found = sorted(set(df["cid"].unique()))
    df = df.drop_duplicates().reset_index(drop=True)
    if out_all:
        return df, xcats_found, cids_found
    else:
        return df


def update_df(): ...


def qdf_to_wide_df(
    df: QuantamentalDataFrameBase,
    value_column: str = "value",
) -> pd.DataFrame:
    """
    Pivot the DataFrame to a wide format with memory efficiency.
    """
    # Ensure inputs are of the correct type and exist in the DataFrame
    if not isinstance(df, QuantamentalDataFrameBase):
        raise TypeError("`df` must be a QuantamentalDataFrame.")
    if not isinstance(value_column, str):
        raise TypeError("`value_column` must be a string.")
    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in DataFrame.")

    # Leverage categorical data to avoid extensive data copying
    cid_labels = df["cid"].cat.categories[df["cid"].cat.codes]
    xcat_labels = df["xcat"].cat.categories[df["xcat"].cat.codes]

    # Create the 'ticker' column directly without separate temporary structures
    df["ticker"] = pd.Categorical(
        [f"{cid}_{xcat}" for cid, xcat in zip(cid_labels, xcat_labels)],
        categories=pd.Categorical(
            [f"{cid}_{xcat}" for cid, xcat in zip(cid_labels, xcat_labels)]
        ).categories,
    )

    # Perform the pivot directly within the assignment to reduce memory footprint
    return df.pivot(
        index="real_date", columns="ticker", values=value_column
    ).rename_axis(None, axis=1)
