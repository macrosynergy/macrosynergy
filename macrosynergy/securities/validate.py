import logging
from datetime import timedelta, datetime, date
from typing import Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def _validate_frequency(freq: str, param_name: str) -> None:
    """
    Raise ValueError if *freq* is not a supported rebalancing/output frequency.

    Parameters
    ----------
    freq : str
        Frequency string to validate.
    param_name : str
        Name of the calling parameter, used in the error message.

    Raises
    ------
    ValueError
        If "freq" is not one of {"B", "W", "M", "Q", "Y"}.
    """

    VALID_FREQUENCIES = {"B", "W", "M", "Q", "Y"}
    if freq not in VALID_FREQUENCIES:
        raise ValueError(
            f"'{param_name}' must be one of {VALID_FREQUENCIES}, got '{freq}'."
        )


def _validate_constituents(df: pd.DataFrame) -> None:
    """
    Assert that *df* meets the minimum contract for a constituents DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or QuantamentalDataFrame
        DataFrame expected to have columns "cid", "real_date", and
        "membership" with binary (0/1) values and no duplicate
        (cid, real_date) pairs.

    Raises
    ------
    AssertionError
        If any required column is missing, "membership" contains values other
        than 0 or 1, or there are duplicate (cid, real_date) pairs.
    """
    required = {"cid", "real_date", "membership"}
    missing = required - set(df.columns)
    assert len(missing) == 0, (
        f"constituents DataFrame missing columns: {missing}. "
        f"Expected columns: {sorted(required)}."
    )
    assert df["membership"].isin([0, 1]).all(), (
        "constituents['membership'] must contain only 0 and 1. "
        f"Found values: {sorted(df['membership'].unique())}."
    )
    assert not df[["cid", "real_date"]].duplicated().any(), (
        "constituents has duplicate (cid, real_date) pairs. "
        "Each stock must have at most one row per date."
    )


def _validate_returns(df: pd.DataFrame) -> None:
    """
    Assert that *df* meets the minimum contract for a (single-xcat) returns DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or QuantamentalDataFrame
        DataFrame expected to have columns "cid", "real_date", "xcat",
        and "value" with no duplicate (cid, real_date) pairs.  Callers should
        filter to a single xcat before passing.

    Raises
    ------
    AssertionError
        If any required column is missing or there are duplicate (cid, real_date)
        pairs.
    """
    required = {"cid", "real_date", "xcat", "value"}
    missing = required - set(df.columns)
    assert len(missing) == 0, (
        f"returns DataFrame missing columns: {missing}. "
        f"Expected columns: {sorted(required)}."
    )
    assert not df[["cid", "real_date"]].duplicated().any(), (
        "returns has duplicate (cid, real_date) pairs. "
        "Filter to a single xcat before passing."
    )


def _validate_index_returns(df: pd.DataFrame) -> None:
    """
    Assert that *df* meets the minimum contract for an index-returns DataFrame.

    Parameters
    ----------
    df : pd.DataFrame or QuantamentalDataFrame
        DataFrame expected to have columns "real_date" and "value" with no
        duplicate "real_date" entries.

    Raises
    ------
    AssertionError
        If any required column is missing or "real_date" contains duplicates.
    """
    required = {"real_date", "value"}
    missing = required - set(df.columns)
    assert len(missing) == 0, (
        f"index returns DataFrame missing columns: {missing}. "
        f"Expected columns: {sorted(required)}."
    )
    assert (
        not df["real_date"].duplicated().any()
    ), "index returns has duplicate real_date entries."
