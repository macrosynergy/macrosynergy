"""
Input contracts for the single-security index and portfolio calculations.
"""

import logging
from typing import Tuple

import pandas as pd

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


def _validate_weights_col(df: pd.DataFrame, weights_col: str) -> None:
    """
    Assert that *weights_col* is a supported weighting column present in *df*.

    Parameters
    ----------
    df : pd.DataFrame or QuantamentalDataFrame
        Constituents DataFrame expected to contain "weights_col".
    weights_col : str
        Name of the column holding the target weighting input. Must be one of
        {"membership", "raw_weight"}.

    Raises
    ------
    AssertionError
        If "weights_col" is not one of the supported options, is absent from "df",
        or - for "raw_weight" - holds non-numeric or negative values.
    """
    VALID_WEIGHTS_COLS = {"membership", "raw_weight"}
    assert isinstance(weights_col, str) and weights_col in VALID_WEIGHTS_COLS, (
        f"'weights_col' must be one of {sorted(VALID_WEIGHTS_COLS)}, got "
        f"'{weights_col}'. Use 'membership' for equal weighting, or 'raw_weight' "
        "for a custom weighting option."
    )
    assert weights_col in df.columns, (
        f"constituents DataFrame missing column '{weights_col}', required for "
        f"weights_col='{weights_col}'."
    )
    if weights_col != "membership":
        assert pd.api.types.is_numeric_dtype(df[weights_col]), (
            f"constituents['{weights_col}'] must be numeric, got dtype "
            f"'{df[weights_col].dtype}'."
        )
        negatives = df[weights_col].dropna() < 0
        assert not negatives.any(), (
            f"constituents['{weights_col}'] must be non-negative. "
            f"Found {int(negatives.sum())} negative value(s)."
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


# A weighting input can be provided at a coarser cadence than the rebalancing
# frequency it is meant to support. compute_daily_weights forward-fills over gaps,
# which would otherwise mask that staleness silently.


def weight_grid_coverage(
    df: pd.DataFrame,
    freq: str,
    weights_col: str = "value",
) -> Tuple[int, int]:
    """
    Periods of a given frequency holding a weight observation, against those spanned.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with a "real_date" column and the weighting input in
        `weights_col`.
    freq : str
        Pandas period alias of the cadence to test, e.g. "M" or "Q".
    weights_col : str
        Column holding the weighting input. Default is "value".

    Returns
    -------
    tuple of int
        Number of `freq` periods holding at least one observation, and the number of
        periods the observations span. The span is taken from the observed dates
        themselves, so a security set whose members simply did not exist early in the
        sample does not register as a gap.
    """
    dates = pd.DatetimeIndex(
        pd.to_datetime(df.loc[df[weights_col].notna(), "real_date"]).unique()
    ).sort_values()
    return (
        dates.to_period(freq).nunique(),
        pd.period_range(dates.min(), dates.max(), freq=freq).size,
    )


def assert_weight_grid(
    df: pd.DataFrame,
    freq: str,
    weights_col: str = "value",
) -> None:
    """
    Fail if the weighting input is staler than the rebalancing cadence.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe holding the weighting input, as for
        `weight_grid_coverage`.
    freq : str
        Pandas period alias of the rebalancing cadence the weights have to support.
    weights_col : str
        Column holding the weighting input. Default is "value".

    Raises
    ------
    AssertionError
        If any `freq` period spanned by the weighting input holds no observation.
    """
    observed, expected = weight_grid_coverage(df, freq, weights_col=weights_col)
    assert observed == expected, (
        f"Weighting input covers {observed} of {expected} '{freq}' periods: rebalancing "
        f"at '{freq}' would reset weights to a stale target. Provide the weighting "
        f"input at '{freq}' frequency or higher, or rebalance/reconstitute less often."
    )
