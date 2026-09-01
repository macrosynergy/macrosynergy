"""
Comparison helpers for the parity tests, which check that the package's own output stays
self-consistent. Every comparison is exact for labels and dtypes and tolerant only for
floating-point values, so a parity failure names the column that moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.perf.panel_sizes import QUANTAMENTAL_INDEX_COLUMNS


def assert_frame_parity(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """
    Assert two DataFrames agree on columns, shape, dtypes and values.

    Row labels are ignored, so a frame that was reindexed but not reordered still
    matches. Numeric columns compare with a floating-point tolerance and treat NaN as
    equal to NaN; every other column compares exactly.

    Parameters
    ----------
    actual : pd.DataFrame
        The DataFrame produced by the code under test.
    expected : pd.DataFrame
        The DataFrame it should equal.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the column lists, shapes, a dtype or a column's values differ.
    """
    assert list(actual.columns) == list(expected.columns), (
        f"columns differ: {list(actual.columns)} != {list(expected.columns)}"
    )
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    for column in expected.columns:
        actual_column = actual[column].reset_index(drop=True)
        expected_column = expected[column].reset_index(drop=True)
        assert str(actual_column.dtype) == str(expected_column.dtype), (
            f"dtype for {column}: {actual_column.dtype} != {expected_column.dtype}"
        )
        if pd.api.types.is_numeric_dtype(expected_column):
            assert np.allclose(
                actual_column.to_numpy(), expected_column.to_numpy(), equal_nan=True
            ), f"values differ in {column}"
        else:
            assert actual_column.equals(expected_column), f"values differ in {column}"


def assert_qdf_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """
    Assert two quantamental DataFrames agree, ignoring row order.

    Parameters
    ----------
    actual : pd.DataFrame
        The DataFrame produced by the code under test.
    expected : pd.DataFrame
        The DataFrame it should equal.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the frames differ once both are sorted on cid, xcat and real_date.
    """
    sort_columns = list(QUANTAMENTAL_INDEX_COLUMNS)
    actual_sorted = (
        pd.DataFrame(actual).sort_values(sort_columns).reset_index(drop=True)
    )
    expected_sorted = (
        pd.DataFrame(expected).sort_values(sort_columns).reset_index(drop=True)
    )
    assert_frame_parity(actual_sorted, expected_sorted)


def assert_categorical_equal(
    actual: pd.Categorical, expected: pd.Categorical
) -> None:
    """
    Assert two categoricals agree on their category set, order flag and codes.

    Comparing codes rather than values catches a reordered category set, which changes
    how downstream code groups and sorts even when the labels read the same.

    Parameters
    ----------
    actual : pd.Categorical
        The categorical produced by the code under test.
    expected : pd.Categorical
        The categorical it should equal.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the categories, their order, the ordered flag or the codes differ.
    """
    assert list(actual.categories) == list(expected.categories), (
        "category set/order differs"
    )
    assert actual.ordered == expected.ordered, "ordered flag differs"
    assert np.array_equal(actual.codes, expected.codes), "codes differ"
