"""Parity helpers: byte-identical comparison for QDFs, frames, and categoricals + golden I/O."""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pandas as pd

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"
_SORT = ["cid", "xcat", "real_date"]


def assert_frame_parity(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert list(actual.columns) == list(expected.columns), (
        f"columns differ: {list(actual.columns)} != {list(expected.columns)}"
    )
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    for col in expected.columns:
        a, e = actual[col], expected[col]
        assert str(a.dtype) == str(e.dtype), f"dtype for {col}: {a.dtype} != {e.dtype}"
        if pd.api.types.is_numeric_dtype(e):
            assert np.allclose(a.to_numpy(), e.to_numpy(), equal_nan=True), f"values differ in {col}"
        else:
            assert a.reset_index(drop=True).equals(e.reset_index(drop=True)), f"values differ in {col}"


def assert_qdf_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    a = pd.DataFrame(actual).sort_values(_SORT).reset_index(drop=True)
    e = pd.DataFrame(expected).sort_values(_SORT).reset_index(drop=True)
    assert_frame_parity(a, e)


def assert_categorical_equal(actual: pd.Categorical, expected: pd.Categorical) -> None:
    assert list(actual.categories) == list(expected.categories), "category set/order differs"
    assert actual.ordered == expected.ordered, "ordered flag differs"
    assert np.array_equal(actual.codes, expected.codes), "codes differ"


def save_golden(name: str, df: pd.DataFrame) -> str:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_golden(name: str) -> pd.DataFrame:
    return pd.read_parquet(GOLDEN_DIR / f"{name}.parquet")
