# tests/unit/panel/test_panel_ewm_sum.py
import numpy as np
import pandas as pd
import pytest

from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel import panel_ewm_sum


def test_basic_ewm_sum_and_naming():
    # Dense daily (business-day) panel, no gaps.
    df = make_test_df(
        cids=["AUD", "CAD"], xcats=["GROWTH", "INFL"],
        start="2020-01-01", end="2020-06-30",
    )
    out = panel_ewm_sum(df, halflife=5)

    # New categories are named with the _{h}DXMS suffix.
    assert set(out["xcat"].unique()) == {
        "GROWTH_5DXMS", "INFL_5DXMS",
    }
    # Value-only standard columns, in order.
    assert list(out.columns) == ["cid", "xcat", "real_date", "value"]

    # Matches a hand-built reference for one series (already dense -> reindex is identity).
    ref = (
        df[(df["cid"] == "AUD") & (df["xcat"] == "GROWTH")]
        .set_index("real_date")["value"]
        .ewm(halflife=5).sum()
    )
    got = (
        out[(out["cid"] == "AUD") & (out["xcat"] == "GROWTH_5DXMS")]
        .set_index("real_date")["value"]
    )
    pd.testing.assert_series_equal(
        got.astype(float), ref.astype(float),
        check_names=False, check_freq=False,
    )
