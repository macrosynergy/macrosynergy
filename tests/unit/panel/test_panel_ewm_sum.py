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


def _sparse_qdf():
    # AUD_GROWTH observed only on 2020-01-01 (value 10) and 2020-01-10 (value 0),
    # nothing in between -> tests interior zero-fill decay and leading NaN.
    rows = [
        ("AUD", "GROWTH", pd.Timestamp("2020-01-01"), 10.0),
        ("AUD", "GROWTH", pd.Timestamp("2020-01-10"), 0.0),
    ]
    return pd.DataFrame(rows, columns=["cid", "xcat", "real_date", "value"])


def test_leading_region_is_nan_then_present():
    df = _sparse_qdf()
    out = panel_ewm_sum(df, halflife=5, mask_leading=True)
    # First business day of output equals the first observation date, not earlier.
    assert out["real_date"].min() == pd.Timestamp("2020-01-01")
    # mask_leading=False still cannot precede the grid start (== first obs here).
    out2 = panel_ewm_sum(df, halflife=5, mask_leading=False)
    assert out2["real_date"].min() == pd.Timestamp("2020-01-01")


def test_zero_fill_decays_between_releases():
    df = _sparse_qdf()
    out = panel_ewm_sum(df, halflife=5).set_index("real_date")["value"]
    # On 2020-01-01 the sum is the first value itself.
    assert out.loc["2020-01-01"] == pytest.approx(10.0)
    # Business days 02..09 have zero input, so the sum decays geometrically.
    alpha_hl = 0.5 ** (1 / 5)
    bdays = pd.date_range("2020-01-01", "2020-01-09", freq="B")
    expected_09 = 10.0 * alpha_hl ** (len(bdays) - 1)
    assert out.loc["2020-01-09"] == pytest.approx(expected_09, rel=1e-9)
