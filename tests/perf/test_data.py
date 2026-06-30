import pandas as pd
import pytest

from macrosynergy.management.types import QuantamentalDataFrame
from tests.perf.data import (
    SCALE_TIERS, make_perf_qdf, qdf_for_tier, wide_ticker_frame,
    update_df_pieces, srr_panel,
)


def test_scale_tiers_defined():
    assert set(SCALE_TIERS) == {"tiny", "small", "medium", "large"}
    for tier in SCALE_TIERS.values():
        assert {"n_cids", "n_xcats", "n_days"} <= set(tier)


def test_make_perf_qdf_columns_and_dtype():
    df = make_perf_qdf(3, 4, 50)
    assert list(df.columns) == ["cid", "xcat", "real_date", "value"]
    assert df["cid"].dtype == object  # object by default (notebook's slow case)
    assert df["cid"].nunique() == 3 and df["xcat"].nunique() == 4


def test_make_perf_qdf_is_deterministic():
    a = make_perf_qdf(3, 4, 50, seed=7)
    b = make_perf_qdf(3, 4, 50, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_make_perf_qdf_categorical_variant():
    df = make_perf_qdf(3, 4, 50, categorical=True)
    assert isinstance(df, QuantamentalDataFrame)
    assert df["cid"].dtype.name == "category"


def test_qdf_for_tier_tiny_is_small_enough():
    df = qdf_for_tier("tiny")
    assert len(df) < 50_000


def test_wide_ticker_frame_shape():
    w = wide_ticker_frame(n_tickers=10, n_days=30)
    assert w.shape[1] == 10
    assert all("_" in str(c) for c in w.columns)
    assert isinstance(w.index, pd.DatetimeIndex)


def test_update_df_pieces_returns_base_and_list():
    base, pieces = update_df_pieces("tiny", n_pieces=4)
    assert isinstance(base, pd.DataFrame) and len(pieces) == 4
    assert all(set(["cid", "xcat", "real_date", "value"]) <= set(p.columns) for p in pieces)
    assert all(len(p) > 0 for p in pieces)


def test_update_df_pieces_more_pieces_than_xcats_are_nonempty():
    base, pieces = update_df_pieces("tiny", n_pieces=10)  # tiny has 3 xcats
    assert len(pieces) == 10
    assert all(len(p) > 0 for p in pieces)


def test_scale_tiers_row_count_ordering():
    def _rows(t):
        c = SCALE_TIERS[t]
        return c["n_cids"] * c["n_xcats"] * c["n_days"]
    assert _rows("tiny") < _rows("small") < _rows("medium") < _rows("large")


def test_srr_panel_has_signal_and_return_xcats():
    df = srr_panel(n_cids=4, n_dates=200, n_signals=2, n_returns=3)
    assert df["xcat"].nunique() == 5
