# tests/unit/panel/test_panel_ewm_sum.py
import numpy as np
import pandas as pd
import pytest

from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.panel import panel_ewm_sum, panel_calculator


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
    # Value-only standard columns, in canonical QDF order.
    assert list(out.columns) == ["real_date", "cid", "xcat", "value"]

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


def test_multiple_halflives():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    out = panel_ewm_sum(df, halflife=[3, 5])
    assert set(out["xcat"].unique()) == {"GROWTH_3DXMS", "GROWTH_5DXMS"}


def test_postfix_override_scalar():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    out = panel_ewm_sum(df, halflife=5, postfix="EWMSUM")
    assert set(out["xcat"].unique()) == {"GROWTH_EWMSUM"}


def test_postfix_string_with_list_halflife_raises():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    with pytest.raises(AssertionError):
        panel_ewm_sum(df, halflife=[3, 5], postfix="EWMSUM")


def test_matches_panel_calculator_on_dense_daily_panel():
    # Already-dense daily-B panel: reindex is identity, fillna a no-op in the interior,
    # so panel_ewm_sum must equal the panel_calculator EWM-sum on the shared region.
    cids = ["AUD", "CAD"]
    df = make_test_df(cids=cids, xcats=["GROWTH"], start="2020-01-01", end="2020-06-30")

    fast = panel_ewm_sum(df, halflife=5)
    ref = panel_calculator(
        df, calcs=["GROWTH_5DXMS = GROWTH.ewm(halflife=5).sum()"], cids=cids
    )

    fast_i = fast.set_index(["cid", "xcat", "real_date"])["value"].sort_index()
    ref_i = ref.set_index(["cid", "xcat", "real_date"])["value"].sort_index()
    # Compare on the intersection of indices (both start at first valid).
    common = fast_i.index.intersection(ref_i.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(
        fast_i.loc[common].astype(float),
        ref_i.loc[common].astype(float),
        check_names=False,
    )


def test_diverges_from_per_event_calc_on_sparse_panel():
    # Sparse panel: panel_calculator decays per release event; panel_ewm_sum decays per
    # business day. They must differ, and panel_ewm_sum must match a dense-grid reference.
    rows = [
        ("AUD", "GROWTH", pd.Timestamp("2020-01-01"), 5.0),
        ("AUD", "GROWTH", pd.Timestamp("2020-02-03"), 5.0),
        ("AUD", "GROWTH", pd.Timestamp("2020-03-02"), 5.0),
    ]
    df = pd.DataFrame(rows, columns=["cid", "xcat", "real_date", "value"])

    fast = panel_ewm_sum(df, halflife=5).set_index("real_date")["value"]
    per_event = panel_calculator(
        df, calcs=["GROWTH_5DXMS = GROWTH.ewm(halflife=5).sum()"], cids=["AUD"]
    ).set_index("real_date")["value"]

    # On the last release date the two definitions disagree.
    last = pd.Timestamp("2020-03-02")
    assert fast.loc[last] != pytest.approx(per_event.loc[last])


def test_cids_and_xcats_subsetting():
    df = make_test_df(
        cids=["AUD", "CAD", "GBP"], xcats=["GROWTH", "INFL"],
        start="2020-01-01", end="2020-03-31",
    )
    out = panel_ewm_sum(df, xcats=["GROWTH"], cids=["AUD", "CAD"], halflife=5)
    assert set(out["xcat"].unique()) == {"GROWTH_5DXMS"}
    assert set(out["cid"].unique()) == {"AUD", "CAD"}


def test_blacklist_excludes_range():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-06-30")
    black = {"AUD": ["2020-03-01", "2020-04-30"]}
    out = panel_ewm_sum(df, halflife=5, blacklist=black)
    masked = out[(out["cid"] == "AUD") &
                 (out["real_date"] >= "2020-03-01") &
                 (out["real_date"] <= "2020-04-30")]
    assert masked.empty

    # Post-window values must reflect that the blacklisted input rows never entered
    # the computation. Build a reference where those rows are dropped from the input
    # directly (instead of blacklisted after the fact): the two must agree once the
    # window has passed, while a plain (non-blacklisted) run -- which keeps the real,
    # non-zero AUD_GROWTH values through the window -- must diverge from both.
    in_window = (
        (df["cid"] == "AUD")
        & (df["real_date"] >= "2020-03-01")
        & (df["real_date"] <= "2020-04-30")
    )
    df_dropped = df[~in_window]

    ref = panel_ewm_sum(df_dropped, halflife=5).set_index("real_date")["value"]
    plain = panel_ewm_sum(df, halflife=5).set_index("real_date")["value"]
    got = out.set_index("real_date")["value"]

    post_window = pd.Timestamp("2020-05-01")
    assert got.loc[post_window] == pytest.approx(ref.loc[post_window], rel=1e-9)
    assert got.loc[post_window] != pytest.approx(plain.loc[post_window])


def test_empty_selection_returns_empty():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    out = panel_ewm_sum(df, xcats=["NOTHERE"])
    assert list(out.columns) == ["real_date", "cid", "xcat", "value"]
    assert out.empty


def test_explicit_nan_in_value_raises():
    # A standardised panel is expected to be dense apart from blacklisted ranges, so an
    # explicit NaN in `value` is a data-quality signal, not a gap to zero-fill.
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    df.loc[df.index[5], "value"] = np.nan
    with pytest.raises(ValueError):
        panel_ewm_sum(df, halflife=5)


def test_all_nan_series_raises():
    # An all-NaN series must not silently vanish (or crash); it raises like any other
    # explicit-NaN input.
    rows = [
        ("AUD", "GROWTH", pd.Timestamp("2020-01-01"), np.nan),
        ("AUD", "GROWTH", pd.Timestamp("2020-01-10"), np.nan),
    ]
    df = pd.DataFrame(rows, columns=["cid", "xcat", "real_date", "value"])
    with pytest.raises(ValueError):
        panel_ewm_sum(df, halflife=5)


def test_categorical_round_trip():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    qdf = QuantamentalDataFrame(df)  # categorical cid/xcat
    out = panel_ewm_sum(qdf, halflife=5)
    assert isinstance(out["cid"].dtype, pd.CategoricalDtype)
    assert isinstance(out["xcat"].dtype, pd.CategoricalDtype)


def test_single_cid_single_xcat():
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-02-28")
    out = panel_ewm_sum(df, halflife=3)
    assert set(out["xcat"].unique()) == {"GROWTH_3DXMS"}
    assert not out.empty
