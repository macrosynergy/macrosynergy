# tests/unit/panel/test_panel_ewm_sum.py
import numpy as np
import pandas as pd
import pytest
from packaging import version

from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.panel import panel_ewm_sum, panel_calculator

_HAS_EWM_SUM = version.parse(pd.__version__) >= version.parse("1.4.0")


def _ewm_sum_reference(values, halflife):
    """Version-independent EWM sum (adjust=True): ``y_t = x_t + (1-alpha)*y_{t-1}`` with
    ``1-alpha = 0.5**(1/halflife)``. Equals pandas' native ``ewm().sum()`` and lets the
    numeric check run (exercising the fallback path) where that method is unavailable."""
    decay = 0.5 ** (1.0 / halflife)
    acc = 0.0
    out = []
    for v in values:
        acc = v + decay * acc
        out.append(acc)
    return out


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
    # Use a version-independent recurrence rather than native ``.ewm().sum()`` so this
    # numeric check also runs on old pandas, where it exercises the fallback path.
    ser = (
        df[(df["cid"] == "AUD") & (df["xcat"] == "GROWTH")]
        .set_index("real_date")["value"]
    )
    ref = pd.Series(_ewm_sum_reference(ser, 5), index=ser.index)
    got = (
        out[(out["cid"] == "AUD") & (out["xcat"] == "GROWTH_5DXMS")]
        .set_index("real_date")["value"]
    )
    pd.testing.assert_series_equal(
        got.astype(float), ref.astype(float),
        check_names=False, check_freq=False,
    )


def test_leading_and_trailing_regions_excluded():
    # AUD spans the full range; CAD starts later and ends earlier. Each series' output must
    # be bounded by its own first and last observation -- no leading/trailing padding leaks
    # in from the shared business-day grid.
    aud = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-03-31")
    cad = make_test_df(cids=["CAD"], xcats=["GROWTH"], start="2020-02-03", end="2020-02-28")
    df = pd.concat([aud, cad], ignore_index=True)

    out = panel_ewm_sum(df, halflife=5)
    for cid, sub in df.groupby("cid"):
        got = out[out["cid"] == cid]["real_date"]
        assert got.min() == sub["real_date"].min()
        assert got.max() == sub["real_date"].max()


def test_interior_business_day_gap_raises():
    # Dense daily-B series with a single interior business day removed -> data-quality gap.
    df = make_test_df(cids=["AUD"], xcats=["GROWTH"], start="2020-01-01", end="2020-02-28")
    drop = pd.Timestamp("2020-01-15")  # a Wednesday, strictly inside the observed span
    assert (df["real_date"] == drop).any()
    df = df[df["real_date"] != drop]
    with pytest.raises(ValueError):
        panel_ewm_sum(df, halflife=5)


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


@pytest.mark.skipif(
    not _HAS_EWM_SUM,
    reason="panel_calculator's `.ewm().sum()` needs pandas >= 1.4.0",
)
def test_matches_panel_calculator_on_dense_daily_panel():
    # Already-dense daily-B panel: reindex is identity, so panel_ewm_sum must equal the
    # panel_calculator EWM-sum on the shared region.
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

@pytest.mark.skipif(
    _HAS_EWM_SUM,
    reason="complement of the native-method test: runs only where `.ewm().sum()` is "
    "absent (the py37/pandas<1.4 floor)",
)
def test_matches_panel_calculator_on_dense_daily_panel_python37():
    # py37/old-pandas counterpart to the test above. panel_calculator cannot express
    # `.ewm().sum()` below pandas 1.4.0, so instead of comparing against it we compare
    # panel_ewm_sum against the version-independent recurrence oracle applied per series.
    cids = ["AUD", "CAD"]
    df = make_test_df(cids=cids, xcats=["GROWTH"], start="2020-01-01", end="2020-06-30")

    fast = panel_ewm_sum(df, halflife=5)
    fast_i = fast.set_index(["cid", "real_date"])["value"].sort_index()

    ref_parts = []
    for cid in cids:
        ser = df[df["cid"] == cid].set_index("real_date")["value"].sort_index()
        r = pd.Series(_ewm_sum_reference(ser, 5), index=ser.index)
        r.index = pd.MultiIndex.from_product(
            [[cid], r.index], names=["cid", "real_date"]
        )
        ref_parts.append(r)
    ref_i = pd.concat(ref_parts).sort_index()

    common = fast_i.index.intersection(ref_i.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(
        fast_i.loc[common].astype(float),
        ref_i.loc[common].astype(float),
        check_names=False,
    )


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

    # Post-window values must reflect that blacklisted dates contributed zero to the sum
    # (not their real values). Build a reference by zeroing the blacklisted values in the
    # input and running without a blacklist -- the panel stays dense, so it does not raise
    # -- and it must agree with the blacklisted run once the window has passed, while a
    # plain run (which keeps the real, non-zero values through the window) diverges.
    in_window = (
        (df["cid"] == "AUD")
        & (df["real_date"] >= "2020-03-01")
        & (df["real_date"] <= "2020-04-30")
    )
    df_zeroed = df.copy()
    df_zeroed.loc[in_window, "value"] = 0.0

    ref = panel_ewm_sum(df_zeroed, halflife=5).set_index("real_date")["value"]
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
