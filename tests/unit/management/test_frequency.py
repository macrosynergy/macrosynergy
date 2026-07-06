# tests/unit/management/test_frequency.py
import pandas as pd

from macrosynergy.management.utils import infer_release_frequency


def _eop_series(dates):
    d = pd.to_datetime(dates)
    return pd.Series(d, index=range(len(d)))


def test_pure_monthly_and_quarterly():
    monthly = _eop_series(pd.date_range("2020-01-31", periods=12, freq="ME"))
    quarterly = _eop_series(pd.date_range("2020-03-31", periods=8, freq="QE"))
    assert (infer_release_frequency(monthly) == "M").all()
    assert (infer_release_frequency(quarterly) == "Q").all()


def test_quarterly_to_monthly_break():
    # AUD-CPI-like: 8 quarterly eops then 12 monthly eops in one series.
    q = pd.date_range("2015-03-31", periods=8, freq="QE")
    m = pd.date_range(q[-1] + pd.offsets.MonthEnd(1), periods=12, freq="ME")
    s = _eop_series(list(q) + list(m))
    labels = infer_release_frequency(s, window=3)

    # Early observations (the quarterly era, indices 0-7) are quarterly.
    assert (labels.iloc[:8] == "Q").all()
    # Measured lag: the first monthly eop (index 8) still reads "Q" because the
    # rolling window is still dominated by quarterly spacing; the label settles
    # to "M" by index 9 -- one release after the break. Bound this tightly so a
    # regression that pushes the lag out toward ~6 releases fails here.
    assert (labels.iloc[9:] == "M").all()


def test_one_off_irregular_gap_does_not_flip():
    # Monthly cadence with a single delayed print (a ~2-month gap once).
    base = list(pd.date_range("2020-01-31", periods=5, freq="ME"))
    base += [base[-1] + pd.offsets.MonthEnd(2)]           # one skipped month
    base += list(pd.date_range(base[-1] + pd.offsets.MonthEnd(1), periods=5, freq="ME"))
    labels = infer_release_frequency(_eop_series(base), window=3)
    # Rolling median absorbs the single 2-month gap -> stays monthly throughout.
    assert (labels == "M").all()


def test_revisions_share_eop_frequency():
    # Two observations (revisions) with identical eop inherit the same frequency.
    dates = list(pd.date_range("2020-01-31", periods=6, freq="ME"))
    eop = _eop_series(dates + [dates[-1]])                 # a revision of the last eop
    labels = infer_release_frequency(eop)
    assert labels.iloc[-1] == labels.iloc[-2] == "M"


def test_snap_boundaries_weekly_monthly_quarterly():
    # Weekly-ish (7d) -> W; ~30d -> M; ~91d -> Q.
    weekly = _eop_series(pd.date_range("2020-01-03", periods=10, freq="W-FRI"))
    assert (infer_release_frequency(weekly) == "W").all()

    # A ~45-day cadence sits between M (30.4) and Q (91.3); in log space it is nearer M.
    mid = _eop_series(pd.to_datetime(
        ["2020-01-31", "2020-03-16", "2020-04-30", "2020-06-14"]
    ))
    assert (infer_release_frequency(mid) == "M").all()


def test_no_gap_fallback_is_order_independent():
    # One distinct eop => no measurable gap. The fallback must not depend on the order
    # freqs is passed in, and must pick the coarsest candidate (here "A"), not freqs[0].
    single = _eop_series(pd.to_datetime(["2020-01-31"]))
    assert infer_release_frequency(single).iloc[0] == "A"
    assert (
        infer_release_frequency(single, freqs=("D", "W", "M", "Q", "A")).iloc[0]
        == infer_release_frequency(single, freqs=("A", "Q", "M", "W", "D")).iloc[0]
        == "A"
    )
