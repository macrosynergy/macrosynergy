# tests/unit/management/test_frequency.py
import pandas as pd
import pytest

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

    # Early observations are quarterly.
    assert (labels.iloc[:6] == "Q").all()
    # Late observations are monthly (allow ~1-2 releases of lag at the break).
    assert (labels.iloc[-4:] == "M").all()


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
