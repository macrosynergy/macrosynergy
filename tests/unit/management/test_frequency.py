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
