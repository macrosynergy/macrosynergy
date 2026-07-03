import numpy as np
import pandas as pd
import pytest

from macrosynergy.management.constants import ANNUALIZATION_FACTORS
from macrosynergy.panel import annualize_by_release_frequency


def _qdf_with_eop(cid, xcat, real_dates, eops, values):
    return pd.DataFrame({
        "cid": cid, "xcat": xcat,
        "real_date": pd.to_datetime(real_dates),
        "eop": pd.to_datetime(eops),
        "value": values,
    })


def test_pure_monthly_weight_matches_static():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    df = _qdf_with_eop("AUD", "CPIH", dates, dates, [1.0] * 12)
    out = annualize_by_release_frequency(df, postfix="A")
    assert set(out["xcat"].unique()) == {"CPIHA"}
    # Monthly -> value * sqrt(1/12).
    assert np.allclose(out["value"].to_numpy(), 1.0 * np.sqrt(1 / 12))


def test_missing_eop_raises():
    df = pd.DataFrame({
        "cid": ["AUD"], "xcat": ["CPIH"],
        "real_date": pd.to_datetime(["2020-01-31"]), "value": [1.0],
    })
    with pytest.raises(ValueError):
        annualize_by_release_frequency(df)


def test_pure_quarterly_weight():
    dates = pd.date_range("2015-03-31", periods=8, freq="QE")
    df = _qdf_with_eop("AUD", "CPIH", dates, dates, [1.0] * 8)
    out = annualize_by_release_frequency(df)
    # Quarterly -> value * sqrt(1/4) = 0.5.
    assert np.allclose(out["value"].to_numpy(), 0.5)


def test_break_transitions_weight():
    q = pd.date_range("2015-03-31", periods=8, freq="QE")
    m = pd.date_range(q[-1] + pd.offsets.MonthEnd(1), periods=12, freq="ME")
    dates = list(q) + list(m)
    df = _qdf_with_eop("AUD", "CPIH", dates, dates, [1.0] * len(dates))
    out = annualize_by_release_frequency(df).sort_values("real_date")
    vals = out["value"].to_numpy()
    # Early (quarterly) weight 0.5; late (monthly) weight sqrt(1/12).
    assert np.isclose(vals[0], 0.5)
    assert np.isclose(vals[-1], np.sqrt(1 / 12))


def test_uses_annualization_factors_constant():
    # Guards against re-hard-coding 4/12: quarterly weight must equal the constant.
    dates = pd.date_range("2015-03-31", periods=8, freq="QE")
    df = _qdf_with_eop("AUD", "CPIH", dates, dates, [1.0] * 8)
    out = annualize_by_release_frequency(df)
    assert np.allclose(out["value"].to_numpy(), np.sqrt(1 / ANNUALIZATION_FACTORS["Q"]))


def test_empty_selection_returns_empty():
    # A cids/xcats filter matching nothing must not crash on the from_long_df path
    # (an empty frame there raises "Input DataFrame is empty.").
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    df = _qdf_with_eop("AUD", "CPIH", dates, dates, [1.0] * 12)
    out = annualize_by_release_frequency(df, cids=["USD"])
    assert list(out.columns) == ["cid", "xcat", "real_date", "value"]
    assert out.empty
