import numpy as np
import pandas as pd
import pytest

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
