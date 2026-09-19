import numpy as np
import pandas as pd
import pytest

from tests.perf.parity import (
    assert_frame_parity, assert_qdf_equal, assert_categorical_equal,
)


def test_frame_parity_passes_for_equal_with_nan():
    a = pd.DataFrame({"x": [1.0, np.nan], "s": ["p", "q"]})
    b = a.copy()
    assert_frame_parity(a, b)  # no raise


def test_frame_parity_fails_on_value_diff():
    a = pd.DataFrame({"x": [1.0, 2.0]})
    b = pd.DataFrame({"x": [1.0, 2.5]})
    with pytest.raises(AssertionError):
        assert_frame_parity(a, b)


def test_qdf_equal_ignores_row_order():
    a = pd.DataFrame({"cid": ["A", "B"], "xcat": ["X", "Y"],
                      "real_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                      "value": [1.0, 2.0]})
    b = a.iloc[::-1].reset_index(drop=True)
    assert_qdf_equal(a, b)


def test_categorical_equal_detects_order_diff():
    a = pd.Categorical(["x_a", "y_b"], categories=["x_a", "y_b"], ordered=True)
    b = pd.Categorical(["x_a", "y_b"], categories=["y_b", "x_a"], ordered=True)
    with pytest.raises(AssertionError):
        assert_categorical_equal(a, b)


def test_save_and_load_golden_roundtrip(tmp_path, monkeypatch):
    import tests.perf.parity as parity
    monkeypatch.setattr(parity, "GOLDEN_DIR", tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})
    h = parity.save_golden("roundtrip", df)
    assert isinstance(h, str) and len(h) == 64
    pd.testing.assert_frame_equal(parity.load_golden("roundtrip"), df)
