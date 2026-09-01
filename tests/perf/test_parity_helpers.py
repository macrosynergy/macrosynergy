"""Tests for the comparison helpers in parity.py."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.perf.parity import (
    assert_categorical_equal,
    assert_frame_parity,
    assert_qdf_equal,
)


class TestAll(unittest.TestCase):
    def test_frame_parity_passes_for_equal_with_nan(self):
        actual = pd.DataFrame({"x": [1.0, np.nan], "s": ["p", "q"]})
        assert_frame_parity(actual, actual.copy())

    def test_frame_parity_fails_on_value_diff(self):
        actual = pd.DataFrame({"x": [1.0, 2.0]})
        expected = pd.DataFrame({"x": [1.0, 2.5]})
        with self.assertRaises(AssertionError):
            assert_frame_parity(actual, expected)

    def test_frame_parity_fails_on_dtype_diff(self):
        actual = pd.DataFrame({"x": [1, 2]})
        expected = pd.DataFrame({"x": [1.0, 2.0]})
        with self.assertRaises(AssertionError):
            assert_frame_parity(actual, expected)

    def test_frame_parity_fails_on_column_name_diff(self):
        actual = pd.DataFrame({"x": [1.0]})
        expected = pd.DataFrame({"y": [1.0]})
        with self.assertRaises(AssertionError):
            assert_frame_parity(actual, expected)

    def test_qdf_equal_ignores_row_order(self):
        actual = pd.DataFrame(
            {
                "cid": ["A", "B"],
                "xcat": ["X", "Y"],
                "real_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "value": [1.0, 2.0],
            }
        )
        assert_qdf_equal(actual, actual.iloc[::-1].reset_index(drop=True))

    def test_categorical_equal_detects_order_diff(self):
        actual = pd.Categorical(["x_a", "y_b"], categories=["x_a", "y_b"], ordered=True)
        expected = pd.Categorical(
            ["x_a", "y_b"], categories=["y_b", "x_a"], ordered=True
        )
        with self.assertRaises(AssertionError):
            assert_categorical_equal(actual, expected)

    def test_categorical_equal_passes_for_identical(self):
        actual = pd.Categorical(["x_a", "y_b"], categories=["x_a", "y_b"], ordered=True)
        assert_categorical_equal(actual, actual.copy())


if __name__ == "__main__":
    unittest.main()
