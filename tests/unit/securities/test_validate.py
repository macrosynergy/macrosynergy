import unittest
import numpy as np
import pandas as pd
from typing import List

from macrosynergy.securities.validate import (
    _validate_frequency,
    _validate_constituents,
    _validate_returns,
    _validate_index_returns,
)


def _make_constituents_df(
    cids: List[str] = None,
    start: str = "2020-01-02",
    end: str = "2020-03-31",
) -> pd.DataFrame:
    if cids is None:
        cids = ["AAA", "BBB"]
    bdays = pd.bdate_range(start, end)
    rows = [
        {"cid": cid, "real_date": dt, "membership": 1}
        for cid in cids
        for dt in bdays
    ]
    return pd.DataFrame(rows)


def _make_returns_df(
    cids: List[str] = None,
    start: str = "2020-01-02",
    end: str = "2020-03-31",
    xcat: str = "EQXR",
) -> pd.DataFrame:
    if cids is None:
        cids = ["AAA", "BBB"]
    bdays = pd.bdate_range(start, end)
    rows = [
        {"cid": cid, "real_date": dt, "xcat": xcat, "value": 0.0}
        for cid in cids
        for dt in bdays
    ]
    return pd.DataFrame(rows)


class TestValidateFrequency(unittest.TestCase):
    def test_valid_frequencies_pass(self):
        for freq in ["B", "W", "M", "Q", "Y"]:
            _validate_frequency(freq, "freq")

    def test_invalid_frequency_raises(self):
        for bad in ["D", "A", "X", "m", "", "daily", "monthly"]:
            with self.assertRaises(ValueError):
                _validate_frequency(bad, "my_param")

    def test_error_message_includes_param_name(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_frequency("D", "rebalance_freq")
        self.assertIn("rebalance_freq", str(ctx.exception))


class TestValidateConstituents(unittest.TestCase):
    def setUp(self):
        self.df = _make_constituents_df()

    def test_valid_df_passes(self):
        _validate_constituents(self.df)

    def test_missing_column_raises(self):
        for col in ["cid", "real_date", "membership"]:
            with self.assertRaises(AssertionError):
                _validate_constituents(self.df.drop(columns=[col]))

    def test_non_binary_membership_raises(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], "membership"] = 2
        with self.assertRaises(AssertionError):
            _validate_constituents(bad)

    def test_float_membership_value_raises(self):
        bad = self.df.copy()
        bad["membership"] = bad["membership"].astype(float)
        bad.loc[bad.index[0], "membership"] = 0.5
        with self.assertRaises(AssertionError):
            _validate_constituents(bad)

    def test_duplicate_cid_date_raises(self):
        dup = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            _validate_constituents(dup)

    def test_zero_membership_is_valid(self):
        df = self.df.copy()
        df["membership"] = 0
        _validate_constituents(df)


class TestValidateReturns(unittest.TestCase):
    def setUp(self):
        self.df = _make_returns_df()

    def test_valid_df_passes(self):
        _validate_returns(self.df)

    def test_missing_column_raises(self):
        for col in ["cid", "real_date", "xcat", "value"]:
            with self.assertRaises(AssertionError):
                _validate_returns(self.df.drop(columns=[col]))

    def test_duplicate_cid_date_raises(self):
        dup = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            _validate_returns(dup)


class TestValidateIndexReturns(unittest.TestCase):
    def setUp(self):
        bdays = pd.bdate_range("2020-01-02", "2020-03-31")
        self.df = pd.DataFrame(
            {"real_date": bdays, "value": np.zeros(len(bdays))}
        )

    def test_valid_df_passes(self):
        _validate_index_returns(self.df)

    def test_missing_column_raises(self):
        for col in ["real_date", "value"]:
            with self.assertRaises(AssertionError):
                _validate_index_returns(self.df.drop(columns=[col]))

    def test_duplicate_dates_raises(self):
        dup = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            _validate_index_returns(dup)


if __name__ == "__main__":
    unittest.main()
