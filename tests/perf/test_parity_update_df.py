"""
Self-consistency guards for `update_df`. Adding rows a panel already holds must leave it
unchanged, and adding rows that overlap it must keep one row per key, take the added
value, and leave the caller's frame alone.
"""

from __future__ import annotations

import unittest

import pandas as pd

from macrosynergy.management.utils import update_df
from tests.perf.panel_sizes import PANEL_SIZES, QUANTAMENTAL_INDEX_COLUMNS
from tests.perf.parity import assert_qdf_equal


class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        self.panel_size = PANEL_SIZES["tiny"]
        self.qdf = self.panel_size.as_qdf()
        self.slice = self.qdf.iloc[: len(self.qdf) // 3].copy()
        self.sort_columns = list(QUANTAMENTAL_INDEX_COLUMNS)

    def test_updating_with_a_slice_of_itself_is_a_no_op(self):
        updated = update_df(self.panel_size.as_qdf_copy(), self.slice)
        assert_qdf_equal(pd.DataFrame(updated), self.qdf)

    def test_updating_with_the_whole_frame_is_a_no_op(self):
        updated = update_df(
            self.panel_size.as_qdf_copy(), self.panel_size.as_qdf_copy()
        )
        assert_qdf_equal(pd.DataFrame(updated), self.qdf)

    def test_an_overlapping_update_keeps_one_row_per_key(self):
        bumped = self.slice.copy()
        bumped["value"] = bumped["value"] + 100.0
        updated = pd.DataFrame(update_df(self.panel_size.as_qdf_copy(), bumped))
        self.assertEqual(len(updated), len(self.qdf))
        self.assertFalse(updated.duplicated(subset=self.sort_columns).any())

    def test_an_overlapping_update_takes_the_added_value(self):
        bumped = self.slice.copy()
        bumped["value"] = bumped["value"] + 100.0
        updated = pd.DataFrame(update_df(self.panel_size.as_qdf_copy(), bumped))
        merged = updated.merge(
            bumped, on=self.sort_columns, suffixes=("_updated", "_added")
        )
        self.assertEqual(len(merged), len(bumped))
        pd.testing.assert_series_equal(
            merged["value_updated"], merged["value_added"], check_names=False
        )

    def test_the_result_is_sorted_on_the_index_columns(self):
        updated = pd.DataFrame(update_df(self.panel_size.as_qdf_copy(), self.slice))
        keys = updated[self.sort_columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            keys, keys.sort_values(self.sort_columns).reset_index(drop=True)
        )

    def test_the_caller_frame_is_not_modified(self):
        original = self.panel_size.as_qdf_copy()
        snapshot = original.copy(deep=True)
        update_df(original, self.slice)
        pd.testing.assert_frame_equal(original, snapshot)


if __name__ == "__main__":
    unittest.main()
