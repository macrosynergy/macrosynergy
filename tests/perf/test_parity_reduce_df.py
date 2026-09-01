"""
Self-consistency guards for `reduce_df`. Filtering a panel by every cid and xcat it
already contains must return that panel unchanged, and the terminal de-duplication must
remove only rows that repeat.
"""

from __future__ import annotations

import unittest

import pandas as pd

from macrosynergy.management.utils import reduce_df
from tests.perf.panel_sizes import PANEL_SIZES
from tests.perf.parity import assert_qdf_equal


class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        self.panel_size = PANEL_SIZES["tiny"]
        self.qdf = self.panel_size.as_qdf()

    def test_reducing_by_every_cid_and_xcat_is_a_no_op(self):
        reduced = reduce_df(
            self.panel_size.as_qdf_copy(),
            cids=self.panel_size.cids,
            xcats=self.panel_size.xcats,
        )
        assert_qdf_equal(pd.DataFrame(reduced), self.qdf)

    def test_reducing_without_filters_is_a_no_op(self):
        reduced = reduce_df(self.panel_size.as_qdf_copy())
        assert_qdf_equal(pd.DataFrame(reduced), self.qdf)

    def test_duplicate_rows_are_removed(self):
        duplicated = pd.concat([self.qdf, self.qdf.iloc[:100]], ignore_index=True)
        reduced = reduce_df(duplicated)
        self.assertEqual(len(reduced), len(self.qdf))
        assert_qdf_equal(pd.DataFrame(reduced), self.qdf)

    def test_reducing_to_one_xcat_keeps_its_rows_only(self):
        wanted = self.panel_size.xcats[:1]
        reduced = pd.DataFrame(
            reduce_df(self.panel_size.as_qdf_copy(), xcats=wanted)
        )
        self.assertEqual(sorted(set(reduced["xcat"].astype(str))), wanted)
        self.assertEqual(
            len(reduced), self.panel_size.cid_count * self.panel_size.date_count
        )


if __name__ == "__main__":
    unittest.main()
