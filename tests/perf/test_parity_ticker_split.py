"""
Self-consistency guards for the two format conversions and for ticker splitting.
`qdf_to_ticker_df` and `ticker_df_to_qdf` are inverses on a rectangular panel, and a
ticker rejoined from its parts must equal the ticker it came from.
"""

from __future__ import annotations

import unittest

import pandas as pd
from parameterized import parameterized

from macrosynergy.management.utils import (
    get_cid,
    get_xcat,
    qdf_to_ticker_df,
    split_ticker,
    ticker_df_to_qdf,
)
from tests.perf.panel_sizes import PANEL_SIZES
from tests.perf.parity import assert_qdf_equal


class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        self.panel_size = PANEL_SIZES["tiny"]
        self.qdf = self.panel_size.as_qdf()
        self.tickers = sorted(
            f"{cid}_{xcat}"
            for cid in self.panel_size.cids
            for xcat in self.panel_size.xcats
        )

    def test_qdf_survives_a_round_trip_through_the_wide_format(self):
        wide = qdf_to_ticker_df(self.panel_size.as_qdf_copy())
        assert_qdf_equal(pd.DataFrame(ticker_df_to_qdf(wide)), self.qdf)

    def test_wide_format_has_one_row_per_date_and_one_column_per_ticker(self):
        wide = qdf_to_ticker_df(self.panel_size.as_qdf_copy())
        self.assertEqual(wide.shape, self.panel_size.ticker_df_shape)
        self.assertEqual(sorted(wide.columns), self.tickers)
        self.assertEqual(wide.index.name, "real_date")

    def test_ticker_df_to_qdf_returns_the_standard_columns(self):
        wide = self.panel_size.as_ticker_df()
        self.assertEqual(
            list(pd.DataFrame(ticker_df_to_qdf(wide)).columns),
            ["real_date", "cid", "xcat", "value"],
        )

    def test_a_ticker_rejoined_from_its_parts_is_unchanged(self):
        rejoined = [f"{get_cid(ticker)}_{get_xcat(ticker)}" for ticker in self.tickers]
        self.assertEqual(rejoined, self.tickers)

    @parameterized.expand([("cid",), ("xcat",)])
    def test_split_ticker_agrees_with_the_dedicated_accessor(self, mode):
        accessor = {"cid": get_cid, "xcat": get_xcat}[mode]
        for ticker in self.tickers:
            self.assertEqual(split_ticker(ticker, mode), accessor(ticker))


if __name__ == "__main__":
    unittest.main()
