"""
Self-consistency guards for the ticker-series targets. The ticker column a
`QuantamentalDataFrame` derives must agree with the cid and xcat columns it was derived
from, whichever dtype the frame carries.
"""

from __future__ import annotations

import unittest

import pandas as pd
from parameterized import parameterized

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.types.qdf.methods import _get_tickers_series
from tests.perf.panel_sizes import PANEL_SIZES
from tests.perf.parity import assert_categorical_equal


class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        self.panel_size = PANEL_SIZES["tiny"]
        self.qdf = self.panel_size.as_qdf()
        self.tickers = [
            f"{cid}_{xcat}"
            for cid, xcat in zip(
                self.qdf["cid"].astype(str), self.qdf["xcat"].astype(str)
            )
        ]

    def test_tickers_series_matches_the_columns_it_derives_from(self):
        series = _get_tickers_series(self.panel_size.as_qdf(categorical=True))
        self.assertEqual([str(value) for value in series], self.tickers)

    def test_tickers_series_categories_are_the_distinct_tickers(self):
        series = _get_tickers_series(self.panel_size.as_qdf(categorical=True))
        self.assertEqual(
            sorted(str(category) for category in series.categories),
            sorted(set(self.tickers)),
        )
        self.assertTrue(series.ordered)

    def test_tickers_series_is_reproducible(self):
        first = _get_tickers_series(self.panel_size.as_qdf(categorical=True))
        second = _get_tickers_series(
            QuantamentalDataFrame(self.panel_size.as_qdf_copy(), categorical=True)
        )
        assert_categorical_equal(pd.Categorical(first), pd.Categorical(second))

    @parameterized.expand([("object", False), ("categorical", True)])
    def test_add_ticker_column_matches_the_columns_it_derives_from(
        self, _name, categorical
    ):
        df = QuantamentalDataFrame(
            self.panel_size.as_qdf_copy(), categorical=categorical
        ).add_ticker_column()
        self.assertIn("ticker", df.columns)
        self.assertEqual([str(value) for value in df["ticker"]], self.tickers)

    def test_reduce_df_by_ticker_keeps_exactly_the_tickers_asked_for(self):
        df = QuantamentalDataFrame(self.panel_size.as_qdf_copy(), categorical=True)
        wanted = sorted(set(self.tickers))[:5]
        reduced = df.reduce_df_by_ticker(tickers=wanted)
        found = sorted(
            {
                f"{cid}_{xcat}"
                for cid, xcat in zip(
                    reduced["cid"].astype(str), reduced["xcat"].astype(str)
                )
            }
        )
        self.assertEqual(found, wanted)
        self.assertEqual(len(reduced), len(wanted) * self.panel_size.date_count)


if __name__ == "__main__":
    unittest.main()
