"""Output-parity guard for T2 split_ticker / ticker_df_to_qdf (default gate)."""

import pandas as pd

from macrosynergy.management.utils import ticker_df_to_qdf
from tests.perf.data import wide_ticker_frame
from tests.perf.parity import assert_qdf_equal, load_golden


def test_ticker_df_to_qdf_matches_golden():
    out = ticker_df_to_qdf(wide_ticker_frame(12, 60))
    assert_qdf_equal(pd.DataFrame(out), load_golden("ticker_df_to_qdf_tiny"))


def test_ticker_df_to_qdf_columns():
    out = ticker_df_to_qdf(wide_ticker_frame(6, 20))
    assert list(pd.DataFrame(out).columns) == ["real_date", "cid", "xcat", "value"]
