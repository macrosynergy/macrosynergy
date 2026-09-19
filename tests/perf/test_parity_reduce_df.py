"""Output-parity guard for T3 reduce_df (default gate)."""

import pandas as pd

from macrosynergy.management.utils import reduce_df
from tests.perf.data import qdf_for_tier
from tests.perf.parity import assert_qdf_equal, load_golden


def test_reduce_df_dedup_matches_golden():
    clean = qdf_for_tier("tiny")
    dupd = pd.concat([clean, clean.iloc[:100]], ignore_index=True)
    out = reduce_df(dupd)
    # dedup must remove the 100 exact-duplicate rows
    assert len(out) == len(clean)
    assert_qdf_equal(pd.DataFrame(out), load_golden("reduce_df_dedup_tiny"))


def test_reduce_df_no_spurious_row_drop_on_clean_panel():
    qdf = qdf_for_tier("tiny")
    out = reduce_df(qdf)
    # clean panel has no full-row duplicates -> reduce_df must not drop rows
    assert len(out) == len(qdf.drop_duplicates())
