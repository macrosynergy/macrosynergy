"""Output-parity guard for T1 update_df (default gate)."""

import pandas as pd

from macrosynergy.management.utils import update_df
from tests.perf.data import update_df_pieces
from tests.perf.parity import assert_qdf_equal, load_golden


def test_update_df_loop_matches_golden():
    base, pieces = update_df_pieces("tiny", n_pieces=3)
    acc = base
    for p in pieces:
        acc = update_df(acc, p)
    expected = load_golden("update_df_loop_tiny")
    assert_qdf_equal(pd.DataFrame(acc), expected)


def test_update_df_invariants_last_wins_and_sorted():
    base, pieces = update_df_pieces("tiny", n_pieces=2)
    out = update_df(base, pieces[0])
    # no duplicate (cid, xcat, real_date) keys
    assert not pd.DataFrame(out).duplicated(subset=["real_date", "xcat", "cid"]).any()
    # sorted ascending by cid, xcat, real_date (IDX_COLS_SORT_ORDER)
    s = pd.DataFrame(out)[["cid", "xcat", "real_date"]].reset_index(drop=True)
    assert s.equals(s.sort_values(["cid", "xcat", "real_date"]).reset_index(drop=True))


def test_update_df_does_not_mutate_input():
    base, pieces = update_df_pieces("tiny", n_pieces=2)
    snapshot = base.copy(deep=True)
    update_df(base, pieces[0])
    pd.testing.assert_frame_equal(base, snapshot)
