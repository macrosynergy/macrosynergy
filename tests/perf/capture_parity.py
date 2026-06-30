"""Regenerate golden output snapshots from CURRENT package code.

Run once on clean feature/performance (and deliberately with --update to refresh):
    python tests/perf/capture_parity.py --update
Each perf/<slug> branch re-runs the default gate, whose test_parity_*.py assert against these.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Allow running as a bare script (`python tests/perf/capture_parity.py`): ensure the
# repo root is importable so `tests.perf.*` resolves without PYTHONPATH being set.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd

from macrosynergy.management.utils import update_df, reduce_df, ticker_df_to_qdf
from tests.perf.data import qdf_for_tier, wide_ticker_frame, update_df_pieces
from tests.perf.parity import GOLDEN_DIR, save_golden


def _build_goldens() -> dict:
    """Return {name: (kind, DataFrame)} computed on current code at tiny scale."""
    out = {}

    # T3 reduce_df
    out["reduce_df_tiny"] = ("qdf", pd.DataFrame(reduce_df(qdf_for_tier("tiny"))))

    # T1 update_df growing loop
    base, pieces = update_df_pieces("tiny", n_pieces=3)
    acc = base
    for p in pieces:
        acc = update_df(acc, p)
    out["update_df_loop_tiny"] = ("qdf", pd.DataFrame(acc))

    # T2 ticker_df_to_qdf
    out["ticker_df_to_qdf_tiny"] = ("qdf", pd.DataFrame(ticker_df_to_qdf(wide_ticker_frame(12, 60))))

    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="overwrite existing goldens")
    args = parser.parse_args(argv)

    index_path = GOLDEN_DIR / "index.json"
    if index_path.exists() and not args.update:
        print(f"Goldens already exist at {index_path}; pass --update to regenerate.")
        return 1

    index = {}
    for name, (kind, df) in _build_goldens().items():
        h = save_golden(name, df)
        index[name] = {"kind": kind, "hash": h}
        print(f"  captured {name}: {kind} sha256={h[:12]}…")

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {index_path} ({len(index)} goldens).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
