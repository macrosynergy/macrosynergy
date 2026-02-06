import unittest
import tempfile
from pathlib import Path

import pandas as pd

from macrosynergy.download.dataquery_file_api import (
    FileSelector,
    JPMAQS_DATASET_THEME_MAPPING,
)

"""
JPMaQS publishes data as multiple parquet datasets plus related metadata files.

All data files share a common schema:

```
- real_date: date
- ticker: string
- value: float
- grading: float
- eop_lag: float
- mop_lag: float
- last_updated: datetime (the time the row was produced/published)
```

The "full view" of a dataset is expected to be unique on
(real_date, ticker, last_updated).

To arrive at a canonical representation (what should be used for analysis),
select the row with the greatest last_updated for each (real_date, ticker) pair.

Minimal row example (note: last_updated is a column value, not the file timestamp):

```
# earlier published row
real_date=2026-02-04, ticker=AAA, value=1.0, last_updated=2026-02-05T06:10:00Z
# later correction published
real_date=2026-02-04, ticker=AAA, value=1.1, last_updated=2026-02-06T06:10:00Z
# canonical keeps the later row (value=1.1)
```

File naming and types

Datasets appear in two main file types:

1. Snapshot files (daily "full snapshots")

   * One file per dataset per day
   * Filename ends with YYYYMMDD (no intra-day time component)

   Example:
   JPMAQS_MACROECONOMIC_TRENDS_20260206.parquet

2. Delta files (intra-day increments)

   * Multiple files per dataset per day
   * Filename ends with YYYYMMDDTHHMMSS

   Example:
   JPMAQS_GENERIC_RETURNS_DELTA_20260206T140336.parquet

Monthly historical delta files

At the end of each month, the source publishes a historical delta file that is
a stitched/appended version of all the delta files published during that month.

The only differentiating factor of a historical delta file is its timestamp:

- It is ALWAYS on the last day of the month at 23:59:59 (i.e. ...T235959)
- Regular intra-day delta files are guaranteed NOT to have ...T235959

Example (historical delta):
JPMAQS_GENERIC_RETURNS_DELTA_20260131T235959.parquet

Example (regular delta, never T235959):
JPMAQS_GENERIC_RETURNS_DELTA_20260206T140336.parquet

Availability and deletion behavior

Files do not "update" in place: once published, a file's contents remain fixed.
However, the source may remove files over time.

End-of-month deletion and consolidation behavior:

- Full snapshot files for older dates may be removed.
- All smaller intra-day delta files for that month may be removed.
- A single historical delta file for that month remains (the ...T235959 file).

How canonical reconstruction works (high-level)

To reconstruct a canonical dataset "as of" some target time:

Inputs:

- A set of snapshot files (if available)
- A set of delta files up to the target time (regular deltas and/or monthly historical deltas)
- Rows are canonicalized by max(last_updated) per (real_date, ticker)

Minimal pseudocode:

```
rows = []

if snapshot exists for dataset on or before target time:
    rows += read(snapshot_YYYYMMDD.parquet)

    rows += read(all delta files for dataset with file_timestamp > snapshot_timestamp
                 and file_timestamp <= target_time)

else:
    rows += read(all delta files for dataset with file_timestamp <= target_time)
    # this may include monthly historical delta files (...T235959) and/or remaining regular deltas

canonical_rows = for each (real_date, ticker): keep row with max(last_updated)
```

FileSelector responsibilities (what tests should focus on)

The FileSelector class decides which files should be downloaded and/or loaded to
support the reconstruction logic above, given:

* a view of what the source currently makes available (api listing)
* a view of what exists locally (local listing)

Tests for FileSelector typically assert:

* Correct inclusion/exclusion of snapshot vs delta files based on time range and flags
* Correct preference for monthly historical delta files (...T235959) when they cover the range
* Correct handling of overwrite flags (force download even if local exists)
* Correct handling of missing/invalid local paths (treat as not present)
* Correct handling of source-side deletions (file absent from api listing => cannot rely on it)

The JPMAQS_METADATA_CATALOG, when paired with the JPMAQS_DATASET_THEME_MAPPING
dictionary, allows identification of which datasets contain which tickers.
"""


class TestFileSelector(unittest.TestCase):
    def test_init_outer_merge_keeps_all_file_names(self):
        api_df = pd.DataFrame(
            {
                "file-name": ["DATASET1_20240101.parquet"],
                "file-datetime": [pd.Timestamp("2024-01-01T00:00:00Z")],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATASET2_20240101.parquet"
            p.write_bytes(b"x")
            local_df = pd.DataFrame(
                {
                    "file-name": ["DATASET2_20240101.parquet"],
                    "path": [str(p)],
                }
            )
            fs = FileSelector(api_df, local_df)

            self.assertCountEqual(
                fs.files_df["file-name"].astype(str).tolist(),
                ["DATASET1_20240101.parquet", "DATASET2_20240101.parquet"],
            )

    def test_select_files_for_download_overwrite(self):
        api_df = pd.DataFrame(
            {
                "file-name": [
                    "DATASET1_20240101.parquet",
                    "DATASET1_20240102.parquet",
                    "DATASET1_DELTA_20240102T010101.parquet",
                ],
                "file-datetime": [
                    pd.Timestamp("2024-01-01T00:00:00Z"),
                    pd.Timestamp("2024-01-02T00:00:00Z"),
                    pd.Timestamp("2024-01-02T01:01:01Z"),
                ],
            }
        )
        fs = FileSelector(api_df, pd.DataFrame())
        out = fs.select_files_for_download(
            overwrite=True,
            since_datetime="20240101",
            to_datetime="20240102T235959",
            include_delta_files=True,
        )
        self.assertCountEqual(
            out,
            ["DATASET1_20240102.parquet", "DATASET1_DELTA_20240102T010101.parquet"],
        )

    def test_select_files_for_download_load_minimal(self):
        api_df = pd.DataFrame(
            {
                "file-name": [
                    "DATASET1_20240101.parquet",
                    "DATASET1_20240102.parquet",
                    "DATASET1_DELTA_20240102T010101.parquet",
                ],
                "file-datetime": [
                    pd.Timestamp("2024-01-01T00:00:00Z"),
                    pd.Timestamp("2024-01-02T00:00:00Z"),
                    pd.Timestamp("2024-01-02T01:01:01Z"),
                ],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            p2 = Path(td) / "DATASET1_20240102.parquet"
            p2.write_bytes(b"x")
            local_df = pd.DataFrame(
                {
                    "file-name": ["DATASET1_20240102.parquet"],
                    "path": [str(p2)],
                }
            )
            fs = FileSelector(api_df, local_df)
            out = fs.select_files_for_download(
                since_datetime="20240101",
                to_datetime="20240102T235959",
                include_delta_files=True,
            )
            self.assertEqual(out, ["DATASET1_DELTA_20240102T010101.parquet"])

    def test_select_files_for_download_load_delta_only_large_delta_cover(self):
        api_df = pd.DataFrame(
            {
                "file-name": ["DATASETX_DELTA_20260131T235959.parquet"],
                "file-datetime": [pd.Timestamp("2026-01-31T23:59:59Z")],
            }
        )
        fs = FileSelector(api_df, pd.DataFrame())
        out = fs.select_files_for_download(
            to_datetime="20260115",
            include_delta_files=True,
        )
        self.assertEqual(out, ["DATASETX_DELTA_20260131T235959.parquet"])

    def test_select_files_for_download_ignores_missing_local_path(self):
        api_df = pd.DataFrame(
            {
                "file-name": ["DATASET1_20240102.parquet"],
                "file-datetime": [pd.Timestamp("2024-01-02T00:00:00Z")],
            }
        )
        local_df = pd.DataFrame(
            {"file-name": ["DATASET1_20240102.parquet"], "path": ["does-not-exist"]}
        )
        fs = FileSelector(api_df, local_df)
        out = fs.select_files_for_download(to_datetime="20240102")
        self.assertEqual(out, ["DATASET1_20240102.parquet"])

    def test_select_files_for_download_redownloads_if_api_last_modified_newer(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATASET1_20240102.parquet"
            p.write_bytes(b"x")
            api_df = pd.DataFrame(
                {
                    "file-name": ["DATASET1_20240102.parquet"],
                    "file-datetime": [pd.Timestamp("2024-01-02T00:00:00Z")],
                    "last-modified": [pd.Timestamp("2026-01-02T00:00:00Z")],
                }
            )
            local_df = pd.DataFrame(
                {
                    "file-name": ["DATASET1_20240102.parquet"],
                    "path": [str(p)],
                    "last-modified": [pd.Timestamp("2026-01-01T00:00:00Z")],
                }
            )
            fs = FileSelector(api_df, local_df)
            out = fs.select_files_for_download(to_datetime="20240102")
            self.assertEqual(out, ["DATASET1_20240102.parquet"])

    def test_select_files_for_load_drops_invalid_paths(self):
        with tempfile.TemporaryDirectory() as td:
            p_snap = Path(td) / "DATASET1_20240102.parquet"
            p_delta = Path(td) / "DATASET1_DELTA_20240102T010101.parquet"
            p_snap.write_bytes(b"x")
            p_delta.write_bytes(b"x")

            local_df = pd.DataFrame(
                {
                    "file-name": [
                        "DATASET1_20240102.parquet",
                        "DATASET1_DELTA_20240102T010101.parquet",
                        "DATASET1_20240101.parquet",
                    ],
                    "path": [
                        str(p_snap),
                        str(p_delta),
                        str(Path(td) / "missing.parquet"),
                    ],
                }
            )
            fs = FileSelector(pd.DataFrame(), local_df)
            out = fs.select_files_for_load(
                since_datetime="20240101",
                to_datetime="20240102T235959",
                include_delta_files=True,
            )
            self.assertCountEqual(
                out["filename"].astype(str).tolist(),
                ["DATASET1_20240102.parquet", "DATASET1_DELTA_20240102T010101.parquet"],
            )
