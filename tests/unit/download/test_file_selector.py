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
real_date=2026-02-04, ticker=USD_EQXR_NSA_TRF1, value=1.0, last_updated=2026-02-05T06:10:00Z
# later correction published
real_date=2026-02-04, ticker=USD_EQXR_NSA_TRF1, value=1.1, last_updated=2026-02-06T06:10:00Z
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

- It is at month-end "23:59:59" (i.e. ...T235959). If month-end falls on a
  weekend/holiday, the timestamp can also be the previous business day at 23:59:59.
  This means both, the last day of the month and the last business day of the month, should
  be checked for historical deltas.
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

This module also contains a small number of integration-style tests for
`DataQueryFileAPIClient.download()` that assert the *historical delta bootstrap*
parameters and file selection passed into lower-level download helpers.
"""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from macrosynergy.download.dataquery_file_api import (
    DataQueryFileAPIClient,
    FileSelector,
    JPMAQS_DATASET_THEME_MAPPING,
    JPMAQS_EARLIEST_FILE_DATE,
)

EMPTY_API_FILES_DF = pd.DataFrame(
    columns=["file-name", "file-datetime", "last-modified"]
)
EMPTY_LOCAL_FILES_DF = pd.DataFrame(columns=["file-name", "path", "last-modified"])


class TestFileSelectorInit(unittest.TestCase):
    def test_init_accepts_empty_inventories(self):
        fs = FileSelector(EMPTY_API_FILES_DF.copy(), EMPTY_LOCAL_FILES_DF.copy())
        self.assertTrue(isinstance(fs.files_df, pd.DataFrame))
        self.assertIn("file-name", fs.files_df.columns)

    def test_init_raises_if_api_missing_file_name_column(self):
        api_df = pd.DataFrame({"file-datetime": [pd.Timestamp("2024-01-01T00:00:00Z")]})
        with self.assertRaisesRegex(ValueError, "Missing `file-name` in api_files_df"):
            FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())

    def test_init_raises_if_local_missing_file_name_column(self):
        local_df = pd.DataFrame({"path": ["x"]})
        with self.assertRaisesRegex(
            ValueError, "Missing `file-name` in local_files_df"
        ):
            FileSelector(EMPTY_API_FILES_DF.copy(), local_df)

    def test_init_outer_merge_keeps_all_file_names(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "group-id": "JPMAQS",
                    "file-group-id": "DATASET1",
                    "is-available": True,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATASET2_20240101.parquet"
            p.write_bytes(b"x")
            local_df = pd.DataFrame(
                [{"file-name": "DATASET2_20240101.parquet", "path": str(p)}]
            )
            fs = FileSelector(api_df, local_df)

            self.assertCountEqual(
                fs.files_df["file-name"].astype(str).tolist(),
                ["DATASET1_20240101.parquet", "DATASET2_20240101.parquet"],
            )

    def test_init_accepts_filename_alias_column(self):
        api_df = pd.DataFrame(
            {
                "filename": ["DATASET1_20240101.parquet"],
                "file-datetime": [pd.Timestamp("2024-01-01T00:00:00Z")],
            }
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATASET1_20240101.parquet"
            p.write_bytes(b"x")
            local_df = pd.DataFrame(
                {"filename": ["DATASET1_20240101.parquet"], "path": [str(p)]}
            )

            fs = FileSelector(api_df, local_df)
            self.assertIn("file-name", fs.api_files_df.columns)
            self.assertIn("file-name", fs.local_files_df.columns)
            self.assertEqual(
                fs.files_df["file-name"].astype(str).tolist(),
                ["DATASET1_20240101.parquet"],
            )

    def test_init_dedupes_by_last_modified_prefers_latest(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "last-modified": pd.Timestamp("2024-01-01T06:00:00Z"),
                },
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "last-modified": pd.Timestamp("2024-01-01T07:00:00Z"),
                },
            ]
        )
        local_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "path": "does-not-exist",
                    "last-modified": pd.Timestamp("2024-01-01T08:00:00Z"),
                },
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "path": "also-does-not-exist",
                    "last-modified": pd.Timestamp("2024-01-01T09:00:00Z"),
                },
            ]
        )
        fs = FileSelector(api_df, local_df)
        self.assertEqual(len(fs.api_files_df), 1)
        self.assertEqual(len(fs.local_files_df), 1)
        self.assertEqual(
            fs.api_files_df.iloc[0]["last-modified"],
            pd.Timestamp("2024-01-01T07:00:00Z"),
        )
        self.assertEqual(
            fs.local_files_df.iloc[0]["last-modified"],
            pd.Timestamp("2024-01-01T09:00:00Z"),
        )


class TestFileSelectorSelectFilesForDownload(unittest.TestCase):
    def test_select_files_for_download_returns_empty_when_api_empty(self):
        fs = FileSelector(EMPTY_API_FILES_DF.copy(), EMPTY_LOCAL_FILES_DF.copy())
        self.assertEqual(fs.select_files_for_download(to_datetime="20240102"), [])

    def test_select_files_for_download_parses_timestamp_from_filename_when_file_datetime_column_missing(
        self,
    ):
        api_df = pd.DataFrame({"file-name": ["DATASET1_20240102.parquet"]})
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(to_datetime="20240102")
        self.assertEqual(out, ["DATASET1_20240102.parquet"])

    def test_select_files_for_download_skips_rows_with_invalid_timestamp(self):
        api_df = pd.DataFrame({"file-name": ["DATASET1_NOT_A_TS.parquet"]})
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        with self.assertRaises(ValueError):
            fs.select_files_for_download(to_datetime="20240102")

    def test_select_files_for_download_skips_api_rows_with_missing_file_datetime(self):
        api_df = pd.DataFrame(
            [
                {"file-name": "DATASET1_20240102.parquet", "file-datetime": pd.NaT},
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(to_datetime="20240102")
        self.assertEqual(out, ["DATASET1_20240101.parquet"])

    def test_select_files_for_download_date_only_to_datetime_is_inclusive(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_20240102.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(to_datetime="20240102")
        self.assertEqual(out, ["DATASET1_20240102.parquet"])

    def test_select_files_for_download_overwrite(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_20240102.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_DELTA_20240102T010101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T01:01:01Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
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
            [
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_20240102.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_DELTA_20240102T010101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T01:01:01Z"),
                },
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p2 = Path(td) / "DATASET1_20240102.parquet"
            p2.write_bytes(b"x")
            local_df = pd.DataFrame(
                [{"file-name": "DATASET1_20240102.parquet", "path": str(p2)}]
            )
            fs = FileSelector(api_df, local_df)
            out = fs.select_files_for_download(
                since_datetime="20240101",
                to_datetime="20240102T235959",
                include_delta_files=True,
            )
            self.assertEqual(out, ["DATASET1_DELTA_20240102T010101.parquet"])

    def test_select_files_for_download_latest_snapshot_plus_newer_deltas(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_20240102.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_DELTA_20240101T010101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T01:01:01Z"),
                },
                {
                    "file-name": "DATASET1_DELTA_20240102T010101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T01:01:01Z"),
                },
                {
                    "file-name": "DATASET1_DELTA_20240102T230000.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T23:00:00Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(
            since_datetime="20240101",
            to_datetime="20240102T235959",
            include_delta_files=True,
        )
        self.assertEqual(
            out,
            [
                "DATASET1_20240102.parquet",
                "DATASET1_DELTA_20240102T010101.parquet",
                "DATASET1_DELTA_20240102T230000.parquet",
            ],
        )

    def test_select_files_for_download_swaps_window_when_since_is_after_to(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_20240102.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_DELTA_20240102T010101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T01:01:01Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(
            since_datetime="20240103",
            to_datetime="20240102T000000",
            include_delta_files=True,
        )
        self.assertEqual(
            out,
            [
                "DATASET1_20240102.parquet",
                "DATASET1_DELTA_20240102T010101.parquet",
            ],
        )

    def test_select_files_for_download_excludes_deltas_when_include_delta_false(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240102.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_DELTA_20240102T010101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T01:01:01Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(
            to_datetime="20240102T235959",
            include_delta_files=False,
        )
        self.assertEqual(out, ["DATASET1_20240102.parquet"])

    def test_select_files_for_download_ignores_metadata_files(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240102.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_METADATA_20240102.json",
                    "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_METADATA_20240102T010101.json",
                    "file-datetime": pd.Timestamp("2024-01-02T01:01:01Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(to_datetime="20240102T235959")
        self.assertEqual(out, ["DATASET1_20240102.parquet"])

    def test_select_files_for_download_load_delta_only_large_delta_cover(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240229T235959.parquet",
                    "file-datetime": pd.Timestamp("2024-02-29T23:59:59Z"),
                },
                {
                    "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240329T235959.parquet",
                    "file-datetime": pd.Timestamp("2024-03-29T23:59:59Z"),
                },
                {
                    "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240331T235959.parquet",
                    "file-datetime": pd.Timestamp("2024-03-31T23:59:59Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(
            since_datetime="20240320",
            to_datetime="20240315",
            include_delta_files=True,
        )
        self.assertEqual(
            out,
            [
                "JPMAQS_GENERIC_RETURNS_DELTA_20240229T235959.parquet",
                "JPMAQS_GENERIC_RETURNS_DELTA_20240329T235959.parquet",
            ],
        )

    def test_select_files_for_download_max_last_updated_overrides_file_vintage_for_cover_delta(
        self,
    ):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "JPMAQS_SHOCKS_RISK_MEASURES_DELTA_20250930T235959.parquet",
                    "file-datetime": pd.Timestamp("2025-09-30T23:59:59Z"),
                },
                {
                    "file-name": "JPMAQS_SHOCKS_RISK_MEASURES_DELTA_20251031T235959.parquet",
                    "file-datetime": pd.Timestamp("2025-10-31T23:59:59Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(
            to_datetime="20250915",
            max_last_updated="20251008",
            include_delta_files=True,
        )
        self.assertEqual(
            out,
            [
                "JPMAQS_SHOCKS_RISK_MEASURES_DELTA_20250930T235959.parquet",
                "JPMAQS_SHOCKS_RISK_MEASURES_DELTA_20251031T235959.parquet",
            ],
        )

    def test_select_files_for_download_delta_only_history_excludes_when_include_delta_false(
        self,
    ):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240329T235959.parquet",
                    "file-datetime": pd.Timestamp("2024-03-29T23:59:59Z"),
                }
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(
            to_datetime="20240315",
            include_delta_files=False,
        )
        self.assertEqual(out, [])

    def test_select_files_for_download_ignores_missing_local_path(self):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240102.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                }
            ]
        )
        local_df = pd.DataFrame(
            [{"file-name": "DATASET1_20240102.parquet", "path": "does-not-exist"}]
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

    def test_select_files_for_download_does_not_redownload_when_local_file_exists(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATASET1_20240102.parquet"
            p.write_bytes(b"x")
            api_df = pd.DataFrame(
                [
                    {
                        "file-name": "DATASET1_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    }
                ]
            )
            local_df = pd.DataFrame(
                [{"file-name": "DATASET1_20240102.parquet", "path": str(p)}]
            )
            fs = FileSelector(api_df, local_df)
            out = fs.select_files_for_download(to_datetime="20240102")
            self.assertEqual(out, [])

    def test_select_files_for_download_last_modified_only_considered_for_required(self):
        with tempfile.TemporaryDirectory() as td:
            p_req = Path(td) / "DATASET1_20240102.parquet"
            p_req.write_bytes(b"x")
            p_not = Path(td) / "DATASET1_20240101.parquet"
            p_not.write_bytes(b"x")

            api_df = pd.DataFrame(
                [
                    {
                        "file-name": "DATASET1_20240101.parquet",
                        "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                        "last-modified": pd.Timestamp("2024-01-01T10:00:00Z"),
                    },
                    {
                        "file-name": "DATASET1_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                        "last-modified": pd.Timestamp("2024-01-02T10:00:00Z"),
                    },
                ]
            )
            local_df = pd.DataFrame(
                [
                    {
                        "file-name": "DATASET1_20240101.parquet",
                        "path": str(p_not),
                        "last-modified": pd.Timestamp("2024-01-01T00:00:00Z"),
                    },
                    {
                        "file-name": "DATASET1_20240102.parquet",
                        "path": str(p_req),
                        "last-modified": pd.Timestamp("2024-01-02T11:00:00Z"),
                    },
                ]
            )

            fs = FileSelector(api_df, local_df)
            out = fs.select_files_for_download(to_datetime="20240102")
            self.assertEqual(out, [])

    def test_select_files_for_download_overwrite_downloads_even_if_present_locally(
        self,
    ):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "DATASET1_20240102.parquet"
            p.write_bytes(b"x")
            api_df = pd.DataFrame(
                [
                    {
                        "file-name": "DATASET1_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    }
                ]
            )
            local_df = pd.DataFrame(
                [{"file-name": "DATASET1_20240102.parquet", "path": str(p)}]
            )
            fs = FileSelector(api_df, local_df)
            out = fs.select_files_for_download(
                overwrite=True,
                to_datetime="20240102",
            )
            self.assertEqual(out, ["DATASET1_20240102.parquet"])

    @patch("macrosynergy.download.dataquery_file_api.logger")
    def test_select_files_for_download_warns_when_window_has_only_deltas(
        self, mock_logger
    ):
        api_df = pd.DataFrame(
            [
                {
                    "file-name": "DATASET1_20240101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-01T00:00:00Z"),
                },
                {
                    "file-name": "DATASET1_DELTA_20240102T010101.parquet",
                    "file-datetime": pd.Timestamp("2024-01-02T01:01:01Z"),
                },
            ]
        )
        fs = FileSelector(api_df, EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_download(
            since_datetime="20240102",
            to_datetime="20240102T235959",
            include_delta_files=True,
            warn_if_no_full_snapshots=True,
        )
        self.assertEqual(out, ["DATASET1_DELTA_20240102T010101.parquet"])
        self.assertTrue(mock_logger.warning.called)
        msg = " ".join(str(a) for a in mock_logger.warning.call_args[0])
        self.assertIn("No full snapshots available in the requested window", msg)


class TestFileSelectorSelectFilesForLoad(unittest.TestCase):
    def test_select_files_for_load_returns_empty_when_local_empty(self):
        fs = FileSelector(EMPTY_API_FILES_DF.copy(), EMPTY_LOCAL_FILES_DF.copy())
        out = fs.select_files_for_load(to_datetime="20240102")
        self.assertTrue(out.empty)

    def test_select_files_for_load_returns_empty_df_when_path_column_missing(self):
        local_df = pd.DataFrame({"file-name": ["DATASET1_20240102.parquet"]})
        fs = FileSelector(EMPTY_API_FILES_DF.copy(), local_df)
        out = fs.select_files_for_load(to_datetime="20240102")
        self.assertTrue(out.empty)

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
            fs = FileSelector(EMPTY_API_FILES_DF.copy(), local_df)
            out = fs.select_files_for_load(
                since_datetime="20240101",
                to_datetime="20240102T235959",
                include_delta_files=True,
            )
            self.assertCountEqual(
                out["filename"].astype(str).tolist(),
                ["DATASET1_20240102.parquet", "DATASET1_DELTA_20240102T010101.parquet"],
            )

    def test_select_files_for_load_excludes_metadata_files(self):
        with tempfile.TemporaryDirectory() as td:
            p_snap = Path(td) / "DATASET1_20240102.parquet"
            p_meta = Path(td) / "DATASET1_METADATA_20240102.json"
            p_snap.write_bytes(b"x")
            p_meta.write_bytes(b"x")

            local_df = pd.DataFrame(
                [
                    {"file-name": p_snap.name, "path": str(p_snap)},
                    {"file-name": p_meta.name, "path": str(p_meta)},
                ]
            )
            fs = FileSelector(EMPTY_API_FILES_DF.copy(), local_df)
            out = fs.select_files_for_load(to_datetime="20240102T235959")
            self.assertEqual(out["filename"].astype(str).tolist(), [p_snap.name])

    def test_select_files_for_load_latest_snapshot_plus_newer_deltas(self):
        with tempfile.TemporaryDirectory() as td:
            p_snap_old = Path(td) / "DATASET1_20240101.parquet"
            p_snap_new = Path(td) / "DATASET1_20240102.parquet"
            p_delta_old = Path(td) / "DATASET1_DELTA_20240101T010101.parquet"
            p_delta_new = Path(td) / "DATASET1_DELTA_20240102T010101.parquet"
            for p in (p_snap_old, p_snap_new, p_delta_old, p_delta_new):
                p.write_bytes(b"x")

            local_df = pd.DataFrame(
                [
                    {"file-name": p_snap_old.name, "path": str(p_snap_old)},
                    {"file-name": p_snap_new.name, "path": str(p_snap_new)},
                    {"file-name": p_delta_old.name, "path": str(p_delta_old)},
                    {"file-name": p_delta_new.name, "path": str(p_delta_new)},
                ]
            )
            fs = FileSelector(EMPTY_API_FILES_DF.copy(), local_df)
            out = fs.select_files_for_load(
                to_datetime="20240102T235959", include_delta_files=True
            )
            self.assertCountEqual(
                out["filename"].astype(str).tolist(),
                [
                    "DATASET1_20240102.parquet",
                    "DATASET1_DELTA_20240102T010101.parquet",
                ],
            )

    def test_select_files_for_load_excludes_deltas_when_include_delta_false(self):
        with tempfile.TemporaryDirectory() as td:
            p_snap = Path(td) / "DATASET1_20240102.parquet"
            p_delta = Path(td) / "DATASET1_DELTA_20240102T010101.parquet"
            p_snap.write_bytes(b"x")
            p_delta.write_bytes(b"x")
            local_df = pd.DataFrame(
                [
                    {"file-name": p_snap.name, "path": str(p_snap)},
                    {"file-name": p_delta.name, "path": str(p_delta)},
                ]
            )
            fs = FileSelector(EMPTY_API_FILES_DF.copy(), local_df)
            out = fs.select_files_for_load(
                to_datetime="20240102T235959", include_delta_files=False
            )
            self.assertEqual(out["filename"].astype(str).tolist(), [p_snap.name])

    def test_select_files_for_load_delta_only_includes_covering_large_delta(self):
        with tempfile.TemporaryDirectory() as td:
            p_large = Path(td) / "JPMAQS_GENERIC_RETURNS_DELTA_20240329T235959.parquet"
            p_prev = Path(td) / "JPMAQS_GENERIC_RETURNS_DELTA_20240229T235959.parquet"
            p_large.write_bytes(b"x")
            p_prev.write_bytes(b"x")

            local_df = pd.DataFrame(
                [
                    {"file-name": p_prev.name, "path": str(p_prev)},
                    {"file-name": p_large.name, "path": str(p_large)},
                ]
            )
            fs = FileSelector(EMPTY_API_FILES_DF.copy(), local_df)
            out = fs.select_files_for_load(
                since_datetime="20240320",
                to_datetime="20240315",
                include_delta_files=True,
            )
            self.assertCountEqual(
                out["filename"].astype(str).tolist(),
                [p_prev.name, p_large.name],
            )

    def test_select_files_for_load_ignores_snapshots_after_vintage_and_uses_large_delta_cover(
        self,
    ):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            p_prev = (
                td_path
                / "JPMAQS_SHOCKS_RISK_MEASURES_DELTA_20250930T235959.parquet"
            )
            p_cover = (
                td_path
                / "JPMAQS_SHOCKS_RISK_MEASURES_DELTA_20251031T235959.parquet"
            )
            p_future_snap = td_path / "JPMAQS_SHOCKS_RISK_MEASURES_20260204.parquet"
            for p in (p_prev, p_cover, p_future_snap):
                p.write_bytes(b"x")

            local_df = pd.DataFrame(
                [
                    {"file-name": p_prev.name, "path": str(p_prev)},
                    {"file-name": p_cover.name, "path": str(p_cover)},
                    {"file-name": p_future_snap.name, "path": str(p_future_snap)},
                ]
            )
            fs = FileSelector(EMPTY_API_FILES_DF.copy(), local_df)
            out = fs.select_files_for_load(
                to_datetime="20251008", include_delta_files=True
            )
            self.assertCountEqual(
                out["filename"].astype(str).tolist(),
                [p_prev.name, p_cover.name],
            )

    def test_select_files_for_load_max_last_updated_overrides_file_vintage_for_cover_delta(
        self,
    ):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            p_sep = (
                td_path / "JPMAQS_SHOCKS_RISK_MEASURES_DELTA_20250930T235959.parquet"
            )
            p_oct = (
                td_path / "JPMAQS_SHOCKS_RISK_MEASURES_DELTA_20251031T235959.parquet"
            )
            for p in (p_sep, p_oct):
                p.write_bytes(b"x")

            local_df = pd.DataFrame(
                [
                    {"file-name": p_sep.name, "path": str(p_sep)},
                    {"file-name": p_oct.name, "path": str(p_oct)},
                ]
            )
            fs = FileSelector(EMPTY_API_FILES_DF.copy(), local_df)
            out = fs.select_files_for_load(
                to_datetime="20250915",
                max_last_updated="20251008",
                include_delta_files=True,
            )
            self.assertCountEqual(
                out["filename"].astype(str).tolist(),
                [p_sep.name, p_oct.name],
            )

    def test_select_files_for_load_delta_only_excludes_when_include_delta_false(self):
        with tempfile.TemporaryDirectory() as td:
            p_large = Path(td) / "JPMAQS_GENERIC_RETURNS_DELTA_20240329T235959.parquet"
            p_large.write_bytes(b"x")
            local_df = pd.DataFrame([{"file-name": p_large.name, "path": str(p_large)}])
            fs = FileSelector(EMPTY_API_FILES_DF.copy(), local_df)
            out = fs.select_files_for_load(
                to_datetime="20240315", include_delta_files=False
            )
            self.assertTrue(out.empty)


class TestFileSelectorTickerFiltering(unittest.TestCase):
    def test_filters_api_files_to_datasets_for_tickers_using_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            cat_path = Path(td) / "JPMAQS_METADATA_CATALOG_20240102.parquet"
            cat_path.write_bytes(b"not a real parquet - patched read")

            api_df = pd.DataFrame(
                [
                    {
                        "file-name": "JPMAQS_GENERIC_RETURNS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                    {
                        "file-name": "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                ]
            )

            mocked_catalog_df = pd.DataFrame(
                {
                    "Ticker": ["USD_EQXR_NSA_TRF1", "EUR_RIR_NSA_TRFX"],
                    "Theme": ["Generic returns", "Macroeconomic trends"],
                }
            )
            with patch(
                "macrosynergy.download.dataquery_file_api.pd.read_parquet",
                autospec=True,
                return_value=mocked_catalog_df,
            ) as mock_read:
                fs = FileSelector(
                    api_df,
                    EMPTY_LOCAL_FILES_DF.copy(),
                    tickers=["usd_eqxr_nsa_trf1"],
                    catalog_file=cat_path,
                )
                mock_read.assert_called_once()
                self.assertEqual(mock_read.call_args[0][0], cat_path)
                self.assertIn("Generic returns", JPMAQS_DATASET_THEME_MAPPING)

                out = fs.select_files_for_download(
                    overwrite=True, to_datetime="20240102"
                )
                self.assertEqual(out, ["JPMAQS_GENERIC_RETURNS_20240102.parquet"])

    def test_resolves_catalog_from_local_inventory_when_catalog_file_not_provided(self):
        with tempfile.TemporaryDirectory() as td:
            catalog_old = Path(td) / "JPMAQS_METADATA_CATALOG_20240101.parquet"
            catalog_new = Path(td) / "JPMAQS_METADATA_CATALOG_20240102.parquet"
            catalog_old.write_bytes(b"x")
            catalog_new.write_bytes(b"x")

            local_df = pd.DataFrame(
                [
                    {
                        "file-name": catalog_old.name,
                        "path": str(catalog_old),
                        "dataset": "JPMAQS_METADATA_CATALOG",
                        "file-timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                    },
                    {
                        "file-name": catalog_new.name,
                        "path": str(catalog_new),
                        "dataset": "JPMAQS_METADATA_CATALOG",
                        "file-timestamp": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                ]
            )

            api_df = pd.DataFrame(
                [
                    {
                        "file-name": "JPMAQS_GENERIC_RETURNS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                    {
                        "file-name": "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                ]
            )

            mocked_catalog_df = pd.DataFrame(
                {"Ticker": ["USD_EQXR_NSA_TRF1"], "Theme": ["Generic returns"]}
            )
            with patch(
                "macrosynergy.download.dataquery_file_api.pd.read_parquet",
                autospec=True,
                return_value=mocked_catalog_df,
            ) as mock_read:
                fs = FileSelector(api_df, local_df, tickers=["USD_EQXR_NSA_TRF1"])
                self.assertEqual(mock_read.call_args[0][0], catalog_new)

                out = fs.select_files_for_download(
                    overwrite=True, to_datetime="20240102"
                )
                self.assertEqual(out, ["JPMAQS_GENERIC_RETURNS_20240102.parquet"])

    def test_case_sensitive_catalog_matching(self):
        with tempfile.TemporaryDirectory() as td:
            cat_path = Path(td) / "JPMAQS_METADATA_CATALOG_20240102.parquet"
            cat_path.write_bytes(b"not a real parquet - patched read")

            api_df = pd.DataFrame(
                [
                    {
                        "file-name": "JPMAQS_GENERIC_RETURNS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                    {
                        "file-name": "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                ]
            )

            mocked_catalog_df = pd.DataFrame(
                {"Ticker": ["USD_EQXR_NSA_TRF1"], "Theme": ["Generic returns"]}
            )
            with patch(
                "macrosynergy.download.dataquery_file_api.pd.read_parquet",
                autospec=True,
                return_value=mocked_catalog_df,
            ):
                fs_case_insensitive = FileSelector(
                    api_df,
                    EMPTY_LOCAL_FILES_DF.copy(),
                    tickers=["usd_eqxr_nsa_trf1"],
                    catalog_file=cat_path,
                    case_sensitive=False,
                )
                out_case_insensitive = fs_case_insensitive.select_files_for_download(
                    overwrite=True, to_datetime="20240102"
                )
                self.assertEqual(
                    out_case_insensitive, ["JPMAQS_GENERIC_RETURNS_20240102.parquet"]
                )

                fs_case_sensitive = FileSelector(
                    api_df,
                    EMPTY_LOCAL_FILES_DF.copy(),
                    tickers=["usd_eqxr_nsa_trf1"],
                    catalog_file=cat_path,
                    case_sensitive=True,
                )
                out_case_sensitive = fs_case_sensitive.select_files_for_download(
                    overwrite=True, to_datetime="20240102"
                )
                self.assertCountEqual(
                    out_case_sensitive,
                    [
                        "JPMAQS_GENERIC_RETURNS_20240102.parquet",
                        "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet",
                    ],
                )

    def test_ticker_filtering_falls_back_to_unfiltered_when_catalog_unreadable(self):
        with tempfile.TemporaryDirectory() as td:
            cat_path = Path(td) / "JPMAQS_METADATA_CATALOG_20240102.parquet"
            cat_path.write_bytes(b"x")
            api_df = pd.DataFrame(
                [
                    {
                        "file-name": "JPMAQS_GENERIC_RETURNS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                    {
                        "file-name": "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                ]
            )
            with patch(
                "macrosynergy.download.dataquery_file_api.pd.read_parquet",
                autospec=True,
                side_effect=ValueError("boom"),
            ):
                fs = FileSelector(
                    api_df,
                    EMPTY_LOCAL_FILES_DF.copy(),
                    tickers=["USD_EQXR_NSA_TRF1"],
                    catalog_file=cat_path,
                )
                out = fs.select_files_for_download(
                    overwrite=True, to_datetime="20240102"
                )
                self.assertCountEqual(
                    out,
                    [
                        "JPMAQS_GENERIC_RETURNS_20240102.parquet",
                        "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet",
                    ],
                )

    def test_ticker_filtering_falls_back_to_unfiltered_when_theme_not_mapped(self):
        with tempfile.TemporaryDirectory() as td:
            cat_path = Path(td) / "JPMAQS_METADATA_CATALOG_20240102.parquet"
            cat_path.write_bytes(b"x")
            api_df = pd.DataFrame(
                [
                    {
                        "file-name": "JPMAQS_GENERIC_RETURNS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                    {
                        "file-name": "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                ]
            )
            mocked_catalog_df = pd.DataFrame(
                {"Ticker": ["USD_EQXR_NSA_TRF1"], "Theme": ["Not a theme"]}
            )
            with patch(
                "macrosynergy.download.dataquery_file_api.pd.read_parquet",
                autospec=True,
                return_value=mocked_catalog_df,
            ):
                fs = FileSelector(
                    api_df,
                    EMPTY_LOCAL_FILES_DF.copy(),
                    tickers=["USD_EQXR_NSA_TRF1"],
                    catalog_file=cat_path,
                )
                out = fs.select_files_for_download(
                    overwrite=True, to_datetime="20240102"
                )
                self.assertCountEqual(
                    out,
                    [
                        "JPMAQS_GENERIC_RETURNS_20240102.parquet",
                        "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet",
                    ],
                )

    def test_ticker_filtering_supports_lowercase_catalog_columns(self):
        with tempfile.TemporaryDirectory() as td:
            cat_path = Path(td) / "JPMAQS_METADATA_CATALOG_20240102.parquet"
            cat_path.write_bytes(b"x")
            api_df = pd.DataFrame(
                [
                    {
                        "file-name": "JPMAQS_GENERIC_RETURNS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                    {
                        "file-name": "JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet",
                        "file-datetime": pd.Timestamp("2024-01-02T00:00:00Z"),
                    },
                ]
            )
            mocked_catalog_df = pd.DataFrame(
                {
                    "ticker": ["USD_EQXR_NSA_TRF1", "EUR_RIR_NSA_TRFX"],
                    "theme": ["Generic returns", "Macroeconomic trends"],
                }
            )
            with patch(
                "macrosynergy.download.dataquery_file_api.pd.read_parquet",
                autospec=True,
                return_value=mocked_catalog_df,
            ):
                fs = FileSelector(
                    api_df,
                    EMPTY_LOCAL_FILES_DF.copy(),
                    tickers=["usd_eqxr_nsa_trf1"],
                    catalog_file=cat_path,
                )
                out = fs.select_files_for_download(
                    overwrite=True, to_datetime="20240102"
                )
                self.assertEqual(out, ["JPMAQS_GENERIC_RETURNS_20240102.parquet"])


class TestFileSelectorHistoricalDeltaFullSelection(unittest.TestCase):
    def test_select_files_for_download_full_history_for_delta_only_dataset(self):
        month_ends = pd.date_range("2022-01-31", "2024-02-29", freq="ME")
        large_delta_ts = [d.strftime("%Y%m%dT235959") for d in month_ends]
        large_delta_ts += ["20240329T235959", "20240331T235959"]
        large_delta_ts += ["20240430T235959"]

        api_rows = [
            {
                "file-name": f"JPMAQS_GENERIC_RETURNS_DELTA_{ts}.parquet",
                "file-datetime": pd.to_datetime(ts, format="%Y%m%dT%H%M%S", utc=True),
            }
            for ts in large_delta_ts
        ]
        api_rows += [
            {
                "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240310T120000.parquet",
                "file-datetime": pd.Timestamp("2024-03-10T12:00:00Z"),
            },
            {
                "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240315T010101.parquet",
                "file-datetime": pd.Timestamp("2024-03-15T01:01:01Z"),
            },
            {
                "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240330T010101.parquet",
                "file-datetime": pd.Timestamp("2024-03-30T01:01:01Z"),
            },
        ]

        api_rows += [
            {
                "file-name": "JPMAQS_MACROECONOMIC_TRENDS_20240313.parquet",
                "file-datetime": pd.Timestamp("2024-03-13T00:00:00Z"),
            },
            {
                "file-name": "JPMAQS_MACROECONOMIC_TRENDS_20240314.parquet",
                "file-datetime": pd.Timestamp("2024-03-14T00:00:00Z"),
            },
            {
                "file-name": "JPMAQS_MACROECONOMIC_TRENDS_DELTA_20240313T230000.parquet",
                "file-datetime": pd.Timestamp("2024-03-13T23:00:00Z"),
            },
            {
                "file-name": "JPMAQS_MACROECONOMIC_TRENDS_DELTA_20240314T010101.parquet",
                "file-datetime": pd.Timestamp("2024-03-14T01:01:01Z"),
            },
            {
                "file-name": "JPMAQS_MACROECONOMIC_TRENDS_DELTA_20240315T120000.parquet",
                "file-datetime": pd.Timestamp("2024-03-15T12:00:00Z"),
            },
            {
                "file-name": "JPMAQS_METADATA_CATALOG_20240315.parquet",
                "file-datetime": pd.Timestamp("2024-03-15T00:00:00Z"),
            },
        ]

        api_df = pd.DataFrame(api_rows)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            already_downloaded = [
                "JPMAQS_GENERIC_RETURNS_DELTA_20220131T235959.parquet",
                "JPMAQS_GENERIC_RETURNS_DELTA_20220228T235959.parquet",
                "JPMAQS_MACROECONOMIC_TRENDS_20240314.parquet",
            ]
            for fn in already_downloaded:
                (td_path / fn).write_bytes(b"x")

            local_df = pd.DataFrame(
                [
                    {
                        "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20220131T235959.parquet",
                        "path": str(
                            td_path
                            / "JPMAQS_GENERIC_RETURNS_DELTA_20220131T235959.parquet"
                        ),
                    },
                    {
                        "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20220228T235959.parquet",
                        "path": str(
                            td_path
                            / "JPMAQS_GENERIC_RETURNS_DELTA_20220228T235959.parquet"
                        ),
                    },
                    {
                        "file-name": "JPMAQS_MACROECONOMIC_TRENDS_20240314.parquet",
                        "path": str(
                            td_path / "JPMAQS_MACROECONOMIC_TRENDS_20240314.parquet"
                        ),
                    },
                    {
                        "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20220331T235959.parquet",
                        "path": "does-not-exist",
                    },
                ]
            )

            fs = FileSelector(api_df, local_df)
            selected = fs.select_files_for_download(
                since_datetime="20240310",
                to_datetime="20240315",
                include_delta_files=True,
            )

            expected_macro = [
                "JPMAQS_MACROECONOMIC_TRENDS_20240314.parquet",
                "JPMAQS_MACROECONOMIC_TRENDS_DELTA_20240314T010101.parquet",
                "JPMAQS_MACROECONOMIC_TRENDS_DELTA_20240315T120000.parquet",
            ]
            expected_macro = [
                x
                for x in expected_macro
                if x != "JPMAQS_MACROECONOMIC_TRENDS_20240314.parquet"
            ]

            expected_generic = [
                f"JPMAQS_GENERIC_RETURNS_DELTA_{ts}.parquet"
                for ts in large_delta_ts
                if ts <= "20240329T235959"
            ]
            expected_generic += [
                "JPMAQS_GENERIC_RETURNS_DELTA_20240310T120000.parquet",
                "JPMAQS_GENERIC_RETURNS_DELTA_20240315T010101.parquet",
            ]
            expected_generic = sorted(
                set(expected_generic)
                - {
                    "JPMAQS_GENERIC_RETURNS_DELTA_20220131T235959.parquet",
                    "JPMAQS_GENERIC_RETURNS_DELTA_20220228T235959.parquet",
                }
            )

            expected = sorted(set(expected_macro + expected_generic))
            self.assertListEqual(selected, expected)


class TestDataQueryFileAPIClientHistoricalDeltaBootstrap(unittest.TestCase):
    @patch(
        "macrosynergy.download.dataquery_file_api.get_client_id_secret",
        side_effect=AssertionError(
            "Unexpected credential lookup via get_client_id_secret."
        ),
    )
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_mid_month_historical_vintage_bootstraps_full_delta_history(
        self, _mock_oauth, _mock_get_client
    ):
        with tempfile.TemporaryDirectory() as td:
            client = DataQueryFileAPIClient(
                client_id="id",
                client_secret="secret",
                out_dir=td,
            )

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        client, "download_catalog_file", return_value="catalog.parquet"
                    )
                )
                stack.enter_context(
                    patch.object(
                        client,
                        "filter_to_valid_tickers",
                        return_value=["USD_EQXR_NSA_TRF1"],
                    )
                )
                mock_get_ds = stack.enter_context(
                    patch.object(
                        client,
                        "get_datasets_for_indicators",
                        return_value=["JPMAQS_GENERIC_RETURNS"],
                    )
                )
                stack.enter_context(
                    patch.object(
                        client,
                        "_get_effective_snapshot_switchover_ts",
                        return_value=pd.Timestamp("2024-04-01T00:00:00Z"),
                    )
                )
                mock_download = stack.enter_context(
                    patch.object(client, "download_full_snapshot")
                )
                mock_load = stack.enter_context(
                    patch.object(client, "load_data", return_value=pd.DataFrame())
                )

                client.download(
                    tickers=["USD_EQXR_NSA_TRF1"],
                    since_datetime="20240301",
                    to_datetime="20240315",
                    include_delta_files=True,
                    include_metadata_files=False,
                    show_progress=False,
                )

                mock_get_ds.assert_called_once()

                dl_kwargs = mock_download.call_args.kwargs
                self.assertEqual(dl_kwargs["since_datetime"], JPMAQS_EARLIEST_FILE_DATE)
                self.assertEqual(dl_kwargs["to_datetime"], "20240331T235959")
                self.assertFalse(dl_kwargs["include_full_snapshots"])
                self.assertTrue(dl_kwargs["include_delta"])
                self.assertFalse(dl_kwargs["include_metadata"])
                self.assertIn("JPMAQS_GENERIC_RETURNS", dl_kwargs["file_group_ids"])
                self.assertIn(
                    "JPMAQS_GENERIC_RETURNS_DELTA", dl_kwargs["file_group_ids"]
                )

                load_kwargs = mock_load.call_args.kwargs
                self.assertIsNone(load_kwargs["since_datetime"])
                self.assertEqual(load_kwargs["to_datetime"], "20240315")

                _mock_get_client.assert_not_called()

    @patch(
        "macrosynergy.download.dataquery_file_api.get_client_id_secret",
        side_effect=AssertionError(
            "Unexpected credential lookup via get_client_id_secret."
        ),
    )
    @patch("macrosynergy.download.dataquery_file_api.DataQueryFileAPIOauth")
    def test_download_full_history_download_selects_complete_delta_set(
        self, _mock_oauth, _mock_get_client
    ):
        with tempfile.TemporaryDirectory() as td:
            client = DataQueryFileAPIClient(
                client_id="id",
                client_secret="secret",
                out_dir=td,
            )

            available_df = pd.DataFrame(
                [
                    {
                        "file-group-id": "JPMAQS_GENERIC_RETURNS_DELTA",
                        "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240229T235959.parquet",
                        "file-datetime": pd.Timestamp("2024-02-29T23:59:59Z"),
                    },
                    {
                        "file-group-id": "JPMAQS_GENERIC_RETURNS_DELTA",
                        "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240310T120000.parquet",
                        "file-datetime": pd.Timestamp("2024-03-10T12:00:00Z"),
                    },
                    {
                        "file-group-id": "JPMAQS_GENERIC_RETURNS_DELTA",
                        "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240329T235959.parquet",
                        "file-datetime": pd.Timestamp("2024-03-29T23:59:59Z"),
                    },
                    {
                        "file-group-id": "JPMAQS_GENERIC_RETURNS_DELTA",
                        "file-name": "JPMAQS_GENERIC_RETURNS_DELTA_20240331T235959.parquet",
                        "file-datetime": pd.Timestamp("2024-03-31T23:59:59Z"),
                    },
                    {
                        "file-group-id": "JPMAQS_MACROECONOMIC_TRENDS_DELTA",
                        "file-name": "JPMAQS_MACROECONOMIC_TRENDS_DELTA_20240331T235959.parquet",
                        "file-datetime": pd.Timestamp("2024-03-31T23:59:59Z"),
                    },
                    {
                        "file-group-id": "JPMAQS_GENERIC_RETURNS",
                        "file-name": "JPMAQS_GENERIC_RETURNS_METADATA_20240315.json",
                        "file-datetime": pd.Timestamp("2024-03-15T00:00:00Z"),
                    },
                ]
            )

            def fake_filter_available_files_by_datetime(
                *,
                since_datetime,
                to_datetime,
                include_full_snapshots,
                include_delta,
                include_metadata,
            ):
                self.assertEqual(since_datetime, JPMAQS_EARLIEST_FILE_DATE)
                self.assertEqual(to_datetime, "20240331T235959")
                self.assertFalse(include_full_snapshots)
                self.assertTrue(include_delta)
                self.assertFalse(include_metadata)
                return available_df[
                    ~available_df["file-name"].str.lower().str.contains("_metadata")
                ].copy()

            downloaded_df = pd.DataFrame(
                {
                    "file-name": [
                        "JPMAQS_GENERIC_RETURNS_DELTA_20240310T120000.parquet"
                    ],
                    "path": [
                        str(
                            Path(td)
                            / "JPMAQS_GENERIC_RETURNS_DELTA_20240310T120000.parquet"
                        )
                    ],
                }
            )
            Path(downloaded_df.iloc[0]["path"]).write_bytes(b"x")

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        client, "download_catalog_file", return_value="catalog.parquet"
                    )
                )
                stack.enter_context(
                    patch.object(
                        client,
                        "filter_to_valid_tickers",
                        return_value=["USD_EQXR_NSA_TRF1"],
                    )
                )
                stack.enter_context(
                    patch.object(
                        client,
                        "get_datasets_for_indicators",
                        return_value=["JPMAQS_GENERIC_RETURNS"],
                    )
                )
                stack.enter_context(
                    patch.object(
                        client,
                        "_get_effective_snapshot_switchover_ts",
                        return_value=pd.Timestamp("2024-04-01T00:00:00Z"),
                    )
                )
                stack.enter_context(
                    patch.object(
                        client,
                        "filter_available_files_by_datetime",
                        side_effect=fake_filter_available_files_by_datetime,
                    )
                )
                stack.enter_context(
                    patch.object(
                        client, "list_downloaded_files", return_value=downloaded_df
                    )
                )
                mock_dlm = stack.enter_context(
                    patch.object(client, "download_multiple_files")
                )
                stack.enter_context(
                    patch.object(client, "load_data", return_value=pd.DataFrame())
                )

                client.download(
                    tickers=["USD_EQXR_NSA_TRF1"],
                    since_datetime="20240301",
                    to_datetime="20240315",
                    include_delta_files=True,
                    include_metadata_files=False,
                    show_progress=False,
                )

                expected = [
                    "JPMAQS_GENERIC_RETURNS_DELTA_20240229T235959.parquet",
                    "JPMAQS_GENERIC_RETURNS_DELTA_20240329T235959.parquet",
                ]
                called = mock_dlm.call_args.kwargs["filenames"]
                self.assertListEqual(called, expected)

                _mock_get_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
