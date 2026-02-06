import unittest
import tempfile
from pathlib import Path

import pandas as pd

from macrosynergy.download.dataquery_file_api import FileSelector, JPMAQS_DATASET_THEME_MAPPING


"""
Every day JPMaQS publish all data as multiple parquet datasets and related metadata files.

All data files share the common schema of:
- real_date: date
- ticker: string
- value: float
- grading: float
- eop_lag: float
- mop_lag: float
- last_updated: datetime

The "full view" of the dataset is meant to be unique on (real_date, ticker, last_updated).
To arrive at a 'canonical' representation (what should be used for analysis), 
the last_updated value is used to pick the most recent value for each (real_date, ticker) pair.


The files are divided into datasets such as "JPMAQS_GENERIC_RETURNS", "JPMAQS_MACROECONOMIC_TRENDS", etc. - these are "snapshot" files.
These datasets also have complementary "delta" datasets, named "JPMAQS_GENERIC_RETURNS_DELTA", "JPMAQS_MACROECONOMIC_TRENDS_DELTA", etc. - these are delta files.


Snapshot files, are published every day, with all 'real_date' values for the given dataset for all tickers.
Delta files are published intra-daily, and contain only rows that have "updated" since the last delta file was published.
The snapshot for real_date T, will have entries up to T-1, and will contain ONLY the latest rows for each (real_date, ticker) pair.
Delta files however, contain multiple rows for the same (real_date, ticker) which are unique on (real_date, ticker, last_updated).
If all delta files were collected and stitched together, they would contain all the rows that have ever been published for the dataset.

There are also historical delta files, published at the last second of every month (23:59:59 on the last day of the month).
These are a stitched/glued version of all the delta files published in that month, and contain all the rows published in that month.
The source then deletes the smaller delta files, as well as the snapshots, and the users are expected to use the historical delta files.

To get a canonical representation of the dataset of any given day delta files, the user would need to:

1) Download a snapshot file. Download ALL deltas since the snapshot file's last_updated value, up to the day of interest.
2) Download absolutely all delta files since the beginning of time, and stitch them together, and select as required. 

Given that the source removes snapshot files, for days in far past, only option 2 is available.

The FileSelector class is responsible for evaluting which files should be downloaded, as well as returning a list of files
that need to be loaded to arrive at the canonical representation of the dataset for a given day.]

The JPMAQS_METADATA_CATALOG, when paired with mapping in the `JPMAQS_DATASET_THEME_MAPPING` dictionary, allows us to identify
which datasets contain which tickers.
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




