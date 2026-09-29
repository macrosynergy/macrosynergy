"""
Tests for `macrosynergy.download.data_explorer`.

No network: delta files are written to a temp directory in the layout
`DataQueryFileAPIClient.download_file` produces, then read back through the real
`_downloaded_files_df` parser so the tests exercise the same frame the explorer sees.
"""

import datetime
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import polars as pl

from macrosynergy.download.data_explorer import (
    BASE_SHARD,
    DeltaFileLoader,
    DeltaManifest,
    compile_delta_files,
    scan_individual_file,
    transform_delta_qdf_to_revisions_matrix,
)
from macrosynergy.download.dataquery_file_api import _downloaded_files_df

DATASET = "JPMAQS_GENERIC_RETURNS"
TICKER = "USD_EQXR_NSA"


def _write_delta_file(root: Path, file_datetime: str, n_rows: int = 3) -> Path:
    """Write one delta file where `download_file` would put it, and return its path."""
    file_path = (
        root
        / pd.Timestamp(file_datetime).strftime("%Y-%m-%d")
        / f"{DATASET}_DELTA_{file_datetime}.parquet"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "real_date": [datetime.date(2025, 1, i + 1) for i in range(n_rows)],
            "ticker": [TICKER] * n_rows,
            "value": [float(i) for i in range(n_rows)],
            "grading": [1.0] * n_rows,
            "eop_lag": [0.0] * n_rows,
            "mop_lag": [0.0] * n_rows,
            "last_updated": [pd.Timestamp(file_datetime).to_pydatetime()] * n_rows,
        }
    ).write_parquet(file_path)
    return file_path


class TestScanIndividualFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.file_path = _write_delta_file(self.temp_dir, "20250902T060000")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_metric_subset_is_honoured(self):
        lazy_df = scan_individual_file(self.file_path, metrics=["value"])
        self.assertEqual(
            lazy_df.collect_schema().names(), ["ticker", "real_date", "value"]
        )

    def test_unavailable_metrics_are_dropped(self):
        lazy_df = scan_individual_file(self.file_path, metrics=["value", "not_a_metric"])
        self.assertEqual(
            lazy_df.collect_schema().names(), ["ticker", "real_date", "value"]
        )

    def test_last_updated_filter_survives_projection(self):
        # `last_updated` is filtered on but not selected: the filter must still apply
        kept = scan_individual_file(
            self.file_path, metrics=["value"], min_last_updated="20250901T000000"
        ).collect()
        dropped = scan_individual_file(
            self.file_path, metrics=["value"], min_last_updated="20260101T000000"
        ).collect()
        self.assertEqual(kept.height, 3)
        self.assertEqual(dropped.height, 0)


class TestDeltaManifest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_round_trip(self):
        manifest = DeltaManifest(self.temp_dir)
        self.assertEqual(manifest.consumed(), set())
        manifest.record(DATASET, BASE_SHARD, ["a.parquet", "b.parquet"], rows=6)
        manifest.save()

        reloaded = DeltaManifest(self.temp_dir)
        self.assertEqual(reloaded.consumed(), {"a.parquet", "b.parquet"})
        self.assertEqual(
            reloaded.shard_paths(DATASET),
            [self.temp_dir / DATASET / f"{BASE_SHARD}.parquet"],
        )

    def test_record_merges_sources_for_an_existing_shard(self):
        manifest = DeltaManifest(self.temp_dir)
        manifest.record(DATASET, "20250902", ["a.parquet"], rows=3)
        manifest.record(DATASET, "20250902", ["b.parquet"], rows=6)
        self.assertEqual(manifest.consumed(), {"a.parquet", "b.parquet"})
        self.assertEqual(manifest.datasets[DATASET]["20250902"]["rows"], 6)


class TestCompileDeltaFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.download_dir = self.temp_dir / "jpmaqs-delta-explorer"
        self.manifest = DeltaManifest(self.temp_dir / "jpmaqs-delta-compiled")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _compile(self, *file_datetimes, n_rows=3):
        for file_datetime in file_datetimes:
            _write_delta_file(self.download_dir, file_datetime, n_rows=n_rows)
        files_df = _downloaded_files_df(self.download_dir, include_metadata_files=True)
        return compile_delta_files(files_df, self.manifest)

    def test_first_run_compiles_all_history_into_one_base_shard(self):
        self._compile("20250901T060000", "20250902T060000")

        shards = self.manifest.datasets[DATASET]
        self.assertEqual(list(shards), [BASE_SHARD])
        self.assertEqual(shards[BASE_SHARD]["rows"], 6)
        base_path = self.manifest.root / DATASET / f"{BASE_SHARD}.parquet"
        self.assertTrue(base_path.exists())
        self.assertEqual(pl.read_parquet(base_path).height, 6)

    def test_later_runs_shard_by_day(self):
        self._compile("20250901T060000")
        self._compile("20250902T060000", "20250903T060000")

        self.assertEqual(
            sorted(self.manifest.datasets[DATASET]), ["20250902", "20250903", BASE_SHARD]
        )

    def test_second_delta_on_the_same_day_rewrites_that_shard(self):
        self._compile("20250901T060000")
        self._compile("20250902T060000")
        self._compile("20250902T180000")

        shard = self.manifest.datasets[DATASET]["20250902"]
        self.assertEqual(shard["rows"], 6)
        self.assertEqual(
            shard["sources"],
            [
                f"{DATASET}_DELTA_20250902T060000.parquet",
                f"{DATASET}_DELTA_20250902T180000.parquet",
            ],
        )
        shard_path = self.manifest.root / DATASET / "20250902.parquet"
        self.assertEqual(pl.read_parquet(shard_path).height, 6)

    def test_compiled_files_are_deleted_and_empty_dirs_pruned(self):
        self._compile("20250901T060000")
        self.assertFalse((self.download_dir / "2025-09-01").exists())

    def test_already_compiled_files_are_skipped(self):
        self._compile("20250901T060000")
        # the delete failed / was interrupted: the file is back on disk but consumed
        _write_delta_file(self.download_dir, "20250901T060000")
        files_df = _downloaded_files_df(self.download_dir, include_metadata_files=True)
        compile_delta_files(files_df, self.manifest)

        self.assertEqual(list(self.manifest.datasets[DATASET]), [BASE_SHARD])
        self.assertEqual(self.manifest.datasets[DATASET][BASE_SHARD]["rows"], 3)


class TestDeltaFileLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.download_dir = self.temp_dir / "jpmaqs-delta-explorer"
        self.manifest = DeltaManifest(self.temp_dir / "jpmaqs-delta-compiled")
        _write_delta_file(self.download_dir, "20250901T060000")
        _write_delta_file(self.download_dir, "20250902T060000")
        compile_delta_files(
            _downloaded_files_df(self.download_dir, include_metadata_files=True),
            self.manifest,
        )
        # the second file lands in its own shard, so the loader must read both
        _write_delta_file(self.download_dir, "20250903T060000")
        compile_delta_files(
            _downloaded_files_df(self.download_dir, include_metadata_files=True),
            self.manifest,
        )
        self.catalog_df = pd.DataFrame(
            {"Ticker": [TICKER], "Theme": ["Generic returns"]}
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_reads_every_shard_for_the_dataset(self):
        loader = DeltaFileLoader(self.manifest, self.catalog_df)
        self.assertEqual(loader.load_ticker_data(TICKER).collect().height, 9)

    def test_ticker_lookup_is_case_insensitive(self):
        loader = DeltaFileLoader(self.manifest, self.catalog_df)
        self.assertEqual(loader.load_ticker_data(TICKER.lower()).collect().height, 9)

    def test_unknown_ticker_raises(self):
        loader = DeltaFileLoader(self.manifest, self.catalog_df)
        with self.assertRaises(ValueError):
            loader.load_ticker_data("NOT_A_TICKER")

    def test_uncompiled_dataset_raises(self):
        catalog_df = pd.DataFrame(
            {"Ticker": ["USD_XGDP_NSA"], "Theme": ["Macroeconomic trends"]}
        )
        loader = DeltaFileLoader(self.manifest, catalog_df)
        with self.assertRaises(ValueError):
            loader.load_ticker_data("USD_XGDP_NSA")

    def test_catalog_df_is_not_mutated(self):
        DeltaFileLoader(self.manifest, self.catalog_df)
        self.assertNotIn("Dataset", self.catalog_df.columns)


class TestTransformDeltaQdfToVintage(unittest.TestCase):
    def _delta_df(self, last_updated, values, metric="value"):
        return pd.DataFrame(
            {
                "ticker": [TICKER] * len(values),
                "real_date": [datetime.date(2025, 1, 1)] * len(values),
                "last_updated": pd.to_datetime(last_updated),
                metric: values,
            }
        )

    def test_metric_argument_is_honoured(self):
        df = self._delta_df(["2025-09-01T06:00:00"], [7.0], metric="grading")
        out = transform_delta_qdf_to_revisions_matrix(df, metric="grading")
        self.assertEqual(out.to_numpy().tolist(), [[7.0]])

    def test_updates_after_the_cutoff_roll_to_the_next_day(self):
        df = self._delta_df(["2025-09-01T06:00:00", "2025-09-01T23:00:00"], [1.0, 2.0])
        out = transform_delta_qdf_to_revisions_matrix(df, end_of_day_time="12:00:00")
        self.assertEqual(
            list(out.columns), [datetime.date(2025, 9, 1), datetime.date(2025, 9, 2)]
        )

    def test_updates_before_the_cutoff_collapse_to_one_release(self):
        df = self._delta_df(["2025-09-01T06:00:00", "2025-09-01T23:00:00"], [1.0, 2.0])
        out = transform_delta_qdf_to_revisions_matrix(df, end_of_day_time="23:59:59")
        self.assertEqual(list(out.columns), [datetime.date(2025, 9, 1)])
        self.assertEqual(out.to_numpy().tolist(), [[2.0]])  # latest update wins

    def test_missing_metric_raises(self):
        df = self._delta_df(["2025-09-01T06:00:00"], [1.0])
        with self.assertRaises(ValueError):
            transform_delta_qdf_to_revisions_matrix(df, metric="eop_lag")

    def test_multiple_tickers_raise(self):
        df = self._delta_df(["2025-09-01T06:00:00", "2025-09-01T07:00:00"], [1.0, 2.0])
        df["ticker"] = [TICKER, "EUR_EQXR_NSA"]
        with self.assertRaises(ValueError):
            transform_delta_qdf_to_revisions_matrix(df)


if __name__ == "__main__":
    unittest.main()
