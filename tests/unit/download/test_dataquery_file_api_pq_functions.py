import unittest
import tempfile
import datetime
from pathlib import Path
from unittest.mock import patch
import functools
import logging
import polars as pl
import pandas as pd
import shutil

from macrosynergy.download.dataquery_file_api import (
    _check_lazy_load_inputs,
    _list_downloaded_files,
    _downloaded_files_df,
    _filter_to_latest_files,
    lazy_load_from_parquets,
    _identify_schema_type,
    JPMaQSParquetSchemaKind,
    _ensure_columns,
    _to_output_schema,
    _filter_lazy_frame_by_tickers,
    _delete_corrupt_files,
    _delete_jpmaqs_file,
    _is_jpmaqs_file,
)
from macrosynergy.compat import PYTHON_3_8_OR_LATER


def suppress_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.disable(logging.CRITICAL)
        try:
            return func(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)

    return wrapper


def _make_sample_parquet(path: Path) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "ticker": ["USD_GROWTH", "JPY_INFL"],
            "real_date": [
                datetime.date(2024, 1, 31),
                datetime.date(2024, 2, 29),
            ],
            "value": [1.1, 2.2],
            "grading": ["A", "B"],
            "eop_lag": [0, 1],
            "mop_lag": [0, 1],
            "last_updated": ["2024-03-01", "2024-03-02"],
        }
    )
    df.write_parquet(path)
    return df


def _make_ticker_parquet(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_parquet(path)


def _make_qdf_parquet(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_parquet(path)


def pd_to_datetime_compat(x, **kwargs):
    return pd.to_datetime(x, errors="coerce", **kwargs)


@unittest.skipUnless(PYTHON_3_8_OR_LATER, "Requires Python 3.8+")
class TestLazyLoad(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

        _make_ticker_parquet(
            self.tmpdir / "JPMAQS_DATASET1_20240101.parquet",
            {
                "ticker": ["USD_INFL", "EUR_INFL"],
                "real_date": [datetime.date(2023, 1, 1), datetime.date(2023, 1, 1)],
                "value": [1.0, 2.0],
            },
        )
        _make_ticker_parquet(
            self.tmpdir / "JPMAQS_DATASET1_20240102.parquet",
            {
                "ticker": ["USD_INFL", "EUR_INFL", "JPY_INFL"],
                "real_date": [
                    datetime.date(2023, 1, 2),
                    datetime.date(2023, 1, 2),
                    datetime.date(2023, 1, 2),
                ],
                "value": [1.1, 2.1, 3.1],
            },
        )
        (self.tmpdir / "JPMAQS_DATASET1_20240102_DELTA.parquet").touch()

        sub_dir = self.tmpdir / "subdir"
        _make_qdf_parquet(
            sub_dir / "JPMAQS_DATASET2_20240103.parquet",
            {
                "cid": ["USD", "GBP"],
                "xcat": ["GROWTH", "GROWTH"],
                "real_date": [datetime.date(2023, 2, 1), datetime.date(2023, 2, 1)],
                "value": [5.0, 6.0],
            },
        )
        (self.tmpdir / "JPMAQS_DATASET2_20240103_METADATA.json").touch()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_list_downloaded_files(self):
        files = _list_downloaded_files(self.tmpdir, file_format="parquet")
        self.assertEqual(len(files), 4)
        filenames = sorted([p.name for p in files])
        self.assertIn("JPMAQS_DATASET1_20240101.parquet", filenames)
        self.assertIn("JPMAQS_DATASET1_20240102.parquet", filenames)
        self.assertIn("JPMAQS_DATASET1_20240102_DELTA.parquet", filenames)
        self.assertIn("JPMAQS_DATASET2_20240103.parquet", filenames)

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_downloaded_files_df(self):
        df = _downloaded_files_df(self.tmpdir, file_format="parquet")
        self.assertEqual(len(df), 4)
        self.assertNotIn(
            "JPMAQS_DATASET2_20240103_METADATA.json", df["filename"].to_list()
        )
        ds1_latest = df[df["filename"] == "JPMAQS_DATASET1_20240102.parquet"].iloc[0]
        self.assertEqual(ds1_latest["dataset"], "JPMAQS_DATASET1")
        self.assertEqual(ds1_latest["file-timestamp"], pd.Timestamp("2024-01-02"))

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_filter_to_latest_files(self):
        df = _downloaded_files_df(self.tmpdir, file_format="parquet")
        latest = _filter_to_latest_files(df)
        self.assertEqual(len(latest), 2)
        filenames = latest["filename"].to_list()
        self.assertIn("JPMAQS_DATASET1_20240102.parquet", filenames)
        self.assertIn("JPMAQS_DATASET2_20240103.parquet", filenames)
        self.assertNotIn("JPMAQS_DATASET1_20240102_DELTA.parquet", filenames)

    def test_identify_schema_type(self):
        lf_ticker = pl.LazyFrame({"ticker": ["A_B"], "value": [1]})
        lf_qdf = pl.LazyFrame({"cid": ["A"], "xcat": ["B"], "value": [1]})
        lf_bad = pl.LazyFrame({"col1": ["A"], "col2": ["B"]})
        self.assertEqual(
            _identify_schema_type(lf_ticker), JPMaQSParquetSchemaKind.TICKER
        )
        self.assertEqual(_identify_schema_type(lf_qdf), JPMaQSParquetSchemaKind.QDF)
        with self.assertRaises(ValueError):
            _identify_schema_type(lf_bad)

    def test_ensure_columns(self):
        lf = pl.LazyFrame({"a": [1], "b": [2]})
        ensured = _ensure_columns(lf, ["a", "c", "d"])
        df = ensured.collect()
        self.assertIn("a", df.columns)
        self.assertIn("c", df.columns)
        self.assertIn("d", df.columns)
        self.assertTrue(df["c"].is_null().all())

    def test_to_output_schema(self):
        lf_ticker = pl.LazyFrame({"ticker": ["A_B"], "value": [1]})
        df_qdf = _to_output_schema(
            lf_ticker, JPMaQSParquetSchemaKind.TICKER, want_qdf=True
        ).collect()
        self.assertIn("cid", df_qdf.columns)
        self.assertIn("xcat", df_qdf.columns)
        self.assertNotIn("ticker", df_qdf.columns)

        lf_qdf = pl.LazyFrame({"cid": ["C"], "xcat": ["D"], "value": [2]})
        df_ticker = _to_output_schema(
            lf_qdf, JPMaQSParquetSchemaKind.QDF, want_qdf=False
        ).collect()
        self.assertIn("ticker", df_ticker.columns)
        self.assertNotIn("cid", df_ticker.columns)
        self.assertEqual(df_ticker["ticker"][0], "C_D")

    def test_filter_lazy_frame_by_tickers(self):
        lf = pl.LazyFrame(
            {
                "ticker": ["A_B", "C_D"],
                "real_date": [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)],
            }
        )
        filt = _filter_lazy_frame_by_tickers(
            lf, JPMaQSParquetSchemaKind.TICKER, ["A_B"], None, None
        )
        self.assertEqual(filt.collect().shape[0], 1)
        self.assertEqual(filt.collect()["ticker"][0], "A_B")

        filt_date = _filter_lazy_frame_by_tickers(
            lf, JPMaQSParquetSchemaKind.TICKER, ["A_B", "C_D"], "2023-01-15", None
        )
        self.assertEqual(filt_date.collect().shape[0], 2)

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_lazy_load_basic_filtering(self):
        df = lazy_load_from_parquets(self.tmpdir, tickers=["JPY_INFL"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["cid"], "JPY")
        self.assertEqual(df.iloc[0]["value"], 3.1)

        df_qdf = lazy_load_from_parquets(self.tmpdir, cids=["USD"], xcats=["GROWTH"])
        self.assertEqual(len(df_qdf), 1)
        self.assertEqual(df_qdf.iloc[0]["cid"], "USD")
        self.assertEqual(df_qdf.iloc[0]["xcat"], "GROWTH")
        self.assertEqual(df_qdf.iloc[0]["value"], 5.0)

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_lazy_load_date_and_dataset_filters(self):
        df = lazy_load_from_parquets(
            self.tmpdir, tickers=["USD_GROWTH"], end_date="2023-01-31"
        )
        self.assertEqual(len(df), 0)

        df_ds = lazy_load_from_parquets(
            self.tmpdir,
            datasets=["JPMAQS_DATASET2"],
            cids=["USD", "GBP"],
            xcats=["GROWTH"],
        )
        self.assertEqual(len(df_ds), 2)
        self.assertTrue(set(df_ds["cid"].to_list()) == {"USD", "GBP"})

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_lazy_load_output_formats(self):
        pl_df = lazy_load_from_parquets(
            self.tmpdir, tickers=["USD_INFL"], dataframe_type="polars"
        )
        self.assertIsInstance(pl_df, pl.DataFrame)
        self.assertEqual(pl_df.shape[0], 1)

        lazy_df = lazy_load_from_parquets(
            self.tmpdir, tickers=["USD_INFL"], dataframe_type="polars-lazy"
        )
        self.assertIsInstance(lazy_df, pl.LazyFrame)
        self.assertEqual(lazy_df.collect().shape[0], 1)

        df_wide = lazy_load_from_parquets(
            self.tmpdir, cids=["USD"], xcats=["GROWTH"], dataframe_format="tickers"
        )
        self.assertIn("ticker", df_wide.columns)
        self.assertNotIn("cid", df_wide.columns)
        self.assertEqual(df_wide.iloc[0]["ticker"], "USD_GROWTH")

    def test_check_lazy_load_inputs_raises(self):
        with self.assertRaises(FileNotFoundError):
            _check_lazy_load_inputs(
                "nonexistent_dir",
                "parquet",
                [],
                [],
                [],
                [],
                None,
                None,
                "qdf",
                "pandas",
                True,
            )
        with self.assertRaises(ValueError):
            _check_lazy_load_inputs(
                self.tmpdir,
                "parquet",
                [],
                ["USD"],
                None,
                [],
                None,
                None,
                "qdf",
                "pandas",
                True,
            )
        with self.assertRaises(ValueError):
            _check_lazy_load_inputs(
                self.tmpdir,
                "parquet",
                [],
                [],
                [],
                [],
                None,
                None,
                "bad",
                "pandas",
                True,
            )


class TestCorruptedFilesHandling(unittest.TestCase):
    def setUp(self):
        # _make_sample_parquet with 4 paths
        self.tmpdir = Path(tempfile.mkdtemp())
        self.created_filenames = [
            "JPMAQS_GOOD_20260701.parquet",
            "JPMAQS_GOOD_20260702.parquet",
            "JPMAQS_CORRUPT_20260703.parquet",
            "JPMAQS_GOOD_20260704.parquet",
        ]
        for fname in self.created_filenames:
            path = self.tmpdir / fname
            if "CORRUPT" in fname:
                with open(path, "wb") as f:
                    f.write(b"not a parquet file")
            else:
                _make_sample_parquet(path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_delete_corrupt_files(self):
        corrupt_file_path = self.tmpdir / "JPMAQS_CORRUPT_20260703.parquet"
        self.assertTrue(corrupt_file_path.exists())
        parquet_files = list(Path(self.tmpdir).glob("*.parquet"))
        _delete_corrupt_files(parquet_files)

        self.assertFalse(corrupt_file_path.exists())
        current_files = list(Path(self.tmpdir).glob("*.parquet"))
        self.assertEqual(len(current_files), 3)
        self.assertFalse(corrupt_file_path in current_files)


class TestJpmaqsFileDeletion(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @suppress_logging
    def test_delete_jpmaqs_file_refuses_non_jpmaqs(self):
        good = self.tmpdir / "JPMAQS_GENERIC_RETURNS_20260728.parquet"
        bad = self.tmpdir / "not_a_jpmaqs_file.parquet"
        good.write_bytes(b"x")
        bad.write_bytes(b"x")

        self.assertTrue(_delete_jpmaqs_file(good))
        self.assertFalse(good.exists())

        self.assertFalse(_delete_jpmaqs_file(bad))
        self.assertTrue(bad.exists())  # non-JPMaQS file is never deleted

    @suppress_logging
    def test_delete_corrupt_files_ignores_non_jpmaqs(self):
        bad = self.tmpdir / "corrupted.parquet"
        bad.write_bytes(b"not a parquet file")
        _delete_corrupt_files([bad])
        self.assertTrue(bad.exists())  # non-JPMaQS file left untouched

    def test_is_jpmaqs_file_variants(self):
        for name in (
            "JPMAQS_GENERIC_RETURNS_20260728.parquet",
            "JPMAQS_METADATA_CATALOG_20260728.csv",
            "JPMAQS_METADATA_NOTIFICATIONS_20260728T060000.json",
            "JPMAQS_X_20260728.PARQUET",  # extension case-insensitive
        ):
            self.assertTrue(_is_jpmaqs_file(self.tmpdir / name), name)
        for name in (
            "random.parquet",
            "jpmaqs_lowercase_20260728.parquet",  # prefix is case-sensitive
            "not_JPMAQS_prefixed.parquet",
            "JPMAQS_X_20260728.txt",  # unsupported extension
            "JPMAQS_X_20260728",  # no extension
        ):
            self.assertFalse(_is_jpmaqs_file(self.tmpdir / name), name)

    @suppress_logging
    def test_delete_jpmaqs_file_missing_is_ok(self):
        missing = self.tmpdir / "JPMAQS_GENERIC_RETURNS_20260728.parquet"
        # file does not exist; JPMaQS-named deletion is a no-op, not an error
        self.assertTrue(_delete_jpmaqs_file(missing))

    @suppress_logging
    def test_list_downloaded_files_excludes_non_jpmaqs(self):
        (self.tmpdir / "JPMAQS_GENERIC_RETURNS_20260728.parquet").write_bytes(b"x")
        (self.tmpdir / "random.parquet").write_bytes(b"x")
        names = [f.name for f in _list_downloaded_files(self.tmpdir, "parquet")]
        self.assertEqual(names, ["JPMAQS_GENERIC_RETURNS_20260728.parquet"])

    @suppress_logging
    def test_list_downloaded_files_recurses_subdirs(self):
        sub = self.tmpdir / "2026-07-28"
        sub.mkdir()
        (sub / "JPMAQS_GENERIC_RETURNS_20260728.parquet").write_bytes(b"x")
        (sub / "notes.parquet").write_bytes(b"x")
        names = [f.name for f in _list_downloaded_files(self.tmpdir, "parquet")]
        self.assertEqual(names, ["JPMAQS_GENERIC_RETURNS_20260728.parquet"])


if __name__ == "__main__":
    unittest.main()
