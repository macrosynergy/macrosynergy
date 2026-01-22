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
    _ensure_columns,
    _to_output_schema,
    _filter_lazy_frame_by_tickers,
    _delete_corrupt_files,
    _large_delta_file_datetimes,
    _expr_split_ticker,
    _lazy_load_filtered_parquets,
)


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
            "grading": [1.0, 3.0],
            "eop_lag": [0.0, 1.0],
            "mop_lag": [0.0, 1.0],
            "last_updated": [
                datetime.datetime(2024, 3, 1, 12, 0, 0),
                datetime.datetime(2024, 3, 2, 12, 0, 0),
            ],
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


class TestLazyLoad(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

        _make_ticker_parquet(
            self.tmpdir / "DATASET1_20240101.parquet",
            {
                "ticker": ["USD_INFL", "EUR_INFL"],
                "real_date": [datetime.date(2023, 1, 1), datetime.date(2023, 1, 1)],
                "value": [1.0, 2.0],
            },
        )
        _make_ticker_parquet(
            self.tmpdir / "DATASET1_20240102.parquet",
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
        _make_ticker_parquet(
            self.tmpdir / "DATASET1_DELTA_20240102T010101.parquet",
            {
                "ticker": ["USD_INFL"],
                "real_date": [datetime.date(2023, 1, 2)],
                "value": [1.2],
            },
        )

        sub_dir = self.tmpdir / "subdir"
        _make_ticker_parquet(
            sub_dir / "DATASET2_20240103.parquet",
            {
                "ticker": ["USD_GROWTH", "GBP_GROWTH"],
                "real_date": [datetime.date(2023, 2, 1), datetime.date(2023, 2, 1)],
                "value": [5.0, 6.0],
            },
        )
        (self.tmpdir / "DATASET2_20240103_METADATA.json").touch()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_list_downloaded_files(self):
        files = _list_downloaded_files(self.tmpdir, file_format="parquet")
        self.assertEqual(len(files), 4)
        filenames = sorted([p.name for p in files])
        self.assertIn("DATASET1_20240101.parquet", filenames)
        self.assertIn("DATASET1_20240102.parquet", filenames)
        self.assertIn("DATASET1_DELTA_20240102T010101.parquet", filenames)
        self.assertIn("DATASET2_20240103.parquet", filenames)

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_downloaded_files_df(self):
        df = _downloaded_files_df(self.tmpdir, file_format="parquet")
        self.assertEqual(len(df), 4)
        self.assertNotIn("DATASET2_20240103_METADATA.json", df["filename"].to_list())
        ds1_latest = df[df["filename"] == "DATASET1_20240102.parquet"].iloc[0]
        self.assertEqual(ds1_latest["dataset"], "DATASET1")
        self.assertEqual(ds1_latest["file-timestamp"], pd.Timestamp("2024-01-02"))

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_downloaded_files_df_effective_dataset_delta_and_non_delta(self):
        _make_ticker_parquet(
            self.tmpdir / "BETA_20240101.parquet",
            {
                "ticker": ["USD_INFL"],
                "real_date": [datetime.date(2024, 1, 1)],
                "value": [1.0],
            },
        )
        (self.tmpdir / "BETA_DELTA_20240101T010101.parquet").touch()

        df = _downloaded_files_df(self.tmpdir, file_format="parquet")

        non_delta = df[df["filename"] == "BETA_20240101.parquet"].iloc[0]
        self.assertEqual(non_delta["dataset"], "BETA")
        self.assertEqual(non_delta["e-dataset"], "BETA")

        delta = df[df["filename"] == "BETA_DELTA_20240101T010101.parquet"].iloc[0]
        self.assertEqual(delta["dataset"], "BETA_DELTA")
        self.assertEqual(delta["e-dataset"], "BETA")

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_filter_to_latest_files(self):
        df = _downloaded_files_df(self.tmpdir, file_format="parquet")
        latest = _filter_to_latest_files(df)
        self.assertEqual(len(latest), 3)
        filenames = latest["filename"].to_list()
        self.assertIn("DATASET1_20240102.parquet", filenames)
        self.assertIn("DATASET2_20240103.parquet", filenames)
        # Delta is older than the latest snapshot for DATASET1 and should not be selected.
        self.assertIn("DATASET1_DELTA_20240102T010101.parquet", filenames)

    @patch("macrosynergy.download.dataquery_file_api.logger")
    def test_filter_to_latest_files_warns_when_window_has_only_deltas(
        self, mock_logger
    ):
        df = pd.DataFrame(
            {
                "path": [
                    self.tmpdir / "DATASET1_20231231.parquet",
                    self.tmpdir / "DATASET1_DELTA_20240101T010101.parquet",
                ],
                "filename": [
                    "DATASET1_20231231.parquet",
                    "DATASET1_DELTA_20240101T010101.parquet",
                ],
                "dataset": ["DATASET1", "DATASET1_DELTA"],
                "e-dataset": ["DATASET1", "DATASET1"],
                "file-timestamp": [
                    pd.Timestamp("2023-12-31T00:00:00Z"),
                    pd.Timestamp("2024-01-01T01:01:01Z"),
                ],
            }
        )
        _filter_to_latest_files(
            files_df=df,
            include_delta_files=True,
            warn_if_no_full_snapshots=True,
            since_datetime="20240101",
            to_datetime="20240101",
        )
        mock_logger.warning.assert_any_call(
            "No full snapshots available in the requested window "
            "since=2024-01-01T00:00:00Z to=2024-01-01T23:59:59Z "
            "earliest_snapshot=2023-12-31T00:00:00Z"
        )

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
            lf_ticker,
            include_file_column=None,
            want_qdf=True,
        ).collect()
        self.assertIn("cid", df_qdf.columns)
        self.assertIn("xcat", df_qdf.columns)
        self.assertNotIn("ticker", df_qdf.columns)

        lf_ticker2 = pl.LazyFrame({"ticker": ["C_D"], "value": [2]})
        df_ticker = _to_output_schema(
            lf_ticker2,
            include_file_column=None,
            want_qdf=False,
        ).collect()
        self.assertIn("ticker", df_ticker.columns)
        self.assertNotIn("cid", df_ticker.columns)
        self.assertNotIn("xcat", df_ticker.columns)
        self.assertEqual(df_ticker["ticker"][0], "C_D")

    def test_filter_lazy_frame_by_tickers(self):
        lf = pl.LazyFrame(
            {
                "ticker": ["A_B", "C_D"],
                "real_date": [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)],
            }
        )
        filt = _filter_lazy_frame_by_tickers(
            lf,
            ["A_B"],
            None,
            None,
            None,
            None,
        )
        self.assertEqual(filt.collect().shape[0], 1)
        self.assertEqual(filt.collect()["ticker"][0], "A_B")

        filt_date = _filter_lazy_frame_by_tickers(
            lf,
            ["A_B", "C_D"],
            "2023-01-15",
            None,
            None,
            None,
        )
        self.assertEqual(filt_date.collect().shape[0], 1)

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_lazy_load_basic_filtering(self):
        df = lazy_load_from_parquets(
            self.tmpdir,
            tickers=["JPY_INFL"],
            since_datetime="20240101",
            include_delta_files=False,
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["cid"], "JPY")
        self.assertEqual(df.iloc[0]["value"], 3.1)

        df_qdf = lazy_load_from_parquets(
            self.tmpdir,
            tickers=["USD_GROWTH"],
            since_datetime="20240101",
            include_delta_files=False,
        )
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
            datasets=["DATASET2"],
            tickers=["USD_GROWTH", "GBP_GROWTH"],
        )
        self.assertEqual(len(df_ds), 2)
        self.assertTrue(set(df_ds["cid"].to_list()) == {"USD", "GBP"})

    def test_lazy_load_raises_when_no_indicators(self):
        with self.assertRaisesRegex(ValueError, "No tickers specified"):
            lazy_load_from_parquets(self.tmpdir, include_delta_files=False)

    def test_lazy_load_raises_when_tickers_not_found(self):
        # Add a local JPMaQS catalog file so validation uses the catalog (not per-parquet scans).
        catalog_file = self.tmpdir / "JPMAQS_METADATA_CATALOG_20240101.parquet"
        _make_qdf_parquet(
            catalog_file,
            {
                "Ticker": [
                    "USD_INFL",
                    "EUR_INFL",
                    "JPY_INFL",
                    "USD_GROWTH",
                    "GBP_GROWTH",
                ],
                "Theme": ["DATASET1"] * 5,
            },
        )
        with self.assertRaisesRegex(ValueError, "JPMaQS catalog"):
            lazy_load_from_parquets(
                self.tmpdir,
                tickers=["ZZZ_DOES_NOT_EXIST"],
                include_delta_files=False,
                catalog_file=str(catalog_file),
            )

    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_lazy_load_output_formats(self):
        pl_df = lazy_load_from_parquets(
            self.tmpdir,
            tickers=["USD_INFL"],
            dataframe_type="polars",
            since_datetime="20240101",
            include_delta_files=False,
        )
        self.assertIsInstance(pl_df, pl.DataFrame)
        self.assertEqual(pl_df.shape[0], 1)

        lazy_df = lazy_load_from_parquets(
            self.tmpdir,
            tickers=["USD_INFL"],
            dataframe_type="polars-lazy",
            since_datetime="20240101",
            include_delta_files=False,
        )
        self.assertIsInstance(lazy_df, pl.LazyFrame)
        self.assertEqual(lazy_df.collect().shape[0], 1)

        df_wide = lazy_load_from_parquets(
            self.tmpdir,
            tickers=["USD_GROWTH"],
            dataframe_format="tickers",
            since_datetime="20240101",
            include_delta_files=False,
        )
        self.assertIn("ticker", df_wide.columns)
        self.assertNotIn("cid", df_wide.columns)
        self.assertEqual(df_wide.iloc[0]["ticker"], "USD_GROWTH")

    def test_lazy_load_datasets_type_validation(self):
        with self.assertRaisesRegex(ValueError, "`datasets` must be a list of strings"):
            lazy_load_from_parquets(
                self.tmpdir,
                tickers=["USD_INFL"],
                datasets="DATASET2",  # invalid type: should be list[str]
                include_delta_files=False,
            )

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
                None,
                None,
                "latest",
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
                None,
                None,
                "latest",
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
                None,
                None,
                "latest",
                "bad",
                "pandas",
                True,
            )


class TestCorruptedFilesHandling(unittest.TestCase):
    def setUp(self):
        # _make_sample_parquet with 4 paths
        self.tmpdir = Path(tempfile.mkdtemp())
        self.created_filenames = [
            "jpmaqs_good1.parquet",
            "jpmaqs_good2.parquet",
            "corrupted.parquet",
            "jpmaqs_corrupted.parquet",
            "jpmaqs_good3.parquet",
        ]
        for fname in self.created_filenames:
            path = self.tmpdir / fname
            if "corrupted" in fname:
                with open(path, "wb") as f:
                    f.write(b"not a parquet file")
            else:
                _make_sample_parquet(path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_delete_corrupt_files(self):
        corrupt_file = "corrupted.parquet"
        corrupt_file_path = self.tmpdir / corrupt_file
        self.assertTrue(corrupt_file_path.exists())
        parquet_files = list(Path(self.tmpdir).glob("*.parquet"))
        _delete_corrupt_files(parquet_files)

        self.assertFalse(corrupt_file_path.exists())
        current_files = list(Path(self.tmpdir).glob("*.parquet"))
        self.assertEqual(len(current_files), 3)
        self.assertFalse(corrupt_file_path in current_files)


class TestLargeDeltaFileDatetimes(unittest.TestCase):
    def test_large_delta_datetimes_formatting(self):
        as_strings = _large_delta_file_datetimes()
        as_timestamps = _large_delta_file_datetimes(as_str=False)

        self.assertGreater(len(as_strings), 0)
        self.assertEqual(len(as_strings), len(as_timestamps))
        self.assertEqual(len(as_strings), len(set(as_strings)))
        self.assertTrue(all(isinstance(x, str) for x in as_strings))
        self.assertTrue(all(isinstance(x, pd.Timestamp) for x in as_timestamps))
        self.assertListEqual(
            as_strings, [ts.strftime("%Y%m%dT%H%M%S") for ts in as_timestamps]
        )
        self.assertTrue(
            all(
                ts.hour == 23 and ts.minute == 59 and ts.second == 59
                for ts in as_timestamps
            )
        )
        self.assertListEqual(as_strings, sorted(as_strings))


class TestLazyLoadFilteredParquets(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.ticker_file = self.tmpdir / "DATASETX_20240101.parquet"
        self.qdf_file = self.tmpdir / "DATASETX_20240102.parquet"

        ticker_df = pl.DataFrame(
            {
                "ticker": ["USD_INFL", "EUR_INFL"],
                "real_date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)],
                "value": [1.0, 2.0],
                "last_updated": [
                    datetime.datetime(2024, 1, 2, 0, 0),
                    datetime.datetime(2024, 1, 2, 0, 0),
                ],
            }
        )
        ticker_df.write_parquet(self.ticker_file)

        qdf_df = pl.DataFrame(
            {
                "ticker": ["USD_INFL", "USD_INFL"],
                "real_date": [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)],
                "value": [10.0, 20.0],
                "last_updated": [
                    datetime.datetime(2024, 1, 3, 0, 0),
                    datetime.datetime(2024, 2, 2, 0, 0),
                ],
            }
        )
        qdf_df.write_parquet(self.qdf_file)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_expr_split_ticker_handles_extra_segments(self):
        cid_expr, xcat_expr = _expr_split_ticker(pl.col("ticker"))
        lf = pl.LazyFrame({"ticker": ["USD_GROWTH_EXTRA", "EUR_CPI"]})
        result = lf.select(cid_expr.alias("cid"), xcat_expr.alias("xcat")).collect()

        self.assertEqual(result["cid"].to_list(), ["USD", "EUR"])
        self.assertEqual(result["xcat"].to_list(), ["GROWTH_EXTRA", "CPI"])

    def test_lazy_load_filtered_parquets_latest_dedup(self):
        lf = _lazy_load_filtered_parquets(
            paths=[str(self.ticker_file), str(self.qdf_file)],
            tickers=["USD_INFL", "EUR_INFL"],
            start_date=None,
            end_date=None,
            min_last_updated=None,
            max_last_updated=None,
            delta_treatment="latest",
            include_file_column="source_file",
            return_qdf=True,
        )
        df = lf.collect()

        self.assertIn("source_file", df.columns)
        self.assertEqual(df.height, 3)

        usd_jan = df.filter(
            (pl.col("cid") == "USD")
            & (pl.col("xcat") == "INFL")
            & (pl.col("real_date") == datetime.date(2024, 1, 1))
        )
        self.assertEqual(usd_jan.height, 1)
        self.assertEqual(usd_jan["value"][0], 10.0)
        self.assertEqual(Path(str(usd_jan["source_file"][0])).name, self.qdf_file.name)

        eur_row = df.filter((pl.col("cid") == "EUR") & (pl.col("xcat") == "INFL"))
        self.assertEqual(eur_row.height, 1)
        self.assertEqual(eur_row["value"][0], 2.0)
        self.assertEqual(
            Path(str(eur_row["source_file"][0])).name, self.ticker_file.name
        )

    def test_lazy_load_filtered_parquets_earliest_ticker_schema(self):
        lf = _lazy_load_filtered_parquets(
            paths=[str(self.ticker_file), str(self.qdf_file)],
            tickers=["USD_INFL"],
            start_date=None,
            end_date=None,
            min_last_updated=None,
            max_last_updated=None,
            delta_treatment="earliest",
            include_file_column="source_file",
            return_qdf=False,
        )
        df = lf.collect()

        self.assertIn("ticker", df.columns)
        self.assertNotIn("cid", df.columns)
        self.assertEqual(df.filter(pl.col("ticker") == "USD_INFL").height, 2)

        usd_jan = df.filter(
            (pl.col("ticker") == "USD_INFL")
            & (pl.col("real_date") == datetime.date(2024, 1, 1))
        )
        self.assertEqual(usd_jan.height, 1)
        self.assertEqual(usd_jan["value"][0], 1.0)
        self.assertEqual(
            Path(str(usd_jan["source_file"][0])).name, self.ticker_file.name
        )

    def test_lazy_load_filtered_parquets_requires_paths(self):
        call_kwargs = dict(
            paths=[],
            tickers=["USD_INFL"],
            start_date=None,
            end_date=None,
            min_last_updated=None,
            max_last_updated=None,
            delta_treatment="all",
            include_file_column=None,
            return_qdf=True,
        )
        with self.assertRaises(ValueError):
            _lazy_load_filtered_parquets(**call_kwargs)


class TestLazyLoadDatasetsFilterSemantics(unittest.TestCase):
    @patch(
        "macrosynergy.download.dataquery_file_api.pd_to_datetime_compat",
        pd_to_datetime_compat,
    )
    def test_datasets_filter_matches_effective_dataset_and_respects_include_delta(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            snap_path = tmpdir / "DATASET1_20240102.parquet"
            delta_path = tmpdir / "DATASET1_DELTA_20240102T010101.parquet"

            _make_ticker_parquet(
                snap_path,
                {
                    "ticker": ["USD_INFL"],
                    "real_date": [datetime.date(2024, 1, 1)],
                    "value": [1.0],
                    "last_updated": [datetime.datetime(2024, 1, 2, 0, 0)],
                },
            )
            _make_ticker_parquet(
                delta_path,
                {
                    "ticker": ["USD_INFL"],
                    "real_date": [datetime.date(2024, 1, 1)],
                    "value": [2.0],
                    "last_updated": [datetime.datetime(2024, 1, 3, 0, 0)],
                },
            )

            # Base dataset selection includes delta files when `include_delta_files=True`.
            df = lazy_load_from_parquets(
                tmpdir,
                datasets=["DATASET1"],
                tickers=["USD_INFL"],
                include_delta_files=True,
                delta_treatment="latest",
            )
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["value"], 2.0)

            # Same dataset selection excludes delta files when `include_delta_files=False`.
            df_no_delta = lazy_load_from_parquets(
                tmpdir,
                datasets=["DATASET1"],
                tickers=["USD_INFL"],
                include_delta_files=False,
                delta_treatment="latest",
            )
            self.assertEqual(len(df_no_delta), 1)
            self.assertEqual(df_no_delta.iloc[0]["value"], 1.0)

            # Passing a delta dataset name still filters by the effective (base) dataset;
            # `include_delta_files` remains the sole control for delta inclusion.
            df_delta_name = lazy_load_from_parquets(
                tmpdir,
                datasets=["DATASET1_DELTA"],
                tickers=["USD_INFL"],
                include_delta_files=True,
                delta_treatment="latest",
            )
            self.assertEqual(len(df_delta_name), 1)
            self.assertEqual(df_delta_name.iloc[0]["value"], 2.0)

            df_delta_name_no_delta = lazy_load_from_parquets(
                tmpdir,
                datasets=["DATASET1_DELTA"],
                tickers=["USD_INFL"],
                include_delta_files=False,
                delta_treatment="latest",
            )
            self.assertEqual(len(df_delta_name_no_delta), 1)
            self.assertEqual(df_delta_name_no_delta.iloc[0]["value"], 1.0)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
