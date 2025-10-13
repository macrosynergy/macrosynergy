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
    _atomic_sink_csv,
    _atomic_sink_parquet,
    convert_ticker_based_parquet_file_to_qdf_pl,
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


@unittest.skipUnless(PYTHON_3_8_OR_LATER, "Requires Python 3.8+")
class TestQDFConvertPolars(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        for item in self.tmpdir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    for sub in item.iterdir():
                        sub.unlink(missing_ok=True)
                    item.rmdir()
            except Exception:
                pass
        try:
            self.tmpdir.rmdir()
        except Exception:
            pass

    def _read_csv_df(self, path: Path) -> pl.DataFrame:
        return pl.read_csv(path, infer_schema_length=0)

    def _no_sidecars(self) -> bool:
        return not list(self.tmpdir.glob(".*.inprogress"))

    def test_passthrough_to_csv_without_qdf(self):
        src = self.tmpdir / "input.parquet"
        original = _make_sample_parquet(src)

        convert_ticker_based_parquet_file_to_qdf_pl(
            filename=str(src), as_csv=True, qdf=False, keep_raw_data=True
        )

        out_csv = self.tmpdir / "input.csv"
        self.assertTrue(out_csv.is_file())

        got = self._read_csv_df(out_csv).with_columns(
            pl.col("real_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )

        self.assertSetEqual(set(got.columns), set(original.columns))
        self.assertEqual(
            sorted(got["ticker"].to_list()), sorted(original["ticker"].to_list())
        )
        self.assertTrue(self._no_sidecars())

    def test_qdf_parquet_overwrite_in_place(self):
        src = self.tmpdir / "market.parquet"
        _make_sample_parquet(src)

        convert_ticker_based_parquet_file_to_qdf_pl(
            filename=str(src), as_csv=False, qdf=True, keep_raw_data=False
        )

        self.assertTrue(src.is_file())
        got = pl.read_parquet(src)
        self.assertNotIn("ticker", got.columns)
        self.assertTrue({"cid", "xcat"}.issubset(set(got.columns)))
        self.assertEqual(got["cid"].to_list(), ["USD", "JPY"])
        self.assertEqual(got["xcat"].to_list(), ["GROWTH", "INFL"])
        self.assertTrue(self._no_sidecars())

    def test_qdf_parquet_keep_raw_data(self):
        src = self.tmpdir / "series.parquet"
        _make_sample_parquet(src)

        convert_ticker_based_parquet_file_to_qdf_pl(
            filename=str(src), as_csv=False, qdf=True, keep_raw_data=True
        )

        qdf_path = self.tmpdir / "series_qdf.parquet"
        self.assertTrue(src.is_file())
        self.assertTrue(qdf_path.is_file())

        got = pl.read_parquet(qdf_path)
        self.assertNotIn("ticker", got.columns)
        self.assertTrue({"cid", "xcat"}.issubset(set(got.columns)))
        self.assertEqual(got["cid"].to_list(), ["USD", "JPY"])

    def test_qdf_csv_outputs_selected_columns(self):
        src = self.tmpdir / "x.parquet"
        _make_sample_parquet(src)

        convert_ticker_based_parquet_file_to_qdf_pl(
            filename=str(src), as_csv=True, qdf=True, keep_raw_data=True
        )

        out_csv = self.tmpdir / "x_qdf.csv"
        self.assertTrue(out_csv.is_file())

        got = self._read_csv_df(out_csv)
        expected = {
            "real_date",
            "value",
            "grading",
            "eop_lag",
            "mop_lag",
            "last_updated",
            "cid",
            "xcat",
        }
        self.assertTrue(expected.issubset(set(got.columns)))
        self.assertNotIn("ticker", got.columns)

    @suppress_logging
    def test_missing_file_raises(self):
        missing = self.tmpdir / "nope.parquet"
        with self.assertRaises(FileNotFoundError):
            convert_ticker_based_parquet_file_to_qdf_pl(filename=str(missing))

    def test_atomic_sink_parquet_cleans_sidecar_on_failure(self):
        src = self.tmpdir / "boom.parquet"
        _make_sample_parquet(src)

        lf = pl.scan_parquet(str(src))
        final_out = self.tmpdir / "target.parquet"
        sidecar = self.tmpdir / ".target.parquet.inprogress"

        with patch.object(
            pl.LazyFrame, "sink_parquet", side_effect=RuntimeError("fail")
        ):
            with self.assertRaises(RuntimeError):
                _atomic_sink_parquet(lf, final_out, sidecar, compression="zstd")

        self.assertFalse(sidecar.exists())
        self.assertFalse(final_out.exists())

    def test_atomic_sink_csv_cleans_sidecar_on_failure(self):
        src = self.tmpdir / "boomcsv.parquet"
        _make_sample_parquet(src)

        lf = pl.scan_parquet(str(src))
        final_out = self.tmpdir / "out.csv"
        sidecar = self.tmpdir / ".out.csv.inprogress"

        with patch.object(
            pl.LazyFrame, "sink_csv", side_effect=RuntimeError("csvfail")
        ):
            with self.assertRaises(RuntimeError):
                _atomic_sink_csv(lf, final_out, sidecar)

        self.assertFalse(sidecar.exists())
        self.assertFalse(final_out.exists())

    @suppress_logging
    def test_inplace_overwrite_preserves_source_on_failure(self):
        src = self.tmpdir / "inplace.parquet"
        original_df = _make_sample_parquet(src)

        with patch.object(pl.LazyFrame, "sink_parquet", side_effect=IOError("forced")):
            with self.assertRaises(IOError):
                convert_ticker_based_parquet_file_to_qdf_pl(
                    filename=str(src), as_csv=False, qdf=True, keep_raw_data=False
                )

        self.assertTrue(src.is_file())
        roundtrip = pl.read_parquet(src)
        self.assertSetEqual(set(roundtrip.columns), set(original_df.columns))
        self.assertEqual(roundtrip.shape, original_df.shape)
        self.assertTrue(self._no_sidecars())

    def test_qdf_handles_multi_part_xcat_and_various_cids(self):
        src = self.tmpdir / "variety.parquet"
        tickers = [
            "USD_GROWTH_X1_D1M1",
            "CAD_GDP_XY",
            "JPY_INFL",
            "INR_PROD_ABC_DEF",
            "CNY_SALES_A1",
            "CHF_CPI_12M",
            "EUR_RATE_ABC_12",
            "AUD_TRADE_BAL",
        ]
        df = pl.DataFrame(
            {
                "ticker": tickers,
                "real_date": [datetime.date(2024, 1, 1)] * len(tickers),
                "value": list(range(len(tickers))),
                "grading": ["A"] * len(tickers),
                "eop_lag": [0] * len(tickers),
                "mop_lag": [0] * len(tickers),
                "last_updated": ["2024-03-01"] * len(tickers),
            }
        )
        df.write_parquet(src)

        convert_ticker_based_parquet_file_to_qdf_pl(
            filename=str(src), as_csv=False, qdf=True, keep_raw_data=True
        )

        qdf_path = self.tmpdir / "variety_qdf.parquet"
        got = pl.read_parquet(qdf_path).select("cid", "xcat")

        expected_cid = [t.split("_", 1)[0] for t in tickers]
        expected_xcat = [t.split("_", 1)[1] for t in tickers]

        self.assertEqual(got["cid"].to_list(), expected_cid)
        self.assertEqual(got["xcat"].to_list(), expected_xcat)


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
        (self.tmpdir / "DATASET1_20240102_DELTA.parquet").touch()

        sub_dir = self.tmpdir / "subdir"
        _make_qdf_parquet(
            sub_dir / "DATASET2_20240103.parquet",
            {
                "cid": ["USD", "GBP"],
                "xcat": ["GROWTH", "GROWTH"],
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
        self.assertIn("DATASET1_20240102_DELTA.parquet", filenames)
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
    def test_filter_to_latest_files(self):
        df = _downloaded_files_df(self.tmpdir, file_format="parquet")
        latest = _filter_to_latest_files(df)
        self.assertEqual(len(latest), 2)
        filenames = latest["filename"].to_list()
        self.assertIn("DATASET1_20240102.parquet", filenames)
        self.assertIn("DATASET2_20240103.parquet", filenames)
        self.assertNotIn("DATASET1_20240102_DELTA.parquet", filenames)

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
            datasets=["DATASET2"],
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


if __name__ == "__main__":
    unittest.main()
