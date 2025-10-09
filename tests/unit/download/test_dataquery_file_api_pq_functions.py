import unittest
import tempfile
import datetime
from pathlib import Path
from unittest.mock import patch
import functools
import logging
import polars as pl

from macrosynergy.download.dataquery_file_api import (
    _atomic_sink_csv,
    _atomic_sink_parquet,
    convert_ticker_based_parquet_file_to_qdf_pl,
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
    """Create a small valid parquet file for testing."""
    df = pl.DataFrame(
        {
            "ticker": ["US_GROWTH", "JP_INFL"],
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


class TestQDFConvertPolars(unittest.TestCase):
    """Unit tests for Polars-based QDF conversion functions."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        # Clean up temp dir completely
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

    # ---------- Tests ----------

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
        self.assertEqual(got["cid"].to_list(), ["US", "JP"])
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
        self.assertEqual(got["cid"].to_list(), ["US", "JP"])

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


if __name__ == "__main__":
    unittest.main()
