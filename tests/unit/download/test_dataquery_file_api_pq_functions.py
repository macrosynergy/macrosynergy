"""
Tests for the parquet loading path of `macrosynergy.download.dataquery_file_api`.

Organised by the unit under test, in three layers:

1. pure helpers       - no filesystem, no fixture
2. filesystem helpers - one temp directory each
3. end to end         - `lazy_load_from_parquets` over the shared fixture below

Each helper has exactly one home. The end-to-end classes only assert behaviour that
emerges from composition; they do not re-test what the layers below already cover.
"""

import datetime
import functools
import logging
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl

from macrosynergy.compat import PYTHON_3_8_OR_LATER
from macrosynergy.download.dataquery_file_api import (
    EXPECTED_JPMAQS_PARQUET_SCHEMA,
    _apply_delta_treatment,
    _check_lazy_load_inputs,
    _collect_naming_paths,
    _delete_corrupt_files,
    _delete_jpmaqs_file,
    _downloaded_files_df,
    _filter_lazy_frame_by_tickers,
    _filter_to_latest_files,
    _is_jpmaqs_file,
    _list_downloaded_files,
    _read_catalog,
    _scan_and_prepare_single_parquet,
    _scan_check_and_cast_single_parquet,
    _split_jpmaqs_filename,
    _to_output_schema,
    build_filtered_lazy_frames_df,
    lazy_load_from_parquets,
)
from macrosynergy.management.constants import JPMAQS_METRICS


def suppress_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.disable(logging.CRITICAL)
        try:
            return func(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)

    return wrapper


# Dataset ids must be real JPMaQS ones: the catalog resolves Theme -> dataset through
# JPMAQS_DATASET_THEME_MAPPING, and anything unmapped becomes "Unknown" and is skipped.
MACRO_DS = "JPMAQS_MACROECONOMIC_TRENDS"
MACRO_THEME = "Macroeconomic trends"
RETURNS_DS = "JPMAQS_GENERIC_RETURNS"
RETURNS_THEME = "Generic returns"

D1 = datetime.date(2023, 1, 1)
D2 = datetime.date(2023, 1, 2)

# publication timestamps: a snapshot lands at 06:00 UTC, its delta at end of day
LU_SNAPSHOT = datetime.datetime(2024, 1, 2, 6, 0, 0)
LU_DELTA = datetime.datetime(2024, 1, 2, 23, 59, 59)


def write_parquet(path: Path, data: dict) -> Path:
    """Write `data` to `path` verbatim, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_parquet(path)
    return path


def ticker_rows(tickers, dates, values, last_updated, grading=None, **extra) -> dict:
    """Rows in the ticker-based shape JPMaQS actually delivers."""
    n = len(tickers)
    data = {
        "ticker": tickers,
        "real_date": dates,
        "value": values,
        "grading": [1.0] * n if grading is None else grading,
        "eop_lag": [0.0] * n,
        "mop_lag": [0.0] * n,
        "last_updated": last_updated,
    }
    data.update(extra)
    return data


def write_rows(path: Path, tickers, dates, values, lu=LU_SNAPSHOT, **kwargs) -> Path:
    """Write ticker-shaped rows. `lu` is broadcast to every row unless it is a list."""
    if not isinstance(lu, list):
        lu = [lu] * len(tickers)
    return write_parquet(path, ticker_rows(tickers, dates, values, lu, **kwargs))


def write_catalog(path: Path, ticker_themes: dict) -> Path:
    """Minimal stand-in for JPMAQS_METADATA_CATALOG: Ticker -> Theme."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"Ticker": list(ticker_themes), "Theme": list(ticker_themes.values())}
    ).to_parquet(path)
    return path


def ticker_lazyframe(**overrides) -> pl.LazyFrame:
    """A LazyFrame carrying every column `_to_output_schema` requires."""
    data = {
        "real_date": [D2],
        "ticker": ["USD_INFL"],
        "value": [1.1],
        "eop_lag": [0.0],
        "mop_lag": [0.0],
        "grading": [1.0],
        "last_updated": [LU_SNAPSHOT],
    }
    data.update(overrides)
    return pl.LazyFrame(data)


class TempDirCase(unittest.TestCase):
    """Gives each test its own directory in `self.tmpdir`."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir, True)


# ---------------------------------------------------------------------------------
# 1. pure helpers
# ---------------------------------------------------------------------------------


class TestFilenameHelpers(unittest.TestCase):
    def test_split_jpmaqs_filename(self):
        cases = {
            "JPMAQS_GENERIC_RETURNS_20250501.parquet": (
                "JPMAQS_GENERIC_RETURNS",
                "20250501",
            ),
            "JPMAQS_X_DELTA_20240102T235959.parquet": (
                "JPMAQS_X_DELTA",
                "20240102T235959",
            ),
            "JPMAQS_METADATA_NOTIFICATIONS_20260119.json": (
                "JPMAQS_METADATA_NOTIFICATIONS",
                "20260119",
            ),
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(_split_jpmaqs_filename(filename), expected)

    def test_split_jpmaqs_filename_without_an_underscore_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _split_jpmaqs_filename("file.parquet")
        self.assertIn("Invalid filename format", str(ctx.exception))

    def test_is_jpmaqs_file(self):
        cases = {
            "JPMAQS_GENERIC_RETURNS_20260728.parquet": True,
            "JPMAQS_METADATA_CATALOG_20260728.csv": True,
            "JPMAQS_METADATA_NOTIFICATIONS_20260728T060000.json": True,
            "JPMAQS_X_20260728.PARQUET": True,  # extension is case-insensitive
            "random.parquet": False,
            "jpmaqs_lowercase_20260728.parquet": False,  # prefix is case-sensitive
            "not_JPMAQS_prefixed.parquet": False,
            "JPMAQS_X_20260728.txt": False,  # unsupported extension
            "JPMAQS_X_20260728": False,  # no extension
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(_is_jpmaqs_file(Path("data") / name), expected)


@unittest.skipUnless(PYTHON_3_8_OR_LATER, "Requires Python 3.8+")
class TestFilterLazyFrameByTickers(unittest.TestCase):
    def setUp(self):
        self.lf = pl.LazyFrame(
            {
                "ticker": ["A_B", "C_D"],
                "real_date": [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)],
            }
        )

    def test_filters_by_ticker(self):
        filtered = _filter_lazy_frame_by_tickers(self.lf, ["A_B"], None, None)
        self.assertEqual(filtered.collect()["ticker"].to_list(), ["A_B"])

    def test_empty_ticker_list_does_not_filter(self):
        filtered = _filter_lazy_frame_by_tickers(self.lf, [], None, None)
        self.assertEqual(filtered.collect().shape[0], 2)

    def test_date_bounds(self):
        both = ["A_B", "C_D"]
        cases = [
            ("2023-01-15", None, ["C_D"]),
            (None, "2023-01-15", ["A_B"]),
            ("2023-01-15", "2023-02-15", ["C_D"]),
            (None, None, both),
        ]
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                filtered = _filter_lazy_frame_by_tickers(self.lf, both, start, end)
                self.assertEqual(filtered.collect()["ticker"].to_list(), expected)


@unittest.skipUnless(PYTHON_3_8_OR_LATER, "Requires Python 3.8+")
class TestToOutputSchema(unittest.TestCase):
    """Projects the normalised ticker-shaped frame to the requested shape."""

    def test_ticker_shape_is_passed_through(self):
        df = _to_output_schema(ticker_lazyframe(), want_qdf=False).collect()
        self.assertEqual(df.columns, list(EXPECTED_JPMAQS_PARQUET_SCHEMA))
        self.assertEqual(df["ticker"][0], "USD_INFL")

    def test_qdf_shape_splits_the_ticker_in_place(self):
        df = _to_output_schema(ticker_lazyframe(), want_qdf=True).collect()
        # cid/xcat take ticker's position, so the qdf column order is stable
        self.assertEqual(df.columns[:3], ["real_date", "cid", "xcat"])
        self.assertEqual((df["cid"][0], df["xcat"][0]), ("USD", "INFL"))

    def test_only_the_first_underscore_splits_the_ticker(self):
        df = _to_output_schema(
            ticker_lazyframe(ticker=["USD_DU05YXR_VT10"]), want_qdf=True
        ).collect()
        self.assertEqual((df["cid"][0], df["xcat"][0]), ("USD", "DU05YXR_VT10"))

    def test_source_file_is_kept_only_when_requested(self):
        lf = ticker_lazyframe(source_file=["JPMAQS_X_20240102"])
        kept = _to_output_schema(lf, want_qdf=True, include_source_file=True).collect()
        self.assertEqual(kept["source_file"][0], "JPMAQS_X_20240102")
        dropped = _to_output_schema(lf, want_qdf=True).collect()
        self.assertNotIn("source_file", dropped.columns)

    def test_raises_naming_the_missing_column(self):
        with self.assertRaises(ValueError) as ctx:
            _to_output_schema(
                pl.LazyFrame({"ticker": ["USD_INFL"], "real_date": [D2]}),
                want_qdf=False,
            )
        self.assertIn("value", str(ctx.exception))

    def test_raises_when_source_file_is_requested_but_absent(self):
        with self.assertRaises(ValueError) as ctx:
            _to_output_schema(
                ticker_lazyframe(), want_qdf=True, include_source_file=True
            )
        self.assertIn("source_file", str(ctx.exception))


@unittest.skipUnless(PYTHON_3_8_OR_LATER, "Requires Python 3.8+")
class TestApplyDeltaTreatment(unittest.TestCase):
    """USD_INFL is restated for the same real_date; EUR_INFL is not."""

    def frame(self):
        return pl.LazyFrame(
            {
                "cid": ["USD", "USD", "EUR"],
                "xcat": ["INFL", "INFL", "INFL"],
                "real_date": [D2, D2, D2],
                "value": [1.1, 9.9, 2.1],
                "last_updated": [LU_SNAPSHOT, LU_DELTA, LU_SNAPSHOT],
            }
        )

    def test_treatments(self):
        cases = [("latest", 2, 9.9), ("earliest", 2, 1.1), ("all", 3, None)]
        for treatment, rows, usd_value in cases:
            with self.subTest(treatment=treatment):
                df = _apply_delta_treatment(self.frame(), treatment).collect()
                self.assertEqual(df.shape[0], rows)
                if usd_value is not None:
                    usd = df.filter(pl.col("cid") == "USD")
                    self.assertEqual(usd["value"].to_list(), [usd_value])

    def test_keys_on_ticker_when_not_qdf(self):
        lf = pl.LazyFrame(
            {
                "ticker": ["USD_INFL", "USD_INFL"],
                "real_date": [D2, D2],
                "value": [1.1, 9.9],
                "last_updated": [LU_SNAPSHOT, LU_DELTA],
            }
        )
        df = _apply_delta_treatment(lf, "latest", return_qdf=False).collect()
        self.assertEqual(df["value"].to_list(), [9.9])

    def test_invalid_treatment_raises(self):
        with self.assertRaises(ValueError):
            _apply_delta_treatment(self.frame(), "newest")


@unittest.skipUnless(PYTHON_3_8_OR_LATER, "Requires Python 3.8+")
class TestCollectNamingPaths(unittest.TestCase):
    # the cap that `_collect_naming_paths` applies to the paths it lists
    PATH_CAP = 20

    def failing_frame(self):
        return pl.LazyFrame({"a": ["x"]}).select(pl.col("a").cast(pl.Float64))

    def test_returns_the_frame_when_collect_succeeds(self):
        df = _collect_naming_paths(pl.LazyFrame({"a": [1]}), ["one.parquet"])
        self.assertEqual(df["a"].to_list(), [1])

    def test_failure_names_the_files(self):
        with self.assertRaises(ValueError) as ctx:
            _collect_naming_paths(self.failing_frame(), ["one.parquet", "two.parquet"])
        self.assertIn("one.parquet", str(ctx.exception))
        self.assertIn("two.parquet", str(ctx.exception))

    def test_path_list_is_capped(self):
        paths = [f"f{i}.parquet" for i in range(self.PATH_CAP + 10)]
        with self.assertRaises(ValueError) as ctx:
            _collect_naming_paths(self.failing_frame(), paths)
        message = str(ctx.exception)
        self.assertEqual(message.count(".parquet"), self.PATH_CAP)
        self.assertIn("and 10 more", message)


class TestCheckLazyLoadInputs(TempDirCase):
    def setUp(self):
        super().setUp()
        write_rows(
            self.tmpdir / f"{MACRO_DS}_20240102.parquet",
            ["USD_INFL"],
            [D2],
            [1.1],
        )

    def args(self, **overrides):
        args = dict(
            files_dir=self.tmpdir,
            file_format="parquet",
            tickers=[],
            cids=[],
            xcats=[],
            metrics=["value"],
            start_date=None,
            end_date=None,
            dataframe_format="qdf",
            dataframe_type="pandas",
            categorical_dataframe=True,
        )
        args.update(overrides)
        return list(args.values())

    def test_accepts_valid_arguments(self):
        _check_lazy_load_inputs(*self.args())  # must not raise

    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            _check_lazy_load_inputs(*self.args(files_dir="nonexistent_dir"))

    def test_rejections(self):
        cases = {
            "cids without xcats": dict(cids=["USD"], xcats=None),
            "unknown dataframe_format": dict(dataframe_format="bad"),
            "unknown dataframe_type": dict(dataframe_type="bad"),
            "non-bool categorical_dataframe": dict(categorical_dataframe=1),
            "wide with several metrics": dict(
                dataframe_format="wide", metrics=["value", "grading"]
            ),
            "wide with the default metric set": dict(
                dataframe_format="wide", metrics=list(JPMAQS_METRICS)
            ),
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    _check_lazy_load_inputs(*self.args(**overrides))


# ---------------------------------------------------------------------------------
# 2. filesystem helpers
# ---------------------------------------------------------------------------------


class TestListDownloadedFiles(TempDirCase):
    def setUp(self):
        super().setUp()
        self.top = f"{MACRO_DS}_20240102.parquet"
        (self.tmpdir / self.top).write_bytes(b"x")
        (self.tmpdir / "random.parquet").write_bytes(b"x")  # not a JPMaQS file
        nested = self.tmpdir / "2024-01-02"
        nested.mkdir()
        self.nested = f"{RETURNS_DS}_20240102.parquet"
        (nested / self.nested).write_bytes(b"x")
        (nested / "notes.parquet").write_bytes(b"x")

    @suppress_logging
    def test_recurses_and_excludes_non_jpmaqs_files(self):
        found = [f.name for f in _list_downloaded_files(self.tmpdir, "parquet")]
        self.assertEqual(sorted(found), sorted([self.top, self.nested]))

    @suppress_logging
    def test_filters_by_file_format(self):
        (self.tmpdir / f"{MACRO_DS}_20240102.json").write_bytes(b"{}")
        found = [f.name for f in _list_downloaded_files(self.tmpdir, "json")]
        self.assertEqual(found, [f"{MACRO_DS}_20240102.json"])

    def test_unknown_file_format_raises(self):
        with self.assertRaises(ValueError):
            _list_downloaded_files(self.tmpdir, "xlsx")


class TestDownloadedFilesDf(TempDirCase):
    def setUp(self):
        super().setUp()
        write_rows(
            self.tmpdir / f"{MACRO_DS}_20240102.parquet",
            ["USD_INFL"],
            [D2],
            [1.1],
        )
        write_rows(
            self.tmpdir / f"{MACRO_DS}_DELTA_20240102T235959.parquet",
            ["USD_INFL"],
            [D2],
            [9.9],
            [LU_DELTA],
        )
        # real metadata names carry the _METADATA token before the date
        self.metadata = "JPMAQS_METADATA_CATALOG_20240102.parquet"
        write_rows(
            self.tmpdir / self.metadata, ["USD_INFL"], [D2], [1.1], [LU_SNAPSHOT]
        )

    def test_parses_dataset_type_and_timestamp(self):
        df = _downloaded_files_df(self.tmpdir, "parquet").set_index("file-name")
        snapshot = df.loc[f"{MACRO_DS}_20240102.parquet"]
        self.assertEqual(snapshot["dataset"], MACRO_DS)
        self.assertEqual(snapshot["file-datetime"], "20240102")
        self.assertEqual(snapshot["file-type"], "parquet")
        self.assertEqual(
            snapshot["file-timestamp"], pd.Timestamp("2024-01-02", tz="UTC")
        )

        delta = df.loc[f"{MACRO_DS}_DELTA_20240102T235959.parquet"]
        self.assertEqual(delta["dataset"], f"{MACRO_DS}_DELTA")
        self.assertEqual(
            delta["file-timestamp"], pd.Timestamp("2024-01-02 23:59:59", tz="UTC")
        )

    def test_metadata_files_are_excluded_unless_requested(self):
        without = set(_downloaded_files_df(self.tmpdir, "parquet")["file-name"])
        self.assertNotIn(self.metadata, without)
        with_meta = set(
            _downloaded_files_df(self.tmpdir, "parquet", include_metadata_files=True)[
                "file-name"
            ]
        )
        self.assertIn(self.metadata, with_meta)

    def test_missing_directory_returns_an_empty_frame(self):
        self.assertTrue(_downloaded_files_df(self.tmpdir / "nope", "parquet").empty)


class TestFilterToLatestFiles(TempDirCase):
    """
    Two snapshot dates for one dataset, a delta for the newer one, and a second
    dataset on the newer date.
    """

    def setUp(self):
        super().setUp()
        write_rows(
            self.tmpdir / f"{MACRO_DS}_20240101.parquet",
            ["USD_INFL"],
            [D1],
            [1.0],
            [datetime.datetime(2024, 1, 1, 6)],
        )
        write_rows(
            self.tmpdir / f"{MACRO_DS}_20240102.parquet",
            ["USD_INFL"],
            [D2],
            [1.1],
        )
        write_rows(
            self.tmpdir / f"{MACRO_DS}_DELTA_20240102T235959.parquet",
            ["USD_INFL"],
            [D2],
            [9.9],
            [LU_DELTA],
        )
        write_rows(
            self.tmpdir / f"{RETURNS_DS}_20240102.parquet",
            ["USD_XR"],
            [D2],
            [5.0],
        )
        self.files_df = _downloaded_files_df(self.tmpdir, "parquet")
        self.snapshot = f"{MACRO_DS}_20240102.parquet"
        self.delta = f"{MACRO_DS}_DELTA_20240102T235959.parquet"
        self.other = f"{RETURNS_DS}_20240102.parquet"
        self.older = f"{MACRO_DS}_20240101.parquet"

    def names(self, **kwargs):
        return sorted(_filter_to_latest_files(self.files_df, **kwargs)["file-name"])

    def test_deltas_are_included_by_default(self):
        self.assertEqual(self.names(), sorted([self.snapshot, self.delta, self.other]))

    def test_deltas_can_be_excluded(self):
        self.assertEqual(
            self.names(include_delta_files=False), sorted([self.snapshot, self.other])
        )

    def test_the_older_snapshot_is_always_dropped(self):
        for include_deltas in (True, False):
            with self.subTest(include_delta_files=include_deltas):
                self.assertNotIn(
                    self.older, self.names(include_delta_files=include_deltas)
                )

    def test_effective_dataset_groups_a_delta_with_its_snapshot(self):
        by_name = _filter_to_latest_files(self.files_df).set_index("file-name")
        self.assertEqual(by_name.loc[self.snapshot, "e-dataset"], MACRO_DS)
        self.assertEqual(by_name.loc[self.delta, "e-dataset"], MACRO_DS)
        self.assertEqual(by_name.loc[self.other, "e-dataset"], RETURNS_DS)

    def test_result_is_sorted_with_a_clean_index(self):
        latest = _filter_to_latest_files(self.files_df)
        self.assertEqual(list(latest.index), list(range(len(latest))))
        self.assertEqual(list(latest["dataset"]), sorted(latest["dataset"]))

    def test_empty_input_passes_through(self):
        empty = pd.DataFrame(columns=["file-name", "file-datetime", "file-timestamp"])
        self.assertTrue(_filter_to_latest_files(empty).empty)

    def test_raises_without_a_full_snapshot(self):
        delta_only = self.files_df[self.files_df["file-name"].str.contains("_DELTA")]
        with self.assertRaises(ValueError) as ctx:
            _filter_to_latest_files(delta_only)
        self.assertIn("No full-snapshot files", str(ctx.exception))


class TestFileDeletion(TempDirCase):
    @suppress_logging
    def test_deletes_jpmaqs_files_and_refuses_others(self):
        good = self.tmpdir / f"{MACRO_DS}_20240102.parquet"
        bad = self.tmpdir / "not_a_jpmaqs_file.parquet"
        good.write_bytes(b"x")
        bad.write_bytes(b"x")

        self.assertTrue(_delete_jpmaqs_file(good))
        self.assertFalse(good.exists())
        self.assertFalse(_delete_jpmaqs_file(bad))
        self.assertTrue(bad.exists())

    @suppress_logging
    def test_deleting_a_missing_jpmaqs_file_is_not_an_error(self):
        missing = self.tmpdir / f"{MACRO_DS}_20240102.parquet"
        self.assertTrue(_delete_jpmaqs_file(missing))

    @suppress_logging
    def test_delete_corrupt_files(self):
        good = write_rows(
            self.tmpdir / f"{MACRO_DS}_20240102.parquet",
            ["USD_INFL"],
            [D2],
            [1.1],
        )
        corrupt = self.tmpdir / f"{MACRO_DS}_20240103.parquet"
        corrupt.write_bytes(b"not a parquet file")
        not_jpmaqs = self.tmpdir / "corrupted.parquet"
        not_jpmaqs.write_bytes(b"not a parquet file")

        removed = _delete_corrupt_files([good, corrupt, not_jpmaqs])

        self.assertEqual(removed, [str(corrupt)])
        self.assertTrue(good.exists())
        self.assertFalse(corrupt.exists())
        self.assertTrue(not_jpmaqs.exists())  # JPMaQS files only, never touched


@unittest.skipUnless(PYTHON_3_8_OR_LATER, "Requires Python 3.8+")
class TestScanCheckAndCastSingleParquet(TempDirCase):
    """Every parquet is normalised to the ticker-based schema on scan."""

    def setUp(self):
        super().setUp()
        self.path = self.tmpdir / f"{MACRO_DS}_20240102.parquet"

    def ticker_file(self, **kwargs):
        return write_rows(self.path, ["USD_INFL"], [D2], [1.1], [LU_SNAPSHOT], **kwargs)

    def test_output_matches_the_expected_schema(self):
        schema = dict(
            _scan_check_and_cast_single_parquet(self.ticker_file()).collect_schema()
        )
        for column, dtype in EXPECTED_JPMAQS_PARQUET_SCHEMA.items():
            with self.subTest(column=column):
                self.assertEqual(schema[column], dtype)

    def test_metric_dtypes_are_coerced(self):
        path = write_parquet(
            self.path,
            {
                "ticker": ["USD_INFL"],
                "real_date": [D2],
                "value": [1.1],
                "grading": ["1.5"],  # string
                "eop_lag": [0],  # int
                "mop_lag": [1],  # int
                "last_updated": [LU_SNAPSHOT],
            },
        )
        df = _scan_check_and_cast_single_parquet(path).collect()
        self.assertEqual(df.schema["grading"], pl.Float64)
        self.assertEqual(df["grading"].to_list(), [1.5])
        self.assertEqual(df.schema["eop_lag"], pl.Float64)
        self.assertEqual(df.schema["mop_lag"], pl.Float64)

    def test_missing_metric_columns_are_backfilled_as_typed_nulls(self):
        path = write_parquet(
            self.path, {"ticker": ["USD_INFL"], "real_date": [D2], "value": [1.1]}
        )
        df = _scan_check_and_cast_single_parquet(path).collect()
        for column in ("grading", "eop_lag", "mop_lag", "last_updated"):
            with self.subTest(column=column):
                self.assertTrue(df[column].is_null().all())
                self.assertEqual(
                    df.schema[column], EXPECTED_JPMAQS_PARQUET_SCHEMA[column]
                )

    def test_legacy_qdf_schema_is_converted_and_warns(self):
        path = write_parquet(
            self.path,
            {
                "cid": ["USD"],
                "xcat": ["INFL"],
                "real_date": [D2],
                "value": [1.1],
                "grading": [1.0],
                "eop_lag": [0.0],
                "mop_lag": [0.0],
                "last_updated": [LU_SNAPSHOT],
            },
        )
        with self.assertWarns(UserWarning) as cm:
            df = _scan_check_and_cast_single_parquet(path).collect()
        self.assertIn("modified schema", str(cm.warning))
        self.assertIn(self.path.name, str(cm.warning))  # names the offending file
        self.assertEqual(df["ticker"].to_list(), ["USD_INFL"])
        self.assertNotIn("cid", df.columns)

    def test_unusable_schemas_raise(self):
        cases = {
            "cid without xcat": (
                {"cid": ["USD"], "real_date": [D2], "value": [1.1]},
                "both 'cid' and 'xcat'",
            ),
            "no ticker and no cid": (
                {"real_date": [D2], "value": [1.1]},
                "ticker",
            ),
            "no real_date": (
                {"ticker": ["USD_INFL"], "value": [1.1]},
                "real_date",
            ),
        }
        for index, (label, (data, expected)) in enumerate(cases.items()):
            with self.subTest(case=label):
                path = write_parquet(
                    self.tmpdir / f"{MACRO_DS}_2024010{index}.parquet", data
                )
                with self.assertRaises(ValueError) as ctx:
                    _scan_check_and_cast_single_parquet(path)
                self.assertIn(expected, str(ctx.exception))

    def test_source_file_column(self):
        for categorical, dtype in [(True, pl.Categorical), (False, pl.String)]:
            with self.subTest(categorical=categorical):
                df = _scan_check_and_cast_single_parquet(
                    self.ticker_file(),
                    include_source_file=True,
                    categorical_source_file_column=categorical,
                ).collect()
                self.assertEqual(df["source_file"].to_list(), [f"{MACRO_DS}_20240102"])
                self.assertEqual(df.schema["source_file"], dtype)

    def test_a_pre_existing_source_file_column_raises_when_requested(self):
        path = self.ticker_file(source_file=["somewhere_else"])
        with self.assertRaises(ValueError) as ctx:
            _scan_check_and_cast_single_parquet(path, include_source_file=True)
        self.assertIn("source_file", str(ctx.exception))

    def test_a_pre_existing_source_file_column_is_projected_away_otherwise(self):
        path = self.ticker_file(source_file=["somewhere_else"])
        prepared = _scan_and_prepare_single_parquet(
            path=path,
            tickers=["USD_INFL"],
            start_date=None,
            end_date=None,
            return_qdf=True,
        )
        self.assertNotIn("source_file", prepared.collect().columns)


class TestReadCatalog(TempDirCase):
    def setUp(self):
        super().setUp()
        _read_catalog.cache_clear()
        self.addCleanup(_read_catalog.cache_clear)
        self.path = write_catalog(
            self.tmpdir / "JPMAQS_METADATA_CATALOG_20240102.parquet",
            {"USD_INFL": MACRO_THEME},
        )

    def test_reads_the_catalog(self):
        self.assertEqual(_read_catalog(self.path)["Ticker"].tolist(), ["USD_INFL"])

    def test_repeated_reads_hit_the_cache(self):
        calls = []
        real = pd.read_parquet

        def spy(path, *args, **kwargs):
            calls.append(path)
            return real(path, *args, **kwargs)

        with patch("pandas.read_parquet", spy):
            _read_catalog(self.path)
            _read_catalog(self.path)
        self.assertEqual(len(calls), 1)

    def test_callers_share_one_frame_and_must_copy_before_mutating(self):
        # the contract its docstring states; build_filtered_lazy_frames_df copies
        self.assertIs(_read_catalog(self.path), _read_catalog(self.path))


# ---------------------------------------------------------------------------------
# 3. end to end
# ---------------------------------------------------------------------------------


@unittest.skipUnless(PYTHON_3_8_OR_LATER, "Requires Python 3.8+")
class LazyLoadFixture(unittest.TestCase):
    """
    Two datasets laid out the way `download_files` writes them:

        JPMAQS_MACROECONOMIC_TRENDS_20240101.parquet         USD_INFL, EUR_INFL on D1
        JPMAQS_MACROECONOMIC_TRENDS_20240102.parquet         USD_INFL, EUR_INFL on D2
        JPMAQS_MACROECONOMIC_TRENDS_DELTA_20240102T235959    USD_INFL on D2, restated 9.9
        subdir/JPMAQS_GENERIC_RETURNS_20240102.parquet       USD_XR on D2

    The catalog lives outside the data directory so it is never scanned as data. It also
    holds ZZZ_MYSTERY under an unmapped theme, for the "Unknown dataset" path.

    `load()` includes delta files, as `download()` does. `load_snapshot_only()` excludes
    them, so values come from the snapshot and are unaffected by delta resolution.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.catalog_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.addCleanup(shutil.rmtree, self.catalog_dir, True)

        self.older_snapshot = write_rows(
            self.tmpdir / f"{MACRO_DS}_20240101.parquet",
            ["USD_INFL", "EUR_INFL"],
            [D1, D1],
            [1.0, 2.0],
            [datetime.datetime(2024, 1, 1, 6)] * 2,
        )
        # grading arrives as a string in some snapshots; it is cast on scan
        self.latest_snapshot = write_rows(
            self.tmpdir / f"{MACRO_DS}_20240102.parquet",
            ["USD_INFL", "EUR_INFL"],
            [D2, D2],
            [1.1, 2.1],
            grading=["1.0", "2.0"],
        )
        self.latest_delta = write_rows(
            self.tmpdir / f"{MACRO_DS}_DELTA_20240102T235959.parquet",
            ["USD_INFL"],
            [D2],
            [9.9],
            [LU_DELTA],
        )
        self.returns_snapshot = write_rows(
            self.tmpdir / "subdir" / f"{RETURNS_DS}_20240102.parquet",
            ["USD_XR"],
            [D2],
            [5.0],
        )
        self.catalog_path = write_catalog(
            self.catalog_dir / "JPMAQS_METADATA_CATALOG_20240102.parquet",
            {
                "USD_INFL": MACRO_THEME,
                "EUR_INFL": MACRO_THEME,
                "USD_XR": RETURNS_THEME,
                "ZZZ_MYSTERY": "Not a real theme",
            },
        )

    def load(self, **kwargs):
        """Load from the fixture, scoped to MACRO_DS unless told otherwise."""
        kwargs.setdefault("catalog_path", self.catalog_path)
        kwargs.setdefault("datasets", [MACRO_DS])
        return lazy_load_from_parquets(self.tmpdir, **kwargs)

    def load_snapshot_only(self, **kwargs):
        kwargs.setdefault("include_delta_files", False)
        return self.load(**kwargs)


class TestBuildFilteredLazyFramesDf(LazyLoadFixture):
    def build(self, paths, tickers, **kwargs):
        kwargs.setdefault("start_date", None)
        kwargs.setdefault("end_date", None)
        kwargs.setdefault("return_qdf", True)
        return build_filtered_lazy_frames_df(
            paths=paths, tickers=tickers, catalog_path=self.catalog_path, **kwargs
        )

    def test_maps_each_dataset_to_its_files_and_tickers(self):
        df = self.build(
            [self.latest_snapshot, self.latest_delta, self.returns_snapshot],
            ["USD_INFL", "EUR_INFL", "USD_XR"],
        ).set_index("e-dataset")

        self.assertEqual(sorted(df.index), sorted([MACRO_DS, RETURNS_DS]))
        # a snapshot and its delta land on the same dataset row
        self.assertEqual(
            sorted(Path(p).name for p in df.loc[MACRO_DS, "path"]),
            sorted([self.latest_snapshot.name, self.latest_delta.name]),
        )
        self.assertEqual(df.loc[MACRO_DS, "ticker"], ["EUR_INFL", "USD_INFL"])
        self.assertTrue(df["lazyframe"].notna().all())
        self.assertNotIn("_merge", df.columns)  # the merge indicator stays internal

    def test_lazyframes_resolve_to_the_requested_tickers(self):
        df = self.build([self.latest_snapshot], ["USD_INFL"])
        out = df.iloc[0]["lazyframe"].collect()
        self.assertEqual(out["cid"].unique().to_list(), ["USD"])

    def test_string_paths_are_accepted(self):
        df = self.build([str(self.latest_snapshot)], ["USD_INFL"])
        self.assertEqual(list(df["e-dataset"]), [MACRO_DS])

    @suppress_logging
    def test_tickers_of_an_unmapped_theme_are_skipped(self):
        df = self.build([self.latest_snapshot], ["USD_INFL", "ZZZ_MYSTERY"])
        self.assertEqual(list(df["e-dataset"]), [MACRO_DS])

    def test_unmatched_datasets_raise_naming_the_missing_side(self):
        cases = {
            "requested ticker with no file": (
                [self.latest_snapshot],
                ["USD_INFL", "USD_XR"],
                "no downloaded file",
            ),
            "file no ticker was requested for": (
                [self.latest_snapshot, self.returns_snapshot],
                ["USD_INFL"],
                "no requested tickers",
            ),
        }
        for label, (paths, tickers, expected) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as ctx:
                    self.build(paths, tickers)
                self.assertIn(RETURNS_DS, str(ctx.exception))
                self.assertIn(expected, str(ctx.exception))


class TestLazyLoadQdf(LazyLoadFixture):
    """The default output: a QuantamentalDataFrame."""

    QDF_COLUMNS = [
        "real_date",
        "cid",
        "xcat",
        "value",
        "eop_lag",
        "mop_lag",
        "grading",
        "last_updated",
    ]

    def test_columns_and_values(self):
        df = self.load_snapshot_only(tickers=["USD_INFL"])
        self.assertEqual(list(df.columns), self.QDF_COLUMNS)
        self.assertEqual(len(df), 1)  # the older snapshot is not loaded
        row = df.iloc[0]
        self.assertEqual((row["cid"], row["xcat"], row["value"]), ("USD", "INFL", 1.1))
        self.assertEqual(row["grading"], 1.0)  # the string grading was cast

    def test_cids_and_xcats_expand_to_tickers(self):
        df = self.load_snapshot_only(cids=["USD", "EUR"], xcats=["INFL"])
        self.assertEqual(sorted(df["cid"].astype(str)), ["EUR", "USD"])

    def test_ticker_casing_is_resolved_against_the_catalog(self):
        df = self.load_snapshot_only(tickers=["usd_infl"])
        self.assertEqual(sorted(df["cid"].astype(str).unique()), ["USD"])

    def test_datasets_argument_restricts_the_files_read(self):
        df = self.load_snapshot_only(tickers=["USD_XR"], datasets=[RETURNS_DS])
        self.assertEqual(df["xcat"].astype(str).to_list(), ["XR"])

    def test_several_datasets_load_together(self):
        df = self.load_snapshot_only(tickers=["USD_INFL", "USD_XR"], datasets=None)
        self.assertEqual(sorted(df["xcat"].astype(str)), ["INFL", "XR"])

    def test_date_bounds(self):
        within = self.load_snapshot_only(tickers=["USD_INFL"], start_date="2023-01-02")
        self.assertEqual(
            within["real_date"].dt.date.astype(str).to_list(), ["2023-01-02"]
        )
        outside = self.load_snapshot_only(tickers=["USD_INFL"], end_date="2022-12-31")
        self.assertEqual(len(outside), 0)

    def test_metrics_subset_keeps_the_qdf_column_order(self):
        df = self.load_snapshot_only(tickers=["USD_INFL"], metrics=["grading", "value"])
        self.assertEqual(
            list(df.columns), ["real_date", "cid", "xcat", "value", "grading"]
        )

    def test_categorical_dataframe_flag(self):
        on = self.load_snapshot_only(tickers=["USD_INFL"])
        self.assertEqual(on["cid"].dtype.name, "category")
        off = self.load_snapshot_only(tickers=["USD_INFL"], categorical_dataframe=False)
        self.assertNotEqual(off["cid"].dtype.name, "category")


class TestLazyLoadOutputFormats(LazyLoadFixture):
    """`dataframe_format` and `dataframe_type` combinations."""

    def wide(self, **kwargs):
        kwargs.setdefault("tickers", ["USD_INFL", "EUR_INFL"])
        return self.load_snapshot_only(
            dataframe_format="wide", metrics=["value"], **kwargs
        )

    def test_tickers_format(self):
        df = self.load_snapshot_only(tickers=["USD_INFL"], dataframe_format="tickers")
        self.assertEqual(list(df.columns), list(EXPECTED_JPMAQS_PARQUET_SCHEMA))
        self.assertEqual(df.iloc[0]["ticker"], "USD_INFL")

    def test_polars_and_polars_lazy_types(self):
        as_polars = self.load_snapshot_only(
            tickers=["USD_INFL"], dataframe_type="polars"
        )
        self.assertIsInstance(as_polars, pl.DataFrame)
        as_lazy = self.load_snapshot_only(
            tickers=["USD_INFL"], dataframe_type="polars-lazy"
        )
        self.assertIsInstance(as_lazy, pl.LazyFrame)
        self.assertEqual(as_lazy.collect().shape[0], as_polars.shape[0])

    def test_wide_pandas_indexes_by_date(self):
        df = self.wide()
        self.assertEqual(sorted(df.columns), ["EUR_INFL", "USD_INFL"])
        self.assertEqual(df.index.name, "real_date")
        self.assertEqual(df.loc[pd.Timestamp(D2), "USD_INFL"], 1.1)

    def test_wide_polars_keeps_the_date_as_a_column(self):
        df = self.wide(dataframe_type="polars")
        self.assertEqual(df.columns, ["real_date", "EUR_INFL", "USD_INFL"])
        self.assertEqual(df.sort("real_date")["USD_INFL"].to_list()[-1], 1.1)

    def test_wide_polars_lazy_is_sorted_by_date(self):
        dates = self.wide(dataframe_type="polars-lazy").collect()["real_date"].to_list()
        self.assertEqual(dates, sorted(dates))

    def test_wide_columns_are_the_requested_tickers_on_both_backends(self):
        # USD_XR has no rows in MACRO_DS, so it covers "requested but absent"
        for dataframe_type in ("pandas", "polars"):
            with self.subTest(dataframe_type=dataframe_type):
                df = self.wide(dataframe_type=dataframe_type)
                columns = [c for c in df.columns if c != "real_date"]
                self.assertEqual(sorted(columns), ["EUR_INFL", "USD_INFL"])


class TestLazyLoadDeltas(LazyLoadFixture):
    """A delta restating a row must supersede it in every output shape."""

    def test_deltas_are_included_by_default(self):
        df = self.load(tickers=["USD_INFL"])
        usd = df[df["real_date"] == pd.Timestamp(D2)]
        self.assertEqual(usd["value"].tolist(), [9.9])

    def test_qdf_keeps_one_row_per_cid_xcat_date(self):
        df = self.load(tickers=["USD_INFL", "EUR_INFL"])
        self.assertFalse(df.duplicated(subset=["cid", "xcat", "real_date"]).any())
        by_cid = df.set_index(df["cid"].astype(str))["value"]
        self.assertEqual(by_cid["USD"], 9.9)  # restated by the delta
        self.assertEqual(by_cid["EUR"], 2.1)  # not restated

    def test_tickers_format_keeps_one_row_per_ticker_date(self):
        df = self.load(tickers=["USD_INFL", "EUR_INFL"], dataframe_format="tickers")
        self.assertFalse(df.duplicated(subset=["ticker", "real_date"]).any())
        usd = df[(df["ticker"] == "USD_INFL") & (df["real_date"] == pd.Timestamp(D2))]
        self.assertEqual(usd["value"].tolist(), [9.9])

    def test_wide_format_resolves_restated_rows(self):
        for dataframe_type in ("pandas", "polars"):
            with self.subTest(dataframe_type=dataframe_type):
                df = self.load(
                    tickers=["USD_INFL", "EUR_INFL"],
                    dataframe_format="wide",
                    metrics=["value"],
                    dataframe_type=dataframe_type,
                )
                if dataframe_type == "pandas":
                    value = df.loc[pd.Timestamp(D2), "USD_INFL"]
                else:
                    value = df.filter(pl.col("real_date") == D2)["USD_INFL"][0]
                self.assertEqual(value, 9.9)


class TestLazyLoadSourceFile(LazyLoadFixture):
    """`include_source_file` in the returned frame. Dtypes are covered at scan level."""

    def test_absent_unless_requested(self):
        self.assertNotIn("source_file", self.load(tickers=["USD_INFL"]).columns)

    def test_appended_last_and_categorical_in_pandas(self):
        df = self.load_snapshot_only(tickers=["USD_INFL"], include_source_file=True)
        self.assertEqual(list(df.columns)[-1], "source_file")
        self.assertEqual(df.iloc[0]["source_file"], f"{MACRO_DS}_20240102")
        self.assertEqual(df["source_file"].dtype.name, "category")

    def test_names_the_file_whose_row_survived_dedup(self):
        df = self.load(tickers=["USD_INFL", "EUR_INFL"], include_source_file=True)
        by_cid = df.set_index(df["cid"].astype(str))["source_file"].astype(str)
        self.assertEqual(by_cid["USD"], f"{MACRO_DS}_DELTA_20240102T235959")
        self.assertEqual(by_cid["EUR"], f"{MACRO_DS}_20240102")

    def test_survives_a_metrics_subset(self):
        df = self.load_snapshot_only(
            tickers=["USD_INFL"], metrics=["value"], include_source_file=True
        )
        self.assertEqual(
            list(df.columns), ["real_date", "cid", "xcat", "value", "source_file"]
        )

    def test_listing_it_in_metrics_warns_and_enables_the_flag(self):
        with self.assertWarns(UserWarning) as cm:
            df = self.load_snapshot_only(
                tickers=["USD_INFL"], metrics=["value", "source_file"]
            )
        self.assertIn("include_source_file", str(cm.warning))
        self.assertIn("source_file", df.columns)
        self.assertEqual(df.iloc[0]["value"], 1.1)  # the requested metric survives


class TestLazyLoadValidation(LazyLoadFixture):
    """Input mistakes are reported, not turned into silent or cryptic failures."""

    def test_missing_catalog_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            lazy_load_from_parquets(
                self.tmpdir, tickers=["USD_INFL"], catalog_path=None
            )
        self.assertIn("catalog_path", str(ctx.exception))

    def test_non_bool_source_file_flags_raise(self):
        for flag in ("include_source_file", "categorical_source_file_column"):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError) as ctx:
                    self.load(tickers=["USD_INFL"], **{flag: 1})
                self.assertIn(flag, str(ctx.exception))

    def test_unknown_tickers_are_reported(self):
        with self.assertWarns(UserWarning) as cm:
            df = self.load(tickers=["USD_INFL", "ZZZ_TYPO"])
        self.assertIn("ZZZ_TYPO", str(cm.warning))
        self.assertEqual(sorted(df["cid"].astype(str).unique()), ["USD"])

    def test_invalid_cid_xcat_combinations_are_reported(self):
        with self.assertWarns(UserWarning) as cm:
            df = self.load(cids=["USD", "ZZZ"], xcats=["INFL"])
        self.assertIn("ZZZ_INFL", str(cm.warning))
        self.assertEqual(sorted(df["cid"].astype(str).unique()), ["USD"])

    def test_all_tickers_unknown_raises(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError) as ctx:
                self.load(tickers=["ZZZ_TYPO"])
        self.assertIn("ZZZ_TYPO", str(ctx.exception))

    def test_requested_ticker_without_a_downloaded_file_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.load(tickers=["USD_INFL", "USD_XR"])
        self.assertIn(RETURNS_DS, str(ctx.exception))

    def test_wide_rejects_include_source_file(self):
        with self.assertRaises(ValueError) as ctx:
            self.load(
                tickers=["USD_INFL"],
                dataframe_format="wide",
                metrics=["value"],
                include_source_file=True,
            )
        self.assertIn("include_source_file", str(ctx.exception))

    def test_caller_arguments_are_not_mutated(self):
        tickers = ["USD_INFL"]
        metrics = ["value", "source_file"]
        constant = list(JPMAQS_METRICS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.load(tickers=tickers, cids=["EUR"], xcats=["INFL"], metrics=metrics)
        self.assertEqual(tickers, ["USD_INFL"])
        self.assertEqual(metrics, ["value", "source_file"])
        self.assertEqual(JPMAQS_METRICS, constant)


if __name__ == "__main__":
    unittest.main()
