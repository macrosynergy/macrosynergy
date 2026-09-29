import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import pandas as pd
import polars as pl

from macrosynergy.download.dataquery_file_api import (
    JPMAQS_DATASET_THEME_MAPPING,
    JPMAQS_EARLIEST_FILE_DATE,
    JPMAQS_METRICS,
    DataQueryFileAPIClient,
    _delete_jpmaqs_file,
    pd_to_datetime_compat,
    utc_now,
)

MANIFEST_NAME = "_manifest.json"
BASE_SHARD = "base"


class DataQueryFileAPIClientAdapter(DataQueryFileAPIClient):
    def _get_save_dir(self) -> str:
        """
        Override the save directory to use `jpmaqs-delta-explorer` under the
        base directory if the base directory is not already `jpmaqs-delta-explorer`.
        """
        base_dir = Path(self.out_dir)
        if base_dir.name != "jpmaqs-delta-explorer":
            return str(base_dir / "jpmaqs-delta-explorer")
        return str(base_dir)


def scan_individual_file(
    file_path: Union[str, Path],
    tickers: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_last_updated: Optional[str] = None,
    min_last_updated: Optional[str] = None,
    include_source_file: bool = False,
    categorical_ticker_column: bool = True,
    categorical_source_file_column: bool = True,
) -> pl.LazyFrame:
    """
    Scan a single Parquet file and return a Polars LazyFrame.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    lazy_df = pl.scan_parquet(str(file_path))
    lf_schema = lazy_df.collect_schema()
    key_cols = ["ticker", "real_date"]
    assert all(col in lf_schema for col in key_cols), (
        f"File {file_path} does not contain required columns: {key_cols}"
    )
    if include_source_file:
        if "source_file" in lf_schema:
            raise ValueError(
                "The column 'source_file' already exists in the file. Cannot add it again."
            )
        filename = Path(file_path).name.split(".")[0]
        lazy_df = lazy_df.with_columns(pl.lit(filename).alias("source_file"))

    # only the metrics the file actually carries; a requested subset is honoured as given
    available_metrics = [
        col for col in lf_schema if col in JPMAQS_METRICS and col not in key_cols
    ]
    metrics = [m for m in metrics or [] if m in available_metrics] or available_metrics
    if include_source_file:
        metrics = metrics + ["source_file"]

    # filter before selecting, so a projection dropping `last_updated` cannot break it
    if tickers is not None:
        lazy_df = lazy_df.filter(pl.col("ticker").is_in(tickers))
    if start_date is not None:
        lazy_df = lazy_df.filter(
            pl.col("real_date") >= pd_to_datetime_compat(start_date).date()
        )
    if end_date is not None:
        lazy_df = lazy_df.filter(
            pl.col("real_date") <= pd_to_datetime_compat(end_date).date()
        )
    # `last_updated` is stored tz-naive in UTC, so drop the tz before comparing
    if max_last_updated is not None:
        lazy_df = lazy_df.filter(
            pl.col("last_updated")
            <= pd_to_datetime_compat(max_last_updated).tz_localize(None)
        )
    if min_last_updated is not None:
        lazy_df = lazy_df.filter(
            pl.col("last_updated")
            >= pd_to_datetime_compat(min_last_updated).tz_localize(None)
        )

    lazy_df = lazy_df.select(key_cols + metrics)

    if categorical_ticker_column:
        lazy_df = lazy_df.with_columns(pl.col("ticker").cast(pl.Categorical))
    if include_source_file and categorical_source_file_column:
        lazy_df = lazy_df.with_columns(pl.col("source_file").cast(pl.Categorical))

    return lazy_df


class DeltaManifest(object):
    """
    Record of which delta files have been compiled into which Parquet shard, and the
    sole input to deciding what still needs downloading.

    Layout under `root`::

        _manifest.json
        JPMAQS_<THEME>/base.parquet         # first run, all history in one file
        JPMAQS_<THEME>/<YYYYMMDD>.parquet   # one shard per day thereafter
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser()
        self.path = self.root / MANIFEST_NAME
        self.datasets: Dict[str, Dict[str, dict]] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.datasets = json.load(f).get("datasets", {})

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "datasets": self.datasets}, f, indent=4)
        os.replace(tmp_path, self.path)

    def consumed(self) -> Set[str]:
        """The names of every delta file already compiled into a shard."""
        return {
            file_name
            for shards in self.datasets.values()
            for shard in shards.values()
            for file_name in shard["sources"]
        }

    def shard_paths(self, dataset: str) -> List[Path]:
        shards = self.datasets.get(dataset, {})
        keys = sorted(shards, key=lambda key: (key != BASE_SHARD, key))
        return [self.root / shards[key]["file"] for key in keys]

    def record(self, dataset: str, shard_key: str, sources: List[str], rows: int):
        shard = self.datasets.setdefault(dataset, {}).setdefault(shard_key, {})
        shard["file"] = f"{dataset}/{shard_key}.parquet"
        shard["sources"] = sorted(set(shard.get("sources", [])) | set(sources))
        shard["rows"] = rows
        shard["updated"] = utc_now().isoformat()


class DeltaFileLoader(object):
    def __init__(
        self,
        manifest: DeltaManifest,
        catalog_df: pd.DataFrame,
    ):
        self.manifest: DeltaManifest = manifest
        self.catalog_df: pd.DataFrame = catalog_df.copy()
        if not "Dataset" in self.catalog_df.columns:
            self.catalog_df["Dataset"] = (
                self.catalog_df["Theme"]
                .map(JPMAQS_DATASET_THEME_MAPPING)
                .fillna("Unknown")
            )
        assert self.catalog_df["Ticker"].is_unique, (
            "Ticker column must be unique in catalog_df."
        )

    def load_ticker_data(self, ticker: str, **kwargs) -> pl.LazyFrame:
        if "tickers" in kwargs:
            raise ValueError("The 'tickers' argument is not allowed in kwargs.")
        matches = self.catalog_df[
            self.catalog_df["Ticker"].str.lower() == ticker.lower()
        ]
        if matches.empty:
            raise ValueError(f"Ticker {ticker} not found in catalog.")

        dataset = matches["Dataset"].iloc[0]
        shard_paths = self.manifest.shard_paths(dataset)
        if not shard_paths:
            raise ValueError(
                f"No compiled files for dataset {dataset}. "
                "Run `JPMaQSDataExplorer.download_all_delta_files()` first."
            )
        kwargs["tickers"] = [matches["Ticker"].iloc[0]]
        return pl.concat(
            [scan_individual_file(f, **kwargs) for f in shard_paths], how="vertical"
        )


def transform_delta_qdf_to_revisions_matrix(
    df: pd.DataFrame,
    metric: str = "value",
    collapse_to_eod_values: bool = True,
    end_of_day_time: str = "23:59:59",
    end_of_day_tz: str = "UTC",
) -> pd.DataFrame:
    cols_to_keep = ["real_date", "last_updated", metric]
    missing = [c for c in cols_to_keep + ["ticker"] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    if df["ticker"].nunique(dropna=False) > 1:
        raise ValueError(
            "The DataFrame contains multiple tickers. Please filter to a single ticker."
        )

    out = df[cols_to_keep].copy()

    if collapse_to_eod_values:
        ts = out["last_updated"]
        ts = (
            ts.dt.tz_localize(end_of_day_tz)
            if ts.dt.tz is None
            else ts.dt.tz_convert(end_of_day_tz)
        )
        # `end_of_day_time` is the release cut-off: anything later is the next day's release
        _t = pd.Timestamp(end_of_day_time).time()
        eod_offset = pd.Timedelta(
            hours=_t.hour,
            minutes=_t.minute,
            seconds=_t.second,
            microseconds=_t.microsecond,
        )
        rolls_over = ts > (ts.dt.normalize() + eod_offset)
        out["effective_last_updated"] = (
            ts.dt.normalize() + rolls_over * pd.Timedelta(days=1)
        ).dt.date
    else:
        out["effective_last_updated"] = out["last_updated"]

    # latest record per (real_date, effective_last_updated); stable sort keeps input order on exact ties
    sort_cols = ["real_date", "effective_last_updated", "last_updated"]
    drop_dup_cols = ["real_date", "effective_last_updated"]
    new_last_updated_col = (
        "jpmaqs_release_date" if collapse_to_eod_values else "jpmaqs_release_datetime"
    )
    out = (
        out.sort_values(by=sort_cols, kind="stable")
        .drop_duplicates(subset=drop_dup_cols, keep="last")
        .reset_index(drop=True)
        .rename(columns={"effective_last_updated": new_last_updated_col})
    )

    out = out.pivot(
        columns=new_last_updated_col, index="real_date", values=metric
    ).ffill(axis=1)
    return out


def compile_delta_files(
    files_df: pd.DataFrame,
    manifest: DeltaManifest,
    delete_source_files: bool = True,
) -> DeltaManifest:
    """
    Fold the delta files in `files_df` into the manifest's Parquet shards, rewriting
    each touched shard in place. A dataset with no shards yet compiles all of its
    history into a single `base` shard; after that, deltas are sharded by the calendar
    day of their file timestamp, so a day's shard is rewritten as its deltas arrive.
    """
    files_df = files_df[files_df["file-name"].str.contains("_DELTA")]
    files_df = files_df[~files_df["file-name"].isin(manifest.consumed())]
    if files_df.empty:
        return manifest

    files_df = files_df.assign(
        **{
            "e-dataset": files_df["dataset"].str.replace("_DELTA", "", regex=False),
            "shard-key": files_df["file-timestamp"].dt.strftime("%Y%m%d"),
        }
    )
    first_run = ~files_df["e-dataset"].isin(manifest.datasets)
    files_df.loc[first_run, "shard-key"] = BASE_SHARD

    for (dataset, shard_key), group in files_df.groupby(["e-dataset", "shard-key"]):
        shard_path = manifest.root / dataset / f"{shard_key}.parquet"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        paths = list(map(Path, group["path"]))
        if shard_path.exists():
            paths.append(shard_path)

        # collect then write: the shard is one of the sources, so it cannot be sunk into
        shard_df = pl.concat(
            [scan_individual_file(f) for f in paths], how="vertical"
        ).collect(engine="streaming")
        tmp_path = shard_path.with_suffix(".parquet.tmp")
        shard_df.write_parquet(tmp_path)
        os.replace(tmp_path, shard_path)

        # record and delete per shard, so an interrupted run leaves no shard whose
        # sources are unrecorded and would be folded in a second time
        manifest.record(dataset, shard_key, group["file-name"].tolist(), shard_df.height)
        manifest.save()
        if delete_source_files:
            _delete_compiled_files(group["path"])

    return manifest


def _delete_compiled_files(paths: List[Union[str, Path]]) -> None:
    """Delete compiled delta files, then any directory they leave empty."""
    parent_dirs = set()
    for path in map(Path, paths):
        if _delete_jpmaqs_file(path):
            parent_dirs.add(path.parent)
    for parent_dir in parent_dirs:
        if parent_dir.is_dir() and not any(parent_dir.iterdir()):
            parent_dir.rmdir()


class JPMaQSDataExplorer(object):
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
    ):
        if data_path is None:
            data_path = Path("~/jpmaqs-data").expanduser()

        self._data_path = Path(data_path).expanduser()
        self.downloader = DataQueryFileAPIClientAdapter(out_dir=self._data_path)
        # a sibling of the download directory, so `list_downloaded_files` never scans it
        self.manifest = DeltaManifest(self._data_path / "jpmaqs-delta-compiled")
        legacy_dir = Path(self.downloader._get_save_dir()) / "combined-delta-files"
        if legacy_dir.is_dir():
            shutil.rmtree(legacy_dir)

    @property
    def file_loader(self) -> DeltaFileLoader:
        if getattr(self, "_file_loader", None) is None:
            self._file_loader = DeltaFileLoader(
                manifest=self.manifest,
                catalog_df=self.catalog_df,
            )
        return self._file_loader

    @property
    def catalog_file(self) -> str:
        if getattr(self, "_catalog_file", None) is None:
            self._catalog_file = self.downloader.download_catalog_file()
        return self._catalog_file

    @property
    def catalog_df(self) -> pd.DataFrame:
        if getattr(self, "_catalog_df", None) is None:
            self._catalog_df = pd.read_parquet(self.catalog_file)
            self._catalog_df["Dataset"] = (
                self._catalog_df["Theme"]
                .map(JPMAQS_DATASET_THEME_MAPPING)
                .fillna("Unknown")
            )

        return self._catalog_df

    def init(self):
        """
        Initialize the data explorer by downloading the catalog file.
        """
        self.download_all_delta_files()
        assert bool(self.catalog_file), "Failed to download the catalog file."

    def download_all_delta_files(
        self,
        since_datetime: str = JPMAQS_EARLIEST_FILE_DATE,
        include_metadata: bool = True,
        **kwargs,
    ):
        if "include_full_snapshots" in kwargs or "include_delta" in kwargs:
            raise ValueError("This utility only downloads delta files.")
        if include_metadata:
            # metadata files are never compiled, so their on-disk skip check still holds
            self.downloader.download_files(
                since_datetime=since_datetime,
                include_full_snapshots=False,
                include_delta=False,
                include_metadata=True,
                **kwargs,
            )

        available_df = self.downloader.filter_available_files_by_datetime(
            since_datetime=since_datetime,
            include_full_snapshots=False,
            include_delta=True,
            include_metadata=False,
        )
        consumed = self.manifest.consumed()
        pending = [f for f in available_df["file-name"] if str(f) not in consumed]
        if pending:
            # ponytail: the first run holds every delta since 2022 on disk before it
            # compiles; batch the download by month if that peak becomes a problem
            self.downloader.download_multiple_files(filenames=pending, **kwargs)

        # any delta left on disk is either new or was compiled but not deleted (a crash
        # between the manifest write and the delete); `compile_delta_files` skips the latter
        compile_delta_files(self.downloader.list_downloaded_files(), self.manifest)
        self._file_loader = None

    def load_ticker_data(
        self, ticker: str, collect: bool = False, as_pandas: bool = False
    ) -> Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]:
        """
        Load the data for a specific ticker from the compiled files.
        """
        if as_pandas:
            collect = True
        lazy_df = self.file_loader.load_ticker_data(ticker)
        if collect:
            lazy_df = lazy_df.collect()
            if as_pandas:
                return lazy_df.to_pandas()

        return lazy_df

    def load_ticker_revision_matrix(
        self,
        ticker: str,
        metric: str = "value",
        collapse_to_eod_values: bool = True,
        end_of_day_time: str = "23:59:59",
        end_of_day_tz: str = "UTC",
        as_pandas: bool = True,
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Load the revision matrix for a specific ticker from the compiled files.
        """
        if not as_pandas:
            warnings.warn(
                "This method is is best suited for pandas DataFrames, and internally "
                "converts back to the required polars format. Consider directly "
                "consuming the pandas DataFrame output for best performance."
            )

        vintage_df = transform_delta_qdf_to_revisions_matrix(
            self.load_ticker_data(ticker, as_pandas=True),
            metric=metric,
            collapse_to_eod_values=collapse_to_eod_values,
            end_of_day_time=end_of_day_time,
            end_of_day_tz=end_of_day_tz,
        )
        return vintage_df if as_pandas else pl.from_pandas(vintage_df)


if __name__ == "__main__":
    explorer = JPMaQSDataExplorer(data_path="~/jpmaqs-data")
    explorer.init()
    df = explorer.load_ticker_revision_matrix("USD_EQXR_NSA")
    print(df)
