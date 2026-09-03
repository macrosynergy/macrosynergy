import functools
import hashlib
import itertools
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import pandas as pd
import polars as pl

from macrosynergy.compat import PD_2_0_OR_LATER
from macrosynergy.download.dataquery_file_api import (
    JPMAQS_DATASET_THEME_MAPPING,
    JPMAQS_EARLIEST_FILE_DATE,
    JPMAQS_METRICS,
    DataQueryFileAPIClient,
)


class DataQueryFileAPIClientAdapter(DataQueryFileAPIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_save_dir(self) -> str:
        """
        Override the save directory to use `jpmaqs-delta-explorer` under the
        base directory if the base directory is not already `jpmaqs-delta-explorer`.
        """
        base_dir = Path(self.out_dir)
        if base_dir.name != "jpmaqs-delta-explorer":
            return str(base_dir / "jpmaqs-delta-explorer")
        return str(base_dir)

    def list_downloaded_files(self):
        files_df = super().list_downloaded_files()
        files_df = files_df["file-name"].contains()


def _pd_to_datetime_compat(ts: str, utc: bool) -> pd.Timestamp:
    formats = [
        "%Y%m%d",
        "%Y%m%dT%H%M%S",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        # ISO with timezone information
        "%Y-%m-%dT%H:%M:%SZ",  # UTC with Z (e.g. 2025-09-16T12:34:56Z)
        "%Y-%m-%dT%H:%M:%S%z",  # With numeric offset (e.g. 2025-09-16T12:34:56+02:00 or +0200)
    ]
    formats_str = f"[{', '.join(formats).replace('%', '').upper()}]"
    for fmt in formats:
        try:
            return pd.to_datetime(ts, format=fmt, utc=utc)
        except (ValueError, TypeError):
            continue
    raise ValueError(
        f"Timestamp '{ts}' does not match expected formats. Use one of {formats_str}."
    )


@overload
def pd_to_datetime_compat(
    ts: str,
    format: str = "mixed",
    utc: bool = True,
) -> pd.Timestamp: ...


@overload
def pd_to_datetime_compat(
    ts: pd.Series,
    format: str = "mixed",
    utc: bool = True,
) -> pd.Series: ...


def pd_to_datetime_compat(
    ts: Union[str, pd.Series],
    format: str = "mixed",
    utc: bool = True,
) -> Union[pd.Timestamp, pd.Series]:
    if PD_2_0_OR_LATER:
        return pd.to_datetime(ts, format=format, utc=utc)
    if isinstance(ts, pd.Series):
        return ts.apply(lambda x: _pd_to_datetime_compat(x, utc=utc))
    return _pd_to_datetime_compat(ts, utc=utc)


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
    available_metrics_list = JPMAQS_METRICS.copy()
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

    available_metrics_list = list(
        (set(available_metrics_list) | set(lf_schema)) - set(key_cols)
    )
    if metrics is not None:
        metrics = [m for m in metrics if m in available_metrics_list]
    if not bool(metrics):
        metrics = available_metrics_list

    if include_source_file:
        metrics = list(set(metrics + ["source_file"]))

    assert set(
        list(lf_schema.keys()) + (["source_file"] if include_source_file else [])
    ) == set(key_cols + metrics)

    expected_columns = key_cols + metrics
    lazy_df = lazy_df.select([pl.col(c) for c in expected_columns])

    # Filter by tickers if provided
    if tickers is not None:
        lazy_df = lazy_df.filter(pl.col("ticker").is_in(tickers))

    # convert start_date, end_date to date - expect YYYY-MM-DD, YYYYMMDD
    if start_date is not None:
        start_date = pd_to_datetime_compat(start_date).date()
        lazy_df = lazy_df.filter(pl.col("real_date") >= start_date)
    if end_date is not None:
        end_date = pd_to_datetime_compat(end_date).date()
        lazy_df = lazy_df.filter(pl.col("real_date") <= end_date)

    # convert max_last_updated, min_last_updated to datetime - expect YYYY-MM-DDTHH:MM:SS, YYYYMMDDTHHMMSS
    if max_last_updated is not None:
        max_last_updated = pd_to_datetime_compat(max_last_updated)
        lazy_df = lazy_df.filter(pl.col("last_updated") <= max_last_updated)
    if min_last_updated is not None:
        min_last_updated = pd_to_datetime_compat(min_last_updated)
        lazy_df = lazy_df.filter(pl.col("last_updated") >= min_last_updated)

    if categorical_ticker_column:
        lazy_df = lazy_df.with_columns(pl.col("ticker").cast(pl.Categorical))
    if include_source_file and categorical_source_file_column:
        lazy_df = lazy_df.with_columns(pl.col("source_file").cast(pl.Categorical))

    return lazy_df


class DeltaFileLoader(object):
    def __init__(
        self,
        files_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
    ):
        self.files_df: pd.DataFrame = files_df
        self.catalog_df: pd.DataFrame = catalog_df
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
        if "ticker" in kwargs:
            raise ValueError("The 'ticker' argument is not allowed in kwargs.")
        if ticker.lower() not in self.catalog_df["Ticker"].str.lower().values:
            raise ValueError(f"Ticker {ticker} not found in catalog.")

        files = self._get_files_for_ticker(ticker)

        kwargs["tickers"] = [ticker] if isinstance(ticker, str) else ticker
        lf = pl.concat(
            [scan_individual_file(f, **kwargs) for f in files], how="vertical"
        )
        return lf

    def _get_files_for_ticker(self, ticker: str) -> List[Path]:
        """
        Get the list of delta files for a specific ticker.
        """
        dataset_name: str = self.catalog_df[self.catalog_df["Ticker"] == ticker][
            "Dataset"
        ].iloc[0]
        files_list = self.files_df[self.files_df["e-dataset"] == dataset_name][
            "path"
        ].tolist()
        return list(map(Path, files_list))


def _needs_download(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        assert isinstance(self, JPMaQSDataExplorer), (
            "This decorator can only be used on methods of JPMaQSDataExplorer."
        )
        if not self._download_ran_successfully:
            ...
            # raise RuntimeError(
            #     "Please run the `init()` method to initialize the data explorer"
            # )
        return func(self, *args, **kwargs)

    return wrapper


def transform_delta_qdf_to_vintage(
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
        _t = pd.Timestamp(end_of_day_time).time()
        eod_offset = pd.Timedelta(
            hours=_t.hour,
            minutes=_t.minute,
            seconds=_t.second,
            microseconds=_t.microsecond,
        )
        out["effective_last_updated"] = ts.dt.normalize() + eod_offset
        out["effective_last_updated"] = out["effective_last_updated"].dt.date
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
        columns=new_last_updated_col, index="real_date", values="value"
    ).ffill(axis=1)
    return out


def _get_file_hash(file_path, algo: str = "sha256") -> str:
    _CHUNK = 8 * 1024 * 1024  # 8 MiB
    with open(file_path, "rb") as f:
        try:
            return hashlib.file_digest(f, algo).hexdigest()  # 3.11+, releases GIL
        except AttributeError:  # <3.11 fallback
            h = hashlib.new(algo)
            mv = memoryview(bytearray(_CHUNK))
            while True:
                n = f.readinto(mv)
                if not n:
                    break
                h.update(mv[:n])

            return h.hexdigest()


def combine_dataset_files(
    files_df: pd.DataFrame,
    out_path: Union[str, Path],
    file_suffix: str = "_COMBINED",
    categorical_ticker_column: bool = True,
    include_source_file: bool = True,
    categorical_source_file_column: bool = True,
    delete_source_files: bool = False,
) -> List[Path]:
    found_data_index = None
    if (Path(out_path) / "_index.json").exists():
        # found_data_index = Path(out_path) / "_index.json"
        with open(Path(out_path) / "_index.json", "r") as f:
            found_data_index = json.load(f)

    if found_data_index:
        # dfx = pd.DataFrame(found_data_index)
        files_already_used = set(
            itertools.chain.from_iterable(
                [found_data_index[_]["source_files"] for _ in found_data_index]
            )
        )
        files_df = files_df[
            ~(
                files_df["file-name"]
                .apply(lambda x: str(x).split(".")[0])
                .isin(files_already_used)
            )
        ]
    if files_df.empty:
        return []
    dfx = (
        files_df[files_df["file-name"].str.contains("_DELTA")]
        .groupby("e-dataset")["path"]
        .agg(sorted)
        .reset_index()
    )

    Path(out_path).mkdir(parents=True, exist_ok=True)
    file_suffix = "_" + file_suffix.lstrip("_").rstrip(".parquet") + ".parquet"
    dfx["out_file_path"] = (dfx["e-dataset"] + file_suffix).apply(
        lambda x: Path(out_path) / x
    )
    conflicting_dirs = dfx[dfx["out_file_path"].apply(lambda x: Path(x).is_dir())][
        "out_file_path"
    ].tolist()
    if len(conflicting_dirs) > 0:
        raise ValueError(
            f"The following output paths are directories, which conflicts with the expected output file paths: {conflicting_dirs}. Please remove these directories or choose a different output path."
        )
    for _file in dfx["out_file_path"]:
        if not Path(_file).exists():
            continue
        mask = dfx["path"].apply(str) == str(_file)
        dfx.loc[mask, "path"] = dfx.loc[mask, "path"] + [_file]
    _scan_args = dict(  # noqa: C408
        categorical_ticker_column=categorical_ticker_column,
        include_source_file=include_source_file,
        categorical_source_file_column=categorical_source_file_column,
    )

    def _scan_files(paths: List[Union[str, Path]]) -> pl.LazyFrame:
        return pl.concat(
            [scan_individual_file(p, **_scan_args) for p in paths],
            how="vertical",
        )

    dfx["lf"] = dfx["path"].apply(lambda x: _scan_files(x))

    data_index_dict = {}
    for _, (dataset, dataset_paths, out_file) in dfx[
        ["e-dataset", "path", "out_file_path"]
    ].iterrows():
        data_index_dict[dataset] = {
            "file_path": out_file,
            "source_files": sorted({Path(p).name.split(".")[0] for p in dataset_paths}),
        }

    lf_out_pairs: List[Tuple[pl.LazyFrame, Path]] = dfx[
        ["lf", "out_file_path"]
    ].values.tolist()
    print(f"Sinking {len(lf_out_pairs)} combined files...")
    pl.collect_all(
        [_lf.sink_parquet(_outx, lazy=True) for _lf, _outx in lf_out_pairs],
        engine="streaming",
    )
    print("Done sinking combined delta files")

    for i, (_dataset, _out_path) in enumerate(
        dfx[["e-dataset", "out_file_path"]].values.tolist()
    ):
        if Path(_out_path).exists():
            data_index_dict[_dataset]["hash"] = _get_file_hash(_out_path)
        else:
            warnings.warn(
                f"Combined file for dataset '{_dataset}' was not created at {_out_path}. "
                "Please check for errors in the previous steps."
            )

    # save this in out_path/_index.json
    index_file_path = Path(out_path) / "_index.json"
    with open(index_file_path, "w") as f:
        json.dump(data_index_dict, f, indent=4)

    # delete all source files
    if delete_source_files:
        out_files = dfx["out_file_path"].apply(lambda x: Path(x).resolve()).tolist()
        deleted_paths = []
        for _paths in dfx["path"]:
            for _path in _paths:
                if Path(_path).resolve() not in out_files:
                    Path(_path).unlink()
                    deleted_paths.append(_path)

        deleted_path_folders = list(
            set([Path(_path).parent for _path in deleted_paths])
        )
        for _folder in deleted_path_folders:
            if not any(Path(_folder).iterdir()):
                Path(_folder).rmdir()

    return data_index_dict


class JPMaQSDataExplorer(object):
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
    ):
        if data_path is None:
            data_path = Path("~/jpmaqs-data").expanduser()

        self._data_path = Path(data_path).expanduser()
        self.downloader = DataQueryFileAPIClientAdapter(out_dir=self._data_path)
        self._download_ran_successfully = False
        self.combined_files_path = (
            Path(self.downloader._get_save_dir()) / "combined-delta-files"
        )

    @property
    @_needs_download
    def file_loader(self) -> DeltaFileLoader:
        if not hasattr(self, "_file_loader") or self._file_loader is None:
            self._file_loader = DeltaFileLoader(
                files_df=self.files_df,
                catalog_df=self.catalog_df,
            )
        return self._file_loader

    @property
    @_needs_download
    def catalog_file(self) -> str:
        if not hasattr(self, "_catalog_file") or self._catalog_file is None:
            self._catalog_file = self.downloader.download_catalog_file()
        return self._catalog_file

    @property
    @_needs_download
    def catalog_df(self) -> pd.DataFrame:
        if not hasattr(self, "_catalog_df") or self._catalog_df is None:
            self._catalog_df = pd.read_parquet(self.catalog_file)
            self._catalog_df["Dataset"] = (
                self._catalog_df["Theme"]
                .map(JPMAQS_DATASET_THEME_MAPPING)
                .fillna("Unknown")
            )

        return self._catalog_df

    @property
    @_needs_download
    def files_df(self) -> pd.DataFrame:
        if not hasattr(self, "_files_df") or self._files_df is None:
            self._files_df_updater()
        return self._files_df

    def _files_df_updater(
        self,
    ):
        self._files_df = self.downloader.list_downloaded_files()
        assert "dataset" in self._files_df.columns, (
            "The files_df must have a 'dataset' column."
        )
        if "e-dataset" not in self._files_df.columns:
            self._files_df["e-dataset"] = self._files_df["dataset"].str.replace(
                "_DELTA", "", regex=False
            )

    def init(self):
        """
        Initialize the data explorer by downloading the catalog file.
        """
        self.download_all_delta_files()
        assert bool(self.catalog_file), "Failed to download the catalog file."

    def _combine_dataset_files(self) -> List[Path]:
        """
        Combine all delta files for a specific dataset into a single Parquet file.
        """
        return combine_dataset_files(
            files_df=self.files_df, out_path=self.combined_files_path
        )

    def download_all_delta_files(
        self,
        since_datetime: str = JPMAQS_EARLIEST_FILE_DATE,
        include_metadata: bool = True,
        **kwargs,
    ):
        if "include_full_snapshots" in kwargs:
            raise ValueError("This utility does not support snapshot files.")
        self.downloader.download_files(
            since_datetime=since_datetime,
            include_full_snapshots=False,
            include_delta=True,
            include_metadata=include_metadata,
            **kwargs,
        )
        self._combine_dataset_files()
        self._files_df_updater()
        self._download_ran_successfully = True

    def load_ticker_data(
        self, ticker: str, collect: bool = False, as_pandas: bool = False
    ) -> Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]:
        """
        Load the data for a specific ticker from the downloaded files.
        """
        if as_pandas:
            collect = True
        lazy_df = self.file_loader.load_ticker_data(ticker)
        if collect:
            lazy_df = lazy_df.collect()
            if as_pandas:
                return lazy_df.to_pandas()

        return lazy_df

    def load_ticker_vintage_data(
        self,
        ticker: str,
        metric: str = "value",
        collapse_to_eod_values: bool = True,
        end_of_day_time: str = "23:59:59",
        end_of_day_tz: str = "UTC",
        collect: bool = True,
        as_pandas: bool = True,
    ) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
        """
        Load the vintage data for a specific ticker from the downloaded files.
        """
        if not as_pandas:
            warnings.warn(
                "This method is is best suited for pandas DataFrames, and internally "
                "converts back to the required polars format. Consider directly "
                "consuming the pandas DataFrame output for best performance."
            )

        df = self.file_loader.load_ticker_data(ticker).collect().to_pandas()
        vintage_df = transform_delta_qdf_to_vintage(
            df,
            metric=metric,
            collapse_to_eod_values=collapse_to_eod_values,
            end_of_day_time=end_of_day_time,
            end_of_day_tz=end_of_day_tz,
        )
        if not as_pandas:
            if collect:
                return pl.from_pandas(vintage_df)
            else:
                return pl.from_pandas(vintage_df).lazy()
        return vintage_df


if __name__ == "__main__":
    explorer = JPMaQSDataExplorer(data_path="~/jpmaqs-data")
    explorer.init()
    # explorer.init()
    # print(explorer.files_df.head())
    # df = explorer.load_ticker_data("USD_EQXR_NSA")
    # print(df.head(50).collect())
    df = explorer.load_ticker_vintage_data("USD_EQXR_NSA")
    print(df)
