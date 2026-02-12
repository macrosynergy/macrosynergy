from typing import Dict, List, Optional, Sequence, Tuple, Union
import os
from pathlib import Path
import pandas as pd
import polars as pl
import logging
from enum import Enum

from macrosynergy.management.constants import JPMAQS_METRICS

from macrosynergy.compat import PYTHON_3_8_OR_LATER
from macrosynergy.download.dataquery_file_api.common import (
    JPMaQSParquetExpectedColumns,
    pl_string_type,
    pd_to_datetime_compat,
    get_current_or_last_business_day,
    _downloaded_files_df,
    _list_downloaded_files,
    _normalize_file_timestamp_cutoff,
)
from macrosynergy.download.dataquery_file_api.file_selector import FileSelector

logger = logging.getLogger(__name__)


def lazy_load_from_parquets(
    files_dir: Union[str, Path],
    file_format: str = "parquet",
    tickers: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    min_last_updated: Optional[Union[str, pd.Timestamp]] = None,
    max_last_updated: Optional[Union[str, pd.Timestamp]] = None,
    dataframe_format: str = "qdf",
    dataframe_type: str = "pandas",
    categorical_dataframe: bool = True,
    datasets: Optional[List[str]] = None,
    include_delta_files: bool = True,
    delta_treatment: str = "latest",
    since_datetime: Optional[Union[str, pd.Timestamp]] = None,
    to_datetime: Optional[Union[str, pd.Timestamp]] = None,
    include_file_column: bool = True,
    catalog_file: Optional[str] = None,
    warn_if_no_full_snapshots: bool = False,
) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
    """
    This function helps to lazily load JPMaQS parquet files from a specified directory.
    It operates using the exact ticker names provided.

    Notes
    -----
    The `datasets` argument applies to
    the "effective dataset" (e-dataset), meaning that delta datasets (those ending with
    `_DELTA`) are treated as updates to their base dataset, not as separate datasets.

    Vintage selection (`to_datetime`)
    -------------------------------
    When JPMaQS removes older full snapshots, historical data may only be reconstructible
    from delta files. In monthly "large delta" regimes, the delta file for a given month
    is timestamped at month-end (or the previous business day), which can fall *after*
    an in-month `to_datetime` (e.g., `to_datetime="2025-03-15"`).

    In that case the loader will still select the covering month-end delta file and you
    should use `max_last_updated <= to_datetime` to exclude updates beyond the requested
    vintage. `DataQueryFileAPIClient.load_data()` applies this default automatically
    when `to_datetime` is provided without `max_last_updated`.
    """
    files_dir = Path(files_dir)
    if (not metrics) or (metrics == "all") or ("all" in metrics):
        metrics = JPMAQS_METRICS

    delta_treatment = delta_treatment.lower()

    _check_lazy_load_inputs(
        files_dir=files_dir,
        file_format=file_format,
        tickers=tickers,
        cids=None,  # intentionally set to None
        xcats=None,  # intentionally set to None
        metrics=metrics,
        start_date=start_date,
        end_date=end_date,
        min_last_updated=min_last_updated,
        max_last_updated=max_last_updated,
        delta_treatment=delta_treatment,
        dataframe_format=dataframe_format,
        dataframe_type=dataframe_type,
        categorical_dataframe=categorical_dataframe,
        datasets=datasets,
    )

    all_data_files_df: pd.DataFrame = _downloaded_files_df(
        files_dir=files_dir,
        file_format=file_format,
        include_metadata_files=False,  # no metadata files - cannot scan with QDF like schema
    )
    if to_datetime is not None and (not all_data_files_df.empty):
        if "file-timestamp" in all_data_files_df.columns:
            oldest_local_ts = all_data_files_df["file-timestamp"].min()
            if pd.notna(oldest_local_ts):
                to_cutoff = _normalize_file_timestamp_cutoff(to_datetime)
                if to_cutoff < oldest_local_ts:
                    raise ValueError(
                        "`to_datetime` predates the oldest JPMaQS data file timestamp "
                        "found in the local cache "
                        f"({oldest_local_ts.strftime('%Y-%m-%dT%H:%M:%SZ')})."
                    )
    effective_to_datetime = to_datetime
    if (
        warn_if_no_full_snapshots
        and (since_datetime is not None)
        and (to_datetime is None)
    ):
        effective_to_datetime = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    fs = FileSelector(api_files_df=None, local_files_df=all_data_files_df)
    available_files_df: pd.DataFrame = fs.select_files_for_load(
        since_datetime=since_datetime,
        to_datetime=effective_to_datetime,
        include_delta_files=include_delta_files,
        warn_if_no_full_snapshots=warn_if_no_full_snapshots,
        min_last_updated=min_last_updated,
        max_last_updated=max_last_updated,
    )
    if datasets:
        datasets = sorted(set([d.replace("_DELTA", "") for d in datasets]))
        if "e-dataset" in available_files_df.columns:
            available_files_df = available_files_df.loc[
                available_files_df["e-dataset"].isin(datasets)
            ]
        else:
            available_files_df = available_files_df.iloc[0:0].copy()

    include_file_column = "source_file" if include_file_column else None

    tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]

    if not tickers:
        raise ValueError(
            "No tickers specified. Provide `tickers=[...]` (or `cids` and `xcats`). "
            "If you want to load all tickers, pass `tickers` as a list of all tickers "
            "you want to load."
        )

    paths = (
        sorted(available_files_df["path"])
        if (not available_files_df.empty and ("path" in available_files_df.columns))
        else []
    )
    if not paths:
        total_parquets = len(_list_downloaded_files(files_dir, file_format="parquet"))
        total_data_parquets = len(all_data_files_df)
        today_utc = pd.Timestamp.utcnow().normalize()
        last_bd = get_current_or_last_business_day(today_utc)
        datasets_str = (
            ", ".join(list(datasets)[:5]) + ("..." if len(datasets) > 5 else "")
            if datasets
            else "N/A"
        )
        extra_hint = ""
        cache_hint = (
            f"Local cache scanned: '{files_dir.resolve()}'. "
            f"Found {total_parquets} parquet file(s) total."
        )
        if total_parquets > 0 and total_data_parquets == 0:
            extra_hint = (
                " Found parquet file(s), but they appear to be metadata/catalog only "
                "(no data snapshot/delta files)."
            )
        raise FileNotFoundError(
            "No JPMaQS data snapshot/delta parquet files were found in the local cache "
            f"to load for the requested selection (datasets={datasets_str}).{extra_hint} "
            f"{cache_hint} "
            "This is a local-cache issue (not necessarily a DataQuery availability issue). "
            "Common causes: only metadata files were downloaded for the selected vintage, "
            "or cache cleanup removed older files. "
            "If you are making a historical/vintage request, try setting "
            "`cleanup_old_files_n_days=None`. "
            "JPMaQS data files are published on business days only (the catalog is daily). "
            f"Today (UTC) is {today_utc.date()}; latest business day is {last_bd.date()}. "
            "Try running `download()` again (with `skip_download=False`) or set "
            f"`since_datetime='{last_bd.strftime('%Y%m%d')}'` (or earlier)."
        )

    if catalog_file:
        catalog_path = Path(catalog_file)
        if not catalog_path.is_file():
            raise FileNotFoundError(f"No such file: {catalog_path}")

        catalog_lf = pl.scan_parquet(str(catalog_path))
        if PYTHON_3_8_OR_LATER:
            schema_cols = catalog_lf.collect_schema().names()
        else:
            schema_cols = catalog_lf.schema.keys()
        ticker_col = "Ticker" if "Ticker" in schema_cols else "ticker"
        tickers_lf_col = catalog_lf.select(pl.col(ticker_col)).drop_nulls().unique()
        if ticker_col in schema_cols:
            catalog_tickers = tickers_lf_col.collect()[ticker_col].to_list()
            if catalog_tickers:
                catalog_set = {str(t).lower() for t in catalog_tickers}
                missing = sorted({t for t in tickers if t.lower() not in catalog_set})
                if missing:
                    raise ValueError(
                        f"Ticker(s) not present in JPMaQS catalog: {', '.join(missing)}."
                    )

    lf: pl.LazyFrame = _lazy_load_filtered_parquets(
        paths=paths,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        delta_treatment=delta_treatment,
        min_last_updated=min_last_updated,
        max_last_updated=max_last_updated,
        return_qdf=(dataframe_format == "qdf"),
        include_file_column=include_file_column,
    )
    if (metrics and set(metrics) != set(JPMAQS_METRICS)) or include_file_column:
        cols_to_keep = ["real_date", "cid", "xcat", "ticker"] + metrics
        if include_file_column:
            cols_to_keep.append(include_file_column)
        if PYTHON_3_8_OR_LATER:
            lf = lf.select(
                [pl.col(c) for c in cols_to_keep if c in lf.collect_schema().names()]
            )
        else:
            lf = lf.select([pl.col(c) for c in cols_to_keep if c in lf.schema.keys()])
    cat_cols = ["cid", "xcat", "ticker"]
    if include_file_column:
        cat_cols.append(include_file_column)
    if dataframe_type in {"polars", "polars-lazy"}:
        if categorical_dataframe:
            categorical_dtype = getattr(pl, "Categorical", None)
            if categorical_dtype is not None:
                _names = (
                    lf.collect_schema().names()
                    if PYTHON_3_8_OR_LATER
                    else lf.schema.keys()
                )
                cols = [c for c in cat_cols if c in _names]
                for c in cols:
                    try:
                        lf = lf.with_columns(pl.col(c).cast(categorical_dtype))
                    except Exception:
                        logger.warning(
                            f"Failed to cast '{c}' to Categorical; keeping as string."
                        )
        return lf if dataframe_type == "polars-lazy" else lf.collect()
    if dataframe_type == "pandas":
        df = lf.collect().to_pandas()
        if categorical_dataframe:
            cols = [c for c in cat_cols if c in df.columns]
            if cols:
                df[cols] = df[cols].astype("category")
        return df

    raise ValueError("Unknown dataframe type")


def _check_lazy_load_inputs(
    files_dir: Union[str, Path],
    file_format: str,
    tickers: Optional[List[str]],
    cids: Optional[List[str]],
    xcats: Optional[List[str]],
    metrics: Optional[List[str]],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    min_last_updated: Optional[Union[str, pd.Timestamp]],
    max_last_updated: Optional[Union[str, pd.Timestamp]],
    delta_treatment: str,
    dataframe_format: str,
    dataframe_type: str,
    categorical_dataframe: bool,
    datasets: Optional[List[str]] = None,
):
    files_dir = Path(files_dir)
    if not files_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {files_dir}")

    if file_format != "parquet":
        raise ValueError("`file_format` must be 'parquet'.")
    # check whether or not there are any parquet files in the glob directory -recursive
    if not _list_downloaded_files(files_dir, file_format):
        raise FileNotFoundError(
            f"No {file_format} files found in directory: {files_dir}"
        )
    if delta_treatment not in ["latest", "earliest", "all"]:
        raise ValueError(
            "`delta_treatment` must be one of 'latest', 'earliest', or 'all'."
        )

    for param, name in [
        (tickers, "tickers"),
        (cids, "cids"),
        (xcats, "xcats"),
        (metrics, "metrics"),
        (datasets, "datasets"),
    ]:
        if param is not None and (
            not isinstance(param, list) or not all(isinstance(x, str) for x in param)
        ):
            raise ValueError(f"If provided, `{name}` must be a list of strings.")

    if bool(cids) ^ bool(xcats):
        raise ValueError(
            "Both `cids` and `xcats` must be provided together, or neither."
        )

    tickers_list = [
        t.strip() for t in (tickers or []) if isinstance(t, str) and t.strip()
    ]
    if not (bool(tickers_list) or (bool(cids) and bool(xcats))):
        raise ValueError(
            "No tickers specified. Provide `tickers=[...]` (or `cids` and `xcats`). "
            "If you want to load all tickers, pass `tickers` as a list of all tickers "
            "you want to load."
        )

    for param, name in [
        (start_date, "start_date"),
        (end_date, "end_date"),
        (min_last_updated, "min_last_updated"),
        (max_last_updated, "max_last_updated"),
    ]:
        if param is not None and not isinstance(param, (str, pd.Timestamp)):
            raise ValueError(f"`{name}` must be a string or pandas Timestamp.")
        if isinstance(param, str):
            try:
                pd_to_datetime_compat(param, utc=True)
            except ValueError:
                raise ValueError(
                    f"`{name}` has invalid timestamp format. Use YYYY-MM-DD or a "
                    "recognized timestamp format with timezone."
                )

    if dataframe_format not in ["qdf", "tickers"]:
        raise ValueError("`dataframe_format` must be one of 'qdf' or 'tickers'.")

    if dataframe_type not in ["pandas", "polars", "polars-lazy"]:
        raise ValueError(
            "`dataframe_type` must be one of 'pandas', 'polars', 'polars-lazy'."
        )
    if not isinstance(categorical_dataframe, bool):
        raise ValueError("`categorical_dataframe` must be a boolean.")


class JPMaQSParquetSchemaKind(Enum):
    TICKER = "ticker"
    QDF = "qdf"


def _expr_split_ticker(ticker_expr: pl.Expr) -> Tuple[pl.Expr, pl.Expr]:
    """
    Robust split of 'CID_XCAT...' into (cid, xcat) WITHOUT using splitn().
    Works across Polars versions (avoids struct vs list return type issues).
    """
    splitx = ticker_expr.str.splitn("_", 2)
    cid = splitx.struct.field("field_0")
    xcat = splitx.struct.field("field_1")
    return cid, xcat


def _ensure_columns(
    lf: pl.LazyFrame,
    cols: Sequence[str],
    dtypes: Optional[Dict[str, "pl.DataType"]] = None,
) -> pl.LazyFrame:
    """
    Ensure all `cols` exist before .select(...).
    This runs schema-only (lf.collect_schema()), not a materialization.
    """
    if PYTHON_3_8_OR_LATER:
        have = set(lf.collect_schema().keys())
    else:
        have = set(lf.schema.keys())
    missing = [c for c in cols if c not in have]
    if not missing:
        return lf

    add_exprs = {}
    for c in missing:
        expr = pl.lit(None)
        if dtypes and c in dtypes:
            expr = expr.cast(dtypes[c])
        add_exprs[c] = expr

    return lf.with_columns(**add_exprs)


def _filter_lazy_frame_by_tickers(
    lf: pl.LazyFrame,
    tickers: Sequence[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    min_last_updated: Optional[Union[str, pd.Timestamp]],
    max_last_updated: Optional[Union[str, pd.Timestamp]],
) -> pl.LazyFrame:
    tickers_list = [t for t in tickers if t]
    lf = lf.filter(pl.col("ticker").is_in(tickers_list))
    if start_date:
        start_date = pd_to_datetime_compat(start_date).strftime("%Y-%m-%d")
        lf = lf.filter(pl.col("real_date") >= pl.lit(start_date).str.to_date())
    if end_date:
        end_date = pd_to_datetime_compat(end_date).strftime("%Y-%m-%d")
        lf = lf.filter(pl.col("real_date") <= pl.lit(end_date).str.to_date())
    if min_last_updated:
        min_last_updated = pd_to_datetime_compat(min_last_updated).to_datetime64()
        lf = lf.filter(pl.col("last_updated") >= pl.lit(min_last_updated))
    if max_last_updated:
        max_last_updated = pd_to_datetime_compat(max_last_updated).to_datetime64()
        lf = lf.filter(pl.col("last_updated") <= pl.lit(max_last_updated))

    return lf


def _to_output_schema(
    lf: pl.LazyFrame,
    include_file_column: Optional[str],
    want_qdf: bool,
) -> pl.LazyFrame:
    """Normalize columns to qdf or ticker-based shape."""
    cols = "real_date.ticker.value.eop_lag.mop_lag.grading.last_updated"
    if include_file_column:
        cols += "." + include_file_column
    ticker_cols = cols.split(".")
    qdf_cols = cols.replace("ticker", "cid.xcat").split(".")

    dtype_map = dict(JPMaQSParquetExpectedColumns.TICKER.value)
    dtype_map.update({"cid": pl_string_type(), "xcat": pl_string_type()})

    if want_qdf:
        cid_expr, xcat_expr = _expr_split_ticker(pl.col("ticker"))
        lf = lf.with_columns(cid=cid_expr, xcat=xcat_expr)
        lf = _ensure_columns(lf, qdf_cols, dtypes=dtype_map)
        return lf.select(qdf_cols)

    lf = _ensure_columns(lf, ticker_cols, dtypes=dtype_map)
    return lf.select(ticker_cols)


def _build_filtered_parquet_lazyframe(
    paths: Sequence[Union[str, os.PathLike]],
    tickers_list: Sequence[str],
    *,
    start_date: Optional[Union[pd.Timestamp, str]] = None,
    end_date: Optional[Union[pd.Timestamp, str]] = None,
    min_last_updated: Optional[Union[pd.Timestamp, str]] = None,
    max_last_updated: Optional[Union[pd.Timestamp, str]] = None,
    include_file_column: Optional[str] = None,
    return_qdf: bool = False,
) -> pl.LazyFrame:
    """
    Scan multiple parquet paths into a single LazyFrame, optionally adding a file-path
    column in a way compatible with Polars 0.17.13 (Python 3.7).

    NOTE: categorical casting is intentionally not done here. Casting per-file columns
    to `pl.Categorical` before concatenation can break on older Polars/Python (e.g.
    Polars 0.17.x on Python 3.7). Callers should cast categoricals after concat.
    """
    lazy_parts: List[pl.LazyFrame] = []

    for pth in paths:
        file_base_name = Path(pth).name
        pth_str = os.fspath(pth)
        lf = pl.scan_parquet(pth_str)

        lf = _filter_lazy_frame_by_tickers(
            lf=lf,
            tickers=tickers_list,
            start_date=start_date,
            end_date=end_date,
            min_last_updated=min_last_updated,
            max_last_updated=max_last_updated,
        )

        if include_file_column:
            lf = lf.with_columns(pl.lit(file_base_name).alias(include_file_column))
        lf = _to_output_schema(
            lf=lf,
            include_file_column=include_file_column,
            want_qdf=return_qdf,
        )

        lazy_parts.append(lf)

    if not lazy_parts:
        return pl.DataFrame().lazy()

    return pl.concat(lazy_parts, how="vertical")


def _lazy_load_filtered_parquets(
    paths: List[str],
    tickers: List[str],
    start_date: Optional[Union[str, pd.Timestamp]],
    end_date: Optional[Union[str, pd.Timestamp]],
    min_last_updated: Optional[Union[str, pd.Timestamp]],
    max_last_updated: Optional[Union[str, pd.Timestamp]],
    delta_treatment: str,
    include_file_column: Optional[str],
    return_qdf: bool = True,
) -> pl.LazyFrame:
    if not paths:
        raise ValueError("No paths provided")

    tickers_list: List[str] = list(dict.fromkeys(tickers))

    out: pl.LazyFrame = _build_filtered_parquet_lazyframe(
        paths=paths,
        tickers_list=tickers_list,
        start_date=start_date,
        end_date=end_date,
        min_last_updated=min_last_updated,
        max_last_updated=max_last_updated,
        include_file_column=include_file_column,
        return_qdf=return_qdf,
    )

    key_cols = ["cid", "xcat"] if return_qdf else ["ticker"]
    full_key = key_cols + ["real_date"]

    if delta_treatment != "all":
        if delta_treatment == "latest":
            out = out.sort(
                full_key + ["last_updated"], descending=[False] * len(full_key) + [True]
            ).unique(subset=full_key, keep="first")
        elif delta_treatment == "earliest":
            out = out.sort(
                full_key + ["last_updated"],
                descending=[False] * len(full_key) + [False],
            ).unique(subset=full_key, keep="first")
        else:
            raise ValueError(f"Unknown delta_treatment: {delta_treatment}")
    sort_cols = ["cid", "xcat"] if return_qdf else ["ticker"]
    out = out.sort(sort_cols + ["real_date"])

    return out
