from typing import List, Optional, Union
from pathlib import Path
import logging

import pandas as pd

from .constants import JPMAQS_DATASET_THEME_MAPPING

from .common import (
    pd_to_datetime_compat,
    _normalize_last_updated_cutoff,
    _normalize_file_timestamp_cutoff,
    _is_date_only_string,
    _covering_large_delta_timestamp,
)

logger = logging.getLogger(__name__)


class FileSelector:
    """Helper class to reconcile API vs local file inventories."""

    def __init__(
        self,
        api_files_df: Optional[pd.DataFrame],
        local_files_df: Optional[pd.DataFrame],
        file_name_col: str = "file-name",
        tickers: Optional[List[str]] = None,
        catalog_file: Optional[Union[str, Path]] = None,
        case_sensitive: bool = False,
    ) -> None:
        self.file_name_col = str(file_name_col)
        self.api_files_df = (
            api_files_df.copy()
            if isinstance(api_files_df, pd.DataFrame)
            else pd.DataFrame()
        )
        self.local_files_df = (
            local_files_df.copy()
            if isinstance(local_files_df, pd.DataFrame)
            else pd.DataFrame()
        )

        if self.file_name_col not in self.api_files_df.columns:
            if "filename" in self.api_files_df.columns:
                self.api_files_df[self.file_name_col] = self.api_files_df["filename"]
            elif not self.api_files_df.empty:
                raise ValueError(f"Missing `{self.file_name_col}` in api_files_df.")
            else:
                self.api_files_df[self.file_name_col] = pd.Series(dtype="object")

        if self.file_name_col not in self.local_files_df.columns:
            if "filename" in self.local_files_df.columns:
                self.local_files_df[self.file_name_col] = self.local_files_df[
                    "filename"
                ]
            elif not self.local_files_df.empty:
                raise ValueError(f"Missing `{self.file_name_col}` in local_files_df.")
            else:
                self.local_files_df[self.file_name_col] = pd.Series(dtype="object")

        self.tickers = (
            [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
            if tickers
            else []
        )
        self.catalog_file = Path(catalog_file) if catalog_file else None
        self.case_sensitive = bool(case_sensitive)
        self.datasets_for_tickers: List[str] = self._resolve_datasets_for_tickers()

        self._dedupe_inventories()
        self.files_df = self.api_files_df.merge(
            self.local_files_df,
            on=self.file_name_col,
            how="outer",
            suffixes=("_api", "_local"),
        )
        # Backwards compatible alias (internal / not user-facing).
        self.merged_df = self.files_df

    def refresh(
        self,
        *,
        api_files_df: Optional[pd.DataFrame] = None,
        local_files_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Refresh cached API and/or local inventories in-place.

        This is intended for reusing a single `FileSelector` instance across multiple
        selection operations (for example when the client downloads files and the local
        inventory changes).
        """
        if api_files_df is not None:
            self.api_files_df = (
                api_files_df.copy()
                if isinstance(api_files_df, pd.DataFrame)
                else pd.DataFrame()
            )
        if local_files_df is not None:
            self.local_files_df = (
                local_files_df.copy()
                if isinstance(local_files_df, pd.DataFrame)
                else pd.DataFrame()
            )

        for attr in ("api_files_df", "local_files_df"):
            df = getattr(self, attr)
            if self.file_name_col not in df.columns:
                if "filename" in df.columns:
                    df[self.file_name_col] = df["filename"]
                elif df.empty:
                    df[self.file_name_col] = pd.Series(dtype="object")
                else:
                    raise ValueError(f"Missing `{self.file_name_col}` in {attr}.")
            setattr(self, attr, df)

        # Prefer enriching local inventory with API `last-modified` without forcing an
        # upstream API call in `list_downloaded_files()`.
        if (
            (not self.local_files_df.empty)
            and ("last-modified" not in self.local_files_df.columns)
            and ("last-modified" in self.api_files_df.columns)
        ):
            lm = self.api_files_df[[self.file_name_col, "last-modified"]].copy()
            lm = lm.drop_duplicates(subset=[self.file_name_col], keep="first")
            self.local_files_df = self.local_files_df.merge(lm, on=self.file_name_col, how="left")

        if self.tickers:
            self.datasets_for_tickers = self._resolve_datasets_for_tickers()

        self._dedupe_inventories()
        self.files_df = self.api_files_df.merge(
            self.local_files_df,
            on=self.file_name_col,
            how="outer",
            suffixes=("_api", "_local"),
        )
        self.merged_df = self.files_df

    def effective_snapshot_switchover_ts(
        self,
        *,
        file_group_ids: List[str],
        catalog_file_group_id: Optional[str] = None,
    ) -> Optional[pd.Timestamp]:
        """
        Return the effective (per-request) earliest full-snapshot timestamp.

        Notes
        -----
        JPMaQS can remove older full snapshots over time. For a given set of datasets
        we define the "switchover" as the *latest* of the datasets' earliest currently
        available full snapshots. If any dataset has no full snapshots at all, returns
        None.
        """
        if not file_group_ids:
            return None

        base_datasets = sorted(
            {
                str(d).replace("_DELTA", "")
                for d in file_group_ids
                if isinstance(d, str) and d and ("_METADATA" not in d.upper())
            }
        )
        if catalog_file_group_id is not None:
            base_datasets = [d for d in base_datasets if d != catalog_file_group_id]
        if not base_datasets:
            return None

        api_like = self._as_local_like_df(self.api_files_df, source="api")
        if api_like.empty:
            return None

        filenames = api_like["filename"].astype(str)
        is_snapshot = ~filenames.str.contains("_DELTA", case=False, na=False) & ~filenames.str.contains(
            "_METADATA", case=False, na=False
        )

        earliest_by_dataset: List[pd.Timestamp] = []
        for ds in base_datasets:
            ds_snapshots = api_like.loc[is_snapshot & api_like["dataset"].astype(str).eq(ds)]
            if ds_snapshots.empty:
                return None
            earliest_by_dataset.append(ds_snapshots["file-timestamp"].min())

        return max(earliest_by_dataset) if earliest_by_dataset else None

    def _dedupe_inventories(self) -> None:
        for attr in ("api_files_df", "local_files_df"):
            df = getattr(self, attr)
            if df.empty:
                continue
            if "last-modified" in df.columns:
                df = df.sort_values("last-modified", ascending=False)
            df = df.drop_duplicates(subset=[self.file_name_col], keep="first")
            setattr(self, attr, df.reset_index(drop=True))

    def _resolve_datasets_for_tickers(self) -> List[str]:
        if not self.tickers:
            return []
        catalog_path: Optional[Path] = None
        if self.catalog_file and self.catalog_file.is_file():
            catalog_path = self.catalog_file
        elif not self.local_files_df.empty and ("path" in self.local_files_df.columns):
            df = self.local_files_df.copy()
            if "dataset" in df.columns:
                df = df[df["dataset"] == "JPMAQS_METADATA_CATALOG"]
            else:
                df = df[
                    df[self.file_name_col]
                    .astype(str)
                    .str.startswith("JPMAQS_METADATA_CATALOG_")
                ]
            df = df[df["path"].notna()]
            if (not df.empty) and ("file-timestamp" in df.columns):
                df = df.sort_values("file-timestamp", ascending=False)
            if not df.empty:
                try:
                    candidate = Path(str(df.iloc[0]["path"]))
                    if candidate.is_file():
                        catalog_path = candidate
                except Exception:
                    catalog_path = None

        if catalog_path is None:
            return []

        try:
            cat = pd.read_parquet(catalog_path)
        except Exception:
            return []

        ticker_col = "Ticker" if "Ticker" in cat.columns else "ticker"
        theme_col = "Theme" if "Theme" in cat.columns else "theme"
        if ticker_col not in cat.columns or theme_col not in cat.columns:
            return []

        if self.case_sensitive:
            mask = cat[ticker_col].astype(str).isin(self.tickers)
        else:
            req = {t.lower() for t in self.tickers}
            mask = cat[ticker_col].astype(str).str.lower().isin(req)

        ds = (
            cat.loc[mask, theme_col]
            .map(JPMAQS_DATASET_THEME_MAPPING)
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        return sorted(set(ds))

    def _as_local_like_df(self, df: pd.DataFrame, *, source: str) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        out = df.copy()
        if "filename" not in out.columns:
            out["filename"] = out[self.file_name_col].astype(str)

        if "dataset" not in out.columns:
            base = out["filename"].astype(str).str.split(".", n=1).str[0]
            out["dataset"] = base.str.rsplit("_", n=1).str[0]

        if "e-dataset" not in out.columns:
            out["e-dataset"] = (
                out["dataset"].astype(str).str.replace(r"_DELTA$", "", regex=True)
            )

        if "file-timestamp" not in out.columns:
            if source == "api" and "file-datetime" in out.columns:
                out["file-timestamp"] = pd_to_datetime_compat(
                    out["file-datetime"], utc=True
                )
            else:
                base = out["filename"].astype(str).str.split(".", n=1).str[0]
                ts_str = base.str.rsplit("_", n=1).str[-1]
                out["file-timestamp"] = pd_to_datetime_compat(ts_str, utc=True)

        out = out[out["file-timestamp"].notna()].copy()
        if self.datasets_for_tickers:
            out = out[out["e-dataset"].isin(self.datasets_for_tickers)].copy()

        return out

    def select_files_for_download(
        self,
        overwrite: bool = False,
        since_datetime: Optional[Union[str, pd.Timestamp]] = None,
        to_datetime: Optional[Union[str, pd.Timestamp]] = None,
        file_group_ids: Optional[List[str]] = None,
        include_full_snapshots: bool = True,
        include_delta_files: bool = True,
        include_metadata_files: bool = False,
        warn_if_no_full_snapshots: bool = False,
        last_modified_col: str = "last-modified",
        min_last_updated: Optional[Union[str, pd.Timestamp]] = None,
        max_last_updated: Optional[Union[str, pd.Timestamp]] = None,
    ) -> List[str]:
        """Select API file-name(s) required for a load vintage that are missing/outdated locally."""
        api_like = self._as_local_like_df(self.api_files_df, source="api")
        if api_like.empty:
            return []

        if file_group_ids is not None:
            if (not isinstance(file_group_ids, list)) or (not all(isinstance(x, str) for x in file_group_ids)):
                raise ValueError("`file_group_ids` must be a list of strings.")
            if "file-group-id" in api_like.columns:
                api_like = api_like[api_like["file-group-id"].isin(file_group_ids)].copy()

        if not include_full_snapshots:
            is_snapshot = (
                ~api_like["filename"].astype(str).str.contains("_DELTA", case=False, na=False)
                & ~api_like["filename"].astype(str).str.contains("_METADATA", case=False, na=False)
            )
            api_like = api_like.loc[~is_snapshot].copy()

        selected_api = _select_local_files_for_load(
            api_like,
            since_datetime=since_datetime,
            to_datetime=to_datetime,
            include_delta_files=include_delta_files,
            warn_if_no_full_snapshots=warn_if_no_full_snapshots,
            min_last_updated=min_last_updated,
            max_last_updated=max_last_updated,
        )
        required = set(selected_api["filename"].astype(str).tolist())

        # Metadata files are also vintage-sensitive (the date window matters). When
        # requested, include only metadata files that fall within the same vintage
        # window semantics as snapshots/deltas (date-only strings are treated as
        # whole-day cutoffs).
        if include_metadata_files:
            meta_mask = (
                api_like["filename"]
                .astype(str)
                .str.contains("_METADATA", case=False, na=False)
            )
            meta_df = api_like.loc[meta_mask].copy()
            if not meta_df.empty:
                meta_since_ts = (
                    pd_to_datetime_compat(since_datetime)
                    if since_datetime is not None
                    else None
                )
                if meta_since_ts is not None and _is_date_only_string(since_datetime):
                    meta_since_ts = meta_since_ts.normalize()

                meta_to_ts = (
                    _normalize_file_timestamp_cutoff(to_datetime)
                    if to_datetime is not None
                    else meta_df["file-timestamp"].max()
                )

                if meta_since_ts is not None and meta_since_ts > meta_to_ts:
                    meta_since_ts, meta_to_ts = meta_to_ts, meta_since_ts

                if meta_since_ts is not None:
                    meta_df = meta_df[
                        meta_df["file-timestamp"].between(meta_since_ts, meta_to_ts)
                    ].copy()
                else:
                    meta_df = meta_df[meta_df["file-timestamp"].le(meta_to_ts)].copy()

                required |= set(meta_df["filename"].astype(str).tolist())
        if not required:
            return []
        if overwrite:
            return sorted(required)

        local = self.local_files_df.copy()
        if (not local.empty) and ("path" in local.columns):
            local = local[
                local["path"].notna() & local["path"].astype(str).str.len().gt(0)
            ]
            local = local[local["path"].apply(lambda p: Path(str(p)).is_file())]
        present = (
            set(local[self.file_name_col].astype(str).tolist())
            if not local.empty
            else set()
        )
        to_download = set(required - present)

        if (
            (last_modified_col in self.api_files_df.columns)
            and (not self.local_files_df.empty)
            and (last_modified_col in self.local_files_df.columns)
        ):
            adf = self.api_files_df.set_index(self.file_name_col)[last_modified_col]
            ldf = self.local_files_df.set_index(self.file_name_col)[last_modified_col]
            common = adf.index.intersection(ldf.index).intersection(list(required))
            if len(common) > 0:
                updated = common[adf.loc[common] > ldf.loc[common]]
                to_download |= set(map(str, updated.tolist()))

        return sorted(to_download)

    def select_files_for_load(
        self,
        since_datetime: Optional[Union[str, pd.Timestamp]] = None,
        to_datetime: Optional[Union[str, pd.Timestamp]] = None,
        include_delta_files: bool = True,
        warn_if_no_full_snapshots: bool = False,
        min_last_updated: Optional[Union[str, pd.Timestamp]] = None,
        max_last_updated: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Select local snapshot/delta files to load from disk (drops rows without a valid file `path`)."""
        if self.local_files_df.empty:
            return self.local_files_df.copy()

        if "path" not in self.local_files_df.columns:
            return pd.DataFrame()

        local_df = self.local_files_df.copy()
        local_df = local_df[
            local_df["path"].notna() & local_df["path"].astype(str).str.len().gt(0)
        ]
        local_df = local_df[local_df["path"].apply(lambda p: Path(str(p)).is_file())]
        local_df = self._as_local_like_df(local_df, source="local")
        if local_df.empty:
            return local_df

        return _select_local_files_for_load(
            local_df,
            since_datetime=since_datetime,
            to_datetime=to_datetime,
            include_delta_files=include_delta_files,
            warn_if_no_full_snapshots=warn_if_no_full_snapshots,
            min_last_updated=min_last_updated,
            max_last_updated=max_last_updated,
        )


def _select_local_files_for_load(
    files_df: pd.DataFrame,
    *,
    since_datetime: Optional[Union[str, pd.Timestamp]] = None,
    to_datetime: Optional[Union[str, pd.Timestamp]] = None,
    include_delta_files: bool = True,
    warn_if_no_full_snapshots: bool = False,
    min_last_updated: Optional[Union[str, pd.Timestamp]] = None,
    max_last_updated: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Single-responsibility helper: choose which local snapshot/delta files to load.

    The selection is per effective dataset ("e-dataset"):
    - If full snapshots exist for the dataset and at least one snapshot is present in the
      requested file-vintage window, load the latest snapshot in the window and any delta
      files newer than that snapshot (also within the window).
    - If no full snapshots exist at or before the requested vintage (effective delta-only
      history), load *all* available deltas up to the requested vintage. For monthly "large
      delta" regimes, also include the covering month-end delta file even if it timestamps
      after `to_datetime` (row-level filtering is handled via `max_last_updated`).
    """
    if files_df.empty:
        return files_df

    df = files_df.copy()

    if "e-dataset" in df.columns:
        group_col = "e-dataset"
    else:
        group_col = "dataset"
        if group_col not in df.columns:
            raise ValueError("Expected column 'dataset' in files_df")
        df[group_col] = (
            df[group_col].astype(str).str.replace(r"_DELTA$", "", regex=True)
        )

    if "file-timestamp" not in df.columns:
        raise ValueError("Expected column 'file-timestamp' in files_df")

    # ---- normalize inputs (window + vintages) ---------------------------------
    since_ts = (
        pd_to_datetime_compat(since_datetime) if since_datetime is not None else None
    )
    if since_ts is not None and _is_date_only_string(since_datetime):
        since_ts = since_ts.normalize()

    vintage_to_ts = (
        _normalize_file_timestamp_cutoff(to_datetime)
        if to_datetime is not None
        else df["file-timestamp"].max()
    )

    # `last_updated` timestamps are row-content vintages. They can be non-monotonic with
    # file timestamps, especially in monthly "large delta" regimes. Use them to guide
    # delta coverage without allowing snapshots beyond the file-vintage cutoff.
    content_to_ts: Optional[pd.Timestamp] = None
    if (max_last_updated is not None) or (to_datetime is not None):
        content_to_ts = _normalize_last_updated_cutoff(
            max_last_updated if max_last_updated is not None else to_datetime
        )

    # Snapshot-led selection keeps the historical behaviour of treating
    # (`since_datetime`, `to_datetime`) as an unordered window and swapping if needed.
    window_since_ts = since_ts
    window_to_ts = vintage_to_ts
    if window_since_ts is not None and window_since_ts > window_to_ts:
        window_since_ts, window_to_ts = window_to_ts, window_since_ts

    # ---- file type flags -------------------------------------------------------
    filenames = df["filename"].astype(str)
    is_delta = filenames.str.contains("_DELTA", case=False, na=False)
    is_metadata = filenames.str.contains("_METADATA", case=False, na=False)
    is_snapshot = ~is_delta & ~is_metadata

    ts = df["file-timestamp"]
    grp = df[group_col]
    in_window = (
        ts.le(window_to_ts)
        if window_since_ts is None
        else ts.between(window_since_ts, window_to_ts)
    )

    # ---- snapshot availability summary ----------------------------------------
    # Delta-only history (effective): no snapshots exist at or before the requested
    # file-vintage cutoff (`vintage_to_ts`).
    has_snapshot_upto_vintage = (is_snapshot & ts.le(vintage_to_ts)).groupby(grp).any()
    has_snapshot_upto_vintage = grp.map(has_snapshot_upto_vintage).fillna(False)
    is_delta_only_history = ~has_snapshot_upto_vintage

    latest_snapshot_in_window = (
        df.loc[is_snapshot & in_window].groupby(group_col)["file-timestamp"].max()
    )
    latest_snapshot_in_window = grp.map(latest_snapshot_in_window)
    has_snapshot_in_window = latest_snapshot_in_window.notna()

    # ---- large delta coverage summary (per dataset) ---------------------------
    to_base_ts = content_to_ts if content_to_ts is not None else vintage_to_ts
    need_cover = (to_datetime is not None) or (max_last_updated is not None)

    cover_ts_by_group = pd.Series(dtype="object")
    has_regular_in_month_by_group = pd.Series(dtype="bool")
    if need_cover and bool(is_delta.any()):
        deltas = df.loc[is_delta, [group_col, "file-timestamp"]].copy()

        cover_ts_by_group = deltas.groupby(group_col)["file-timestamp"].apply(
            lambda s: _covering_large_delta_timestamp(
                to_ts=to_base_ts, delta_file_timestamps=s.tolist()
            )
        )

        def _has_regular_in_month(s: pd.Series) -> bool:
            in_month = (s.dt.year == to_base_ts.year) & (s.dt.month == to_base_ts.month)
            is_large_delta_ts = (
                (s.dt.hour == 23) & (s.dt.minute == 59) & (s.dt.second == 59)
            )
            return bool((in_month & (s.le(to_base_ts)) & (~is_large_delta_ts)).any())

        has_regular_in_month_by_group = deltas.groupby(group_col)[
            "file-timestamp"
        ].apply(_has_regular_in_month)

    cover_ts = grp.map(cover_ts_by_group)
    cover_ts = pd.to_datetime(cover_ts, utc=True, errors="coerce")
    has_regular_in_month = (
        grp.map(has_regular_in_month_by_group).astype("boolean").fillna(False)
    )

    # ---- selection masks (vectorized) -----------------------------------------
    cutoff_ts = vintage_to_ts
    if content_to_ts is not None and max_last_updated is not None:
        cutoff_ts = max(cutoff_ts, content_to_ts)

    # Delta-only history: select all deltas up to the cutoff plus a covering large-delta
    # file (which can be after the vintage cutoff).
    mask_delta_only = (
        include_delta_files & is_delta_only_history & is_delta & ts.le(cutoff_ts)
    )
    mask_delta_only_cover = (
        include_delta_files
        & need_cover
        & is_delta_only_history
        & is_delta
        & cover_ts.notna()
        & ts.eq(cover_ts)
        & ts.gt(cutoff_ts)
    )

    # Snapshot-led history: window semantics apply.
    mask_latest_snapshot = (
        is_snapshot
        & in_window
        & has_snapshot_in_window
        & ts.eq(latest_snapshot_in_window)
    )

    mask_window_deltas_no_snapshot = (
        include_delta_files
        & is_delta
        & in_window
        & has_snapshot_upto_vintage
        & (~has_snapshot_in_window)
    )
    mask_window_deltas_after_snapshot = (
        include_delta_files
        & is_delta
        & in_window
        & has_snapshot_in_window
        & ts.ge(latest_snapshot_in_window)
    )

    # Snapshot-led cover delta (only when regular in-month deltas are absent).
    mask_snapshot_cover = (
        include_delta_files
        & need_cover
        & is_delta
        & has_snapshot_in_window
        & cover_ts.notna()
        & (~has_regular_in_month)
        & cover_ts.ge(latest_snapshot_in_window)
        & ts.eq(cover_ts)
        & (~mask_window_deltas_after_snapshot)
    )

    out = df.loc[
        mask_delta_only
        | mask_delta_only_cover
        | mask_latest_snapshot
        | mask_window_deltas_no_snapshot
        | mask_window_deltas_after_snapshot
        | mask_snapshot_cover
    ].copy()

    if out.empty:
        return df.iloc[0:0].copy()

    # ---- warnings / final output shaping --------------------------------------
    earliest_snapshot_ts: Optional[pd.Timestamp] = None
    if warn_if_no_full_snapshots and window_since_ts is not None:
        earliest_snapshot_ts = df.loc[is_snapshot, "file-timestamp"].min()
        if pd.isna(earliest_snapshot_ts):
            earliest_snapshot_ts = None

    if warn_if_no_full_snapshots and window_since_ts is not None:
        is_delta_out = (
            out["filename"].astype(str).str.contains("_DELTA", case=False, na=False)
        )
        snapshots_out = out.loc[~is_delta_out].copy()
        if snapshots_out.empty and bool(is_delta_out.any()):
            earliest_snapshot_str = None
            if earliest_snapshot_ts is not None:
                earliest_snapshot_str = earliest_snapshot_ts.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            else:
                earliest_snapshot_str = "N/A"
            logger.warning(
                "No full snapshots available in the requested window "
                f"since={window_since_ts.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                f"to={window_to_ts.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                f"earliest_snapshot={earliest_snapshot_str}"
            )

    if not include_delta_files:
        # keep only snapshots
        is_delta = (
            out["filename"].astype(str).str.contains("_DELTA", case=False, na=False)
        )
        is_metadata = (
            out["filename"].astype(str).str.contains("_METADATA", case=False, na=False)
        )
        out = out.loc[~is_delta & ~is_metadata].copy()

    out = out.sort_values([group_col, "file-timestamp", "filename"]).reset_index(
        drop=True
    )
    return out
