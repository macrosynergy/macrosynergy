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
        include_delta_files: bool = True,
        warn_if_no_full_snapshots: bool = False,
        last_modified_col: str = "last-modified",
        min_last_updated: Optional[Union[str, pd.Timestamp]] = None,
        max_last_updated: Optional[Union[str, pd.Timestamp]] = None,
    ) -> List[str]:
        """Select API file-name(s) required for a load vintage that are missing/outdated locally."""
        api_like = self._as_local_like_df(self.api_files_df, source="api")
        if api_like.empty:
            return []

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

    # For snapshot-led selection we keep the historical behaviour of treating
    # (`since_datetime`, `to_datetime`) as an unordered window and swapping if needed.
    window_since_ts = since_ts
    window_to_ts = vintage_to_ts
    if window_since_ts is not None and window_since_ts > window_to_ts:
        window_since_ts, window_to_ts = window_to_ts, window_since_ts

    earliest_snapshot_ts: Optional[pd.Timestamp] = None
    if warn_if_no_full_snapshots and window_since_ts is not None:
        is_delta_all = (
            df["filename"].astype(str).str.contains("_DELTA", case=False, na=False)
        )
        is_metadata_all = (
            df["filename"].astype(str).str.contains("_METADATA", case=False, na=False)
        )
        snapshots_all = df.loc[~is_delta_all & ~is_metadata_all].copy()
        earliest_snapshot_ts = snapshots_all["file-timestamp"].min()
        if pd.isna(earliest_snapshot_ts):
            earliest_snapshot_ts = None

    selected = []
    for _, g in df.groupby(group_col):
        if g.empty:
            continue

        is_delta_g = (
            g["filename"].astype(str).str.contains("_DELTA", case=False, na=False)
        )
        is_metadata_g = (
            g["filename"].astype(str).str.contains("_METADATA", case=False, na=False)
        )
        snapshots_all_g = g.loc[~is_delta_g & ~is_metadata_g].copy()

        # Delta-only history (effective): no snapshots exist at or before the requested
        # vintage. Snapshots can exist *later* in time (e.g. after source-side deletion
        # of older snapshots), but are unusable for reconstructing an earlier vintage
        # and should not block delta-only selection.
        snapshots_upto_vintage_g = snapshots_all_g[
            snapshots_all_g["file-timestamp"].le(vintage_to_ts)
        ].copy()
        if snapshots_upto_vintage_g.empty:
            if not include_delta_files:
                continue

            deltas_all = g.loc[is_delta_g].copy()
            if deltas_all.empty:
                continue

            effective_to_ts = vintage_to_ts
            to_base_ts = content_to_ts if content_to_ts is not None else vintage_to_ts
            if content_to_ts is not None and max_last_updated is not None:
                effective_to_ts = max(effective_to_ts, content_to_ts)

            if (to_datetime is not None) or (max_last_updated is not None):
                # Prefer regular in-month deltas when they exist up to the requested
                # vintage; only fall back to the covering month-end ("large delta")
                # file when needed (regular deltas may be deleted/absent).
                ts_ser = deltas_all["file-timestamp"]
                in_month = (ts_ser.dt.year == to_base_ts.year) & (
                    ts_ser.dt.month == to_base_ts.month
                )
                is_large_delta_ts = (
                    (ts_ser.dt.hour == 23)
                    & (ts_ser.dt.minute == 59)
                    & (ts_ser.dt.second == 59)
                )
                has_regular_in_month = bool(
                    (in_month & (ts_ser.le(to_base_ts)) & (~is_large_delta_ts)).any()
                )

                cover_ts = _covering_large_delta_timestamp(
                    to_ts=to_base_ts,
                    delta_file_timestamps=deltas_all["file-timestamp"].tolist(),
                )
                if cover_ts is not None and not has_regular_in_month:
                    effective_to_ts = max(effective_to_ts, cover_ts)

            deltas_sel = deltas_all[
                deltas_all["file-timestamp"].le(effective_to_ts)
            ].copy()
            selected.append(deltas_sel)
            continue

        # Windowed candidate set (matches historical behaviour for snapshot-led selection).
        if window_since_ts is not None:
            g_window = g[
                g["file-timestamp"].between(window_since_ts, window_to_ts)
            ].copy()
        else:
            g_window = g[g["file-timestamp"].le(window_to_ts)].copy()
        if g_window.empty:
            continue

        is_delta_w = (
            g_window["filename"]
            .astype(str)
            .str.contains("_DELTA", case=False, na=False)
        )
        is_metadata_w = (
            g_window["filename"]
            .astype(str)
            .str.contains("_METADATA", case=False, na=False)
        )
        snapshots_w = g_window.loc[~is_delta_w & ~is_metadata_w].copy()

        # Snapshot-led selection (preserves the window semantics):
        if snapshots_w.empty:
            # No snapshots within the requested window: fall back to deltas in-window (if any).
            if include_delta_files:
                selected.append(g_window.loc[is_delta_w].copy())
            continue

        latest_snapshot_ts = snapshots_w["file-timestamp"].max()
        snapshots_sel = snapshots_w[
            snapshots_w["file-timestamp"] == latest_snapshot_ts
        ].copy()
        if not include_delta_files:
            selected.append(snapshots_sel)
            continue

        deltas_w = g_window.loc[is_delta_w].copy()
        deltas_sel = deltas_w[deltas_w["file-timestamp"] >= latest_snapshot_ts].copy()

        # Monthly "large delta" regimes may require the covering month-end delta file even
        # when it timestamps after the file-vintage window (`to_datetime`). Include it when
        # regular in-month deltas are absent and a covering large-delta timestamp exists.
        to_base_ts = content_to_ts if content_to_ts is not None else vintage_to_ts
        deltas_all = g.loc[is_delta_g].copy()
        if (not deltas_all.empty) and (
            (to_datetime is not None) or (max_last_updated is not None)
        ):
            ts_ser = deltas_all["file-timestamp"]
            in_month = (ts_ser.dt.year == to_base_ts.year) & (
                ts_ser.dt.month == to_base_ts.month
            )
            is_large_delta_ts = (
                (ts_ser.dt.hour == 23)
                & (ts_ser.dt.minute == 59)
                & (ts_ser.dt.second == 59)
            )
            has_regular_in_month = bool(
                (in_month & (ts_ser.le(to_base_ts)) & (~is_large_delta_ts)).any()
            )
            cover_ts = _covering_large_delta_timestamp(
                to_ts=to_base_ts,
                delta_file_timestamps=ts_ser.tolist(),
            )
            if (
                cover_ts is not None
                and (not has_regular_in_month)
                and cover_ts >= latest_snapshot_ts
            ):
                cover_df = deltas_all[deltas_all["file-timestamp"] == cover_ts].copy()
                if not cover_df.empty:
                    deltas_sel = pd.concat([deltas_sel, cover_df], ignore_index=True)
        selected.append(pd.concat([snapshots_sel, deltas_sel], ignore_index=True))

    out = (
        pd.concat(selected, ignore_index=True).reset_index(drop=True)
        if selected
        else df.iloc[0:0].copy()
    )

    if out.empty:
        return out

    if warn_if_no_full_snapshots and window_since_ts is not None:
        is_delta_out = (
            out["filename"].astype(str).str.contains("_DELTA", case=False, na=False)
        )
        is_metadata_out = (
            out["filename"].astype(str).str.contains("_METADATA", case=False, na=False)
        )
        snapshots_out = out.loc[~is_delta_out & ~is_metadata_out].copy()
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
