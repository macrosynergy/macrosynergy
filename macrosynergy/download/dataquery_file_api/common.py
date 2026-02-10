from __future__ import annotations

import calendar
import datetime
import functools
from typing import List, Optional, Union, Any, Sequence
import pandas as pd
import polars as pl
import threading
import logging
import time

from macrosynergy.compat import PD_2_0_OR_LATER, POLARS_0_17_13_OR_EARLIER

from .constants import JPMAQS_EARLIEST_FILE_DATE

logger = logging.getLogger(__name__)


class RateLimitedRequester:
    """
    Provides a thread-safe rate-limiting mechanism for API requests.
    """

    def __init__(self, api_delay: float):
        if api_delay < 0:
            raise ValueError("`api_delay` must be non-negative.")
        self._api_delay = api_delay
        self._rate_limit_lock = threading.Lock()
        self._last_api_call: Optional[datetime.datetime] = None

    def _wait_for_api_call(self, api_delay: Optional[float] = None) -> bool:
        """
        Blocks until the required delay since the last API call has passed.
        """
        delay = self._api_delay if api_delay is None else api_delay
        if delay <= 0:
            return True

        with self._rate_limit_lock:
            now = datetime.datetime.now()
            if self._last_api_call is None:
                self._last_api_call = now
                return True

            diff = (now - self._last_api_call).total_seconds()
            sleep_for = delay - diff
            if sleep_for > 0:
                if sleep_for > 1:
                    logger.info(
                        f"Sleeping for {sleep_for:.2f} seconds for API rate limit."
                    )
                time.sleep(sleep_for)

            self._last_api_call = datetime.datetime.now()
        return True


def _pd_to_datetime_compat(ts: str, utc: bool):
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


def _pd_to_utc_timestamp(ts: pd.Timestamp, utc: bool) -> pd.Timestamp:
    if not utc:
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _pd_scalar_to_datetime_compat(
    ts: Union[str, datetime.date, datetime.datetime, pd.Timestamp],
    *,
    format: str,
    utc: bool,
) -> pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        return _pd_to_utc_timestamp(ts, utc=utc)
    if isinstance(ts, datetime.datetime):
        return _pd_to_utc_timestamp(pd.Timestamp(ts), utc=utc)
    if isinstance(ts, datetime.date):
        return _pd_to_utc_timestamp(pd.Timestamp(ts), utc=utc)
    if isinstance(ts, str):
        if PD_2_0_OR_LATER:
            return pd.to_datetime(ts, format=format, utc=utc)
        return _pd_to_datetime_compat(ts, utc=utc)
    raise TypeError(
        "`ts` must be a string, date, datetime, pandas Timestamp, or a Series."
    )


def pd_to_datetime_compat(
    ts: Union[str, datetime.date, datetime.datetime, pd.Timestamp, pd.Series],
    format: str = "mixed",
    utc: bool = True,
):
    """
    Parse common timestamp-like inputs into pandas datetime objects.

    - Scalars return a `pd.Timestamp`
    - `pd.Series` returns a Series of Timestamps

    Notes
    -----
    - Strings accept the same formats as `_pd_to_datetime_compat` (and in pandas>=2.0
      also support `format="mixed"`).
    - Non-string scalars (date/datetime/Timestamp) are converted via `pd.Timestamp`
      and optionally localized/converted to UTC.
    """

    if isinstance(ts, pd.Series):
        if PD_2_0_OR_LATER:
            return pd.to_datetime(ts, format=format, utc=utc)
        return ts.apply(
            lambda x: _pd_scalar_to_datetime_compat(x, format=format, utc=utc)
        )

    return _pd_scalar_to_datetime_compat(ts, format=format, utc=utc)


def pd_timestamp_compat(
    ts: Optional[Union[str, datetime.date, datetime.datetime, pd.Timestamp]] = None,
    *,
    utc: bool = True,
) -> pd.Timestamp:
    """
    Convert common timestamp-like inputs into a pandas Timestamp.

    This is a thin wrapper around `pd_to_datetime_compat` for scalar inputs. It keeps
    the legacy convenience of `ts=None` defaulting to "now".
    """
    if ts is None:
        return pd.Timestamp.utcnow() if utc else pd.Timestamp.now()

    out = pd_to_datetime_compat(ts, utc=utc)
    if isinstance(out, pd.Series):
        raise TypeError("`pd_timestamp_compat` expects a scalar input, not a Series.")
    return out


def get_current_or_last_business_day(
    now_utc: Optional[
        Union[str, datetime.date, datetime.datetime, pd.Timestamp]
    ] = None,
) -> pd.Timestamp:
    """
    Return "today" (UTC) if it's a weekday, otherwise the previous business day.

    Notes
    -----
    - "Business day" is Monday-Friday (no holiday calendar).
    - Returned Timestamp is UTC and normalized to midnight.
    """
    now_ts = pd_to_datetime_compat(
        now_utc if now_utc is not None else pd.Timestamp.utcnow(), utc=True
    ).normalize()
    # Monday=0 ... Sunday=6; weekend => >=5
    if now_ts.weekday() >= 5:
        return (now_ts - pd.offsets.BDay(1)).normalize()
    return now_ts


def validate_dq_timestamp(
    ts: str, var_name: str = None, raise_error: bool = True
) -> bool:
    """Validate a timestamp string for DataQuery API."""
    try:
        if PD_2_0_OR_LATER:
            pd.to_datetime(ts, format="mixed", utc=True)
        else:
            pd_to_datetime_compat(ts, utc=True)
        return True
    except (ValueError, TypeError):
        if raise_error:
            vn = f"`{var_name}`" if var_name else "Timestamp"
            raise ValueError(
                f"Invalid {vn} format. Use YYYYMMDD, YYYYMMDDTHHMMSS, or a "
                "recognized timestamp format with timezone."
            )
        else:
            return False


def _month_ends_between(
    start: datetime.date,
    end: datetime.date,
) -> List[datetime.date]:
    year, month = start.year, start.month
    out = []
    while (year, month) <= (end.year, end.month):
        dtx = datetime.date(year, month, calendar.monthrange(year, month)[1])
        if start <= dtx <= end:
            out.append(dtx)
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return out


def _previous_business_day(d: datetime.date) -> datetime.date:
    if isinstance(d, datetime.datetime):
        d = d.date()
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= datetime.timedelta(days=1)
    return d


@functools.lru_cache(maxsize=1)
def _large_delta_file_datetimes(as_str: bool = True) -> List[str]:
    sd = pd_to_datetime_compat(JPMAQS_EARLIEST_FILE_DATE).date()
    if isinstance(sd, datetime.datetime):
        sd = sd.date()
    ed = datetime.date.today()

    listA = _month_ends_between(sd, ed)
    listB = [_previous_business_day(d) for d in listA]

    all_dates = sorted(set(listA + listB))
    dt_list = [
        datetime.datetime.combine(d, datetime.time(23, 59, 59)) for d in all_dates
    ]
    dt_list = sorted(map(pd_to_datetime_compat, dt_list))
    if not as_str:
        return dt_list

    return [d.strftime("%Y%m%dT%H%M%S") for d in dt_list]


def pl_string_type():
    if POLARS_0_17_13_OR_EARLIER:
        return pl.Utf8
    return pl.String


def _is_date_only_string(x: Any) -> bool:
    """
    True for date-only strings like "YYYYMMDD" or "YYYY-MM-DD".

    This is used to interpret "date-only" `since_datetime`/`to_datetime` inputs as
    whole-day windows when selecting files or filtering by `last_updated`.
    """
    if not isinstance(x, str):
        return False
    return ("T" not in x) and (":" not in x)


def _normalize_file_timestamp_cutoff(
    x: Union[str, pd.Timestamp, datetime.date, datetime.datetime],
) -> pd.Timestamp:
    """
    Normalize a file-vintage cutoff.

    - For date-only strings, interpret as *end of that day* (inclusive).
    - For datetime-like inputs, keep the exact timestamp.
    """
    ts = pd_to_datetime_compat(x)
    if _is_date_only_string(x):
        ts = ts.normalize() + pd.DateOffset(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def _normalize_last_updated_cutoff(
    x: Union[str, pd.Timestamp, datetime.date, datetime.datetime],
) -> pd.Timestamp:
    """
    Normalize a `last_updated` cutoff.

    - For date-only strings, interpret as *end of that day* (inclusive) so that
      updates on that date are retained.
    - For datetime-like inputs, keep the exact timestamp.
    """
    return _normalize_file_timestamp_cutoff(x)


def _covering_large_delta_timestamp(
    to_ts: pd.Timestamp, delta_file_timestamps: Sequence[pd.Timestamp]
) -> Optional[pd.Timestamp]:
    """
    For monthly "large delta" regimes, return the first month-end-ish delta timestamp
    that covers `to_ts`, even if it is after `to_ts`.

    This is needed because the monthly delta file for a given month can be timestamped
    at month-end (or the previous business day), which may fall *after* a user's
    requested `to_datetime` within the month. In that case, we still need to load the
    covering delta file and then filter rows using `max_last_updated <= to_datetime`.
    """
    if not delta_file_timestamps:
        return None

    delta_ts_set = set(delta_file_timestamps)
    large_delta_ts_all = _large_delta_file_datetimes(as_str=False)
    if not large_delta_ts_all:
        return None

    # Candidates are "large delta" timestamps in the same (year, month) at or after to_ts.
    candidates = [
        ts
        for ts in large_delta_ts_all
        if (ts.year == to_ts.year and ts.month == to_ts.month and ts >= to_ts)
    ]
    for ts in sorted(candidates):
        if ts in delta_ts_set:
            return ts
    return None
