"""
Data manager for the Refinitiv / LSEG Datastream Web Service (DSWS).

This module provides :class:`DatastreamDataManager`, which sits on top of
:class:`~macrosynergy.download.datastream.connection.DatastreamConnection` and offers
high-level methods for:

* Fetching index / list constituents.
* Retrieving static / snapshot metadata for a set of instruments.
* Downloading time-series data with automatic request chunking that respects the
  official DSWS API limits (≤ 50 instruments, ≤ 50 datatypes, ≤ 100 items per call).
* Post-processing the raw DataFrames returned by ``DatastreamPy`` into tidy formats
  suitable for downstream analysis.

API limits (enforced by this module)
-------------------------------------
* ``MAX_INSTRUMENTS_PER_REQUEST = 50``
* ``MAX_DATATYPES_PER_REQUEST  = 50``
* ``MAX_ITEMS_PER_REQUEST      = 100``   (instruments × datatypes)

Typical usage::

    from macrosynergy.download.datastream.data_manager import DatastreamDataManager

    mgr = DatastreamDataManager(username="DS:YOUR_ID", password="secret")

    # Snapshot metadata
    raw_meta = mgr.get_metadata(["VOD", "BP", "HSBA"], fields=["NAME", "RIC", "PCUR"])
    meta_df  = DatastreamDataManager.process_metadata(raw_meta)

    # Time-series prices
    raw_ts   = mgr.get_data(["VOD", "BP"], fields=["P", "RI"], start="-1Y", end="0D")
    ts_dict  = DatastreamDataManager.process_timeseries_data(raw_ts)
"""

import logging
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .connection import DatastreamConnection

# ---------------------------------------------------------------------------
# Module-level logger (matches project convention in dataquery.py)
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)
_debug_handler = logging.StreamHandler(io.StringIO())
_debug_handler.setLevel(logging.NOTSET)
_debug_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s :: %(message)s"
    )
)
logger.addHandler(_debug_handler)

# ---------------------------------------------------------------------------
# API limits
# ---------------------------------------------------------------------------
MAX_INSTRUMENTS_PER_REQUEST: int = 50
MAX_DATATYPES_PER_REQUEST: int = 50
MAX_ITEMS_PER_REQUEST: int = 100

# ---------------------------------------------------------------------------
# Default metadata fields
# ---------------------------------------------------------------------------
DEFAULT_STATIC_FIELDS: Dict[str, str] = {
    "RIC": "Reuters Instrument Code",
    "DSCD": "Datastream code",
    "NAME": "Company / instrument name",
    # "EXCH": "Exchange code",
    "EXNAME": "Exchange name",
    "MNEM": "Datastream mnemonic / quote ID",
    "TYPE": "Instrument type",
    "ESTAT": "Equity status",
    "PRIMQUOTE": "Primary quote DS code",
    "ISINID": "Primary / secondary quote flag",
    "BDATE": "Base listing date",
    "DEADDT": "Security-level delisting date",
    "PCUR": "Price currency",
    "P.U": "Unit of adjusted price",
}


class DatastreamDataManager:
    """High-level data-retrieval interface for the Datastream Web Service.

    The manager handles chunking, ticker-formatting, and response concatenation so
    that callers can request arbitrarily large instrument universes without worrying
    about DSWS request-size limits.

    Parameters
    ----------
    connection : DatastreamConnection, optional
        A pre-built, optionally already-connected :class:`DatastreamConnection`
        instance.  Mutually exclusive with *username* / *password*.
    username : str, optional
        DSWS child ID.  Used only when *connection* is ``None``.
    password : str, optional
        DSWS password.  Used only when *connection* is ``None``.

    Raises
    ------
    ValueError
        If neither *connection* nor both *username* and *password* are supplied.

    Examples
    --------
    Using a pre-built connection::

        conn = DatastreamConnection("DS:ID", "secret")
        mgr  = DatastreamDataManager(connection=conn)

    Passing credentials directly (connection created internally)::

        mgr = DatastreamDataManager(username="DS:ID", password="secret")

    Context-manager pattern (connection is managed externally)::

        with DatastreamConnection("DS:ID", "secret") as conn:
            mgr = DatastreamDataManager(connection=conn)
            df  = mgr.get_metadata("VOD")
    """

    def __init__(
        self,
        connection: Optional[DatastreamConnection] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        if connection is not None:
            self._connection = connection
        elif username and password:
            logger.debug(
                "No DatastreamConnection supplied — creating one from credentials."
            )
            self._connection = DatastreamConnection(
                username=username, password=password
            )
        else:
            raise ValueError(
                "Provide either a DatastreamConnection object via 'connection', "
                "or supply both 'username' and 'password'."
            )

    # ------------------------------------------------------------------
    # Public data-retrieval methods
    # ------------------------------------------------------------------

    def get_constituents(
        self,
        index_code: str,
        date: Optional[Union[str, datetime]] = None,
    ) -> List[str]:
        """Fetch the mnemonic codes for all constituents of a Datastream list.

        The method appends the ``|L`` suffix if not already present and issues a
        snapshot request for the ``MNEM`` datatype.

        Parameters
        ----------
        index_code : str
            Datastream list code, e.g. ``'LFTSE100'``, ``'LS&PCOMP'``.  The ``|L``
            suffix is appended automatically when absent.
        date : str or datetime, optional
            Point-in-time date for the constituent snapshot.  ``None`` (default)
            returns the current composition.

        Returns
        -------
        list of str
            Sorted, deduplicated list of Datastream mnemonic strings.  Returns an
            empty list when the API returns no data.

        Raises
        ------
        ValueError
            If the API response contains an error marker (``'$$ER:'``).
        """
        list_code = index_code if index_code.endswith("|L") else f"{index_code}|L"
        start_date = self._format_date(date) if date is not None else "0D"

        logger.info("Fetching constituents for list '%s' at date '%s'.", list_code, start_date)

        ds = self._connection.get_connection()
        try:
            result = ds.get_data(
                tickers=list_code,
                fields=["MNEM"],
                start=start_date,
                kind=0,
            )
        except Exception as exc:
            logger.error("API call failed for constituents of '%s': %s", list_code, exc)
            raise

        if result is None or result.empty:
            logger.warning("Empty response for constituents of '%s'.", list_code)
            return []

        mnemonics: List[str] = []

        # Detect error markers in any Value/value column.
        value_col = next(
            (c for c in result.columns if c.lower() == "value"), None
        )
        if value_col and result[value_col].astype(str).str.startswith("$$ER:").any():
            error_sample = result[value_col].astype(str).iloc[0]
            logger.error("API error in constituents response: %s", error_sample)
            raise ValueError(
                f"Datastream returned an error for list '{list_code}': {error_sample}"
            )

        # Parse MNEM column — handle three possible shapes:
        #   1. A 'MNEM' column with one mnemonic per row.
        #   2. A single 'Value' column with one mnemonic per row.
        #   3. A 'Value' cell containing comma-separated mnemonics.
        if "MNEM" in result.columns:
            raw_values = result["MNEM"].dropna().astype(str).tolist()
        elif value_col:
            raw_values = result[value_col].dropna().astype(str).tolist()
        else:
            raw_values = result.iloc[:, 0].dropna().astype(str).tolist()

        for raw in raw_values:
            # A single cell may carry comma-separated codes.
            for part in raw.split(","):
                stripped = part.strip()
                if stripped and not stripped.startswith("$$ER:"):
                    mnemonics.append(stripped)

        unique_sorted = sorted(set(mnemonics))
        logger.info(
            "Retrieved %d constituent(s) for list '%s'.",
            len(unique_sorted),
            list_code,
        )
        return unique_sorted

    def get_metadata(
        self,
        tickers: Union[str, List[str], Tuple[str, ...]],
        fields: Union[str, List[str], Tuple[str, ...]] = "NAME",
    ) -> pd.DataFrame:
        """Retrieve static / snapshot metadata for a set of instruments.

        Automatically chunks the request to stay within DSWS limits and
        concatenates the results into a single long-format DataFrame.

        Parameters
        ----------
        tickers : str, list, or tuple
            One or more Datastream instrument codes.  Comma-separated strings are
            accepted (e.g. ``'VOD,BP,HSBA'``).
        fields : str, list, or tuple, optional
            One or more Datastream datatype codes.  Defaults to ``'NAME'``.

        Returns
        -------
        pd.DataFrame
            Raw long-format DataFrame from the DSWS API with columns such as
            ``Instrument``, ``Datatype``, ``Value``, ``Currency``, ``Dates``.
            Use :meth:`process_metadata` to pivot to wide format.

        Raises
        ------
        ValueError
            If *tickers* or *fields* normalises to an empty list.
        """
        ticker_list = self._normalize_to_list(tickers)
        field_list = self._normalize_to_list(fields)
        self._validate_inputs(ticker_list, field_list)

        ticker_chunks, field_chunks = self._build_chunks(ticker_list, field_list)
        frames: List[pd.DataFrame] = []

        for t_chunk in ticker_chunks:
            row_frames: List[pd.DataFrame] = []
            for f_chunk in field_chunks:
                ticker_arg = self._format_tickers_arg(t_chunk, multi_field=len(f_chunk) > 1)
                logger.debug(
                    "get_metadata chunk: tickers=%s, fields=%s", ticker_arg, f_chunk
                )
                ds = self._connection.get_connection()
                chunk_df = ds.get_data(
                    tickers=ticker_arg,
                    fields=f_chunk,
                    # start="-0D",
                    # end="0D",
                    kind=0
                )
                if chunk_df is not None and not chunk_df.empty:
                    row_frames.append(chunk_df)

            if row_frames:
                frames.append(pd.concat(row_frames, axis=0, ignore_index=True))

        if not frames:
            logger.warning("get_metadata returned no data for tickers=%s.", ticker_list)
            return pd.DataFrame()

        result = pd.concat(frames, axis=0, ignore_index=True)
        logger.info(
            "get_metadata: retrieved %d row(s) for %d ticker(s) and %d field(s).",
            len(result),
            len(ticker_list),
            len(field_list),
        )
        return result

    def get_data(
        self,
        tickers: Union[str, List[str], Tuple[str, ...]],
        fields: Union[str, List[str], Tuple[str, ...]] = "P",
        start: str = "-1Y",
        end: str = "0D",
        freq: str = "D",
    ) -> pd.DataFrame:
        """Retrieve time-series data for a set of instruments.

        Automatically chunks the request to stay within DSWS limits.  Results
        are concatenated along ``axis=1`` (columns) so the returned DataFrame
        has a ``DatetimeIndex`` and ``MultiIndex`` columns of
        ``(Instrument, Field, Currency)``.

        Parameters
        ----------
        tickers : str, list, or tuple
            One or more Datastream instrument codes.
        fields : str, list, or tuple, optional
            One or more Datastream datatype codes.  Defaults to ``'P'`` (price).
        start : str, optional
            Start date.  Accepts relative (``'-1Y'``, ``'-3M'``) or absolute
            (``'2020-01-01'``) formats.  Defaults to ``'-1Y'``.
        end : str, optional
            End date.  Defaults to ``'0D'`` (today).
        freq : str, optional
            Data frequency: ``'D'`` daily, ``'W'`` weekly, ``'M'`` monthly,
            ``'Q'`` quarterly, ``'Y'`` yearly.  Defaults to ``'D'``.

        Returns
        -------
        pd.DataFrame
            Raw time-series DataFrame with a ``DatetimeIndex`` and
            ``MultiIndex`` columns.  Use :meth:`process_timeseries_data` to
            reshape to a tidy dict of DataFrames.

        Raises
        ------
        ValueError
            If *tickers* or *fields* normalises to an empty list.
        """
        ticker_list = self._normalize_to_list(tickers)
        field_list = self._normalize_to_list(fields)
        self._validate_inputs(ticker_list, field_list)

        ticker_chunks, field_chunks = self._build_chunks(ticker_list, field_list)
        frames: List[pd.DataFrame] = []

        for t_chunk in ticker_chunks:
            for f_chunk in field_chunks:
                ticker_arg = self._format_tickers_arg(t_chunk, multi_field=len(f_chunk) > 1)
                logger.debug(
                    "get_data chunk: tickers=%s, fields=%s, start=%s, end=%s, freq=%s",
                    ticker_arg,
                    f_chunk,
                    start,
                    end,
                    freq,
                )
                ds = self._connection.get_connection()
                chunk_df = ds.get_data(
                    tickers=ticker_arg,
                    fields=f_chunk,
                    start=start,
                    end=end,
                    freq=freq,
                )
                if chunk_df is not None and not chunk_df.empty:
                    frames.append(chunk_df)

        if not frames:
            logger.warning("get_data returned no data for tickers=%s.", ticker_list)
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        # Drop duplicate columns that can appear when chunks share a date index.
        result = result.loc[:, ~result.columns.duplicated()]
        logger.info(
            "get_data: retrieved %d row(s) for %d ticker(s) and %d field(s).",
            len(result),
            len(ticker_list),
            len(field_list),
        )
        return result

    # ------------------------------------------------------------------
    # Post-processing static methods
    # ------------------------------------------------------------------

    @staticmethod
    def process_metadata(
        metadata: pd.DataFrame,
        static_fields_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Pivot a raw long-format metadata DataFrame to wide format.

        Takes the long-format output of :meth:`get_metadata` and returns a
        wide DataFrame indexed by ``ticker`` with one column per datatype.

        Parameters
        ----------
        metadata : pd.DataFrame
            Raw DataFrame as returned by :meth:`get_metadata`.
        static_fields_map : dict, optional
            Mapping of datatype code → description used for documentation
            purposes only.  Defaults to :data:`DEFAULT_STATIC_FIELDS`.

        Returns
        -------
        pd.DataFrame
            Wide-format DataFrame indexed by ``ticker``.  Each metadata
            field becomes a column.  Rows with a missing ``currency`` are
            dropped as they typically represent API error responses.  The
            ``DEADDT`` column (if present) is cast to ``datetime64``.
        """
        if static_fields_map is None:
            static_fields_map = DEFAULT_STATIC_FIELDS

        if metadata is None or metadata.empty:
            logger.warning("process_metadata received an empty DataFrame.")
            return pd.DataFrame()

        df = metadata.copy()

        # Normalise column names (API may return varying capitalisation).
        col_map = {c: c for c in df.columns}
        for original in df.columns:
            lower = original.lower()
            if lower == "instrument":
                col_map[original] = "ticker"
            elif lower == "currency":
                col_map[original] = "currency"
            elif lower == "datatype":
                col_map[original] = "field"
            elif lower == "value":
                col_map[original] = "value"
        df = df.rename(columns=col_map)

        required = {"ticker", "field", "value"}
        missing = required - set(df.columns)
        if missing:
            logger.error(
                "process_metadata: required columns %s not found. Available: %s",
                missing,
                list(df.columns),
            )
            return df

        # Add currency column if absent (snapshot without currency info).
        if "currency" not in df.columns:
            df["currency"] = pd.NA

        # Drop rows that represent API-level errors (Value starts with $$ER).
        df["value"] = df["value"].astype(str)
        df = df[~df["value"].str.startswith("$$ER:")]

        # Drop rows where currency is NaN — these are invalid / error records.
        df = df.dropna(subset=["currency"])

        # Pivot from long to wide format.
        try:
            pivot = (
                df.set_index(["ticker", "currency", "field"])["value"]
                .unstack("field")
                .reset_index(level="currency")
            )
        except Exception as exc:
            logger.error("process_metadata pivot failed: %s", exc)
            return df

        # Cast DEADDT to datetime if present.
        if "DEADDT" in pivot.columns:
            pivot["DEADDT"] = pd.to_datetime(pivot["DEADDT"], errors="coerce")

        pivot.index.name = "ticker"
        logger.debug("process_metadata: pivot shape %s.", pivot.shape)
        return pivot

    @staticmethod
    def process_timeseries_data(
        tsdata: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Reshape a raw time-series DataFrame into a dict of tidy DataFrames.

        Parameters
        ----------
        tsdata : pd.DataFrame
            Raw DataFrame returned by :meth:`get_data`.  Expected to have a
            ``DatetimeIndex`` (named ``'Dates'``) and ``MultiIndex`` columns
            of ``(Instrument, Field, Currency)`` or at minimum ``(Instrument, Field)``.

        Returns
        -------
        dict of {str: pd.DataFrame}
            Keys are field codes (e.g. ``'P'``, ``'RI'``).  Each value is a
            DataFrame with columns ``['real_date', 'ticker', 'currency', 'value']``
            sorted by ``(real_date, ticker)``.
        """
        if tsdata is None or tsdata.empty:
            logger.warning("process_timeseries_data received an empty DataFrame.")
            return {}

        df = tsdata.copy()

        # Ensure the index is named consistently.
        if df.index.name != "Dates":
            df.index.name = "Dates"

        # Stack all column levels to produce a long DataFrame.
        try:
            long = df.stack(level=list(range(df.columns.nlevels)), future_stack=True)
        except TypeError:
            # Older pandas versions do not support future_stack keyword.
            long = df.stack(level=list(range(df.columns.nlevels)))

        long = long.reset_index()

        # Standardise column names regardless of MultiIndex depth.
        n_cols = len(long.columns)
        if n_cols == 4:
            # (Dates, Instrument, Field, value) — no Currency level.
            long.columns = ["real_date", "ticker", "field", "value"]
            long["currency"] = pd.NA
        elif n_cols == 5:
            # (Dates, Instrument, Field, Currency, value) — full MultiIndex.
            long.columns = ["real_date", "ticker", "field", "currency", "value"]
        else:
            logger.warning(
                "process_timeseries_data: unexpected column count %d after stack.", n_cols
            )
            long.columns = (
                ["real_date"]
                + [f"level_{i}" for i in range(n_cols - 2)]
                + ["value"]
            )
            return {"unknown": long}

        # Drop rows with missing values (NaN / $$ER).
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])

        # Split by field code.
        result: Dict[str, pd.DataFrame] = {}
        for field_code, group in long.groupby("field"):
            out = (
                group[["real_date", "ticker", "currency", "value"]]
                .sort_values(["real_date", "ticker"])
                .reset_index(drop=True)
            )
            result[str(field_code)] = out

        logger.info(
            "process_timeseries_data: produced %d field DataFrame(s): %s.",
            len(result),
            list(result.keys()),
        )
        return result

    # ------------------------------------------------------------------
    # Private / internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_to_list(
        value: Union[str, List[str], Tuple[str, ...]],
    ) -> List[str]:
        """Convert a flexible input type to a list of stripped strings.

        Parameters
        ----------
        value : str, list, or tuple
            A single code, a comma-separated string, or a sequence.

        Returns
        -------
        list of str
            Each element is stripped of leading/trailing whitespace.
        """
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, (list, tuple)):
            result: List[str] = []
            for item in value:
                result.extend(
                    v.strip() for v in str(item).split(",") if v.strip()
                )
            return result
        raise TypeError(
            f"_normalize_to_list expects str, list, or tuple; got {type(value).__name__}."
        )

    @staticmethod
    def _validate_inputs(tickers: List[str], fields: List[str]) -> None:
        """Raise ``ValueError`` when either list is empty.

        Parameters
        ----------
        tickers : list of str
            Normalised ticker list.
        fields : list of str
            Normalised field list.

        Raises
        ------
        ValueError
            If *tickers* or *fields* is empty.
        """
        if not tickers:
            raise ValueError("tickers must not be empty after normalisation.")
        if not fields:
            raise ValueError("fields must not be empty after normalisation.")

    @staticmethod
    def _format_tickers_arg(tickers: List[str], multi_field: bool) -> str:
        """Produce the ``tickers`` string expected by ``ds.get_data()``.

        Rules (from DSWS documentation):

        * Single ticker, single field  → plain string ``'VOD'``.
        * Single ticker, multiple fields → angle-bracket wrapped ``'<VOD>'``.
        * Multiple tickers             → comma-separated ``'VOD,BP,HSBA'``
          (regardless of field count).

        Parameters
        ----------
        tickers : list of str
            One or more ticker codes.
        multi_field : bool
            ``True`` when multiple datatype fields are requested.

        Returns
        -------
        str
            Formatted tickers argument ready to pass to ``ds.get_data()``.
        """
        if len(tickers) == 1:
            ticker = tickers[0]
            return f"<{ticker}>" if multi_field else ticker
        return ",".join(tickers)

    @staticmethod
    def _format_date(date: Optional[Union[str, datetime]]) -> str:
        """Convert a date value to the string format expected by DSWS.

        Parameters
        ----------
        date : None, str, or datetime
            * ``None``     → ``'0D'`` (today).
            * ``datetime`` → ``'YYYY-MM-DD'``.
            * ``str``      → returned unchanged.

        Returns
        -------
        str
            Date string suitable for ``start`` / ``end`` parameters.
        """
        if date is None:
            return "0D"
        if isinstance(date, datetime):
            return date.strftime("%Y-%m-%d")
        return str(date)

    @staticmethod
    def _compute_chunk_sizes(n_tickers: int, n_fields: int) -> Tuple[int, int]:
        """Compute chunk sizes that maximise request size within DSWS limits.

        Limits enforced:
        * tickers ≤ :data:`MAX_INSTRUMENTS_PER_REQUEST` (50)
        * fields  ≤ :data:`MAX_DATATYPES_PER_REQUEST`   (50)
        * tickers × fields ≤ :data:`MAX_ITEMS_PER_REQUEST` (100)

        Parameters
        ----------
        n_tickers : int
            Total number of tickers to request.
        n_fields : int
            Total number of fields / datatypes to request.

        Returns
        -------
        tuple of (int, int)
            ``(ticker_chunk_size, field_chunk_size)`` — both guaranteed ≥ 1.
        """
        max_t = min(n_tickers, MAX_INSTRUMENTS_PER_REQUEST)
        max_f = min(n_fields, MAX_DATATYPES_PER_REQUEST)

        if max_t * max_f <= MAX_ITEMS_PER_REQUEST:
            return max_t, max_f

        # Product exceeds 100 — shrink the smaller dimension first.
        if n_fields <= n_tickers:
            # More tickers than fields: hold fields constant, reduce tickers.
            f_chunk = max_f
            while f_chunk >= 1:
                t_chunk = MAX_ITEMS_PER_REQUEST // f_chunk
                t_chunk = min(t_chunk, max_t)
                if t_chunk >= 1:
                    return t_chunk, f_chunk
                f_chunk -= 1
        else:
            # More fields than tickers: hold tickers constant, reduce fields.
            t_chunk = max_t
            while t_chunk >= 1:
                f_chunk = MAX_ITEMS_PER_REQUEST // t_chunk
                f_chunk = min(f_chunk, max_f)
                if f_chunk >= 1:
                    return t_chunk, f_chunk
                t_chunk -= 1

        # Absolute fallback — should never be reached in practice.
        logger.warning(
            "_compute_chunk_sizes fell through to fallback (1, 1) for "
            "n_tickers=%d, n_fields=%d.",
            n_tickers,
            n_fields,
        )
        return 1, 1

    @staticmethod
    def _build_chunks(
        tickers: List[str],
        fields: List[str],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Partition tickers and fields into sub-lists that respect API limits.

        Parameters
        ----------
        tickers : list of str
            Full list of ticker codes.
        fields : list of str
            Full list of datatype codes.

        Returns
        -------
        tuple of (list of list, list of list)
            ``(ticker_chunks, field_chunks)`` where each inner list is a
            sub-batch ready to send as a single API call.
        """
        t_size, f_size = DatastreamDataManager._compute_chunk_sizes(
            len(tickers), len(fields)
        )
        logger.debug(
            "_build_chunks: %d tickers / %d fields → chunk sizes (%d, %d).",
            len(tickers),
            len(fields),
            t_size,
            f_size,
        )

        ticker_chunks = [
            tickers[i: i + t_size] for i in range(0, len(tickers), t_size)
        ]
        field_chunks = [
            fields[i: i + f_size] for i in range(0, len(fields), f_size)
        ]
        return ticker_chunks, field_chunks

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DatastreamDataManager("
            f"connection={self._connection!r})"
        )
