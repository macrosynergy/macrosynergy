"""
Unit tests for DatastreamDataManager and parse_list_name.

DatastreamPy is an optional external dependency that is not available in CI,
so it is stubbed out in sys.modules before any import of the modules under
test.
"""

import sys
import unittest
from datetime import date, datetime
from unittest.mock import MagicMock, call, patch

import pandas as pd

# ---------------------------------------------------------------------------
# Stub DatastreamPy before importing the modules under test.
# ---------------------------------------------------------------------------
sys.modules.setdefault("DatastreamPy", MagicMock())

from macrosynergy.download.datastream.connection import DatastreamConnection  # noqa: E402
from macrosynergy.download.datastream.data_manager import (  # noqa: E402
    DEFAULT_STATIC_FIELDS,
    MAX_DATATYPES_PER_REQUEST,
    MAX_INSTRUMENTS_PER_REQUEST,
    MAX_ITEMS_PER_REQUEST,
    DatastreamDataManager,
    parse_list_name,
)
_USERNAME = "DS:ID"
# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_mock_ds(get_data_return=None):
    """Return (mock_connection, mock_ds_client) with get_data pre-configured."""
    mock_ds = MagicMock()
    mock_ds.get_data.return_value = get_data_return
    mock_conn = MagicMock(spec=DatastreamConnection)
    mock_conn.get_connection.return_value = mock_ds
    return mock_conn, mock_ds


def _make_manager(get_data_return=None, show_usage_stats=False):
    """Return (DatastreamDataManager, mock_ds_client)."""
    mock_conn, mock_ds = _make_mock_ds(get_data_return)
    mgr = DatastreamDataManager(connection=mock_conn, show_usage_stats=show_usage_stats)
    return mgr, mock_ds


def _constituents_df(mnemonics):
    return pd.DataFrame({"MNEM": mnemonics})


def _metadata_df(instruments, fields, values, currency="USD"):
    rows = [
        {"Instrument": i, "Datatype": f, "Value": v, "Currency": currency}
        for i, f, v in zip(instruments, fields, values)
    ]
    return pd.DataFrame(rows)


def _timeseries_df(dates, instruments, fields):
    idx = pd.DatetimeIndex(dates, name="Dates")
    cols = pd.MultiIndex.from_tuples(
        [(inst, fld, "USD") for inst in instruments for fld in fields],
        names=["Instrument", "Field", "Currency"],
    )
    data = [[1.0] * len(cols)] * len(dates)
    return pd.DataFrame(data, index=idx, columns=cols)


def _usage_stats_df(datapoints=3004):
    return pd.DataFrame(
        {
            "Dates": ["2026-06-02"] * 7,
            "Instrument": ["STATS"] * 7,
            "Datatype": [
                "User",
                "Hits",
                "Requests",
                "Datatypes",
                "Datapoints",
                "Start Date",
                "End Date",
            ],
            "Value": [
                _USERNAME,
                "18",
                "12",
                "3004",
                str(datapoints),
                "2026-06-01",
                "2026-06-30",
            ],
            "Currency": ["NA"] * 7,
        }
    )


def _stats_calls(mock_ds):
    """Return only the get_data calls that target the STATS ticker."""
    return [c for c in mock_ds.get_data.call_args_list if c[1].get("tickers") == "STATS"]


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit(unittest.TestCase):
    def test_init_with_connection(self):
        mock_conn = MagicMock(spec=DatastreamConnection)
        mgr = DatastreamDataManager(connection=mock_conn)
        self.assertIs(mgr._connection, mock_conn)
        self.assertFalse(mgr._show_usage_stats)

    def test_init_with_credentials(self):
        with patch(
            "macrosynergy.download.datastream.data_manager.DatastreamConnection"
        ) as MockConn:
            DatastreamDataManager(username="DS:ID", password="secret")
            MockConn.assert_called_once_with(username="DS:ID", password="secret")

    def test_init_missing_args_raises(self):
        with self.assertRaises(ValueError):
            DatastreamDataManager()

    def test_show_usage_stats_stored_true(self):
        mock_conn = MagicMock(spec=DatastreamConnection)
        mgr = DatastreamDataManager(connection=mock_conn, show_usage_stats=True)
        self.assertTrue(mgr._show_usage_stats)

    def test_show_usage_stats_default_false(self):
        mock_conn = MagicMock(spec=DatastreamConnection)
        mgr = DatastreamDataManager(connection=mock_conn)
        self.assertFalse(mgr._show_usage_stats)


# ---------------------------------------------------------------------------
# _log_usage_stats
# ---------------------------------------------------------------------------


class TestLogUsageStats(unittest.TestCase):
    def test_no_api_call_when_flag_false(self):
        mgr, mock_ds = _make_manager(show_usage_stats=False)
        mgr._log_usage_stats()
        mock_ds.get_data.assert_not_called()

    def test_calls_stats_endpoint_when_flag_true(self):
        today = date(2026, 6, 2)
        mgr, mock_ds = _make_manager(
            get_data_return=_usage_stats_df(), show_usage_stats=True
        )
        with patch(
            "macrosynergy.download.datastream.data_manager.date"
        ) as mock_date:
            mock_date.today.return_value = today
            mgr._log_usage_stats()
        mock_ds.get_data.assert_called_once_with(
            tickers="STATS",
            fields=["DS.USERSTATS"],
            kind=0,
            start="2026-06-02",
        )

    def test_prints_stats_output(self):
        mgr, _ = _make_manager(
            get_data_return=_usage_stats_df(), show_usage_stats=True
        )
        with patch("builtins.print") as mock_print:
            mgr._log_usage_stats()
        printed = "\n".join(str(c[0][0]) for c in mock_print.call_args_list)
        self.assertIn("Hits", printed)
        self.assertIn("Datapoints", printed)
        self.assertIn("monthly quota", printed)

    def test_user_row_not_printed(self):
        mgr, _ = _make_manager(
            get_data_return=_usage_stats_df(), show_usage_stats=True
        )
        with patch("builtins.print") as mock_print:
            mgr._log_usage_stats()
        printed = "\n".join(str(c[0][0]) for c in mock_print.call_args_list)
        self.assertNotIn(_USERNAME, printed)

    def test_does_not_print_when_flag_false(self):
        mgr, _ = _make_manager(show_usage_stats=False)
        with patch("builtins.print") as mock_print:
            mgr._log_usage_stats()
        mock_print.assert_not_called()

    def test_quota_warning_above_90_pct(self):
        mgr, _ = _make_manager(
            get_data_return=_usage_stats_df(datapoints=950_000),
            show_usage_stats=True,
        )
        with patch("builtins.print") as mock_print:
            mgr._log_usage_stats()
        printed = "\n".join(str(c[0][0]) for c in mock_print.call_args_list)
        self.assertIn("WARNING", printed)
        self.assertIn("95.0%", printed)

    def test_no_quota_warning_below_90_pct(self):
        mgr, _ = _make_manager(
            get_data_return=_usage_stats_df(datapoints=500_000),
            show_usage_stats=True,
        )
        with patch("builtins.print") as mock_print:
            mgr._log_usage_stats()
        printed = "\n".join(str(c[0][0]) for c in mock_print.call_args_list)
        self.assertNotIn("WARNING", printed)

    def test_api_exception_is_suppressed(self):
        mgr, mock_ds = _make_manager(show_usage_stats=True)
        mock_ds.get_data.side_effect = RuntimeError("network failure")
        mgr._log_usage_stats()  # must not raise

    def test_empty_stats_response_does_not_crash(self):
        mgr, _ = _make_manager(get_data_return=pd.DataFrame(), show_usage_stats=True)
        mgr._log_usage_stats()  # must not raise


# ---------------------------------------------------------------------------
# get_constituents
# ---------------------------------------------------------------------------


class TestGetConstituents(unittest.TestCase):
    def test_returns_sorted_deduplicated_list(self):
        df = _constituents_df(["VOD", "BP", "HSBA", "VOD"])
        mgr, _ = _make_manager(get_data_return=df)
        self.assertEqual(mgr.get_constituents("LFTSE100"), ["BP", "HSBA", "VOD"])

    def test_appends_list_suffix_when_absent(self):
        mgr, mock_ds = _make_manager(get_data_return=_constituents_df(["VOD"]))
        mgr.get_constituents("LFTSE100")
        self.assertEqual(mock_ds.get_data.call_args[1]["tickers"], "LFTSE100|L")

    def test_does_not_double_suffix(self):
        mgr, mock_ds = _make_manager(get_data_return=_constituents_df(["VOD"]))
        mgr.get_constituents("LFTSE100|L")
        self.assertEqual(mock_ds.get_data.call_args[1]["tickers"], "LFTSE100|L")

    def test_empty_response_returns_empty_list(self):
        mgr, _ = _make_manager(get_data_return=pd.DataFrame())
        self.assertEqual(mgr.get_constituents("LFTSE100"), [])

    def test_error_marker_raises_value_error(self):
        df = pd.DataFrame({"Value": ["$$ER: unknown list"]})
        mgr, _ = _make_manager(get_data_return=df)
        with self.assertRaises(ValueError):
            mgr.get_constituents("BADLIST")

    def test_api_exception_propagates(self):
        mgr, mock_ds = _make_manager()
        mock_ds.get_data.side_effect = ConnectionError("timeout")
        with self.assertRaises(ConnectionError):
            mgr.get_constituents("LFTSE100")

    def test_usage_stats_called_once_when_flag_true(self):
        mgr, mock_ds = _make_manager(
            get_data_return=_constituents_df(["VOD"]), show_usage_stats=True
        )
        mgr.get_constituents("LFTSE100")
        self.assertEqual(len(_stats_calls(mock_ds)), 1)

    def test_usage_stats_not_called_when_flag_false(self):
        mgr, mock_ds = _make_manager(
            get_data_return=_constituents_df(["VOD"]), show_usage_stats=False
        )
        mgr.get_constituents("LFTSE100")
        self.assertEqual(len(_stats_calls(mock_ds)), 0)


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------


class TestGetMetadata(unittest.TestCase):
    def test_returns_dataframe(self):
        df = _metadata_df(["VOD", "VOD"], ["NAME", "RIC"], ["Vodafone", "VOD.L"])
        mgr, _ = _make_manager(get_data_return=df)
        result = mgr.get_metadata(["VOD"], fields=["NAME", "RIC"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_empty_response_returns_empty_dataframe(self):
        mgr, _ = _make_manager(get_data_return=pd.DataFrame())
        result = mgr.get_metadata(["VOD"], fields=["NAME"])
        self.assertTrue(result.empty)

    def test_raises_on_empty_tickers(self):
        mgr, _ = _make_manager()
        with self.assertRaises(ValueError):
            mgr.get_metadata([], fields=["NAME"])

    def test_raises_on_empty_fields(self):
        mgr, _ = _make_manager()
        with self.assertRaises(ValueError):
            mgr.get_metadata(["VOD"], fields=[])

    def test_comma_separated_string_tickers(self):
        df = _metadata_df(["VOD", "BP"], ["NAME", "NAME"], ["Vodafone", "BP plc"])
        mgr, mock_ds = _make_manager(get_data_return=df)
        result = mgr.get_metadata("VOD,BP", fields="NAME")
        self.assertFalse(result.empty)

    def test_usage_stats_called_once_when_flag_true(self):
        df = _metadata_df(["VOD"], ["NAME"], ["Vodafone"])
        mgr, mock_ds = _make_manager(get_data_return=df, show_usage_stats=True)
        mgr.get_metadata(["VOD"], fields=["NAME"])
        self.assertEqual(len(_stats_calls(mock_ds)), 1)

    def test_usage_stats_not_called_when_flag_false(self):
        df = _metadata_df(["VOD"], ["NAME"], ["Vodafone"])
        mgr, mock_ds = _make_manager(get_data_return=df, show_usage_stats=False)
        mgr.get_metadata(["VOD"], fields=["NAME"])
        self.assertEqual(len(_stats_calls(mock_ds)), 0)

    def test_usage_stats_called_on_empty_response_when_flag_true(self):
        mgr, mock_ds = _make_manager(get_data_return=pd.DataFrame(), show_usage_stats=True)
        mgr.get_metadata(["VOD"], fields=["NAME"])
        self.assertEqual(len(_stats_calls(mock_ds)), 1)


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------


class TestGetData(unittest.TestCase):
    def test_returns_dataframe(self):
        df = _timeseries_df(["2024-01-01", "2024-01-02"], ["VOD"], ["P"])
        mgr, _ = _make_manager(get_data_return=df)
        result = mgr.get_data(["VOD"], fields=["P"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_empty_response_returns_empty_dataframe(self):
        mgr, _ = _make_manager(get_data_return=pd.DataFrame())
        result = mgr.get_data(["VOD"], fields=["P"])
        self.assertTrue(result.empty)

    def test_raises_on_empty_tickers(self):
        mgr, _ = _make_manager()
        with self.assertRaises(ValueError):
            mgr.get_data([], fields=["P"])

    def test_raises_on_empty_fields(self):
        mgr, _ = _make_manager()
        with self.assertRaises(ValueError):
            mgr.get_data(["VOD"], fields=[])

    def test_passes_start_end_freq_to_api(self):
        df = _timeseries_df(["2024-01-01"], ["VOD"], ["P"])
        mgr, mock_ds = _make_manager(get_data_return=df)
        mgr.get_data(["VOD"], fields=["P"], start="-3M", end="0D", freq="W")
        data_call = mock_ds.get_data.call_args
        self.assertEqual(data_call[1]["start"], "-3M")
        self.assertEqual(data_call[1]["end"], "0D")
        self.assertEqual(data_call[1]["freq"], "W")

    def test_usage_stats_called_once_when_flag_true(self):
        df = _timeseries_df(["2024-01-01"], ["VOD"], ["P"])
        mgr, mock_ds = _make_manager(get_data_return=df, show_usage_stats=True)
        mgr.get_data(["VOD"], fields=["P"])
        self.assertEqual(len(_stats_calls(mock_ds)), 1)

    def test_usage_stats_not_called_when_flag_false(self):
        df = _timeseries_df(["2024-01-01"], ["VOD"], ["P"])
        mgr, mock_ds = _make_manager(get_data_return=df, show_usage_stats=False)
        mgr.get_data(["VOD"], fields=["P"])
        self.assertEqual(len(_stats_calls(mock_ds)), 0)

    def test_usage_stats_called_on_empty_response_when_flag_true(self):
        mgr, mock_ds = _make_manager(get_data_return=pd.DataFrame(), show_usage_stats=True)
        mgr.get_data(["VOD"], fields=["P"])
        self.assertEqual(len(_stats_calls(mock_ds)), 1)


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


class TestNormalizeToList(unittest.TestCase):
    def test_single_string(self):
        self.assertEqual(DatastreamDataManager._normalize_to_list("VOD"), ["VOD"])

    def test_comma_separated_string(self):
        self.assertEqual(
            DatastreamDataManager._normalize_to_list("VOD, BP, HSBA"),
            ["VOD", "BP", "HSBA"],
        )

    def test_list_passthrough(self):
        self.assertEqual(
            DatastreamDataManager._normalize_to_list(["VOD", "BP"]), ["VOD", "BP"]
        )

    def test_tuple_passthrough(self):
        self.assertEqual(
            DatastreamDataManager._normalize_to_list(("VOD", "BP")), ["VOD", "BP"]
        )

    def test_strips_whitespace(self):
        self.assertEqual(
            DatastreamDataManager._normalize_to_list("  VOD  ,  BP  "), ["VOD", "BP"]
        )

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            DatastreamDataManager._normalize_to_list(123)


class TestValidateInputs(unittest.TestCase):
    def test_raises_on_empty_tickers(self):
        with self.assertRaises(ValueError):
            DatastreamDataManager._validate_inputs([], ["P"])

    def test_raises_on_empty_fields(self):
        with self.assertRaises(ValueError):
            DatastreamDataManager._validate_inputs(["VOD"], [])

    def test_valid_does_not_raise(self):
        DatastreamDataManager._validate_inputs(["VOD"], ["P"])


class TestFormatTickersArg(unittest.TestCase):
    def test_single_ticker_single_field(self):
        self.assertEqual(
            DatastreamDataManager._format_tickers_arg(["VOD"], multi_field=False), "VOD"
        )

    def test_single_ticker_multi_field_wrapped(self):
        self.assertEqual(
            DatastreamDataManager._format_tickers_arg(["VOD"], multi_field=True), "<VOD>"
        )

    def test_multi_ticker_comma_joined(self):
        self.assertEqual(
            DatastreamDataManager._format_tickers_arg(["VOD", "BP"], multi_field=False),
            "VOD,BP",
        )

    def test_multi_ticker_multi_field_comma_joined(self):
        self.assertEqual(
            DatastreamDataManager._format_tickers_arg(["VOD", "BP"], multi_field=True),
            "VOD,BP",
        )


class TestFormatDate(unittest.TestCase):
    def test_none_returns_today_placeholder(self):
        self.assertEqual(DatastreamDataManager._format_date(None), "0D")

    def test_string_passthrough(self):
        self.assertEqual(DatastreamDataManager._format_date("2024-01-15"), "2024-01-15")

    def test_datetime_formats_as_iso(self):
        self.assertEqual(
            DatastreamDataManager._format_date(datetime(2024, 3, 5)), "2024-03-05"
        )


class TestComputeChunkSizes(unittest.TestCase):
    def test_small_request_unchanged(self):
        t, f = DatastreamDataManager._compute_chunk_sizes(5, 5)
        self.assertEqual(t, 5)
        self.assertEqual(f, 5)

    def test_respects_max_instruments(self):
        t, _ = DatastreamDataManager._compute_chunk_sizes(200, 1)
        self.assertLessEqual(t, MAX_INSTRUMENTS_PER_REQUEST)

    def test_respects_max_datatypes(self):
        _, f = DatastreamDataManager._compute_chunk_sizes(1, 200)
        self.assertLessEqual(f, MAX_DATATYPES_PER_REQUEST)

    def test_product_within_item_limit(self):
        t, f = DatastreamDataManager._compute_chunk_sizes(50, 50)
        self.assertLessEqual(t * f, MAX_ITEMS_PER_REQUEST)

    def test_minimum_one_for_any_input(self):
        t, f = DatastreamDataManager._compute_chunk_sizes(1, 1)
        self.assertGreaterEqual(t, 1)
        self.assertGreaterEqual(f, 1)


# ---------------------------------------------------------------------------
# process_metadata
# ---------------------------------------------------------------------------


class TestProcessMetadata(unittest.TestCase):
    def _wide_input(self):
        return pd.DataFrame(
            {
                "Instrument": ["VOD", "VOD"],
                "Datatype": ["NAME", "RIC"],
                "Value": ["Vodafone", "VOD.L"],
                "Currency": ["USD", "USD"],
            }
        )

    def test_returns_wide_dataframe_with_field_columns(self):
        result = DatastreamDataManager.process_metadata(self._wide_input())
        self.assertIn("NAME", result.columns)
        self.assertIn("RIC", result.columns)
        self.assertEqual(result.index.name, "ticker")

    def test_empty_input_returns_empty(self):
        self.assertTrue(
            DatastreamDataManager.process_metadata(pd.DataFrame()).empty
        )

    def test_error_rows_dropped(self):
        df = pd.DataFrame(
            {
                "Instrument": ["VOD", "ERR"],
                "Datatype": ["NAME", "NAME"],
                "Value": ["Vodafone", "$$ER: bad ticker"],
                "Currency": ["USD", "USD"],
            }
        )
        result = DatastreamDataManager.process_metadata(df)
        self.assertNotIn("ERR", result.index)

    def test_null_currency_rows_dropped(self):
        df = pd.DataFrame(
            {
                "Instrument": ["VOD", "BP"],
                "Datatype": ["NAME", "NAME"],
                "Value": ["Vodafone", "BP plc"],
                "Currency": ["USD", None],
            }
        )
        result = DatastreamDataManager.process_metadata(df)
        self.assertNotIn("BP", result.index)

    def test_deaddt_cast_to_datetime(self):
        df = pd.DataFrame(
            {
                "Instrument": ["VOD"],
                "Datatype": ["DEADDT"],
                "Value": ["2020-06-01"],
                "Currency": ["USD"],
            }
        )
        result = DatastreamDataManager.process_metadata(df)
        self.assertEqual(result["DEADDT"].dtype, "datetime64[ns]")


# ---------------------------------------------------------------------------
# process_timeseries_data
# ---------------------------------------------------------------------------


class TestProcessTimeseriesData(unittest.TestCase):
    def test_returns_dict_keyed_by_field(self):
        df = _timeseries_df(["2024-01-01", "2024-01-02"], ["VOD", "BP"], ["P", "RI"])
        result = DatastreamDataManager.process_timeseries_data(df)
        self.assertIn("P", result)
        self.assertIn("RI", result)

    def test_output_has_correct_columns(self):
        df = _timeseries_df(["2024-01-01"], ["VOD"], ["P"])
        result = DatastreamDataManager.process_timeseries_data(df)
        self.assertSetEqual(
            set(result["P"].columns), {"real_date", "ticker", "currency", "value"}
        )

    def test_sorted_by_real_date_and_ticker(self):
        df = _timeseries_df(["2024-01-02", "2024-01-01"], ["VOD", "BP"], ["P"])
        result = DatastreamDataManager.process_timeseries_data(df)
        dates = result["P"]["real_date"].tolist()
        self.assertEqual(dates, sorted(dates))

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(DatastreamDataManager.process_timeseries_data(pd.DataFrame()), {})

    def test_nan_values_dropped(self):
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="Dates")
        cols = pd.MultiIndex.from_tuples(
            [("VOD", "P", "USD")], names=["Instrument", "Field", "Currency"]
        )
        df = pd.DataFrame([[1.0], [float("nan")]], index=idx, columns=cols)
        result = DatastreamDataManager.process_timeseries_data(df)
        self.assertEqual(len(result["P"]), 1)


# ---------------------------------------------------------------------------
# parse_list_name
# ---------------------------------------------------------------------------


class TestParseListName(unittest.TestCase):
    def test_january_2024(self):
        self.assertEqual(parse_list_name("DOW30_list_0124"), date(2024, 1, 31))

    def test_december_2023(self):
        self.assertEqual(parse_list_name("SP500_1223"), date(2023, 12, 31))

    def test_february_2024_leap_year(self):
        self.assertEqual(parse_list_name("LIST_0224"), date(2024, 2, 29))

    def test_1990s_century_inference(self):
        self.assertEqual(parse_list_name("LIST_0198"), date(1998, 1, 31))

    def test_1980s_century_inference(self):
        # Leading digit '8' maps to 1900s per the documented convention.
        self.assertEqual(parse_list_name("LIST_1189"), date(1989, 11, 30))


if __name__ == "__main__":
    unittest.main()
