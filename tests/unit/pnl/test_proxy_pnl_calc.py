import unittest
import matplotlib.pyplot
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import warnings
import matplotlib
from macrosynergy.compat import PD_NEW_DATE_FREQ
from macrosynergy.pnl.proxy_pnl_calc import (
    _apply_trading_costs,
    _calculate_trading_costs,
    _check_df,
    _generate_roll_dates,
    _get_rebal_dates,
    _pnl_excl_costs,
    _portfolio_sums,
    _preprocess_positions_for_costs,
    _prep_dfs_for_pnl_calcs,
    _replace_strs,
    _split_returns_positions_df,
    _split_returns_positions_tickers,
    _warn_and_drop_nans,
    proxy_pnl_calc,
    plot_pnl,
)
from macrosynergy.pnl.transaction_costs import (
    TransactionCosts,
    TransactionCostsDictAdapter,
)

from macrosynergy.download.transaction_costs import (  # noqa
    AVAIALBLE_COSTS,
    AVAILABLE_STATS,
    AVAILABLE_CTYPES,
    AVAILABLE_CATS,
)
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (  # noqa
    ticker_df_to_qdf,
    qdf_to_ticker_df,
    reduce_df,
    update_df,
    get_cid,
    get_xcat,
    _map_to_business_day_frequency,
)
import string
import random


def random_string(length: int = 10) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=length))


def make_cost_entry(bid_offer, rollcost, size):
    """Build a TransactionCostsDictAdapter cost entry in the nested schema.

    `bid_offer`, `rollcost` and `size` are each (median, pct90) tuples for the
    respective cost-type cost / size anchors. The size anchors are shared by
    both cost types.
    """

    def anchors(pair):
        return {"median": pair[0], "pct90": pair[1]}

    return {
        "bid_offer": {"size": anchors(size), "cost": anchors(bid_offer)},
        "rollcost": {"size": anchors(size), "cost": anchors(rollcost)},
    }


def flat_cost_entry(unit_cost=1.0, size_median=1.0, size_pct90=10.0):
    """Cost entry with a flat per-unit cost (median == pct90) for both cost
    types, so extrapolate_cost returns `unit_cost` for any trade size."""
    return make_cost_entry(
        bid_offer=(unit_cost, unit_cost),
        rollcost=(unit_cost, unit_cost),
        size=(size_median, size_pct90),
    )


KNOWN_FID_ENDINGS = [f"{t}_{s}" for t in AVAIALBLE_COSTS for s in AVAILABLE_STATS]


def make_tx_cost_df(
    cids: List[str] = None,
    tickers: List[str] = None,
    start="2020-01-01",
    end="2025-01-01",
) -> pd.DataFrame:
    err = "Either cids or tickers must be provided (not both)"
    assert bool(cids) or bool(tickers), err
    assert bool(cids) ^ bool(tickers), err

    if cids is None:
        tiks = tickers
    else:
        tiks = [f"{c}_{k}" for c in cids for k in AVAILABLE_CATS]

    freq = _map_to_business_day_frequency("M")
    date_range = pd.bdate_range(start=start, end=end, freq=freq)

    val_dict = {
        "BIDOFFER_MEDIAN": (0.1, 0.2),
        "BIDOFFER_90PCTL": (0.5, 2),
        "ROLLCOST_MEDIAN": (0.001, 0.006),
        "ROLLCOST_90PCTL": (0.007, 0.01),
        "SIZE_MEDIAN": (10, 20),
        "SIZE_90PCTL": (50, 70),
    }

    ct_map = {
        "FX": (10, 20),
        "IRS": (100, 150),
        "CDS": (1000, 1500),
    }

    df = pd.DataFrame(index=date_range)
    for tik in tiks:
        cid = get_cid(tik)
        # add all cols in val_dict
        for cost_type, (mn, mx) in val_dict.items():
            for fid_type, (rn, rx) in ct_map.items():
                df[f"{cid}_{fid_type}{cost_type}"] = np.random.uniform(
                    mn, mx, len(df)
                ) * np.random.uniform(rn, rx)

    df.index.name = "real_date"
    # forward will this to complete for every day
    new_index = pd.bdate_range(start=start, end=end, freq="B")
    df = df.reindex(new_index).ffill().bfill()

    return QuantamentalDataFrame.from_wide(df)


class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP"]
        self.xcats = ["FX", "EQ", "IRS", "CDS"]
        _tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats]
        self.spos = "SNAME_POS"
        self.rstring = "RETURNS"
        self.tickers: List = [f"{tk}_{self.spos}" for tk in _tickers]
        self.tickers += [f"{tk}{self.rstring}" for tk in _tickers]

        self.rd_idx = pd.Series(
            pd.bdate_range(start="2020-01-01", end="2020-12-31"), name="real_date"
        )
        self.df_wide = pd.DataFrame(
            data=np.random.randn(len(self.rd_idx), len(self.tickers)),
            index=self.rd_idx,
            columns=self.tickers,
        )
        # self.qdf = ticker_df_to_qdf(df=self.df_wide)

    def test_replace_strs(self):
        strs = [str(i) for i in range(100)]
        strs = _replace_strs(list_of_strs=strs, old_str="1", new_str="2")
        self.assertFalse("1" in "".join(strs))

        # check that the other strings are not affected
        for i in range(100):
            if "1" in str(i):
                self.assertFalse("1" in strs[i])

    def test_split_returns_positions_tickers(self):
        ret_tickers, pos_tickers = _split_returns_positions_tickers(
            tickers=self.tickers, rstring=self.rstring, spos=self.spos
        )
        self.assertEqual(sorted(get_cid(ret_tickers)), sorted(get_cid(pos_tickers)))

        for _ in range(len(ret_tickers)):
            rx = random.randint(0, len(ret_tickers) - 1)
            test_list = ret_tickers.copy()
            test_list.pop(rx)
            with self.assertRaises(ValueError):
                _split_returns_positions_tickers(
                    tickers=test_list, rstring=self.rstring, spos=self.spos
                )

    def test_split_returns_positions_df(self):
        spos = "SNAME_POS"
        cids = ["USD", "EUR", "JPY", "GBP"]
        xcats = ["FX", "EQ", "IRS", "CDS"]
        rstring = "RETURNS"
        _tickers = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
        tickers = [f"{tk}{rstring}" for tk in _tickers]
        tickers += [f"{tk}_{spos}" for tk in _tickers]

        rd_idx = pd.Series(
            pd.bdate_range(start="2020-01-01", end="2020-12-31"), name="real_date"
        )
        df = pd.DataFrame(
            data=np.random.randn(len(rd_idx), len(tickers)),
            index=rd_idx,
            columns=tickers,
        )
        ret_df, pos_df = _split_returns_positions_df(
            df_wide=df, rstring=rstring, spos=spos
        )
        self.assertEqual(
            sorted(get_cid(ret_df.columns)), sorted(get_cid(pos_df.columns))
        )

        for _ in range(len(ret_df.columns)):
            rx = random.randint(0, len(ret_df.columns) - 1)
            test_df = df.copy()
            test_df.drop(columns=[test_df.columns[rx]], inplace=True)
            with self.assertRaises(ValueError):
                _split_returns_positions_df(df_wide=test_df, rstring=rstring, spos=spos)

    def test_get_rebal_dates(self):
        _freq = "BME" if PD_NEW_DATE_FREQ else "BM"
        rd_idx = pd.Series(
            pd.bdate_range(start="2020-01-01", end="2020-12-31", freq=_freq),
            name="real_date",
        )
        df_test = self.df_wide.copy()
        df_test.loc[~df_test.index.isin(rd_idx), :] = np.nan
        rebal_dates = _get_rebal_dates(df_wide=df_test)
        self.assertEqual(set(rebal_dates), set(rd_idx))

    def test_warn_and_drop_nans(self):
        dfw = self.df_wide.copy()
        # set 10 random rows to nan, then check if they are dropped
        drop_rows = []
        for _ in range(10):
            rx = random.randint(0, len(dfw) - 1)
            drop_rows.append(dfw.index[rx])
            dfw.loc[dfw.index[rx], :] = np.nan

        rows_regex = "Warning: The following rows are all NaNs and have been dropped"
        with self.assertWarnsRegex(UserWarning, rows_regex):
            dfw = _warn_and_drop_nans(dfw)
        # check that the dropped rows are not in the index
        self.assertFalse(set(drop_rows).intersection(set(dfw.index)))

        ## Repeat test for columns
        dfw = self.df_wide.copy()
        # set 10 random columns to nan, then check if they are dropped
        drop_cols = []
        for _ in range(10):
            rx = random.randint(0, len(dfw.columns) - 1)
            drop_cols.append(dfw.columns[rx])
            dfw.loc[:, dfw.columns[rx]] = np.nan

        cols_regex = "Warning: The following columns are all NaNs and have been dropped"
        with self.assertWarnsRegex(UserWarning, cols_regex):
            dfw = _warn_and_drop_nans(dfw)

        # check that the dropped columns are not in the columns
        self.assertFalse(set(drop_cols).intersection(set(dfw.columns)))

    def test_prep_dfs_for_pnl_calcs(self):
        dfw = self.df_wide.copy()
        _prets, _ppos = _split_returns_positions_df(
            df_wide=dfw, rstring=self.rstring, spos=self.spos
        )

        rebal_dates = _get_rebal_dates(_ppos)
        _ppos.columns = _replace_strs(_ppos.columns, f"_{self.spos}")
        _prets.columns = _replace_strs(_prets.columns, self.rstring)
        _ppos, _prets = _ppos[sorted(_ppos.columns)], _prets[sorted(_prets.columns)]
        _pnl_df = (
            pd.DataFrame(
                index=pd.bdate_range(
                    _ppos.first_valid_index(), _ppos.last_valid_index()
                ),
                columns=_ppos.columns.tolist(),
            )
            .reset_index()
            .rename(columns={"index": "real_date"})
            .set_index("real_date")
        )
        expc_result_tuple = (_pnl_df, _ppos, _prets, rebal_dates)
        result_tuple = _prep_dfs_for_pnl_calcs(
            df_wide=dfw, rstring=self.rstring, spos=self.spos
        )
        # check that they equals
        for x, y in zip(expc_result_tuple, result_tuple):
            if isinstance(x, pd.DataFrame):
                self.assertTrue(x.equals(y))
            else:
                self.assertEqual(set(x), set(y))

        with self.assertRaises(AssertionError):
            tdfw = dfw.copy()
            tdfw.index.name = "date"
            _prep_dfs_for_pnl_calcs(df_wide=tdfw, rstring=self.rstring, spos=self.spos)

        # randomly set a col to nan, then check if it raises an error
        tdfw = dfw.copy()
        # cant be the first or the last real_date
        tdfw.loc[tdfw.index[50], tdfw.columns[0]] = np.nan
        with self.assertWarns(UserWarning):
            _prep_dfs_for_pnl_calcs(df_wide=tdfw, rstring=self.rstring, spos=self.spos)

    def test_check_df(self):
        df = ticker_df_to_qdf(df=self.df_wide)

        _check_df(df=df, rstring=self.rstring, spos=self.spos)

        with self.assertRaises(TypeError):
            _check_df(df=123, rstring=self.rstring, spos=self.spos)

    def test_preprocess_positions_for_costs_rejects_bad_last_row(self):
        # A position panel whose latest row carries no information (all-NaN or
        # all-zero) is rejected: we can't tell whether positions are still held,
        # so any downstream cost attribution would be guessing.
        idx = pd.bdate_range("2020-01-01", periods=5, name="real_date")
        cols = [f"USD_FX_{self.spos}", f"EUR_FX_{self.spos}"]
        df = pd.DataFrame(np.random.randn(5, len(cols)), idx, cols)
        for sentinel in (np.nan, 0.0):
            bad = df.copy()
            bad.iloc[-1] = sentinel
            with self.assertRaises(ValueError):
                _preprocess_positions_for_costs(bad)

    def test_preprocess_positions_for_costs_prepends_zero_anchor_and_fills_nans(self):
        # Leading NaNs (e.g. a contract that has not started trading yet) are
        # filled with zero, and a synthetic zero-position row is prepended one
        # business day before the panel starts. This makes the opening trade enter
        # as a regular absolute change `abs(pos[t] - pos[t-1])` rather than needing a
        # special-cased first-date charge.
        idx = pd.bdate_range("2020-01-01", periods=5, name="real_date")
        cols = [f"USD_FX_{self.spos}", f"EUR_FX_{self.spos}"]
        df = pd.DataFrame(
            [[np.nan, 1.0], [2.0, np.nan], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            index=idx,
            columns=cols,
        )
        out = _preprocess_positions_for_costs(df)
        self.assertEqual(len(out), len(df) + 1)
        self.assertEqual(out.iloc[0].tolist(), [0.0, 0.0])
        self.assertLess(out.index[0], df.index.min())
        self.assertFalse(out.isna().any().any())

    def test_generate_roll_dates_subsets_index(self):
        # Roll dates always sit on trading days that exist in the position panel,
        # and quarterly rolls are sparser than monthly, which are sparser than
        # weekly, over a multi-year window.
        idx = pd.bdate_range("2020-01-01", "2022-12-31")
        for freq in ("D", "W", "M", "Q"):
            roll_dates = _generate_roll_dates(idx, roll_freq=freq)
            self.assertTrue(set(roll_dates).issubset(set(idx)))
        m = _generate_roll_dates(idx, roll_freq="M")
        q = _generate_roll_dates(idx, roll_freq="Q")
        w = _generate_roll_dates(idx, roll_freq="W")
        self.assertGreater(len(w), len(m))
        self.assertGreater(len(m), len(q))

    def test_generate_roll_dates_date_alignment(self):
        # Date alignment: every returned roll date must be a trading day that
        # actually exists in the position panel. When a calendar period-end falls
        # on a weekend or holiday, or is otherwise absent from the panel (e.g. an
        # exchange-closure gap), the helper snaps back to the prior available
        # trading day for that period rather than silently dropping the roll.
        # This keeps the residual roll-cost calculation well-defined even when
        # the data calendar is imperfect.

        # (a) Weekend calendar month-end: 31 May 2020 is a Sunday, so May's roll
        # is anchored to the prior Friday (29 May 2020).
        idx = pd.bdate_range("2020-01-01", "2020-06-30")
        rolls = _generate_roll_dates(idx, roll_freq="M")
        self.assertIn(pd.Timestamp("2020-05-29"), rolls)
        self.assertNotIn(pd.Timestamp("2020-05-31"), rolls)

        # (b) Mid-sample gap: drop 28 Feb 2020 (a Friday and February's natural
        # month-end). The helper falls back to the prior trading day (27 Feb,
        # Thursday) so the February roll is still booked.
        idx_with_gap = idx.drop(pd.Timestamp("2020-02-28"))
        rolls = _generate_roll_dates(idx_with_gap, roll_freq="M")
        self.assertIn(pd.Timestamp("2020-02-27"), rolls)
        self.assertNotIn(pd.Timestamp("2020-02-28"), rolls)

        # (c) Truncated trailing period: data ends 15 Jan 2020 (a Wednesday) so
        # there is no real January month-end available. The last in-period
        # trading day is used; positions are still observed there.
        idx_short = pd.bdate_range("2020-01-01", "2020-01-15")
        rolls = _generate_roll_dates(idx_short, roll_freq="M")
        self.assertEqual(list(rolls), [pd.Timestamp("2020-01-15")])

        # (d) Leap year with the leap day on a weekday: 29 Feb 2024 is a
        # Thursday and is itself February's month-end - no snap-back is needed,
        # and 28 Feb must not appear instead.
        idx_leap_weekday = pd.bdate_range("2024-01-01", "2024-04-30")
        rolls = _generate_roll_dates(idx_leap_weekday, roll_freq="M")
        self.assertIn(pd.Timestamp("2024-02-29"), rolls)
        self.assertNotIn(pd.Timestamp("2024-02-28"), rolls)

        # (e) Leap year with the leap day on a weekend: 29 Feb 2020 is a
        # Saturday and so absent from the business-day panel. The February
        # roll snaps back to Friday 28 Feb, exactly as a non-leap-year
        # February-on-Saturday would.
        idx_leap_weekend = pd.bdate_range("2020-01-01", "2020-04-30")
        rolls = _generate_roll_dates(idx_leap_weekend, roll_freq="M")
        self.assertIn(pd.Timestamp("2020-02-28"), rolls)
        self.assertNotIn(pd.Timestamp("2020-02-29"), rolls)

        # (f) Weekend quarter-end: in 2024 both 31 Mar and 30 Jun fall on a
        # Sunday, so the corresponding quarterly rolls snap to the prior
        # Fridays. The Q3 end (30 Sep, a Monday) and Q4 end (31 Dec, a Tuesday)
        # fall on weekdays and remain as-is.
        idx_q = pd.bdate_range("2024-01-01", "2024-12-31")
        q_rolls = _generate_roll_dates(idx_q, roll_freq="Q")
        self.assertIn(pd.Timestamp("2024-03-29"), q_rolls)
        self.assertIn(pd.Timestamp("2024-06-28"), q_rolls)
        self.assertIn(pd.Timestamp("2024-09-30"), q_rolls)
        self.assertIn(pd.Timestamp("2024-12-31"), q_rolls)
        self.assertNotIn(pd.Timestamp("2024-03-31"), q_rolls)
        self.assertNotIn(pd.Timestamp("2024-06-30"), q_rolls)

        # (g) Subset invariant must hold for every alignment scenario above.
        cases = [
            ("M", idx),
            ("M", idx_with_gap),
            ("M", idx_short),
            ("M", idx_leap_weekday),
            ("M", idx_leap_weekend),
            ("Q", idx_q),
        ]
        for freq, test_idx in cases:
            rolls = _generate_roll_dates(test_idx, roll_freq=freq)
            self.assertTrue(set(rolls).issubset(set(test_idx)))

    def test_generate_roll_dates_rejects_unsupported_frequency(self):
        # Only "D", "W", "M", "Q" are supported; anything else is a caller error.
        idx = pd.bdate_range("2020-01-01", "2022-12-31")
        with self.assertRaises(ValueError):
            _generate_roll_dates(idx, roll_freq="A")


def mock_pnl_excl_costs(
    df_wide: pd.DataFrame, spos: str, rstring: str, pnle_name: str
) -> pd.DataFrame:
    pnl_df, pivot_pos, pivot_returns, rebal_dates = _prep_dfs_for_pnl_calcs(
        df_wide=df_wide, spos=spos, rstring=rstring
    )
    rebal_dates = sorted(
        set(rebal_dates + [pd.Timestamp(pivot_pos.last_valid_index())])
    )

    prices_df: pd.DataFrame = pd.DataFrame(
        data=1.0,
        index=pivot_returns.index,
        columns=pivot_returns.columns,
    )
    for dt1, dt2 in zip(rebal_dates[:-1], rebal_dates[1:]):
        dt1x = dt1 + pd.offsets.BDay(1)
        prices_df.loc[dt1x:dt2] = (1 + pivot_returns.loc[dt1x:dt2] / 100).cumprod()

    pnl_df = (pivot_returns / 100) * pivot_pos.shift(1) * prices_df.shift(1)
    pnl_df = pnl_df.loc[pnl_df.abs().sum(axis=1) > 0]
    pnl_df.columns = [f"{col}_{spos}_{pnle_name}" for col in pnl_df.columns]
    return pnl_df


def mock_calculate_trading_costs(
    df_wide: pd.DataFrame,
    spos: str,
    rstring: str,
    transaction_costs: TransactionCosts,
    tc_name: str,
    roll_freq: str = "M",
    bidoffer_name: str = "BIDOFFER",
    rollcost_name: str = "ROLLCOST",
) -> pd.DataFrame:
    # Naive per-(ticker, date) oracle for `_calculate_trading_costs` -- built
    # independently from production so a vectorisation bug shows up as a diff.
    _, pivot_pos = _split_returns_positions_df(
        df_wide=df_wide, spos=spos, rstring=rstring
    )
    roll_dates = _generate_roll_dates(pivot_pos.index, roll_freq)
    pivot_pos = _preprocess_positions_for_costs(pivot_pos)
    tickers = pivot_pos.columns.tolist()
    tc_cols = [
        f"{c}_{tc_name}_{k}" for c in tickers for k in (bidoffer_name, rollcost_name)
    ]
    tc_df = pd.DataFrame(0.0, index=pivot_pos.index, columns=tc_cols)
    roll_list = sorted(roll_dates)

    for ticker in tickers:
        fid = ticker.replace(f"_{spos}", "")
        col_bo = f"{ticker}_{tc_name}_{bidoffer_name}"
        col_rc = f"{ticker}_{tc_name}_{rollcost_name}"
        prev = pivot_pos[ticker].shift(1)

        for dt in pivot_pos.index:
            d = abs(pivot_pos[ticker].loc[dt] - prev.loc[dt])
            if d > 0:
                bo = transaction_costs.bidoffer(trade_size=d, fid=fid, real_date=dt)
                tc_df.loc[dt, col_bo] = d * bo / 100

        prev_bday = pivot_pos[ticker].shift(1)
        for dt in roll_list:
            # Compare the position on each roll date with the position on the
            # immediately-prior row of the position panel. The zero anchor
            # prepended by _preprocess_positions_for_costs gives the first
            # observed date a well-defined predecessor of 0, so an opening
            # trade books no roll cost.
            curr_pos = pivot_pos[ticker].loc[dt]
            prev_pos = prev_bday.loc[dt]
            held_long = curr_pos > 0 and prev_pos > 0
            held_short = curr_pos < 0 and prev_pos < 0
            if held_long or held_short:
                r = min(abs(curr_pos), abs(prev_pos))
                rc = transaction_costs.rollcost(trade_size=r, fid=fid, real_date=dt)
                tc_df.loc[dt, col_rc] = r * rc / 100

    for ticker in tickers:
        tc_df[f"{ticker}_{tc_name}"] = tc_df[
            [
                f"{ticker}_{tc_name}_{bidoffer_name}",
                f"{ticker}_{tc_name}_{rollcost_name}",
            ]
        ].sum(axis=1)

    return tc_df.loc[tc_df.abs().sum(axis=1) > 0]


class TestCalculations(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP"]
        self.xcats = ["FX", "EQ", "IRS", "CDS"]
        _tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats]
        self.spos = "SNAME_POS"
        self.rstring = "RETURNS"
        self.tickers: List = [f"{tk}_{self.spos}" for tk in _tickers]
        self.tickers += [f"{tk}{self.rstring}" for tk in _tickers]
        self.fids = [
            f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats if xcat != "EQ"
        ]

        self.rd_idx = pd.Series(
            pd.bdate_range(start="2020-01-01", end="2020-12-31"), name="real_date"
        )
        self.df_wide = pd.DataFrame(
            data=np.random.randn(len(self.rd_idx), len(self.tickers)),
            index=self.rd_idx,
            columns=self.tickers,
        )

        # self.qdf = ticker_df_to_qdf(df=self.df_wide)

    def test_pnl_excl_costs(self):
        def _test_eq(argsx: Dict[str, Any]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertTrue(
                    mock_pnl_excl_costs(**argsx).equals(_pnl_excl_costs(**argsx))
                )

        # test ideal case
        argsx = {
            "df_wide": self.df_wide.copy(),
            "spos": self.spos,
            "rstring": self.rstring,
            "pnle_name": "pnl",
        }
        _test_eq(argsx)

        # add nans in random places and check if it raises an error
        for _ in range(10):
            dfw = self.df_wide.copy()
            date_factor = 0.1
            col_factor = 0.1
            tdates = np.random.choice(
                dfw.index, int(date_factor * len(dfw.index)), replace=False
            )
            tcols = np.random.choice(
                dfw.columns, int(col_factor * len(dfw.columns)), replace=False
            )
            for dt in tdates:
                for col in tcols:
                    dfw.loc[dt, col] = np.nan
            argsx["df_wide"] = dfw
            _test_eq(argsx)

    def test_calculate_trading_costs(self):
        df_wide = self.df_wide.copy()
        df_wide = df_wide.abs()
        spos = self.spos
        rstring = self.rstring
        tc = make_tx_cost_df(cids=self.cids)
        transaction_costs = TransactionCosts(df=tc, fids=self.fids)

        def _test_eq(argsx: Dict[str, Any]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertTrue(
                    mock_calculate_trading_costs(**argsx).equals(
                        _calculate_trading_costs(**argsx)
                    )
                )

        argsx = {
            "df_wide": df_wide,
            "spos": spos,
            "rstring": rstring,
            "transaction_costs": transaction_costs,
            "tc_name": "tc",
            "roll_freq": "M",
        }

        _test_eq(argsx)

    def test_apply_trading_costs(self):
        # Extract a subset of tickers to use for pnlx_wide_df and tc_wide_df
        pnl_df = _pnl_excl_costs(
            df_wide=self.df_wide, spos=self.spos, rstring=self.rstring, pnle_name="PNLe"
        )
        tc_df = _calculate_trading_costs(
            df_wide=self.df_wide,
            spos=self.spos,
            rstring=self.rstring,
            transaction_costs=TransactionCosts(
                df=make_tx_cost_df(cids=self.cids), fids=self.fids
            ),
            tc_name="TC",
            roll_freq="M",
        )

        # Call the function
        tc_name, pnl_name, pnle_name = "TC", "PNL", "PNLe"
        bidoffer_name, rollcost_name = "BIDOFFER", "ROLLCOST"
        output_df = _apply_trading_costs(
            pnlx_wide_df=pnl_df,
            tc_wide_df=tc_df,
            spos=self.spos,
            tc_name=tc_name,
            pnl_name=pnl_name,
            pnle_name=pnle_name,
            bidoffer_name=bidoffer_name,
            rollcost_name=rollcost_name,
        )

        expc_output = pnl_df.copy()
        pnl_list = sorted(pnl_df.columns)
        tcs_list = [
            tc
            for tc in tc_df.columns.tolist()
            if not str(tc).endswith(
                (f"_{tc_name}_{bidoffer_name}", f"_{tc_name}_{rollcost_name}")
            )
        ]
        tcs_list = sorted(set(tcs_list))
        for pnl_col, tc_col in zip(pnl_list, tcs_list):
            assert pnl_col.replace(f"_{self.spos}_{pnle_name}", "") == tc_col.replace(
                f"_{self.spos}_{tc_name}", ""
            )
            expc_output[pnl_col] = expc_output[pnl_col].sub(tc_df[tc_col], fill_value=0)
        expc_output = expc_output.rename(
            columns=lambda x: str(x).replace(
                f"_{self.spos}_{pnle_name}", f"_{self.spos}_{pnl_name}"
            )
        )

        pd.testing.assert_frame_equal(output_df, expc_output)

    def test_portfolio_sums(self):
        pnl_name, tc_name, pnle_name = "PNL", "TC", "PNLE"
        bidoffer_name, rollcost_name = "BIDOFFER", "ROLLCOST"
        portfolio_name = "PORTFOLIO"

        pnl_wide = _pnl_excl_costs(
            df_wide=self.df_wide,
            spos=self.spos,
            rstring=self.rstring,
            pnle_name=pnle_name,
        )
        tc_wide = _calculate_trading_costs(
            df_wide=self.df_wide,
            spos=self.spos,
            rstring=self.rstring,
            transaction_costs=TransactionCosts(
                df=make_tx_cost_df(cids=self.cids), fids=self.fids
            ),
            tc_name=tc_name,
            roll_freq="M",
        )
        pnl_incl_costs = _apply_trading_costs(
            pnlx_wide_df=pnl_wide,
            tc_wide_df=tc_wide,
            spos=self.spos,
            tc_name=tc_name,
            pnl_name=pnl_name,
            pnle_name=pnle_name,
            bidoffer_name=bidoffer_name,
            rollcost_name=rollcost_name,
        )

        df_outs = {
            "pnl_incl_costs": pnl_incl_costs,
            "pnl_excl_costs": pnl_wide,
            "tc_wide": tc_wide,
        }

        df_outs = _portfolio_sums(
            df_outs=df_outs,
            spos=self.spos,
            portfolio_name=portfolio_name,
            pnl_name=pnl_name,
            tc_name=tc_name,
            pnle_name=pnle_name,
            bidoffer_name=bidoffer_name,
            rollcost_name=rollcost_name,
        )

        glb_pnl_incl_costs = df_outs["pnl_incl_costs"].sum(axis=1, skipna=True)
        glb_pnl_excl_costs = df_outs["pnl_excl_costs"].sum(axis=1, skipna=True)
        tcs_list = [
            tc
            for tc in df_outs["tc_wide"].columns.tolist()
            if not str(tc).endswith(
                (f"_{tc_name}_{bidoffer_name}", f"_{tc_name}_{rollcost_name}")
            )
        ]
        tcs_list = sorted(set(tcs_list))

        glb_tcosts = df_outs["tc_wide"].loc[:, tcs_list].sum(axis=1, skipna=True)
        expc_df_outs = {
            "pnl_incl_costs": pnl_incl_costs,
            "pnl_excl_costs": pnl_wide,
            "tc_wide": tc_wide,
        }
        expc_df_outs["pnl_incl_costs"].loc[
            :, f"{portfolio_name}_{self.spos}_{pnl_name}"
        ] = glb_pnl_incl_costs
        expc_df_outs["pnl_excl_costs"].loc[
            :, f"{portfolio_name}_{self.spos}_{pnle_name}"
        ] = glb_pnl_excl_costs

        expc_df_outs["tc_wide"].loc[
            :, f"{portfolio_name}_{self.spos}_{tc_name}"
        ] = glb_tcosts

        for key in df_outs.keys():
            pd.testing.assert_frame_equal(df_outs[key], expc_df_outs[key])

    def _hand_calc_setup(self):
        # 4-contract, 6 rebal-date fixture; rebal == roll == "M".
        #
        # Cost adapter: flat_cost_entry() gives each fid a flat per-unit cost
        # of 1.0 (median == pct90) for both bid-offer and roll cost, so
        # extrapolate_cost returns 1.0 for any trade size (a flat per-unit
        # cost in percentage points). The production cost path computes
        # `dollar_cost = trade_size * pct / 100`, so a trade of size N here
        # produces a dollar cost of N / 100. The expected cost tables in
        # the tests below are therefore the integer trade-size and roll-
        # residual values divided by 100.
        #
        # Position trajectories cover the regimes the cost path must handle:
        #   X_FID: long increase, long decrease, close, open short, held short
        #   Y_FID: open long, sign flip, held short, sign flip, held long
        #   Z_FID: held short, sign flip, held long x2, close
        #   S_FID: open short, short increase, short decrease, held short,
        #          short increase, held short (negative-magnitude coverage).
        spos, rstring = "SNAME_POS", "RETURNS"
        contracts = ["X_FID", "Y_FID", "Z_FID", "S_FID"]
        idx = pd.bdate_range("2020-01-01", "2020-07-31", name="real_date")
        rebal = _generate_roll_dates(idx, roll_freq="M")[:6]

        rows = pd.DataFrame(
            [
                [10, 0, -4, -5],
                [12, 5, -4, -7],
                [8, -5, 4, -3],
                [0, -5, 4, -3],
                [-5, 3, 4, -8],
                [-5, 3, 0, -8],
            ],
            index=rebal,
            columns=contracts,
            dtype=float,
        )
        positions = pd.DataFrame(np.nan, index=idx, columns=contracts)
        positions.loc[rebal] = rows.values
        positions = positions.ffill().fillna(0.0)

        pos_df = positions.add_suffix(f"_{spos}")
        ret_df = pd.DataFrame(
            0.0, index=positions.index, columns=positions.columns
        ).add_suffix(rstring)
        df_wide = pd.concat([pos_df, ret_df], axis=1)
        df_wide.index.name = "real_date"

        cost_dict = {fid: flat_cost_entry() for fid in contracts}
        adapter = TransactionCostsDictAdapter(cost_dict=cost_dict, fids=contracts)

        return df_wide, spos, rstring, adapter, rebal, contracts

    def test_calculate_trading_costs_bidoffer_matches_hand_calc(self):
        # Bid-offer is charged on each trade; the trade size is abs(new
        # target - previous target). The expected table is exactly those
        # per-contract per-rebal-date absolute deltas. S_FID exercises the
        # negative-magnitude moves explicitly: a short going from 5 to 7
        # is a trade of size 2, and a short shrinking from 7 to 3 is a
        # trade of size 4.
        df_wide, spos, rstring, adapter, rebal, contracts = self._hand_calc_setup()
        tc_df = _calculate_trading_costs(
            df_wide, spos, rstring, adapter, "TC", roll_freq="M"
        )
        expected = (
            pd.DataFrame(
                [
                    [10, 0, 4, 5],
                    [2, 5, 0, 2],
                    [4, 10, 8, 4],
                    [8, 0, 0, 0],
                    [5, 8, 0, 5],
                    [0, 0, 4, 0],
                ],
                index=rebal,
                columns=contracts,
                dtype=float,
            )
            / 100
        )
        for col in contracts:
            actual = tc_df.reindex(rebal)[f"{col}_{spos}_TC_BIDOFFER"].fillna(0.0)
            pd.testing.assert_series_equal(
                actual, expected[col].rename(actual.name), check_exact=False
            )

    def test_calculate_trading_costs_rollcost_matches_hand_calc(self):
        # Roll cost is charged on the size that survives the roll with its
        # direction unchanged: min(abs(prev_bday), abs(curr)) when both have
        # the same sign, and zero on opens, closes and sign flips. S_FID
        # covers the negative-magnitude case: a held short of size 3 carries
        # 3, a short shrinking from 7 to 3 carries 3 (the part that survived),
        # and a short growing from 3 to 8 also carries 3 (the prior size).
        df_wide, spos, rstring, adapter, rebal, contracts = self._hand_calc_setup()
        tc_df = _calculate_trading_costs(
            df_wide, spos, rstring, adapter, "TC", roll_freq="M"
        )
        expected = (
            pd.DataFrame(
                [
                    [0, 0, 0, 0],
                    [10, 0, 4, 5],
                    [8, 0, 0, 3],
                    [0, 5, 4, 3],
                    [0, 0, 4, 3],
                    [5, 3, 0, 8],
                ],
                index=rebal,
                columns=contracts,
                dtype=float,
            )
            / 100
        )
        for col in contracts:
            actual = tc_df.reindex(rebal)[f"{col}_{spos}_TC_ROLLCOST"].fillna(0.0)
            pd.testing.assert_series_equal(
                actual, expected[col].rename(actual.name), check_exact=False
            )

    def test_calculate_trading_costs_flat_rebal_charges_no_bidoffer(self):
        # A rebal date that does not change the target position is not a
        # trade and must book zero bid-offer. Contract Y's target on the
        # fifth rebal date equals its target on the fourth (both 3), so
        # its bid-offer there must be zero. Contract X moves from zero
        # to minus five on the fourth rebal, so its bid-offer there must
        # be strictly positive.
        df_wide, spos, rstring, adapter, rebal, _ = self._hand_calc_setup()
        tc_df = _calculate_trading_costs(
            df_wide, spos, rstring, adapter, "TC", roll_freq="M"
        )
        d4, d5 = rebal[4], rebal[5]
        self.assertEqual(tc_df.reindex([d5])[f"Y_FID_{spos}_TC_BIDOFFER"].iloc[0], 0.0)
        self.assertGreater(
            tc_df.reindex([d4])[f"X_FID_{spos}_TC_BIDOFFER"].iloc[0], 0.0
        )

    def test_calculate_trading_costs_alternating_signs_book_no_rollcost(self):
        # Single-contract panel that flips sign at every rebal date
        # (+5, -5, +5, -5, +5, -5). Under the prev-bday charge basis the
        # position one row before each rebal carries the prior rebal's
        # opposite sign, so the sign check fails and no held position
        # survives any flip - roll cost is exactly zero on every rebal.
        # Bid-offer is strictly positive on every rebal: the opening
        # trade of size 5, then a close-and-reopen of size 10 at each
        # subsequent flip.
        spos, rstring = "SNAME_POS", "RETURNS"
        contracts = ["F_FID"]
        idx = pd.bdate_range("2020-01-01", "2020-07-31", name="real_date")
        rebal = _generate_roll_dates(idx, roll_freq="M")[:6]

        flips = pd.DataFrame(
            [[5], [-5], [5], [-5], [5], [-5]],
            index=rebal,
            columns=contracts,
            dtype=float,
        )
        positions = pd.DataFrame(np.nan, index=idx, columns=contracts)
        positions.loc[rebal] = flips.values
        positions = positions.ffill().fillna(0.0)

        pos_df = positions.add_suffix(f"_{spos}")
        ret_df = pd.DataFrame(
            0.0, index=positions.index, columns=positions.columns
        ).add_suffix(rstring)
        df_wide = pd.concat([pos_df, ret_df], axis=1)
        df_wide.index.name = "real_date"

        cost_dict = {"F_FID": flat_cost_entry()}
        adapter = TransactionCostsDictAdapter(cost_dict=cost_dict, fids=contracts)

        tc_df = _calculate_trading_costs(
            df_wide, spos, rstring, adapter, "TC", roll_freq="M"
        )
        rc = tc_df.reindex(rebal)[f"F_FID_{spos}_TC_ROLLCOST"].fillna(0.0)
        bo = tc_df.reindex(rebal)[f"F_FID_{spos}_TC_BIDOFFER"].fillna(0.0)
        self.assertTrue((rc == 0.0).all())
        self.assertTrue((bo > 0.0).all())

    def test_calculate_trading_costs_rollcost_uses_prev_bday_not_prev_roll(self):
        # Spec ref: macrosynergy/academy#2132 comment 4461473326. Roll cost
        # compares the position on the roll date to the position on the
        # immediately-previous trading day, not to the position on the
        # previous roll date. Build a four-monthly-roll panel where the
        # position changes between consecutive rolls so the two semantics
        # disagree:
        #
        #   anchor (Dec 31 2019): 0   (zero-anchor prepended by preprocessing)
        #   Jan 02 - Jan 31: +20  -> roll[0] = Jan 31 (Fri)
        #   Feb 03 - Feb 28: +50  -> roll[1] = Feb 28 (Fri); intra-period grow
        #   Mar 02 - Mar 30: +50
        #   Mar 31:          +10  -> roll[2] = Mar 31 (Tue); same-day shrink
        #   Apr 01 - Apr 29: +10
        #   Apr 30:          -7   -> roll[3] = Apr 30 (Thu); same-day sign flip
        #
        # Cost adapter: flat 1 % cost per USD, so dollar charge = size / 100.
        # Per-roll expected ROLLCOST (with the prev-roll-date semantic shown
        # in parentheses for contrast):
        #
        #   roll[0]: prev_bday=Jan 30=+20, curr=+20 -> min=20 -> 0.20
        #            (prev_roll semantic: no prev roll -> 0.00)
        #   roll[1]: prev_bday=Feb 27=+50, curr=+50 -> min=50 -> 0.50
        #            (prev_roll semantic: min(20, 50)=20  -> 0.20)
        #   roll[2]: prev_bday=Mar 30=+50, curr=+10 -> min=10 -> 0.10
        #            (prev_roll semantic: min(50, 10)=10 -> 0.10, same)
        #   roll[3]: prev_bday=Apr 29=+10, curr=-7  -> sign flip -> 0.00
        #            (prev_roll semantic: sign flip   -> 0.00, same)
        spos, rstring = "SNAME_POS", "RETURNS"
        contracts = ["T_FID"]
        idx = pd.bdate_range("2020-01-01", "2020-04-30", name="real_date")
        rolls = _generate_roll_dates(idx, roll_freq="M")
        self.assertEqual(len(rolls), 4)

        pos = pd.Series(20.0, index=idx, name=f"T_FID_{spos}")
        pos.iloc[idx.get_loc(rolls[0]) + 1 :] = 50.0
        pos.iloc[idx.get_loc(rolls[2]) :] = 10.0
        pos.iloc[idx.get_loc(rolls[3]) :] = -7.0

        df_wide = pd.concat(
            [
                pos.to_frame(),
                pd.DataFrame(0.0, index=idx, columns=[f"T_FID{rstring}"]),
            ],
            axis=1,
        )
        df_wide.index.name = "real_date"

        adapter = TransactionCostsDictAdapter(
            cost_dict={"T_FID": flat_cost_entry()},
            fids=contracts,
        )
        tc_df = _calculate_trading_costs(
            df_wide, spos, rstring, adapter, "TC", roll_freq="M"
        )
        rc = tc_df.reindex(rolls)[f"T_FID_{spos}_TC_ROLLCOST"].fillna(0.0)
        self.assertAlmostEqual(rc.iloc[0], 0.20, places=12)
        self.assertAlmostEqual(rc.iloc[1], 0.50, places=12)
        self.assertAlmostEqual(rc.iloc[2], 0.10, places=12)
        self.assertAlmostEqual(rc.iloc[3], 0.00, places=12)

    def test_calculate_trading_costs_rejects_bad_last_row(self):
        # An all-zero (or all-NaN) latest position row leaves the cost
        # path with no information about what is currently held.
        # _preprocess_positions_for_costs rejects such a panel with a
        # ValueError; this test asserts that the rejection surfaces all
        # the way through _calculate_trading_costs and is not silently
        # swallowed.
        df_wide, spos, rstring, adapter, _, _ = self._hand_calc_setup()
        pos_cols = [c for c in df_wide.columns if c.endswith(f"_{spos}")]
        df_wide.loc[df_wide.index[-1], pos_cols] = 0.0
        with self.assertRaises(ValueError):
            _calculate_trading_costs(
                df_wide, spos, rstring, adapter, "TC", roll_freq="M"
            )

    def test_proxy_pnl_calc_matches_hand_calc_via_public_api(self):
        # Public-API parity with the bid-offer and roll-cost hand-calc
        # tables, run end-to-end through proxy_pnl_calc instead of the
        # private cost path. The shared _hand_calc_setup fixture exercises
        # opens, closes, sign flips, held longs, held shorts and negative-
        # magnitude transitions across four contracts.
        #
        # Edge cases covered in a single test:
        #   - exact bid-offer value per (rebal, contract) under "M"
        #   - exact roll-cost value per (rebal, contract) under "M"
        #   - zero bid-offer and zero roll-cost on every non-rebal day
        #     (positions are flat between rebals, so no charge can book)
        #   - pure sign flip (Y_FID, rebal[2]: 5 -> -5) books bid-offer
        #     on size 10 and zero roll-cost
        #   - held short (X_FID, rebal[5]: -5 -> -5) books roll-cost on
        #     size 5 and zero bid-offer
        #   - close (Z_FID, rebal[5]: 4 -> 0) books bid-offer on size 4
        #     and zero roll-cost
        #   - bid-offer is invariant under roll_freq (it depends on
        #     position changes, not on the roll schedule)
        #   - roll-cost is strictly reduced when roll_freq tightens from
        #     "M" to "Q" (fewer roll dates means fewer charging dates)
        df_wide, spos, rstring, adapter, rebal, contracts = self._hand_calc_setup()
        # _pnl_excl_costs drops all-zero pnl rows, which empties the panel
        # if every return is zero. Inject a small constant return on every
        # business day so the public-API path produces a non-empty PnL.
        # Costs are independent of returns, so the bid-offer and roll-cost
        # tables checked below are unchanged by this.
        ret_cols = [c for c in df_wide.columns if c.endswith(rstring)]
        df_wide.loc[:, ret_cols] = 0.01
        qdf = ticker_df_to_qdf(df_wide)

        common_kwargs = dict(
            df=qdf,
            spos=spos,
            rstring=rstring,
            transaction_costs_object=adapter,
            return_pnl_excl_costs=True,
            return_costs=True,
            concat_dfs=False,
        )

        _, _, tc_qdf_m = proxy_pnl_calc(roll_freq="M", **common_kwargs)
        tc_wide_m = qdf_to_ticker_df(tc_qdf_m)

        # _generate_roll_dates returns a sorted python list, so the expected
        # tables need an explicit DatetimeIndex name to match the "real_date"
        # name carried through tc_wide_m from qdf_to_ticker_df.
        rebal_index = pd.DatetimeIndex(rebal, name="real_date")
        expected_bidoffer = (
            pd.DataFrame(
                [
                    [10, 0, 4, 5],
                    [2, 5, 0, 2],
                    [4, 10, 8, 4],
                    [8, 0, 0, 0],
                    [5, 8, 0, 5],
                    [0, 0, 4, 0],
                ],
                index=rebal_index,
                columns=contracts,
                dtype=float,
            )
            / 100
        )
        expected_rollcost = (
            pd.DataFrame(
                [
                    [0, 0, 0, 0],
                    [10, 0, 4, 5],
                    [8, 0, 0, 3],
                    [0, 5, 4, 3],
                    [0, 0, 4, 3],
                    [5, 3, 0, 8],
                ],
                index=rebal_index,
                columns=contracts,
                dtype=float,
            )
            / 100
        )

        bo_cols = [f"{c}_{spos}_TCOST_BIDOFFER" for c in contracts]
        rc_cols = [f"{c}_{spos}_TCOST_ROLLCOST" for c in contracts]

        for col, bo_col, rc_col in zip(contracts, bo_cols, rc_cols):
            bo_actual = tc_wide_m.reindex(rebal)[bo_col].fillna(0.0)
            rc_actual = tc_wide_m.reindex(rebal)[rc_col].fillna(0.0)
            pd.testing.assert_series_equal(
                bo_actual,
                expected_bidoffer[col].rename(bo_actual.name),
                check_exact=False,
            )
            pd.testing.assert_series_equal(
                rc_actual,
                expected_rollcost[col].rename(rc_actual.name),
                check_exact=False,
            )

        # Off-schedule charges must be zero. Bid-offer can only book on a
        # rebal date (positions are constant in between); roll-cost can
        # only book on a roll date (the cost path's roll schedule). Both
        # sets are derived directly from the panel rather than hard-coded.
        full_roll_dates = _generate_roll_dates(df_wide.index, "M")
        non_rebal = tc_wide_m.index.difference(pd.DatetimeIndex(rebal))
        non_roll = tc_wide_m.index.difference(pd.DatetimeIndex(full_roll_dates))
        bo_off = tc_wide_m.loc[non_rebal, bo_cols].fillna(0.0)
        rc_off = tc_wide_m.loc[non_roll, rc_cols].fillna(0.0)
        self.assertTrue((bo_off.to_numpy() == 0.0).all())
        self.assertTrue((rc_off.to_numpy() == 0.0).all())

        # A roll date past the last rebal (Jul 31 here) still books
        # roll-cost on whatever was held coming out of the prior rebal,
        # because the position is carried unchanged across the roll. This
        # exercises the "held across a roll with no trade" regime.
        extra_roll_dates = [
            d for d in full_roll_dates if d not in pd.DatetimeIndex(rebal)
        ]
        self.assertEqual(len(extra_roll_dates), 1)
        last_roll = extra_roll_dates[0]
        expected_held_after_last_rebal = {
            "X_FID": 5,  # -5 held short
            "Y_FID": 3,  # 3 held long
            "Z_FID": 0,  # closed at rebal[5]
            "S_FID": 8,  # -8 held short
        }
        for fid, expected in expected_held_after_last_rebal.items():
            self.assertAlmostEqual(
                tc_wide_m.loc[last_roll, f"{fid}_{spos}_TCOST_ROLLCOST"],
                expected / 100,
            )
            # No trade happens on this roll date, so bid-offer must be zero.
            self.assertAlmostEqual(
                tc_wide_m.loc[last_roll, f"{fid}_{spos}_TCOST_BIDOFFER"], 0.0
            )

        # Regime-boundary spot checks pulled out of the tables above so
        # a regression in any one regime fails with an obvious message.
        self.assertAlmostEqual(
            tc_wide_m.loc[rebal[2], f"Y_FID_{spos}_TCOST_ROLLCOST"], 0.0
        )
        self.assertAlmostEqual(
            tc_wide_m.loc[rebal[2], f"Y_FID_{spos}_TCOST_BIDOFFER"], 10 / 100
        )
        self.assertAlmostEqual(
            tc_wide_m.loc[rebal[5], f"X_FID_{spos}_TCOST_BIDOFFER"], 0.0
        )
        self.assertAlmostEqual(
            tc_wide_m.loc[rebal[5], f"X_FID_{spos}_TCOST_ROLLCOST"], 5 / 100
        )
        self.assertAlmostEqual(
            tc_wide_m.loc[rebal[5], f"Z_FID_{spos}_TCOST_ROLLCOST"], 0.0
        )
        self.assertAlmostEqual(
            tc_wide_m.loc[rebal[5], f"Z_FID_{spos}_TCOST_BIDOFFER"], 4 / 100
        )

        # Re-run with quarterly rolls. Bid-offer is driven by position
        # changes and must be identical on every rebal date; roll-cost
        # is driven by the roll schedule and must shrink strictly in
        # total when the schedule sparsens.
        _, _, tc_qdf_q = proxy_pnl_calc(roll_freq="Q", **common_kwargs)
        tc_wide_q = qdf_to_ticker_df(tc_qdf_q)

        bo_m_on_rebal = tc_wide_m.reindex(rebal)[bo_cols].fillna(0.0)
        bo_q_on_rebal = tc_wide_q.reindex(rebal)[bo_cols].fillna(0.0)
        pd.testing.assert_frame_equal(bo_m_on_rebal, bo_q_on_rebal, check_exact=False)
        rc_total_m = tc_wide_m[rc_cols].abs().sum().sum()
        rc_total_q = tc_wide_q[rc_cols].abs().sum().sum()
        self.assertGreater(rc_total_m, rc_total_q)
        self.assertGreater(rc_total_q, 0.0)


class TestProxyPNLCalc(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP"]
        self.xcats = ["FX", "EQ", "IRS", "CDS"]
        _tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats]
        self.spos = "SNAME_POS"
        self.rstring = "RETURNS"
        pos_tickers: List[str] = [f"{tk}_{self.spos}" for tk in _tickers]
        ret_tickers: List[str] = [f"{tk}{self.rstring}" for tk in _tickers]
        self.tickers: List[str] = pos_tickers + ret_tickers
        self.fids = [
            f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats if xcat != "EQ"
        ]

        self.rd_idx = pd.Series(
            pd.bdate_range(start="2020-01-01", end="2020-12-31"), name="real_date"
        )
        # Returns: a synthetic daily return series for each contract.
        # Positions: drawn on the monthly rebalancing dates and forward-filled
        # in between, so the cost path sees a realistic monthly rebalancing
        # cadence rather than a fresh position on every business day.
        rebal_idx = _generate_roll_dates(
            index=pd.DatetimeIndex(self.rd_idx), roll_freq="M"
        )
        pos_panel = pd.DataFrame(index=self.rd_idx, columns=pos_tickers, dtype=float)
        pos_panel.loc[rebal_idx] = np.random.randn(len(rebal_idx), len(pos_tickers))
        pos_panel = pos_panel.ffill().bfill()
        ret_panel = pd.DataFrame(
            data=np.random.randn(len(self.rd_idx), len(ret_tickers)),
            index=self.rd_idx,
            columns=ret_tickers,
        )
        self.df_wide = pd.concat([pos_panel, ret_panel], axis=1)[self.tickers]
        self.qdf = ticker_df_to_qdf(df=self.df_wide)
        self.tc = TransactionCosts(df=make_tx_cost_df(cids=self.cids), fids=self.fids)
        self.good_args = {
            "df": self.qdf,
            "spos": self.spos,
            "rstring": self.rstring,
            "transaction_costs_object": self.tc,
            "roll_freq": None,
            "start": None,
            "end": None,
            "blacklist": None,
            "portfolio_name": "GLB",
            "pnl_name": "PNL",
            "tc_name": "TCOST",
            "bidoffer_name": "BIDOFFER",
            "rollcost_name": "ROLLCOST",
            "return_pnl_excl_costs": True,
            "return_costs": True,
            "concat_dfs": False,
        }

    def test_proxy_pnl_calc(self):
        full_result = proxy_pnl_calc(**self.good_args)

        self.assertIsInstance(full_result, tuple)
        self.assertEqual(len(full_result), 3)
        for res in full_result:
            self.assertIsInstance(res, QuantamentalDataFrame)

        pnl_found_tickers = QuantamentalDataFrame(full_result[0]).list_tickers()
        pnle_found_tickers = QuantamentalDataFrame(full_result[1]).list_tickers()
        trunc_pnle_found_tickers = [t[:-1] for t in pnle_found_tickers]
        self.assertEqual(trunc_pnle_found_tickers, pnl_found_tickers)

        found_tc_tickers = QuantamentalDataFrame(full_result[2]).list_tickers()
        trunc_tc_tickers = [t.split(self.spos)[0] for t in found_tc_tickers]
        trunc_pnl_tickers = [t.split(self.spos)[0] for t in pnl_found_tickers]
        self.assertEqual(set(trunc_tc_tickers), set(trunc_pnl_tickers))

        # now check with concat_dfs=True
        self.good_args["concat_dfs"] = True
        concat_result: QuantamentalDataFrame = proxy_pnl_calc(**self.good_args)
        self.assertIsInstance(concat_result, QuantamentalDataFrame)
        expected_result: QuantamentalDataFrame = pd.concat(full_result, axis=0)
        self.assertTrue(
            expected_result.sort_values(by=QuantamentalDataFrame.IndexCols)
            .reset_index(drop=True)
            .equals(
                concat_result.sort_values(
                    by=QuantamentalDataFrame.IndexCols
                ).reset_index(drop=True)
            )
        )

    def test_proxy_pnl_calc_return_args(self):
        # call with return_costs=False
        full_result = proxy_pnl_calc(**self.good_args)
        args_copy = self.good_args.copy()

        args_copy["return_costs"] = False
        args_copy["concat_dfs"] = True
        result = proxy_pnl_calc(**args_copy)

        expected_result = pd.concat([full_result[0], full_result[1]], axis=0)

        self.assertTrue(
            expected_result.sort_values(by=QuantamentalDataFrame.IndexCols)
            .reset_index(drop=True)
            .equals(
                result.sort_values(by=QuantamentalDataFrame.IndexCols).reset_index(
                    drop=True
                )
            )
        )

    def test_proxy_pnl_calc_return_args2(self):
        full_result = proxy_pnl_calc(**self.good_args)
        args_copy = self.good_args.copy()
        args_copy["return_pnl_excl_costs"] = False
        args_copy["return_costs"] = False
        args_copy["concat_dfs"] = True

        result = proxy_pnl_calc(**args_copy)

        expected_result = full_result[0]

        self.assertTrue(
            expected_result.sort_values(by=QuantamentalDataFrame.IndexCols)
            .reset_index(drop=True)
            .equals(
                result.sort_values(by=QuantamentalDataFrame.IndexCols).reset_index(
                    drop=True
                )
            )
        )

    def test_proxy_pnl_calc_with_cost_dict(self):
        pos_tickers = [tk for tk in self.tickers if tk.endswith(f"_{self.spos}")]
        fids = sorted({tk.replace(f"_{self.spos}", "") for tk in pos_tickers})

        # Distinct bid-offer and roll-cost anchors to confirm the adapter
        # routes each cost type independently (matching the panel-backed
        # TransactionCosts object built below).
        bo_cost = (0.2, 0.4)  # (median, pct90)
        ro_cost = (0.1, 0.3)
        size = (35, 90)
        cost_dict = {
            fid: make_cost_entry(bid_offer=bo_cost, rollcost=ro_cost, size=size)
            for fid in fids
        }

        df_const = pd.DataFrame(index=self.df_wide.index)
        for fid in fids:
            df_const[f"{fid}BIDOFFER_MEDIAN"] = bo_cost[0]
            df_const[f"{fid}BIDOFFER_90PCTL"] = bo_cost[1]
            df_const[f"{fid}ROLLCOST_MEDIAN"] = ro_cost[0]
            df_const[f"{fid}ROLLCOST_90PCTL"] = ro_cost[1]
            df_const[f"{fid}SIZE_MEDIAN"] = size[0]
            df_const[f"{fid}SIZE_90PCTL"] = size[1]
        df_const.index.name = "real_date"

        tc_qdf = QuantamentalDataFrame.from_wide(df_const)
        tc_obj = TransactionCosts(df=tc_qdf, fids=fids)

        args_obj = self.good_args.copy()
        args_obj["transaction_costs_object"] = tc_obj
        args_dict = self.good_args.copy()
        args_dict["transaction_costs_object"] = cost_dict

        result_obj = proxy_pnl_calc(**args_obj)
        result_dict = proxy_pnl_calc(**args_dict)

        for res_a, res_b in zip(result_obj, result_dict):
            self.assertTrue(
                res_a.sort_values(by=QuantamentalDataFrame.IndexCols)
                .reset_index(drop=True)
                .equals(
                    res_b.sort_values(by=QuantamentalDataFrame.IndexCols).reset_index(
                        drop=True
                    )
                )
            )

    def test_proxy_pnl_calc_accepts_str_roll_freq(self):
        # proxy_pnl_calc must accept a frequency string for `roll_freq` other
        # than the default and propagate it into the cost path. Quarterly's
        # schedule is strictly sparser than monthly's, so on the same input
        # the M run must book non-zero roll cost on more dates and a higher
        # total roll cost than the Q run -- if the kwarg were ignored, the
        # two would match. (The cost QDF emits one row per cost-eligible
        # date for each contract, so total row counts coincide; the count
        # that varies with `roll_freq` is the count of non-zero entries.)
        #
        # A small inline fixture is used (2 contracts, 1 year, flat-cost
        # adapter) so the test stays cheap relative to TestProxyPNLCalc.setUp.
        cids = ["USD", "EUR"]
        spos, rstring = "SNAME_POS", "XR"
        idx = pd.bdate_range("2020-01-01", "2020-12-31")
        rebal = _generate_roll_dates(idx, roll_freq="M")
        pos_tickers = [f"{cid}_FX_{spos}" for cid in cids]
        ret_tickers = [f"{cid}_FX{rstring}" for cid in cids]
        pos_panel = pd.DataFrame(np.nan, index=idx, columns=pos_tickers)
        pos_panel.loc[rebal] = np.random.randn(len(rebal), len(pos_tickers))
        pos_panel = pos_panel.ffill().bfill()
        # _pnl_excl_costs drops all-zero pnl rows; with zero returns the
        # panel would be empty. A small constant return keeps the PnL
        # frame populated without affecting the cost-path assertions.
        ret_panel = pd.DataFrame(0.01, index=idx, columns=ret_tickers)
        df_wide = pd.concat([pos_panel, ret_panel], axis=1)
        df_wide.index.name = "real_date"
        qdf = ticker_df_to_qdf(df_wide)

        fids = [f"{cid}_FX" for cid in cids]
        adapter = TransactionCostsDictAdapter(
            cost_dict={fid: flat_cost_entry() for fid in fids},
            fids=fids,
        )

        base = dict(
            df=qdf,
            spos=spos,
            rstring=rstring,
            transaction_costs_object=adapter,
            return_pnl_excl_costs=True,
            return_costs=True,
            concat_dfs=False,
        )

        _, _, tc_m = proxy_pnl_calc(roll_freq="M", **base)
        _, _, tc_q = proxy_pnl_calc(roll_freq="Q", **base)

        rc_m = tc_m[tc_m.xcat.str.endswith("_ROLLCOST")]
        rc_q = tc_q[tc_q.xcat.str.endswith("_ROLLCOST")]
        self.assertGreater((rc_m.value != 0).sum(), (rc_q.value != 0).sum())
        self.assertGreater(rc_m.value.abs().sum(), rc_q.value.abs().sum())
        self.assertGreater((rc_q.value != 0).sum(), 0)

    def test_proxy_pnl_calc_rejects_dict_roll_freq(self):
        # Per-fid dict form of `roll_freq` is reserved for a future release;
        # passing a dict must raise NotImplementedError so callers know to
        # pass a single frequency string (or None) for now. The rejection
        # happens before any data processing, so a one-row QDF is enough.
        qdf = QuantamentalDataFrame(
            pd.DataFrame(
                {
                    "cid": ["USD"],
                    "xcat": ["FX_SNAME_POS"],
                    "real_date": [pd.Timestamp("2020-01-01")],
                    "value": [1.0],
                }
            )
        )
        with self.assertRaises(NotImplementedError):
            proxy_pnl_calc(
                df=qdf,
                spos="SNAME_POS",
                rstring="XR",
                transaction_costs_object=None,
                roll_freq={"FX": "M"},
            )

    def test_plot_pnl(self):
        self.good_args["concat_dfs"] = True
        pnl_df = proxy_pnl_calc(**self.good_args)
        matplotlib.pyplot.close("all")
        mpl_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_pnl(pnl_df, portfolio_name=self.good_args["portfolio_name"])
        matplotlib.pyplot.close("all")
        matplotlib.use(mpl_backend)


if __name__ == "__main__":
    unittest.main()
