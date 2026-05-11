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
from macrosynergy.pnl.transaction_costs import TransactionCosts

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
        # as a regular absolute change `|pos[t] - pos[t-1]|` rather than needing a
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
    bidoffer_name: str = "BIDOFFER",
    rollcost_name: str = "ROLLCOST",
) -> pd.DataFrame:
    _, pivot_pos = _split_returns_positions_df(
        df_wide=df_wide, spos=spos, rstring=rstring
    )
    rebal_dates = _get_rebal_dates(pivot_pos)
    _end = pd.Timestamp(pivot_pos.last_valid_index())
    rebal_dates = sorted(set(rebal_dates + [_end]))
    pos_cols = pivot_pos.columns.tolist()
    tc_cols = [
        f"{col}_{tc_name}_{cost_type}"
        for col in pos_cols
        for cost_type in [bidoffer_name, rollcost_name]
    ]
    # Create a dataframe to store the trading costs with all 0s
    tc_df = pd.DataFrame(data=0.0, index=pivot_pos.index, columns=tc_cols)
    tickers = pivot_pos.columns.tolist()
    ## Taking the 1st position
    ## Here, only the bidoffer is considered, as there is nothing to roll
    first_pos = pivot_pos.loc[rebal_dates[0]]
    for ticker in tickers:
        _fid = ticker.replace(f"_{spos}", "")
        bidoffer = transaction_costs.bidoffer(
            trade_size=first_pos[ticker],
            fid=_fid,
            real_date=rebal_dates[0],
        )
        # Add a 0 for rollcost
        tc_df.loc[rebal_dates[0], f"{ticker}_{tc_name}_{rollcost_name}"] = 0
        tc_df.loc[rebal_dates[0], f"{ticker}_{tc_name}_{bidoffer_name}"] = (
            bidoffer / 100
        )

    for ix, (dt1, dt2) in enumerate(zip(rebal_dates[:-1], rebal_dates[1:])):
        dt2x = dt2 - pd.offsets.BDay(1)
        prev_pos, next_pos = pivot_pos.loc[dt1], pivot_pos.loc[dt2]
        curr_pos = pivot_pos.loc[dt1:dt2x]
        avg_pos: pd.Series = curr_pos.abs().mean(axis=0)
        delta_pos = (next_pos - prev_pos).abs()
        for ticker in tickers:
            _fid = ticker.replace(f"_{spos}", "")
            _rcn = f"{ticker}_{tc_name}_{rollcost_name}"
            _bon = f"{ticker}_{tc_name}_{bidoffer_name}"
            rollcost = transaction_costs.rollcost(
                trade_size=avg_pos[ticker],
                fid=_fid,
                real_date=dt2,
            )
            bidoffer = transaction_costs.bidoffer(
                trade_size=delta_pos[ticker],
                fid=_fid,
                real_date=dt2,
            )
            # delta_pos and avg_pos are already in absolute terms
            tc_df.loc[dt2, _rcn] = avg_pos[ticker] * rollcost / 100
            tc_df.loc[dt2, _bon] = delta_pos[ticker] * bidoffer / 100

    # Sum TICKER_TCOST_BIDOFFER and TICKER_TCOST_ROLLCOST into TICKER_TCOST
    for ticker in tickers:
        tc_df[f"{ticker}_{tc_name}"] = tc_df[
            [
                f"{ticker}_{tc_name}_{bidoffer_name}",
                f"{ticker}_{tc_name}_{rollcost_name}",
            ]
        ].sum(axis=1)
    # Drop rows with no trading costs
    tc_df = tc_df.loc[tc_df.abs().sum(axis=1) > 0]
    # check that remaining dates are part of rebal_dates
    assert set(tc_df.index) <= set(rebal_dates)
    assert not (tc_df < 0).any().any()
    return tc_df


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


class TestProxyPNLCalc(unittest.TestCase):

    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP"]
        self.xcats = ["FX", "EQ", "IRS", "CDS"]
        _tickers = [f"{cid}_{xcat}" for cid in self.cids for xcat in self.xcats]
        self.spos = "SNAME_POS"
        self.rstring = "RETURNS"
        self.tickers: List[str] = [f"{tk}_{self.spos}" for tk in _tickers]
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
        self.qdf = ticker_df_to_qdf(df=self.df_wide)
        self.tc = TransactionCosts(df=make_tx_cost_df(cids=self.cids), fids=self.fids)
        self.good_args = {
            "df": self.qdf,
            "spos": self.spos,
            "rstring": self.rstring,
            "transaction_costs_object": self.tc,
            "roll_freqs": None,
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

        cost_template = {
            "median_cost": 0.2,
            "median_size": 35,
            "pct90_cost": 0.4,
            "pct90_size": 90,
        }
        cost_dict = {fid: dict(cost_template) for fid in fids}

        df_const = pd.DataFrame(index=self.df_wide.index)
        for fid in fids:
            df_const[f"{fid}BIDOFFER_MEDIAN"] = cost_template["median_cost"]
            df_const[f"{fid}BIDOFFER_90PCTL"] = cost_template["pct90_cost"]
            df_const[f"{fid}ROLLCOST_MEDIAN"] = cost_template["median_cost"]
            df_const[f"{fid}ROLLCOST_90PCTL"] = cost_template["pct90_cost"]
            df_const[f"{fid}SIZE_MEDIAN"] = cost_template["median_size"]
            df_const[f"{fid}SIZE_90PCTL"] = cost_template["pct90_size"]
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
