import unittest
from unittest import mock
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, Set
from numbers import Number
import warnings
from macrosynergy.compat import PD_NEW_DATE_FREQ
from macrosynergy.pnl.proxy_pnl_calc import (
    _apply_trading_costs,
    _calculate_trading_costs,
    _check_df,
    _get_rebal_dates,
    _pnl_excl_costs,
    _portfolio_sums,
    _prep_dfs_for_pnl_calcs,
    _replace_strs,
    _split_returns_positions_df,
    _split_returns_positions_tickers,
    _warn_and_drop_nans,
    proxy_pnl_calc,
)


from macrosynergy.pnl.historic_portfolio_volatility import (
    historic_portfolio_vol,
    RETURN_SERIES_XCAT,
)
from macrosynergy.pnl.contract_signals import contract_signals
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.utils import (
    is_valid_iso_date,
    standardise_dataframe,
    ticker_df_to_qdf,
    get_sops,
    qdf_to_ticker_df,
    reduce_df,
    update_df,
    get_cid,
    get_xcat,
)
import string
import random


def random_string(length: int = 10) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=length))


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
            with self.assertRaises(AssertionError):
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
            with self.assertRaises(AssertionError):
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


def mock_pnl_excl_costs(
    df_wide: pd.DataFrame, spos: str, rstring: str, pnl_name: str
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
    pnl_df.columns = [f"{col}_{spos}_{pnl_name}" for col in pnl_df.columns]
    return pnl_df


class TestCalculations(unittest.TestCase):
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
            "pnl_name": "pnl",
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


if __name__ == "__main__":
    unittest.main()
