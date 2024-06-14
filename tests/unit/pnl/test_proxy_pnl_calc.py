import unittest
from unittest import mock
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, Set
from numbers import Number
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


class TestFunctions(unittest.TestCase):
    def setUp(): ...

    def test_replace_strs(self):
        strs = [str(i) for i in range(100)]
        strs = _replace_strs(list_of_strs=strs, old="1", new="2")
        self.assertFalse("1" in "".join(strs))

        # check that the other strings are not affected
        for i in range(100):
            if "1" in str(i):
                self.assertFalse("1" in strs[i])

    def test_split_returns_positions_tickers(self):
        spos = "SNAME_POS"
        rstring = "RETURNS"
        cids = [random_string(3) for _ in range(20)]
        tickers = [f"{cid}_{rstring}" for cid in cids]
        tickers += [f"{cid}_{spos}" for cid in cids]

        ret_tickers, pos_tickers = _split_returns_positions_tickers(
            tickers=tickers, rstring=rstring, spos=spos
        )
        
