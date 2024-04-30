import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, Set
from numbers import Number
from macrosynergy.pnl.notional_positions import (
    notional_positions,
    _apply_slip,
    _check_df_for_contract_signals,
    _vol_target_positions,
    _leverage_positions,
    notional_positions,
)
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.utils import (
    is_valid_iso_date,
    standardise_dataframe,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
    reduce_df,
    update_df,
    get_cid,
    get_xcat,
)


class TestNotionalPositions(unittest.TestCase):
    def test__apply_slip(self):
        cids = ["USD", "EUR", "JPY", "GBP"]
        fcats = ["FX", "CDS", "IRS"]
        tdf = make_test_df(start="2019-01-01", end="2019-01-10", cids=cids, xcats=fcats)
        fids = [f"{cid}_{fcat}" for cid in cids for fcat in fcats]
        removed_fid = fids.pop(np.random.randint(len(fids)))
        result = _apply_slip(df=tdf, slip=4, fids=fids)
        out_tickers: List[str] = sorted(set(result["cid"] + "_" + result["xcat"]))
        self.assertTrue(removed_fid not in out_tickers)
        self.assertIsInstance(result, QuantamentalDataFrame)

    def test__check_df_for_contract_signals(self):
        cids = ["USD", "EUR", "JPY", "GBP"]
        fcats = ["FX", "CDS", "IRS"]
        sname = "testX"
        sig_ident: str = f"_CSIG_{sname}"

        wide_df = qdf_to_ticker_df(
            make_test_df(start="2019-01-01", end="2019-02-01", cids=cids, xcats=fcats)
        )
        fids = [f"{cid}_{fcat}" for cid in cids for fcat in fcats]

        col_names = [f"{cid}_{fcat}{sig_ident}" for cid in cids for fcat in fcats]
        wide_df.columns = col_names

        _check_df_for_contract_signals(df_wide=wide_df, sname=sname, fids=fids)

        # pop a random column
        removed_col = col_names.pop(np.random.randint(len(col_names)))
        wide_df = wide_df.drop(columns=[removed_col])
        with self.assertRaises(ValueError):
            _check_df_for_contract_signals(df_wide=wide_df, sname=sname, fids=fids)


if __name__ == "__main__":
    unittest.main()
