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
    def setUp(self) -> None:
        self.cids: List[str] = ["USD", "EUR", "JPY", "GBP"]
        self.fcats: List[str] = ["FX", "CDS", "IRS"]
        self.sname: str = "STRATx"
        self.pname: str = "POSz"
        self.sig_ident: str = f"_CSIG_{self.sname}"
        self.fids: List[str] = [
            f"{cid}_{fcat}" for cid in self.cids for fcat in self.fcats
        ]
        ticker_endings = [f"{fcat}{self.sig_ident}" for fcat in self.fcats]
        self.f_tickers: List[str] = [
            f"{cid}_{te}" for cid in self.cids for te in ticker_endings
        ]

        self.mock_df = make_test_df(
            start="2019-01-01",
            end="2019-02-01",
            cids=self.cids,
            xcats=ticker_endings,
        )
        self.mock_df_wide = qdf_to_ticker_df(self.mock_df)

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
        # Test vanilla case
        wide_df = self.mock_df_wide.copy()
        _check_df_for_contract_signals(
            df_wide=wide_df, sname=self.sname, fids=self.fids
        )
        # Test ValueError with missing column
        col_names = list(wide_df.columns)
        removed_col = col_names.pop(np.random.randint(len(col_names)))
        wide_df = wide_df.drop(columns=[removed_col])
        with self.assertRaises(ValueError):
            _check_df_for_contract_signals(
                df_wide=wide_df,
                sname=self.sname,
                fids=self.fids,
            )

    def test__leverage_positions(self):
        df_wide = self.mock_df_wide.copy()
        # set all values to 1
        df_wide.loc[:, :] = 1
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        result = _leverage_positions(
            df_wide=df_wide,
            sname=self.sname,
            pname=self.pname,
            fids=fx_fids,
            leverage=1,
        )
        # col names should be the FID+strat+pos
        expected_cols = [f"{fid}_{self.sname}_{self.pname}" for fid in fx_fids]
        found_cols = list(result.columns)
        self.assertEqual(set(expected_cols), set(found_cols))

        for cola, colb in zip(found_cols[:-1], found_cols[1:]):
            self.assertTrue(result[cola].equals(result[colb]))


if __name__ == "__main__":
    unittest.main()
