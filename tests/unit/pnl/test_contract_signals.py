import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any

from tests.simulate import make_qdf
from macrosynergy.management.simulate_quantamental_data import make_test_df
from macrosynergy.pnl.contract_signals import (
    contract_signals,
    _apply_cscales,
    _apply_hscales,
    _apply_sig_conversion,
    _calculate_contract_signals,
)
from macrosynergy.pnl import Numeric, NoneType


class TestContractSignals(unittest.TestCase):
    def test_apply_cscales_types(self):
        cids: List[str] = ["USD", "EUR", "GBP"]
        ctypes: List[str] = ["FX", "EQ", "IR"]
        cscales: List[float] = [1.0, 2.0, 3.0]
        csigns: List[int] = [1, -1, 1]
        start: str = "2010-01-01"
        end: str = "2010-01-31"
        df: pd.DataFrame = make_test_df(
            cids=cids, xcats=ctypes, start=start, end=end, style="linear"
        )

        good_args: Dict[str, Any] = {
            "df": df,
            "cids": cids,
            "ctypes": ctypes,
            "cscales": cscales,
            "csigns": csigns,
        }

        # Test 0: Check that a dry run works
        _apply_cscales(**good_args)

        # Test 0.1 - breaks with a TypeError when incorrect type is passed
        for argn in good_args.keys():
            with self.assertRaises(TypeError):
                _apply_cscales(**{**good_args, argn: 1})
