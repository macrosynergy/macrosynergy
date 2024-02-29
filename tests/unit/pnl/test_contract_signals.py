import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any

from tests.simulate import make_qdf
from macrosynergy.management.simulate import make_test_df
from macrosynergy.pnl.contract_signals import contract_signals, _check_arg_types
from macrosynergy.management.types import Numeric, QuantamentalDataFrame


class TestContractSignals(unittest.TestCase):
    def setUp(self):
        self.cids: List[str] = ["USD", "EUR", "GBP", "AUD", "CAD"]
        self.xcats: List[str] = ["SIG", "HR"]
        self.start: str = "2000-01-01"
        self.end: str = "2020-12-31"
        self.ctypes: List[str] = ["FX", "IRS", "CDS"]
        self.cscales: List[Numeric] = [1.0, 0.5, 0.1]
        self.csigns: List[int] = [1, -1, 1]
        self.hbasket: List[str] = ["USD_EQ", "EUR_EQ"]
        self.hscales: List[Numeric] = [0.7, 0.3]

    def testDF(self) -> QuantamentalDataFrame:
        return make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start=self.start,
            end=self.end,
        )
