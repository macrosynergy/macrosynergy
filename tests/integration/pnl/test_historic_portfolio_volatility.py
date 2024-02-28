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