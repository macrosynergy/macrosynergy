"""Test historical volatility estimates with simulate returns from random normal distribution"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional
from numbers import Number
from unittest import mock
import warnings

from macrosynergy.download.transaction_costs import (
    AVAIALBLE_COSTS,
    AVAILABLE_STATS,
)
from macrosynergy.pnl.transaction_costs import (
    get_fids,
    check_df_for_txn_stats,
    get_diff_index,
    extrapolate_cost,
    SparseCosts,
    TransactionCosts,
)
from macrosynergy.management.utils import (
    qdf_to_ticker_df,
    get_sops,
    ticker_df_to_qdf,
    _map_to_business_day_frequency,
)
from macrosynergy.management.types import QuantamentalDataFrame, NoneType
from macrosynergy.management.simulate import make_test_df, simulate_returns_and_signals
