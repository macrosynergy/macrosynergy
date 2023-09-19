from macrosynergy.pnl.naive_pnl import NaivePnL

from macrosynergy.pnl.common import Numeric, NoneType, _short_xcat

from macrosynergy.pnl.contract_signals import contract_signals
from macrosynergy.pnl.notional_positions import notional_positions
from macrosynergy.pnl.proxy_pnl import proxy_pnl


TYPES = [Numeric, NoneType]
TYPES = [f.__name__ for f in TYPES]

CLASSES = [NaivePnL]
CLASSES = [f.__name__ for f in CLASSES]

FUNCTIONS = [contract_signals, notional_positions, proxy_pnl]
FUNCTIONS = [f.__name__ for f in FUNCTIONS]

__all__ = TYPES + CLASSES + FUNCTIONS
