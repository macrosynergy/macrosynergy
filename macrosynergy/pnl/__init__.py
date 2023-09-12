from typing import List, Union, Tuple, Optional, SupportsInt, SupportsFloat
import numpy as np


from macrosynergy.pnl.naive_pnl import NaivePnL


NoneType = type(None)
Numeric = Union[int, float, np.int64, np.float64, SupportsInt, SupportsFloat]


__all__ = ['Numeric', 'NoneType', 'NaivePnL']