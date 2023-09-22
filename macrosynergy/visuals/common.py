"""
Common types and functions used across the modules of the `macrosynergy.pnl` subpackage.

---

## **Types** 

---

`:type <Union[int, float, np.int64, np.float64, SupportsInt, SupportsFloat]> Numeric`: 
    a numeric type that supports type checks across `int`, `float`, `np.int64`, 
    `np.float64`, `SupportsInt`, and `SupportsFloat`.

`:type <NoneType> NoneType`: the type of `None`.

"""

from typing import List, Union, SupportsInt, SupportsFloat
import numpy as np

NoneType = type(None)
Numeric = Union[int, float, np.int64, np.float64, SupportsInt, SupportsFloat]


