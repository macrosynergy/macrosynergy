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

from typing import List, Union, SupportsInt, SupportsFloat, Optional
import numpy as np

NoneType = type(None)
Numeric = Union[int, float, np.int64, np.float64, SupportsInt, SupportsFloat]


def _short_xcat(
    ticker: Optional[str] = None,
    xcat: Optional[str] = None,
):
    """
    Get the short version of the cross-section category.

    :param <str> ticker: the ticker of the contract.
    :param <str> xcat: a extended category of the contract.

    :return <str>: the category from the ticker/xcat.
    """
    if ticker is not None and xcat is not None:
        raise ValueError("Either `ticker` or `xcat` must be specified, not both")

    if ticker is not None:
        cid, xcat = ticker.split("_", 1)
        return _short_xcat(xcat=xcat)
    elif xcat is not None:
        return xcat.split("_")[-1]
    else:
        raise ValueError("Either `ticker` or `xcat` must be specified")

