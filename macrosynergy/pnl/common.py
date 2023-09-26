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

from typing import List, Union, SupportsInt, SupportsFloat, Optional, TypeAlias
import numpy as np

from numbers import SupportsInt, SupportsFloat
import numpy as np


class NumericType(type):
    """
    MetaClass to support type checks across `int`, `float`, `np.int64`, `np.float64`,
    `SupportsInt`, and `SupportsFloat`.
    """

    # A tuple of the desired types
    _numeric_types = (int, float, np.int64, np.float64, SupportsInt, SupportsFloat)

    def __instancecheck__(cls, instance):
        return isinstance(instance, cls._numeric_types)


class Numeric(metaclass=NumericType):
    """
    Custom class definition for a numeric type that supports type checks across `int`,
    `float`, `np.int64`, `np.float64`, `SupportsInt`, and `SupportsFloat`.
    """

    # Alternatively, use `numbers.Number` directly
    pass


class NoneType(type):
    """
    MetaClass to support type checks for `None`.
    """

    def __instancecheck__(cls, instance):
        return instance is None


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
