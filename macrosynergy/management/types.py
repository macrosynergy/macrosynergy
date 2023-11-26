"""
Module hosting custom types and meta-classes for use across the package.
"""

# Imports
from typing import Any, Dict, List, Optional, Tuple, Union, SupportsFloat, SupportsInt
import numpy as np
import pandas as pd


class NumericType(type):
    """
    MetaClass to support type checks across `int`, `float`, `np.int64`, `np.float64`,
    `SupportsInt`, and `SupportsFloat`.
    """

    # A tuple of the desired types
    _numeric_types = (int, float, np.int64, np.float64)

    def __instancecheck__(cls, instance):
        # if type(instance) not in cls._numeric_types:
        #     return False
        return isinstance(instance, cls._numeric_types)


class Numeric(metaclass=NumericType):
    """
    Custom class definition for a numeric type that supports type checks across `int`,
    `float`, `np.int64`, `np.float64`, `SupportsInt`, and `SupportsFloat`.
    """

    # Alternatively, use `numbers.Number` directly
    pass


class NoneTypeMeta(type):
    """
    MetaClass to support type checks for `None`.
    """

    def __instancecheck__(cls, instance):
        return instance is None or isinstance(instance, type(None))


class NoneType(metaclass=NoneTypeMeta):
    """
    Custom class definition for a NoneType that supports type checks for `None`.
    """

    pass


class QuantamentalDataFrameMeta(type):
    """
    MetaClass to support type checks for `QuantamentalDataFrame`.
    """

    IndexCols: List[str] = ["cid", "xcat", "real_date"]

    def __instancecheck__(cls, instance):
        IDX_COLS = QuantamentalDataFrame.IndexCols
        result: bool = True
        try:
            # the try except offers a safety net in case the instance is not a pd.DataFrame
            # and one of the checks raises an error
            result = result and isinstance(instance, pd.DataFrame)
            result = result and instance.index.name is None
            result = result and not isinstance(instance.columns, pd.MultiIndex)
            result = result and all([col in instance.columns for col in IDX_COLS])
            result = result and len(instance.columns) > len(IDX_COLS)

            correct_date_type: bool = instance[
                "real_date"
            ].dtype == "datetime64[ns]" or isinstance(
                instance["real_date"].dtype, pd.DatetimeTZDtype
            ) or instance.empty
            result = result and correct_date_type

            # # check if the cid col is all str
            # # check if the xcat col is all str
        except:
            result: bool = False
        finally:
            return result


class QuantamentalDataFrame(metaclass=QuantamentalDataFrameMeta):
    """
    Class definition for a QuantamentalDataFrame that supports type checks for
    `QuantamentalDataFrame`.
    Returns True if the instance is a `pd.DataFrame` with the standard Quantamental
    DataFrame columns ("cid", "xcat", "real_date") and at least one additional column.
    It also checks if the "real_date" column is a datetime type.

    Usage:
    >>> df: pd.DataFrame = make_test_df()
    >>> isinstance(df, QuantamentalDataFrame)
    True
    """

    pass
