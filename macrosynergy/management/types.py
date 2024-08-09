"""
Module hosting custom types and meta-classes for use across the package.
"""

from typing import List, Optional
import pandas as pd


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

    IndexCols: List[str] = ["real_date", "cid", "xcat"]

    def __instancecheck__(cls, instance):
        IDX_COLS = QuantamentalDataFrame.IndexCols
        result: bool = True
        try:
            # the try except offers a safety net in case the instance is not a
            # pd.DataFrame and one of the checks raises an error
            result = result and isinstance(instance, pd.DataFrame)
            result = result and instance.index.name is None
            result = result and not isinstance(instance.columns, pd.MultiIndex)
            result = result and all([col in instance.columns for col in IDX_COLS])
            result = result and len(instance.columns) > len(IDX_COLS)
            result = result and len(instance.columns) == len(set(instance.columns))

            datetypestr = str(instance["real_date"].dtype)
            _condA = datetypestr in ["datetime64[ns]", "datetime64[ms]"]
            _condB = datetypestr.startswith("datetime64[") and datetypestr.endswith("]")
            correct_date_type: bool = (
                _condA
                or _condB
                or isinstance(instance["real_date"].dtype, pd.DatetimeTZDtype)
                or instance.empty
            )
            result = result and correct_date_type

            # # check if the cid col is all str
            # # check if the xcat col is all str
        except:
            result: bool = False
        finally:
            return result


class QuantamentalDataFrame(pd.DataFrame, metaclass=QuantamentalDataFrameMeta):
    """
    ## Type extension of `pd.DataFrame` for Quantamental DataFrames.

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

    def __init__(self, df: Optional[pd.DataFrame] = None):
        if df is not None:
            if not (
                isinstance(df, pd.DataFrame) and isinstance(df, QuantamentalDataFrame)
            ):
                raise TypeError("Input must be a QuantamentalDataFrame (pd.DataFrame).")
        super().__init__(df)
