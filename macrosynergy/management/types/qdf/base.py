"""
Module hosting custom types and meta-classes for use across the package.
"""

from typing import List, Optional, Any
import pandas as pd


class QuantamentalDataFrameMeta(type):
    """
    MetaClass to support type checks for `QuantamentalDataFrame`.
    """

    IndexCols: List[str] = ["real_date", "cid", "xcat"]
    _StrIndexCols: List[str] = ["cid", "xcat"]

    def __instancecheck__(cls, instance):
        IDX_COLS = QuantamentalDataFrameBase.IndexCols
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
                (_condA or _condB)
                or isinstance(instance["real_date"].dtype, pd.DatetimeTZDtype)
                or instance.empty
            )
            result = result and correct_date_type

        except:
            result: bool = False
        finally:
            return result


class QuantamentalDataFrameBase(pd.DataFrame, metaclass=QuantamentalDataFrameMeta):
    """
    ## Base class to to extend `pd.DataFrame` for Quantamental DataFrames.

    This class is a parent class to macrosynergy.types.QuantamentalDataFrame.
    """

    IndexCols: List[str] = ["real_date", "cid", "xcat"]
    _StrIndexCols: List[str] = ["cid", "xcat"]