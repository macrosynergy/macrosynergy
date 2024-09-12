"""
Module hosting custom types and meta-classes for use across the package.
"""

from typing import List, Optional, Any
import pandas as pd
import warnings

from .methods import change_column_format, reduce_df, update_df
from .base import QuantamentalDataFrameBase


class QuantamentalDataFrame(QuantamentalDataFrameBase):
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

    IndexCols: List[str] = ["real_date", "cid", "xcat"]
    _StrIndexCols: List[str] = ["cid", "xcat"]

    def __init__(self, df: Optional[pd.DataFrame] = None, categorical: bool = True):
        if df is not None:
            if not (
                isinstance(df, pd.DataFrame) and isinstance(df, QuantamentalDataFrame)
            ):
                raise TypeError("Input must be a QuantamentalDataFrame (pd.DataFrame).")
        super().__init__(df)

        if categorical:
            self.to_categorical()

    def is_categorical(self) -> bool:
        """
        Returns True if the QuantamentalDataFrame is categorical.
        """
        strcols = list(set(QuantamentalDataFrame.IndexCols) - {"real_date"})

        for col in strcols:
            if col in self.columns:
                if not self[col].dtype.name == "category":
                    return False

        return True

    def to_categorical(self) -> "QuantamentalDataFrame":
        """
        Converts the QuantamentalDataFrame to a categorical DataFrame.
        """
        return change_column_format(self, cols=self._StrIndexCols, dtype="category")

    def to_string_type(self) -> "QuantamentalDataFrame":
        """
        Converts the QuantamentalDataFrame to a string DataFrame.

        """
        return change_column_format(self, cols=self._StrIndexCols, dtype=str)
