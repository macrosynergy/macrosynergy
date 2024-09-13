"""
Module hosting custom types and meta-classes for use across the package.
"""

from typing import List, Optional, Any, Dict, Mapping, Union, Callable, Sequence

import pandas as pd
import warnings

from .methods import (
    change_column_format,
    reduce_df,
    update_df,
    apply_blacklist,
    pivot_df,
)
from .base import QuantamentalDataFrameBase


class QuantamentalDataFrame(QuantamentalDataFrameBase):
    """
    Type extension of `pd.DataFrame` for Quantamental DataFrames.

    Usage:
    >>> df: pd.DataFrame = load_data()
    >>> qdf = QuantamentalDataFrame(df)
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

    def reduce_df(
        self,
        cids: Optional[Sequence[str]] = None,
        xcats: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Mapping[str, Sequence[Union[str, pd.Timestamp]]] = None,
        inplace: bool = False,
    ) -> "QuantamentalDataFrame":
        """
        Filter DataFrame by `cids`, `xcats`, `tickers`, and `start` & `end` dates.
        """
        func = reduce_df
        return self._inplaceoperation(
            method=func,
            inplace=inplace,
            df=self,
            cids=cids,
            xcats=xcats,
            start=start,
            end=end,
            blacklist=blacklist,
        )

    def apply_blacklist(
        self,
        blacklist: Mapping[str, Sequence[Union[str, pd.Timestamp]]],
        inplace: bool = False,
    ):
        """
        Apply a blacklist to the QuantamentalDataFrame.
        """
        func = apply_blacklist
        return self._inplaceoperation(
            method=func,
            inplace=inplace,
            df=self,
            blacklist=blacklist,
        )

    def update_df(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
    ) -> "QuantamentalDataFrame":
        """
        Update the QuantamentalDataFrame with a new DataFrame.
        """
        func = update_df
        return self._inplaceoperation(
            method=func,
            inplace=inplace,
            df=self,
            new_df=df,
        )

    def pivot_df(
        self,
        value_column: str = "value",
        inplace: bool = False,
    ) -> "QuantamentalDataFrame":
        """
        Pivot the QuantamentalDataFrame.
        """
        func = pivot_df
        return self._inplaceoperation(
            method=func,
            inplace=inplace,
            df=self,
            value_column=value_column,
        )
