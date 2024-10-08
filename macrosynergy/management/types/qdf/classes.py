"""
Module hosting custom types and meta-classes for use across the package.
"""

from typing import Optional, Any, Mapping, Union, Callable, Sequence

import pandas as pd

from .methods import (
    change_column_format,
    reduce_df,
    update_df,
    apply_blacklist,
    qdf_to_wide_df,
    check_is_categorical,
    add_nan_series,
    qdf_from_timseries,
)
from .base import QuantamentalDataFrameBase


class QuantamentalDataFrame(QuantamentalDataFrameBase):
    """
    Type extension of `pd.DataFrame` for Quantamental DataFrames.

    Usage:
    >>> df: pd.DataFrame = load_data()
    >>> qdf = QuantamentalDataFrame(df)
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, categorical: bool = True):
        if df is not None:
            if not (
                isinstance(df, pd.DataFrame) and isinstance(df, QuantamentalDataFrame)
            ):
                raise TypeError("Input must be a QuantamentalDataFrame (pd.DataFrame).")

        if type(df) is QuantamentalDataFrame:
            self = df
            return

        super().__init__(df)
        self.InitializedAsCategorical = check_is_categorical(self)
        if categorical:
            self.to_categorical()

    def _inplaceoperation(
        self, method: Callable[..., Any], inplace: bool = False, *args, **kwargs
    ):
        result = method(*args, **kwargs)
        if inplace:
            self.__init__(result, categorical=self.InitializedAsCategorical)
            return self
        return QuantamentalDataFrame(result, categorical=self.InitializedAsCategorical)

    def is_categorical(self) -> bool:
        """
        Returns True if the QuantamentalDataFrame is categorical.
        """
        return check_is_categorical(self)

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

    def to_original_dtypes(self) -> "QuantamentalDataFrame":
        """
        Converts the QuantamentalDataFrame to its original dtypes (using the
        `InitialisedAsCategorical` attribute).
        """
        if self.InitializedAsCategorical:
            return self.to_categorical()
        return self.to_string_type()

    def reduce_df(
        self,
        cids: Optional[Sequence[str]] = None,
        xcats: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Mapping[str, Sequence[Union[str, pd.Timestamp]]] = None,
        out_all: bool = False,
        intersect: bool = False,
        inplace: bool = False,
    ) -> "QuantamentalDataFrame":
        """
        Filter DataFrame by `cids`, `xcats`, and `start` & `end` dates.
        """
        result = reduce_df(
            df=self,
            cids=cids,
            xcats=xcats,
            start=start,
            end=end,
            blacklist=blacklist,
            out_all=out_all,
            intersect=intersect,
        )

        if out_all:
            result, _xcats, _cids = result

        result = QuantamentalDataFrame(
            result, categorical=self.InitializedAsCategorical
        )

        if out_all:
            return result, _xcats, _cids

        return result

    def apply_blacklist(
        self,
        blacklist: Mapping[str, Sequence[Union[str, pd.Timestamp]]],
        inplace: bool = False,
    ):
        """
        Apply a blacklist to the QuantamentalDataFrame.
        """
        # func = apply_blacklist
        # return self._inplaceoperation(
        #     method=func,
        #     inplace=inplace,
        #     df=self,
        #     blacklist=blacklist,
        # )
        result = apply_blacklist(df=self, blacklist=blacklist)
        return QuantamentalDataFrame(result, categorical=self.InitializedAsCategorical)

    def update_df(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
    ) -> "QuantamentalDataFrame":
        """
        Update the QuantamentalDataFrame with a new DataFrame.
        """
        result = update_df(df=self, new_df=df)
        return QuantamentalDataFrame(result, categorical=self.InitializedAsCategorical)

    def add_nan_series(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        inplace: bool = False,
    ) -> "QuantamentalDataFrame":
        """
        Add a NaN series to the QuantamentalDataFrame.
        """
        return add_nan_series(df=self, ticker=ticker, start=start, end=end)

    def to_wide(
        self,
        value_column: str = "value",
    ) -> "QuantamentalDataFrame":
        """
        Pivot the QuantamentalDataFrame.
        """
        return qdf_to_wide_df(self, value_column=value_column)

    @classmethod
    def from_timeseries(
        cls,
        timeseries: pd.Series,
        ticker: str,
    ) -> "QuantamentalDataFrame":
        """
        Convert a timeseries DataFrame to a QuantamentalDataFrame.
        """
        return qdf_from_timseries(timeseries=timeseries, ticker=ticker)
