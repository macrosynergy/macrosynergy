"""
Module hosting custom types and meta-classes for use across the package.
"""

from typing import Optional, Any, Mapping, Union, Callable, Sequence, List
import pandas as pd

from .methods import (
    get_col_sort_order,
    change_column_format,
    reduce_df,
    update_df,
    apply_blacklist,
    qdf_to_wide_df,
    check_is_categorical,
    add_nan_series,
    rename_xcats,
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
        df = df[get_col_sort_order(df)]
        super().__init__(df)
        self.InitializedAsCategorical = check_is_categorical(self)
        if categorical:
            self.to_categorical()
        else:
            self.to_string_type()

    def _inplaceoperation(
        self, method: Callable[..., Any], inplace: bool = False, *args, **kwargs
    ):
        result = method(*args, **kwargs)
        if inplace:
            self.__init__(result)
            return self
        return QuantamentalDataFrame(result)

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

        result = QuantamentalDataFrame(result)

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
        return QuantamentalDataFrame(result)

    def update_df(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
    ) -> "QuantamentalDataFrame":
        """
        Update the QuantamentalDataFrame with a new DataFrame.
        """
        result = update_df(df=self, new_df=df)
        return QuantamentalDataFrame(result)

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
        result = add_nan_series(df=self, ticker=ticker, start=start, end=end)
        return QuantamentalDataFrame(result)

    def rename_xcats(
        self,
        xcat_map: Optional[Mapping[str, str]] = None,
        select_xcats: Optional[List[str]] = None,
        postfix: Optional[str] = None,
        prefix: Optional[str] = None,
        name_all: Optional[str] = None,
        fmt_string: Optional[str] = None,
        inplace: bool = False,
    ) -> "QuantamentalDataFrame":
        """
        Rename xcats in the QuantamentalDataFrame.
        """
        result = rename_xcats(
            df=self,
            xcat_map=xcat_map,
            select_xcats=select_xcats,
            postfix=postfix,
            prefix=prefix,
            name_all=name_all,
            fmt_string=fmt_string,
        )
        return QuantamentalDataFrame(result)

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
        return QuantamentalDataFrame(
            qdf_from_timseries(
                timeseries=timeseries,
                ticker=ticker,
            )
        )

    @classmethod
    def from_long_df(
        cls,
        df: pd.DataFrame,
        real_date_column: str = "real_date",
        value_column: str = "value",
        cid: Optional[str] = None,
        xcat: Optional[str] = None,
        ticker: Optional[str] = None,
        categorical: bool = True,
    ) -> "QuantamentalDataFrame":
        """
        Convert a long DataFrame to a QuantamentalDataFrame. This is useful when the DataFrame
        may contain only a CID or XCAT column, or in cases where the CID and XCAT columns are
        not named as such.
        """
        # does the real_date column exist?
        if real_date_column not in df.columns:
            raise ValueError(f"No `{real_date_column}` column found in the DataFrame.")
        if value_column not in df.columns:
            raise ValueError(f"No `{value_column}` column found in the DataFrame.")
        if len(df) == 0:
            raise ValueError("Input DataFrame is empty.")

        errstr = "No `{}` column found in the DataFrame and `{}` not specified."
        for var_, name_ in zip([cid, xcat], ["cid", "xcat"]):
            if name_ not in df.columns and var_ is None:
                raise ValueError(errstr.format(name_, name_))

        if ticker is not None:
            if bool(cid) or bool(xcat):
                raise ValueError("Cannot specify `ticker` with `cid` or `xcat`.")
            cid, xcat = ticker.split("_", 1)

        new_df = df[[real_date_column, value_column]].copy()

        # if the cid col is there in the df copy it over as a categorical type
        if "cid" in df.columns:
            new_df["cid"] = df["cid"].astype("category")
        # if the xcat col is there in the df copy it over as a categorical type
        if "xcat" in df.columns:
            new_df["xcat"] = df["xcat"].astype("category")

        # if the cid was specfied then overwrite and initialise as a single cid category dtype
        if cid is not None:
            new_df["cid"] = pd.Categorical.from_codes(
                codes=[0] * len(new_df), categories=[cid]
            )

        # if the xcat was specfied then overwrite and initialise as a single xcat category dtype
        if xcat is not None:
            new_df["xcat"] = pd.Categorical.from_codes(
                codes=[0] * len(new_df), categories=[xcat]
            )

        return QuantamentalDataFrame(new_df, categorical=categorical)
