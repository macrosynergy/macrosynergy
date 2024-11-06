"""
Module hosting custom types and meta-classes for use across the package.
"""

from typing import Optional, Mapping, Union, Sequence, List, Tuple, Dict
import pandas as pd
from macrosynergy.compat import PD_2_0_OR_LATER
from .methods import (
    get_col_sort_order,
    change_column_format,
    _get_tickers_series,
    reduce_df,
    reduce_df_by_ticker,
    update_df,
    apply_blacklist,
    qdf_to_wide_df,
    check_is_categorical,
    add_nan_series,
    drop_nan_series,
    rename_xcats,
    qdf_from_timeseries,
    create_empty_categorical_qdf,
    concat_qdfs,
)
from .base import QuantamentalDataFrameBase


class QuantamentalDataFrame(QuantamentalDataFrameBase):
    """
    Type extension of `pd.DataFrame` for Quantamental DataFrames.  Usage: >>> df:
    pd.DataFrame = load_data() >>> qdf = QuantamentalDataFrame(df)
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        categorical: bool = True,
        _initialized_as_categorical: Optional[bool] = None,
    ):
        if df is not None:
            df_err = "Input must be a standardised Quantamental DataFrame."
            if not isinstance(df, pd.DataFrame):
                raise TypeError(df_err)

            if not isinstance(df, QuantamentalDataFrame):
                if "real_date" in df.columns:
                    df["real_date"] = pd.to_datetime(df["real_date"])

                if not isinstance(df, QuantamentalDataFrame):
                    raise ValueError(df_err)

        if type(df) is QuantamentalDataFrame:
            if _initialized_as_categorical is None:
                _initialized_as_categorical = df.InitializedAsCategorical
        else:
            if df.columns.tolist() != get_col_sort_order(df):
                df = df[get_col_sort_order(df)]

        if PD_2_0_OR_LATER:
            super().__init__(df)
        else:
            super().__init__(df.copy()) # pragma: no cover

        _check_cat = check_is_categorical(self)
        if _initialized_as_categorical is None:
            self.InitializedAsCategorical = _check_cat
        else:
            if not isinstance(_initialized_as_categorical, bool):
                raise TypeError("`_initialized_as_categorical` must be a boolean.")
            self.InitializedAsCategorical = _initialized_as_categorical

        if categorical:
            if not _check_cat:
                self.to_categorical()
        else:
            self.to_string_type()

    def is_categorical(self) -> bool:
        """
        Returns True if the QuantamentalDataFrame is categorical.

        Returns
        -------
        bool
            True if the QuantamentalDataFrame is categorical
        """
        return check_is_categorical(self)

    def to_categorical(self) -> "QuantamentalDataFrame":
        """
        Converts the QuantamentalDataFrame to a categorical DataFrame.

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame with categorical columns.
        """
        return change_column_format(self, cols=self._StrIndexCols, dtype="category")

    def to_string_type(self) -> "QuantamentalDataFrame":
        """
        Converts the QuantamentalDataFrame to a string DataFrame.

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame with string columns.

        """
        return change_column_format(self, cols=self._StrIndexCols, dtype="object")

    def to_original_dtypes(self) -> "QuantamentalDataFrame":
        """
        Converts the QuantamentalDataFrame to its original dtypes (using the
        `InitialisedAsCategorical` attribute).

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame with its original dtypes. The dtype is determined
            by the `InitialisedAsCategorical` attribute. The output dtype will be either
            'category' or 'object'.
        """

        if self.InitializedAsCategorical:
            return self.to_categorical()
        return self.to_string_type()

    def list_tickers(self) -> List[str]:
        """
        List all tickers in the QuantamentalDataFrame.

        Returns
        -------
        List[str]
            A list of all tickers in the QuantamentalDataFrame.
        """
        ltickers: List[str] = sorted(_get_tickers_series(self).unique())
        return ltickers

    def add_ticker_column(self) -> "QuantamentalDataFrame":
        """
        Add a ticker column to the QuantamentalDataFrame. `ticker` is a combination of
        `cid` and `xcat` columns. i.e. `ticker = cid_xcat`.

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame with a `ticker` column.
        """
        ticker_col = _get_tickers_series(self)
        self["ticker"] = ticker_col
        return self

    def drop_ticker_column(self) -> "QuantamentalDataFrame":
        """
        Drop the ticker column from the QuantamentalDataFrame.

        Raises
        ------
        ValueError
            If no `ticker` column is found in the DataFrame.

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame without the `ticker` column.
        """
        if "ticker" not in self.columns:
            raise ValueError("No `ticker` column found in the DataFrame.")
        return self.drop(columns=["ticker"])

    def reduce_df(
        self,
        cids: Optional[Sequence[str]] = None,
        xcats: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Dict[str, Sequence[Union[str, pd.Timestamp]]] = None,
        out_all: bool = False,
        intersect: bool = False,
    ) -> Union[
        "QuantamentalDataFrame", Tuple["QuantamentalDataFrame", List[str], List[str]]
    ]:
        """
        Filter DataFrame by `cids`, `xcats`, and `start` & `end` dates.

        Parameters
        ----------
        cids : Optional[Sequence[str]], optional
            List of CIDs to filter by, by default None
        xcats : Optional[Sequence[str]], optional
            List of XCATs to filter by, by default None
        start : Optional[str], optional
            Start date to filter by, by default None
        end : Optional[str], optional
            End date to filter by, by default None
        blacklist : Dict[str, Sequence[Union[str, pd.Timestamp]]], optional
            Blacklist to apply to the DataFrame, by default None
        out_all : bool, optional
            If True, return the filtered DataFrame, the filtered XCATs, and the filtered
            CIDs, by default False

        Returns
        -------
        QuantamentalDataFrame
            The filtered QuantamentalDataFrame. (if out_all=False, default)

        Tuple[QuantamentalDataFrame, List[str], List[str]]
            The filtered QuantamentalDataFrame, the filtered XCATs, and the filtered CIDs.
            (if out_all=True)
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
            result,
            categorical=self.is_categorical(),
            _initialized_as_categorical=self.InitializedAsCategorical,
        )

        if out_all:
            return result, _xcats, _cids

        return result

    def reduce_df_by_ticker(
        self,
        tickers: Sequence[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Mapping[str, Sequence[Union[str, pd.Timestamp]]] = None,
    ) -> "QuantamentalDataFrame":
        """
        Filter DataFrame by `ticker`, `start` & `end` dates.

        Parameters
        ----------
        tickers : Sequence[str]
            List of tickers to filter by
        start : Optional[str], optional
            Start date to filter by, by default None
        end : Optional[str], optional
            End date to filter by, by default None
        blacklist : Mapping[str, Sequence[Union[str, pd.Timestamp]]], optional
            Blacklist to apply to the DataFrame, by default None

        Returns
        -------
        QuantamentalDataFrame
            The filtered QuantamentalDataFrame.
        """
        result = reduce_df_by_ticker(
            df=self,
            tickers=tickers,
            start=start,
            end=end,
            blacklist=blacklist,
        )
        return QuantamentalDataFrame(
            result,
            categorical=self.is_categorical(),
            _initialized_as_categorical=self.InitializedAsCategorical,
        )

    def apply_blacklist(
        self,
        blacklist: Mapping[str, Sequence[Union[str, pd.Timestamp]]],
    ):
        """
        Apply a blacklist to the QuantamentalDataFrame.
        """
        result = apply_blacklist(df=self, blacklist=blacklist)
        return QuantamentalDataFrame(
            result,
            categorical=self.is_categorical(),
            _initialized_as_categorical=self.InitializedAsCategorical,
        )

    def update_df(
        self,
        df_add: pd.DataFrame,
        xcat_replace: bool = False,
    ) -> "QuantamentalDataFrame":
        """
        Update the QuantamentalDataFrame with a new DataFrame.

        Parameters
        ----------
        df_add : pd.DataFrame
            DataFrame to update the QuantamentalDataFrame with
        xcat_replace : bool, optional
            If True, replace the XCATs in the QuantamentalDataFrame with the XCATs in
            `df_add`, by default False

        Returns
        -------
        QuantamentalDataFrame
            The updated QuantamentalDataFrame.
        """
        result = update_df(df=self, df_add=df_add, xcat_replace=xcat_replace)
        return QuantamentalDataFrame(
            result,
            categorical=self.is_categorical(),
            _initialized_as_categorical=self.InitializedAsCategorical,
        )

    def add_nan_series(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "QuantamentalDataFrame":
        """
        Add a NaN series to the QuantamentalDataFrame.

        Parameters
        ----------
        ticker : str
            Ticker to add the NaN series to
        start : Optional[str], optional
            Start date of the NaN series, by default None
        end : Optional[str], optional
            End date of the NaN series, by default None

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame with the NaN series added.
        """
        result = add_nan_series(df=self, ticker=ticker, start=start, end=end)
        return QuantamentalDataFrame(
            result,
            categorical=self.is_categorical(),
            _initialized_as_categorical=self.InitializedAsCategorical,
        )

    def drop_nan_series(
        self,
        column: str = "value",
        raise_warning: bool = True,
    ) -> "QuantamentalDataFrame":
        """
        Drop NaN series from the QuantamentalDataFrame.

        Parameters
        ----------
        column : str, optional
            Column to check for NaN series, by default "value"
        raise_warning : bool, optional
            If True, raise a warning if NaN series are dropped, by default True

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame with NaN series dropped
        """
        result = drop_nan_series(df=self, column=column, raise_warning=raise_warning)
        return QuantamentalDataFrame(
            result,
            categorical=self.is_categorical(),
            _initialized_as_categorical=self.InitializedAsCategorical,
        )

    def rename_xcats(
        self,
        xcat_map: Optional[Mapping[str, str]] = None,
        select_xcats: Optional[List[str]] = None,
        postfix: Optional[str] = None,
        prefix: Optional[str] = None,
        name_all: Optional[str] = None,
        fmt_string: Optional[str] = None,
    ) -> "QuantamentalDataFrame":
        """
        Rename xcats in the QuantamentalDataFrame.

        Parameters
        ----------
        xcat_map : Optional[Mapping[str, str]], optional
            Mapping of xcats to rename, by default None
        select_xcats : Optional[List[str]], optional
            List of xcats to rename, by default None
        postfix : Optional[str], optional
            Postfix to add to the xcats, by default None
        prefix : Optional[str], optional
            Prefix to add to the xcats, by default None
        name_all : Optional[str], optional
            Name to rename all xcats to, by default None
        fmt_string : Optional[str], optional
            Format string to rename xcats, by default None

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame with the xcats renamed.
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
        return QuantamentalDataFrame(
            result,
            categorical=self.is_categorical(),
            _initialized_as_categorical=self.InitializedAsCategorical,
        )

    def to_wide(
        self,
        value_column: str = "value",
    ) -> "QuantamentalDataFrame":
        """
        Pivot the QuantamentalDataFrame.

        Parameters
        ----------
        value_column : str, optional
            Column to pivot, by default "value"

        Returns
        -------
        QuantamentalDataFrame
            The pivoted QuantamentalDataFrame, with each ticker as a column with the
            values of the `value_column` and the index as the `real_date`.
        """
        result = qdf_to_wide_df(self, value_column=value_column)
        if self.InitializedAsCategorical:
            return result

        result.columns = result.columns.astype("object")
        return result

    @classmethod
    def from_timeseries(
        cls,
        timeseries: pd.Series,
        ticker: str,
        metric: str = "value",
    ) -> "QuantamentalDataFrame":
        """
        Convert a timeseries DataFrame to a QuantamentalDataFrame.

        Parameters
        ----------
        timeseries : pd.Series
            Timeseries to convert to a QuantamentalDataFrame
        ticker : str
            Ticker to assign to the timeseries
        metric : str, optional
            Metric to assign to the timeseries, by default "value"

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame created from the timeseries.
        """
        return QuantamentalDataFrame(
            qdf_from_timeseries(
                timeseries=timeseries,
                ticker=ticker,
                metric=metric,
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
        Convert a long DataFrame to a QuantamentalDataFrame. This is useful when the
        DataFrame may contain only a `cid` or `xcat` column, or in cases where the `cid`
        and `xcat` columns are not named as such.

        Parameters
        ----------
        df : pd.DataFrame
            Long DataFrame to convert to a QuantamentalDataFrame
        real_date_column : str, optional
            Column name of the real date, by default "real_date"
        value_column : str, optional
            Column name of the value, by default "value"
        cid : Optional[str], optional
            `cid` to assign to the DataFrame, by default None. If not specified, the `cid`
            column must be present in the DataFrame.
        xcat : Optional[str], optional
            `xcat` to assign to the DataFrame, by default None
        ticker : Optional[str], optional
            Ticker to assign to the DataFrame, by default None
        categorical : bool, optional
            If True, convert the DataFrame to categorical, by default True

        Raises
        ------
        ValueError
            If the `real_date_column` or `value_column` are not found in the DataFrame,
            or if `ticker` is specified with `cid` or `xcat`.
        ValueError
            If the input DataFrame is empty.
        ValueError
            If the `cid` or `xcat` columns are not found in the DataFrame, and have not
            been specified in the function call.

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame created from the long DataFrame.
        """

        if real_date_column not in df.columns:
            raise ValueError(f"No `{real_date_column}` column found in the DataFrame.")
        else:
            df = df.rename(columns={real_date_column: "real_date"})
            real_date_column = "real_date"

        if value_column not in df.columns:
            raise ValueError(f"No `{value_column}` column found in the DataFrame.")

        if len(df) == 0:
            raise ValueError("Input DataFrame is empty.")

        if ticker is not None:
            if bool(cid) or bool(xcat):
                raise ValueError("Cannot specify `ticker` with `cid` or `xcat`.")
            cid, xcat = ticker.split("_", 1)

        errstr = "No `{}` column found in the DataFrame and `{}` not specified."
        for var_, name_ in zip([cid, xcat], ["cid", "xcat"]):
            if name_ not in df.columns and var_ is None:
                raise ValueError(errstr.format(name_, name_))

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

    @classmethod
    def from_qdf_list(
        cls,
        qdf_list: List["QuantamentalDataFrame"],
        categorical: bool = True,
    ) -> "QuantamentalDataFrame":
        """
        Concatenate a list of QuantamentalDataFrames into a single QuantamentalDataFrame.

        Parameters
        ----------
        qdf_list : List[QuantamentalDataFrame]
            List of QuantamentalDataFrames to concatenate
        categorical : bool, optional
            If True, convert the DataFrame to categorical, by default True

        Raises
        ------
        TypeError
            If any element in the list is not a QuantamentalDataFrame.
        ValueError
            If the input list is empty.

        Returns
        -------
        QuantamentalDataFrame
            The concatenated QuantamentalDataFrame.
        """
        if not all(isinstance(qdf, QuantamentalDataFrame) for qdf in qdf_list):
            raise TypeError("All elements in the list must be QuantamentalDataFrames.")
        if not qdf_list:
            raise ValueError("Input list is empty.")

        qdf_list = concat_qdfs(qdf_list)
        return QuantamentalDataFrame(qdf_list, categorical=categorical)

    @classmethod
    def from_wide(
        cls,
        df: pd.DataFrame,
        value_column: str = "value",
        categorical: bool = True,
    ) -> "QuantamentalDataFrame":
        """
        Convert a wide DataFrame to a QuantamentalDataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Wide DataFrame to convert to a QuantamentalDataFrame
        value_column : str, optional
            Column to pivot, by default "value"
        categorical : bool, optional
            If True, convert the DataFrame to categorical, by default True

        Raises
        ------
        TypeError
            If `df` is not a pandas DataFrame.
        ValueError
            If `df` does not have a datetime index.
        ValueError
            If all columns are not in the format 'cid_xcat'.

        Returns
        -------
        QuantamentalDataFrame
            The QuantamentalDataFrame created from the wide DataFrame.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("`df` must have a datetime index.")

        if not isinstance(value_column, str):
            raise TypeError("`value_column` must be a string.")

        if not all("_" in col for col in df.columns):
            raise ValueError("All columns must be in the format 'cid_xcat'.")

        df.index.name = "real_date"
        df.columns.name = None
        tickers = df.columns.tolist()
        qdfs_list: List[QuantamentalDataFrame] = []

        for tkr in tickers:
            cid, xcat = tkr.split("_", 1)
            qdf = df[[tkr]].reset_index().rename(columns={tkr: value_column})
            df = df.drop(columns=[tkr])
            qdf = QuantamentalDataFrame.from_long_df(
                df=qdf, cid=cid, xcat=xcat, value_column=value_column
            )
            qdfs_list.append(qdf)

        return QuantamentalDataFrame.from_qdf_list(qdfs_list, categorical=categorical)

    @classmethod
    def create_empty_df(
        cls,
        cid: Optional[str] = None,
        xcat: Optional[str] = None,
        ticker: Optional[str] = None,
        metrics: List[str] = ["value"],
        date_range: Optional[pd.DatetimeIndex] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        categorical: bool = True,
    ) -> "QuantamentalDataFrame":
        """
        Create an empty QuantamentalDataFrame.

        Parameters
        ----------
        cid : Optional[str], optional
            `cid` to assign to the DataFrame, by default None
        xcat : Optional[str], optional
            `xcat` to assign to the DataFrame, by default None
        ticker : Optional[str], optional
            Ticker to assign to the DataFrame, by default None. If specified, `cid` and
            `xcat` must not be specified.
        metrics : List[str], optional
            Metrics to assign to the DataFrame, by default ["value"]
        date_range : Optional[pd.DatetimeIndex], optional
            Date range to assign to the DataFrame, by default None. If not specified,
            `start` and `end` must be specified.
        start : Optional[str], optional
            Start date to assign to the DataFrame, by default None
        end : Optional[str], optional
            End date to assign to the DataFrame, by default None
        categorical : bool, optional
            If True, convert the DataFrame to categorical, by default True

        Returns
        -------
        QuantamentalDataFrame
            The empty QuantamentalDataFrame.
        """
        qdf = create_empty_categorical_qdf(
            cid=cid,
            xcat=xcat,
            ticker=ticker,
            metrics=metrics,
            date_range=date_range,
            start=start,
            end=end,
        )
        return QuantamentalDataFrame(qdf, categorical=categorical)
