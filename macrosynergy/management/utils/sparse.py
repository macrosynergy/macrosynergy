from typing import Dict, List, Any, Union, Optional, Callable, Tuple
from collections.abc import KeysView, ValuesView, ItemsView
from numbers import Number
import json
import warnings
import pandas as pd
import numpy as np
from macrosynergy.management.utils import (
    qdf_to_ticker_df,
    ticker_df_to_qdf,
    concat_single_metric_qdfs,
    get_cid,
    get_xcat,
    is_valid_iso_date,
)
from macrosynergy.management.types import QuantamentalDataFrame


SCORE_BY_OPTIONS = {"diff": "diff", "level": "value"}


class InformationStateChanges(object):
    """
    Class to hold information state changes for a set of tickers.
    InformationStateChanges show only data releases where there is an update in the
    indicator's value, grading or eop_lag. This offers a more compact representation of
    the data, where only releases which add information are retained.

    Initialize using the `from_qdf` class method to create an `InformationStateChanges`
    object from a `QuantamentalDataFrame`. The `calculate_score` method can be used to
    calculate scores for the information state changes.

    Example initialization:

    .. code-block:: python

        from macrosynergy.download import JPMaQSDownload
        from macrosynergy.management import InformationStateChanges

        tickers = ["USD_GDPPC_SA", "GBP_GDPPC_SA"]

        with JPMaQSDownload(client_id="cl_id", client_secret="cl_secret") as jpmaqs:
            df = jpmaqs.download(tickers=tickers, metrics="all")

        isc = InformationStateChanges.from_qdf(df)
        usd_gpdppc_isc = isc["USD_GDPPC_SA"]


    Parameters
    ----------
    min_period : pd.Timestamp
        The minimum period to include in the InformationStateChanges object.
    max_period : pd.Timestamp
        The maximum period to include in the InformationStateChanges object.


    .. note::
        Instantiate using the `from_qdf` or `from_isc_df` class methods.  This class is
        subscriptable, i.e. `isc["ticker"]` will return the DataFrame for the given ticker.
    """

    def __init__(
        self,
        min_period: pd.Timestamp = None,
        max_period: pd.Timestamp = None,
    ):
        self.isc_dict: Dict[str, pd.DataFrame] = dict()
        self.density_stats_df: pd.DataFrame = None
        self._min_period: pd.Timestamp = min_period
        self._max_period: pd.Timestamp = max_period
        self._qdf_as_categorical: bool = False

    def __getitem__(self, item) -> pd.DataFrame:
        return self.isc_dict[item]

    def __setitem__(self, key, value):
        self.isc_dict[key] = value

    def __str__(self):
        return f"InformationStateChanges object with tickers: {list(self.keys())}"

    def __repr__(self):
        return f"InformationStateChanges object with {len(self.keys())} tickers"

    def __add__(self, other):
        if not isinstance(other, InformationStateChanges):
            raise TypeError(
                "Unsupported operand type(s) for +: 'InformationStateChanges' and {}".format(
                    type(other)
                )
            )
        new_isc = InformationStateChanges(
            min_period=self._min_period, max_period=self._max_period
        )
        sameticks = sorted(set(self.keys()).intersection(set(other.keys())))
        if len(sameticks) > 0:
            raise ValueError(
                "Tickers overlap between the two "
                "InformationStateChanges, cannot overwrite data.\n"
                "Overlap: {}".format(sameticks)
            )
        new_isc.isc_dict = {**self.isc_dict, **other.isc_dict}
        return new_isc

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, InformationStateChanges):
            return False
        same_keys = set(self.keys()) == set(value.keys())
        if not same_keys:
            return False
        for k in self.keys():
            same_df = (self[k].sort_index()).equals(value[k].sort_index())
            if not same_df:
                return False

        assert same_keys and same_df
        return True

    def keys(self) -> KeysView:
        """
        A list of tickers in the InformationStateChanges object.

        Returns
        -------
        KeysView
            A view of the tickers in the InformationStateChanges object.
        """

        return self.isc_dict.keys()

    def values(self) -> ValuesView:
        """
        Extract the DataFrames from the InformationStateChanges object.

        Returns
        -------
        ValuesView
            A view of the DataFrames in the InformationStateChanges object.
        """

        return self.isc_dict.values()

    def items(self) -> ItemsView:
        """
        Iterate through (ticker, DataFrame) pairs in the InformationStateChanges object.

        Returns
        -------
        ItemsView
            A view of the (ticker, DataFrame) pairs in the InformationStateChanges
            object.
        """

        return self.isc_dict.items()

    @classmethod
    def from_qdf(
        cls: "InformationStateChanges",
        df: QuantamentalDataFrame,
        norm: bool = True,
        score_by: str = "diff",
        **kwargs,
    ) -> "InformationStateChanges":
        """
        Create an InformationStateChanges object from a QuantamentalDataFrame.

        Parameters
        ----------
        qdf : QuantamentalDataFrame
            The QuantamentalDataFrame to create the InformationStateChanges object from.
            This dataframe must contain a `value` column. Additionally, the `eop_lag`
            column is required to calculate the correct `eop` and `version` information.
            If not provided, the information state is assumed to be based on the value
            only. The `grading` column is optional and will be preserved in the output if
            provided.
        norm : bool
            If True, calculate the score for the information state changes.
        score_by : str
            The method to use for scoring. If "diff" (default), the score is calculated
            based on the difference between the information state changes. If "level", the
            score is calculated based on the value ('level') of the information state
            change.
        **kwargs : Any
            Additional keyword arguments to pass to the `calculate_score` Please refer
            to :func:`InformationStateChanges.calculate_score()` for more information.

        Returns
        -------
        InformationStateChanges
            An InformationStateChanges object.
        """
        if score_by not in SCORE_BY_OPTIONS.keys():
            raise ValueError(
                f"`score_by` must be one of {list(SCORE_BY_OPTIONS.keys())}"
            )

        isc: InformationStateChanges = cls(
            min_period=df["real_date"].min(),
            max_period=df["real_date"].max(),
        )

        df = QuantamentalDataFrame(df)
        isc._qdf_as_categorical = df.InitializedAsCategorical

        isc_dict, density_stats_df = create_delta_data(
            df, return_density_stats=True, score_by=score_by
        )

        isc.isc_dict = isc_dict
        isc.density_stats_df = density_stats_df

        if norm:
            isc.calculate_score(score_by=score_by, **kwargs)
        return isc

    @classmethod
    def from_isc_df(
        cls: "InformationStateChanges",
        df: pd.DataFrame,
        ticker: str,
        value_column: str = "value",
        eop_column: str = "eop",
        grading_column: str = "grading",
        real_date_column: str = "real_date",
        norm: bool = True,
        **kwargs,
    ) -> "InformationStateChanges":
        """
        Create an InformationStateChanges object from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to create the InformationStateChanges object from.
        ticker : str
            The ticker to create the InformationStateChanges object for.
        value_column : str
            The name of the column to use as the value.
        eop_column : str
            The name of the column to use as the end of period date.
        grading_column : str
            The name of the column to use as the grading.
        real_date_column : str
            The name of the column to use as the real date.
        norm : bool
            If True, calculate the score for the information state changes.
        **kwargs : Any
            Additional keyword arguments to pass to the `calculate_score` Please refer
            to `InformationStateChanges.calculate_score()` for more information.

        Returns
        -------
        InformationStateChanges
            An InformationStateChanges object.
        """

        isc_df: pd.DataFrame = _load_isc_from_df(
            df=df,
            ticker=ticker,
            value_column=value_column,
            eop_column=eop_column,
            grading_column=grading_column,
            real_date_column=real_date_column,
        )
        isc_dict = {ticker: isc_df}
        density_stats_df = _get_diff_density_stats_from_df(isc_df)
        minx = isc_df["value"].first_valid_index()
        maxx = isc_df["value"].last_valid_index()
        isc: InformationStateChanges = cls(min_period=minx, max_period=maxx)
        setattr(isc, "isc_dict", isc_dict)
        setattr(isc, "density_stats_df", density_stats_df)
        assert isinstance(isc, InformationStateChanges)
        assert len(isc.isc_dict) == 1

        if norm:
            isc.calculate_score(**kwargs)

        return isc

    def to_qdf(
        self,
        value_column: str = "value",
        postfix: str = None,
        metrics: List[str] = ["eop", "grading"],
        thresh: Union[Tuple[float, float], float] = None,
    ) -> pd.DataFrame:
        """
        Convert the InformationStateChanges object to a QuantamentalDataFrame.

        Parameters
        ----------
        value_column : str
            The name of the column to use as the value.
        postfix : str
            A postfix to append to the xcat column. Default is None.
        metrics : List[str]
            A list of metrics to include in the DataFrame. Default is ["eop", "grading"].
            Use `metrics=None` to disregard any non-value columns.
        thresh : Union[Tuple[float, float], float]
            A float or a tuple of two floats to winsorise the data to. Default is None.
            If a single float is provided, it is used for both lower and upper bounds,
            as `(-thresh, thresh)`. If a tuple is provided, it is used as
            `(thresh[0], thresh[1])`.
        Returns
        -------
        pd.DataFrame
            A DataFrame with the information state changes.
        """

        result = sparse_to_dense(
            isc=self.isc_dict,
            value_column=value_column,
            min_period=self._min_period,
            max_period=self._max_period,
            postfix=postfix,
            metrics=metrics,
            thresh=thresh,
        )

        return QuantamentalDataFrame(
            result,
            _initialized_as_categorical=self._qdf_as_categorical,
        ).to_original_dtypes()

    def to_dict(
        self, ticker: str
    ) -> Dict[
        str, Union[List[Tuple[str, float, str, float]], Tuple[str, str, str], str]
    ]:
        data = [
            (f"{index:%Y-%m-%d}", row.value, f"{row.eop:%Y-%m-%d}", row.grading)
            for index, row in self[ticker][["value", "eop", "grading"]].iterrows()
        ]

        columns = ("real_date", "value", "eop", "grading")
        return_dict = {
            "data": data,
            "columns": columns,
            "last_real_date": f"{self._max_period:%Y-%m-%d}",
            "ticker": ticker,
        }
        return return_dict

    def to_json(self, ticker: str) -> str:
        return json.dumps(self.to_dict(ticker))

    def get_releases(
        self,
        from_date: Optional[Union[pd.Timestamp, str]] = pd.Timestamp.today().normalize()
        - pd.offsets.BDay(1),
        to_date: Optional[Union[pd.Timestamp, str]] = pd.Timestamp.today().normalize(),
        excl_xcats: List[str] = None,
        latest_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get the latest releases for the InformationStateChanges object.

        Parameters
        ----------
        from_date : pd.Timestamp
            The start date of the period to get releases for.
        to_date : pd.Timestamp
            The end date of the period to get releases for.
        excl_xcats : List[str]
            A list of xcats to exclude from the releases.
        latest_only : bool
            If True, only the latest release for each ticker is returned. Default is
            True.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the latest releases for each ticker. If `latest_only` is
            False, all releases within the date range are returned.
        """

        if excl_xcats is not None:
            excl_xcat_err = "`excl_xcats` must be a list of strings"
            if not isinstance(excl_xcats, list):
                raise TypeError(excl_xcat_err)
            if not all(isinstance(x, str) for x in excl_xcats):
                raise TypeError(excl_xcat_err)
        else:
            excl_xcats = []

        if not isinstance(latest_only, bool):
            raise ValueError("`latest_only` must be a boolean")

        dt_err = "`{varname}` must be a `pd.Timestamp` or an ISO formatted date"
        for var_name in ["from_date", "to_date"]:
            if not isinstance(eval(var_name), (pd.Timestamp, str, type(None))):
                raise TypeError(dt_err.format(varname=var_name))
            if isinstance(eval(var_name), str):
                is_valid_iso_date(eval(var_name))

        if from_date is None:
            from_date = self._min_period
        elif isinstance(from_date, str):
            from_date = pd.Timestamp(from_date)
        if to_date is None:
            to_date = self._max_period
        elif isinstance(to_date, str):
            to_date = pd.Timestamp(to_date)

        if from_date > to_date:
            from_date, to_date = to_date, from_date
            warnings.warn("`from_date` is greater than `to_date`. Swapping the dates.")

        dfs_list = []
        for k, v in self.items():
            if get_xcat(k) in excl_xcats:
                continue
            s: pd.DataFrame = v.copy()
            s = s[(s.index >= from_date) & (s.index <= to_date)]
            s["ticker"] = k
            if latest_only and not s.empty:
                s = s.loc[[s.last_valid_index()]]

            dfs_list.append(s.reset_index())

        rel = (
            pd.concat(dfs_list, axis=0)
            .sort_values(by=["real_date", "eop", "ticker"])
            .rename(columns={"diff": "change"})
            .reset_index(drop=True)
        )

        if latest_only:
            return rel.set_index("ticker")
        else:
            return rel

    def temporal_aggregator_period(
        self,
        winsorise: int = 10,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Temporal aggregator over periods of changes in the information state.

        Parameters
        ----------
        winsorise : int
            The value to winsorise the data to. Default is 10.
        start : pd.Timestamp
            The start date of the period to aggregate.
        end : pd.Timestamp
            The end date of the period to aggregate.

        Returns
        -------
        QuantamentalDataFrame
            A QuantamentalDataFrame with the aggregated values.
        """

        return temporal_aggregator_period(
            isc=self.isc_dict,
            start=start or self._min_period,
            end=end or self._max_period,
            winsorise=winsorise,
        )

    def calculate_score(
        self,
        std: str = "std",
        halflife: int = None,
        min_periods: int = 10,
        isc_version: int = 0,
        iis: bool = False,
        custom_method: Optional[Callable] = None,
        custom_method_kwargs: Dict = {},
        volatility_forecast: bool = True,
        score_by: str = "diff",
    ):
        """
        Calculate score on sparse indicator for the InformationStateChanges object.

        Parameters
        ----------
        std : str
            The method to use for calculating the standard deviation. Supported methods
            are `std`, `abs`, `exp` and `exp_abs`. See the documentation for
            `StandardDeviationMethods` for more information.
        halflife : int
            The halflife of the exponential weighting. Only used with `exp` and
            `exp_abs` methods. Default is None.
        min_periods : int
            The minimum number of periods required for the calculation. Default is 10.
        isc_version : int
            The version of the information state changes to use. If set to 0 (default),
            only the first version is used. If set to any other positive integer, all
            versions are used.
        iis : bool
            if True (default) zn-scores are also calculated for the initial sample
            period defined by `min_periods`, on an in-sample basis, to avoid losing history.
        custom_method : Callable
            A custom method to use for calculating the standard deviation. Must have the
            signature `custom_method(s: pd.Series, **kwargs) -> pd.Series`.
        custom_method_kwargs : Dict
            Keyword arguments to pass to the custom method.
        volatility_forecast : bool
            If True (default), the volatility forecast is shifted one period forward to
            align with the information state changes.
        score_by : str
            The method to use for scoring. If "diff" (default), the score is calculated
            based on the difference between the information state changes. If "level", the
            score is calculated based on the value ('level') of the information state
            change.

        Returns
        -------
        InformationStateChanges
            The InformationStateChanges object with the scores
        """

        _calculate_score_on_sparse_indicator_for_class(
            cls=self,
            std=std,
            halflife=halflife,
            min_periods=min_periods,
            isc_version=isc_version,
            iis=iis,
            custom_method=custom_method,
            custom_method_kwargs=custom_method_kwargs,
            volatility_forecast=volatility_forecast,
            score_by=score_by,
        )
        return self


def _get_diff_data(
    diff_mask: pd.Series,
    val_series: pd.Series,
    eop_series: pd.Series,
    grading_series: pd.Series,
    fvi: pd.Timestamp,
) -> pd.DataFrame:
    """
    Generate a DataFrame containing versioned differences in values with end-of-period
    adjustments.

    This function processes input time series data and constructs a DataFrame that tracks
    changes in values, their versions, and grading adjustments based on an end-of-period
    lag. The function is intended to help analyze how values evolve over time, accounting
    for end-of-period (EOP) adjustments and releases, with each observation uniquely
    versioned to capture changes.

    Parameters
    ----------
    diff_mask : pd.Series
        A boolean mask indicating where the value has changed.
    val_series : pd.Series
        The value series.
    eop_series : pd.Series
        The end-of-period lag series.
    grading_series : pd.Series
        The grading series.
    fvi : pd.Timestamp
        The first valid index (fvi) for the ticker.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the diff data.
    """

    # get the first index as well
    dates = val_series.index[diff_mask].union([fvi])

    # create the diff dataframe
    df_temp: pd.DataFrame = pd.concat(
        (
            val_series.loc[dates].to_frame("value"),
            eop_series.loc[dates].to_frame("eop_lag"),
            grading_series.loc[dates].to_frame("grading"),
        ),
        axis=1,
        ignore_index=False,
    )

    df_temp["eop"] = df_temp.index - pd.to_timedelta(df_temp["eop_lag"], unit="D")
    df_temp["release"] = df_temp["eop_lag"].diff(periods=1) < 0

    df_temp = df_temp.sort_index().reset_index()
    df_temp["count"] = df_temp.index

    df_temp = pd.merge(
        left=df_temp,
        right=df_temp.groupby(["eop"], as_index=False)["count"].min(),
        on=["eop"],
        how="outer",
        suffixes=(None, "_min"),
    )
    df_temp["version"] = df_temp["count"] - df_temp["count_min"]
    df_temp["diff"] = df_temp["value"].diff(periods=1)
    df_temp = df_temp.set_index("real_date")[
        ["value", "eop", "version", "grading", "diff"]
    ]

    return df_temp


def _load_isc_from_df(
    df: pd.DataFrame,
    ticker: str,
    value_column: str = "value",
    eop_column: str = "eop",
    grading_column: str = "grading",
    real_date_column: str = "real_date",
) -> pd.DataFrame:
    """
    Load and prepare versioned data from a DataFrame for a given ticker, including
    end-of-period and grading adjustments.

    This function extracts specified columns from the input DataFrame and processes
    them to generate versioned differences in the values. It creates a structured
    DataFrame suitable for tracking value changes over time with associated end-of-period
    and grading information. Data validation is included to ensure column presence and
    correct grading ranges.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing data in wide format with required columns.
    ticker : str
        The identifier or ticker for the data, used to identify and validate the source.
    value_column : str, optional
        Column name for the primary values in the data, by default "value".
    eop_column : str, optional
        Column name for end-of-period data, by default "eop".
    grading_column : str, optional
        Column name for grading data, by default "grading".
    real_date_column : str, optional
        Column name for the date associated with each observation, by default "real_date".

    Returns
    -------
    pd.DataFrame
        A DataFrame with the versioned data for the specified ticker.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("`df` must be a DataFrame")
    if not isinstance(ticker, str):
        raise ValueError("`ticker` must be a string")
    if df.index.name == real_date_column:
        df = df.reset_index()

    all_cols_present = set(
        [value_column, eop_column, grading_column, real_date_column]
    ).issubset(df.columns)

    if not (all_cols_present):
        dx = {
            var_str: var_val
            for var_str, var_val in [
                ("value_column", value_column),
                ("eop_column", eop_column),
                ("grading_column", grading_column),
                ("real_date_column", real_date_column),
            ]
        }
        raise ValueError(
            "`df` must contain columns specified in `value_column`, `eop_column`,"
            " `grading_column` and `real_date_column`, or have an index named with the value"
            " of `real_date_column`.\n"
            f"Args: {dx}"
            f"Columns: {df.columns}"
        )
    df_temp = (
        df[[real_date_column, value_column, eop_column, grading_column]]
        .copy()
        .sort_index()
        .reset_index()
    )
    df_temp["count"] = df_temp.index
    df_temp = pd.merge(
        left=df_temp,
        right=df_temp.groupby([eop_column], as_index=False)["count"].min(),
        on=[eop_column],
        how="outer",
        suffixes=(None, "_min"),
    )

    # force all columns to be float
    df_temp[value_column] = df_temp[value_column].astype(dtype="float64")
    df_temp[grading_column] = df_temp[grading_column].astype(dtype="float64")

    df_temp["version"] = df_temp["count"] - df_temp["count_min"]
    df_temp["diff"] = df_temp[value_column].diff(periods=1)
    df_temp = df_temp.set_index(real_date_column)[
        [value_column, eop_column, "version", grading_column, "diff"]
    ]

    if any(df_temp[grading_column] > 3):
        df_temp[grading_column] = df_temp[grading_column].astype(dtype="float64") / 10.0
    if any(1 > df_temp[grading_column]) or any(df_temp[grading_column] > 3):
        raise ValueError(
            "Grading values must be between 1.0 and 3.0 (incl.),"
            " or integers between 10 and 30"
        )

    return df_temp


def _get_diff_density_stats_from_df(
    isc_df: pd.DataFrame,
) -> Dict[str, Union[float, str]]:
    """
    Get the density stats for a given ticker from a DataFrame with the changes in the
    information state.

    Parameters
    ----------
    isc_df : pd.DataFrame
        A DataFrame with the changes in the information state.

    Returns
    -------
    Dict[str, Union[float, str]
        A dictionary with the density stats.
    """

    diff_mask = isc_df["diff"].abs() > 1e-12
    diff_density = 100 * diff_mask.sum() / (~isc_df["value"].isna()).sum()
    fvi, lvi = isc_df["value"].first_valid_index(), isc_df["value"].last_valid_index()
    dtrange_str = f"{fvi.strftime('%Y-%m-%d')} : {lvi.strftime('%Y-%m-%d')}"
    return {"diff_density": diff_density, "date_range": dtrange_str}


def _get_diff_density_stats(
    diff_mask: pd.Series, val_series: pd.Series, fvi: pd.Timestamp, lvi: pd.Timestamp
) -> Dict[str, Union[float, str]]:
    """
    Get the density stats for a given ticker from a boolean mask indicating where the
    value has changed and the value series.

    Parameters
    ----------
    diff_mask : pd.Series
        A boolean mask indicating where the value has changed.
    val_series : pd.Series
        The value series.

    Returns
    -------
    Dict[str, Union[float, str]
        A dictionary with the density stats.
    """

    diff_density: Number = 100 * diff_mask.sum() / (~val_series.isna()).sum()
    dtrange_str = f"{fvi.strftime('%Y-%m-%d')} : {lvi.strftime('%Y-%m-%d')}"
    return {"diff_density": diff_density, "date_range": dtrange_str}


def create_delta_data(
    df: QuantamentalDataFrame,
    return_density_stats: bool = False,
    score_by: str = "diff",
) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Creates a dictionary of dataframes with the changes in the information state for
    each ticker in the QuantamentalDataFrame. Optionally, returns a DataFrame with the
    statistics for change frequency, density and date range for each ticker.

    Parameters
    ----------
    df : QuantamentalDataFrame
        The QuantamentalDataFrame to calculate the changes for.
    return_density_stats : bool
        If True, returns a DataFrame with the density stats for each ticker.
    score_by : str
        The method to use for scoring. If "diff" (default), the score is calculated based

    Returns
    -------
    Union[Dict[str, pd.DataFrame], pd.DataFrame]
        A dictionary of DataFrames with the changes in the information state for each
        ticker.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise ValueError("`df` must be a QuantamentalDataFrame")
    if not isinstance(return_density_stats, bool):
        raise ValueError("`return_density_stats` must be a boolean")
    if "value" not in df.columns:
        raise ValueError("`df` must contain a `value` column")
    if "eop_lag" not in df.columns:
        warnings.warn(
            "`df` does not contain an `eop_lag` column. Differences calculated will not be "
            " based on end-of-period adjustments."
        )
        df["eop_lag"] = np.nan
    if "grading" not in df.columns:
        df["grading"] = np.nan

    df = QuantamentalDataFrame(df)
    values_df = qdf_to_ticker_df(df, value_column="value")
    eop_df = qdf_to_ticker_df(df, value_column="eop_lag")
    grading_df = qdf_to_ticker_df(df, value_column="grading")
    assert set(values_df.columns) == set(eop_df.columns) == set(grading_df.columns)
    all_tickers: List[str] = values_df.columns.tolist()

    # create dicts to store the dataframes and density stats
    isc_dict: Dict[str, Any] = {}
    # density_stats: Dict[str, Dict[str, Any]] = {}

    if score_by == "diff":
        diff_mask = values_df.diff(axis=0).abs() > 1e-12

        # get the first valid index for each column
        fvi_series: pd.Series = values_df.apply(lambda x: x.first_valid_index())
    elif score_by == "level":
        diff_mask = values_df.abs() > 1e-12

        # get the first valid index for each column
        labels, values = zip(
            *[
                (col, values_df.loc[diff_mask[col], col].first_valid_index())
                for col in values_df
            ]
        )
        fvi_series: pd.Series = pd.Series(values, index=labels)
    else:
        raise ValueError(f"Invalid value for `score_by`: {score_by}")

    lvi_series: pd.Series = values_df.apply(lambda x: x.last_valid_index())
    density_stats: Dict[str, Dict[str, Union[float, str]]] = {}

    for ticker in all_tickers:
        isc_dict[ticker] = _get_diff_data(
            diff_mask=diff_mask[ticker],
            val_series=values_df[ticker],
            eop_series=eop_df[ticker],
            grading_series=grading_df[ticker],
            fvi=fvi_series[ticker],
        )
        density_stats[ticker] = _get_diff_density_stats(
            diff_mask=diff_mask[ticker],
            val_series=values_df[ticker],
            fvi=fvi_series[ticker],
            lvi=lvi_series[ticker],
        )

    # flatten the density stats
    _dstats_flat = [
        (k, v["diff_density"], v["date_range"]) for k, v in density_stats.items()
    ]

    density_stats_df = (
        pd.DataFrame(_dstats_flat, columns=["ticker", "changes_density", "date_range"])
        .sort_values(by="changes_density", ascending=False)
        .reset_index(drop=True)
    )
    if return_density_stats:
        return isc_dict, density_stats_df
    else:
        return isc_dict


class SubscriptableMeta(type):
    """
    Convenience metaclass to allow subscripting of methods on a class.
    """

    def __getitem__(cls, item):
        if hasattr(cls, item) and callable(getattr(cls, item)):
            return getattr(cls, item)
        else:
            raise KeyError(f"{item} is not a valid method name")


class VolatilityEstimationMethods(metaclass=SubscriptableMeta):
    """
    Class to hold methods for calculating standard deviations. Each method must comply
    to the following signature: `func(s: pd.Series, **kwargs) -> pd.Series`  Currently
    supported methods are:  - `std`: Standard deviation - `abs`: Mean absolute deviation
    - `exp`: Exponentially weighted standard deviation - `exp_abs`: Exponentially
    weighted mean absolute deviation
    """

    @staticmethod
    def std(s: pd.Series, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the expanding standard deviation of a Series.

        Parameters
        ----------
        s : pd.Series
            The Series to calculate the standard deviation for.
        min_periods : int
            The minimum number of periods required for the calculation.

        Returns
        -------
        pd.Series
            The standard deviation of the Series.
        """

        return s.expanding(min_periods=min_periods).std()

    @staticmethod
    def abs(s: pd.Series, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the expanding mean absolute deviation of a Series.

        Parameters
        ----------
        s : pd.Series
            The Series to calculate the absolute standard deviation for.
        min_periods : int
            The minimum number of periods required for the calculation.

        Returns
        -------
        pd.Series
            The absolute standard deviation of the Series.
        """

        mean = s.expanding(min_periods=min_periods).mean()
        return (s - mean.bfill()).abs().expanding(min_periods=min_periods).mean()

    @staticmethod
    def exp(s: pd.Series, halflife: int, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the exponentially weighted standard deviation of a Series.

        Parameters
        ----------
        s : pd.Series
            The Series to calculate the exponentially weighted standard deviation for.
        halflife : int
            The halflife of the exponential weighting.
        min_periods : int
            The minimum number of periods required for the calculation.

        Returns
        -------
        pd.Series
            The exponentially weighted standard deviation of the Series.
        """

        return s.ewm(halflife=halflife, min_periods=min_periods).std()

    @staticmethod
    def exp_abs(s: pd.Series, halflife: int, min_periods: int, **kwargs) -> pd.Series:
        """
        Calculate the exponentially weighted mean absolute deviation of a Series.

        Parameters
        ----------
        s : pd.Series
            The Series to calculate the exponentially weighted absolute standard
            deviation for.
        halflife : int
            The halflife of the exponential weighting.
        min_periods : int
            The minimum number of periods required for the calculation.

        Returns
        -------
        pd.Series
            The exponentially weighted absolute standard deviation of the Series.
        """

        mean = s.ewm(halflife=halflife, min_periods=min_periods).mean()
        sd = (
            (s - mean.bfill())
            .abs()
            .ewm(halflife=halflife, min_periods=min_periods)
            .mean()
        )
        return sd


CALC_SCORE_CUSTOM_METHOD_ERR_MSG = (
    "Method {std} not supported. "
    f"Supported methods are: {dir(VolatilityEstimationMethods)}. \n"
    "Alternatively, provide a custom method with signature "
    "`custom_method(s: pd.Series, **kwargs) -> pd.Series` "
    "using the `custom_method` and `custom_method_kwargs` arguments."
)


def calculate_score_on_sparse_indicator(
    isc: Dict[str, pd.DataFrame],
    std: str = "std",
    halflife: int = None,
    min_periods: int = 10,
    isc_version: int = 0,
    iis: bool = False,
    custom_method: Optional[Callable] = None,
    custom_method_kwargs: Dict = {},
    volatility_forecast: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate score on sparse indicator

    Parameters
    ----------
    isc : Dict[str, pd.DataFrame]
        A dictionary of DataFrames with the changes in the information state for each
        ticker.
    std : str
        The method to use for calculating the standard deviation. Supported methods are
        `std`, `abs`, `exp` and `exp_abs`. See the documentation for
        `VolatilityEstimationMethods` for more information.
    halflife : int
        The halflife of the exponential weighting. Only used with `exp` and `exp_abs`
        methods. Default is None.
    min_periods : int
        The minimum number of periods required for the calculation. Default is 10.
    isc_version : int
        The version of the information state changes to use. If set to 0 (default), only
        the first version is used. If set to any other positive integer, all versions are
        used.
    iis : bool
        if True (default) zn-scores are also calculated for the initial sample period
        defined by `min_periods`, on an in-sample basis, to avoid losing history.
    custom_method : Callable
        A custom method to use for calculating the standard deviation. Must have the
        signature `custom_method(s: pd.Series, **kwargs) -> pd.Series`.
    custom_method_kwargs : Dict
        Keyword arguments to pass to the custom method.
    volatility_forecast : bool
        If True (default), the volatility forecast is shifted one period forward to
        align with the information state changes.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary of DataFrames with the changes in the information state for each
        ticker.
    """

    # Operations on a per key in data dictionary

    curr_method: Callable[[pd.Series, Optional[Dict[str, Any]]], pd.Series]
    if custom_method is not None:
        if not callable(custom_method):
            raise TypeError("`custom_method` must be a callable")
        if not isinstance(custom_method_kwargs, dict):
            raise TypeError("`custom_method_kwargs` must be a dictionary")
        curr_method = custom_method
    else:
        if not hasattr(VolatilityEstimationMethods, std):
            raise ValueError(CALC_SCORE_CUSTOM_METHOD_ERR_MSG.format(std=std))
        # curr_method = getattr(StandardDeviationMethod, std)
        curr_method = VolatilityEstimationMethods[std]

    method_kwargs: Dict[str, Any] = dict(
        min_periods=min_periods, halflife=halflife, **custom_method_kwargs
    )
    # if not 0, then use all versions
    for key, v in isc.items():
        mask_rel = (v["version"] == 0) if isc_version == 0 else (v["version"] >= 0)
        s = v.loc[mask_rel, "diff"]
        # TODO exponential weights (requires knowledge of frequency...)

        result: pd.Series = curr_method(s, **method_kwargs)

        columns = [kk for kk in v.columns if kk != "std"]
        v = pd.merge(
            left=v[columns],
            right=result.to_frame("std").shift(periods=int(volatility_forecast)),
            how="left",
            left_index=True,
            right_index=True,
        )
        v["std"] = v["std"].ffill()
        if iis:
            v["std"] = v["std"].bfill()
        v["zscore"] = v["diff"] / v["std"]

        isc[key] = v

    return isc


def _infer_frequency_timeseries(eop_lag_series: pd.Series) -> Optional[str]:
    """
    Infer the frequency of a time series based on the pattern of eop_lag values,
    considering resets to 0 to detect period ends.

    Parameters
    ----------
    eop_lag_series : pd.Series
        A Series of eop_lag values.

    Returns
    -------
    Optional[str]
        The inferred frequency of the time series. One of "D", "W", "M", "Q" or "A".
    """

    # Identify resets to 0 which indicate the end of a period
    reset_indices = eop_lag_series[eop_lag_series == 0].index
    if len(reset_indices) < 2:
        return "D"  # Default to daily if insufficient data

    # Calculate periods between resets
    periods = reset_indices.to_series().diff().dropna().dt.days

    # Determine the most common period length
    most_common_period = periods.mode().values[0]

    # Define ranges for different frequencies
    freqs = {"D": 1, "W": 7, "M": 30, "Q": 91, "A": 365}
    # 10% tolerance for frequency ranges
    for freq, freq_range in freqs.items():
        rng = (1 - 0.1) * freq_range, (1 + 0.1) * freq_range
        if rng[0] <= most_common_period <= rng[1]:
            return freq

    return "D"  # Default to daily if no match is found


def infer_frequency(df: QuantamentalDataFrame) -> pd.Series:
    """
    Infer the frequency of a QuantamentalDataFrame based on the most common eop_lag
    values.

    Parameters
    ----------
    df : QuantamentalDataFrame
        The QuantamentalDataFrame to infer the frequency for.

    Returns
    -------
    pd.Series
        A Series with the inferred frequency for each ticker in the
        QuantamentalDataFrame.
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("`df` must be a QuantamentalDataFrame")
    if not "eop_lag" in df.columns:
        raise ValueError("`df` must contain an `eop_lag` column")

    ticker_df = qdf_to_ticker_df(df, value_column="eop_lag")
    freq_dict = {
        ticker: _infer_frequency_timeseries(ticker_df[ticker])
        for ticker in ticker_df.columns
    }
    return pd.Series(freq_dict)


def weight_from_frequency(freq: str, base: float = 252):
    """Weight from frequency"""
    # TODO apply on multiple tickers
    freq_map = {"D": 1, "W": 5, "M": 21, "Q": 93, "A": 252}
    assert freq in freq_map, f"Frequency {freq} not supported"
    return freq_map[freq] / base


def _remove_insignificant_values(
    df: pd.DataFrame, threshold: float = 1e-12
) -> pd.DataFrame:
    """
    Convenience function to remove insignificant values from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to remove insignificant values from.
    threshold : float
        The threshold below which values are considered insignificant.

    Returns
    -------
    pd.DataFrame
        The DataFrame with insignificant values removed.
    """

    return df / (df.cumsum(axis=0).abs() > threshold).astype(int)


def _isc_dict_to_frames(
    isc: Dict[str, pd.DataFrame], metric: str = "value"
) -> List[pd.DataFrame]:
    """
    Convert a dictionary of DataFrames to a list of DataFrames with a specific metric.
    """
    frames = []
    for k, v in isc.items():
        assert isinstance(v, pd.DataFrame)
        assert metric in v.columns
        _fr = v[metric].to_frame(k)
        if _fr.empty:
            warnings.warn(f"Empty frame for {k}")
            continue

        frames.append(_fr)

    return frames


def _get_metric_df_from_isc(
    isc: Dict[str, pd.DataFrame],
    metric: str,
    date_range: pd.DatetimeIndex,
    fill: Union[str, Number] = 0,
) -> pd.DataFrame:
    """
    Get a DataFrame with a specific metric from a dictionary of DataFrames.

    Parameters
    ----------
    isc : Dict[str, pd.DataFrame]
        A dictionary of DataFrames with the changes in the information state for each
        ticker.
    metric : str
        The name of the metric to extract.
    date_range : pd.DatetimeIndex
        The date range to reindex the DataFrame to.
    fill : Union[str, Number]
        The value to fill NaNs with. If 'ffill', forward fill NaNs. Default is 0.
    """

    fill_err: str = "`fill` must be a number to replace NaNs or 'ffill' to forward fill"
    if not isinstance(fill, (str, Number)):
        raise TypeError(fill_err)
    if isinstance(fill, str):
        if not fill == "ffill":
            raise ValueError(fill_err)

    df_frames: List[pd.DataFrame] = _isc_dict_to_frames(isc, metric=metric)
    tdf: pd.DataFrame = pd.concat(df_frames, axis=1).reindex(date_range)
    if fill == "ffill":
        tdf = tdf.ffill()
    else:
        tdf = tdf.fillna(fill)
    tdf.columns.name, tdf.index.name = "ticker", "real_date"
    return tdf


def sparse_to_dense(
    isc: Dict[str, pd.DataFrame],
    value_column: str,
    min_period: pd.Timestamp,
    max_period: pd.Timestamp,
    postfix: str = None,
    metrics: List[str] = ["eop", "grading"],
    thresh: Union[Tuple[float, float], float] = None,
) -> pd.DataFrame:
    """
    Convert a dictionary of DataFrames with changes in the information state to a dense
    DataFrame (QuantamentalDataFrame).

    Parameters
    ----------
    isc : Dict[str, pd.DataFrame]
        A dictionary of DataFrames with the changes in the information state for each
        ticker.
    value_column : str
        The name of the column to use as the value.
    min_period : pd.Timestamp
        The minimum period to include in the DataFrame.
    max_period : pd.Timestamp
        The maximum period to include in the DataFrame.
    postfix : str
        A postfix to append to the xcat column. Default is None.
    metrics : Optional[List[str]]
        A list of metrics to include in the DataFrame. Default is ["eop", "grading"].
        Use `metrics=None` to disregard any non-value columns.
    thresh : Union[Tuple[float, float], float]
        A float or a tuple of two floats to winsorise the data to. Default is None.
        If a single float is provided, it is used for both lower and upper bounds,
        as `(-thresh, thresh)`. If a tuple is provided, it is used as
        `(thresh[0], thresh[1])`.
    Returns
    -------
    pd.DataFrame
        A DataFrame with the dense information state.
    """

    # TODO store real_date min and max in object...
    # Note: default behaviour includes both start and end dates
    dtrange = pd.date_range(
        start=min_period,
        end=max_period,
        freq="B",
    )

    tdf = _get_metric_df_from_isc(isc=isc, metric=value_column, date_range=dtrange)
    tdf = _remove_insignificant_values(tdf, threshold=1e-12)

    wins_lower, wins_upper = None, None
    if thresh is not None:
        if isinstance(thresh, tuple):
            if (len(thresh) != 2) or not all(isinstance(x, Number) for x in thresh):
                raise ValueError(
                    "If `thresh` is a tuple, it must contain two numeric values."
                )
            wins_lower, wins_upper = thresh
        elif isinstance(thresh, Number):
            wins_lower, wins_upper = -thresh, thresh
        else:
            raise ValueError("`thresh` must be a number or a tuple of two numbers.")

        tdf = tdf.clip(lower=wins_lower, upper=wins_upper)

    sm_qdfs: List[QuantamentalDataFrame] = [ticker_df_to_qdf(tdf)]
    if metrics is None:
        metrics = []
    for metric_name in metrics:
        wdf = _get_metric_df_from_isc(
            isc=isc, metric=metric_name, date_range=dtrange, fill="ffill"
        )
        if wdf.empty or wdf.isna().all().all():
            dfs = [
                QuantamentalDataFrame.create_empty_df(
                    ticker=tickerx,
                    date_range=dtrange,
                    metrics=[metric_name],
                )
                for tickerx in wdf.columns
            ]
            m_qdf = QuantamentalDataFrame.from_qdf_list(dfs)
        else:
            m_qdf = ticker_df_to_qdf(wdf, metric=metric_name)

        sm_qdfs.append(m_qdf)

    qdf: QuantamentalDataFrame = QuantamentalDataFrame.from_qdf_list(sm_qdfs)
    if "eop" in metrics:
        qdf["eop_lag"] = (qdf["real_date"] - qdf["eop"]).dt.days
        qdf = QuantamentalDataFrame(qdf)

    if postfix:
        qdf.rename_xcats(postfix=postfix)

    return qdf


def temporal_aggregator_exponential(
    df: QuantamentalDataFrame,
    halflife: int = 5,
    winsorise: float = None,
) -> pd.DataFrame:
    """
    Temporal aggregator using exponential moving average.

    Parameters
    ----------
    df : QuantamentalDataFrame
        The QuantamentalDataFrame to aggregate.
    halflife : int
        The halflife of the exponential moving average.
    winsorise : float
        The value to winsorise the data to. Default is None.

    Returns
    -------
    QuantamentalDataFrame
        A QuantamentalDataFrame with the aggregated values.
    """

    tdf: pd.DataFrame = qdf_to_ticker_df(df)
    if winsorise:
        tdf = tdf.clip(lower=-winsorise, upper=winsorise)
    # Exponential moving average weights
    tdf = tdf.ewm(halflife=halflife).mean()
    qdf: QuantamentalDataFrame = ticker_df_to_qdf(tdf)
    qdf["xcat"] += f"EWM{halflife:d}D"
    return qdf


def temporal_aggregator_mean(
    df: QuantamentalDataFrame,
    window: int = 21,
    winsorise: float = None,
) -> pd.DataFrame:
    """
    Temporal aggregator using a rolling mean.

    Parameters
    ----------
    df : QuantamentalDataFrame
        The QuantamentalDataFrame to aggregate.
    window : int
        The window size for the rolling mean.
    winsorise : float
        The value to winsorise the data to. Default is None.

    Returns
    -------
    QuantamentalDataFrame
        A QuantamentalDataFrame with the aggregated values.
    """

    tdf: pd.DataFrame = qdf_to_ticker_df(df)
    if winsorise:
        tdf = tdf.clip(lower=-winsorise, upper=winsorise)

    tdf = tdf.rolling(window=window).mean()
    qdf: QuantamentalDataFrame = ticker_df_to_qdf(tdf)
    qdf["xcat"] += f"MA{window:d}D"
    return qdf


def temporal_aggregator_period(
    isc: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    winsorise: int = 10,
    postfix: str = "_NCSUM",
) -> pd.DataFrame:
    """
    Temporal aggregator over periods of changes in the information state.

    Parameters
    ----------
    isc : Dict[str, pd.DataFrame]
        A dictionary of DataFrames with the changes in the information state for each
        ticker.
    start : pd.Timestamp
        The start date of the period to aggregate.
    end : pd.Timestamp
        The end date of the period to aggregate.
    winsorise : int
        The value to winsorise the data to. Default is 10.
    postfix : str
        A postfix to append to the xcat column. Default is "_NCSUM".

    Returns
    -------
    QuantamentalDataFrame
        A QuantamentalDataFrame with the aggregated values.
    """

    # Note: default behaviour includes both start and end dates
    dt_range = pd.date_range(
        start=start,
        end=end,
        freq="B",
    )
    tdf: pd.DataFrame = _get_metric_df_from_isc(
        isc=isc,
        metric="zscore_norm_squared",
        date_range=dt_range,
        fill=0,
    )
    # Winsorise and remove insignificant values
    tdf = _remove_insignificant_values(tdf, threshold=1e-12)
    tdf = tdf.clip(lower=-winsorise, upper=winsorise)

    # Map out the eop dates
    p_eop: pd.DataFrame = _get_metric_df_from_isc(
        isc=isc,
        metric="eop",
        date_range=dt_range,
        fill="ffill",
    )

    qdf: QuantamentalDataFrame = concat_single_metric_qdfs(
        [
            ticker_df_to_qdf(tdf),
            ticker_df_to_qdf(p_eop, metric="eop"),
        ]
    )

    # group by cid, xcat, eop
    qdf["value"] = (
        qdf.groupby(["cid", "xcat", "eop"])["value"].cumsum().reset_index(drop=True)
    )
    if postfix:
        qdf["xcat"] += postfix
    return qdf


def _calculate_score_on_sparse_indicator_for_class(
    cls: "InformationStateChanges",
    std: str = "std",
    halflife: int = None,
    min_periods: int = 10,
    isc_version: int = 0,
    iis: bool = False,
    custom_method: Optional[Callable] = None,
    custom_method_kwargs: Dict = {},
    volatility_forecast: bool = True,
    score_by: str = "diff",
    threshold: float = 1e-12,
):
    """
    Calculate score on sparse indicator for a class. Effectively a re-implementation of
    the function `calculate_score_on_sparse_indicator` that specifically operates on an
    `InformationStateChanges` object.
    """

    assert isinstance(
        cls, InformationStateChanges
    ), "`cls` must be an `InformationStateChanges` object"
    assert hasattr(cls, "isc_dict") and isinstance(
        cls.isc_dict, dict
    ), "`InformationStateChanges` object not initialized"

    if score_by not in SCORE_BY_OPTIONS.keys():
        raise ValueError(f"`score_by` must be one of {list(SCORE_BY_OPTIONS.keys())}")

    score_by_column = SCORE_BY_OPTIONS[score_by]

    for key, v in cls.isc_dict.items():
        if not score_by_column in v.columns:
            raise ValueError(f"Column `{score_by_column}` not in for ticker {key}")

    curr_method: Callable[[pd.Series, Optional[Dict[str, Any]]], pd.Series]
    if custom_method is not None:
        if not callable(custom_method):
            raise TypeError("`custom_method` must be a callable")
        if not isinstance(custom_method_kwargs, dict):
            raise TypeError("`custom_method_kwargs` must be a dictionary")
        curr_method = custom_method
    else:
        if not hasattr(VolatilityEstimationMethods, std):
            raise ValueError(CALC_SCORE_CUSTOM_METHOD_ERR_MSG.format(std=std))
        curr_method = VolatilityEstimationMethods[std]

    method_kwargs: Dict[str, Any] = dict(
        min_periods=min_periods, halflife=halflife, **custom_method_kwargs
    )
    for key, v in cls.isc_dict.items():
        mask_rel = (v["version"] == 0) if isc_version == 0 else (v["version"] >= 0)
        s = v.loc[mask_rel, score_by_column]
        result: pd.Series = curr_method(s, **method_kwargs)
        columns = [kk for kk in v.columns if kk != "std"]
        v = pd.merge(
            left=v[columns],
            right=result.to_frame("std").shift(periods=int(volatility_forecast)),
            how="left",
            left_index=True,
            right_index=True,
        )
        v["std"] = v["std"].ffill()
        if iis:
            v["std"] = v["std"].bfill()
        v["zscore"] = v[score_by_column] / v["std"]

        cls.isc_dict[key] = v


if __name__ == "__main__":
    # df = pd.read_csv(
    #     "data/isc.csv",
    #     parse_dates=["real_date", "eop"],
    #     date_format="%Y%m%d",
    # )
    # ticker = "TRY_CTOT_NSA_PI"
    # isc = InformationStateChanges.from_isc_df(df, ticker=ticker, iis=True)
    # print(isc)

    from macrosynergy.download import JPMaQSDownload
    from macrosynergy.management import InformationStateChanges

    tickers = ["USD_GDPPC_SA", "GBP_GDPPC_SA"]

    with JPMaQSDownload() as jpmaqs:
        df = jpmaqs.download(tickers=tickers, metrics="all")

    isc = InformationStateChanges.from_qdf(df[["cid", "xcat", "real_date", "value"]])
    iqdf = isc.to_qdf()
    print(list(isc.keys()))
