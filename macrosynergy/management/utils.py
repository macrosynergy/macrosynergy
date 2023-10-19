"""
Generic dataframe and type conversion functions specific to the
Macrosynergy package and JPMaQS dataframes/data.
"""

import pandas as pd
import numpy as np
import datetime
from typing import Any, List, Dict, Optional, Union, Set, Iterable, overload
import requests, requests.compat
import warnings


##############################
#   Overloads
##############################


@overload
def get_cid(ticker: str) -> str:
    ...


@overload
def get_cid(ticker: Iterable[str]) -> List[str]:
    ...


@overload
def get_xcat(ticker: str) -> str:
    ...


@overload
def get_xcat(ticker: Iterable[str]) -> List[str]:
    ...


@overload
def split_ticker(ticker: str) -> str:
    ...


@overload
def split_ticker(ticker: Iterable[str]) -> List[str]:
    ...


##############################
#   Helpful Functions
##############################


def split_ticker(ticker: Union[str, Iterable[str]], mode: str) -> Union[str, List[str]]:
    """
    Returns either the cross-sectional identifier (cid) or the category (xcat) from a
    ticker. The function is overloaded to accept either a single ticker or an iterable
    (e.g. list, tuple, pd.Series, np.array) of tickers.

    :param <str> ticker: The ticker to be converted.
    :param <str> mode: The mode to be used. Must be either "cid" or "xcat".

    Returns
    :return <str>: The cross-sectional identifier or category.
    """
    if not isinstance(mode, str):
        raise TypeError("Argument `mode` must be a string.")

    mode: str = mode.lower().strip()
    if mode not in ["cid", "xcat"]:
        raise ValueError("Argument `mode` must be either 'cid' or 'xcat'.")

    if not isinstance(ticker, str):
        if isinstance(ticker, Iterable):
            if len(ticker) == 0:
                raise ValueError("Argument `ticker` must not be empty.")
            return [split_ticker(t, mode) for t in ticker]
        else:
            raise TypeError(
                "Argument `ticker` must be a string or an iterable of strings."
            )

    if "_" not in ticker:
        raise ValueError(
            "Argument `ticker` must be a string" " with at least one underscore."
            f" Received '{ticker}' instead."
        )

    cid, xcat = str(ticker).split("_", 1)
    rStr: str = cid if mode == "cid" else xcat
    if len(rStr.strip()) == 0:
        raise ValueError(
            f"Unable to extract {mode} from ticker {ticker}."
            " Please check the ticker."
        )

    return rStr


def get_cid(ticker: Union[str, Iterable[str]]) -> Union[str, List[str]]:
    """
    Returns the cross-sectional identifier (cid) from a ticker.

    :param <str> ticker: The ticker to be converted.

    Returns
    :return <str>: The cross-sectional identifier.
    """
    return split_ticker(ticker, mode="cid")


def get_xcat(ticker: Union[str, Iterable[str]]) -> str:
    """
    Returns the category (xcat) from a ticker.

    :param <str> ticker: The ticker to be converted.

    Returns
    :return <str>: The category.
    """
    return split_ticker(ticker, mode="xcat")


def generate_random_date(
    start: Optional[Union[str, datetime.datetime, pd.Timestamp]] = "1990-01-01",
    end: Optional[Union[str, datetime.datetime, pd.Timestamp]] = "2020-01-01",
) -> str:
    """
    Generates a random date between two dates.

    :param <str> start: The start date, in the ISO format (YYYY-MM-DD).
    :param <str> end: The end date, in the ISO format (YYYY-MM-DD).

    Returns
    :return <str>: The random date.
    """

    if not isinstance(start, (str, datetime.datetime, pd.Timestamp)):
        raise TypeError(
            "Argument `start` must be a string, datetime.datetime, or pd.Timestamp."
        )
    if not isinstance(end, (str, datetime.datetime, pd.Timestamp)):
        raise TypeError(
            "Argument `end` must be a string, datetime.datetime, or pd.Timestamp."
        )

    start: pd.Timestamp = pd.Timestamp(start)
    end: pd.Timestamp = pd.Timestamp(end)
    if start == end:
        return start.strftime("%Y-%m-%d")
    else:
        return pd.Timestamp(
            np.random.randint(start.value, end.value, dtype=np.int64)
        ).strftime("%Y-%m-%d")


def get_dict_max_depth(d: dict) -> int:
    """
    Returns the maximum depth of a dictionary.

    :param <dict> d: The dictionary to be searched.

    Returns
    :return <int>: The maximum depth of the dictionary.
    """
    return (
        1 + max(map(get_dict_max_depth, d.values()), default=0)
        if isinstance(d, dict)
        else 0
    )


def rec_search_dict(d: dict, key: str, match_substring: bool = False, match_type=None):
    """
    Recursively searches a dictionary for a key and returns the value
    associated with it.

    :param <dict> d: The dictionary to be searched.
    :param <str> key: The key to be searched for.
    :param <bool> match_substring: If True, the function will return
        the value of the first key that contains the substring
        specified by the key parameter. If False, the function will
        return the value of the first key that matches the key
        parameter exactly. Default is False.
    :param <Any> match_type: If not None, the function will look for
        a key that matches the search parameters and has
        the specified type. Default is None.
    :return Any: The value associated with the key, or None if the key
        is not found.
    """
    if not isinstance(d, dict):
        return None

    for k, v in d.items():
        if match_substring:
            if key in k:
                if match_type is None or isinstance(v, match_type):
                    return v
        else:
            if k == key:
                if match_type is None or isinstance(v, match_type):
                    return v

        if isinstance(v, dict):
            result = rec_search_dict(v, key, match_substring, match_type)
            if result is not None:
                return result

    return None


def is_valid_iso_date(date: str) -> bool:
    if not isinstance(date, str):
        raise TypeError("Argument `date` must be a string.")

    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def convert_iso_to_dq(date: str) -> str:
    if is_valid_iso_date(date):
        r = date.replace("-", "")
        assert len(r) == 8, "Date formatting failed"
        return r
    else:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD")


def convert_dq_to_iso(date: str) -> str:
    if len(date) == 8:
        r = datetime.datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
        assert is_valid_iso_date(r), "Failed to format date"
        return r
    else:
        raise ValueError("Incorrect date format, should be YYYYMMDD")


def form_full_url(url: str, params: Dict = {}) -> str:
    """
    Forms a full URL from a base URL and a dictionary of parameters.
    Useful for logging and debugging.

    :param <str> url: base URL.
    :param <dict> params: dictionary of parameters.

    :return <str>: full URL
    """
    return requests.compat.quote(
        (f"{url}?{requests.compat.urlencode(params)}" if params else url),
        safe="%/:=&?~#+!$,;'@()*[]",
    )


def common_cids(df: pd.DataFrame, xcats: List[str]):
    """
    Returns a list of cross-sectional identifiers (cids) for which the specified categories
       (xcats) are available.

    :param <pd.Dataframe> df: Standardized JPMaQS DataFrame with necessary columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> xcats: A list with least two categories whose cross-sectional
        identifiers are being considered.

    return <List[str]>: List of cross-sectional identifiers for which all categories in `xcats`
        are available.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")

    if not isinstance(xcats, list):
        raise TypeError("Argument `xcats` must be a list.")
    elif not all(isinstance(elem, str) for elem in xcats):
        raise TypeError("Argument `xcats` must be a list of strings.")
    elif len(xcats) < 2:
        raise ValueError("Argument `xcats` must contain at least two category tickers.")
    elif not set(xcats).issubset(set(df["xcat"].unique())):
        raise ValueError("All categories in `xcats` must be present in the DataFrame.")

    cid_sets: List[set] = []
    for xc in xcats:
        sc: set = set(df[df["xcat"] == xc]["cid"].unique())
        if sc:
            cid_sets.append(sc)

    ls: List[str] = list(cid_sets[0].intersection(*cid_sets[1:]))
    return sorted(ls)


##############################
#   Dataframe Functions
##############################


def standardise_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    idx_cols: List[str] = ["cid", "xcat", "real_date"]
    commonly_used_cols: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
    if not set(df.columns).issuperset(set(idx_cols)):
        fail_str: str = (
            f"Error : Tried to standardize DataFrame but failed."
            f"DataFrame not in the correct format. Please ensure "
            f"that the DataFrame has the following columns: "
            f"'cid', 'xcat', 'real_date', along with any other "
            "variables you wish to include (e.g. 'value', 'mop_lag', "
            "'eop_lag', 'grading')."
        )

        try:
            dft: pd.DataFrame = df.reset_index()
            found_cols: list = dft.columns.tolist()
            fail_str += f"\nFound columns: {found_cols}"
            if not set(dft.columns).issuperset(set(idx_cols)):
                raise ValueError(fail_str)
            df = dft.copy()
        except:
            raise ValueError(fail_str)

        # check if there is atleast one more column
        if len(df.columns) < 4:
            raise ValueError(fail_str)

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
    df["cid"] = df["cid"].astype(str)
    df["xcat"] = df["xcat"].astype(str)
    df = df.sort_values(by=["real_date", "cid", "xcat"]).reset_index(drop=True)

    remaining_cols: Set[str] = set(df.columns) - set(idx_cols)

    df = df[idx_cols + sorted(list(remaining_cols))]

    # for every remaining col, try to convert to float
    for col in remaining_cols:
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    non_idx_cols: list = sorted(list(set(df.columns) - set(idx_cols)))
    return df[idx_cols + non_idx_cols]


def drop_nan_series(df: pd.DataFrame, raise_warning: bool = False) -> pd.DataFrame:
    """
    Drops any series that are entirely NaNs.
    Raises a user warning if any series are dropped.

    :param <pd.DataFrame> df: The dataframe to be cleaned.
    :param <bool> raise_warning: Whether to raise a warning if any series are dropped.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Error: The input must be a pandas DataFrame.")
    elif not set(df.columns).issuperset(set(["cid", "xcat", "value"])):
        raise ValueError(
            "Error: The input DataFrame must have columns 'cid', 'xcat', 'value'."
        )
    elif not df["value"].isna().any():
        return df

    if not isinstance(raise_warning, bool):
        raise TypeError("Error: The raise_warning argument must be a boolean.")

    df_orig: pd.DataFrame = df.copy()
    for cd, xc in df_orig.groupby(["cid", "xcat"]).groups:
        sel_series: pd.Series = df_orig[
            (df_orig["cid"] == cd) & (df_orig["xcat"] == xc)
        ]["value"]
        if sel_series.isna().all():
            if raise_warning:
                warnings.warn(
                    message=f"The series {cd}_{xc} is populated "
                    "with NaNs only, and will be dropped.",
                    category=UserWarning,
                )
            df = df[~((df["cid"] == cd) & (df["xcat"] == xc))]

    return df.reset_index(drop=True)


def qdf_to_ticker_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a standardized JPMaQS DataFrame to a wide format DataFrame
    with each column representing a ticker.

    :param <pd.DataFrame> df: A standardised quantamental dataframe.
    :return <pd.DataFrame>: The converted DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")

    STD_COLS: List[str] = ["cid", "xcat", "real_date", "value"]
    if not set(df.columns).issuperset(set(STD_COLS)):
        df: pd.DataFrame = standardise_dataframe(df)[STD_COLS]

    df["ticker"] = df["cid"] + "_" + df["xcat"]
    # drop cid and xcat
    df = (
        df.drop(columns=["cid", "xcat"])
        .pivot(index="real_date", columns="ticker", values="value")
        .rename_axis(None, axis=1)
    )

    return df


def ticker_df_to_qdf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a wide format DataFrame (with each column representing a ticker)
    to a standardized JPMaQS DataFrame.

    :param <pd.DataFrame> df: A wide format DataFrame.
    :return <pd.DataFrame>: The converted DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument `df` must be a pandas DataFrame.")

    # pivot to long format
    df = (
        df.stack(level=0)
        .reset_index()
        .rename(columns={0: "value", "level_1": "ticker"})
    )
    # split ticker using get_cid and get_xcat
    df["cid"] = get_cid(df["ticker"])
    df["xcat"] = get_xcat(df["ticker"])
    # drop ticker column

    df = df.drop(columns=["ticker"])

    # standardise and return
    return standardise_dataframe(df=df)

def apply_slip(df: pd.DataFrame, slip: int,
                    cids: List[str], xcats: List[str],
                    metrics: List[str], raise_error: bool = True) -> pd.DataFrame:
        """
        Applied a slip, i.e. a negative lag, to the target DataFrame 
        for the given cross-sections and categories, on the given metrics.
        
        :param <pd.DataFrame> target_df: DataFrame to which the slip is applied.
        :param <int> slip: Slip to be applied.
        :param <List[str]> cids: List of cross-sections.
        :param <List[str]> xcats: List of categories.
        :param <List[str]> metrics: List of metrics to which the slip is applied.
        :return <pd.DataFrame> target_df: DataFrame with the slip applied.
        :raises <TypeError>: If the provided parameters are not of the expected type.
        :raises <ValueError>: If the provided parameters are semantically incorrect.
        """

        df = df.copy()
        if not (isinstance(slip, int) and slip >= 0):
            raise ValueError("Slip must be a non-negative integer.")
        
        if cids is None:
            cids = df['cid'].unique().tolist()
        if xcats is None:
            xcats = df['xcat'].unique().tolist()

        sel_tickers : List[str] = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
        df['tickers'] = df['cid'] + '_' + df['xcat']

        if not set(sel_tickers).issubset(set(df['tickers'].unique())):
            if raise_error:
                raise ValueError("Tickers targetted for applying slip are not present in the DataFrame.\n"
                f"Missing tickers: {sorted(list(set(sel_tickers) - set(df['tickers'].unique())))}")
            else:
                warnings.warn("Tickers targetted for applying slip are not present in the DataFrame.\n"
                f"Missing tickers: {sorted(list(set(sel_tickers) - set(df['tickers'].unique())))}")

        slip : int = slip.__neg__()
        
        df[metrics] = df.groupby('tickers')[metrics].shift(slip)
        df = df.drop(columns=['tickers'])
        
        return df

def downsample_df_on_real_date(
    df: pd.DataFrame,
    groupby_columns: List[str] = [],
    freq: str = "M",
    agg: str = "mean",
):
    """
    Downsample JPMaQS DataFrame.

    :param <pd.Dataframe> df: standardized JPMaQS DataFrame with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List> groupby_columns: a list of columns used to group the DataFrame.
    :param <str> freq: frequency option. Per default the correlations are calculated
        based on the native frequency of the datetimes in 'real_date', which is business
        daily. Downsampling options include weekly ('W'), monthly ('M'), or quarterly
        ('Q') mean.
    :param <str> agg: aggregation method. Must be one of "mean" (default), "median",
        "min", "max", "first" or "last".

    :return <pd.DataFrame>: the downsampled DataFrame.
    """

    if not set(groupby_columns).issubset(df.columns):
        raise ValueError(
            "The columns specified in 'groupby_columns' were not found in the DataFrame."
        )

    if not isinstance(freq, str):
        raise TypeError("`freq` must be a string")
    else:
        freq: str = freq.upper()
        if freq not in ["D", "W", "M", "Q", "A"]:
            raise ValueError("`freq` must be one of 'D', 'W', 'M', 'Q' or 'A'")

    if not isinstance(agg, str):
        raise TypeError("`agg` must be a string")
    else:
        agg: str = agg.lower()
        if agg not in ["mean", "median", "min", "max", "first", "last"]:
            raise ValueError(
                "`agg` must be one of 'mean', 'median', 'min', 'max', 'first', 'last'"
            )

    return (
        df.set_index("real_date")
        .groupby(groupby_columns)
        .resample(freq)
        .agg(agg, numeric_only=True)
        .reset_index()
    )
