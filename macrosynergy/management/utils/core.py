"""
This module contains core utility functions, as well as stand-alone functions
that are used across the package.
"""

import datetime
import itertools

from macrosynergy.management.types import QuantamentalDataFrame
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Union, overload

import numpy as np
import pandas as pd
import requests
import requests.compat


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
            "Argument `ticker` must be a string"
            " with at least one underscore."
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
