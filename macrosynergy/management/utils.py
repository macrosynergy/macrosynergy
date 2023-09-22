"""
Generic dataframe and type conversion functions specific to the
Macrosynergy package and JPMaQS dataframes/data.
"""

import pandas as pd
import numpy as np
import datetime
from typing import Any, List, Dict, Optional, Union, Set
import requests, requests.compat
import warnings

##############################
#   Helpful Functions
##############################


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
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def convert_to_iso_format(date: Any = None) -> str:
    raise NotImplementedError("This function is not yet implemented.")
    """
    Converts a datetime like object or string to an ISO date string.

    Parameters
    :param <Any> date: The date to be converted. This can be a
        datetime object, a string, pd.Timestamp, or np.datetime64.

    Returns
    :return <str>: The ISO date string (YYYY-MM-DD).
    """
    if date is None:
        ValueError("Argument `date` cannot be None.")

    r: Optional[str] = None
    if isinstance(date, str):
        r: Optional[str] = None
        if is_valid_iso_date(date):
            r = date
        else:
            if len(date) == 8:
                try:
                    r = convert_dq_to_iso(date)
                except Exception as e:
                    if isinstance(e, (ValueError, AssertionError)):
                        pass
            else:
                for sep in ["-", "/", ".", " "]:
                    if sep in date:
                        try:
                            sd = date.split(sep)
                            dx = date
                            if len(sd) == 3:
                                if len(sd[1]) == 3:
                                    sd[1] = {
                                        "JAN": "01",
                                        "FEB": "02",
                                        "MAR": "03",
                                        "APR": "04",
                                        "MAY": "05",
                                        "JUN": "06",
                                        "JUL": "07",
                                        "AUG": "08",
                                        "SEP": "09",
                                        "OCT": "10",
                                        "NOV": "11",
                                        "DEC": "12",
                                    }[sd[1].upper()]
                                    dx = sep.join(sd)
                                r = datetime.datetime.strptime(
                                    dx, "%d" + sep + "%m" + sep + "%Y"
                                ).strftime("%Y-%m-%d")
                                break
                        except Exception as e:
                            if isinstance(e, ValueError):
                                pass
                            else:
                                raise e

        if r is None:
            raise RuntimeError("Could not convert date to ISO format.")
    elif isinstance(date, (datetime.datetime, pd.Timestamp, np.datetime64)):
        r = date.strftime("%Y-%m-%d")
    else:
        raise TypeError(
            "Argument `date` must be a string, datetime.datetime, pd.Timestamp or np.datetime64."
        )

    assert is_valid_iso_date(r), "Failed to convert date to ISO format."
    return r


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


def wide_to_long(
    df: pd.DataFrame,
    wide_var: str = "cid",
    val_col: str = "value",
) -> pd.DataFrame:
    """
    Converts a wide dataframe to a long dataframe.

    :param <pd.DataFrame> df: The dataframe to be converted.
    :param <str> wide_var: The variable name of the wide variable.
        In case the columns are ... cid_1, cid_2, cid_3, ... then
        wide_var should be "cid", else "xcat" or "real_date" must be
        passed.

    Returns
    :return <pd.DataFrame>: The converted dataframe.
    """
    idx_cols: list = ["cid", "xcat", "real_date"]

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Error: The input must be a pandas DataFrame.")

    if wide_var not in ["cid", "xcat", "real_date"]:
        raise ValueError(
            "Error: The wide_var must be one of 'cid', 'xcat', 'real_date'."
        )

    """ 
    if wide_var == "cid":
     then the columns are real_date, xcat, cidX, cidY, cidZ, ...
     convert to real_date, xcat, cid, value
    """
    # use stack and unstack to convert to long format
    df = df.set_index(idx_cols).stack().reset_index()
    df.columns = idx_cols + [wide_var, val_col]

    return standardise_dataframe(df)
