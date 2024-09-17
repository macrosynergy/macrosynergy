"""
Module with functions for processing "blacklist" data for cross-sections in a quantamental
DataFrame.
"""

import numpy as np
import pandas as pd
from typing import List
from itertools import groupby
from macrosynergy.management.utils import reduce_df
from macrosynergy.management.simulate import make_qdf_black, make_qdf
from macrosynergy.management.types import QuantamentalDataFrame


def startend(dti, start, length):
    """Return start and end dates of a sequence as tuple

    :param <DateTimeIndex> dti: datetime series of working days
    :param <int> start: index of start
    :param <int> length: number of sequential days

    :return <Tuple[pd.Timestamp, pd.Timestamp]>: tuple of start and end date
    """

    tup = (dti[start], dti[start + (length - 1)])
    return tup


def make_blacklist(
    df: QuantamentalDataFrame,
    xcat: str,
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    nan_black: bool = False,
):
    """
    Converts binary category of standardized dataframe into a standardized dictionary
    that can serve as a blacklist for cross-sections in further analyses

    :param <QuantamentalDataFrame> df: standardized DataFrame with following columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <str> xcat: category with binary values, where 1 means blacklisting and 0
        means not blacklisting.
    :param List<str> cids: list of cross-sections that are considered in the formation
        of the blacklist. Per default, all available cross sections are considered.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the respective category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the respective category is available is used.
    :param <bool> nan_black: if True NaNs are blacklisted (coverted to ones). Defaults is
        False, i.e. NaNs are converted to zeroes.

    :return <dict>: standardized dictionary with cross-sections as keys and tuples of
        start and end dates of the blacklist periods in ISO formats as values.
        If one cross section has multiple blacklist periods, numbers are added to the
        keys (i.e. TRY_1, TRY_2, etc.)
    """

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("df must be a standardized quantamental dataframe")

    dfd = reduce_df(df=df, xcats=[xcat], cids=cids, start=start, end=end)

    if "value" not in dfd.columns:
        raise ValueError("`value` column not found in df")

    if not all(np.isin(dfd["value"].dropna().unique(), [0, 1])):
        raise ValueError("blacklist values must all be 0/1")

    df_pivot = dfd.pivot(index="real_date", columns="cid", values="value")
    dates = df_pivot.index
    cids_df = list(df_pivot.columns)

    # replace NaNs
    df_pivot[df_pivot.isna()] = int(nan_black)  # 1 if nan_black else 0

    dates_dict = {}
    for cid in cids_df:
        count = 0
        column = df_pivot[cid].to_numpy()
        si = 0
        for k, v in groupby(column):  # iterator of consecutive keys and values
            v = list(v)  # instantiate the iterable in memory.
            length = len(v)
            if v[0] == 1:  # if blacklist period
                if count == 0:
                    dates_dict[cid] = startend(dates, si, length)
                elif count == 1:
                    val = dates_dict.pop(cid)
                    dates_dict[cid + "_1"] = val  # change key if more than 1 per cid
                    count += 1
                    dates_dict[cid + "_" + str(count)] = startend(dates, si, length)
                else:
                    dates_dict[cid + "_" + str(count)] = startend(dates, si, length)
                count += 1
            si += length
    return dates_dict


if __name__ == "__main__":
    cids = ["AUD", "GBP", "CAD", "USD"]
    cols = ["earliest", "latest", "mean_add", "sd_mult"]
    df_cid1 = pd.DataFrame(index=cids, columns=cols)

    df_cid1.loc["AUD"] = ["2010-01-01", "2020-12-31", 0, 1]
    df_cid1.loc["GBP"] = ["2011-01-01", "2020-11-30", 0, 1]
    df_cid1.loc["CAD"] = ["2011-01-01", "2021-11-30", 0, 1]
    df_cid1.loc["USD"] = ["2011-01-01", "2020-12-30", 0, 1]

    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
    df_xcat1 = pd.DataFrame(index=["FXXR_NSA", "FXCRY_NSA"], columns=cols)
    df_xcat1.loc["FXXR_NSA"] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
    df_xcat1.loc["FXCRY_NSA"] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
    df1 = make_qdf(df_cid1, df_xcat1, back_ar=0.05)

    df_xcat2 = pd.DataFrame(index=["FXNONTRADE_NSA"], columns=["earliest", "latest"])
    df_xcat2.loc["FXNONTRADE_NSA"] = ["2010-01-01", "2021-11-30"]
    black = {
        "AUD": ("2010-01-12", "2010-06-14"),
        "USD": ("2011-08-17", "2011-09-20"),
        "CAD_1": ("2011-01-04", "2011-01-23"),
        "CAD_2": ("2013-01-09", "2013-04-10"),
        "CAD_3": ("2015-01-12", "2015-03-12"),
        "CAD_4": ("2021-11-01", "2021-11-20"),
    }

    print(black)
    df2 = make_qdf_black(df_cid1, df_xcat2, blackout=black)

    df = pd.concat([df1, df2]).reset_index()

    dates_dict = make_blacklist(
        df, xcat="FXNONTRADE_NSA", cids=None, start=None, end=None
    )

    # If the output, from the below printed dictionary, differs from the above defined
    # dictionary, it should be by a date or two, as the construction of the dataframe,
    # using make_qdf_black(), will account for the dates received, in the dictionary,
    # being weekends. Therefore, if any of the dates, for the start or end of the
    # blackout period are Saturday or Sunday, the date for will be shifted to the
    # following Monday. Hence, a break in alignment from "blackout" to "dates_dict".
    print(dates_dict)
