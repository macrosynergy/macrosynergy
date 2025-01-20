import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management import reduce_df


def extend_history(
    df: pd.DataFrame,
    new_xcat: str,
    cids: Optional[List[str]] = None,
    hierarchy: List[str] = [],
    backfill: bool = False,
    start: str = None,
):
    """
    Extends the history of a dataframe by creating a new xcat by combining hierarchical categories.
    The method prioritizes superior categories for the new xcat and supplements with inferior ones
    where superior category data is unavailable.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing categories that are to be extended.
    new_xcat : str
        The name of the new xcat.
    cids : List[str], optional
        The cross sections to extend. If None, all cids available for any category in 'hierarchy' are extended.
    hierarchy : List[str]
         list of categories from best to worst for representation of the concept.
         Inferior categories are only used to extend the history of the superior ones.
         The new category consists of the best representation category values
         and inferior category values that are available prior to any superior.
    backfill : bool, optional
        If True, the new xcat is backfilled to the start date specified by the 'start' parameter.
    start : str, optional
        The start date of the new xcat. If backfill is True, this values will be backfilled up to this date.

    Returns
    -------
    ~pandas.DataFrame
        standardized DataFrame for the new xcat with extended history, with the columns:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    df = QuantamentalDataFrame(df)
    result_as_categorical = df.InitializedAsCategorical
    _extend_history_checks(
        df=df,
        new_xcat=new_xcat,
        cids=cids,
        hierarchy=hierarchy,
        backfill=backfill,
        start=start,
    )
    start = pd.to_datetime(start)

    extended_results = []

    df, _, cids_in_df = reduce_df(df=df, xcats=hierarchy, cids=cids, out_all=True)

    if df.empty:
        raise ValueError("No data available for the specified cids and categories.")

    if cids is None:
        cids = cids_in_df
    else:
        missing_cids = list(set(cids) - set(cids_in_df))
        cids = cids_in_df
        if len(missing_cids) > 0:
            warnings.warn(
                f"Warning: cids {missing_cids} do not exist for any category in hierarchy. They will be ignored."
            )

    for cid in cids:

        cid_df = df[df["cid"] == cid]

        extended_series = pd.DataFrame()

        for category in hierarchy:

            cat_df = cid_df[cid_df["xcat"] == category].sort_values("real_date")

            if extended_series.empty:
                extended_series = cat_df.copy()
            else:
                min_real_date = extended_series["real_date"].min()

                inferior_values = cat_df[cat_df["real_date"] < min_real_date]
                extended_series = pd.concat([extended_series, inferior_values])

        extended_series = extended_series.sort_values("real_date")

        extended_series["xcat"] = new_xcat
        extended_series["cid"] = cid

        if backfill:
            if not extended_series.empty:
                min_date = extended_series["real_date"].min()
                if min_date > start:
                    backfilled_data = pd.DataFrame(
                        {
                            "real_date": pd.bdate_range(
                                start=start, end=min_date - pd.Timedelta(days=1)
                            ),
                            "value": extended_series.iloc[0]["value"],
                            "cid": cid,
                            "xcat": new_xcat,
                        }
                    )
                    extended_series = pd.concat([backfilled_data, extended_series])
        elif start is not None:
            extended_series = extended_series[extended_series["real_date"] >= start]

        # Add new_xcat and cid
        extended_series["xcat"] = new_xcat
        extended_series["cid"] = cid

        extended_results.append(extended_series)

    extended_df = pd.concat(extended_results, ignore_index=True)
    extended_df = extended_df.sort_values(["cid", "real_date"])

    return QuantamentalDataFrame(extended_df, categorical=result_as_categorical)


def _extend_history_checks(
    df: pd.DataFrame,
    new_xcat: str,
    cids: Optional[List[str]] = None,
    hierarchy: List[str] = [],
    backfill: bool = False,
    start: str = None,
):
    """
    Checks for inputs to `extend_history`.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing categories that are to be extended.
    new_xcat : str
        The name of the new xcat.
    cids : List[str], optional
        The cross sections to extend. If None, all cids available for any category in 'hierarchy' are extended.
    hierarchy : List[str]
         list of categories from best to worst for representation of the concept.
         Inferior categories are only used to extend the history of the superior ones.
         The new category consists of the best representation category values
         and inferior category values that are available prior to any superior.
    backfill : bool, optional
        If True, the new xcat is backfilled to the start date specified by the 'start' parameter.
    start : str, optional
        The start date of the new xcat. If backfill is True, this values will be backfilled up to this date.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(new_xcat, str):
        raise TypeError("new_xcat must be a string")
    if cids is not None:
        if not isinstance(cids, list):
            raise TypeError("cids must be a list")
        if not all(isinstance(cid, str) for cid in cids):
            raise TypeError("cids must be a list of strings")
    if not isinstance(hierarchy, list):
        raise TypeError("hierarchy must be a list")
    if not isinstance(backfill, bool):
        raise TypeError("backfill must be a boolean")
    if start is not None and not isinstance(start, str):
        raise TypeError("start must be a string")
    if not all(isinstance(cat, str) for cat in hierarchy):
        raise TypeError("hierarchy must be a list of strings")

    if backfill and start is None:
        raise ValueError("start must be provided if backfill is True")


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "USD", "NZD"]
    xcats = ["INFL", "INFL0"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )

    df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.5, 2]
    df_cids.loc["CAD"] = ["2011-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-11-30", -0.2, 0.5]
    df_cids.loc["USD"] = ["2010-01-01", "2020-12-30", -0.2, 0.5]
    df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
    df_cids.loc["EUR"] = ["2002-01-01", "2020-09-30", -0.2, 2]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["INFL"] = ["2010-01-01", "2020-09-30", 1, 2, 0.8, 0.5]
    df_xcats.loc["INFL0"] = ["2000-01-01", "2020-09-30", 1, 3, 0.5, 0.2]

    np.random.seed(0)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    df = extend_history(
        dfd, "INFL1", cids, ["INFL", "INFL0"], backfill=False, start=None
    )

    pass
