from typing import List

import pandas as pd
import numpy as np

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.types import QuantamentalDataFrame


def extend_history(
    df: pd.DataFrame,
    new_xcat: str,
    cids: List[str] = None,
    hierarchy: List[str] = [],
    backfill: bool = False,
    start: str = None,
):
    """
    Extends the history of a dataframe by creating a new xcat with the same values as the original xcat.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to extend.
    new_xcat : str
        The name of the new xcat.
    cids : List[str], optional
        The cids to extend. If None, all cids are extended.
    hierarchy : List[str], optional
        The hierarchy of the xcat to extend. If empty, the xcat is extended as is.
    backfill : bool, optional
        If True, the new xcat is backfilled to the start date of the original xcat.
    start : str, optional
        The start date of the new xcat. If backfill is True, this is the start date of the backfill.
    
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
        else:
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
    cids: List[str] = None,
    hierarchy: List[str] = [],
    backfill: bool = False,
    start: str = None,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(new_xcat, str):
        raise TypeError("new_xcat must be a string")
    if not isinstance(cids, list):
        raise TypeError("cids must be a list")
    if not isinstance(hierarchy, list):
        raise TypeError("hierarchy must be a list")
    if not isinstance(backfill, bool):
        raise ValueError("backfill must be a boolean")
    if not isinstance(start, str):
        raise TypeError("start must be a string")
    if not all(isinstance(cid, str) for cid in cids):
        raise TypeError("cids must be a list of strings")
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
        dfd, "INFL1", cids, ["INFL", "INFL0"], backfill=True, start="1995-01-01"
    )

    pass
