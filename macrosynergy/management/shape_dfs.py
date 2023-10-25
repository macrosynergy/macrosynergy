"""
Utilities and common functions for the manipulating DataFrames.
"""

import pandas as pd
import random
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management import decorators
from macrosynergy.management.utils import (
    reduce_df as _reduce_df,
    reduce_df_by_ticker as _reduce_df_by_ticker,
    categories_df as _categories_df,
)

DEPRECATION_VERSION: str = "0.1.1"

WARN_STR: str = (
    "`macrosynergy.management.shape_dfs.{old_method}` was deprecated in version {deprecate_version}"
    "and moved to `macrosynergy.management.utils.{new_method}()`. "
    "This module and path will be removed in a future release."
)


@decorators.deprecate(
    new_func=_reduce_df, deprecate_version=DEPRECATION_VERSION, message=WARN_STR
)
def reduce_df(*args, **kwargs):
    """
    Deprecated. Moved to `macrosynergy.management.utils.reduce_df()`.
    """
    return _reduce_df(*args, **kwargs)


def reduce_df_by_ticker(*args, **kwargs):
    """
    Deprecated. Moved to `macrosynergy.management.utils.reduce_df_by_ticker()`.
    """
    return _reduce_df_by_ticker(*args, **kwargs)


def categories_df(*args, **kwargs):
    """
    Deprecated. Moved to `macrosynergy.management.utils.categories_df()`.
    """
    return _categories_df(*args, **kwargs)


if __name__ == "__main__":
    cids = ["NZD", "AUD", "GBP", "CAD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
    df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
    df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfd_x1 = reduce_df(
        dfd, xcats=xcats[:-1], cids=cids[0], start="2012-01-01", end="2018-01-31"
    )

    tickers = [cid + "_XR" for cid in cids]
    dfd_xt = reduce_df_by_ticker(dfd, ticks=tickers, blacklist=black)

    # Testing categories_df().
    dfc1 = categories_df(
        dfd,
        xcats=["GROWTH", "CRY"],
        cids=cids,
        val="value",
        freq="W",
        lag=1,
        xcat_aggs=["mean", "mean"],
        start="2000-01-01",
        blacklist=black,
    )
