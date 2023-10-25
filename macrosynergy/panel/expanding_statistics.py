"""
Module for storing functions for computing expanding statistics 
on time series data.
"""

import pandas as pd
import numpy as np
from itertools import accumulate
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management import decorators

from macrosynergy.management.utils import (
    expanding_mean_with_nan as _expanding_mean_with_nan,
)

DEPRECATION_VERSION: str = "0.1.1"

WARN_STR: str = (
    "`macrosynergy.panel.{old_method}` was deprecated in version {deprecate_version} "
    "and moved to `macrosynergy.management.utils.{new_method}()`. "
    "This module and path will be removed in a future release."
)


@decorators.deprecate(
    new_func=_expanding_mean_with_nan,
    deprecate_version=DEPRECATION_VERSION,
    message=WARN_STR,
)
def expanding_mean_with_nan(*args, **kwargs):
    """
    Deprecated. Moved to `macrosynergy.management.utils.expanding_mean_with_nan()`.
    """
    return _expanding_mean_with_nan(*args, **kwargs)


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "USD", "NZD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )

    # Define the cross-sections over different timestamps such that the pivoted DataFrame
    # includes NaN values: more realistic testcase.
    df_cids.loc["AUD"] = ["2022-01-01", "2022-02-01", 0.5, 2]
    df_cids.loc["CAD"] = ["2022-01-10", "2022-02-01", 0.5, 2]
    df_cids.loc["GBP"] = ["2022-01-20", "2022-02-01", -0.2, 0.5]
    df_cids.loc["USD"] = ["2022-01-01", "2022-02-01", -0.2, 0.5]
    df_cids.loc["NZD"] = ["2022-01-05", "2022-02-01", -0.1, 2]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR"] = ["2010-01-01", "2022-02-01", 0, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2011-01-01", "2022-02-01", 1, 2, 0.9, 0.5]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2022-02-01", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2013-01-01", "2022-02-01", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd_xr = dfd[dfd["xcat"] == "XR"]

    dfw = dfd_xr.pivot(index="real_date", columns="cid", values="value")
    no_rows = dfw.shape[0]

    ret_mean = expanding_mean_with_nan(dfw)
