"""
Tools used to update and modify a quantamental DataFrame.

::docs::update_df::sort_first::

"""

import pandas as pd
import warnings

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.make_relative_value import make_relative_value
from functools import wraps, update_wrapper
from inspect import signature
from macrosynergy.management import decorators

from macrosynergy.management.utils import (
    df_tickers as _df_tickers,
    update_df as _update_df,
    update_categories as _update_categories,
    update_tickers as _update_tickers,
)

DEPRECATION_VERSION: str = "0.1.1"

WARN_STR: str = (
    "`macrosynergy.management.update_df.{old_method}` has been deprecated "
    "and moved to `macrosynergy.management.utils.{new_method}()`. "
    "This module and path will be removed in a future release ({deprecate_version})."
)


@decorators.deprecate(
    new_func=_update_df, deprecate_version=DEPRECATION_VERSION, message=WARN_STR
)
def update_df(*args, **kwargs):
    return _update_df(*args, **kwargs)


@decorators.deprecate(
    new_func=_update_categories, deprecate_version=DEPRECATION_VERSION, message=WARN_STR
)
def update_categories(*args, **kwargs):
    return _update_categories(*args, **kwargs)


@decorators.deprecate(
    new_func=_df_tickers, deprecate_version=DEPRECATION_VERSION, message=WARN_STR
)
def df_tickers(*args, **kwargs):
    return _df_tickers(*args, **kwargs)


@decorators.deprecate(
    new_func=_update_tickers, deprecate_version=DEPRECATION_VERSION, message=WARN_STR
)
def update_tickers(*args, **kwargs):
    return _update_tickers(*args, **kwargs)


if __name__ == "__main__":
    # Simulate dataframe.

    cids = ["AUD", "CAD", "GBP", "NZD"]
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

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    tickers = df_tickers(dfd)

    black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    # Test the above method by using the in-built make_relative_value() method.
    dfd_1_rv = make_relative_value(
        dfd,
        xcats=["GROWTH", "INFL"],
        cids=None,
        blacklist=None,
        rel_meth="subtract",
        rel_xcats=None,
        postfix="RV",
    )

    dfd_add = update_categories(df=dfd, df_add=dfd_1_rv)

    dfd_1_rv_blacklist = make_relative_value(
        dfd,
        xcats=["GROWTH", "INFL"],
        cids=None,
        blacklist=None,
        rel_meth="divide",
        rel_xcats=None,
        postfix="RV",
    )

    dfd_add_2 = update_df(df=dfd_add, df_add=dfd_1_rv_blacklist, xcat_replace=True)
