"""
Module implementing the `make_relative_category()` function, which calculates indicator
values relative to a set of categories. Designed with sectoral equity returns in mind,
this function can be extended to other data types with a similar structure.
Unlike `make_relative_value()`, which computes values relative to a cross-sectional mean
for a given category, `make_relative_category()` returns values relative to an average
across categories.
"""

import pandas as pd
from typing import List, Set

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import reduce_df
from macrosynergy.management.types import QuantamentalDataFrame
import warnings


def make_relative_category(
    df: pd.DataFrame,
    xcats: List[str] = None,
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    basket: List[str] = None,
    complete_set: bool = False,
    rel_meth: str = "subtract",
    rel_xcats: List[str] = None,
    postfix: str = "RC",
):
    """
    For each cross section specified in the panel, relative values are calculated by
    either subtracting or dividing by the average across categories at each `real_date`.

    Parameters
    ----------
    df : ~pandas.DataFrame
        standardized JPMaQS DataFrame with the necessary columns: 'cid', 'xcat',
        'real_date' and 'value'.
    xcats : List[str]
        all extended categories for which relative values are to be calculated. The user
        must provide the set of categories to be used in the calculation.
    cids : List[str]
        cross sections for which relative values are calculated.
    start : str
        earliest date in ISO format. Default is None and earliest date for which the
        respective category is available is used.
    end : str
        latest date in ISO format. Default is None and latest date for which the
        respective category is available is used.
    blacklist : dict
        cross sections with date ranges that should be excluded from the output.
    basket : List[str]
        categories to be used for the relative value benchmark. The default is every
        category in the DataFrame that is available over the time-period.
        However, the basket parameter can specify a subset of the available
        categories.
    complete_set : bool
        boolean parameter that determines whether each cross-section is required to have
        the full set of extended categories in the basket for a relative value calculation
        to occur. If set to True, the cross-section will be excluded from the output if
        some desired categories are missing. Default is False. If False, the mean, for the
        relative value, will use the subset of categories available for that cross-section.
    rel_meth : str
        method for calculating relative value. Default is 'subtract'. Alternative is
        'divide'.
    rel_xcats : List[str]
        extended category name of the relative values. Will displace the original
        category names: xcat + postfix. The order should reflect the order of the passed
        categories.
    postfix : str
        acronym to be appended to 'xcat' string to give the name for relative value
        category. Only applies if rel_xcats is None. Default is 'RC' relative category

    Returns
    -------
    ~pandas.DataFrame
        standardized DataFrame with the relative values, with the columns:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    col_names = ["cid", "xcat", "real_date", "value"]
    operations = {
        "divide": pd.Series.div,
        "subtract": pd.Series.sub,
    }
    if rel_meth not in operations:
        raise ValueError("rel_meth must be 'subtract' or 'divide'")

    if not isinstance(xcats, (list, str)):
        raise TypeError("xcats must be a list of strings or a single string.")

    if isinstance(xcats, str):
        xcats = [xcats]

    if rel_xcats is not None:
        if not (
            isinstance(rel_xcats, list) and all([isinstance(x, str) for x in rel_xcats])
        ):
            raise ValueError("List of strings expected for `rel_xcats`.")

        if len(xcats) != len(rel_xcats):
            raise ValueError(
                "`rel_xcats` must have the same number of elements as `xcats`."
            )

        rel_xcats_dict = dict(zip(xcats, rel_xcats))
    else:
        rel_xcats_dict = {x: x + postfix for x in xcats}

    df = QuantamentalDataFrame(df)
    # Intersect parameter set to False. Therefore, cross sections across the categories can vary.
    if basket:
        all_xcats: List[str] = list(set(xcats).union(set(basket)))
    else:
        all_xcats = xcats
    if len(all_xcats) < 1:
        all_xcats = None
    dfx = reduce_df(df, all_xcats, cids, start, end, blacklist, out_all=False)

    if cids is None:
        # All cross sections available - union across categories.
        cids = list(dfx["cid"].unique())

    if basket is not None:
        # Basket must be a subset of the available xcats
        miss: Set = set(basket) - set(df["xcat"])
        error_basket = f"The category basket elements {miss} are not specified or are not available."
        if not len(miss) == 0:
            raise ValueError(error_basket)
    else:
        # Default basket is all available cross sections.
        basket = xcats

    available_cids = dfx["cid"].unique()

    if len(xcats) == len(basket) == 1:
        run_error = (
            "Computing the relative value on a single category using a "
            "basket consisting exclusively of the aforementioned category "
            "is an incorrect usage of the function."
        )
        raise ValueError(run_error)

    df_list: List[pd.DataFrame] = []

    # Each cid could have an incomplete basket, we are allowing it to be flexible
    for i, cid in enumerate(available_cids):
        df_cid = dfx[dfx["cid"] == cid]
        xcats_avl = df_cid["xcat"].unique()

        df_cid: pd.DataFrame = df_cid[["xcat", "real_date", "value"]]

        dfb, basket = _prepare_category_basket(
            df=df_cid,
            cid=cid,
            basket=basket,
            xcats_avl=xcats_avl,
            complete_set=complete_set,
        )

        if len(basket) > 0:
            bm = dfb.groupby(by="real_date").mean(numeric_only=True).reset_index()
        else:
            continue

        dfa = df_cid.copy()
        dfa["count"] = dfa.groupby("real_date")["value"].transform("count")
        dfa = dfa.loc[dfa["count"] > 1]

        # The time-index will be delimited by the respective category.
        dfa = pd.merge(dfa, bm, how="left", on=["real_date"], suffixes=["", "_bm"])
        dfa = dfa.sort_values(by=["xcat", "real_date"])

        dfa["value"] = operations[rel_meth](dfa["value"], dfa["value_bm"], axis=0)

        # cleaning
        df_new = dfa.drop(columns=["value_bm", "count"])
        df_new["xcat"] = df_new["xcat"].map(rel_xcats_dict).astype("category")
        df_new["cid"] = pd.Categorical.from_codes(
            codes=[0] * len(df_new), categories=[cid]
        )

        if (
            df_new.sort_values(["xcat", "real_date"])[col_names].isna().all().all()
            or (len(df_new) == 0)
            or df_new.empty
        ):
            continue

        df_list.append(df_new.sort_values(["xcat", "real_date"])[col_names])

    return QuantamentalDataFrame(
        pd.concat(df_list).reset_index(drop=True),
        categorical=df.InitializedAsCategorical,
    )


def _prepare_category_basket(
    df: pd.DataFrame,
    cid: str,
    basket: List[str],
    xcats_avl: List[str],
    complete_set: bool,
):
    """
    Relative cid-specific indicators can be defined over different sets of categories.
    We will determine the respective basket given the available categories for the
    respective cross section.

    Parameters
    ----------
    df : ~pandas.DataFrame
        long JPMaQS DataFrame of single category.
    cid : str
        target cross section for the relative value calculation.
    basket : ~pandas.DataFrame
        set of categories to be used for the relative value benchmark if available.
    xcats_avl : List[str]
        categories available for the target cross section.
    complete_set : bool
        if True, the basket is only calculated if all categories, held in the basket,
        are available for that respective category.

    Returns
    -------
    ~pandas.DataFrame
        DataFrame with the categories available for the target cross section.
    List[str]
        List of categories available for the target cross section.
    """

    xcats_used: List[str] = sorted(set(basket) & set(xcats_avl))
    xcats_miss: List[str] = list(set(basket) - set(xcats_used))

    # if any missing cids, warn
    if xcats_miss:
        if complete_set:
            xcats_used = []
            err_str: str = (
                f"The cross section, {cid}, is missing {xcats_miss} which are included "
                f"in the basket {basket}. Therefore, the cross section will be excluded "
                "from the returned DataFrame."
            )
            print(err_str)
            warnings.warn(err_str, UserWarning)
        else:
            err_str: str = (
                f"The cross section, {cid}, is missing {xcats_miss} from the requested "
                f"basket. The new basket will be {xcats_used}."
            )
            print(err_str)
            warnings.warn(err_str, UserWarning)

    # Reduce the DataFrame to the specified basket given the available cross sections.
    dfb = df.loc[df["xcat"].isin(xcats_used)]

    return dfb, xcats_used


if __name__ == "__main__":
    # Simulate DataFrame.

    cids = ["AUD", "CAD", "GBP", "NZD"]
    xcats = ["XR1", "XR2", "CRY1", "CRY2"]
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
    df_xcats.loc["XR1"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["XR2"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["CRY1"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["CRY2"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Simulate blacklist
    black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    # Applications
    dfd_relative_xr = make_relative_category(
        dfd,
        xcats=["XR1", "XR2"],
        cids=None,
        blacklist=None,
        rel_meth="subtract",
        rel_xcats=None,
    )

    dfd_1_black = make_relative_category(
        dfd,
        xcats=["CRY1", "CRY2"],
        cids=None,
        blacklist=black,
        rel_meth="divide",
        rel_xcats=None,
        postfix="_RELRATIO",
    )
