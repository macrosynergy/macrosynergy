"""
Implementation of `make_relative_value()` function as a module. The function is used to
calculate values for indicators relative to a basket of cross sections.
"""

import pandas as pd
from typing import List, Set

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import reduce_df
from macrosynergy.management.types import QuantamentalDataFrame
import warnings


def make_relative_value(
    df: pd.DataFrame,
    xcats: List[str],
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    basket: List[str] = None,
    complete_cross: bool = False,
    rel_meth: str = "subtract",
    rel_xcats: List[str] = None,
    postfix: str = "R",
):
    """
    For each category specified in the panel, relative values are calculated
    by either subtracting or dividing by the cross-sectional mean at each `real_date`.

    Parameters
    ----------
    df : ~pandas.DataFrame
        standardized JPMaQS DataFrame with the necessary columns: 'cid', 'xcat',
        'real_date' and 'value'.
    xcats : List[str]
        all extended categories for which relative values are to be calculated.
    cids : List[str]
        cross sections for which relative values are calculated. Default is all cross 
        section available for the respective category.
    start : str
        earliest date in ISO format. Default is None and earliest date for which the
        respective category is available is used.
    end : str
        latest date in ISO format. Default is None and latest date for which the
        respective category is available is used.
    blacklist : dict
        cross sections with date ranges that should be excluded from the output.
    basket : List[str]
        cross sections to be used for the relative value benchmark. The default is every
        cross section in the `cids` argument that is available in the DataFrame over the
        respective time-period. However, the basket can be reduced to a valid subset of the
        available cross sections.
    complete_cross : bool
        boolean parameter that determines whether each category is required to have the
        full set of cross sections held by the basket parameter for a relative value
        calculation to occur. If set to True, the category will be excluded from the output
        if cross sections are missing. Default is False. If False, the mean, for the
        relative value, will use the subset that is available for that category. For
        instance, if `basket = ['AUD', 'CAD', 'GBP', 'NZD']` but available `cids = ['GBP',
        'NZD']`, the basket will be implicitly updated to basket = ['GBP', 'NZD'] for that
        respective category.
    rel_meth : str
        method for calculating relative value. Default is 'subtract'. Alternative is
        'divide'.
    rel_xcats : List[str]
        extended category name of the relative values. Will displace the original
        category names: xcat + postfix. The order should reflect the order of the passed
        categories.
    postfix : str
        acronym to be appended to `xcat` string to give the name for relative value
        category. Only applies if rel_xcats is None. Default is 'R'

    Returns
    -------
    ~pandas.DataFrame
        standardized DataFrame with the relative values, with the columns:
        'cid', 'xcat', 'real_date' and 'value'.
    """

    col_names = ["cid", "xcat", "real_date", "value"]

    operations = {
        "divide": pd.DataFrame.div,
        "subtract": pd.DataFrame.sub,
    }
    if rel_meth not in operations:
        raise ValueError("rel_meth must be 'subtract' or 'divide'")

    if not isinstance(xcats, (list, str)):
        raise TypeError("xcats must be a list of strings or a single string.")

    if isinstance(xcats, str):
        xcats = [xcats]

    if rel_xcats is not None:
        error_rel_xcat = "List of strings or single string expected for `rel_xcats`."
        if not isinstance(rel_xcats, (list, str)):
            raise TypeError(error_rel_xcat)

        if isinstance(rel_xcats, str):
            rel_xcats = [rel_xcats]

        if len(xcats) != len(rel_xcats):
            raise ValueError(
                "`rel_xcats` must have the same number of elements as `xcats`."
            )

        rel_xcats_dict = dict(zip(xcats, rel_xcats))

    df = QuantamentalDataFrame(df)
    # Intersect parameter set to False. Therefore, cross sections across the categories
    # can vary.
    all_cids: List[str] = []
    for cvar in [cids, basket]:
        if cvar is not None:
            all_cids.extend(cvar)
    all_cids = list(set(all_cids))
    if len(all_cids) < 1:
        all_cids = None
    dfx = reduce_df(df, xcats, all_cids, start, end, blacklist, out_all=False)

    if cids is None:
        # All cross sections available - union across categories.
        cids = list(dfx["cid"].unique())

    if basket is not None:
        # Basket must be a subset of the available cross sections.
        set_diff = set(basket).difference(set(dfx["cid"].unique()))
        if not (len(set_diff) == 0):
            raise ValueError(
                f"The basket elements {set_diff} are not available in the DataFrame."
            )
    else:
        # Default basket is all available cross sections.
        basket = cids

    available_xcats = dfx["xcat"].unique()

    if len(cids) == len(basket) == 1:
        run_error = (
            "Computing the relative value on a single cross section using a "
            "basket consisting exclusively of the aforementioned cross section "
            "is an incorrect usage of the function."
        )
        raise RuntimeError(run_error)

    df_list: List[pd.DataFrame] = []
    # Categories can be defined over a different set of cross sections.
    for i, xcat in enumerate(available_xcats):
        df_xcat = dfx[dfx["xcat"] == xcat]
        available_cids = df_xcat["cid"].unique()

        dfx_xcat: pd.DataFrame = df_xcat[["cid", "real_date", "value"]]

        dfb, basket = _prepare_basket(
            df=dfx_xcat,
            xcat=xcat,
            basket=basket,
            cids_avl=available_cids,
            complete_cross=complete_cross,
        )

        if len(basket) > 1:
            # Mean of (available) cross sections at each point in time. If all
            # cross sections defined in the "basket" data structure are not available for
            # a specific date, compute the mean over the available subset.
            bm = dfb.groupby(by="real_date").mean(numeric_only=True)
        elif len(basket) == 1:
            # Relative value is mapped against a single cross section.
            bm = dfb.set_index("real_date")[["value"]]
        else:
            # Category is not defined over all cross sections in the basket and
            # 'complete_cross' equals True.
            continue

        dfw: pd.DataFrame = dfx_xcat.pivot(
            index="real_date", columns="cid", values="value"
        )

        # Computing the relative value is only justified if the number of cross sections,
        # for the respective date, exceeds one. Therefore, if any rows have only a single
        # cross section, remove the dates from the DataFrame.
        dfw = dfw[dfw.count(axis=1) > 1]
        # The time-index will be delimited by the respective category.
        if isinstance(dfw.columns, pd.CategoricalIndex):
            if "value" not in dfw.columns.categories:
                dfw.columns = dfw.columns.add_categories(["value"])
            dfw["value"] = bm["value"]
            dfa = dfw
            dfw = dfw.drop("value", axis=1)
        else:
            dfa = pd.merge(dfw, bm, how="left", left_index=True, right_index=True)
        dfo: pd.DataFrame = operations[rel_meth](dfa[dfw.columns], dfa["value"], axis=0)

        # Re-stack.
        df_new = (
            dfo.stack().reset_index().rename({"level_1": "cid", 0: "value"}, axis=1)
        )

        if rel_xcats is None:
            df_new["xcat"] = xcat + postfix
        else:
            df_new["xcat"] = rel_xcats_dict[xcat]

        if (
            df_new.sort_values(["cid", "real_date"])[col_names].isna().all().all()
            or (len(df_new) == 0)
            or df_new.empty
        ):
            continue

        df_list.append(df_new.sort_values(["cid", "real_date"])[col_names])

    return QuantamentalDataFrame(
        pd.concat(df_list).reset_index(drop=True),
        categorical=df.InitializedAsCategorical,
    )


def _prepare_basket(
    df: pd.DataFrame,
    xcat: str,
    basket: List[str],
    cids_avl: List[str],
    complete_cross: bool,
):
    """
    Categories can be defined over different cross sections. Will determine the
    basket given the available cross sections for the respective category.

    Parameters
    ----------
    df : ~pandas.DataFrame
        long JPMaQS DataFrame of single category.
    xcat : str
        respective category for the relative value calculation.
    basket : pd.DataFrame
        cross sections to be used for the relative value benchmark if available.
    cids_avl : List[str]
        cross sections available for the category.
    complete_cross : bool
        if True, the basket is only calculated if all cross sections, held in the
        basket, are available for that respective category.

    Returns
    -------
    ~pandas.DataFrame
        DataFrame with the cross sections available for the respective category.
    List[str]
        List of cross sections available for the respective category.
    """

    cids_used: List[str] = sorted(set(basket) & set(cids_avl))
    cids_miss: List[str] = list(set(basket) - set(cids_used))

    # if any missing cids, warn
    if cids_miss:
        if complete_cross:
            cids_used = []
            err_str: str = (
                f"The category, {xcat}, is missing {cids_miss} which are included "
                f"in the basket {basket}. Therefore, the category will be excluded "
                "from the returned DataFrame."
            )
            warnings.warn(err_str, UserWarning)
        else:
            err_str: str = (
                f"The category, {xcat}, is missing {cids_miss} from the requested "
                f"basket. The new basket will be {cids_used}."
            )
            warnings.warn(err_str, UserWarning)

    # Reduce the DataFrame to the specified basket given the available cross sections.
    dfb = df[df["cid"].isin(cids_used)]

    return dfb, cids_used


if __name__ == "__main__":
    # Simulate DataFrame.

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

    # Simulate blacklist
    black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    # Applications
    dfd_1 = make_relative_value(
        dfd,
        xcats=["GROWTH"],
        cids=None,
        blacklist=None,
        rel_meth="subtract",
        rel_xcats=None,
        postfix="RV",
        basket=["AUD", "CAD"],
    )

    rel_xcats = ["GROWTH_sRV", "INFL_sRV"]
    dfd_2 = make_relative_value(
        dfd,
        xcats=["GROWTH", "INFL"],
        cids=None,
        blacklist=None,
        rel_meth="subtract",
        rel_xcats=rel_xcats,
        postfix="RV",
    )

    dfd_1_black = make_relative_value(
        dfd,
        xcats=["GROWTH", "INFL"],
        cids=None,
        blacklist=black,
        rel_meth="subtract",
        rel_xcats=rel_xcats,
        postfix="RV",
    )
