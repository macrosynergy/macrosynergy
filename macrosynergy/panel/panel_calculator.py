"""
Implementation of panel calculation functions for quantamental data. The functionality
allows applying mathematical operations on time-series data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import reduce_df, drop_nan_series, get_cid, get_xcat
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy import PYTHON_3_8_OR_LATER
import re
import random
import string


def panel_calculator(
    df: pd.DataFrame,
    calcs: List[str] = None,
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    external_func: dict = {},
) -> pd.DataFrame:
    """
    Calculates new data panels through a given input formula which is performed on
    existing panels.

    Parameters
    ----------
    df : ~pandas.Dataframe
        standardized dataframe with following necessary columns: 'cid', 'xcat',
        'real_date' and 'value'.
    calcs : List[str]
        list of formulas denoting operations on panels of categories. Words in capital
        letters denote category panels. Otherwise the formulas can include numpy functions
        and standard binary operators. See notes below.
    cids : List[str]
        cross sections over which the panels are defined.
    start : str
        earliest date in ISO format. Default is None and earliest date in df is used.
    end : str
        latest date in ISO format. Default is None and latest date in df is used.
    blacklist : dict
        cross sections with date ranges that should be excluded from the dataframe. If
        one cross section has several blacklist periods append numbers to the cross-section
        code.
    external_func : dict
        dictionary of external functions to be used in the panel calculation. The key is
        the name of the function and the value is the function object itself. e.g.
        {"my_func": my_func}.

    Returns
    -------
    ~pandas.Dataframe
        standardized dataframe with all new categories in standard format, i.e the
        columns 'cid', 'xcat', 'real_date' and 'value'.


    Notes
    -----
    Panel calculation strings can use numpy functions and unary/binary operators on
    category panels. The category is indicated by capital letters, underscores and
    numbers. Panel category names that are not at the beginning or end of the string
    must always have a space before and after the name. Calculated category and
    panel operations must be separated by '='.

    Examples:

    .. code-block:: python

        NEWCAT = ( OLDCAT1 + 0.5) * OLDCAT2
    or

    .. code-block:: python

        NEWCAT = np.log( OLDCAT1 ) - np.abs( OLDCAT2 ) ** 1/2

    Panel calculation can also involve individual indicator
    series (to be applied to all series in the panel by using th 'i' as prefix), such
    as:

    .. code-block:: python

        NEWCAT = OLDCAT1 - np.sqrt( iUSD_OLDCAT2 )

    These strings are passed as a list of strings (`calcs`) to the function.

    If more than one new category is calculated, the resulting panels can be used
    sequentially in the calculations, such as:
    .. code-block:: python

        ["NEWCAT1 = 1 + OLDCAT1 / 100", "NEWCAT2 = OLDCAT2 * NEWCAT1"]

    .. code-block:: python

        calcs = [
            "NEWCAT = OLDCAT1 + OLDCAT2",
            "NEWCAT2 = CAT_A * CAT_B - CAT_C * 0.5",
            "NEWCAT3 = OLDCAT1 - np.sqrt(iUSD_OLDCAT2)",
        ]

        df = panel_calculator(df=df, calcs=calcs, ...)
    """

    # A. Asserts

    cols = ["cid", "xcat", "real_date", "value"]

    col_error = f"The DataFrame must contain the necessary columns: {cols}."
    assert set(cols).issubset(set(df.columns)), col_error
    # Removes any columns beyond the required.
    df = QuantamentalDataFrame(df[cols])
    _as_categorical = df.InitializedAsCategorical
    assert isinstance(calcs, list), "List of functions expected."

    error_formula = "Each formula in the panel calculation list must be a string."
    assert all([isinstance(elem, str) for elem in calcs]), error_formula
    assert isinstance(cids, list), "List of cross-sections expected."

    _check_calcs(calcs)

    safe_globals = {"np": np, "pd": pd, **external_func}

    # B. Collect new category names and their formulas.

    ops = {}
    for calc in calcs:
        calc_parts = calc.split("=", maxsplit=1)
        ops[calc_parts[0].strip()] = calc_parts[1].strip()

    # C. Check if all required categories are in the dataframe.

    old_xcats_used, singles_used, single_cids = _get_xcats_used(ops)

    old_xcats_used = list(set(old_xcats_used))
    missing = sorted(set(old_xcats_used) - set(df["xcat"].unique()))

    new_xcats = list(ops.keys())
    if len(missing) > 0 and not set(missing).issubset(set(new_xcats)):
        raise ValueError(f"Missing categories: {missing}.")

    # If any of the elements of single_cids are not in cids, add them to cids.
    cids_used = list(set(single_cids + cids))

    # D. Reduce dataframe with intersection requirement.

    dfx = reduce_df(
        df,
        xcats=old_xcats_used,
        cids=cids_used,
        start=start,
        end=end,
        blacklist=blacklist,
        intersect=False,
    )

    # E. Create all required wide dataframes with category names.
    df = df.add_ticker_column()
    data_map = {}
    for xcat in old_xcats_used:
        dfxx = dfx[dfx["xcat"] == xcat]
        dfw = dfxx.pivot(index="real_date", columns="cid", values="value")
        dfw = _replace_zeros(df=dfw)
        data_map[xcat] = dfw

    for single in singles_used:
        ticker = single[1:]
        dfxx = df[(df["ticker"]) == ticker]
        if dfxx.empty:
            raise ValueError(f"Ticker, {ticker}, missing from the dataframe.")
        else:
            dfx1 = dfxx.set_index("real_date")["value"].to_frame()
            dfx1 = dfx1.truncate(before=start, after=end)

            dfw = pd.concat([dfx1] * len(cids), axis=1, ignore_index=True)
            dfw.columns = cids
            dfw = _replace_zeros(df=dfw)
            data_map[single] = dfw

    # F. Calculate the panels and collect.
    df_out: pd.DataFrame
    for new_xcat, formula in ops.items():
        dfw_add = eval(formula, safe_globals, data_map)
        df_add = pd.melt(dfw_add.reset_index(), id_vars=["real_date"]).rename(
            {"variable": "cid"}, axis=1
        )
        df_add = QuantamentalDataFrame.from_long_df(df_add, xcat=new_xcat)
        if new_xcat == list(ops.keys())[0]:
            df_out = df_add[cols]
        else:
            df_out = pd.concat([df_out, df_add[cols]], axis=0, ignore_index=True)
        dfw_add = _replace_zeros(df=dfw_add)
        data_map[new_xcat] = dfw_add

    if df_out.isna().any().any():
        df_out = drop_nan_series(df=df_out, raise_warning=True)

    df_out = QuantamentalDataFrame(df_out, categorical=_as_categorical)
    return df_out


def time_series_check(formula: str, index: int):
    """
    Determine if the panel has any time-series methods applied. If a time-series
    conversion is applied, the function will return the terminal index of the respective
    category. Further, a boolean parameter is also returned to confirm the presence of a
    time-series operation.

    Parameters
    ----------
    formula : str

    index : int
        starting index to iterate over.

    Returns
    -------
    Tuple[int, bool]
    """

    check = lambda a, b, c: (
        (a.isupper() or a.isnumeric()) and b == "." and c.islower()
    )

    f = formula
    length = len(f)
    clause = False
    for i in range(index, (length - 2)):
        if check(f[i], f[i + 1], f[i + 2]):
            clause = True
            break

    return i, clause


def is_valid_xcat(xcat_str: str) -> bool:
    """
    Heuristic to determine if a string is a valid category (`xcat`).
    Conditions:
        - Only composed of alphanumeric characters and underscores
        - Must contain at least one uppercase letter
        - If starts with "i", must be a ticker, i.e containing an underscore

    Parameters
    ----------
    xcat_str : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a valid category (`xcat`), False otherwise.
    """
    xcat_chars = string.ascii_letters + string.digits + "_"
    if xcat_str.startswith("i"):
        if (get_cid(xcat_str) + "_" + get_xcat(xcat_str)) != xcat_str:
            return False
    if len(set(xcat_str) - set(xcat_chars)) > 0:
        return False
    if not any(c in string.ascii_uppercase for c in xcat_str):
        return False
    return True


def xcat_isolator(calc_rhs_str: str) -> List[str]:
    """
    Split the category from the right hand side (RHS) of the panel calculation formula.
    The function will return a list of categories found in the RHS string.

    Parameters
    ----------
    calc_rhs_str : str
        right hand side of the panel calculation formula.
    """
    xcat_chars = string.ascii_letters + string.digits + "_"
    # keep original calc_rhs_str for for later
    rhs_chars = [c for c in calc_rhs_str if not c.isspace()]
    mask = [c in xcat_chars for c in rhs_chars]

    found_xcats_with_pos = []
    current = ""
    start_idx = None

    for ic, char in enumerate(rhs_chars):
        if mask[ic]:
            if current == "":
                start_idx = ic
            current += char
        elif current != "":
            found_xcats_with_pos.append((current, start_idx))
            current = ""
            start_idx = None

    if current != "":
        found_xcats_with_pos.append((current, start_idx))

    found_xcats: List[str] = []
    n = len(rhs_chars)

    for xcat, start in found_xcats_with_pos:
        if start is None:
            continue

        end = start + len(xcat) - 1

        has_eq_left = start > 0 and rhs_chars[start - 1] == "="
        has_eq_right = end < n - 1 and rhs_chars[end + 1] == "="

        if has_eq_left or has_eq_right:
            continue

        if is_valid_xcat(xcat):
            found_xcats.append(xcat)

    if not found_xcats:
        raise ValueError(
            "This calculation does not contain any valid categories (XCATs).\n\t:"
            f"{calc_rhs_str}"
        )

    return found_xcats


def _get_xcats_used(ops: dict) -> Tuple[List[str], List[str]]:
    """
    Collect all categories used in the panel calculation.

    Parameters
    ----------
    ops : dict
        dictionary of panel calculation formulas.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        all_xcats_used, singles_used, single_cids
    """

    xcats_used: List[str] = []
    singles_used: List[str] = []
    for op in ops.values():
        xcats_found = xcat_isolator(op)
        new_single_tickers = [x for x in xcats_found if x.startswith("i")]
        new_xcats_used = [x for x in xcats_found if x not in new_single_tickers]

        singles_used += new_single_tickers
        xcats_used += new_xcats_used

    single_xcats = [get_xcat(x) for x in singles_used]
    single_cids = [get_cid(x) for x in singles_used]

    # removing the "i" prefix from single_cids
    single_cids = [x.lstrip("i") for x in single_cids]

    all_xcats_used = xcats_used + single_xcats
    return all_xcats_used, singles_used, single_cids


def _check_calcs(formulas: List[str]):
    """
    Check formulas for invalid characters in xcats.

    Parameters
    ----------
    calcs : List[str]
        list of formulas.

    Returns
    -------
    List[str]
        list of formulas.
    """

    pattern = r"[-+*()/](?=i?[A-Z])|(?<=[A-Z])[-+*()/]"

    for formula in formulas:
        for term in formula.split():
            # Search for any occurrences of the pattern in the input string
            if re.search(pattern, term):
                raise ValueError(
                    f"Invalid character found next to a capital letter or 'i' in string: {term}. "
                    + "Arithmetic operators and parentheses must be separated by spaces."
                )


def _replace_zeros(df: pd.DataFrame):
    """
    Replace zeros with NaNs in the dataframe.

    Parameters
    ----------
    df : ~pandas.DataFrame
        dataframe to be cleaned.

    Returns
    -------
    ~pandas.DataFrame
        cleaned dataframe.
    """

    if not PYTHON_3_8_OR_LATER:  # pragma: no cover
        for col in df.columns:
            df[col] = df[col].replace(pd.NA, np.nan)
            df[col] = df[col].astype("float64")
        return df
    else:
        return df

    return df


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "USD", "NZD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]

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
    df_xcats.loc["XR"] = ["2012-01-01", "2020-12-31", 0, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2012-01-01", "2020-09-30", 1, 2, 0.8, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Example blacklist.
    black = {"AUD": ["2000-01-01", "2003-12-31"]}

    start = "2010-01-01"
    end = "2020-12-31"

    # Example filter for dataframe.
    filt1 = (dfd["xcat"] == "XR") | (dfd["xcat"] == "CRY")
    dfdx = dfd[filt1]

    # First testcase.

    f1 = "NEW_VAR1 = GROWTH - iEUR_INFL"
    formulas = [f1]
    cidx = ["AUD", "CAD"]
    df_calc = panel_calculator(
        df=dfd, calcs=formulas, cids=cidx, start=start, end=end, blacklist=black
    )
    # Second testcase: EUR is not passed in as one of the cross-sections in "cids"
    # parameter but is defined in the dataframe. Therefore, code will not break.
    cids = ["AUD", "CAD", "GBP", "USD", "NZD"]
    formula = "NEW1 = XR - iUSD_XR"
    formula_2 = "NEW2 = GROWTH - iEUR_INFL"
    formulas = [formula, formula_2]
    df_calc = panel_calculator(
        df=dfd, calcs=formulas, cids=cids, start=start, end=end, blacklist=black
    )
