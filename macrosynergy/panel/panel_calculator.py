"""
Implementation of panel calculation functions for quantamental data. The functionality
allows applying mathematical operations on time-series data.
"""

import collections
import random
import re
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd

from macrosynergy import PYTHON_3_8_OR_LATER
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import drop_nan_series, reduce_df


def panel_calculator(
    df: pd.DataFrame,
    calcs: List[str] = None,
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    external_func: dict = None,
    sort_execution_order: bool = True,
    use_parallel: bool = True,
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
    sort_execution_order : bool
        if True, the function will sort the execution order of the calculations to
        minimize dependencies to ensure dependency resolution. If False, the function will
        use the order of the calculations as provided in the input. Default is True.
    use_parallel : bool
        if True, the function will create disjoint subgraphs of the calculations and
        distribute them across multiple processes for parallel execution. If False,
        the function will execute the calculations sequentially. Default is True.

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
    if use_parallel:
        return panel_calc_pll(
            df=df,
            calcs=calcs,
            cids=cids,
            start=start,
            end=end,
            blacklist=blacklist,
            external_func=external_func,
        )
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
    new_variables_existances: Dict[str, bool] = {}  # True where the variable exists
    df = df.add_ticker_column()
    data_map = {}
    for xcat in old_xcats_used:
        dfxx = dfx[dfx["xcat"] == xcat]
        dfw = dfxx.pivot(index="real_date", columns="cid", values="value")
        dfw = _replace_zeros(df=dfw)
        data_map[xcat] = dfw
        new_variables_existances[xcat] = not dfw.empty

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
            new_variables_existances[single] = not dfw.empty

    if sort_execution_order:
        ops_tuples = sort_execution_order_func(ops, new_variables_existances)
        first_op_lhs = ops_tuples[0][0]
    else:
        ops_tuples = list(ops.items())
        first_op_lhs = ops_tuples[0][0]

    # F. Calculate the panels and collect.
    df_out: pd.DataFrame
    for new_xcat, formula in ops_tuples:
        dfw_add = eval(formula, safe_globals, data_map)
        df_add = pd.melt(dfw_add.reset_index(), id_vars=["real_date"]).rename(
            {"variable": "cid"}, axis=1
        )
        df_add = QuantamentalDataFrame.from_long_df(df_add, xcat=new_xcat)
        if new_xcat == first_op_lhs:
            df_out = df_add[cols]
        else:
            df_out = pd.concat([df_out, df_add[cols]], axis=0, ignore_index=True)
        dfw_add = _replace_zeros(df=dfw_add)
        data_map[new_xcat] = dfw_add

    if df_out.isna().any().any():
        df_out = drop_nan_series(df=df_out, raise_warning=True)

    df_out = QuantamentalDataFrame(df_out, categorical=_as_categorical)
    return df_out


def panel_calc_pll(
    df: pd.DataFrame,
    calcs: List[str] = None,
    cids: List[str] = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    external_func: dict = None,
):
    ops = {}
    for calc in calcs:
        calc_parts = calc.split("=", maxsplit=1)
        ops[calc_parts[0].strip()] = calc_parts[1].strip()

    old_xcats_used, singles_used, single_cids = _get_xcats_used(ops)
    avail = set(df["xcat"]) & set(old_xcats_used)
    results = joblib.Parallel(n_jobs=4)(
        joblib.delayed(panel_calculator)(
            df,
            list(map(lambda x: x.formula, _cl)),
            cids=cids,
            external_func=external_func,
            start=start,
            end=end,
            blacklist=blacklist,
            sort_execution_order=True,
            use_parallel=False,
        )
        for _cl in CalcList(calcs, avail).get_independent_subgraphs()
    )
    return QuantamentalDataFrame.from_qdf_list(results)


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

    def check(a: str, b: str, c: str) -> bool:
        return (a.isupper() or a.isnumeric()) and b == "." and c.islower()

    f = formula
    length = len(f)
    clause = False
    for i in range(index, (length - 2)):
        if check(f[i], f[i + 1], f[i + 2]):
            clause = True
            break

    return i, clause


def xcat_isolator(expression: str, start_index: str, index: int) -> Tuple[str, int]:
    """
    Split the category from the time-series operation. The function will return the
    respective category.

    Parameters
    ----------
    expression : str

    start_index : str
        starting index to search over.
    index : int
        defines the end of the search space over the expression.

    Returns
    -------
    Tuple[str, int]
        xcat string, and the string index where the xcat ends.
    """

    op_copy = expression[start_index : index + 1]

    start = next(i for i, elem in enumerate(op_copy) if elem.isupper())

    xcat = op_copy[start : index + 1]

    return xcat, start_index + start + len(xcat)


def _get_xcats_used(ops: dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Collect all categories used in the panel calculation.

    Parameters
    ----------
    ops : dict
        dictionary of panel calculation formulas.

    Returns
    -------
    Tuple[List[str], List[str]]
        all_xcats_used, singles_used.
    """

    xcats_used: List[str] = []
    singles_used: List[str] = []
    for op in ops.values():
        index, clause = time_series_check(formula=op, index=0)
        start_index = 0
        if clause:
            while clause:
                xcat, end_ = xcat_isolator(op, start_index, index)
                xcats_used.append(xcat)
                index, clause = time_series_check(op, index=end_)
                start_index = end_
        else:
            op_list = op.split(" ")
            xcats_used += [x for x in op_list if re.match("^[A-Z]", x)]
            singles_used += [s for s in op_list if re.match("^i", s)]

    single_xcats = [x.split("_", 1)[1] for x in singles_used]
    single_cids = [x.split("_", 1)[0] for x in single_xcats]
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


def sort_execution_order_func(
    ops: Dict[str, Set],
    new_variables_existances: Dict[str, bool],
) -> List[Tuple[str, str]]:
    formulas = [f"{k} = {v}" for k, v in ops.items()]
    existing_vars = [k for k, v in new_variables_existances.items() if v]
    sorted_calcs = CalcList(calcs=formulas, already_existing_vars=existing_vars).calcs
    ops = {calc.lhs: calc.rhs for calc in sorted_calcs}
    return list(ops.items())


class SingleCalc:
    """
    Class to represent a single calculation.
    """

    formula: str
    lhs: str
    rhs: str
    dct: dict[str, str]

    def __init__(self, formula: str) -> None:
        self.formula = formula
        self.lhs, self.rhs = [_.strip() for _ in formula.split(" = ", maxsplit=1)]
        self.dct = {self.lhs: self.rhs}

    def dependencies(self) -> List[str]:
        rhs = self.rhs
        tokens: List[str] = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", rhs)
        exclude = set([self.lhs, "np", "pd"])
        deps = [t for t in tokens if t.isupper() and t not in exclude]
        return sorted(set(deps))

    def creates(self) -> str:
        return self.lhs

    def __repr__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def __str__(self) -> str:
        return self.__repr__()


class CalcList:
    """
    Manages a collection of SingleCalc formulas, identifies which are feasible
    from the existing variables, and groups them into truly independent subgraphs.
    Also provides a topologically-sorted list (self.calcs) of all feasible calculations.
    """

    def __init__(self, calcs: List[str], already_existing_vars: List[str]):
        self.all_calcs = [SingleCalc(c) for c in calcs]
        self.already_existing_vars = set(already_existing_vars)

        # Identify which calculations are feasible from the existing variables
        self.feasible_calcs = self._find_feasible_calcs()

        # Build an undirected graph over only the feasible calculations
        self.graph = self._build_graph(self.feasible_calcs)

        # For convenience, also store a topologically-sorted list of all feasible calculations in self.calcs
        self.calcs = self._sort_calculations()

    def _find_feasible_calcs(self) -> List[SingleCalc]:
        """
        Returns only those SingleCalc objects whose dependencies
        can eventually be satisfied starting from self.already_existing_vars.
        """
        known_vars = set(self.already_existing_vars)
        feasible = []
        calcs_remaining = set(self.all_calcs)  # Make a copy

        # We'll keep picking off feasible calculations until no more can be found
        progress = True
        while progress:
            progress = False
            for calc in list(calcs_remaining):  # copy to iterate
                deps = calc.dependencies()
                # If all dependencies are known, this calc is feasible
                if all(d in known_vars for d in deps):
                    feasible.append(calc)
                    known_vars.add(calc.creates())
                    calcs_remaining.remove(calc)
                    progress = True

        # At the end, anything left in calcs_remaining has dependencies
        # that can't be satisfied from the known_vars, so it is not feasible.
        return feasible

    def _sort_calculations(self) -> List[SingleCalc]:
        """
        Topologically sort the feasible calculations so that
        each calculation's dependencies are created prior to it.
        """
        sorted_calcs = []
        known_vars = set(self.already_existing_vars)
        remaining = self.feasible_calcs[:]
        cyc_err = "Cyclic or unresolvable dependencies in calculations."
        while remaining:
            # find all calcs whose dependencies are satisfied
            placeable: List[SingleCalc] = [
                calc
                for calc in remaining
                if all(dep in known_vars for dep in calc.dependencies())
            ]
            if not placeable:
                raise ValueError(cyc_err)

            # We add them in the order we encounter them
            # sort is only done to make it strictly deterministic
            for calc in sorted(placeable, key=lambda x: x.creates()):
                sorted_calcs.append(calc)
                known_vars.add(calc.creates())
                remaining.remove(calc)

        return sorted_calcs

    def _build_graph(self, calcs: List[SingleCalc]) -> Dict[int, Set[int]]:
        """
        Build an undirected graph of feasible calculations for finding
        connected components (truly independent subgraphs).
        We'll store a mapping: calc_index -> set_of_calc_indices_that_are_connected
        """
        idx_map = {calc: i for i, calc in enumerate(calcs)}
        graph = collections.defaultdict(set)

        for i, cA in enumerate(calcs):
            outA = cA.creates()
            for j, cB in enumerate(calcs):
                if i == j:
                    continue
                # if cB depends on cA's output, or cA depends on cB's output,
                # they are in the same connected component
                if outA in cB.dependencies():
                    graph[i].add(j)
                    graph[j].add(i)

        return dict(graph)

    def get_independent_subgraphs(self) -> List[List[SingleCalc]]:
        """
        Returns a list of subgraphs (each subgraph is a list of SingleCalc),
        where each subgraph is truly independent from the others.
        These are just the connected components of the feasible graph.
        """
        visited = set()
        subgraphs = []
        calcs = self.feasible_calcs
        idx_map = {i: calcs[i] for i in range(len(calcs))}

        for i in range(len(calcs)):
            if i not in visited:
                # BFS/DFS to get all connected nodes
                component = []
                queue = collections.deque([i])
                visited.add(i)
                while queue:
                    node = queue.popleft()
                    component.append(idx_map[node])
                    # For each neighbor in the undirected graph:
                    for neighbor in self.graph.get(node, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                subgraphs.append(component)

        return subgraphs

    def get_subgraph_parallel_blocks(
        self, subgraph: List[SingleCalc]
    ) -> List[List[SingleCalc]]:
        """
        For a single subgraph (list of SingleCalc), return parallel "layers"
        by topological level, i.e. each layer can be executed in parallel.
        """
        remaining = subgraph[:]
        known_vars = set(self.already_existing_vars)
        blocks = []

        placed = True
        while placed and remaining:
            placed = False
            this_block = []
            for calc in list(remaining):
                deps = calc.dependencies()
                if all(d in known_vars for d in deps):
                    this_block.append(calc)
            if this_block:
                for calc in this_block:
                    remaining.remove(calc)
                    known_vars.add(calc.creates())
                blocks.append(this_block)
                placed = True

        if remaining:
            # Something is unplaced => cyclical or incomplete dependencies
            raise ValueError(
                "Cannot form parallel layers. Possibly a cycle in subgraph."
            )

        return blocks

    def __repr__(self):
        feasible_formulas = "\n".join(str(c) for c in self.feasible_calcs)
        sorted_formulas = "\n".join(str(c) for c in self.calcs)
        return (
            f"Feasible calculations:\n{feasible_formulas}\n\n"
            f"Topologically-sorted feasible calculations (CalcList.calcs):\n{sorted_formulas}\n"
        )


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
