import numpy as np
import pandas as pd
from typing import List
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
import re
import random
from collections import OrderedDict
from itertools import repeat


def symbol_finder(expression: str, index: int = 0):
    """
    Method used to understand if the category involved in the expression only concerns a
    single cross-section.

    :param <str> expression: associated formula.
    :param <int> index: variable used to track the index of the expression.

    :return <List[int]> list of indices where the "@" symbol occurs.
    """

    indices = []
    adjustment = 0

    while index != -1:

        index = expression.find("@")
        if index != -1:
            indices.append(index + adjustment)
            adjustment = (index + 1) + adjustment

            index += 1
            expression = expression[index:]

    return indices


def iterator_func(expression: str, symbol: str, index: int):
    """
    Indefinite iterator until a specific symbol is found. Will return the index of the
    respective symbol.

    :param <str> expression:
    :param <str> symbol:
    :param <int> index: starting index of the iterator.

    :return <int>: index of the associated symbol.
    """

    elem = expression[index]
    iterator = index
    while elem != symbol:
        iterator += 1
        elem = expression[iterator]

    return iterator


def formula_reconstruction(formula: str, indices: List[int]):
    """
    Reconstruct the formula to handle single cross-sections. The purpose is to return the
    formula but with the removal of the single cross-section declaration and instead
    leave the xcat. For instance, "@USD_GROWTH" will be reduced to "GROWTH". The
    originally referenced cross-section will be stored in a separate data structure and
    incorporated in the formula in a different capacity.

    :param <str> formula: "((@USD_GROWTH - np.sqrt( @USD_INFL )) / @USD_XR )"
    :param <List[int]> indices: list of indices where the "@" symbol occurs in the
        expression.

    :return <str, dict>: returns the updated formula and a tracking mechanism for the
        cross-sections.
    """

    cid_tracker = {}
    xcat_tracker = {}

    for index in indices:

        iterator = iterator_func(formula, "_", index)
        cid_tracker[index] = formula[(index + 1): iterator]

        start = iterator
        end = iterator_func(formula, " ", index=iterator)

        xcat_tracker[(start + 1)] = formula[(start + 1): end]

    reconstruction = tuple(zip(cid_tracker.keys(), xcat_tracker.keys()))
    adjustment = 0
    for tup in reconstruction:

        xcat_start = tup[1] - adjustment
        cid_start = tup[0] - adjustment

        value = cid_tracker[tup[0]]
        cid_tracker.pop(tup[0])
        cid_tracker[(tup[0] - adjustment)] = value

        adjustment += (xcat_start - cid_start)
        formula = formula[:cid_start] + formula[xcat_start:]

    return formula, cid_tracker


def formula_handler(calcs: List[str]):
    """
    Separate the functions, contained in the list, y = f(x), on the equality sign. The
    codomain will be the key and the function will be the value.

    :param <List[str]> calcs:

    :return <dict, dict>: dictionary hosting the function.
    """

    ops = {}
    expression_cid = {}
    for i, calc in enumerate(calcs):

        calc_parts = calc.split('=', maxsplit=1)
        # Suppress to imperative inclusion of terminal parenthesis.
        value = calc_parts[1].strip()
        indices = symbol_finder(expression=value, index=0)

        if indices:
            value, cid_dict = formula_reconstruction(formula=value, indices=indices)
            expression_cid[i] = cid_dict

        ops[calc_parts[0].strip()] = value

    return ops, expression_cid

def time_series_check(formula: str, index: int):
    """
    Determine if the panel has any time-series methods applied. If a time-series
    conversion is applied, the function will return the terminal index of the respective
    category. Further, a boolean parameter is also returned to confirm the presence of a
    time-series operation.

    :param <str> formula:
    :param <int> index: starting index to iterate over.

    :return <int, bool>:
    """

    check = lambda a, b, c: (a.isupper() and b == "." and c.islower())

    f = formula
    length = len(f)
    clause = False
    for i in range(index, (length - 2)):
        if check(f[i], f[i + 1], f[i + 2]):
            clause = True
            break
        else:
            continue

    return i, clause

def xcat_isolator(expression: str, start_index: str, index: int):
    """
    Split the category from the time-series operation. The function will return the
    respective category.

    :param <str> expression:
    :param <str> start_index: starting index to search over.
    :param <int> index: defines the end of the search space over the expression.

    :return <str> xcat.
    """

    op_copy = expression[start_index:index + 1]

    start = 0
    elem = op_copy[start_index]
    while not elem.isupper():
        start += 1
        elem = op_copy[start]

    xcat = op_copy[start:(index + 1)]

    return xcat, (start_index + start + len(xcat))

def involved_xcats(ops: dict):
    """
    Function used to understand the original categories involved in the panel
    calculations. To isolate the involved categories in the expression, spaces are
    required either side of the category. For instance, NEWCAT1 = np.abs( XR ).
    Further, each category will exclusively involve capital letters which aids
    determining the categories in the string expression.

    :param <dict> ops: dictionary containing the panel calculation where the key is the
        newly formed category and the value the calculation.

    :return <set> xcats_used: unique categories referenced across all formulas.
    """

    xcats_used = []
    new_xcats = list(ops.keys())

    for op in ops.values():
        index, clause = time_series_check(formula=op, index=0)
        start_index = 0
        if clause:
            while clause:
                xcat, end = xcat_isolator(op, start_index, index)
                xcats_used.append(xcat)
                index, clause = time_series_check(op, index=end)
                start_index = end
        else:
            op_list = op.split(' ')
            xcats_used += [x for x in op_list if re.match('^[A-Z]', x)
                           and x not in new_xcats]

    return set(xcats_used)


def pandas_alignment(dates_dict: dict, expression: str):
    """
    Function designed to return the latest start date and earliest end date across the
    categories involved in the expression. Reduce all dataframes to be defined over the
    same time-period.

    :param <dict> dates_dict:
    :param <str> expression:

    return <pd.Timestamp>:
    """

    cats_tuple = []
    cats_indices = {}
    for k in dates_dict.keys():

        pattern = re.compile(k)
        indices = pattern.finditer(expression)
        result = [match.span() for match in indices]

        if result:
            no = len(result)
            if no > 1:
                xcat_l = repeat(k, no)
                tuple_ = list(zip(xcat_l, result))
                cats_tuple += tuple_
            else:
                cats_tuple.append((k, next(iter(result))))
            cats_indices[k] = result
        else:
            continue

    s_date = pd.Timestamp.min
    e_date = pd.Timestamp.max

    for k in cats_indices.keys():
        if dates_dict[k][0] > s_date:
            s_date = dates_dict[k][0]
        if dates_dict[k][1] < e_date:
            e_date = dates_dict[k][1]

    return s_date, e_date, cats_indices, cats_tuple


def category_order(cats_indices: List[tuple]):
    """
    Order the List according to the indices of the respective categories. The
    categories occurring earliest in the expression will account for the first elements
    in the returned List. Utilises a polynomial sorting algorithm called Insertion Sort.

    :param <List[tuple]> cats_indices:

    return <List[tuple]> cats_indices: ordered list.
    """

    no_xcats = len(cats_indices)
    for i in range(1, no_xcats):
        current = cats_indices[i]
        j = (i - 1)

        while j >= 0 and cats_indices[j][1][0] > current[1][0]:
            cats_indices[j + 1] = cats_indices[j]
            j -= 1
        cats_indices[j + 1] = current

    return cats_indices

def cid_append(expression: str, index_cid: dict, dates_dict: dict):
    """
    Subroutine designed to modify the formula to account for the presence of single
    cross-sections on certain categories. For instance, np.sqrt(@USD_OLDCAT2).
    Further, will align the involved dataframes which is required due to the conversion
    to a np.ndarray.

    :param <str> expression:
    :param <dict> index_cid: the dictionary's key will be the concerning category's
        starting index and the value will be the relevant cross-section.
    :param <dict> dates_dict: dictionary consisting of each category and their respective
        start and end date across the panel series.

    :return <str> expression: updated formula.
    """
    global s_date, e_date

    index_cid = OrderedDict(sorted(index_cid.items()))

    add_length = 0
    for k, v in index_cid.items():

        cat_index = (k + add_length)
        end_cat = iterator_func(expression, symbol=" ", index=cat_index)

        addition = "['" + v + "'].to_numpy()[:, np.newaxis]"
        expression = expression[:end_cat] + addition + expression[end_cat:]

        add_length += len(addition)

    s_date, e_date, cats_indices, cats_tuple = pandas_alignment(dates_dict, expression)
    cats_tuple = category_order(cats_tuple)

    trunc = ".loc[s_date:e_date]"
    length = 0
    counter = 0
    for elem in cats_tuple:
        indices = elem[1]
        index = indices[1] + (length * counter)
        expression = expression[:index] + trunc + expression[index:]
        counter += 1
        length = len(trunc) * counter

    return expression


def panel_calculator(df: pd.DataFrame, calcs: List[str] = None, cids: List[str] = None,
                     start: str = None, end: str = None,
                     blacklist: dict = None):
    """
    Calculates new data panels through operations on existing panels.

    :param <pd.Dataframe> df: standardized dataframe with following necessary columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> calcs:  list of formulas denoting operations on panels of
        categories. Words in capital letters denote category panels.
        Otherwise the formulas can include numpy functions and standard binary operators.
        See notes below.
    :param <List[str]> cids: cross sections over which the panels are defined.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in
        df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is
        used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the dataframe. If one cross section has several blacklist periods append numbers
        to the cross-section code.

    :return <pd.Dataframe>: standardized dataframe with all new categories in standard
        format, i.e the columns 'cid', 'xcat', 'real_date' and 'value'.

    Notes:
    Panel calculation strings can use numpy functions and unary/binary operators go
    category panels, whereby the category is indicated by capital letters, underscores
    and numbers.
    Calculated category and panel operations must be separated by '='.
        "NEWCAT = (OLDCAT1 + 0.5) * OLDCAT2"
        "NEWCAT = np.log(OLDCAT1) - np.abs(OLDCAT2) ** 1/2"
    Panel calculation can also involve individual indicator series (to be applied
    to all series in the corresponding panel by using the @ as prefix), such as:
        "NEWCAT = OLDCAT1 - np.sqrt(@USD_OLDCAT2)"
    If more than one new category is calculated, the resulting panels can be used
    sequentially in the calculations, such as:
    ["NEWCAT1 = 1 + OLDCAT1/100", "NEWCAT2 = OLDCAT2 * NEWCAT1"]

    """

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert set(cols).issubset(set(df.columns))
    assert isinstance(calcs, list), "List of functions expected."
    assert all([isinstance(elem, str) for elem in calcs]), \
        "Each formula in the panel calculation list must be a string."
    assert isinstance(cids, list), "List of cross-sections expected."

    ops, expression_cid = formula_handler(calcs)

    old_xcats_used = involved_xcats(ops=ops)

    available_xcats = set(df['xcat'].unique())
    missing = sorted(old_xcats_used - available_xcats)
    assert len(missing) == 0, f"Missing categories: {missing}"

    # Reduce the dataframe to the cross-sections available in all categories.
    dfx = reduce_df(df, xcats=list(old_xcats_used), cids=cids, start=start,
                    end=end, blacklist=blacklist, intersect=True)

    dates_xcat = {}
    for xcat in old_xcats_used:
        dfxx = dfx[dfx['xcat'] == xcat]
        dfw = dfxx.pivot(index='real_date', columns='cid', values='value')
        dates_xcat[xcat] = (dfw.index[0], dfw.index[-1])

        exec(f"{xcat} = dfw")

    output_df = []
    index = 0
    for new_xcat, formula in ops.items():

        if index in expression_cid.keys():
            formula = cid_append(formula, index_cid=expression_cid[index],
                                 dates_dict=dates_xcat)
        dfw_add = eval(formula)
        df_add = pd.melt(dfw_add.reset_index(), id_vars=['real_date'])
        df_add['xcat'] = new_xcat
        output_df.append(df_add)
        exec(f'{new_xcat} = dfw_add')
        index += 1

        # Update the dates dictionary to include the newly formed category.
        dates_xcat[new_xcat] = (dfw_add.index[0], dfw_add.index[-1])

    df_calc = pd.concat(output_df)[cols]
    df_calc.reset_index(drop=True)

    return df_calc


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2010-01-01', '2020-12-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2010-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    start = '2010-01-01'
    end = '2020-12-31'

    # Start of the testing. Various testcases included to understand the capabilities of
    # the designed function.

    formula_3 = "NEW3 = GROWTH - np.square( @USD_INFL )"
    formulas = ["NEW1 = np.abs( XR ) + 0.552 + 2 * CRY", "NEW2 = NEW1 * 2", formula_3]
    df_calc = panel_calculator(df=dfd, calcs=formulas, cids=cids, start=start, end=end)

    # Secondary testcase.
    formula = "NEW1 = (( GROWTH - np.square( @USD_INFL )) / @USD_INFL )"
    formulas = [formula]
    df_calc = panel_calculator(df=dfd, calcs=formulas, cids=cids, start=start, end=end)

    # Third testcase. Referencing the newly formed category.
    formula = "NEW1 = ( GROWTH * @USD_INFL )"
    formula_2 = "NEW2 = ( XR - @USD_GROWTH )"
    formulas = [formula, formula_2]
    df_calc = panel_calculator(df=dfd, calcs=formulas, cids=cids, start=start, end=end)

    # Fourth testcase. Referencing the newly formed category.
    formula = "NEW1 = ( GROWTH - INFL )"  # Growth adjusted for inflation.
    formula_2 = "NEW2 = ( XR - @USD_NEW1 )"
    formulas = [formula, formula_2]
    df_calc = panel_calculator(df=dfd, calcs=formulas, cids=cids, start=start, end=end)

    # Fifth testcase.
    # Integration of time-series operations.
    formula = "NEW1 = GROWTH.pct_change(periods=1, fill_method='pad')"
    formulas = [formula]
    df_calc = panel_calculator(df=dfd, calcs=formulas, cids=cids, start=start, end=end)

    # Sixth testcase.
    # Further testing of time-series operations.
    formula = "NEW1 = GROWTH.pct_change(periods=1, fill_method='pad') - " \
              "INFL.pct_change(periods=1, fill_method='pad')"
    formulas = [formula]
    df_calc = panel_calculator(df=dfd, calcs=formulas, cids=cids, start=start, end=end)

    # Seventh testcase.
    formulas = ["NEW1 = np.square(np.abs( XR ))"]
    filt1 = (dfd['xcat'] == 'XR')
    dfdx = dfd[filt1]

    df_calc = panel_calculator(df=dfdx, calcs=formulas, cids=cids, start=start, end=end)
