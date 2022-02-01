import numpy as np
import pandas as pd
from typing import List
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
import re
import random
from queue import LifoQueue

def separation(function: str):
    """
    Method used to split the mathematical function on the equal sign to return two
    separate strings. For example, NEWCAT = ((OLDCAT1+0.5)*OLDCAT2) will be returned as a
    tuple (NEWCAT, ((OLDCAT1+0.5)*OLDCAT2)).

    :param <str> function: mathematical function applied to a specific panel.

    :return <tuple> (NEWCAT, ((OLDCAT1+0.5)*OLDCAT2)).
    """

    clean = lambda elem: elem.strip()

    split = function.split(' = ')
    if len(split) != 2:
        assert "Expected form of formula is y = f(x)."
    else:
        key_value = tuple(map(clean, split))

    return key_value

def involved_xcats(dictionary: dict, xcats_available: List[str]):
    """
    Collect categories involved in panel calculations

    :param <dict> dictionary: keys of new category names and values of related formula
        in string format.
    :param <List[str]> xcats_available: sample space of possible categories able to be
        referenced in each calculation.

    :return <dict>: list of the categories that are referenced in the panel
        calculations. The list of categories must be either be a complete set of the
        available categories, or a valid subset.
    """

    xcats_copy = xcats_available.copy()
    xcats = []

    # Polynomial algorithm.
    for v in dictionary.values():
        for xcat in xcats_copy:

            pattern = re.compile(xcat)
            indices = pattern.finditer(v)
            try:
                next(iter(indices))
            except StopIteration:
                continue
            else:
                xcats.append(xcat)

    return list(set(xcats))


def checkExpression(expression: str):
    """
    There are three aspects of the expression that must be confirmed prior to initiating
    the panel_calculator() function using the aforementioned expression. Firstly,
    spaces are not permitted. For instance, "NEWCAT = (OLDCAT1 + 0.5)" must be
    "NEWCAT = (OLDCAT1+0.5)". Secondly, the number of matching parenthesis, included in
    the expression, is delimited by the number of arithmetic operations. Again,
    "NEWCAT = (OLDCAT1 + 0.5) * OLDCAT2" must be in the form: "((OLDCAT1+0.5)*OLDCAT2)".
    And, lastly, check each opening bracket has a matching pair.
    If the above criteria are satisfied, the expression has been defined correctly.

    :param <str> expression: mathematical function applied to specific category.

    return <bool>: binary value representing whether the parenthesis are correct or not.
    """

    stack = LifoQueue(maxsize=100)
    length = len(expression)

    parenthesis_counter = 0
    arithmetic_counter = 0
    numpy_counter = 0
    for i, c in enumerate(expression):
        assert c != " ", "Expression must not contain spaces."
        if c == "(":
            parenthesis_counter += 1
            stack.put(c)
        elif c == ")":

            condition = stack.empty()
            if not condition and stack.get() == "(":
                continue
            else:
                return False
        elif c in ["+", "-", "*", "/"]:
            arithmetic_counter += 1
        elif (c + expression[i + 1: i + 3]) == "np.":
            numpy_counter += 1
        else:
            continue

    error_message = "Invalid expression. Each arithmetic operator, and numpy expression," \
                    " requires parenthesis."
    assert parenthesis_counter == (arithmetic_counter + numpy_counter), error_message
    if stack.empty():
        return True
    else:
        return False


def single_cross(c_list: List[str], category_copy: List[str]):
    """
    Method designed to break up expression where the mathematical function is applied
    to a single cross-section. Will return the category and associated cross-section.

    :param <List[str]> c_list: expression held inside a List.
    :param <List[str]> category_copy: above list but in reverse order.

    :return <tuple(str, str)>:
    """

    c_index = 0
    terminal_index = 0
    for c in category_copy:
        if c != "_" and c != ")":
            c_index += 1
        elif c == ")":
            terminal_index += 1
        else:
            break

    c_index += terminal_index
    cid_index = 0
    for c in category_copy[c_index:]:
        if c != "@":
            cid_index += 1
        else:
            break

    xcat = "".join(c_list[-c_index:-terminal_index])
    adjust = (c_index + 1)
    cid = "".join(c_list[-(cid_index + c_index):-adjust])
    numpy_formula = "".join(c_list[:-(cid_index + adjust)])

    return xcat, cid, numpy_formula


def dataframe_pivot(df: pd.DataFrame, category: str, single_cid: bool):
    """
    Returns a pivoted dataframe on a single panel: each cross-section will be handled by
    a column. Used to support the recursive evaluation method.
    Further, any numpy function will be evaluated on the respective category and the
    pivoted dataframe will be returned having applied np.function(). For instance,
    np.abs(XR) -> np.abs(df[df['xcat'] == XR]).

    :param <pd.DataFrame> df:
    :param <str> category: category to pivot the dataframe on.
    :param <bool> single_cid: boolean parameter indicating whether a mathematical
        function is applied to the complete panel or a single cross-section.

    :return <pd.DataFrame>: pivoted dataframe.
    """
    numpy_clause = (category[:3] == "np.")
    if numpy_clause:
        c_index = 0
        c_list = list(category)
        category_copy = c_list.copy()
        category_copy.reverse()

        if not single_cid:
            for c in category_copy:
                if c != "(":
                    c_index += 1
                else:
                    break
            xcat = "".join(c_list[-c_index:-1])
        else:
            xcat, cid, numpy_formula = single_cross(c_list=c_list,
                                                    category_copy=category_copy)

        dfx = df[df['xcat'] == xcat]
        dfw = dfx.pivot(index='real_date', columns='cid', values='value')

        if not single_cid:
            adjustment = category[:-c_index] + "dfw" + ")"
            return eval(adjustment)
        else:
            adjustment = numpy_formula + "dfw[cid]" + ")"
            adjustment = "dfw[cid]" + " = " + adjustment
            exec(adjustment)
            return dfw
    else:
        dfx = df[df['xcat'] == category]
        dfw = dfx.pivot(index='real_date', columns='cid', values='value')
        return dfw


def evaluateHelp(df: pd.DataFrame, expression: str, index: int):
    """
    Recursively evaluate the expression. The algorithm will break up the individual
    arithmetic operations, according to the parenthesis, and using a Stack data structure
    "collapse" onto the final output. The most interior parenthesis will be calculated
    first, LIFO principle, and the output will be used to recoil back to calculate the
    remaining binary operations. Will internally handle the inclusion of numpy
    functionality applied to certain categories.

    :param <pd.DataFrame> df:
    :param <str> expression:
    :param <int> index: used for pointer arithmetic - iterating through the string
        object.

    :return <pd.DataFrame>: pivoted dataframe with the mathematical calculation applied.
    """

    l_expression = len(expression)
    char = expression[index]
    if char == "(":

        index += 1
        left, index = evaluateHelp(df, expression, index)
        opr = expression[index]
        index += 1

        right, index = evaluateHelp(df, expression, index)
        index += 1
        if opr == "+":
            return left + right, index
        elif opr == "-":
            return left - right, index
        elif opr == "*":
            return left * right, index
        elif opr == "/":
            return left / right, index
        else:
            concat = left + "right" + ")"
            return eval(concat), index

    elif char.isnumeric():
        start = index

        while char.isnumeric() or char == ".":
            index += 1
            char = expression[index]
        return float(expression[start:index]), index

    elif char.isalpha():
        start = index

        single_cid = False
        while char.isalpha() or char in ["(", ")", ".", "@", "_"]:
            index += 1
            try:
                expression[index]
            except IndexError:
                break
            else:
                char = expression[index]
            if char == "n" and expression[index: (index + 3)] == "np.":
                return expression[start:index], (index - 2)
            elif char == "@":
                single_cid = True

        if index == l_expression and expression[0] == "(":
            index -= 1

        category = expression[start:index]
        dfw = dataframe_pivot(df=df, category=category, single_cid=single_cid)
        return dfw, index

    else:
        return 0


def evaluate(df: pd.DataFrame, expression: str):
    """
    Driver function to initiate the recursive algorithm. Used to define the index
    variable.

    :param <pd.DataFrame> df:
    :param <str> expression:

    :return <pd.DataFrame>
    """
    index = 0

    output = evaluateHelp(df, expression, index)

    if isinstance(output, tuple):
        output = output[0]
    return output


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
    to all series in the panel by using the @ as prefix), such as:
    "NEWCAT = OLDCAT1 - np.sqrt(@USD_OLDCAT2)"
    If more than one new category is calculated, the resulting panels can be used
    sequentially in the calculations, such as:
    ["NEWCAT1 = 1 + OLDCAT1/100", "NEWCAT2 = OLDCAT2 * NEWCAT1"]

    """

    assert isinstance(calcs, list), "List of functions expected."
    assert all([isinstance(elem, str) for elem in calcs]),\
        "Each formula in the panel calculation list must be a string."
    assert isinstance(cids, list), "List of cross-sections expected."

    dict_function = {}
    for calc in calcs:
        separate = separation(calc)
        dict_function[separate[0]] = separate[1]

    unique_categories = list(df['xcat'].unique())
    xcats = involved_xcats(dictionary=dict_function, xcats_available=unique_categories)

    dfx = reduce_df(df, xcats=xcats, cids=cids, start=start,
                    end=end, blacklist=blacklist)

    output_df = []
    col_names = ['cid', 'xcat', 'real_date', 'value']

    for k, v in dict_function.items():
        assert checkExpression(v), f"Parenthesis are incorrect in the function passed."

        dfw = evaluate(df=dfx, expression=v)
        df_out = dfw.stack().to_frame("value").reset_index()
        df_out['xcat'] = k
        df_new = df_out.sort_values(['cid', 'real_date'])[col_names]

        # Integrate the newly formed category into the original dataframe allowing it to
        # be used in other panel calculations.
        dfx = pd.concat([dfx, df_new])
        dfx = dfx.reset_index(drop=True)

        output_df.append(df_new)

    df_calc = pd.concat(output_df)

    return df_calc.reset_index(drop=True)


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

    filt1 = (dfd['xcat'] == 'XR') | (dfd['xcat'] == 'CRY')
    dfdx = dfd[filt1]

    # Start of the testing. Various testcases included to understand the capabilities of
    # the designed function.
    df_calc = panel_calculator(df=dfdx,
                               calcs=["XRCALC = (np.abs(XR)+0.552"],
                               cids=cids, start=start, end=end)

    df_calc = panel_calculator(df=dfdx,
                               calcs=["XRCALC = (np.square(np.abs(XR)+0.5))"],
                               cids=cids, start=start, end=end,
                               blacklist=black)

    # Testing multiple categories being referenced in a single expression.
    df_calc = panel_calculator(df=dfdx,
                               calcs=["XRCALC = (XR+np.abs(CRY))"],
                               cids=cids, start=start, end=end,
                               blacklist=black)

    # Further testcase.
    # Exploring the breadth of the panel calculation.
    df_calc = panel_calculator(df=dfdx,
                               calcs=["XRCALC = (np.sqrt(np.square(np.abs(XR)+0.5)))",
                                      "CRYCALC = (np.abs(CRY)-0.5)"],
                               cids=cids, start=start, end=end,
                               blacklist=black)

    filt2 = dfd['xcat'] == 'CRY'
    dfdx = dfd[filt2]
    df_calc = panel_calculator(df=dfdx, calcs=["CRYCALC = np.square(CRY)"], cids=cids,
                               start=start, end=end, blacklist=black)

    calc_list = ["CRYCALC = ((np.log(np.square(np.abs(CRY)+0.5)))+1)"]
    df_calc = panel_calculator(df=dfdx, calcs=calc_list, cids=cids, start=start,
                               end=end, blacklist=black)

    # Further testcase.
    # Testing the inclusion of a mathematical function being applied to a single
    # cross-section, as opposed to the complete panel.
    filt3 = (dfd['xcat'] == 'XR') | (dfd['xcat'] == 'CRY')
    dfdx = dfd[filt3]
    df_calc = panel_calculator(df=dfdx,
                               calcs=["XRCALC = (XR-np.sqrt(@USD_CRY))",
                                      "CRYCALC = np.abs(CRY)"],
                               cids=cids, start=start, end=end,
                               blacklist=black)

    # Further testcase.
    # Testing the feature of using the internally calculated category in a subsequent
    # calculation. Example: "NEWCAT2 = (OLDCAT2*NEWCAT1)"
    df_calc = panel_calculator(df=dfdx,
                               calcs=["XRCALCONE = (XR+100)",
                                      "XRCALCTWO = np.sqrt(XRCALCONE)"],
                               cids=cids, start=start, end=end,
                               blacklist=black)