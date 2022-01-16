import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
import re
import random
from queue import LifoQueue

def separation(function: str):

    clean = lambda elem: elem.strip()

    split = function.split(' = ')
    if len(split) != 2:
        assert "Expected form of formula is y = f(x)."
    else:
        key_value = tuple(map(clean, split))

    return key_value


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
    for i, c in enumerate(expression):
        assert c != " ", "Expression must not contain spaces."
        if c == "(":
            parenthesis_counter += 1
            stack.put(c)
        elif c == ")":

            if stack.get() == "(":
                continue
            else:
                return False
        elif c in ["+", "-", "*", "/"]:
            arithmetic_counter += 1
        else:
            continue

    error_message = "Invalid expression. Each arithmetic operator requires parenthesis."
    assert arithmetic_counter == parenthesis_counter, error_message
    if stack.empty():
        return True
    else:
        return False

def involved_xcats(xcats: List[str], expression: str):
    """
    Understand the number of categories involved in each specific panel calculation and
    return their respective indices in the expression.
    The chosen work flow, for each expression, is largely predicated on the number of
    involved categories. For instance, if the function is a unary operation, the work
    flow required is tractable. In contrast, if multiple categories are involved, the
    approach has to be more considered.

    :param <List[str]> xcats: the categories held in the dataframe. The categories
        referenced in the expression must be a subset of the categories defined in the
        dataframe.
    :param <str> expression:

    :return <dict>: the keys will be the categories referenced in the expression, and the
        values will be their indices.
    """

    indices_dict = {}
    for category in xcats:

        pattern = re.compile(category)
        indices = pattern.finditer(expression)

        groups = []
        for index in indices:
            groups.append(index.span())
        if groups:
            indices_dict[category] = groups
        else:
            continue

    return indices_dict

def dataframe_pivot(df: pd.DataFrame, category: str):
    dfx = df[df['xcat'] == category]
    dfw = dfx.pivot(index='real_date', columns='cid', values='value')

    return dfw

def evaluateHelp(df: pd.DataFrame, expression: str, index: int):

    char = expression[index]
    if char == "(":

        index += 1
        left, index = evaluateHelp(df, expression, index)
        opr = expression[index]
        index += 1

        right, index = evaluateHelp(df, expression, index)
        index += 1
        if opr == "+":
            return (left + right), index
        elif opr == "-":
            return (left - right), index
        elif opr == "*":
            return (left * right), index
        else:
            return round((left / right), ndigits=2), index
    elif char.isnumeric():
        start = index

        while char.isnumeric() or char == ".":
            index += 1
            char = expression[index]
        return float(expression[start:index]), index

    elif char.isalpha():
        start = index

        while char.isalpha():
            index += 1
            char = expression[index]

        category = expression[start:index]
        dfw = dataframe_pivot(df=df, category=category)
        return dfw, index

    else:
        return 0

def evaluate(df: pd.DataFrame, expression: str):
    index = 0

    print("Called.")
    return evaluateHelp(df, expression, index)

def expression_modify(df: pd.DataFrame, indices_dict: dict, expression: str,
                      main_category: str):

    assert main_category in indices_dict.keys(), "Error in defined function."

    category_df_dict = {}
    index_adjustment = 0
    for category, indices in indices_dict.items():
        c_copy = category

        dfx = df[df['xcat'] == c_copy]
        dfw = dfx.pivot(index='real_date', columns='cid', values='value')

        category_df_dict[category] = dfw
        # The String will be converted to the memory handler for the dataframe in memory.
        dfw_xcat = "dfw_" + category
        locals()[dfw_xcat] = dfw

        # Iterate through the possible indices (where the expression is mentioned) and
        # substitute the corresponding dataframe. For instance, XR = XR + 0.5 will be
        # converted to: (dfw = dfw + 0.5) where dfw is the pivoted dataframe.

        for i, tup in enumerate(indices):

            first = (tup[0] + index_adjustment)
            last = (tup[1] + index_adjustment)

            replace = f"locals()[{dfw_xcat}]"
            expression = expression[:first] + replace + expression[last:]
            index_adjustment = len(replace) - len(category)

    dfw = category_df_dict[main_category]
    # Redefine the variable.
    expression = "dfw = " + expression

    return dfw

def binary_operations(expression: str, indices_dict: dict):
    """
    In mathematics, a binary operation is a rule for combining two elements, operands, to
    produce another element. Therefore, split the expression on the binary operator. The
    purpose of such a procedure is to evaluate the expression in separate components.

    :param <str> expression: an expression involving two or more categories. For
        instance, expression = ((OLDCAT1+0.5)*OLDCAT2) would be deconstructed into two
        separate strings allowing an evaluation of the unary operations first.
    :param <dict> indices_dict: dictionary containing the categories involved in the
        expression and their respective indices in the expression.
    """

    binary_operators = ["+", "-", "*", "/"]
    pass

def panel_calculator(df: pd.DataFrame, calcs: List[str] = None, cids: List[str] = None,
                     xcats: List[str] = None, start: str = None, end: str = None,
                     blacklist: dict = None) -> object:
    """
    Calculates panels based on simple operations in input panels.

    :param <pd.Dataframe> df: standardized dataframe with following necessary columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> calcs:  List containing the functions applied to each respective
        category outlined in the xcats parameter. The function will be specified in the
        form of an equation. For instance, "XR = XR + 0.5".
    :param <List[str]> cids: cross sections for which the new panels are calculated.
    :param <List[str]> xcats: the categories the panel calculator is applied to.
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
    Panel calculation strings can use functions and unary mathematical operators
    in accordance with numpy convention to indicate the desired calculation such as:
        "NEWCAT = (OLDCAT1 + 0.5) * OLDCAT2"
        "NEWCAT = np.log(OLDCAT1) - np.abs(OLDCAT2) ** 1/2"
    Panel calculation can also involve individual indicator series (to be applied
    to all series in the panel by using the @ as prefix, such as:
        "NEWCAT = OLDCAT1 - np.sqrt(@USD_OLDCAT2)"
    If more than one new category is calculated, the resulting panels can be used
    sequentially in the calculations, such as:
        ["NEWCAT1 = 1 + OLDCAT1/100", "NEWCAT2 = OLDCAT2 * NEWCAT1]

    """
    assert isinstance(xcats, list), f"List of categories expected, and not object type:" \
                                    " {type(xcats)}."
    assert isinstance(cids, list), f"Cross-sections passed must be held in a List."
    assert isinstance(calcs, list), "List of functions expected."
    assert all([isinstance(elem, str) for elem in calcs]), "Elements must be strings."

    dfx = reduce_df(df, xcats=xcats, cids=cids, start=start,
                    end=end, blacklist=blacklist)

    dict_function = {}
    for calc in calcs:
        separate = separation(calc)
        dict_function[separate[0]] = separate[1]

    output_df = []
    unique_categories = dfx['xcat'].unique()

    for k, v in dict_function.items():
        assert v[0] == "(" and v[-1] == ")", "Function must be encased in parenthesis."
        assert checkExpression(v), f"Parenthesis are incorrect in the function passed."

        dfw = evaluate(df=dfx, expression=v)

    return dfw


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    start = '2010-01-01'
    end = '2020-12-31'

    filt1 = (dfd['xcat'] == 'XR') | (dfd['xcat'] == 'CRY')
    dfdx = dfd[filt1]

    df_calc = panel_calculator(df=dfdx,
                               calcs=["XR = (XR+0.5)/CRY)"],
                               cids=cids, xcats=['XR', 'CRY'], start=start, end=end,
                               blacklist=black)

