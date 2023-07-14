import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


def contract_signals(
    df: pd.DataFrame,
    sig: str,
    cids: List[str],
    ctypes: List[str],
    cscales: Union(List[float], List[str]) = None,
    csigns: List[int] = None,
    hbasket: List[str] = None,
    hscales: Union(List[float], List[str]) = None,
    hratio: str = None,
    start: str = None,
    end: str = None,
    blacklist: dict = None
    sname = "STRAT",
):
    """
    Caclulate contract specific signals based on cross-section-specific signals

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain the cross-section-specific signals and possibly
        categories for variable scale factors of the main contracts and the contracts
        in the hedging basket, as well as cross-section specific hedge ratios
    :param <str> sig: the cross-section-specific signal that serves as the basis of
        contract signals.
    :param <list[str]> cids: list of cross-sections whose signal is to be used.
    :param <list[str]> ctypes: list of identifiers for the contract types that are
        to be traded. They typically correspond to the contract type acronyms
        that are used in JPMaQS for  generic returns, carry and volatility, such as
        "FX" for FX forwards or "EQ" for equity index futures.
        N.B. Overall a contract is identified by the combination of its cross-section
        and its contract type "<cid>_<ctype>".
    :param <list[float]> cscales: list of scaling factors for the contract signals.
        These can be eigher a list of floats or a list of category tickers that serve
        as basis of translation. The former are fixed across time, the latter variable.
    :param <list[float]> csigns: list of signs for the contract signals. These must be 
        either 1 for long position or -1 for short position.
    :param <list[str]> hbasket: list of contract identifiers in the format "<cid>_<ctype>"
        that serve as constituents of the hedging basket.
    param <list[float]> cscales: list of scaling factors (weights) for the basket.
        These can be eigher a list of floats or a list of category tickers that serve
        as basis of translation. The former are fixed across time, the latter variable.
    :param <str> hratio: category names for cross-section-specific hedge ratios.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded
        from the dataframe.
    :param <str> sname: name of the strategy. Default is "STRAT".


    :return: <pd.DataFrame> with the contract signals for all traded contracts and the
        specified strategy. It has the standard JPMaQS DataFrame. The contract signals
        have the following format "<cid>_<ctype><sname>_CSIG".
              
    """

    pass