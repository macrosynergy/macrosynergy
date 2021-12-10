import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


class RealPnL:

    """Estimates and analyses naive illustrative PnLs with limited signal options and
    disregarding transaction costs

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <List[str]> contracts: base tickers (combinations of cross sections and base
        categories) for which signals are to be generated.
    :param <List[str]> sigs: signal categories. These are signal postfixes that are
        appended to the contracts to identify the signal categories
        For example: 'SIG' is appended to 'EUR_FX' to give the ticker 'EUR_FXSIG.
    :param <str> ret: return category postfix; default is "XR_NSA".
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <dict> blacklist: cross sections with date ranges for which no positions
        should be taken
    """

    def __init__(self, df: pd.DataFrame, contracts: List[str],
                 sigs: List[str], ret: str,
                 cids: List[str] = None,
                 start: str = None, end: str = None,
                 blacklist: dict = None):

        pass

    def make_positions(self):
        """
        Calculate the exact positions for ach contract considering rebalancing frequency
            rebalancing lags, and volaility targets.

        # Todo: define appropriate arguments

        """

    def make_pnls(self):
        """
        Calculate the PnLs under consideration of trading and roll costs.

        # Todo: define appropriate arguments

        """
