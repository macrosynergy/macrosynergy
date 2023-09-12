"""
Module for calculating notional positions based on contract signals, assets-under-management, 
and other relevant parameters.

::docs::notional_positions::sort_first::
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df

def notional_positions(
    df: pd.DataFrame,
    sname: str,
    contids: List[str],
    aum: Union[float, int] = 100,
    dollar_per_signal: Union[float, int] = 1,
    leverage: Union[float, int] = None,
    vol_target: Union[float, int] = None,
    rebal_freq: str = 'm', 
    slip: int = 1,
    lback_periods: int = 21, 
    lback_meth: str = 'ma', 
    half_life=11,
    rstring: str = 'XR',
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    pname: str = 'POS',   
):
    """
    Calculates contract positions based on contract signals, AUM and other specs.

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain the contract-specific signals and possibly
        related return series (for vol-targeting).
    :param <str> sname: the name of the strategy. It must correspond to contract
        signals in the dataframe, which have the format "<cid>_<ctype>_<sname>_CSIG", and
        which are typically calculated by the function contract_signals().
    :param <list[str]> contids: list of contract identifiers in the format 
        "<cid>_<ctype>". It must correspond to contract signals in the dataframe.
    :param <float> aum: the assets under management in USD million (for consistency).
        This is basis for all position sizes. Default is 100.
    :param <float> dollar_per_signal: the amount of notional currency (e.g. USD) per
        contract signal value. Default is 1. The default scale has no specific meaning
        and is merely a basis for tryouts.
    :param <float> leverage: the ratio of the sum of notional positions to AUM.
        This is the main basis for leveraged-based positioning. Since different 
        contracts have diferent eexpected volatility and correlations this method
        does not control expected volatility. Default is None, i.e. the method is not
        applied.
    :param <float> vol_target: the target volatility of the portfolio in % of AUM.
        This is the main parameter for volatility-targeted positioning. That method
        estimates the annualized standard deviation of the signal-based portfolio 
        for a 1 USD per signal portfolio based on past variances and covariances of
        the contract returns. The estimation is managed by the function 
        `historic_portfolio_vol()`.
        Default is None, i.e. the volatility-targeting is not applied.
    :param <str> rebal_freq: the rebalancing frequency. Default is 'm' for monthly.
        Alternatives are 'w' for business weekly, 'd' for daily, and 'q' for quarterly.
        Contract signals are taken from the end of the holding period and applied to
        positions at the beginning of the next period, subject to slippage.
    :slip <int>: the number of days to wait before applying the signal. Default is 1.
        This means that positions are taken at the very end of the first business day
        of the holding period.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21. This passed through to
        the function `historic_portfolio_vol()`.
    :param <str> lback_meth: the method to use for the lookback period of the
        volatility-targeting method. Default is 'ma' for moving average. Alternative is 
        "xma", for exponential moving average. Again this is passed through to
        the function `historic_portfolio_vol()`.
    :param <int> half_life: the half-life of the exponential moving average for the
        volatility-targeting method. Default is 11. This is passed through to
        the function `historic_portfolio_vol()`.
    :param <str> rstring: a general string of the return category. This identifies
        the contract returns that are required for the volatility-targeting method, based
        on the category identifier format <cid>_ <ctype><rstring>_<rstring> in accordance
        with JPMaQS conventions. Default is 'XR'.
    :param <str> start: the start date of the data. Default is None, which means that
        the start date is taken from the dataframe.
    :param <str> end: the end date of the data. Default is None, which means that
        the end date is taken from the dataframe.
    :param <dict> blacklist: a dictionary of contract identifiers to exclude from
        the calculation. Default is None, which means that no contracts are excluded.
    :param <str> pname: the name of the position. Default is 'POS'.

    :return: <pd.DataFrame> with the positions for all traded contracts and the
        specified strategy in USD million. It has the standard JPMaQS DataFrame. 
        The contract signals have the following format "<cid>_<ctype>_<sname>_<pname>".
    
    """

    pass



def historic_portfolio_vol(
    df: pd.DataFrame,
    sname: str,
    contids: List[str],
    est_freq: str = 'm',
    lback_periods: int = 21, 
    lback_meth: str = 'ma', 
    half_life=11,
    rstring: str = 'XR',
    start: str = None,
    end: str = None,
    blacklist: dict = None,
):
    """
    Estimates the annualized standard deviations of a changing portfolio of contracts.

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
        This dataframe must contain the contract-specific signals and return series.
    :param <str> sname: the name of the strategy. It must correspond to contract
        signals in the dataframe, which have the format "<cid>_<ctype>_<sname>_CSIG", and
        which are typically calculated by the function contract_signals().
    :param <list[str]> contids: list of contract identifiers in the format 
        "<cid>_<ctype>". It must correspond to contract signals in the dataframe.
    :param <str> est_freq: the frequency of the volatility estimation. Default is 'm'
        for monthly. Alternatives are 'w' for business weekly, 'd' for daily, and 'q'
        for quarterly. Estimations are conducted for the end of the period.
    :param <float> dollar_per_signal: the amount of notional currency (e.g. USD) per
        contract signal value. Default is 1. The default scale has no specific meaning
        and is merely a basis for tryouts.
    :param <int> lback_periods: the number of periods to use for the lookback period
        of the volatility-targeting method. Default is 21. This passed through to
        the function `historic_portfolio_vol()`.
    :param <str> lback_meth: the method to use for the lookback period of the
        volatility-targeting method. Default is 'ma' for moving average. Alternative is 
        "xma", for exponential moving average. Again this is passed through to
        the function `historic_portfolio_vol()`.
    :param <str> rstring: a general string of the return category. This identifies
        the contract returns that are required for the volatility-targeting method, based
        on the category identifier format <cid>_ <ctype><rstring>_<rstring> in accordance
        with JPMaQS conventions. Default is 'XR'.
    :param <str> start: the start date of the data. Default is None, which means that
        the start date is taken from the dataframe.
    :param <str> end: the end date of the data. Default is None, which means that
        the end date is taken from the dataframe.
    :param <dict> blacklist: a dictionary of contract identifiers to exclude from
        the calculation. Default is None, which means that no contracts are excluded.
    

    :return: <pd.DataFrame> with the annualized standard deviations of the portfolios.
        The values are in % annualized. Values between estimation points are forward
        filled. 
    
    N.B.: If returns in the lookback window are not available the function will replace
    them with the average of the available returns of the same contract type. If no
    returns are available for a contract type the function will reduce the lookback window 
    up to a minimum of 11 days. If no returns are available for a contract type for 
    at least 11 days the function returns an NaN for that date and sends a warning of all 
    the dates for which this happened.

    
    """
    
    pass