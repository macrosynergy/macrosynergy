import numpy as np
import pandas as pd

from macrosynergy.learning.panel_time_series_split import (
    ExpandingIncrementPanelSplit,
    BasePanelSplit,
)

from macrosynergy.management.utils.df_utils import (
    reduce_df,
    categories_df,
    update_df,
)

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from typing import Union, Optional, Dict, Tuple, List, Callable

class MarketBetaEstimator:
    """
    Class for market beta estimation of a panel of contract returns with respect to 
    a time series of general market returns.
    
    Estimation is performed using seemingly unrelated linear regression models within
    an expanding window learning process. At a given estimation date, a hyperparameter
    search is run to determine the optimal model and hyperparameter configuration
    with respect to a given performance metric. After the model is selected, the betas 
    are extracted from each cross-section and stored in a quantamental dataframe. Lastly,
    the out of sample hedged returns are calculated and stored in a quantamental dataframe
    for analysis.  
    """
    def __init__ (
        self,
        df: pd.DataFrame,
        contract_return: str,
        market_return: str,
        market_cid: str,
        contract_cids: List[str] = None,
        blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    ):
        """
        Initializes MarketBetaEstimator. Takes a quantamental dataframe and creates long
        format dataframes for the contract and market returns. 

        :param <pd.DataFrame> df: Daily quantamental dataframe with the following necessary
            columns: 'cid', 'xcat', 'real_date' and 'value'.
        :param <str> contract_return: Extended category name for the category returns.
        :param <str> market_return: Extended category name for the market returns.
        :param <str> market_cid: Cross-section identifier for the market returns. 
        :param <Optional[List[str]> contract_cids: List of cross-section identifiers for
            the contract returns. Default is all in the dataframe. 
        :param <Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]> blacklist: Cross-sections
            with date ranges to be excluded from the dataframe. Default is None. 
        """
        # Checks
        self._checks_init_params(df, contract_return, market_return, contract_cids, blacklist)

        # Assign class variables
        self.contract_return = contract_return
        self.market_return = market_return
        self.market_cid = market_cid
        self.contract_cids = contract_cids
        self.blacklist = blacklist

        # Create pseudo-panel
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])

        for cid in contract_cids:
            dfa = reduce_df(
                df=df,
                xcats=[self.market_return, self.contract_return],
                cids=[self.market_cid, cid],
            )
            dfa["cid"] = f"{cid}v{self.market_cid}"

            dfx = update_df(dfx, dfa)

        # Create long format dataframes

        Xy_long = categories_df(
            df = dfx,
            xcats = [self.market_return, self.contract_return],
            freq="D",
            xcat_aggs=["sum", "sum"],
            lag = 0,
            blacklist=self.blacklist,
        ).dropna()

        self.X = pd.DataFrame(Xy_long[self.market_return])
        self.y = pd.DataFrame(Xy_long[self.contract_return])

    def _checks_init_params(
        self,
        contract_returns: Union[pd.DataFrame, pd.Series],
        market_returns: Union[pd.DataFrame, pd.Series],
        blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
        freqs: Optional[List[str]] = None,
    ):
        pass

    def estimate_beta(
        self,
        name: str, # category name for the panel of estimated contract betas
        outer_splitter: ExpandingIncrementPanelSplit, # outer splitter for expanding window
        inner_splitter: BasePanelSplit, # inner splitter for cross-validation
        models: Dict[str, Union[BaseEstimator, Pipeline]],
        scorer: Callable, # Scikit-learn scorer. Might be a better existing type for this
        hparam_grid: Dict[str, Dict[str, List]],
        hparam_type: str = "grid",
        n_iter: Optional[int] = 10,
        initial_nsplits: Optional[Union[int, np.int_]] = None,
        threshold_ndates: Optional[Union[int, np.int_]] = None,
        n_jobs: Optional[int] = -1,
    ):
        pass

    def _checks_estimate_beta(
        self,
    ):
        pass

    def get_hedged_returns(
        self,
        name: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        pass