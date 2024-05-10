import numpy as np
import pandas as pd

from macrosynergy.learning.panel_time_series_split import (
    ExpandingIncrementPanelSplit,
    ExpandingKFoldPanelSplit,
    BasePanelSplit,
)

from macrosynergy.learning.metrics import neg_mean_abs_market_corr

from macrosynergy.management.utils.df_utils import (
    reduce_df,
    categories_df,
    update_df,
)

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from typing import Union, Optional, Dict, Tuple, List, Callable

from tqdm import tqdm 

import warnings 

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
        blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None, # ignore for now
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
        self._checks_init_params(df, contract_return, market_return, market_cid, contract_cids, blacklist)

        # Assign class variables
        self.contract_return = contract_return
        self.market_return = market_return
        self.market_cid = market_cid
        self.contract_cids = sorted(contract_cids)
        self.blacklist = blacklist

        # Create pseudo-panel
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])

        # TODO: if contract_cids is None, then set it to all cids in the dataframe associated with contract_return
        # Probably sort them too
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
            # blacklist=self.blacklist, comment out for now
        ).dropna()

        self.X = Xy_long[self.market_return]
        self.y = Xy_long[self.contract_return]

        # Create dataframes to store the estimated betas, selected models and chosen 
        # hyperparameters
        self.beta_df = pd.DataFrame(columns=["cid", "real_date", "xcat", "value"])
        self.chosen_models = pd.DataFrame(
            columns=["real_date", "xcat", "model_type", "hparams", "n_splits_used"]
        )

    def _checks_init_params(
        self,
        df: pd.DataFrame,
        contract_return: str,
        market_return: str,
        market_cid: str,
        contract_cids: List[str],
        blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]],
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
        n_jobs_outer: Optional[int] = -1,
        n_jobs_inner: Optional[int] = 1,
    ):
        """
        Method to estimate and store the beta coefficients for each cross-section in the panel. 
        """
        
        # (1) Create a dataframe to store the estimated betas at business-daily frequency
        min_date = self.X.index.get_level_values(1).min()
        max_date = self.X.index.get_level_values(1).max()
        date_range = pd.bdate_range(start=min_date, end=max_date, freq="B")
        idxs = pd.MultiIndex.from_product([sorted(self.contract_cids), date_range], names=["cid","real_date"])
        
        stored_betas = pd.DataFrame(index=idxs, columns=[name], data=np.nan, dtype=np.float64)
        beta_list = []
        chosen_models = []

        # (2) Loop through outer splitter
        for idx, (train_idx, test_idx) in enumerate(tqdm(outer_splitter.split(self.X, self.y))):
            # (2a) Get training and testing samples
            X_train_i = pd.DataFrame(self.X.iloc[train_idx])
            y_train_i = self.y.iloc[train_idx]
            X_test_i = pd.DataFrame(self.X.iloc[test_idx])
            y_test_i = self.y.iloc[test_idx]

            # (2b) Run grid search
            optim_name = None
            optim_model = None 
            optim_score = -np.inf 

            for model_name, model in models.items():
                if hparam_type == "grid":
                    search_object = GridSearchCV(
                        estimator=model,
                        param_grid=hparam_grid[model_name],
                        scoring=scorer,
                        refit=True,
                        cv=inner_splitter,
                        n_jobs=-1,
                    )
                elif hparam_type == "random":
                    search_object = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=hparam_grid[model_name],
                        n_iter=n_iter,
                        scoring=scorer,
                        refit=True,
                        cv=inner_splitter,
                        n_jobs=n_jobs_inner,
                    )
                # Run the grid search
                try:
                    search_object.fit(X_train_i, y_train_i)
                except Exception as e:
                    warnings.warn(
                    f"Error in the grid search for {model_name}: {e}", # Add correct date as well
                    RuntimeWarning,
                )
                # 
                score = search_object.best_score_
                if score > optim_score:
                    optim_name = model_name
                    optim_model = search_object.best_estimator_
                    optim_score = score
                    optim_params = search_object.best_params_

            # Handle case where no model was chosen 
            if optim_model is None:
                # TODO add warning 
                continue
            
            # Get beta estimates for each cross-section
            # These are stored in estimator.coefs_ as a dictionary {cross-section: beta}

            betas = optim_model.coefs_

            # Get OOS hedged returns
            hedged_returns: pd.Series = self.get_hedged_returns(betas, X_test_i, y_test_i)
            # Evaluate hedged returns
            pearson_h, spearman_h, kendall_h = self.get_absolute_correlations(hedged_returns, X_test_i)
            pearson_c, spearman_c, kendall_c = self.get_absolute_correlations(y_test_i, X_test_i)
            pearsons, spearmans, kendalls = [pearson_h, pearson_c], [spearman_h, spearman_c], [kendall_h, kendall_c]
            # TODO: work towards getting a table with columns [real_date, xcat, correlation_type, is_hedged, value]


            # Compute 
            training_time = X_train_i.index.get_level_values(1).max()
            for cross_section, beta in betas.items():
                # Strip the contract cross section from the pseudo cross section names
                cid = cross_section.split("v")[0]
                beta_list.append([cid, training_time, name, beta])

            # Store chosen models and hyperparameters
            chosen_models.append([training_time, name, optim_name, optim_params, inner_splitter.n_splits])

        # (3) Convert beta_list to a dataframe and upsample to business-daily frequency
        for cid, real_date, xcat, value in beta_list:
            stored_betas.loc[(cid, real_date), xcat] = value

        stored_betas = stored_betas.groupby(level=0).ffill()
        stored_betas_long = pd.melt(frame=stored_betas.reset_index(),id_vars=["cid", "real_date"], var_name="xcat", value_name="value")
        
        self.beta_df = pd.concat((self.beta_df, stored_betas_long), axis=0).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float64,
            }
        )
        self.chosen_models = pd.concat((self.chosen_models, pd.DataFrame(chosen_models, columns=["real_date", "xcat", "model_type", "hparams", "n_splits_used"])))

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

if __name__ == "__main__":
    import os 
    from macrosynergy.download import JPMaQSDownload
    from timeit import default_timer as timer
    from datetime import timedelta, date, datetime
    
    from sklearn.base import RegressorMixin
    from sklearn.ensemble import VotingRegressor
    from sklearn.linear_model import LinearRegression 

    from collections import defaultdict
    import scipy.stats as stats

    from macrosynergy.learning.metrics import neg_mean_abs_market_corr
    from macrosynergy.learning.predictors import SURollingLinearRegression
    # Cross-sections of interest

    cids_dmca = [
        "AUD",
        "CAD",
        "CHF",
        "EUR",
        "GBP",
        "JPY",
        "NOK",
        "NZD",
        "SEK",
        "USD",
    ]  # DM currency areas
    cids_dmec = ["DEM", "ESP", "FRF", "ITL", "NLG"]  # DM euro area countries
    cids_latm = ["BRL", "COP", "CLP", "MXN", "PEN"]  # Latam countries
    cids_emea = ["CZK", "HUF", "ILS", "PLN", "RON", "RUB", "TRY", "ZAR"]  # EMEA countries
    cids_emas = [
        "CNY",
        # "HKD",
        "IDR",
        "INR",
        "KRW",
        "MYR",
        "PHP",
        "SGD",
        "THB",
        "TWD",
    ]  # EM Asia countries

    cids_dm = cids_dmca + cids_dmec
    cids_em = cids_latm + cids_emea + cids_emas

    cids = sorted(cids_dm + cids_em)

    # Categories

    main = ["FXXR_NSA", "FXXR_VT10", "FXXRHvGDRB_NSA"]

    econ = []

    mark = [
        "EQXR_NSA",
        "FXXRBETAvGDRB_NSA",
        "FXTARGETED_NSA",
        "FXUNTRADABLE_NSA",
    ]  # related market categories

    xcats = main + econ + mark

    xtix = ["GLB_DRBXR_NSA", "GEQ_DRBXR_NSA"]

    # Download

    # Download series from J.P. Morgan DataQuery by tickers

    start_date = "1990-01-01"
    tickers = [cid + "_" + xcat for cid in cids for xcat in xcats] + xtix
    print(f"Maximum number of tickers is {len(tickers)}")

    # Retrieve credentials

    client_id: str = os.getenv("DQ_CLIENT_ID")
    client_secret: str = os.getenv("DQ_CLIENT_SECRET")

    # Download from DataQuery

    with JPMaQSDownload(client_id=client_id, client_secret=client_secret) as downloader:
        start = timer()
        assert downloader.check_connection()
        df = downloader.download(
            tickers=tickers,
            start_date=start_date,
            metrics=["value", "eop_lag", "mop_lag", "grading"],
            suppress_warning=True,
            show_progress=True,
        )
        end = timer()

    dfd = df.copy()

    print("Download time from DQ: " + str(timedelta(seconds=end - start)))
        
        
    object = MarketBetaEstimator(
        df=dfd,
        contract_return="FXXR_NSA",
        market_return="DRBXR_NSA",
        contract_cids=list(set(cids) - set(["USD"])),
        market_cid="GLB",
    )
    object.estimate_beta(
        name="TEST",
        outer_splitter=ExpandingIncrementPanelSplit(
            train_intervals = 21*12, # Re-estimate beta every year
            min_cids=4,
            min_periods = 21*12*2, # first split has 2 years worth of data for 4 cross sections
            test_size=21 * 3, # evaluate OOS hedged returns over subsequent quarter 
        ),
        inner_splitter=ExpandingKFoldPanelSplit(n_splits = 5),
        models = {
            "LR_ROLL": SURollingLinearRegression(min_xs_samples=21),
        },
        hparam_grid = {
            "LR_ROLL": {"roll": [21, 21 * 3, 21 * 6]},
        },
        scorer=neg_mean_abs_market_corr,
    )