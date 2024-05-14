import numpy as np
import pandas as pd

from macrosynergy.learning.panel_time_series_split import (
    ExpandingIncrementPanelSplit,
    ExpandingKFoldPanelSplit,
    BasePanelSplit,
)

from macrosynergy.learning.metrics import neg_mean_abs_corr

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

class BetaEstimator:
    """
    Beta estimation of a panel of contract returns based on regression.
    
    Estimation is performed using seemingly unrelated linear regression models within
    an expanding window learning process. Statistical learning determines the model and 
    hyperparameter choices sequentially using a given performance metric. 
    Cross-section betas and out-of-sample "hedged" returns are stored in quantamental 
    dataframes.  
    """
    def __init__ (
        self,
        df: pd.DataFrame,
        xcat: str,
        cids: List[str],
        benchmark_return: str,
        blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None, # ignore for now
    ):
        """
        Initializes BetaEstimator. Takes a quantamental dataframe and creates long
        format dataframes for the contract and market returns. 

        :param <pd.DataFrame> df: Daily quantamental dataframe with the following necessary
            columns: 'cid', 'xcat', 'real_date' and 'value'.
        :param <str> xcat: Extended category name for the financial returns to be hedged.
        :param <List[str]> cids: Cross-section identifiers for the financial returns to be hedged.
        :param <str> benchmark_return: Ticker name for the benchmark return to be used in
            the hedging process and beta estimation.  
        :param <Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]> blacklist: Cross-sections
            with date ranges to be excluded from the dataframe. Default is None. 
        """
        # Checks
        self._checks_init_params(df, xcat, cids, benchmark_return, blacklist)

        # Assign class variables
        self.xcat = xcat
        self.cids = sorted(cids)
        self.benchmark_return = benchmark_return
        self.benchmark_cid, self.benchmark_xcat = self.benchmark_return.split("_",1)
        self.blacklist = blacklist

        # Create pseudo-panel
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])

        for cid in self.cids:
            dfa = reduce_df(
                df=df,
                xcats=[self.benchmark_xcat, self.xcat],
                cids=[self.benchmark_cid, cid],
            )
            dfa["cid"] = f"{cid}v{self.benchmark_cid}"

            dfx = update_df(dfx, dfa)

        # Create long format dataframes

        Xy_long = categories_df(
            df = dfx,
            xcats = [self.benchmark_xcat, xcat],
            freq="D",
            xcat_aggs=["sum", "sum"],
            lag = 0,
            # blacklist=self.blacklist, comment out for now
        ).dropna()

        self.X = Xy_long[self.benchmark_xcat]
        self.y = Xy_long[xcat]

        # Create dataframes to store the estimated betas, selected models and chosen 
        # hyperparameters
        self.beta_df = pd.DataFrame(columns=["cid", "real_date", "xcat", "value"])
        self.hedged_returns = pd.DataFrame(columns=["cid", "real_date", "xcat", "value"])
        self.chosen_models = pd.DataFrame(
            columns=["real_date", "xcat", "model_type", "hparams", "n_splits_used"]
        )

    def _checks_init_params(
        self,
        df: pd.DataFrame,
        xcat: str,
        cids: List[str],
        benchmark_return: str,
        blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]],
    ):
        pass

    def estimate_beta(
        self,
        beta_xcat: str,
        hedged_return_xcat: str,
        inner_splitter: BasePanelSplit,
        scorer: Callable, # Possibly find a better type hint for a scikit-learn scorer
        models: Dict[str, Union[BaseEstimator, Pipeline]],
        hparam_grid: Dict[str, Dict[str, List]],
        min_cids: int = 1,
        min_periods: int = 252,
        oos_period: int = 21,
        initial_nsplits: Optional[Union[int, np.int_]] = None, # TODO incorporate this logic later
        threshold_ndates: Optional[Union[int, np.int_]] = None, # TODO incorporate this logic later
        hparam_type: str = "grid",
        n_iter: Optional[int] = 10,
        n_jobs_outer: Optional[int] = -1,
        n_jobs_inner: Optional[int] = 1,
    ):
        """
        Estimate and store the beta coefficients for each cross-section.

        At a given estimation date, a search for optimal linear model hyperparameters and underlying
        dataset frequency is performed. The optimal model is fit to the training set, betas for each 
        cross-section are extracted and hedged returns over a subsequent out-of-sample period are
        derived. The hyperparameter search is based on maximization of a cross-validation score from
        a scikit-learn 'scorer' function. 

        :param <str> beta_xcat: Category name for the panel of estimated contract betas.
        :param <str> hedged_return_xcat: Category name for the panel of derived hedged returns.
        :param <BasePanelSplit> inner_splitter: Panel cross-validation splitter for
            the hyperparameter search. It is recommended to set this to an ExpandingKFoldPanelSplit
            splitter.
        :param <Callable> scorer: Scikit-learn scorer function used in both model and 
            hyperparameter selection for optimization. For beta estimation, it is recommended
            to use `neg_mean_abs_market_corr` from the `macrosynergy.learning` submodule. 
        :param <dict> models: Dictionary of scikit-learn compatible linear regression models. For beta
            estimation, these models should be seemingly unrelated regressions with a 'coefs_' attribute
            storing estimated betas for each cross-section.
        :param <dict> hparam_grid: Nested dictionary defining the hyperparameters to consider
            for each model.
        :param <int> min_cids: smallest number of cross-sections to be in the initial training set.
            Since market betas are estimated for each cross-section, it is recommended to set this to 
            one. This parameter should be used in conjunction with min_periods. Default is 1. 
        :param <int> min_periods: minimum requirement for the initial training set in business days. 
            This parameter is applied in conjunction with min_cids.
            Default is 1 year (252 days).
        :param <int> oos_period: Number of out-of-sample business days for which hedged returns
            are derived. After the hedged returns are determined in each iteration, the training
            panel expands to encapsulate these samples, meaning that this parameter also determines
            re-estimation frequency. Default is 1 month (21 days).
        :param <int> initial_nsplits: Number of splits to be used in cross-validation for the initial
            training set. If None, the number of splits is defined by the inner_splitter. Default is None.
        :param <int> threshold_ndates: Number of business days to pass before the number of cross-validation
            splits increases by one. Default is None.
        :param <str> hparam_type: String indicating the type of hyperparameter search. This can be 
            `grid`, `prior` or `bayes`. Currently the `bayes` option produces a NotImplementedError. 
            Default is "grid".
        :param <int> n_iter: Number of iterations to run for random search. Default is 10.
        :param <int> outer_n_jobs: Number of jobs to run the outer splitter in parallel. Default is -1, which uses
            all available cores.
        :param <int> inner_n_jobs: Number of jobs to run the inner splitter in parallel. Default is 1, which uses
            a single core. 
        """
        
        # (1) Create daily multi-indexed dataframes to store estimated betas and hedged returns
        # This makes indexing easy and the resulting dataframes can be melted to long format later

        min_date = self.X.index.get_level_values(1).min()
        max_date = self.X.index.get_level_values(1).max()
        date_range = pd.bdate_range(start=min_date, end=max_date, freq="B")
        idxs = pd.MultiIndex.from_product([sorted(self.contract_cids), date_range], names=["cid","real_date"])
        
        stored_betas = pd.DataFrame(index=idxs, columns=[beta_xcat], data=np.nan, dtype=np.float64)
        stored_hedged_returns = pd.DataFrame(index=idxs, columns=[hedged_return_xcat], data=np.nan, dtype=np.float64)

        # (2) Set up outer splitter
        outer_splitter = ExpandingIncrementPanelSplit(
            train_intervals=oos_period,
            min_cids=min_cids,
            min_periods=min_periods,
            test_size=oos_period,
        )

        # Create lists to store stored quantities in each iteration
        # This is done to avoid appending to the dataframe in each iteration
        beta_list = []
        hedged_return_list = []
        chosen_models = []

        # (3) Loop through outer splitter, run grid search for optimal model, extract beta 
        # estimates, calculate OOS hedged returns and store results
        train_test_splits = list(outer_splitter.split(X=self.X, y=self.y))
        for idx, (train_idx, test_idx) in tqdm(enumerate(train_test_splits),total=len(train_test_splits)):
            # Get training and testing samples
            X_train_i = pd.DataFrame(self.X.iloc[train_idx])
            y_train_i = self.y.iloc[train_idx]
            X_test_i = pd.DataFrame(self.X.iloc[test_idx])
            y_test_i = self.y.iloc[test_idx]
            training_time = X_train_i.index.get_level_values(1).max()

            # Run grid search
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
                        n_jobs=n_jobs_inner,
                    )
                elif hparam_type == "prior":
                    search_object = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=hparam_grid[model_name],
                        n_iter=n_iter,
                        scoring=scorer,
                        refit=True,
                        cv=inner_splitter,
                        n_jobs=n_jobs_inner,
                    )

                try:
                    search_object.fit(X_train_i, y_train_i)
                except Exception as e:
                    warnings.warn(
                    f"Error running a hyperparameter search for {model_name} at {training_time}: {e}",
                    RuntimeWarning,
                )
                if not hasattr(search_object, "best_score_"):
                    # Then the grid search failed completely
                    warnings.warn(
                        f"No model was selected at time {training_time}. Hence, no beta can be estimated."
                    )
                    continue

                # If a model was selected, extract the score, name, estimator and optimal hyperparameters
                score = search_object.best_score_
                if score > optim_score:
                    optim_name = model_name
                    optim_model = search_object.best_estimator_
                    optim_score = score
                    optim_params = search_object.best_params_
            
            # Get beta estimates for each cross-section
            # These are stored in estimator.coefs_ as a dictionary {cross-section: beta}
            betas = optim_model.coefs_

            # Get OOS hedged returns
            # This will be a List of lists, with inner lists recording the hedged return
            # for a given cross-section and a given OOS timestamp.
            hedged_returns: List = self._get_hedged_returns(betas, X_test_i, y_test_i, hedged_return_xcat)
            hedged_return_list.extend(hedged_returns)

            # Also update the list of betas 
            for cross_section, beta in betas.items():
                # Strip the contract cross section from the pseudo cross section names
                cid = cross_section.split("v")[0]
                beta_list.append([cid, training_time, beta_xcat, beta])

            # Store chosen models and hyperparameters
            chosen_models.append([training_time, beta_xcat, optim_name, optim_params, inner_splitter.n_splits])

        # (5) Convert beta_list and hedged_return_list to dataframes and upsample to business-daily frequency
        for cid, real_date, xcat, value in beta_list:
            stored_betas.loc[(cid, real_date), xcat] = value
        for cid, real_date, xcat, value in hedged_return_list:
            stored_hedged_returns.loc[(cid, real_date), xcat] = value

        stored_betas = stored_betas.groupby(level=0).ffill().dropna()
        stored_hedged_returns = stored_hedged_returns.dropna()
        stored_betas_long = pd.melt(frame=stored_betas.reset_index(),id_vars=["cid", "real_date"], var_name="xcat", value_name="value")
        stored_hrets_long = pd.melt(frame=stored_hedged_returns.reset_index(),id_vars=["cid", "real_date"], var_name="xcat", value_name="value")
        
        self.beta_df = pd.concat((self.beta_df, stored_betas_long), axis=0).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float64,
            }
        )
        self.hedged_returns = pd.concat((self.hedged_returns, stored_hrets_long), axis=0).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float64,
            }
        )

        self.chosen_models = pd.concat((self.chosen_models, pd.DataFrame(chosen_models, columns=["real_date", "xcat", "model_type", "hparams", "n_splits_used"])))

    def _get_hedged_returns(
        self,
        betas,
        X_test_i,
        y_test_i,
        hedged_return_xcat,
    ):
        # TODO: input checks
        betas_series = pd.Series(betas)
        XB = X_test_i.mul(betas_series, level=0, axis=0)
        hedged_returns = y_test_i - XB[self.market_return]
        list_hedged_returns = [[idx[0].replace("vGLB", ""), idx[1]] + [hedged_return_xcat] + [value] for idx, value in hedged_returns.items()]
        return list_hedged_returns


    def _checks_estimate_beta(
        self,
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

    from macrosynergy.learning.metrics import neg_mean_abs_corr
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
        
        
    object = BetaEstimator(
        df=dfd,
        xcat="FXXR_NSA",
        benchmark_return="GLB_DRBXR_NSA",
        cids=list(set(cids) - set(["USD"])),
    )
    object.estimate_beta(
        beta_xcat="BETA_NSA",
        hedged_return_xcat="HEDGED_RETURN_NSA",
        inner_splitter=ExpandingKFoldPanelSplit(n_splits = 5),
        scorer=neg_mean_abs_corr,
        models = {
            "LR_ROLL": SURollingLinearRegression(min_xs_samples=21),
        },
        hparam_grid = {
            "LR_ROLL": {"roll": [21, 21 * 3, 21 * 6, 21 * 12, 21 * 24],"data_freq" : ["M", "W", "D"]},
        },
        min_cids=4,
        min_periods=21*12*2,
        oos_period=21 * 3, # Compute hedged returns every quarter
        n_jobs_outer=1,
        n_jobs_inner=-1,
    )
