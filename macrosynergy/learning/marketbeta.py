import os
import numpy as np
import pandas as pd

from macrosynergy.learning.panel_time_series_split import (
    ExpandingFrequencyPanelSplit,
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

import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

from typing import Union, Optional, Dict, Tuple, List, Callable

from tqdm import tqdm 

import warnings 

class BetaEstimator:
    """
    Beta estimation of a panel of contract returns based on regression.
    
    Estimates betas with seemingly unrelated linear regression (SUR) sequentially with 
    expanding panel data windows. Statistical learning determines the model and 
    hyperparameter choices sequentially using a given performance metric. Cross-sectional 
    betas and out-of-sample "hedged" returns are stored in quantamental dataframes. 
    """
    def __init__ (
        self,
        df: pd.DataFrame,
        xcat: str,
        cids: List[str],
        benchmark_return: str,
    ):
        """
        Initializes BetaEstimator. Takes a quantamental dataframe and creates long
        format dataframes for the contract and benchmark returns. 

        :param <pd.DataFrame> df: Daily quantamental dataframe with the following necessary
            columns: 'cid', 'xcat', 'real_date' and 'value'.
        :param <str> xcat: Extended category name for the financial returns to be hedged.
        :param <List[str]> cids: Cross-section identifiers for the financial returns to be hedged.
        :param <str> benchmark_return: Ticker name for the benchmark return to be used in
            the hedging process and beta estimation.
        """
        # Checks
        self._checks_init_params(df, xcat, cids, benchmark_return)

        # Assign class variables
        self.xcat = xcat
        self.cids = sorted(cids)
        self.benchmark_return = benchmark_return
        self.benchmark_cid, self.benchmark_xcat = self.benchmark_return.split("_",1)

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
    ):
        # Dataframe checks
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not set(["cid", "xcat", "real_date", "value"]).issubset(df.columns):
            raise ValueError("df must contain columns 'cid', 'xcat', 'real_date' and 'value'.")
        if not df["xcat"].nunique() > 1:
            raise ValueError("df must contain at least two xcats. One is required for the contract return panel and another for the benchmark returns.")
        
        # xcat checks
        if not isinstance(xcat, str):
            raise TypeError("xcat must be a string.")
        if xcat not in df["xcat"].unique():
            raise ValueError("xcat must be a valid category in the dataframe.")
        
        # cids checks
        if not isinstance(cids, list):
            raise TypeError("cids must be a list.")
        if not all(isinstance(cid, str) for cid in cids):
            raise TypeError("All elements in cids must be strings.")
        if not all(cid in df["cid"].unique() for cid in cids):
            raise ValueError("All cids must be valid cross-section identifiers in the dataframe.")
        
        # benchmark return checks
        if not isinstance(benchmark_return, str):
            raise TypeError("benchmark_return must be a string.")
        ticker_list = df["cid"] + "_" + df["xcat"]
        if benchmark_return not in ticker_list.unique():
            raise ValueError("benchmark_return must be a valid ticker in the dataframe.")

    def _checks_estimate_beta(
        self,
        beta_xcat,
        hedged_return_xcat,
        inner_splitter,
        scorer,
        models,
        hparam_grid,
        min_cids,
        min_periods,
        est_freq,
        initial_nsplits,
        threshold_ndates,
        hparam_type,
        n_iter,
        n_jobs_outer,
        n_jobs_inner,   
    ):
        # beta_xcat checks
        if not isinstance(beta_xcat, str):
            raise TypeError("beta_xcat must be a string.")
        if beta_xcat in self.beta_df["xcat"].unique():
            raise ValueError(
                "beta_xcat already exists in the stored quantamental dataframe for beta "
                "estimates. Please choose a different category name."
            )
        
        # hedged_return_xcat checks
        if not isinstance(hedged_return_xcat, str):
            raise TypeError("hedged_return_xcat must be a string.")
        if hedged_return_xcat in self.hedged_returns["xcat"].unique():
            raise ValueError(
                "hedged_return_xcat already exists in the stored quantamental dataframe "
                "for hedged returns. Please choose a different category name."
            )
        
        # inner_splitter checks
        if not isinstance(inner_splitter, BasePanelSplit):
            raise TypeError("inner_splitter must be an instance of BasePanelSplit.")
        
        # scorer
        # TODO: add to these checks
        if not callable(scorer):
            raise TypeError("scorer must be a callable function.")
        
        # models grid checks
        if not isinstance(models, dict):
            raise TypeError("models must be a dictionary.")
        if not all(isinstance(model, (BaseEstimator, Pipeline)) for model in models.values()):
            raise TypeError("All values in the models dictionary must be scikit-learn compatible models.")
        if not all(isinstance(model_name, str) for model_name in models.keys()):
            raise TypeError("All keys in the models dictionary must be strings.")
        if not all(hasattr(model, "coefs_") for model in models.values()):
            raise ValueError(
                "All models must be seemingly unrelated linear regressors consistent with"
                "the scikit-learn API. This means that they must be scikit-learn compatible"
                "regressors that have a 'coefs_' attribute storing the estimated betas for each"
                "cross-section, as per the standard imposed by the macrosynergy package."
            )
        
        # hparam_grid checks
        if not isinstance(hparam_grid, dict):
            raise TypeError("hparam_grid must be a dictionary.")
        if not all(isinstance(model_name, str) for model_name in hparam_grid.keys()):
            raise TypeError("All keys in the hparam_grid dictionary must be strings.")
        if not set(models.keys()) == set(hparam_grid.keys()):
            raise ValueError("The keys in the models and hparam_grid dictionaries must match.")
        if not all(isinstance(grid, (dict, list)) for grid in hparam_grid.values()):
            raise TypeError("All values in the hparam_grid dictionary must be dictionaries or lists of dictionaries.")
        for grid in hparam_grid.values():
            if isinstance(grid, dict):
                if not all(isinstance(key, str) for key in grid.keys()):
                    raise TypeError("All keys in the nested dictionaries of hparam_grid must be strings.")
                if not all(isinstance(value, list) for value in grid.values()):
                    raise TypeError("All values in the nested dictionaries of hparam_grid must be lists.")
            if isinstance(grid, list):
                if not all(isinstance(sub_grid, dict) for sub_grid in grid):
                    raise TypeError("All elements in the list of hparam_grid must be dictionaries.")
                for sub_grid in grid:
                    if not all(isinstance(key, str) for key in sub_grid.keys()):
                        raise TypeError("All keys in the nested dictionaries of hparam_grid must be strings.")
                    if not all(isinstance(value, list) for value in sub_grid.values()):
                        raise TypeError("All values in the nested dictionaries of hparam_grid must be lists.")
                    
        # min_cids checks
        if not isinstance(min_cids, int):
            raise TypeError("min_cids must be an integer.")
        if min_cids < 1:
            raise ValueError("min_cids must be greater than 0.")
        
        # min_periods checks
        if not isinstance(min_periods, int):
            raise TypeError("min_periods must be an integer.")
        if min_periods < 1:
            raise ValueError("min_periods must be greater than 0.")
        
        # est_freq checks
        if not isinstance(est_freq, str):
            raise TypeError("est_freq must be a string.")
        if est_freq not in ["D", "W", "M", "Q"]:
            raise ValueError("est_freq must be one of 'D', 'W', 'M' or 'Q'.")
        
        # initial_nsplits checks
        if initial_nsplits is not None:
            if not isinstance(initial_nsplits, int):
                raise TypeError("initial_nsplits must be an integer.")
            if initial_nsplits < 2:
                raise ValueError("initial_nsplits must be greater than 1.")
            
        # threshold_ndates checks
        if threshold_ndates is not None:
            if not isinstance(threshold_ndates, int):
                raise TypeError("threshold_ndates must be an integer.")
            if threshold_ndates < 1:
                raise ValueError("threshold_ndates must be greater than 0.")
            
        # hparam_type checks
        if not isinstance(hparam_type, str):
            raise TypeError("hparam_type must be a string.")
        if hparam_type not in ["grid", "prior", "bayes"]:
            raise ValueError("hparam_type must be one of 'grid', 'prior' or 'bayes'.")
        if hparam_type == "bayes":
            raise NotImplementedError("Bayesian hyperparameter search is not yet implemented.")
        
        # n_iter checks
        if n_iter is not None:
            if not isinstance(n_iter, int):
                raise TypeError("n_iter must be an integer.")
            if n_iter < 1:
                raise ValueError("n_iter must be greater than 0.")
            
        # n_jobs_outer checks
        if not isinstance(n_jobs_outer, int):
            raise TypeError("n_jobs_outer must be an integer.")
        if (n_jobs_outer < -1) or (n_jobs_outer == 0):
            raise ValueError("n_jobs_outer must be greater than zero or equal to -1.")
        
        # n_jobs_inner checks
        # TODO: raise warning if the product of n_jobs_outer and n_jobs_inner is greater than the number of available cores
        if not isinstance(n_jobs_inner, int):
            raise TypeError("n_jobs_inner must be an integer.")
        if (n_jobs_inner < -1) or (n_jobs_inner == 0):
            raise ValueError("n_jobs_inner must be greater than zero or equal to -1.")
        
        # Check that the number of jobs isn't excessive
        n_jobs_outer_adjusted = n_jobs_outer if n_jobs_outer != -1 else os.cpu_count()
        n_jobs_inner_adjusted = n_jobs_inner if n_jobs_inner != -1 else os.cpu_count()

        if n_jobs_outer_adjusted * n_jobs_inner_adjusted > os.cpu_count():
            warnings.warn(
                f"n_jobs_outer * n_jobs_inner is greater than the number of available cores. "
                f"Consider reducing the number of jobs to avoid excessive resource usage."
            )

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
        est_freq: str = "D",
        initial_nsplits: Optional[Union[int, np.int_]] = None, # TODO incorporate this logic later
        threshold_ndates: Optional[Union[int, np.int_]] = None, # TODO incorporate this logic later
        hparam_type: str = "grid",
        n_iter: Optional[int] = 10,
        n_jobs_outer: Optional[int] = -1,
        n_jobs_inner: Optional[int] = 1,
    ):
        """
        Estimates and stores beta coefficients for each cross-section.

        Optimal models and hyperparameters are selected at the end of re-estimation date.
        This includes the frequency of the training dataset. The optimal model is then fit 
        to the training set, betas for each cross-section are extracted.  Based in these 
        betas hedged returns are calculated up to the next re-estimation date.
        Model and hyperparameter search is based on maximization of a cross-validation 
        score from a scikit-learn 'scorer' function.

        :param <str> beta_xcat: Category name for the panel of estimated contract betas.
        :param <str> hedged_return_xcat: Category name for the panel of derived hedged returns.
        :param <BasePanelSplit> inner_splitter: Panel cross-validation splitter for
            the hyperparameter search. It is recommended to set this to an ExpandingKFoldPanelSplit
            splitter.
        :param <Callable> scorer: Scikit-learn scorer function used in both model and 
            hyperparameter selection for optimization. For beta estimation, it is recommended
            to use `neg_mean_abs_corr` from the `macrosynergy.learning` submodule. 
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
        :param <str> est_freq: Frequency forward of each training set for which hedged returns
            are derived. After the hedged returns are determined in each iteration, the training
            panel expands to encapsulate these samples, meaning that this parameter also determines
            re-estimation frequency. This parameter can accept any of the strings "D" (daily), 
            "W" (weekly), "M" (monthly) or "Q" (quarterly). Default is "M".
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
        # Checks 
        self._checks_estimate_beta(
            beta_xcat=beta_xcat,
            hedged_return_xcat=hedged_return_xcat,
            inner_splitter=inner_splitter,
            scorer=scorer,
            models=models,
            hparam_grid=hparam_grid,
            min_cids=min_cids,
            min_periods=min_periods,
            est_freq=est_freq,
            initial_nsplits=initial_nsplits,
            threshold_ndates=threshold_ndates,
            hparam_type=hparam_type,
            n_iter=n_iter,
            n_jobs_outer=n_jobs_outer,
            n_jobs_inner=n_jobs_inner,
        )

        # (1) Create daily multi-indexed dataframes to store estimated betas and hedged returns
        # This makes indexing easy and the resulting dataframes can be melted to long format later

        min_date = self.X.index.get_level_values(1).min()
        max_date = self.X.index.get_level_values(1).max()
        date_range = pd.bdate_range(start=min_date, end=max_date, freq="B")
        idxs = pd.MultiIndex.from_product([self.cids, date_range], names=["cid","real_date"])
        
        stored_betas = pd.DataFrame(index=idxs, columns=[beta_xcat], data=np.nan, dtype=np.float64)
        stored_hedged_returns = pd.DataFrame(index=idxs, columns=[hedged_return_xcat], data=np.nan, dtype=np.float64)

        # (2) Set up outer splitter
        outer_splitter = ExpandingFrequencyPanelSplit(
            expansion_freq=est_freq,
            test_freq=est_freq,
            min_cids=min_cids,
            min_periods=min_periods,
        )

        # Create lists to store stored quantities in each iteration
        # This is done to avoid appending to the dataframe in each iteration
        beta_list = []
        hedged_return_list = []
        chosen_models = []

        # (3) Loop through outer splitter, run grid search for optimal model, extract beta 
        # estimates, calculate OOS hedged returns and store results
        train_test_splits = list(outer_splitter.split(X=self.X, y=self.y))

        results = Parallel(n_jobs=n_jobs_outer)(
            delayed(self._worker)(
                train_idx=train_idx,
                test_idx=test_idx,
                beta_xcat=beta_xcat,
                hedged_return_xcat=hedged_return_xcat,
                inner_splitter=inner_splitter,
                models=models,
                scorer=scorer,
                hparam_grid=hparam_grid,
                hparam_type=hparam_type,
                n_iter=n_iter,
                initial_nsplits=initial_nsplits,
                nsplits_add=(
                    np.floor(idx / threshold_ndates)
                    if initial_nsplits
                    else None
                ),
                n_jobs_inner=n_jobs_inner,
            )
            for idx, (train_idx, test_idx) in tqdm(
                enumerate(train_test_splits),
                total=len(train_test_splits),
            )
        )

        for beta_data, hedged_data, model_data in results:
            beta_list.extend(beta_data)
            hedged_return_list.extend(hedged_data)
            chosen_models.append(model_data)

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

    def _worker(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        beta_xcat: str,
        hedged_return_xcat: str,
        inner_splitter: BasePanelSplit,
        models: Dict[str, Union[BaseEstimator, Pipeline]],
        scorer: Callable,
        hparam_grid: Dict[str, Dict[str, List]],
        hparam_type: str = "grid",
        n_iter: int = 10,
        initial_nsplits: Optional[Union[int, np.int_]] = None,
        nsplits_add: Optional[Union[int, np.int_]] = None,
        n_jobs_inner: int = 1,
    ):
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

        # If nsplits_add is provided, add it to the number of splits
        if initial_nsplits:
            n_splits = initial_nsplits + nsplits_add
            inner_splitter.n_splits = int(n_splits)
        else:
            n_splits = inner_splitter.n_splits

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
                # TODO: handle this case properly
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
        hedged_returns: List = self._calculate_hedged_returns(betas, X_test_i, y_test_i, hedged_return_xcat)

        # Compute betas 
        beta_list = [[cid.split("v")[0], training_time, beta_xcat, beta] for cid, beta in betas.items()]

        # Store chosen models and hyperparameters
        models_list = [training_time, beta_xcat, optim_name, optim_params, inner_splitter.n_splits]

        return beta_list, hedged_returns, models_list

    def _calculate_hedged_returns(
        self,
        betas,
        X_test_i,
        y_test_i,
        hedged_return_xcat,
    ):
        # TODO: input checks
        betas_series = pd.Series(betas)
        XB = X_test_i.mul(betas_series, level=0, axis=0)
        hedged_returns = y_test_i - XB[self.benchmark_xcat]
        list_hedged_returns = [[idx[0].split("v")[0], idx[1]] + [hedged_return_xcat] + [value] for idx, value in hedged_returns.items()]

        return list_hedged_returns
    
    def models_heatmap(
        self,
        beta_xcat: str,
        title: Optional[str] = None,
        cap: Optional[int] = 5,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (12, 8),
    ):
        """
        Visualizing optimal models used for beta estimation.

        :param <str> beta_xcat: Category name for the panel of estimated contract betas.
        :param <Optional[str]> title: Title of the heatmap. Default is None. This creates
            a figure title of the form "Model Selection Heatmap for {name}".
        :param <Optional[int]> cap: Maximum number of models to display. Default
            (and limit) is 5. The chosen models are the 'cap' most frequently occurring
            in the pipeline.
        :param <Optional[Tuple[Union[int, float], Union[int, float]]]> figsize: Tuple of
            floats or ints denoting the figure size. Default is (12, 8).

        Note:
        This method displays the times at which each model in a learning process
        has been optimal and used for beta estimation, as a binary heatmap.
        """
        # Checks
        self._checks_models_heatmap(beta_xcat=beta_xcat, title=title, cap=cap, figsize=figsize)

        # Get the chosen models for the specified pipeline to visualise selection.
        chosen_models = self.chosen_models # TODO: implement self.get_optimal_models()
        chosen_models = chosen_models[chosen_models.xcat == beta_xcat].sort_values(
            by="real_date"
        )
        chosen_models["model_hparam_id"] = chosen_models.apply(
            lambda row: (
                row["model_type"]
                if row["hparams"] == {}
                else f"{row['model_type']}_"
                + "_".join([f"{key}={value}" for key, value in row["hparams"].items()])
            ),
            axis=1,
        )
        chosen_models["real_date"] = chosen_models["real_date"].dt.date
        model_counts = chosen_models.model_hparam_id.value_counts()
        chosen_models = chosen_models[
            chosen_models.model_hparam_id.isin(model_counts.index[:cap])
        ]

        unique_models = chosen_models.model_hparam_id.unique()
        unique_dates = chosen_models.real_date.unique()

        # Fill in binary matrix denoting the selected model at each time
        binary_matrix = pd.DataFrame(0, index=unique_models, columns=unique_dates)
        for _, row in chosen_models.iterrows():
            model_id = row["model_hparam_id"]
            date = row["real_date"]
            binary_matrix.at[model_id, date] = 1

        # Display the heatmap.
        plt.figure(figsize=figsize)
        if binary_matrix.shape[0] == 1:
            sns.heatmap(binary_matrix, cmap="binary_r", cbar=False)
        else:
            sns.heatmap(binary_matrix, cmap="binary", cbar=False)
        plt.title(title)
        plt.show()

    def _checks_models_heatmap(
        self,
        beta_xcat: str,
        title: Optional[str] = None,
        cap: Optional[int] = 5,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (12, 8),
    ):
        if not isinstance(beta_xcat, str):
            raise TypeError("beta_xcat must be a string.")
        if beta_xcat not in self.beta_df["xcat"].unique():
            raise ValueError(
                "beta_xcat must be a valid category in the stored beta dataframe."
                "Check that estimate_beta has been run with the specified beta_xcat."
            )
        if title is None:
            title = f"Model Selection Heatmap for {beta_xcat}"
        if not isinstance(title, str):
            raise TypeError("title must be a string.")
        if not isinstance(cap, int):
            raise TypeError("cap must be an integer.")
        if cap < 1:
            raise ValueError("cap must be greater than 0.")
        if cap > 20:
            warnings.warn(
                f"The maximum number of models to display is 20. The cap has been set to "
                "20.",
                RuntimeWarning,
            )
            cap = 20
        if not isinstance(figsize, tuple):
            raise TypeError("figsize must be a tuple.")
        if not all(isinstance(value, (int, float)) for value in figsize):
            raise TypeError("All elements in figsize must be integers or floats.")
        if len(figsize) != 2:
            raise ValueError("figsize must be a tuple of length 2.")
        
    def get_optimal_models(
        self,
        beta_xcat: Optional[Union[str, List]] = None,
    ) -> pd.DataFrame:
        """
        Returns a dataframe of optimal models selected for one or more beta estimation 
        processes.

        :param <Optional[Union[str, List]]> beta_xcat: Category name or list of category names
            for the panel of estimated contract betas. If None, information from all 
            beta estimation processes held within the class instance is returned. Default is None.

        :return: <pd.DataFrame> A dataframe of optimal models selected for given
            beta estimation processes.
        """
        # Checks 
        self._checks_get_optimal_models(beta_xcat=beta_xcat)

        if beta_xcat is None:
            return self.chosen_models
        else:
            return self.chosen_models[self.chosen_models.xcat.isin(beta_xcat)]

    def _checks_get_optimal_models(
        self,
        beta_xcat: Optional[Union[str, List]],
    ):
        if beta_xcat is not None:
            if isinstance(beta_xcat, str):
                beta_xcat = [beta_xcat]
            if not isinstance(beta_xcat, list):
                raise TypeError("beta_xcat must be a string or a list")
            if not all(isinstance(xcat, str) for xcat in beta_xcat):
                raise TypeError("All elements in beta_xcat, when a list, must be strings.")

    def get_hedged_returns(
        self,
        hedged_return_xcat: Optional[Union[str, List]],
    ):
        """
        Returns a dataframe of out-of-sample hedged returns derived from beta estimation processes
        held within the class instance.

        :param <Optional[Union[str, List]]> hedged_return_xcat: Category name or list of category names
            for the panel of derived hedged returns. If None, information from all 
            beta estimation processes held within the class instance is returned. Default is None.

        :return: <pd.DataFrame> A dataframe of out-of-sample hedged returns derived from beta estimation
            processes.
        """
        # Checks
        self._checks_get_hedged_returns(hedged_return_xcat=hedged_return_xcat)

        if hedged_return_xcat is None:
            return self.hedged_returns
        else:
            return self.hedged_returns[self.hedged_returns.xcat.isin(hedged_return_xcat)]

    def _checks_get_hedged_returns(
        self,
        hedged_return_xcat: Optional[str],
    ):
        if hedged_return_xcat is not None:
            if isinstance(hedged_return_xcat, str):
                hedged_return_xcat = [hedged_return_xcat]
            if not isinstance(hedged_return_xcat, list):
                raise TypeError("hedged_return_xcat must be a string or a list")
            if not all(isinstance(xcat, str) for xcat in hedged_return_xcat):
                raise TypeError("All elements in hedged_return_xcat, when a list, must be strings.")
            
    def get_betas(
        self,
        beta_xcat: Optional[Union[str, List]] = None,
    ):
        """
        Returns a dataframe of estimated betas derived from beta estimation processes
        held within the class instance.

        :param <Optional[Union[str, List]]> beta_xcat: Category name or list of category names
            for the panel of estimated contract betas. If None, information from all 
            beta estimation processes held within the class instance is returned. Default is None.

        :return: <pd.DataFrame> A dataframe of estimated betas derived from beta estimation
            processes.
        """
        # Checks
        self._checks_get_betas(beta_xcat=beta_xcat)

        if beta_xcat is None:
            return self.beta_df
        else:
            return self.beta_df[self.beta_df.xcat.isin(beta_xcat)]
        
    def _checks_get_betas(
        self,
        beta_xcat: Optional[Union[str, List]],
    ):
        if beta_xcat is not None:
            if isinstance(beta_xcat, str):
                beta_xcat = [beta_xcat]
            if not isinstance(beta_xcat, list):
                raise TypeError("beta_xcat must be a string or a list")
            if not all(isinstance(xcat, str) for xcat in beta_xcat):
                raise TypeError("All elements in beta_xcat, when a list, must be strings.")

if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    
    from sklearn.base import RegressorMixin
    from sklearn.ensemble import VotingRegressor
    from sklearn.linear_model import LinearRegression 

    from collections import defaultdict
    import scipy.stats as stats

    from macrosynergy.learning.metrics import neg_mean_abs_corr
    from macrosynergy.learning.predictors import SULinearRegression
    
    # Simulate a panel dataset of benchmark and contract returns
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["BENCH_XR", "CONTRACT_XR"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["BENCH_XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CONTRACT_XR"] = ["2001-01-01", "2020-12-31", 0.1, 1, 0, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Initialize the BetaEstimator object
    # Use for the benchmark return: USD_BENCH_XR.
    be = BetaEstimator(
        df=dfd,
        xcat="CONTRACT_XR",
        benchmark_return="USD_BENCH_XR",
        cids=cids,
    )

    # Define the models and grids to search over
    models = {
        "LR_ROLL": SULinearRegression(min_xs_samples=21*3),
    }
    hparam_grid = {
        "LR_ROLL": [
            {"roll": [21, 21 * 3, 21 * 6, 21 * 12], "data_freq" : ["D"]},
            {"roll": [3, 6, 9, 12], "data_freq" : ["M"]},
        ]
    }
    scorer = neg_mean_abs_corr

    # Now estimate the betas
    be.estimate_beta(
        beta_xcat="BETA_NSA",
        hedged_return_xcat="HEDGED_RETURN_NSA",
        inner_splitter=ExpandingKFoldPanelSplit(n_splits = 5),
        scorer=neg_mean_abs_corr,
        models = models,
        hparam_grid = hparam_grid,
        min_cids=4,
        min_periods=21 * 3,
        est_freq="Q",
        n_jobs_outer=-1,
        n_jobs_inner=1,
    )

    # Get the results
    print(be.get_optimal_models())
    print(be.get_betas())
    print(be.get_hedged_returns())

    be.models_heatmap(beta_xcat="BETA_NSA")