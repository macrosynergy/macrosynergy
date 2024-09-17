import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor

from collections import defaultdict
from tqdm.auto import tqdm

# from tqdm import tqdm

from macrosynergy.learning import (
    BasePanelSplit,
    ExpandingFrequencyPanelSplit,
    ExpandingKFoldPanelSplit,
    neg_mean_abs_corr,
)
from macrosynergy.management import categories_df, reduce_df, update_df


class BetaEstimator:
    """
    Beta estimation of a panel of contract returns based on regression.

    Estimates betas with seemingly unrelated linear regression (SUR) sequentially with
    expanding panel data windows. Statistical learning determines the model and
    hyperparameter choices sequentially using a given performance metric. Cross-sectional
    betas and out-of-sample "hedged" returns are stored in quantamental dataframes.

    .. note::

      This class is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle.
    """

    def __init__(
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
        self.df = df
        self.xcat = xcat
        self.cids = sorted(cids)
        self.benchmark_return = benchmark_return
        self.benchmark_cid, self.benchmark_xcat = self.benchmark_return.split("_", 1)

        # Create pseudo-panel
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])

        for cid in self.cids:
            # Extract cross-section contract returns
            dfa = reduce_df(
                df=self.df,
                xcats=[self.xcat],
                cids=[cid],
            )
            # Extract benchmark returns
            dfb = reduce_df(
                df=self.df,
                xcats=[self.benchmark_xcat],
                cids=[self.benchmark_cid],
            )

            # Combine contract and benchmark returns and rename cross-section identifier
            # in order to match the benchmark return with each cross section in a pseudo
            # panel
            df_cid = pd.concat([dfa, dfb], axis=0)
            df_cid["cid"] = f"{cid}v{self.benchmark_cid}"

            dfx = update_df(dfx, df_cid)

        # Create long format dataframes

        Xy_long = categories_df(
            df=dfx,
            xcats=[self.benchmark_xcat, xcat],
            freq="D",
            xcat_aggs=["sum", "sum"],
            lag=0,
        ).dropna()

        self.X = Xy_long[self.benchmark_xcat]
        self.y = Xy_long[xcat]

        # Create dataframes to store the estimated betas, selected models and chosen
        # hyperparameters
        self.betas = pd.DataFrame(columns=["cid", "real_date", "xcat", "value"])
        self.hedged_returns = pd.DataFrame(
            columns=["cid", "real_date", "xcat", "value"]
        )
        self.chosen_models = pd.DataFrame(
            columns=["real_date", "xcat", "model_type", "hparams", "n_splits"]
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
            raise ValueError(
                "df must contain columns 'cid', 'xcat', 'real_date' and 'value'."
            )
        if not df["xcat"].nunique() > 1:
            raise ValueError(
                "df must contain at least two xcats. One is required for the contract return panel and another for the benchmark returns."
            )
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
            raise ValueError(
                "All cids must be valid cross-section identifiers in the dataframe."
            )

        # benchmark return checks
        if not isinstance(benchmark_return, str):
            raise TypeError("benchmark_return must be a string.")
        ticker_list = df["cid"] + "_" + df["xcat"]
        if benchmark_return not in ticker_list.unique():
            raise ValueError(
                "benchmark_return must be a valid ticker in the dataframe."
            )

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
        use_variance_correction,
        initial_n_splits,
        threshold_n_periods,
        hparam_type,
        n_iter,
        n_jobs_outer,
        n_jobs_inner,
    ):
        # beta_xcat checks
        if not isinstance(beta_xcat, str):
            raise TypeError("beta_xcat must be a string.")
        if beta_xcat in self.betas["xcat"].unique():
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
        if not all(
            isinstance(model, (BaseEstimator, Pipeline)) for model in models.values()
        ):
            raise TypeError(
                "All values in the models dictionary must be scikit-learn compatible models."
            )
        if not all(isinstance(model_name, str) for model_name in models.keys()):
            raise TypeError("All keys in the models dictionary must be strings.")
        for model in models.values():
            if isinstance(model, VotingRegressor):
                for estimator in model.estimators:
                    if not hasattr(estimator[1], "coefs_"):
                        raise ValueError(
                            "All models must be seemingly unrelated linear regressors consistent with "
                            "the scikit-learn API. This means that they must be scikit-learn compatible "
                            "regressors that have a 'coefs_' attribute storing the estimated betas for each "
                            "cross-section, as per the standard imposed by the macrosynergy package. "
                            "Please check that the voting regressor estimators have this attribute."
                        )
            else:
                if not hasattr(model, "coefs_"):
                    raise ValueError(
                        "All models must be seemingly unrelated linear regressors consistent with "
                        "the scikit-learn API. This means that they must be scikit-learn compatible "
                        "regressors that have a 'coefs_' attribute storing the estimated betas for each "
                        "cross-section, as per the standard imposed by the macrosynergy package."
                    )

        # hparam_grid checks
        if not isinstance(hparam_grid, dict):
            raise TypeError("hparam_grid must be a dictionary.")
        if not all(isinstance(model_name, str) for model_name in hparam_grid.keys()):
            raise TypeError("All keys in the hparam_grid dictionary must be strings.")
        if not set(models.keys()) == set(hparam_grid.keys()):
            raise ValueError(
                "The keys in the models and hparam_grid dictionaries must match."
            )
        if not all(isinstance(grid, (dict, list)) for grid in hparam_grid.values()):
            raise TypeError(
                "All values in the hparam_grid dictionary must be dictionaries or lists of dictionaries."
            )
        for grid in hparam_grid.values():
            if isinstance(grid, dict):
                if not all(isinstance(key, str) for key in grid.keys()):
                    raise TypeError(
                        "All keys in the nested dictionaries of hparam_grid must be strings."
                    )
                if not all(isinstance(value, list) for value in grid.values()):
                    raise TypeError(
                        "All values in the nested dictionaries of hparam_grid must be lists."
                    )
            if isinstance(grid, list):
                if not all(isinstance(sub_grid, dict) for sub_grid in grid):
                    raise TypeError(
                        "All elements in the list of hparam_grid must be dictionaries."
                    )
                for sub_grid in grid:
                    if not all(isinstance(key, str) for key in sub_grid.keys()):
                        raise TypeError(
                            "All keys in the nested dictionaries of hparam_grid must be strings."
                        )
                    if not all(isinstance(value, list) for value in sub_grid.values()):
                        raise TypeError(
                            "All values in the nested dictionaries of hparam_grid must be lists."
                        )

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

        # use_variance_correction checks
        if not isinstance(use_variance_correction, bool):
            raise TypeError("use_variance_correction must be a boolean.")

        # initial_n_splits checks
        if initial_n_splits is not None:
            if not isinstance(initial_n_splits, int):
                raise TypeError("initial_n_splits must be an integer.")
            if initial_n_splits < 2:
                raise ValueError("initial_n_splits must be greater than 1.")

        # threshold_n_periods checks
        if threshold_n_periods is not None:
            if not isinstance(threshold_n_periods, int):
                raise TypeError("threshold_n_periods must be an integer.")
            if threshold_n_periods < 1:
                raise ValueError("threshold_n_periods must be greater than 0.")

        # hparam_type checks
        if not isinstance(hparam_type, str):
            raise TypeError("hparam_type must be a string.")
        if hparam_type not in ["grid", "prior", "bayes"]:
            raise ValueError("hparam_type must be one of 'grid', 'prior' or 'bayes'.")
        if hparam_type == "bayes":
            raise NotImplementedError(
                "Bayesian hyperparameter search is not yet implemented."
            )

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
        scorer: Callable,  # Possibly find a better type hint for a scikit-learn scorer
        models: Dict[str, Union[BaseEstimator, Pipeline]],
        hparam_grid: Dict[str, Dict[str, List]],
        min_cids: int = 1,
        min_periods: int = 252,
        est_freq: str = "D",
        use_variance_correction: bool = False,
        initial_n_splits: Optional[Union[int, np.int_]] = None,
        threshold_n_periods: Optional[Union[int, np.int_]] = None,
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
        :param <bool> use_variance_correction: Boolean indicating whether or not to apply a
            correction to cross-validation scores to account for the variation in scores across
            splits. Default is False.
        :param <int> initial_n_splits: Number of splits to be used in cross-validation for the initial
            training set. If None, the number of splits is defined by the inner_splitter. Default is None.
        :param <int> threshold_n_periods: Number of periods, in units of the specified "est_freq"
            to pass before the number of cross-validation splits increases by one. Default is None.
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
            use_variance_correction=use_variance_correction,
            initial_n_splits=initial_n_splits,
            threshold_n_periods=threshold_n_periods,
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
        idxs = pd.MultiIndex.from_product(
            [self.cids, date_range], names=["cid", "real_date"]
        )

        stored_betas = pd.DataFrame(
            index=idxs, columns=[beta_xcat], data=np.nan, dtype=np.float64
        )
        stored_hedged_returns = pd.DataFrame(
            index=idxs, columns=[hedged_return_xcat], data=np.nan, dtype=np.float64
        )

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
                use_variance_correction=use_variance_correction,
                initial_n_splits=initial_n_splits,
                nsplits_add=(
                    np.floor(idx / threshold_n_periods) if initial_n_splits else None
                ),
                n_jobs_inner=n_jobs_inner,
            )
            for idx, (train_idx, test_idx) in tqdm(
                enumerate(train_test_splits),
                total=len(train_test_splits),
            )
        )

        for beta_data, hedged_data, model_data in results:
            if beta_data != []:
                beta_list.extend(beta_data)
            if hedged_data != []:
                hedged_return_list.extend(hedged_data)
            chosen_models.append(model_data)

        for cid, real_date, xcat, value in beta_list:
            stored_betas.loc[(cid, real_date), xcat] = value
        for cid, real_date, xcat, value in hedged_return_list:
            stored_hedged_returns.loc[(cid, real_date), xcat] = value

        stored_betas = stored_betas.groupby(level=0).ffill().dropna()
        stored_hedged_returns = stored_hedged_returns.dropna()
        stored_betas_long = pd.melt(
            frame=stored_betas.reset_index(),
            id_vars=["cid", "real_date"],
            var_name="xcat",
            value_name="value",
        )
        stored_hrets_long = pd.melt(
            frame=stored_hedged_returns.reset_index(),
            id_vars=["cid", "real_date"],
            var_name="xcat",
            value_name="value",
        )

        self.betas = pd.concat((self.betas, stored_betas_long), axis=0).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float64,
            }
        )
        self.hedged_returns = pd.concat(
            (self.hedged_returns, stored_hrets_long), axis=0
        ).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float64,
            }
        )

        self.chosen_models = pd.concat(
            (
                self.chosen_models,
                pd.DataFrame(
                    chosen_models,
                    columns=[
                        "real_date",
                        "xcat",
                        "model_type",
                        "hparams",
                        "n_splits",
                    ],
                ),
            )
        )

    def evaluate_hedged_returns(
        self,
        hedged_rets: Optional[Union[str, List[str]]] = None,
        cids: Optional[Union[str, List[str]]] = None,
        correlation_types: Union[str, List[str]] = "pearson",
        title: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
        freqs: Optional[Union[str, List[str]]] = "M",
    ):
        """
        Method to determine and display a table of average absolute correlations between
        the benchmark return and the computed hedged returns within the class instance, over
        all cross-sections in the panel. Additionally, the correlation table displays the
        same results for the unhedged return specified in the class instance for comparison
        purposes.

        The returned dataframe will be multi-indexed by (benchmark return, return, frequency)
        and will contain each computed absolute correlation coefficient on each column.

        :param <Optional[Union[str, List[str]]> hedged_rets: String or list of strings denoting the hedged returns to be
            evaluated. Default is None, which evaluates all hedged returns within the class instance.
        :param <Optional[Union[str, List[str]]> cids: String or list of strings denoting the cross-sections to evaluate.
            Default is None, which evaluates all cross-sections within the class instance.
        :param <Union[str, List[str] correlation_types: String or list of strings denoting the types of correlations
            to calculate. Options are "pearson", "spearman" and "kendall". If None, all three
            are calculated. Default is "pearson".
        :param <Optional[str]> title: Title for the correlation table. If None, the default
            title is "Average absolute correlations between each return and the chosen benchmark". Default is None.
        :param <Optional[str]> start: String in ISO format. Default is None.
        :param <Optional[str]> end: String in ISO format. Default is None.
        :param <Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]> blacklist: Dictionary of tuples of start and end
            dates to exclude from the evaluation. Default is None.
        :param <Optional[Union[str, List[str]]> freqs: Letters denoting all frequencies
            at which the series may be sampled. This must be a selection of "D", "W", "M", "Q"
            and "A". Default is "M". Each return series will always be summed over the sample
            period.
        """
        # Checks
        self._checks_evaluate_hedged_returns(
            correlation_types=correlation_types,
            hedged_rets=hedged_rets,
            cids=cids,
            start=start,
            end=end,
            blacklist=blacklist,
            freqs=freqs,
        )

        # Parameter handling
        if correlation_types is None:
            correlation_types = ["pearson", "spearman", "kendall"]
        elif isinstance(correlation_types, str):
            correlation_types = [correlation_types]
        if hedged_rets is None:
            hedged_rets = list(self.hedged_returns["xcat"].unique())
        elif isinstance(hedged_rets, str):
            hedged_rets = [hedged_rets]
        if cids is None:
            cids = list(self.hedged_returns["cid"].unique())
        elif isinstance(cids, str):
            cids = [cids]
        if isinstance(freqs, str):
            freqs = [freqs]

        # Construct a quantamental dataframe comprising specified hedged returns as well
        # as the unhedged returns and the benchmark return specified in the class instance
        hedged_df = self.hedged_returns[
            (self.hedged_returns["xcat"].isin(hedged_rets))
            & (self.hedged_returns["cid"].isin(cids))
        ]
        unhedged_df = self.df[
            (self.df["xcat"] == self.xcat) & (self.df["cid"].isin(cids))
        ]
        benchmark_df = self.df[
            (self.df["xcat"] == self.benchmark_xcat)
            & (self.df["cid"] == self.benchmark_cid)
        ]
        combined_df = pd.concat([hedged_df, unhedged_df], axis=0)

        # Create a pseudo-panel to match contract return cross-sections with a replicated
        # benchmark return. This is multi-indexed by (new cid, real_date). The columns
        # are the named hedged returns, with the final column being the benchmark category.
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])

        for cid in cids:
            # Extract unhedged and hedged returns
            dfa = reduce_df(
                df=combined_df,
                xcats=hedged_rets + [self.xcat],
                cids=[cid],
            )
            # Extract benchmark returns
            dfb = reduce_df(
                df=benchmark_df,
                xcats=[self.benchmark_xcat],
                cids=[self.benchmark_cid],
            )
            # Combine and rename cross-section
            df_cid = pd.concat([dfa, dfb], axis=0)
            df_cid["cid"] = f"{cid}v{self.benchmark_cid}"

            dfx = update_df(dfx, df_cid)

        # Create long format dataframes for each specified frequency
        Xy_long_freq = []
        for freq in freqs:
            Xy_long = categories_df(
                df=dfx,
                xcats=hedged_rets + [self.xcat, self.benchmark_xcat],
                cids=[f"{cid}v{self.benchmark_cid}" for cid in cids],
                start=start,
                end=end,
                blacklist=blacklist,
                freq=freq,
                xcat_aggs=["sum", "sum"],
            )
            Xy_long_freq.append(Xy_long)

        # For each xcat and frequency, calculate the mean absolute correlations
        # between the benchmark return and the (hedged and unhedged) market returns
        df_rows = []
        for xcat in hedged_rets + [self.xcat]:
            for freq, Xy_long in zip(freqs, Xy_long_freq):
                calculated_correlations = []
                for correlation in correlation_types:
                    calculated_correlations.append(
                        self._get_mean_abs_corrs(
                            xcat=xcat,
                            df=Xy_long,
                            correlation=correlation,
                            cids=cids,
                        )
                    )
                df_rows.append(calculated_correlations)
        # Create underlying dataframe to store the results
        multiindex = pd.MultiIndex.from_product(
            [[self.benchmark_return], hedged_rets + [self.xcat], freqs],
            names=["benchmark return", "return category", "frequency"],
        )
        corr_df = pd.DataFrame(
            columns=["|" + correlation + "|" for correlation in correlation_types],
            index=multiindex,
            data=df_rows,
        )

        return corr_df

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
        use_variance_correction: bool = False,
        initial_n_splits: Optional[Union[int, np.int_]] = None,
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
        optim_params = None
        optim_score = -np.inf

        # If nsplits_add is provided, add it to the number of splits
        if initial_n_splits:
            n_splits = initial_n_splits + nsplits_add
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

            try:
                search_object.fit(X_train_i, y_train_i)
            except Exception as e:
                warnings.warn(
                    f"Error running a hyperparameter search for {model_name} at {training_time}: {e}",
                    RuntimeWarning,
                )
                continue
            # if not hasattr(search_object, "best_score_"):
            #    # Then the grid search failed completely
            #    warnings.warn(
            #        f"No model was selected at time {training_time}. Hence, no beta can be estimated."
            #    )
            #    # TODO: handle this case properly
            #    continue

            # If a model was selected, extract the score, name, estimator and optimal hyperparameters
            optim_name, optim_model, optim_score, optim_params = (
                self._determine_optimal_model(
                    model_name=model_name,
                    model=model,
                    search_object=search_object,
                    optim_score=optim_score,
                    optim_name=optim_name,
                    optim_model=optim_model,
                    optim_params=optim_params,
                    use_variance_correction=use_variance_correction,
                )
            )

        # Handle case where no model was chosen
        if optim_model is None:
            warnings.warn(
                f"No model was selected at time {training_time}. Hence, no beta can be estimated."
            )
            models_list = [
                training_time,
                beta_xcat,
                "None",
                {},
                inner_splitter.n_splits,
            ]
            beta_list = [
                [cid.split("v")[0], training_time, beta_xcat, np.nan]
                for cid in self.cids
            ]
            hedged_returns = [
                [cid.split("v")[0], training_time] + [hedged_return_xcat] + [np.nan]
                for cid in self.cids
            ]
        else:
            # Get beta estimates for each cross-section
            optim_model.fit(X_train_i, y_train_i)
            if isinstance(optim_model, VotingRegressor):
                estimators = optim_model.estimators_
                coefs_list = [est.coefs_ for est in estimators]
                sum_dict = defaultdict(lambda: [0, 0])

                for coefs in coefs_list:
                    for key, value in coefs.items():
                        sum_dict[key][0] += value
                        sum_dict[key][1] += 1

                betas = {key: sum / count for key, (sum, count) in sum_dict.items()}
            else:
                betas = optim_model.coefs_

            # Get OOS hedged returns
            # This will be a List of lists, with inner lists recording the hedged return
            # for a given cross-section and a given OOS timestamp.
            hedged_returns: List = self._calculate_hedged_returns(
                betas, X_test_i, y_test_i, hedged_return_xcat
            )

            # Compute betas
            beta_list = [
                [cid.split("v")[0], training_time, beta_xcat, beta]
                for cid, beta in betas.items()
            ]

            # Store chosen models and hyperparameters
            models_list = [
                training_time,
                beta_xcat,
                optim_name,
                optim_params,
                inner_splitter.n_splits,
            ]

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
        list_hedged_returns = [
            [idx[0].split("v")[0], idx[1]] + [hedged_return_xcat] + [value]
            for idx, value in hedged_returns.items()
        ]

        return list_hedged_returns

    def _get_mean_abs_corrs(
        self,
        xcat: str,
        cids: str,
        df: pd.DataFrame,
        correlation: pd.DataFrame,
    ):
        """
        Private helper method to calculate the mean absolute correlation between a column
        'xcat' in a dataframe 'df' and the benchmark return (the last column) across all
        cross-sections in 'cids'. The correlation is calculated using the method specified
        in 'correlation'.
        """
        # Get relevant columns
        df_subset = df[[xcat, self.benchmark_xcat]].dropna()

        # Create inner function to calculate the correlation for a given cross-section
        # This is done so that one can groupby cross-section and apply this function directly

        def calculate_correlation(group):
            return abs(group[xcat].corr(group[self.benchmark_xcat], method=correlation))

        # Calculate the mean absolute correlation over all cross sections
        mean_abs_corr = df_subset.groupby("cid").apply(calculate_correlation).mean()

        return mean_abs_corr

    def _checks_evaluate_hedged_returns(
        self,
        correlation_types: Union[str, List[str]],
        hedged_rets: Optional[Union[str, List[str]]],
        cids: Optional[Union[str, List[str]]],
        start: Optional[str],
        end: Optional[str],
        blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]],
        freqs: Optional[Union[str, List[str]]],
    ):
        # correlation_types checks
        if (correlation_types is not None) and (
            not isinstance(correlation_types, (str, list))
        ):
            raise TypeError("correlation_types must be a string or a list of strings.")
        if isinstance(correlation_types, list):
            if not all(
                isinstance(correlation_type, str)
                for correlation_type in correlation_types
            ):
                raise TypeError("All elements in correlation_types must be strings.")
            if not all(
                correlation_type in ["pearson", "spearman", "kendall"]
                for correlation_type in correlation_types
            ):
                raise ValueError(
                    "All elements in correlation_types must be one of 'pearson', 'spearman' or 'kendall'."
                )
        else:
            if correlation_types not in ["pearson", "spearman", "kendall"]:
                raise ValueError(
                    "correlation_types must be one of 'pearson', 'spearman' or 'kendall'."
                )

        # hedged_rets checks
        if hedged_rets is not None:
            if not isinstance(hedged_rets, (str, list)):
                raise TypeError("hedged_rets must be a string or a list of strings.")
            if isinstance(hedged_rets, list):
                if not all(isinstance(hedged_ret, str) for hedged_ret in hedged_rets):
                    raise TypeError("All elements in hedged_rets must be strings.")
                if not all(
                    hedged_ret in self.hedged_returns["xcat"].unique()
                    for hedged_ret in hedged_rets
                ):
                    raise ValueError(
                        "All hedged_rets must be valid hedged return categories within the class instance."
                    )
            else:
                if hedged_rets not in self.hedged_returns["xcat"].unique():
                    raise ValueError(
                        "hedged_rets must be a valid hedged return category within the class instance."
                    )

        # cids checks
        if cids is not None:
            if not isinstance(cids, (str, list)):
                raise TypeError("cids must be a string or a list of strings.")
            if isinstance(cids, list):
                if not all(isinstance(cid, str) for cid in cids):
                    raise TypeError("All elements in cids must be strings.")
                if not all(cid in self.cids for cid in cids):
                    raise ValueError(
                        "All cids must be valid cross-section identifiers within the class instance."
                    )
            else:
                if cids not in self.cids:
                    raise ValueError(
                        "cids must be a valid cross-section identifier within the class instance."
                    )

        # start checks
        if start is not None:
            if not isinstance(start, str):
                raise TypeError("start must be a string.")

        # end checks
        if end is not None:
            if not isinstance(end, str):
                raise TypeError("end must be a string.")

        # blacklist checks
        if blacklist is not None:
            if not isinstance(blacklist, dict):
                raise TypeError("The blacklist argument must be a dictionary.")
            for key, value in blacklist.items():
                # check keys are strings
                if not isinstance(key, str):
                    raise TypeError(
                        "The keys of the blacklist argument must be strings."
                    )
                # check values of tuples of length two
                if not isinstance(value, tuple):
                    raise TypeError(
                        "The values of the blacklist argument must be tuples."
                    )
                if len(value) != 2:
                    raise ValueError(
                        "The values of the blacklist argument must be tuples of length "
                        "two."
                    )
                # ensure each of the dates in the dictionary are timestamps
                for date in value:
                    if not isinstance(date, pd.Timestamp):
                        raise TypeError(
                            "The values of the blacklist argument must be tuples of "
                            "pandas Timestamps."
                        )

        # freqs checks
        if freqs is not None:
            if not isinstance(freqs, (str, list)):
                raise TypeError("freqs must be a string or a list of strings.")
            if isinstance(freqs, list):
                if not all(isinstance(freq, str) for freq in freqs):
                    raise TypeError("All elements in freqs must be strings.")
                if not all(freq in ["D", "W", "M", "Q"] for freq in freqs):
                    raise ValueError(
                        "All elements in freqs must be one of 'D', 'W', 'M' or 'Q'."
                    )
            else:
                if freqs not in ["D", "W", "M", "Q"]:
                    raise ValueError("freqs must be one of 'D', 'W', 'M' or 'Q'.")

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
        self._checks_models_heatmap(
            beta_xcat=beta_xcat, title=title, cap=cap, figsize=figsize
        )

        # Get the chosen models for the specified pipeline to visualise selection.
        chosen_models = self.chosen_models  # TODO: implement self.get_optimal_models()
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
        unique_models = sorted(unique_models, key=lambda x: -model_counts[x])
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

    def _determine_optimal_model(
        self,
        model_name: str,
        model: Union[BaseEstimator, Pipeline],
        search_object: Union[GridSearchCV, RandomizedSearchCV],
        optim_score: float,
        optim_name: str,
        optim_model: Union[BaseEstimator, Pipeline],
        optim_params: Dict,
        use_variance_correction: bool,
    ):
        """
        Private method to determine an optimal model based on cross-validation scores
        at any given estimation time. A given model, with associated model_name, is
        compared to the current optimal model. If the current model has a higher score
        than the optimal model, the current model becomes the optimal model. The new
        optimal score, model name, model and hyperparameters are returned.

        :param <str> model_name: Name of the model being considered.
        :param <Union[BaseEstimator, Pipeline]> model: Model being considered.
        :param <Union[GridSearchCV, RandomizedSearchCV]> search_object: Search object
            containing the results of the hyperparameter search.
        :param <float> optim_score: Current optimal score.
        :param <str> optim_name: Current optimal model name.
        :param <Union[BaseEstimator, Pipeline]> optim_model: Current optimal model.
        :param <Dict> optim_params: Current optimal hyperparameters.
        :param <bool> use_variance_correction: Boolean indicating whether or not to apply a
            correction to cross-validation scores to account for the variation in scores across
            splits.


        """
        if not hasattr(search_object, "cv_results_"):
            return optim_name, optim_model, optim_score, optim_params
        cv_scores = pd.DataFrame(search_object.cv_results_)
        splitscorecols = [
            col
            for col in cv_scores.columns
            if (("split" in col) and ("_test_score" in col))
        ]
        cv_scores = cv_scores[splitscorecols + ["params"]]

        mean_scores = np.nanmean(cv_scores.iloc[:, :-1], axis=1)
        std_scores = np.nanstd(cv_scores.iloc[:, :-1], axis=1)

        if all(np.isnan(mean_scores)):
            return optim_name, optim_model, optim_score, optim_params

        if use_variance_correction:
            # Select model that aims to jointly maximize the mean metric and minimize the
            # standard deviation across splits. Equal weighting is placed on both.
            scaled_mean_scores = (mean_scores - np.nanmin(mean_scores)) / (
                np.nanmax(mean_scores) - np.nanmin(mean_scores)
            )
            scaled_std_scores = (std_scores - np.nanmin(std_scores)) / (
                np.nanmax(std_scores) - np.nanmin(std_scores)
            )
            equalized_scores = (scaled_mean_scores - scaled_std_scores) / 2
            best_index = np.nanargmax(equalized_scores)
            score = equalized_scores[best_index]
        else:
            best_index = np.nanargmax(mean_scores)
            score = mean_scores[best_index]

        if score > optim_score:
            optim_name = model_name
            optim_score = score
            optim_params = cv_scores["params"].values[best_index]
            optim_model = clone(model).set_params(**clone(optim_params, safe=False))

        return optim_name, optim_model, optim_score, optim_params

    def _checks_models_heatmap(
        self,
        beta_xcat: str,
        title: Optional[str] = None,
        cap: Optional[int] = 5,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (12, 8),
    ):
        if not isinstance(beta_xcat, str):
            raise TypeError("beta_xcat must be a string.")
        if beta_xcat not in self.betas["xcat"].unique():
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
                raise TypeError(
                    "All elements in beta_xcat, when a list, must be strings."
                )

    def get_hedged_returns(
        self,
        hedged_return_xcat: Optional[Union[str, List]] = None,
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
            return self.hedged_returns[
                self.hedged_returns.xcat.isin(hedged_return_xcat)
            ]

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
                raise TypeError(
                    "All elements in hedged_return_xcat, when a list, must be strings."
                )

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
            return self.betas
        else:
            return self.betas[self.betas.xcat.isin(beta_xcat)]

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
                raise TypeError(
                    "All elements in beta_xcat, when a list, must be strings."
                )


if __name__ == "__main__":
    from metrics import neg_mean_abs_corr
    from predictors import (
        LADRegressionSystem,
        LinearRegressionSystem,
        CorrelationVolatilitySystem,
    )
    from sklearn.ensemble import VotingRegressor
    from macrosynergy.management.simulate import make_qdf

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

    models = {
        "LR": LADRegressionSystem(min_xs_samples=21 * 3),
    }
    hparam_grid = {"LR": {"fit_intercept": [True, False], "positive": [True, False]}}

    scorer = neg_mean_abs_corr

    be.estimate_beta(
        beta_xcat="BETA_NSA",
        hedged_return_xcat="HEDGED_RETURN_NSA",
        inner_splitter=ExpandingKFoldPanelSplit(n_splits=5),
        scorer=scorer,
        models=models,
        hparam_grid=hparam_grid,
        min_cids=1,
        min_periods=21 * 12,
        est_freq="Q",
        use_variance_correction=False,
        n_jobs_outer=-1,
        n_jobs_inner=1,
    )

    be.models_heatmap(beta_xcat="BETA_NSA")

    evaluation_df = be.evaluate_hedged_returns(
        correlation_types=["pearson", "spearman", "kendall"],
        freqs=["W", "M", "Q"],
    )

    print(evaluation_df)

    models = {
        "VOTE": VotingRegressor(
            [
                ("LR1", LinearRegressionSystem(min_xs_samples=21, data_freq="D")),
                ("LR2", LinearRegressionSystem(min_xs_samples=21, data_freq="W")),
            ]
        )
    }

    hparam_grid = {
        "VOTE": {
            "LR1__roll": [21, 21 * 3, 21 * 6, 21 * 12],
            "LR2__roll": [3, 6, 9, 12],
        },
    }

    scorer = neg_mean_abs_corr

    be.estimate_beta(
        beta_xcat="BETA_NSA",
        hedged_return_xcat="HEDGED_RETURN_NSA",
        inner_splitter=ExpandingKFoldPanelSplit(n_splits=5),
        scorer=neg_mean_abs_corr,
        models=models,
        hparam_grid=hparam_grid,
        min_cids=1,
        min_periods=21 * 12,
        est_freq="Q",
        use_variance_correction=False,
        n_jobs_outer=-1,
        n_jobs_inner=1,
    )

    print(be.get_optimal_models())
    print(be.get_betas())
    print(be.get_hedged_returns())
    be.models_heatmap(beta_xcat="BETA_NSA")

    """models = {
        "CORRVOL": CorrelationVolatilitySystem(
            min_xs_samples=21 * 3,
        ),
    }
    hparam_grid = {
        "CORRVOL": [
            #{
            #    "correlation_lookback": [21*12, 21 * 12 * 2, 21 * 12 * 5, 21 * 12 * 10, None],
            #    "correlation_type": ["pearson", "kendall", "spearman"],
            #    "volatility_lookback": [5, 10, 21, 21 * 3, 21 * 6, 21 * 12],
            #    "volatility_window_type": ["exponential", "rolling"],
            #    "data_freq" : ["D"]
            #},
            {
                "correlation_lookback": [4*12, 4 * 12 * 2, 4 * 12 * 5, 4 * 12 * 10, None],
                "correlation_type": ["pearson", "kendall", "spearman"],
                "volatility_lookback": [2, 4, 4 * 3, 4 * 6, 4 * 12],
                "volatility_window_type": ["exponential", "rolling"],
                "data_freq" : ["W"]
            },
        ]
    }
    scorer = neg_mean_abs_corr

    be.estimate_beta(
        beta_xcat="BETA_NSA",
        hedged_return_xcat="HEDGED_RETURN_NSA",
        inner_splitter=ExpandingKFoldPanelSplit(n_splits = 5),
        scorer=neg_mean_abs_corr,
        models = models,
        hparam_grid = hparam_grid,
        min_cids=1,
        min_periods=21 * 3,
        est_freq="Q",
        use_variance_correction=False,
        n_jobs_outer=-1,
        n_jobs_inner=1,
    )

    print(be.get_optimal_models())
    print(be.get_betas())
    print(be.get_hedged_returns())

    be.models_heatmap(beta_xcat="BETA_NSA")
    # Define the models and grids to search over
    models = {
        "LR_ROLL": LinearRegressionSystem(min_xs_samples=21 * 3),
    }
    hparam_grid = {
        "LR_ROLL": [
            {"roll": [21, 21 * 3, 21 * 6, 21 * 12], "data_freq": ["D"]},
            {"roll": [3, 6, 9, 12], "data_freq": ["M"]},
        ]
    }
    scorer = neg_mean_abs_corr

    # Now estimate the betas
    be.estimate_beta(
        beta_xcat="BETA_NSA",
        hedged_return_xcat="HEDGED_RETURN_NSA",
        inner_splitter=ExpandingKFoldPanelSplit(n_splits=5),
        scorer=neg_mean_abs_corr,
        models=models,
        hparam_grid=hparam_grid,
        min_cids=4,
        min_periods=21 * 3,
        est_freq="Q",
        use_variance_correction=True,
        initial_n_splits=5,
        threshold_n_periods=4,
        n_jobs_outer=-1,
        n_jobs_inner=1,
    )

    # Get the results
    print(be.get_optimal_models())
    print(be.get_betas())
    print(be.get_hedged_returns())

    be.models_heatmap(beta_xcat="BETA_NSA")"""
