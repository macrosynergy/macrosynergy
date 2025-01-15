"""
Class to estimate market betas and calculate out-of-sample hedged returns based on
sequential learning. 
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor

from macrosynergy.learning.forecasting.model_systems.base_regression_system import (
    BaseRegressionSystem,
)
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.learning import ExpandingFrequencyPanelSplit
from macrosynergy.learning.sequential import BasePanelLearner
from macrosynergy.management import categories_df, reduce_df, update_df
from macrosynergy.management.utils.df_utils import (
    concat_categorical,
    _insert_as_categorical,
)


class BetaEstimator(BasePanelLearner):
    """
    Class for sequential beta estimation by learning optimal regression coefficients.
    Out-of-sample hedged returns are additionally calculated and stored.

    Parameters
    ----------
    df : pd.DataFrame
        Daily quantamental dataframe in JPMaQS format containing a panel of features, as
        well as a panel of returns.
    xcats : str or list
        Name of a market return category within the panel specified in `df`.
    benchmark_return : str
        Name of the benchmark return ticker within the panel specified in `df`.
    cids : list, optional
        List of cross-sections for which hedged returns are to be calculated.
        Default is None, which calculates hedged returns for all cross-sections in the
        return panel.
    start : str, optional
        Start date for considered data in subsequent analysis in ISO 8601 format.
        Default is None i.e. the earliest date in the dataframe.
    end : str, optional
        End date for considered data in subsequent analysis in ISO 8601 format.
        Default is None i.e. the latest date in the dataframe.

    Notes
    -----
    The `BetaEstimator` class is used to sequentially estimate macro betas based on a
    panel of contract returns (provided in `xcats`) and a benchmark return ticker
    (provided in `benchmark_return`). The initial conditions of the learning process
    are specified by the dimensions of an initial training set. An optimal model is
    selected out of a provided collection (with associated hyperparameters), a beta is
    extracted for each cross-section (subject to availability) and out-of-sample hedged
    returns are calculated for each cross-section with an estimated beta. The betas and
    hedged returns are stored, and the training set is expanded to include the now-realized
    returns. This process is repeated until the end of the dataset is reached.

    In addition to storing betas and hedged returns, this class also stores useful model
    selection information for analysis, such as the models selected at each point in time.

    Model and hyperparameter selection is performed by cross-validation. Given a
    collection of models and associated hyperparameters to choose from, an HPO is run
    - currently only grid search and random search are supported - to determine the
    optimal choice. This is done by providing a collection of `scikit-learn` compatible
    scoring functions, as well as a collection of `scikit-learn` compatible
    cross-validation splitters and scorers. At each point in time, the cross-validation
    folds are the union of the folds produced by each splitter provided. Each scorer is
    evaluated on each test fold and summarised across test folds by either a custom
    function provided by the user or a common string i.e. 'mean'.

    Consequently, each model and hyperparameter combination has an associated collection
    of scores induced by different metrics, in units of those scorers. In order to form a
    composite score for each hyperparameter, the scores must be normalized across
    model/hyperparameter combinations. This makes scores across scorers comparable, so
    that the average score across adjusted scores can be used as a meaningful estimate
    of each model's generalization ability. Finally, a composite score for each model and
    hyperparameter combination is calculated by averaging the adjusted scores across all
    scorers.

    The optimal model is the one with the largest composite score.
    """

    def __init__(
        self,
        df,
        xcats,
        benchmark_return,
        cids=None,
        start=None,
        end=None,
    ):
        # Checks
        # TODO: Refactor these checks.
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not set(["cid", "xcat", "real_date", "value"]).issubset(df.columns):
            raise ValueError(
                "df must have columns 'cid', 'xcat', 'real_date' and 'value'."
            )

        # cids checks
        if cids is not None:
            if not isinstance(cids, list):
                raise TypeError("cids must be a list.")
            if not all(isinstance(cid, str) for cid in cids):
                raise TypeError("All elements in cids must be strings.")
            for cid in cids:
                if cid not in df["cid"].unique():
                    raise ValueError(f"{cid} not in the dataframe.")

        if not isinstance(benchmark_return, str):
            raise TypeError("benchmark_return must be a string.")

        if isinstance(xcats, str):
            xcats = [xcats]
        elif isinstance(xcats, list):
            if not all(isinstance(xcat, str) for xcat in xcats):
                raise TypeError("All elements in xcats must be strings.")
            elif len(xcats) != 1:
                raise ValueError("xcats must be a string or a list of a single xcat.")
        else:
            raise TypeError("xcats must be a string or a list of a single xcat.")

        # Create pseudo-panel
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])

        self.benchmark_return = benchmark_return
        self.benchmark_cid, self.benchmark_xcat = benchmark_return.split("_", 1)

        for cid in cids:
            # Extract cross-section contract returns
            dfa = reduce_df(
                df=df,
                xcats=xcats if isinstance(xcats, list) else [xcats],
                cids=[cid],
            )
            # Extract benchmark returns
            dfb = reduce_df(
                df=df,
                xcats=[self.benchmark_xcat],
                cids=[self.benchmark_cid],
            )

            # Combine contract and benchmark returns and rename cross-section identifier
            # in order to match the benchmark return with each cross section in a pseudo
            # panel
            df_cid = pd.concat([dfa, dfb], axis=0)
            df_cid["cid"] = f"{cid}v{self.benchmark_cid}"

            dfx = update_df(dfx, df_cid)

        super().__init__(
            df=dfx,
            xcats=(
                [self.benchmark_xcat] + xcats
                if isinstance(xcats, list)
                else [self.benchmark_xcat, xcats]
            ),
            cids=list(dfx["cid"].unique()),
            start=start,
            end=end,
            blacklist=None,
            freq="D",
            lag=0,
        )

        # Create forecast dataframe index
        min_date = min(self.unique_date_levels)
        max_date = max(self.unique_date_levels)
        forecast_date_levels = pd.date_range(start=min_date, end=max_date, freq="B")
        self.forecast_idxs = pd.MultiIndex.from_product(
            [
                [cid.split("v")[0] for cid in self.unique_xs_levels],
                forecast_date_levels,
            ],
            names=["cid", "real_date"],
        )

        # Create initial dataframes to store estimated betas and OOS hedged returns
        self.betas = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"]).astype(
            {
                "real_date": "datetime64[s]",
                "cid": "category",
                "xcat": "category",
                "value": "float32",
            }
        )
        self.hedged_returns = pd.DataFrame(
            columns=["real_date", "cid", "xcat", "value"]
        ).astype(
            {
                "real_date": "datetime64[s]",
                "cid": "category",
                "xcat": "category",
                "value": "float32",
            }
        )

    def estimate_beta(
        self,
        beta_xcat,
        hedged_return_xcat,
        models,
        hyperparameters,
        scorers,
        inner_splitters,
        search_type="grid",
        normalize_fold_results=False,
        cv_summary="mean",
        min_cids=4,
        min_periods=12 * 3,
        est_freq="D",
        max_periods=None,
        split_functions=None,
        n_iter=None,
        n_jobs_outer=-1,
        n_jobs_inner=1,
    ):
        """
        Determines optimal model betas and associated out-of-sample hedged returns.

        Parameters
        ----------
        beta_xcat : str
            Category name for the panel of estimated betas.
        hedged_return_xcat : str
            Category name for the panel of out-of-sample hedged returns.
        models : dict
            Dictionary of models to choose from. The keys are model names and the values
            are scikit-learn compatible models.
        hyperparameters : dict
            Dictionary of hyperparameters to choose from. The keys are model names and
            the values are hyperparameter dictionaries for the corresponding model. The
            keys must match with those provided in `models`.
        scorers : dict
            Dictionary of scoring functions to use in the hyperparameter optimization
            process. The keys are scorer names and the values are scikit-learn compatible
            scoring functions.
        inner_splitters : dict
            Dictionary of inner splitters to use in the hyperparameter optimization
            process. The keys are splitter names and the values are scikit-learn compatible
            cross-validator objects.
        search_type : str
            Type of hyperparameter optimization to perform. Default is "grid". Options are
            "grid" and "prior".
        normalize_fold_results : bool
            Whether to normalize the scores across folds before combining them. Default is
            False.
        cv_summary : str or callable
            Summary function to use to combine scores across cross-validation folds.
            Default is "mean". Options are "mean", "median" or a callable function.
        min_cids : int
            Minimum number of cross-sections required for the initial
            training set. Default is 4.
        min_periods : int
            Minimum number of periods required for the initial training set, in units of
            the frequency `freq` specified in the constructor. Default is 36.
        est_freq : str
            Frequency at which models are refreshed. This corresponds with forward
            frequency of out-of-sample hedged returns and the frequency at which betas
            are estimated.
        max_periods : int
            Maximum length of each training set in units of the frequency `freq` specified
            in the constructor. Default is None, in which case the sequential optimization
            uses expanding training sets, as opposed to rolling windows.
        split_functions : dict, optional
            Dict of callables for determining the number of cross-validation
            splits to add to the initial number as a function of the number of iterations
            passed in the sequential learning process. Default is None. The keys must
            correspond to the keys in `inner_splitters` and should be set to None for any
            splitters that do not require splitter adjustment.
        n_iter : int, optional
            Number of iterations to run in random hyperparameter search. Default is None.
        n_jobs_outer : int, optional
            Number of jobs to run in parallel for the outer sequential loop. Default is -1.
            It is advised for n_jobs_inner * n_jobs_outer (replacing -1 with the number of
            available cores) to be less than or equal to the number of available cores on
            the machine.
        n_jobs_inner : int, optional
            Number of jobs to run in parallel for the inner loop. Default is 1.
            It is advised for n_jobs_inner * n_jobs_outer (replacing -1 with the number of
            available cores) to be less than or equal to the number of available cores on
            the machine.
        """
        # Checks
        # All others are checked in the run method
        if not isinstance(hedged_return_xcat, str):
            raise TypeError("hedged_return_xcat must be a string.")

        self.hedged_return_xcat = hedged_return_xcat

        # Set up outer splitter
        outer_splitter = ExpandingFrequencyPanelSplit(
            expansion_freq=est_freq,
            test_freq=est_freq,
            min_cids=min_cids,
            min_periods=min_periods,
        )

        # Run pipeline
        results = self.run(
            name=beta_xcat,
            outer_splitter=outer_splitter,
            inner_splitters=inner_splitters,
            models=models,
            hyperparameters=hyperparameters,
            scorers=scorers,
            search_type=search_type,
            normalize_fold_results=normalize_fold_results,
            cv_summary=cv_summary,
            split_functions=split_functions,
            n_iter=n_iter,
            n_jobs_outer=n_jobs_outer,
            n_jobs_inner=n_jobs_inner,
        )

        if hedged_return_xcat in self.hedged_returns["xcat"].unique():
            self.hedged_returns = self.hedged_returns[
                self.hedged_returns.xcat != hedged_return_xcat
            ]
        if beta_xcat in self.betas["xcat"].unique():
            self.betas = self.betas[self.betas.xcat != beta_xcat]
        if beta_xcat in self.chosen_models.name.unique():
            self.chosen_models = self.chosen_models[
                self.chosen_models.name != beta_xcat
            ]

        # Collect results from the worker
        beta_data = []
        hedged_return_data = []
        model_choice_data = []

        for split_result in results:
            beta_data.extend(split_result["betas"])
            hedged_return_data.extend(split_result["hedged_returns"])
            model_choice_data.append(split_result["model_choice"])

        stored_betas = pd.DataFrame(
            index=self.forecast_idxs, columns=[beta_xcat], data=np.nan, dtype="float32"
        )
        # Create quantamental dataframes of betas and hedged returns
        for real_date, cid, value in beta_data:
            stored_betas.loc[(cid, real_date), beta_xcat] = value

        stored_betas = stored_betas.groupby(level=0, observed=True).ffill().dropna()
        stored_betas.columns = stored_betas.columns.astype("category")
        stored_betas_long = pd.melt(
            frame=stored_betas.reset_index(),
            id_vars=["real_date", "cid"],
            var_name="xcat",
            value_name="value",
        )

        hedged_returns = (
            pd.DataFrame(hedged_return_data, columns=["real_date", "cid", "value"])
            .sort_values(["cid", "real_date"])
            .dropna()
        ).astype({"cid": "category"})
        hedged_returns = _insert_as_categorical(
            hedged_returns, "xcat", hedged_return_xcat, 2
        )

        self.betas = concat_categorical(self.betas, stored_betas_long)
        self.hedged_returns = concat_categorical(
            self.hedged_returns,
            hedged_returns,
        )

        # Store model selection data
        model_df_long = pd.DataFrame(
            columns=[col for col in self.chosen_models.columns if col != "name"],
            data=model_choice_data,
        ).astype({"model_type": "category"})
        model_df_long = _insert_as_categorical(model_df_long, "name", beta_xcat, 1)

        self.chosen_models = concat_categorical(self.chosen_models, model_df_long)

    def store_split_data(
        self,
        pipeline_name,
        optimal_model,
        optimal_model_name,
        optimal_model_score,
        optimal_model_params,
        inner_splitters_adj,
        X_train,
        y_train,
        X_test,
        y_test,
        timestamp,
        adjusted_test_index,
    ):
        """
        Stores characteristics of the optimal model at each retraining date.

        Parameters
        ----------
        pipeline_name : str
            Name of the signal optimization process.
        optimal_model : BaseRegressionSystem or VotingRegressor
            Optimal model selected at each retraining date.
        optimal_model_name : str
            Name of the optimal model.
        optimal_model_score : float
            Cross-validation score for the optimal model.
        optimal_model_params : dict
            Chosen hyperparameters for the optimal model.
        inner_splitters_adj : dict
            Dictionary of adjusted inner splitters.
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training response variable.
        X_test : pd.DataFrame
            Test feature matrix.
        y_test : pd.Series
            Test response variable.
        timestamp : pd.Timestamp
            Timestamp of the retraining date.
        adjusted_test_index : pd.MultiIndex
            Adjusted test index to account for lagged features.

        Returns
        -------
        dict
            Dictionary containing the betas and hedged returns determined at the
            given retraining date.
        """
        if isinstance(optimal_model, VotingRegressor):
            estimators = optimal_model.estimators_
            coefs_list = [est.coefs_ for est in estimators]
            sum_dict = defaultdict(lambda: [0, 0])

            for coefs in coefs_list:
                for key, value in coefs.items():
                    sum_dict[key][0] += value
                    sum_dict[key][1] += 1
            betas = {key: sum / count for key, (sum, count) in sum_dict.items()}
        elif isinstance(optimal_model, BaseRegressionSystem):
            betas = optimal_model.coefs_
        else:
            X_train.index.get_level_values(0).unique()
            betas = {cid: np.nan for cid in X_train.index.get_level_values(0).unique()}

        betas_list = [
            [
                X_train.index.get_level_values(1).max(),
                cid.split("v")[0],
                beta,
            ]
            for cid, beta in betas.items()
        ]

        # Now calculate the induced hedged returns

        betas_series = pd.Series(betas)
        XB = X_test.mul(betas_series, level=0, axis=0)
        hedged_returns = y_test.values.reshape(-1, 1) - XB.values.reshape(-1, 1)
        hedged_returns_data = [
            [idx[1], idx[0].split("v")[0]] + [hedged_returns[i].item()]
            for i, (idx, _) in enumerate(y_test.items())
        ]
        return {"betas": betas_list, "hedged_returns": hedged_returns_data}

    def evaluate_hedged_returns(
        self,
        hedged_return_xcat=None,
        cids=None,
        correlation_types="pearson",
        title=None,
        start=None,
        end=None,
        blacklist=None,
        freqs="M",
    ):
        """
        Method to determine and display a table of average absolute correlations between
        the benchmark return and the computed hedged returns within the class instance, over
        all cross-sections in the panel. Additionally, the correlation table displays the
        same results for the unhedged return specified in the class instance for comparison
        purposes.

        The returned dataframe will be multi-indexed by (benchmark return, return, frequency)
        and will contain each computed absolute correlation coefficient on each column.

        Parameters
        ----------
        hedged_return_xcat : str or list, optional
            Hedged returns to be evaluated. Default is None, which evaluates all hedged
            returns within the class instance.
        cids : str or list, optional
            Cross-sections for which evaluation of hedged returns takes place.
            Default is None, which evaluates all cross-sections within the class instance.
        correlation_types : str or list, optional
            Types of correlations to calculate.
            Options are "pearson", "spearman" and "kendall". If None, all three
            are calculated. Default is "pearson".
        title : str, optional
            Title for the correlation table. If None, the default
            title is "Average absolute correlations between each return and the chosen
            benchmark". Default is None.
        start : str, optional
            String in ISO format. Default is None.
        end : str, optional
            String in ISO format. Default is None.
        blacklist : dict, optional
            Dictionary of tuples of start and end dates to exclude from the evaluation.
            Default is None.
        freqs: str or list, optional
            Letters denoting all frequencies at which the correlations may be calculated.
            This must be a selection of "D", "W", "M", "Q" and "A". Default is "M".
            Each return series will always be summed over the sample period.

        Returns
        -------
        pd.DataFrame
            A dataframe of average absolute correlations between the benchmark return and the
            computed hedged returns.
        """
        # Checks
        correlation_types, hedged_return_xcat, cids, freqs = (
            self._checks_evaluate_hedged_returns(
                correlation_types=correlation_types,
                hedged_return_xcat=hedged_return_xcat,
                cids=cids,
                start=start,
                end=end,
                blacklist=blacklist,
                freqs=freqs,
            )
        )

        cids_v_benchmark = [f"{cid}v{self.benchmark_cid}" for cid in cids]
        # Construct a quantamental dataframe comprising specified hedged returns as well
        # as the unhedged returns and the benchmark return specified in the class instance
        hedged_df = self.hedged_returns[
            (self.hedged_returns["xcat"].isin(hedged_return_xcat))
            & (self.hedged_returns["cid"].isin(cids))
        ]
        unhedged_df = self.df[
            (self.df["xcat"].isin(self.xcats)) & (self.df["cid"].isin(cids_v_benchmark))
        ]
        benchmark_df = self.df[
            (self.df["xcat"] == self.benchmark_xcat)
            & (self.df["cid"] == f"{self.benchmark_cid}v{self.benchmark_cid}")
        ]

        cid_mapping = dict(zip(cids, cids_v_benchmark))
        hedged_df["cid"] = hedged_df["cid"].replace(cid_mapping)
        combined_df = concat_categorical(hedged_df, unhedged_df)

        # Create a pseudo-panel to match contract return cross-sections with a replicated
        # benchmark return. This is multi-indexed by (new cid, real_date). The columns
        # are the named hedged returns, with the final column being the benchmark category.
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])

        for cid in cids_v_benchmark:
            # Extract unhedged and hedged returns
            dfa = reduce_df(
                df=combined_df,
                xcats=hedged_return_xcat + self.xcats,
                cids=[cid],
            )
            # Extract benchmark returns
            dfb = reduce_df(
                df=benchmark_df,
                xcats=[self.benchmark_xcat],
                cids=[self.benchmark_cid],
            )
            # Combine and rename cross-section
            df_cid = concat_categorical(dfa, dfb)

            dfx = update_df(dfx, df_cid)

        # Create long format dataframes for each specified frequency
        Xy_long_freq = []
        for freq in freqs:
            Xy_long = categories_df(
                df=dfx,
                xcats=hedged_return_xcat + self.xcats,
                cids=[cid for cid in cids_v_benchmark],
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
        xcats_non_benchmark = [
            xcat for xcat in self.xcats if xcat != self.benchmark_xcat
        ]
        for xcat in hedged_return_xcat + xcats_non_benchmark:
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
            [[self.benchmark_return], hedged_return_xcat + xcats_non_benchmark, freqs],
            names=["benchmark return", "return category", "frequency"],
        )
        corr_df = pd.DataFrame(
            columns=[correlation for correlation in correlation_types],
            index=multiindex,
            data=df_rows,
        )

        return corr_df

    def _checks_evaluate_hedged_returns(
        self,
        correlation_types,
        hedged_return_xcat,
        cids,
        start,
        end,
        blacklist,
        freqs,
    ):
        """
        Input checks for the `evaluate_hedged_returns()` method.

        Parameters
        ----------
        correlation_types : str or list
            Types of correlations to calculate.
        hedged_return_xcat : str or list, optional
            Hedged returns to be evaluated.
        cids : str or list, optional
            Cross-sections for which evaluation of hedged returns takes place.
        start : str, optional
            Start date for evaluation.
        end : str, optional
            End date for evaluation.
        blacklist : dict, optional
            Dictionary of tuples of start and end dates to exclude from the evaluation.
        freqs: str or list, optional
            Letters denoting all frequencies at which the correlations may be calculated.
        """
        if isinstance(correlation_types, str):
            correlation_types = [correlation_types]
        elif not isinstance(correlation_types, list):
            raise TypeError("correlation_types must be a string or a list")
        if not all(
            isinstance(correlation_type, str) for correlation_type in correlation_types
        ):
            raise TypeError("All elements in correlation_types must be strings.")
        if not all(
            correlation_type in ["pearson", "spearman", "kendall"]
            for correlation_type in correlation_types
        ):
            raise ValueError(
                "All elements in correlation_types must be one of 'pearson', 'spearman' or 'kendall'."
            )

        if hedged_return_xcat is None:
            hedged_return_xcat = list(self.hedged_returns["xcat"].unique())
        else:
            if isinstance(hedged_return_xcat, str):
                hedged_return_xcat = [hedged_return_xcat]
            elif not isinstance(hedged_return_xcat, list):
                raise TypeError("hedged_return_xcat must be a string or a list")
            if not all(isinstance(xcat, str) for xcat in hedged_return_xcat):
                raise TypeError(
                    "All elements in hedged_return_xcat, when a list, must be strings."
                )
            if not (
                set(hedged_return_xcat).issubset(self.hedged_returns["xcat"].unique())
            ):
                raise ValueError(
                    "hedged_return_xcat must be a valid hedged return category within the class instance."
                )

        if cids is None:
            cids = self.hedged_returns["cid"].unique().tolist()
        else:
            if isinstance(cids, str):
                cids = [cids]
            elif not isinstance(cids, list):
                raise TypeError("cids must be a string or a list")
            if not all(isinstance(cid, str) for cid in cids):
                raise TypeError("All elements in cids must be strings.")
            if not all(cid in self.hedged_returns["cid"].unique() for cid in cids):
                raise ValueError(
                    "All cids must be valid cross-section identifiers within the class instance."
                )

        if start is not None and not isinstance(start, str):
            raise TypeError("start must be a string.")

        if end is not None and not isinstance(end, str):
            raise TypeError("end must be a string.")

        if blacklist is not None:
            if not isinstance(blacklist, dict):
                raise TypeError("The blacklist argument must be a dictionary.")
            if len(blacklist) == 0:
                raise ValueError("The blacklist argument must not be empty.")
            if not all([isinstance(key, str) for key in blacklist.keys()]):
                raise TypeError("The keys of the blacklist argument must be strings.")
            if not all(
                [isinstance(value, (list, tuple)) for value in blacklist.values()]
            ):
                raise TypeError("The values of the blacklist argument must be tuples.")
            if not all([len(value) == 2 for value in blacklist.values()]):
                raise ValueError(
                    "The values of the blacklist argument must be tuples of length two."
                )
            if not all(
                [
                    isinstance(date, pd.Timestamp)
                    for value in blacklist.values()
                    for date in value
                ]
            ):
                raise TypeError(
                    "The values of the blacklist argument must be tuples of pandas Timestamps."
                )

        # freqs checks
        if isinstance(freqs, str):
            freqs = [freqs]
        elif not isinstance(freqs, list):
            raise TypeError("freqs must be a string or a list of strings")
        if not all(isinstance(freq, str) for freq in freqs):
            raise TypeError("All elements in freqs must be strings.")
        if not all(freq in ["D", "W", "M", "Q"] for freq in freqs):
            raise ValueError(
                "All elements in freqs must be one of 'D', 'W', 'M' or 'Q'."
            )

        return correlation_types, hedged_return_xcat, cids, freqs

    def get_hedged_returns(
        self,
        hedged_return_xcat = None,
    ):
        """
        Returns a dataframe of out-of-sample hedged returns derived from beta estimation
        processes held within the class instance.

        Parameters
        ----------
        hedged_return_xcat : str or list, optional
            Category name or list of category names
            for the panel of derived hedged returns. If None, information from all
            beta estimation processes held within the class instance is returned.
            Default is None.

        Returns
        -------
        pd.DataFrame
            A dataframe of out-of-sample hedged returns derived from beta estimation
            processes.
        """
        # Checks
        hedged_return_xcat = self._checks_get_hedged_returns(
            hedged_return_xcat=hedged_return_xcat
        )

        if hedged_return_xcat is None:
            hedged_returns = self.hedged_returns
        else:
            hedged_returns = self.hedged_returns[
                self.hedged_returns.xcat.isin(hedged_return_xcat)
            ]

        return QuantamentalDataFrame(
            hedged_returns, _initialized_as_categorical=self.df.InitializedAsCategorical
        ).to_original_dtypes()

    def _checks_get_hedged_returns(
        self,
        hedged_return_xcat,
    ):
        """
        Input checks for the `get_hedged_returns()` method.

        Parameters
        ----------
        hedged_return_xcat : str or list
            Category name or list of category names for the panel of derived hedged
            returns.

        Returns
        -------
        str or list
            Category name or list of category names for the panel of derived hedged
            returns.
        """
        if hedged_return_xcat is not None:
            if isinstance(hedged_return_xcat, str):
                hedged_return_xcat = [hedged_return_xcat]
            elif not isinstance(hedged_return_xcat, list):
                raise TypeError("hedged_return_xcat must be a string or a list")
            if not all(isinstance(xcat, str) for xcat in hedged_return_xcat):
                raise TypeError(
                    "All elements in hedged_return_xcat, when a list, must be strings."
                )
            if not (
                set(hedged_return_xcat).issubset(self.hedged_returns["xcat"].unique())
            ):
                raise ValueError(
                    "hedged_return_xcat must be a valid hedged return category within the class instance."
                )
        return hedged_return_xcat

    def get_betas(
        self,
        beta_xcat = None,
    ):
        """
        Returns a dataframe of estimated betas derived from beta estimation processes
        held within the class instance.

        Parameters
        ----------
        beta_xcat : str or list
            Category name or list of category names for the panel of estimated contract
            betas. If None, information from all beta estimation processes held within
            the class instance is returned. Default is None.

        Returns
        -------
        pd.DataFrame
            A dataframe of estimated betas derived from beta estimation processes.
        """
        # Checks
        beta_xcat = self._checks_get_betas(beta_xcat=beta_xcat)

        if beta_xcat is None:
            betas = self.betas
        else:
            betas = self.betas[self.betas.xcat.isin(beta_xcat)]

        return QuantamentalDataFrame(
            betas, _initialized_as_categorical=self.df.InitializedAsCategorical
        ).to_original_dtypes()

    def _checks_get_betas(
        self,
        beta_xcat,
    ):
        """
        Input checks for the `get_betas()` method.

        Parameters
        ----------
        beta_xcat : str or list
            Category name or list of category names for the panel of estimated contract
            betas.

        Returns
        -------
        str or list
            Category name or list of category names for the panel of estimated contract
            betas.
        """
        if beta_xcat is not None:
            if isinstance(beta_xcat, str):
                beta_xcat = [beta_xcat]
            if not isinstance(beta_xcat, list):
                raise TypeError("beta_xcat must be a string or a list")
            if not all(isinstance(xcat, str) for xcat in beta_xcat):
                raise TypeError(
                    "All elements in beta_xcat, when a list, must be strings."
                )
            if not (set(beta_xcat).issubset(self.betas["xcat"].unique())):
                raise ValueError(
                    "beta_xcat must be a valid beta category within the class instance."
                )
        return beta_xcat

    def _get_mean_abs_corrs(
        self,
        xcat,
        cids,
        df,
        correlation,
    ):
        """
        Calculate mean absolute correlation between a column 'xcat' in a dataframe 'df'
        and the benchmark return (the last column) across all cross-sections in 'cids'.
        The correlation is calculated using the method specified in 'correlation'.

        Parameters
        ----------
        xcat : str
            Category name for the column in the dataframe.
        cids : str
            Cross-sections for which the correlation is calculated.
        df : pd.DataFrame
            Dataframe containing the relevant columns.
        correlation : str
            Type of correlation to calculate.

        Returns
        -------
        float
            Mean absolute correlation between the column 'xcat' and the benchmark return.
        """
        # Get relevant columns
        df_subset = df[[xcat, self.benchmark_xcat]].dropna()

        # Create inner function to calculate the correlation for a given cross-section
        # This is done so that one can groupby cross-section and apply this function directly

        def calculate_correlation(group):
            return abs(group[xcat].corr(group[self.benchmark_xcat], method=correlation))

        # Calculate the mean absolute correlation over all cross sections
        mean_abs_corr = (
            df_subset.groupby("cid", observed=True).apply(calculate_correlation).mean()
        )

        return mean_abs_corr

    def _check_duplicate_results(self, hedged_return_xcat, beta_xcat):
        """
        Check for duplicate results in the class instance and remove them.

        Parameters
        ----------
        hedged_return_xcat : str
            Category name for the panel of out-of-sample hedged returns.
        beta_xcat : str
            Category name for the panel of estimated betas.
        """
        conditions = [
            ("hedged_returns", "xcat", hedged_return_xcat),
            ("betas", "xcat", beta_xcat),
            ("chosen_models", "name", beta_xcat),
        ]

        self._remove_results(conditions)


if __name__ == "__main__":
    from macrosynergy.learning import (
        ExpandingKFoldPanelSplit,
        LinearRegressionSystem,
        neg_mean_abs_corr,
    )
    from macrosynergy.management.simulate import make_qdf

    # Simulate a panel dataset of benchmark and contract returns
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["BENCH_XR", "CONTRACT_XR"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2015-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2015-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2015-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2015-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["BENCH_XR"] = ["2015-01-01", "2019-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CONTRACT_XR"] = ["2015-01-01", "2019-12-31", 0.1, 1, 0, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Initialize the BetaEstimator object
    # Use for the benchmark return: USD_BENCH_XR.
    be = BetaEstimator(
        df=dfd,
        xcats="CONTRACT_XR",
        benchmark_return="USD_BENCH_XR",
        cids=["AUD", "USD"],
    )

    models = {
        "LR": LinearRegressionSystem(min_xs_samples=21 * 1),
    }
    hparam_grid = {"LR": {"fit_intercept": [True, False], "positive": [True, False]}}

    scorer = {"scorer": neg_mean_abs_corr}

    be.estimate_beta(
        beta_xcat="BETA_NSA",
        hedged_return_xcat="HEDGED_RETURN_NSA",
        models=models,
        hyperparameters=hparam_grid,
        scorers=scorer,
        inner_splitters={"expandingkfold": ExpandingKFoldPanelSplit(n_splits=5)},
        search_type="grid",
        cv_summary="median",
        min_cids=1,
        min_periods=21 * 12,
        est_freq="Q",
        n_jobs_outer=1,
        n_jobs_inner=1,
    )

    evaluation_df = be.evaluate_hedged_returns(
        correlation_types=["pearson", "spearman", "kendall"],
        freqs=["W", "M", "Q"],
    )

    be.models_heatmap(name="BETA_NSA")
    # print(evaluation_df)
