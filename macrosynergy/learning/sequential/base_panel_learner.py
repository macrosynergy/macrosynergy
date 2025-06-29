"""
Sequential learning over a panel.
"""

import numbers
import warnings
from abc import ABC
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from macrosynergy.compat import JOBLIB_RETURN_AS
from macrosynergy.learning.splitters import (
    WalkForwardPanelSplit,
)
from macrosynergy.management import categories_df
from macrosynergy.management.types.qdf import QuantamentalDataFrame


class BasePanelLearner(ABC):
    def __init__(
        self,
        df,
        xcats,
        cids=None,
        start=None,
        end=None,
        blacklist=None,
        freq="M",
        lag=1,
        xcat_aggs=["last", "sum"],
        generate_labels=None,
        skip_checks=False,
        drop_nas=True
    ):
        """
        Initialize a sequential learning process over a panel.

        Parameters
        ----------
        df : pd.DataFrame
            Standardized daily quantamental dataframe with the four columns: "cid",
            "xcat", "real_date" and "value".
        xcats : list
            List of xcats to be used in the learning process. The last category in the
            list is the dependent variable, and all preceding categories are the
            independent variables in a supervised learning framework.
        cids : list, 
            Cross-sections to be included. Default is all in the dataframe.
        start : str, 
            Start date for considered data in subsequent analysis in ISO 8601 format.
            Default is None i.e. the earliest date in the dataframe.
        end : str, 
            End date for considered data in subsequent analysis in ISO 8601 format.
            Default is None i.e. the latest date in the dataframe.
        blacklist : list, 
            Blacklisting dictionary specifying date ranges for which cross-sectional
            information should be excluded. The keys are cross-sections and the values
            are tuples of start and end dates in ISO 8601 format. Default is None.
        freq : str, 
            Frequency of the data. Default is "M" for monthly.
        lag : int, 
            Number of periods to lag the independent variables. Default is 1.
        xcat_aggs : list, 
            List of exactly two aggregation methods for downsampling data to the frequency
            specified in the freq parameter. The first parameter pertains to all
            independent variable downsampling, whilst the second corresponds with the
            target category. Default is ["last", "sum"].
        generate_labels : callable, 
            Function to transform the dependent variable, usually into
            classification labels. Default is None.
        skip_checks : bool, 
            Whether to skip the initialization checks. Default is False.
        drop_nas : bool, 
            Whether to drop rows with NaN values in the dataframe. Default is True.
            If False, only the rows with NaN values in the dependent variable are dropped.

        Notes
        -----
        `BasePanelLearner` is an abstract class that provides the basic structure to
        train statistical machine learning models sequentially over a panel. At each
        model retraining date, an optimal model is chosen out a collection of candidate
        models through a hyperparameter optimization process. Following the model
        selection stage, the model is trained on date preceding the retraining date,
        maintaining the point-in-time principle of JPMaQS. Analytics from the selected
        model are stored for interpretability of the process.

        The hyperparameter optimization process revolves around cross-validation [1]_.
        Cross-validation estimates out-of-sample performance metrics. Each candidate
        model is cross-validated over a collection of possible hyperparameters and the
        model with the best score is chosen. When all possible models are directly
        specified and cross-validated, we call this process a 'grid search'. When models
        are sampled from a given collection, we call this process a 'random search'.
        These are the two methods currently supported by `BasePanelLearner`. In a future
        release, we will add support for Bayesian hyperparameter searches [2]_.

        References
        ----------
        .. [1] Brownlee, J. (2023). A Gentle Introduction to k-fold Cross-Validation.
            https://machinelearningmastery.com/k-fold-cross-validation/

        .. [2] Frazier, P.I., 2018. A tutorial on Bayesian optimization.
            https://arxiv.org/abs/1807.02811
        """
        # Checks
        if not skip_checks:
            self._check_init(
                df,
                xcats,
                cids,
                start,
                end,
                blacklist,
                freq,
                lag,
                xcat_aggs,
                generate_labels,
                drop_nas,
            )
        # Attributes
        self.df = QuantamentalDataFrame(df)
        self.xcats = xcats
        self.cids = cids
        self.start = start
        self.end = end
        self.blacklist = blacklist
        self.freq = freq
        self.lag = lag
        self.xcat_aggs = xcat_aggs
        self.generate_labels = generate_labels
        self.drop_nas = drop_nas

        # Create long-format dataframe
        df_long = (
            categories_df(
                df=self.df,
                xcats=self.xcats,
                cids=self.cids,
                start=self.start,
                end=self.end,
                blacklist=self.blacklist,
                freq=self.freq,
                lag=self.lag,
                xcat_aggs=self.xcat_aggs,
            )
        )

        if self.drop_nas:
            df_long = df_long.dropna().sort_index()
        else:
            # Only drop rows with NaN values in the dependent variable
            df_long = df_long.dropna(subset=[self.xcats[-1]]).sort_index()
            
        # Create X and y
        self.X = df_long.iloc[:, :-1]
        self.y = df_long.iloc[:, -1]

        if self.generate_labels is not None:
            self.y = self.y.apply(self.generate_labels)

        # Store necessary index information
        self.index = self.X.index
        self.date_levels = self.index.get_level_values(1)
        self.xs_levels = self.index.get_level_values(0)
        self.unique_date_levels = sorted(self.date_levels.unique())
        self.unique_xs_levels = sorted(self.xs_levels.unique())

        # Create initial dataframe to store model selection data from the learning process
        self.chosen_models = pd.DataFrame(
            columns=[
                "real_date",
                "name",
                "model_type",
                "score",
                "hparams",
                "n_splits_used",
            ]
        ).astype(
            {
                "real_date": "datetime64[ns]",
                "name": "category",
                "model_type": "category",
                "score": "float32",
                "hparams": "object",
                "n_splits_used": "object",
            }
        )

        self.timestamps = []

    def run(
        self,
        name,
        models,
        outer_splitter,
        inner_splitters=None,
        hyperparameters=None,
        scorers=None,
        search_type="grid",
        normalize_fold_results=False,
        cv_summary="mean",
        include_train_folds=False,
        n_iter=100,
        split_functions=None,
        n_jobs_outer=-1,
        n_jobs_inner=1,
    ):
        """
        Run a learning process over a panel.

        Parameters
        ----------
        name : str
            Category name for the forecasted panel resulting from the learning process.
        models : dict
            Dictionary of model names and compatible `scikit-learn` model objects.
        outer_splitter : WalkForwardPanelSplit
            Outer splitter for the learning process.
        inner_splitters : dict, optional
            Inner splitters for the learning process.
        hyperparameters : dict, optional
            Dictionary of model names and hyperparameter grids.
        scorers : dict, optional
            Dictionary of `scikit-learn` compatible scoring functions.
        search_type : str
            Search type for hyperparameter optimization. Default is "grid".
            Options are "grid", "prior" and "bayes". If no hyperparameter tuning
            is required, this parameter can be disregarded.
        normalize_fold_results : bool
            Whether to normalize the scores across folds before combining them. Default is
            False. If no hyperparameter tuning is required, this parameter can be disregarded.
        cv_summary : str or callable
            Summary function to use to combine scores across cross-validation folds.
            Default is "mean". Options are "mean", "median", "mean-std", "mean/std",
            "mean-std-ge" or a callable function. If no hyperparameter tuning
            is required, this parameter can be disregarded.
        include_train_folds : bool, optional
            Whether to calculate cross-validation statistics on the training folds in 
            additional to the test folds. If no hyperparameter tuning
            is required, this parameter can be disregarded.
        n_iter : int
            Number of iterations for random or bayesian hyperparameter optimization.
            If no hyperparameter tuning is required, this parameter can be disregarded.
        split_functions : dict, optional
            Dictionary of callables for determining the number of cross-validation
            splits to add to the initial number, as a function of the number of iterations
            passed in the sequential learning process. If no hyperparameter tuning
            is required, this parameter can be disregarded.
        n_jobs_outer : int, optional
            Number of jobs to run in parallel for the outer loop. Default is -1. 
        n_jobs_inner : int, optional
            Number of jobs to run in parallel for the inner loop. Default is 1. If no
            hyperparameter tuning is required, this parameter can be disregarded.

        Returns
        -------
        list
            List of dictionaries containing the results of the learning process.
        """
        # Checks
        self._check_run(
            name=name,
            outer_splitter=outer_splitter,
            inner_splitters=inner_splitters,
            models=models,
            hyperparameters=hyperparameters,
            scorers=scorers,
            search_type=search_type,
            normalize_fold_results=normalize_fold_results,
            cv_summary=cv_summary,
            include_train_folds=include_train_folds,
            n_iter=n_iter,
            split_functions=split_functions,
            n_jobs_outer=n_jobs_outer,
            n_jobs_inner=n_jobs_inner,
        )

        # Determine all outer splits and run the learning process in parallel
        train_test_splits = list(outer_splitter.split(self.X, self.y))

        if inner_splitters is not None:
            base_splits = self._get_base_splits(inner_splitters)
        else:
            # No CV is performed, so no base splits are needed
            base_splits = None

        # Return list of results
        optim_results = tqdm(
            Parallel(n_jobs=n_jobs_outer, **JOBLIB_RETURN_AS)(
                delayed(self._worker)(
                    name=name,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    inner_splitters=inner_splitters,
                    models=models,
                    hyperparameters=hyperparameters,
                    scorers=scorers,
                    cv_summary=cv_summary,
                    include_train_folds=include_train_folds,
                    search_type=search_type,
                    normalize_fold_results=normalize_fold_results,
                    n_iter=n_iter,
                    n_splits_add=self._get_n_splits_add(
                        iteration, outer_splitter, split_functions
                    ),
                    n_jobs_inner=n_jobs_inner,
                    base_splits=base_splits,
                )
                for iteration, (train_idx, test_idx) in enumerate(train_test_splits)
            ),
            total=len(train_test_splits),
        )

        return optim_results

    def _worker(
        self,
        name,
        train_idx,
        test_idx,
        inner_splitters,
        models,
        hyperparameters,
        scorers,
        cv_summary,
        include_train_folds,
        search_type,
        normalize_fold_results,
        n_iter,
        n_splits_add,
        n_jobs_inner,
        base_splits,
        timestamp=None,
    ):
        """
        Worker function for parallel processing of the learning process.

        Parameters
        ----------
        name : str
            Category name for the forecasted panel resulting from the learning process.
        train_idx : np.ndarray
            Training indices for the current outer split.
        test_idx : np.ndarray
            Test indices for the current outer split.
        inner_splitters : dict, optional
            Inner splitters for the learning process.
        models : dict
            Compatible `scikit-learn` model objects.
        hyperparameters : dict, optional
            Hyperparameter grids.
        scorers : dict, optional
            Compatible `scikit-learn` scoring functions.
        cv_summary : str or callable, optional
            Summary function to condense cross-validation scores in each fold to a single
            value, against which different hyperparameter choices can be compared.
            If no hyperparameter tuning is required, this parameter can be disregarded.
        include_train_folds : bool, optional
            Whether to calculate cross-validation statistics on the training folds in 
            additional to the test folds.
            If no hyperparameter tuning is required, this parameter can be disregarded.
        search_type : str, optional
            Search type for hyperparameter optimization. Default is "grid".
            Options are "grid", "prior" and "bayes".
            If no hyperparameter tuning is required, this parameter can be disregarded.
        normalize_fold_results : bool, optional
            Whether to normalize the scores across folds before combining them.
            If no hyperparameter tuning is required, this parameter can be disregarded.
        n_iter : int, optional
            Number of iterations for random or bayesian hyperparameter optimization.
            If no hyperparameter tuning is required, this parameter can be disregarded.
        n_splits_add : list, optional
            List of integers to add to the number of splits for each inner splitter.
            Default is None.
        n_jobs_inner : int, optional
            Number of jobs to run in parallel for the inner loop. Default is 1.
            If no hyperparameter tuning is required, this parameter can be disregarded.
        base_splits : dict, optional
            Dictionary of initial number of splits for each inner splitter.
            If no hyperparameter tuning is required, this parameter is None.
        timestamp : pd.Timestamp, optional
            Date to record predictions and model diagnostics. Default is None. If None,
            the earliest date in the test set is used (which is then adjusted for lag).

        Returns
        -------
        dict
            Dictionary comprising model selection data and predictive analytics.
        """
        # Train-test split
        X_train, X_test = self.X.iloc[train_idx, :], self.X.iloc[test_idx, :]
        y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

        # Determine correct timestamps of test set forecasts
        # First get test index information
        test_index = self.index[test_idx]
        test_xs_levels = self.xs_levels[test_idx]
        test_date_levels = self.date_levels[test_idx]
        sorted_test_date_levels = sorted(test_date_levels.unique())

        # Since the features lag behind the targets, the dates need to be adjusted
        # by the lag applied.
        if self.lag != 0:
            locs: np.ndarray = (
                np.searchsorted(
                    self.unique_date_levels, sorted_test_date_levels, side="left"
                )
                - 1
            )
            adj_test_date_levels: pd.DatetimeIndex = pd.DatetimeIndex(
                [self.unique_date_levels[i] if i >= 0 else pd.NaT for i in locs]
            )
            # Now formulate correct index
            date_map = dict(zip(sorted_test_date_levels, adj_test_date_levels))
            mapped_dates = test_date_levels.map(date_map)
            test_index = pd.MultiIndex.from_arrays(
                [test_xs_levels, mapped_dates], names=["cid", "real_date"]
            )
        else:
            adj_test_date_levels = test_date_levels

        if inner_splitters is not None:
            if n_splits_add is not None:
                inner_splitters_adj = inner_splitters.copy()
                for splitter_name, _ in inner_splitters_adj.items():
                    if hasattr(inner_splitters_adj[splitter_name], "n_splits"):
                        inner_splitters_adj[splitter_name].n_splits = (
                            base_splits[splitter_name] + n_splits_add[splitter_name]
                        )

            else:
                inner_splitters_adj = inner_splitters

            optim_name, optim_model, optim_score, optim_params = self._model_search(
                X_train=X_train,
                y_train=y_train,
                inner_splitters=inner_splitters_adj,
                models=models,
                hyperparameters=hyperparameters,
                scorers=scorers,
                search_type=search_type,
                normalize_fold_results=normalize_fold_results,
                n_iter=n_iter,
                cv_summary=cv_summary,
                include_train_folds=include_train_folds,
                n_jobs_inner=n_jobs_inner,
            )
        else:
            optim_name = list(models.keys())[0] 
            optim_model = models[optim_name].fit(X_train, y_train)
            optim_score = np.ubyte(0) # For memory efficiency
            optim_params = {}
            inner_splitters_adj = None

        split_results = self._get_split_results(
            pipeline_name=name,
            optimal_model=optim_model,
            optimal_model_name=optim_name,
            optimal_model_score=optim_score,
            optimal_model_params=optim_params,
            inner_splitters_adj=inner_splitters_adj,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            timestamp=adj_test_date_levels.min() if timestamp is None else timestamp,
            adjusted_test_index=test_index,
        )

        return split_results

    def _model_search(
        self,
        X_train,
        y_train,
        inner_splitters,
        models,
        hyperparameters,
        scorers,
        search_type,
        normalize_fold_results,
        n_iter,
        cv_summary,
        include_train_folds,
        n_jobs_inner,
    ):
        """
        Determine optimal model based on cross-validation from a given training set.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Training target.
        inner_splitters : dict
            Inner splitters for the learning process.
        models : dict
            Compatible `scikit-learn` model objects.
        hyperparameters : dict
            Hyperparameter grids.
        scorers : dict
            Compatible `scikit-learn` scoring functions.
        search_type : str
            Search type for hyperparameter optimization. Default is "grid".
        normalize_fold_results : bool
            Whether to normalize the scores across folds before combining them.
        n_iter : int
            Number of iterations for random or bayesian hyperparameter optimization.
        cv_summary : str or callable
            Summary function to condense cross-validation scores in each fold to a single
            value, against which different hyperparameter choices can be compared.
        include_train_folds : bool, optional
            Whether to calculate cross-validation statistics on the training folds in 
            the cross-validation process.
        n_jobs_inner : int
            Number of jobs to run in parallel for the inner loop.

        Returns
        -------
        tuple
            Optimal model name, optimal model, optimal model score and optimal
            model hyperparameters.
        """
        optim_name = None
        optim_model = None
        optim_score = np.float32("-inf")
        optim_params = {}

        cv_splits = []

        for splitter in inner_splitters.values():
            cv_splits.extend(list(splitter.split(X=X_train, y=y_train)))
        # TODO: instead of picking one model, the best hyperparameters could be selected
        # for each model and then "final" prediction would be the average of the individual
        # predictions. This would be a simple ensemble method.
        for model_name, model in models.items():
            # For each model, find the optimal hyperparameters
            if search_type == "grid":
                search_object = GridSearchCV(
                    estimator=model,
                    param_grid=hyperparameters[model_name],
                    scoring=scorers,
                    n_jobs=n_jobs_inner,
                    refit=partial(
                        self._model_selection,
                        cv_summary=cv_summary,
                        scorers=scorers,
                        normalize_fold_results=normalize_fold_results,
                    ),
                    cv=cv_splits,
                    return_train_score = include_train_folds,
                )
            elif search_type == "prior":
                search_object = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=hyperparameters[model_name],
                    n_iter=n_iter,
                    scoring=scorers,
                    n_jobs=n_jobs_inner,
                    refit=partial(
                        self._model_selection,
                        cv_summary=cv_summary,
                        scorers=scorers,
                        normalize_fold_results=normalize_fold_results,
                    ),
                    cv=cv_splits,
                    return_train_score = include_train_folds,
                )

            try:
                search_object.fit(X_train, y_train)
            except Exception as e:
                warnings.warn(
                    f"Error running a hyperparameter search for {model_name}: {e}",
                    RuntimeWarning,
                )
                continue

            score = self._model_selection(
                search_object.cv_results_,
                cv_summary,
                scorers,
                normalize_fold_results,
                return_index=False,
            )
            if score > optim_score:
                optim_name = model_name
                optim_model = search_object.best_estimator_
                optim_score = score
                optim_params = search_object.best_params_

        return optim_name, optim_model, optim_score, optim_params

    def _model_selection(
        self, cv_results, cv_summary, scorers, normalize_fold_results, return_index=True
    ):
        """
        Select the optimal hyperparameters based on a `scikit-learn` cv_results dataframe.

        Parameters
        ----------
        cv_results : dict
            Cross-validation results dictionary.
        cv_summary : str or callable
            Summary function to condense cross-validation scores in each fold to a single
            value, against which different hyperparameter choices can be compared.
        scorers : dict
            Compatible `scikit-learn` scoring functions.
        normalize_fold_results : bool
            Whether to normalize the scores across folds before combining them.
        return_index : bool
            Whether to return the index of the best estimator or the maximal score itself.
            Default is True.

        Returns
        -------
        int or float
            Either the index of the best estimator or the maximal score itself.
        """
        cv_results = pd.DataFrame(cv_results)
        metric_columns = [
            col
            for col in cv_results.columns
            if col.startswith("split") and ("test" in col or "train" in col)
        ]
        if normalize_fold_results:
            cv_results[metric_columns] = StandardScaler().fit_transform(
                cv_results[metric_columns]
            )

        # For each metric, summarise the scores across folds for each hyperparameter choice
        # using cv_summary
        for scorer in scorers.keys():
            # Extract the test scores for each fold for that scorer
            scorer_columns = [col for col in metric_columns if scorer in col]
            if cv_summary == "mean":
                cv_results[f"{scorer}_summary"] = cv_results[scorer_columns].mean(
                    axis=1,
                )
            elif cv_summary == "median":
                cv_results[f"{scorer}_summary"] = cv_results[scorer_columns].median(
                    axis=1
                )
            elif cv_summary == "mean-std":
                cv_results[f"{scorer}_summary"] = cv_results[scorer_columns].mean(
                    axis=1
                ) - cv_results[scorer_columns].std(axis=1)
            elif cv_summary == "mean/std":
                cv_results[f"{scorer}_summary"] = cv_results[scorer_columns].mean(
                    axis=1
                ) / cv_results[scorer_columns].std(axis=1)
            elif cv_summary == "mean-std-ge":
                # Separate training and columns
                train_columns = [col for col in scorer_columns if "train" in col]
                test_columns = [col for col in scorer_columns if "test" in col]

                # obtain mean and std of test metrics
                mean_test = np.nanmean(cv_results[test_columns], axis=1)
                std_test = np.nanstd(cv_results[test_columns], axis=1)

                # Determine generalization gap.
                # We define this to be the average absolute deviation of the test metrics
                # from their corresponding training metrics.
                generalization_gap = np.nanmean(
                    np.abs(
                        cv_results[train_columns].values - cv_results[test_columns].values
                    ),
                    axis=1
                )

                # Store mean test metric - std test metric - generalization gap
                cv_results[f"{scorer}_summary"] = mean_test - std_test - generalization_gap
            
            else:
                # TODO: handle NAs?
                cv_results[f"{scorer}_summary"] = cv_results[scorer_columns].apply(
                    cv_summary, axis=1
                )

        # Now scale the summary scores for each scorer
        scaler = StandardScaler()

        summary_cols = [f"{scorer}_summary" for scorer in scorers.keys()]

        if len(scorers) > 1:
            scaler = StandardScaler()
            cv_results[summary_cols] = scaler.fit_transform(cv_results[summary_cols])

        # Now average the summary scores for each scorer
        cv_results["final_score"] = cv_results[summary_cols].mean(axis=1)

        # Return index of best estimator
        # TODO: handle case where multiple hyperparameter choices have the same score
        # We currently return the first one
        if return_index:
            return cv_results["final_score"].idxmax()
        else:
            return cv_results["final_score"].max()

    def _get_split_results(
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
        Store model selection information and predictive analytics for training set
        (X_train, y_train).

        Parameters
        ----------
        pipeline_name : str
            Name of the sequential optimization pipeline.
        optimal_model : RegressorMixin or ClassifierMixin or Pipeline
            Optimal model selected for the training set.
        optimal_model_name : str
            Name of the optimal model.
        optimal_model_score : float
            Score of the optimal model.
        optimal_model_params : dict
            Hyperparameters of the optimal model.
        inner_splitters_adj : dict
            Inner splitters for the learning process.
        X_train : pd.DataFrame
            Input feature matrix.
        y_train : pd.Series
            Target variable.
        X_test : pd.DataFrame
            Input feature matrix.
        y_test : pd.Series
            Target variable.
        timestamp : pd.Timestamp
            Model retraining date.
        adjusted_test_index : pd.MultiIndex
            Adjusted test index to account for lagged features.

        Returns
        -------
        dict
            Dictionary containing model selection data and predictive analytics.
        """
        split_result = dict()
        model_result = self._store_model_choice_data(
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
        )

        split_result.update(model_result)

        split_data = self.store_split_data(
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
        )

        split_result.update(split_data)
        return split_result

    def _store_model_choice_data(
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
    ):
        """
        Store model selection information for training set (X_train, y_train).

        Parameters
        ----------
        pipeline_name : str
            Name of the sequential optimization pipeline.
        optimal_model : RegressorMixin or ClassifierMixin or Pipeline
            Optimal model selected for the training set.
        optimal_model_name : str
            Name of the optimal model.
        optimal_model_score : float
            Score of the optimal model.
        optimal_model_params : dict
            Hyperparameters of the optimal model.
        inner_splitters_adj : dict
            Inner splitters for the learning process.
        X_train : pd.DataFrame
            Input feature matrix.
        y_train : pd.Series
            Target variable.
        X_test : pd.DataFrame
            Input feature matrix.
        y_test : pd.Series
            Target variable.
        timestamp : pd.Timestamp
            Model retraining date.

        Returns
        -------
        dict
            Dictionary containing model selection data.
        """
        if optimal_model is None:
            warnings.warn(
                f"No model was selected at time {timestamp}",
                RuntimeWarning,
            )
            optimal_model_name = None
            optimal_model_score = np.float32("-inf")
            optimal_model_params = {}

        data = [
            timestamp,
            optimal_model_name,
            optimal_model_score,
            optimal_model_params,
        ]

        if inner_splitters_adj is not None:
            n_splits = {
                splitter_name: splitter.n_splits
                for splitter_name, splitter in inner_splitters_adj.items()
            }
            data.append(n_splits)
        else:
            data.append(0) # No splits were used because CV wasn't used

        return {"model_choice": data}

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
        Store predictive analytics for training set (X_train, y_train).

        Parameters
        ----------
        pipeline_name : str
            Name of the sequential optimization pipeline.
        optimal_model : RegressorMixin or ClassifierMixin or Pipeline
            Optimal model selected for the training set.
        optimal_model_name : str
            Name of the optimal model.
        optimal_model_score : float
            Score of the optimal model.
        optimal_model_params : dict
            Hyperparameters of the optimal model.
        inner_splitters_adj : dict
            Inner splitters for the learning process.
        X_train : pd.DataFrame
            Input feature matrix.
        y_train : pd.Series
            Target variable.
        X_test : pd.DataFrame
            Input feature matrix.
        y_test : pd.Series
            Target variable.
        timestamp : pd.Timestamp
            Model retraining date.
        adjusted_test_index : pd.MultiIndex
            Adjusted test index to account for lagged features.

        Returns
        -------
        dict
            Dictionary containing predictive analytics.
        """
        return dict()

    def get_optimal_models(self, name=None):
        """
        Returns the sequences of optimal models for one or more processes.

        Parameters
        ----------
        name : str or list, optional
            Label of sequential optimization process. Default is all stored in the class
            instance.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of the optimal models and hyperparameters selected at each
            retraining date.
        """
        if name is None:
            return self.chosen_models
        else:
            if isinstance(name, str):
                name = [name]
            elif not isinstance(name, list):
                raise TypeError(
                    "The process name must be a string or a list of strings."
                )

            for n in name:
                if n not in self.chosen_models.name.unique():
                    raise ValueError(
                        f"""The process name '{n}' is not in the list of already-run
                        pipelines. Please check the name carefully. If correct, please run 
                        calculate_predictions() first.
                        """
                    )
            return self.chosen_models[self.chosen_models.name.isin(name)]

    def models_heatmap(
        self,
        name,
        title=None,
        cap=5,
        figsize=(12, 8),
        title_fontsize=None,
        tick_fontsize=None,
    ):
        """
        Visualized optimal models used for signal calculation.

        Parameters
        ----------
        name : str
            Name of the sequential optimization pipeline.
        title : str, optional
            Title of the heatmap. Default is None. This creates a figure title of the form
            "Model Selection Heatmap for {name}".
        cap : int, optional
            Maximum number of models to display. Default (and limit) is 5. The chosen
            models are the 'cap' most frequently occurring in the pipeline.
        figsize : tuple, optional
            Tuple of floats or ints denoting the figure size. Default is (12, 8).
        title_fontsize : int, optional
            Font size for the title. Default is None.
        tick_fontsize : int, optional
            Font size for the ticks. Default is None.

        Notes
        -----
        This method displays the models selected at each date in time over the span
        of the sequential learning process. A binary heatmap is used to visualise
        the model selection process.
        """
        # Checks
        self._checks_models_heatmap(
            name=name,
            title=title,
            cap=cap,
            figsize=figsize,
            title_fontsize=title_fontsize,
            tick_fontsize=tick_fontsize,
        )

        # Get the chosen models for the specified pipeline to visualise selection.
        chosen_models = self.get_optimal_models(name=name).sort_values(by="real_date")
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
        plt.title(title, fontsize=title_fontsize)

        plt.xticks(fontsize=tick_fontsize)  # X-axis tick font size
        plt.yticks(fontsize=tick_fontsize)
        plt.show()

    def _check_init(
        self,
        df,
        xcats,
        cids,
        start,
        end,
        blacklist,
        freq,
        lag,
        xcat_aggs,
        generate_labels,
        drop_nas,
    ):
        """
        Checks for the constructor.

        Parameters
        ----------
        df : pd.DataFrame
            Long-format dataframe.
        xcats : list
            List of xcats to be used in the learning process.
        cids : list, optional
            List of cids to be used in the learning process. Default is None.
        start : str, optional
            Start date for the learning process. Default is None.
        end : str, optional
            End date for the learning process. Default is None.
        blacklist : dict, optional
            Dictionary of dates to exclude from the learning process. Default is None.
        freq : str
            Frequency of the data. Options are "D", "W", "M", "Q" and "Y".
        lag : int
            Lag to apply to the features. Default is 0.
        xcat_aggs : list
            List of aggregation functions to apply to the independent and
            dependent variables respectively.
        generate_labels : callable, optional
            Function to generate labels for a supervised learning process.
            Default is None.
        drop_nas : bool, optional
            Whether to drop rows with NAs in the dataframe. Default is True.
        """
        # Dataframe checks
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not set(["cid", "xcat", "real_date", "value"]).issubset(df.columns):
            raise ValueError(
                "df must have columns 'cid', 'xcat', 'real_date' and 'value'."
            )
        if len(df) < 1:
            raise ValueError("df must not be empty")

        # categories checks
        if not isinstance(xcats, list):
            raise TypeError("xcats must be a list.")
        if len(xcats) < 2:
            raise ValueError("xcats must have at least two elements.")
        if not all(isinstance(xcat, str) for xcat in xcats):
            raise TypeError("All elements in xcats must be strings.")
        difference_xcats = set(xcats) - set(df["xcat"].unique())
        if difference_xcats != set():
            raise ValueError(f"{str(difference_xcats)} not in the dataframe.")

        # cids checks
        if cids is not None:
            if not isinstance(cids, list):
                raise TypeError("cids must be a list.")
            if not all(isinstance(cid, str) for cid in cids):
                raise TypeError("All elements in cids must be strings.")
            difference_cids = set(cids) - set(df["cid"].unique())
            if difference_cids != set():
                raise ValueError(f"{str(difference_cids)} not in the dataframe.")

        # start checks
        if start is not None:
            if not isinstance(start, str):
                raise TypeError("'start' must be a string.")
            try:
                pd.to_datetime(start)
            except ValueError:
                raise ValueError("'start' must be in ISO 8601 format.")
            if pd.to_datetime(start) > pd.to_datetime(df["real_date"]).max():
                raise ValueError("'start' must be before the last date in the panel.")

        # end checks
        if end is not None:
            if not isinstance(end, str):
                raise TypeError("'end' must be a string.")
            try:
                pd.to_datetime(end)
            except ValueError:
                raise ValueError("'end' must be in ISO 8601 format.")
            if pd.to_datetime(end) < pd.to_datetime(df["real_date"]).min():
                raise ValueError("'end' must be after the first date in the panel.")

        if start is not None and end is not None:
            if pd.to_datetime(start) > pd.to_datetime(end):
                raise ValueError("'start' must be before 'end'.")

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

        # freq checks
        if not isinstance(freq, str):
            raise TypeError("freq must be a string.")
        if freq not in ["D", "W", "M", "Q", "Y"]:
            raise ValueError("freq must be one of 'D', 'W', 'M', 'Q' or 'Y'.")

        # lag checks
        if not isinstance(lag, int):
            raise TypeError("lag must be an integer.")
        if lag < 0:
            raise ValueError("lag must be non-negative.")

        # xcat_aggs checks
        if not isinstance(xcat_aggs, list):
            raise TypeError("xcat_aggs must be a list.")
        if not all(isinstance(xcat_agg, str) for xcat_agg in xcat_aggs):
            raise ValueError("All elements in xcat_aggs must be strings.")
        if len(xcat_aggs) != 2:
            raise ValueError("xcat_aggs must have exactly two elements.")

        # generate_labels checks
        if generate_labels is not None:
            if not callable(generate_labels):
                raise TypeError("generate_labels must be a callable.")
            
        # drop_nas checks
        if not isinstance(drop_nas, bool):
            raise TypeError("drop_nas must be a boolean.")

    def _check_run(
        self,
        name,
        outer_splitter,
        inner_splitters,
        models,
        hyperparameters,
        scorers,
        normalize_fold_results,
        search_type,
        cv_summary,
        include_train_folds,
        n_iter,
        split_functions,
        n_jobs_outer,
        n_jobs_inner,
    ):
        """
        Input parameter checks for the run method.

        Parameters
        ----------
        name : str
            Name of the sequential optimization pipeline.
        models : dict
            Compatible `scikit-learn` model objects.
        outer_splitter : BasePanelSplit
            Outer splitter for the learning process.
        inner_splitters : dict
            Inner splitters for the learning process.
        hyperparameters : dict
            Hyperparameter grids.
        scorers : dict
            Compatible `scikit-learn` scoring functions.
        normalize_fold_results : bool
            Whether to normalize the scores across folds before combining them.
        search_type : str
            Search type for hyperparameter optimization.
        cv_summary : str or callable
            Summary function to condense cross-validation scores in each fold to a single
            value, against which different hyperparameter choices can be compared.
        include_train_folds : bool
            Whether to calculate cross-validation statistics on the training folds in 
            additional to the test folds. If True, the cross-validation estimator will be
            a function of both training data and test data.
        n_iter : int
            Number of iterations for random or bayesian hyperparameter optimization.
        split_functions : dict
            Dictionary of functions associated with each inner splitter describing how to
            increase the number of splits as a function of the number of iterations passed.
        n_jobs_outer : int
            Number of jobs to run in parallel for the outer loop.
        n_jobs_inner : int
            Number of jobs to run in parallel for the inner loop.
        """
        # name
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        
        # models
        if not isinstance(models, dict):
            raise TypeError("The models argument must be a dictionary.")
        if models == {}:
            raise ValueError("The models dictionary cannot be empty.")
        for key in models.keys():
            if not isinstance(key, str):
                raise ValueError("The keys of the models dictionary must be strings.")
            if not isinstance(models[key], (RegressorMixin, ClassifierMixin, Pipeline)):
                raise ValueError(
                    "The values of the models dictionary must be sklearn predictors or "
                    "pipelines."
                )
            # Check that the model is compatible with the data
            if not hasattr(models[key], "fit") or not hasattr(models[key], "predict"):
                raise ValueError(
                    "The entered models must have 'fit' and 'predict' methods."
                )
            if not self.drop_nas:
                # Generate data with nas to check if the model can handle them
                X = pd.DataFrame(
                    data =  np.random.rand(20, 5),
                    columns = [f"feature_{i}" for i in range(5)],
                     # multi index to simulate panel data
                    index = pd.MultiIndex.from_product(
                        [["cid1", "cid2"], pd.date_range("2020-01-01", periods=10, freq="D")],
                        names=["cid", "real_date"],
                    )
                )
                # Add some missing values
                X.iloc[4:8, 3] = np.nan  # Introduce NaNs in the first column
                y = pd.Series(np.random.rand(20), index=X.index)
                try:
                    models[key].fit(X, y)
                except Exception as e:
                    raise ValueError(
                        f"The model {key} cannot handle missing values: {str(e)}"
                    )

        # outer splitter
        if outer_splitter:
            if not isinstance(
                outer_splitter, (WalkForwardPanelSplit)
            ):
                raise TypeError(
                    "outer_splitter must inherit from msl.splitters.WalkForwardPanelSplit"
                )
            
        # First check that both inner_splitters and scorers are None
        # or none of them are None.
        none_condition = all(
            (
                inner_splitters is None,
                scorers is None,
            ),
        )
        not_none_condition = all(
            (
                inner_splitters is not None,
                scorers is not None,
            ),
        )
        if not (none_condition or not_none_condition):
            raise ValueError(
                "Either both inner_splitters and scorers must be None "
                "or none of them can be None."
            )
        
        # Now check that if they are None, then only one model can be specified 
        # and no hyperparameters can be specified.
        if none_condition:
            if hyperparameters is not None or len(models) > 1:
                raise ValueError(
                    "If inner_splitters and scorers are None, then only one model can be "
                    "specified and no hyperparameters can be specified."
                )
        else:
            if hyperparameters is None:
                raise ValueError(
                    "If inner_splitters and scorers are not None, then hyperparameters "
                    "must be specified. This can be a dictionary that specifies "
                    "no hyperparameters if necessary."
                )
        
        # inner splitters
        if inner_splitters is not None:
            if not isinstance(inner_splitters, dict):
                raise TypeError("inner splitters should be specified as a dictionary")
            if inner_splitters == {}:
                raise ValueError("The inner splitters dictionary cannot be empty.")

            for names in inner_splitters.keys():
                if not isinstance(names, str):
                    raise ValueError(
                        "The keys of the inner splitters dictionary must be strings."
                    )
                if not isinstance(inner_splitters[names], BaseCrossValidator):
                    raise ValueError(
                        "The values of the inner splitters dictionary must be instances of BaseCrossValidator."
                    )

        # hyperparameters
        if hyperparameters is not None:
            if not isinstance(hyperparameters, dict):
                raise TypeError("hyperparameters must be a dictionary.")
            for pipe_name, pipe_params in hyperparameters.items():
                if not isinstance(pipe_name, str):
                    raise ValueError(
                        "The keys of the hyperparameters dictionary must be strings."
                    )
                if isinstance(pipe_params, dict):
                    self._check_hyperparam_grid(search_type, pipe_params)
                elif isinstance(pipe_params, list):
                    for param_set in pipe_params:
                        if not isinstance(param_set, dict):
                            raise ValueError(
                                "The values of the hyperparameters dictionary must be dictionaries or lists."
                            )
                        self._check_hyperparam_grid(search_type, param_set)
                else:
                    raise ValueError(
                        "The values of the hyperparameters dictionary must be dictionaries or lists."
                    )
            # Check that the keys of the hyperparameter grid match those in the models dict
            if sorted(hyperparameters.keys()) != sorted(models.keys()):
                raise ValueError(
                    "The keys in the hyperparameter grid must match those in the models "
                    "dictionary."
                )

        # scorers
        if scorers is not None:
            if not isinstance(scorers, dict):
                raise TypeError("scorers must be a dictionary.")
            if scorers == {}:
                raise ValueError("The scorers dictionary cannot be empty.")
            for key in scorers.keys():
                if not isinstance(key, str):
                    raise ValueError("The keys of the scorers dictionary must be strings.")
                if not callable(scorers[key]):
                    raise ValueError(
                        "The values of the scorers dictionary must be callable scoring functions."
                    )

        # search_type
        if not_none_condition:
            if not isinstance(search_type, str):
                raise TypeError("search_type must be a string.")
            if search_type not in ["grid", "prior", "bayes"]:
                raise ValueError("search_type must be one of 'grid', 'prior' or 'bayes'.")
            if search_type == "bayes":
                raise NotImplementedError(
                    "Bayesian hyperparameter search is not yet implemented."
                )

        # cv_summary
        if not_none_condition:
            if not isinstance(cv_summary, str) and not callable(cv_summary):
                raise TypeError("cv_summary must be a string or a callable.")
            if isinstance(cv_summary, str):
                if cv_summary not in [
                    "mean",
                    "median",
                    "mean-std",
                    "mean/std",
                    "mean-std-ge",
                ]:
                    raise ValueError(
                        "cv_summary must be one of 'mean', 'median', 'mean-std', 'mean-std-ge' or ",
                        "'mean/std'"
                    )
            else:
                try:
                    test_summary = cv_summary([1, 2, 3])
                except Exception as e:
                    raise TypeError(
                        "cv_summary must be a function that takes a list of scores and returns "
                        "a single value. Check the validity of cv_summary. Error raised when "
                        "testing the function with [1, 2, 3]: {e}"
                    )
                if not isinstance(test_summary, numbers.Number) and not isinstance(bool):
                    raise TypeError(
                        "cv_summary must be a function that takes a list of scores and returns "
                        "a single value. Check whether the output of cv_summary is a number."
                    )
            
        # include_train_folds
        if not_none_condition:
            if not isinstance(include_train_folds, bool):
                raise TypeError("include_train_folds must be a boolean.")
            if include_train_folds:
                if cv_summary in ["mean", "median", "mean-std", "mean/std"]:
                    warnings.warn(
                        "include_train_folds is True, which means that evaluation on training "
                        "folds is included in the cross-validation estimator. Check that "
                        "cv_summary is set appropriately.",
                        UserWarning,
                    )
            if cv_summary == "mean-std-ge":
                if not include_train_folds:
                    raise ValueError(
                        "include_train_folds must be True if cv_summary is 'mean-std-ge'."
                    )

        # n_iter
        if not_none_condition:
            if search_type == "prior":
                if not isinstance(n_iter, int):
                    raise TypeError("If search_type is 'prior', n_iter must be an integer.")
                if n_iter < 1:
                    raise ValueError("The n_iter argument must be greater than zero.")
            elif n_iter is not None and not isinstance(n_iter, int):
                raise ValueError("n_iter must only be used if search_type is 'prior'.")

        # normalize_fold_results
        if not_none_condition:
            if not isinstance(normalize_fold_results, bool):
                raise TypeError("normalize_fold_results must be a boolean.")
            if normalize_fold_results:
                if search_type == "grid":
                    for model in hyperparameters.keys():
                        num_models = sum(
                            [
                                len(hyperparameters[model][hparam])
                                for hparam in hyperparameters[model].keys()
                            ]
                        )
                        if num_models < 2:
                            raise ValueError(
                                "normalize_fold_results cannot be True if there are less than 2 candidate models. "
                                f"This is the case for the model {model}."
                            )
                        if num_models == 2:
                            warnings.warn(
                                "normalize_fold_results is True but there are only two candidate models for "
                                f"the model {model}. It is recommended for at least three candidate models "
                                "to be available for normalization to be meaningful.",
                                UserWarning,
                            )
                elif search_type == "prior":
                    if n_iter < 2:
                        raise ValueError(
                            "normalize_fold_results cannot be True if n_iter is less than 2."
                        )
                    if n_iter == 2:
                        warnings.warn(
                            "normalize_fold_results is True but n_iter is 2. It is recommended for n_iter to be "
                            "at least 3 for normalization to be meaningful.",
                            UserWarning,
                        )

        # split_functions
        if not_none_condition and split_functions is not None:
            if not isinstance(split_functions, dict):
                raise TypeError("split_functions must be a dictionary.")
            if len(
                set(split_functions.keys()).intersection(set(inner_splitters.keys()))
            ) != len(inner_splitters):
                raise ValueError(
                    "The keys of the split_functions dictionary must match the keys of the inner_splitters dictionary."
                )
            for key in split_functions.keys():
                if not isinstance(key, str):
                    raise ValueError(
                        "The keys of the split_functions dictionary must be strings."
                    )
                if split_functions[key] is not None:
                    if not callable(split_functions[key]):
                        raise ValueError(
                            "The values of the split_functions dictionary must be callables or None."
                        )

        # n_jobs_outer
        if not isinstance(n_jobs_outer, int):
            raise TypeError("n_jobs_outer must be an integer.")
        if n_jobs_outer < 1:
            if n_jobs_outer != -1:
                raise ValueError(
                    "n_jobs_outer must be greater than zero or equal to -1."
                )

        # n_jobs_inner
        if not_none_condition:
            if not isinstance(n_jobs_inner, int):
                raise TypeError("n_jobs_inner must be an integer.")
            if n_jobs_inner < 1:
                if n_jobs_inner != -1:
                    raise ValueError(
                        "n_jobs_inner must be greater than zero or equal to -1."
                    )

    def _check_hyperparam_grid(self, search_type, pipe_params):
        if pipe_params != {}:
            for hparam_key, hparam_values in pipe_params.items():
                if not isinstance(hparam_key, str):
                    raise ValueError(
                        "The keys of the inner hyperparameters dictionaries must be "
                        "strings."
                    )
                if search_type == "grid":
                    if not isinstance(hparam_values, list):
                        raise ValueError(
                            "The values of the inner hyperparameters dictionaries must be "
                            "lists if hparam_type is 'grid'."
                        )
                    if len(hparam_values) == 0:
                        raise ValueError(
                            "The values of the inner hyperparameters dictionaries cannot be "
                            "empty lists."
                        )
                elif search_type == "prior":
                    # hparam_values must either be a list or a scipy.stats distribution
                    # create typeerror
                    if isinstance(hparam_values, list):
                        if len(hparam_values) == 0:
                            raise ValueError(
                                "The values of the inner hyperparameters dictionaries cannot "
                                "be empty lists."
                            )
                    else:
                        if not hasattr(hparam_values, "rvs"):
                            raise ValueError(
                                "Invalid random hyperparameter search dictionary element "
                                f"for hyperparameter {hparam_key}. The dictionary values "
                                "must be scipy.stats distributions."
                            )

    def _checks_models_heatmap(
        self,
        name,
        title=None,
        cap=5,
        figsize=(12, 8),
        title_fontsize=None,
        tick_fontsize=None,
    ):
        """
        Checks for the models_heatmap method.

        Parameters
        ----------
        name : str
            Name of the sequential optimization pipeline.
        title : str, optional
            Title of the heatmap. Default has the form "Model Selection Heatmap for {name}".
        cap : int, optional
            Maximum number of models to display. Default is 5, with limit of 10.
            The chosen models are the 'cap' most frequently occurring in the process.
        figsize : tuple, optional
            Tuple of floats or ints denoting the figure size. Default is (12, 8).
        title_fontsize : int, optional
            Font size for the title. Default is None.
        tick_fontsize : int, optional
            Font size for the ticks. Default is None.
        """
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.chosen_models.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )
        if not isinstance(cap, int):
            raise TypeError("The cap must be an integer.")
        if cap < 0:
            raise ValueError("The cap must be greater than zero.")
        if cap > 10:
            raise ValueError("The cap must be 10 or lower.")

        if title is None:
            title = f"Model Selection Heatmap for {name}"
        if not isinstance(title, str):
            raise TypeError("The figure title must be a string.")

        if not isinstance(figsize, tuple):
            raise TypeError("The figsize argument must be a tuple.")
        if len(figsize) != 2:
            raise ValueError("The figsize argument must be a tuple of length 2.")
        for element in figsize:
            if not isinstance(element, (int, float)):
                raise TypeError(
                    "The elements of the figsize tuple must be floats or ints."
                )

        if title_fontsize is not None:
            if not isinstance(title_fontsize, int):
                raise TypeError("The title_fontsize argument must be an integer.")
            if title_fontsize < 0:
                raise ValueError("The title_fontsize argument must be non-negative.")

        if tick_fontsize is not None:
            if not isinstance(tick_fontsize, int):
                raise TypeError("The tick_fontsize argument must be an integer.")
            if tick_fontsize < 0:
                raise ValueError("The tick_fontsize argument must be non-negative.")

    def _get_base_splits(self, inner_splitters):
        """Get the initial number of splits for each splitter."""
        base_splits = {}
        for splitter_name, splitter in inner_splitters.items():
            if hasattr(splitter, "n_splits"):
                base_splits[splitter_name] = splitter.n_splits
        return base_splits

    def _get_n_splits_add(self, iteration, outer_splitters, split_functions) -> dict:
        """Call split functions to determine the number of splits to add to each splitter."""
        if split_functions is not None:
            return {
                splitter_name: (
                    int(np.ceil(split_function(iteration * outer_splitters.test_size)))
                    if split_function is not None
                    else 0
                )
                for splitter_name, split_function in split_functions.items()
            }
        else:
            return None

    def _remove_results(self, conditions):
        """
        Remove rows from results DataFrames based on conditions.

        Parameters
        ----------
        conditions : list of tuples
            List of tuples containing the attribute name, column name and value to filter on.
        """
        for attr, column, value in conditions:
            df = getattr(self, attr)
            if value in df[column].unique():
                setattr(self, attr, df[df[column] != value])
