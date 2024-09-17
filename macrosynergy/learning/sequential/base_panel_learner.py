"""
Base class for sequential learning over a panel.
"""

import numpy as np
import pandas as pd

from macrosynergy.management import categories_df
from macrosynergy.learning import (
    ExpandingFrequencyPanelSplit,
    ExpandingIncrementPanelSplit,
    BasePanelSplit,
)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import warnings
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from functools import partial


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
        # slip?
        xcat_aggs=["last", "sum"],
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
        cids : list, optional
            Cross-sections to be included. Default is all in the dataframe.
        start : str, optional
            Start date for considered data in subsequent analysis in ISO 8601 format.
            Default is None i.e. the earliest date in the dataframe.
        end : str, optional
            End date for considered data in subsequent analysis in ISO 8601 format.
            Default is None i.e. the latest date in the dataframe.
        blacklist : list, optional
            Blacklisting dictionary specifying date ranges for which cross-sectional
            information should be excluded. The keys are cross-sections and the values
            are tuples of start and end dates in ISO 8601 format. Default is None.
        freq : str, optional
            Frequency of the data. Default is "M" for monthly.
        lag : int, optional
            Number of periods to lag the independent variables. Default is 1.
        xcat_aggs : list, optional
            List of exactly two aggregation methods for downsampling data to the frequency
            specified in the freq parameter. The first parameter pertains to all
            independent variable downsampling, whilst the second corresponds with the
            target category. Default is ["last", "sum"].
        """
        # Checks
        self._check_init(df, xcats, cids, start, end, blacklist, freq, lag, xcat_aggs)

        # Attributes
        self.df = df
        self.xcats = xcats
        self.cids = cids
        self.start = start
        self.end = end
        self.blacklist = blacklist
        self.freq = freq
        self.lag = lag
        self.xcat_aggs = xcat_aggs

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
            .dropna()
            .sort_index()
        )

        # Create X and y
        self.X = df_long.iloc[:, :-1]
        self.y = df_long.iloc[:, -1]

        # Store necessary index information
        self.index = self.X.index
        self.date_levels = self.index.get_level_values(1)
        self.xs_levels = self.index.get_level_values(0)
        self.unique_date_levels = sorted(self.date_levels.unique())
        self.unique_xs_levels = sorted(self.xs_levels.unique())

        # Create initial dataframe to store model selection data from the learning process
        self.chosen_models = pd.DataFrame(
            columns=["real_date", "name", "model_type", "hparams", "n_splits_used"]
        )

    def run(
        self,
        name,
        outer_splitter,
        inner_splitters,
        models,
        hyperparameters,
        scorers,
        search_type="grid",
        cv_summary="mean",  # can be mean, median or lambda function on fold scores.
        n_iter=100,  # number of iterations for random or bayes
        split_functions=None,
        n_jobs_outer=-1,
        n_jobs_inner=1,
        # strategy_eval_periods = None,
    ):
        """
        Run a learning process over a panel.

        Parameters
        ----------
        name : str
            Category name for the forecasted panel resulting from the learning process.
        outer_splitter : WalkForwardPanelSplit
            Outer splitter for the learning process.
        inner_splitters : BaseCrossValidator or list
            Inner splitters for the learning process. If an instance of BaseCrossValidator,
            then this is used for cross-validation. If a list, then each splitter is used
            before the results for each are averaged.
        models : dict
            Dictionary of model names and compatible `scikit-learn` model objects.
        hyperparameters : dict
            Dictionary of model names and hyperparameter grids.
        scorers : callable or list
            `scikit-learn` compatible scoring function or list of scoring functions.
        search_type : str
            Search type for hyperparameter optimization. Default is "grid".
            Options are "grid", "prior" and "bayes".
        cv_summary : str or callable
            Summary function for determining cross-validation scores given scores for
            each validation fold. Default is "mean". Can also be "median" or a function
            that takes a list of scores and returns a single value.
        n_iter : int
            Number of iterations for random or bayesian hyperparameter optimization.
        split_functions : callable or list, optional
            Callable or list of callables for determining the number of cross-validation
            splits to add to the initial number  as a function of the number of iterations passed in the
            sequential learning process. Default is None.
        n_jobs_outer : int, optional
            Number of jobs to run in parallel for the outer loop. Default is -1.
        n_jobs_inner : int, optional
            Number of jobs to run in parallel for the inner loop. Default is 1.
        """
        # Checks
        (
            name,
            outer_splitter,
            inner_splitters,
            models,
            hyperparameters,
            scorers,
            search_type,
            cv_summary,
            n_iter,
            split_functions,
            n_jobs_outer,
            n_jobs_inner,
        ) = self._check_run(
            name,
            outer_splitter,
            inner_splitters,
            models,
            hyperparameters,
            scorers,
            search_type,
            cv_summary,
            n_iter,
            split_functions,
            n_jobs_outer,
            n_jobs_inner,
        )

        # Determine all outer splits and run the learning process in parallel
        train_test_splits = list(outer_splitter.split(self.X, self.y))

        # Return list of results
        optim_results = tqdm(
            Parallel(n_jobs=n_jobs_outer, return_as="generator")(
                delayed(self._worker)(
                    name=name,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    inner_splitters=inner_splitters,
                    models=models,
                    hyperparameters=hyperparameters,
                    scorers=scorers,
                    cv_summary=cv_summary,
                    search_type=search_type,
                    n_iter=n_iter,
                    n_splits_add=(
                        [
                            np.ceil(split_function(iteration))
                            for idx, split_function in enumerate(split_functions)
                        ]
                        if split_functions is not None
                        else None
                    ),
                    n_jobs_inner=n_jobs_inner,
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
        search_type,
        n_iter,
        n_splits_add,
        n_jobs_inner,
    ):
        """
        Worker function for parallel processing of the learning process.
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
                np.searchsorted(self.date_levels, sorted_test_date_levels, side="left")
                - self.lag
            )
            adj_test_date_levels: pd.DatetimeIndex = pd.DatetimeIndex(
                [self.date_levels[i] if i >= 0 else pd.NaT for i in locs]
            )
            # Now formulate correct index
            date_map = dict(zip(test_date_levels, adj_test_date_levels))
            mapped_dates = test_date_levels.map(date_map)
            test_index = pd.MultiIndex.from_arrays(
                [test_xs_levels, mapped_dates], names=["cid", "real_date"]
            )

        if n_splits_add is not None:
            inner_splitters_adj = [inner_splitter for inner_splitter in inner_splitters]
            for idx, inner_splitter in enumerate(inner_splitters_adj):
                inner_splitter.n_splits += n_splits_add[idx]
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
            n_iter=n_iter,
            cv_summary=cv_summary,
            n_jobs_inner=n_jobs_inner,
        )

        # Handle case where no model was selected
        if optim_name is None:
            warnings.warn(
                f"No model was selected for {name} at time {self.date_levels[train_idx].max()}",
                " Hence, resulting signals are set to zero.",
                RuntimeWarning,
            )
            preds = np.zeros(y_test.shape)
            prediction_data = [name, test_index, preds]
            modelchoice_data = [
                self.date_levels[train_idx],
                name,
                None,  # Model selected
                None,  # Hyperparameters selected
                [inner_splitter.n_splits for inner_splitter in inner_splitters_adj],
            ]
            other_data = None

        else:
            # # Then a model was selected
            # optim_model.fit(X_train, y_train)

            # # If optim_model has a create_signal method, use it otherwise use predict
            # if hasattr(optim_model, "create_signal"):
            #     if callable(getattr(optim_model, "create_signal")):
            #         preds: np.ndarray = optim_model.create_signal(X_test)
            #     else:
            #         preds: np.ndarray = optim_model.predict(X_test)
            # else:
            #     preds: np.ndarray = optim_model.predict(X_test)

            # prediction_data = [name, test_index, preds]

            # # Store model choice information
            # modelchoice_data = [
            #     self.date_levels[train_idx],
            #     name,
            #     optim_name,
            #     optim_params,
            #     int(n_splits),
            # ]

            # # Store other information - inherited classes can specify this method to store coefficients, intercepts etc if needed
            # other_data: List[List] = self._extract_model_info(
            #     name,
            #     self.date_levels[train_idx],
            #     optim_model,
            # )
            # optim_model.fit(X_train, y_train)

            # Store model selection data
            modelchoice_data: dict = self.store_modelchoice_data(
                pipeline_name=name,
                optimal_model=optim_model,
                optimal_model_name=optim_name,
                optimal_model_score=optim_score,
                optimal_model_params=optim_params,
                n_splits=[
                    inner_splitter.n_splits for inner_splitter in inner_splitters_adj
                ],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                timestamp=adj_test_date_levels.min(),
            )

            # Store quantamental data
            quantamental_data: dict = self.store_quantamental_data(
                pipeline_name=name,
                model=optim_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                adjusted_test_index=test_index,
            )

            # Store other data
            other_data: dict = self.store_other_data(
                pipeline_name=name,
                optimal_model=optim_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                timestamp=adj_test_date_levels.min(),
            )

        return (
            quantamental_data,
            modelchoice_data,
            other_data,
        )

    def _model_search(
        self,
        X_train,
        y_train,
        inner_splitters,
        models,
        hyperparameters,
        scorers,
        search_type,
        n_iter,
        cv_summary,
        n_jobs_inner,
    ):
        """
        TODO later
        """
        optim_name = None
        optim_model = None
        optim_score = -np.inf
        optim_params = None

        cv_splits = []
        for splitter in inner_splitters:
            cv_splits.extend(list(splitter.split(X=X_train, y=y_train)))
        for model_name, model in models.items():
            # For each model, find the optimal hyperparameters
            if search_type == "grid":
                search_object = GridSearchCV(
                    estimator=model,
                    param_grid=hyperparameters[model_name],
                    scoring=scorers,
                    n_jobs=n_jobs_inner,
                    refit=partial(
                        self._model_selection, cv_summary=cv_summary, scorers=scorers
                    ),
                    cv=cv_splits,
                )
            elif search_type == "prior":
                search_type = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=hyperparameters[model_name],
                    n_iter=n_iter,
                    scoring=scorers,
                    n_jobs=n_jobs_inner,
                    refit=partial(
                        self._model_selection, cv_summary=cv_summary, scorers=scorers
                    ),
                    cv=cv_splits,
                )
            else:
                raise NotImplementedError(
                    f"Search type {search_type} is not implemented."
                )

            try:
                search_object.fit(X_train, y_train)
            except Exception as e:
                warnings.warn(
                    f"Error running a hyperparameter search for {model_name}: {e}",
                    RuntimeWarning,
                )
            score = self._model_selection(
                search_object.cv_results_, cv_summary, scorers, return_index=False
            )
            if score > optim_score:
                optim_name = model_name
                optim_model = search_object.best_estimator_
                optim_score = score
                optim_params = search_object.best_params_

        return optim_name, optim_model, optim_score, optim_params

    def _model_selection(self, cv_results, cv_summary, scorers, return_index=True):
        """
        Determine index of best estimator in a scikit-learn cv_results summary dictionary,
        as well as a cv_summary function indicating how to summarize the cross-validation
        scores across folds.

        For each hyperparameter choice, determine test metrics for each cv fold. Transform
        these scores by min-max scaling for each metric. Then, for each hyperparameter choice,
        average the different metrics for each fold, resulting in a single score for each
        fold. Finally, cv_summary is applied to these scores to determine the best hyperparameter
        choice.
        """
        cv_results = pd.DataFrame(cv_results)
        metric_columns = [
            col
            for col in cv_results.columns
            if col.startswith("split") and "test" in col
        ]

        # For each metric, summarise the scores across folds for each hyperparameter choice
        # using cv_summary
        for scorer in scorers.keys():
            # Extract the test scores for each fold for that scorer
            scorer_columns = [col for col in metric_columns if scorer in col]
            if cv_summary == "mean":
                cv_results[f"{scorer}_summary"] = cv_results[scorer_columns].mean(
                    axis=1
                )
            elif cv_summary == "median":
                cv_results[f"{scorer}_summary"] = cv_results[scorer_columns].median(
                    axis=1
                )
            else:
                cv_results[f"{scorer}_summary"] = cv_results[scorer_columns].apply(
                    cv_summary, axis=1
                )

        # Now apply min-max scaling to the summary scores
        scaler = MinMaxScaler()
        summary_cols = [f"{scorer}_summary" for scorer in scorers.keys()]
        cv_results[summary_cols] = scaler.fit_transform(cv_results[summary_cols])

        # Now average the summary scores for each scorer
        cv_results["final_score"] = cv_results[summary_cols].mean(axis=1)

        # Return index of best estimator
        if return_index:
            return cv_results["final_score"].idxmax()
        else:
            return cv_results["final_score"].max()

    @abstractmethod
    def store_quantamental_data(
        self,
        pipeline_name,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        adjusted_test_index,
    ):
        """
        Abstract method for storing quantamental data.
        """
        pass

    def store_modelchoice_data(
        self,
        pipeline_name,
        optimal_model,
        optimal_model_name,
        optimal_model_score,
        optimal_model_params,
        n_splits,
        X_train,
        y_train,
        X_test,
        y_test,
        timestamp,
    ):
        """
        Store model selection information for training set (X_train, y_train)
        """
        if optimal_model is not None:
            optim_name = optimal_model_name
            optim_score = optimal_model_score
            optim_params = optimal_model_params
        else:

            optim_name = ("None",)
            optim_score = (-np.inf,)
            optim_params = ({},)

        data = [timestamp, pipeline_name, optim_name, optim_score, optim_params]
        data.extend(n_splits)

        return {"model_choice": data}

    @abstractmethod
    def store_other_data(self, optimal_model, X_train, y_train, X_test, y_test):
        """
        Abstract method for storing other data.
        """
        pass

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
    ):
        """
        Checks for the constructor.
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
        for xcat in xcats:
            if xcat not in df["xcat"].unique():
                raise ValueError(f"{xcat} not in the dataframe.")

        # cids checks
        if cids is not None:
            if not isinstance(cids, list):
                raise TypeError("cids must be a list.")
            if not all(isinstance(cid, str) for cid in cids):
                raise TypeError("All elements in cids must be strings.")
            for cid in cids:
                if cid not in df["cid"].unique():
                    raise ValueError(f"{cid} not in the dataframe.")

        # start checks
        if start is not None:
            if not isinstance(start, str):
                raise TypeError("start must be a string.")
            try:
                pd.to_datetime(start)
            except ValueError:
                raise ValueError("start must be in ISO 8601 format.")

        # end checks
        if end is not None:
            if not isinstance(end, str):
                raise TypeError("end must be a string.")
            try:
                pd.to_datetime(end)
            except ValueError:
                raise ValueError("end must be in ISO 8601 format.")

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
        if len(xcat_aggs) != 2:
            raise ValueError("xcat_aggs must have exactly two elements.")

    def _check_run(
        self,
        name,
        outer_splitter,
        inner_splitters,
        models,
        hyperparameters,
        scorers,
        search_type,
        cv_summary,
        n_iter,
        split_functions,
        n_jobs_outer,
        n_jobs_inner,
    ):
        # name
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        # outer splitter
        # TODO: come back and change to WalkForwardPanelSplit
        if not isinstance(
            outer_splitter, (ExpandingFrequencyPanelSplit, ExpandingIncrementPanelSplit)
        ):
            raise TypeError(
                "outer_splitter must be an instance of ExpandingFrequencyPanelSplit or ExpandingIncrementPanelSplit."
            )

        # inner splitter
        if not isinstance(inner_splitters, list):
            if not isinstance(inner_splitters, BasePanelSplit):
                raise TypeError(
                    "inner splitter must be a panel cross-validator subclassing BasePanelSplit."
                )
            inner_splitters = [inner_splitters]
        else:
            for inner_splitter in inner_splitters:
                if not isinstance(inner_splitter, BasePanelSplit):
                    raise TypeError(
                        "inner splitter must be a list of panel cross-validator subclassing BasePanelSplit."
                    )

        # models
        if models == {}:
            raise ValueError("The models dictionary cannot be empty.")
        if not isinstance(models, dict):
            raise TypeError("The models argument must be a dictionary.")
        for key in models.keys():
            if not isinstance(key, str):
                raise TypeError("The keys of the models dictionary must be strings.")
            if not isinstance(models[key], (BaseEstimator, Pipeline)):
                raise TypeError(
                    "The values of the models dictionary must be sklearn predictors or "
                    "pipelines."
                )

        # hyperparameters
        if not isinstance(hyperparameters, dict):
            raise TypeError("The hyperparameters argument must be a dictionary.")
        for pipe_name, pipe_params in hyperparameters.items():
            if not isinstance(pipe_name, str):
                raise TypeError(
                    "The keys of the hyperparameters dictionary must be strings."
                )
            if not isinstance(pipe_params, dict):
                raise TypeError(
                    "The values of the hyperparameters dictionary must be dictionaries."
                )
            if pipe_params != {}:
                for hparam_key, hparam_values in pipe_params.items():
                    if not isinstance(hparam_key, str):
                        raise TypeError(
                            "The keys of the inner hyperparameters dictionaries must be "
                            "strings."
                        )
                    if hyperparameters == "grid":
                        if not isinstance(hparam_values, list):
                            raise TypeError(
                                "The values of the inner hyperparameters dictionaries must be "
                                "lists if hparam_type is 'grid'."
                            )
                        if len(hparam_values) == 0:
                            raise ValueError(
                                "The values of the inner hyperparameters dictionaries cannot be "
                                "empty lists."
                            )
                    elif hyperparameters == "random":
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
        # Check that the keys of the hyperparameter grid match those in the models dict
        if sorted(hyperparameters.keys()) != sorted(models.keys()):
            raise ValueError(
                "The keys in the hyperparameter grid must match those in the models "
                "dictionary."
            )

        # scorers
        if not isinstance(scorers, list):
            if not callable(scorers):
                raise TypeError("scorers must be a callable scoring function.")
            scorers = [scorers]
        else:
            for scorer in scorers:
                if not callable(scorer):
                    raise TypeError(
                        "scorers must be a list of callable scoring functions."
                    )

        # search_type
        if not isinstance(search_type, str):
            raise TypeError("search_type must be a string.")
        if search_type not in ["grid", "prior", "bayes"]:
            raise ValueError("search_type must be one of 'grid', 'prior' or 'bayes'.")

        # cv_summary
        if not isinstance(cv_summary, (str, callable)):
            raise TypeError("cv_summary must be a string or a callable.")
        if isinstance(cv_summary, str):
            if cv_summary not in ["mean", "median"]:
                raise ValueError("cv_summary must be one of 'mean' or 'median'.")
        else:
            try:
                test_summary = cv_summary([1, 2, 3])
            except Exception as e:
                raise TypeError(
                    "cv_summary must be a function that takes a list of scores and returns "
                    "a single value. Check the validity of cv_summary."
                )
            if not isinstance(test_summary, (int, float, np.int_, np.float_)):
                raise TypeError(
                    "cv_summary must be a function that takes a list of scores and returns "
                    "a single value. Check whether the output of cv_summary is a number."
                )

        # n_iter
        if search_type == "random":
            if type(n_iter) != int:
                raise TypeError("The n_iter argument must be an integer.")
            if n_iter < 1:
                raise ValueError("The n_iter argument must be greater than zero.")

        # split_functions
        if split_functions is not None:
            if not isinstance(split_functions, list):
                if not callable(split_functions):
                    raise TypeError("split_functions must be a callable.")
                split_functions = [split_functions]
            if len(split_functions) != len(inner_splitters):
                raise ValueError(
                    "The length of the split_functions list must be the same as the length "
                    "of the inner_splitters list. If some splitters require no adjustment,"
                    "then None elements should be included in the split_functions list."
                )
            for split_function in split_functions:
                if not callable(split_function):
                    raise TypeError("split_functions must be a list of callables.")

        # n_jobs_outer
        if not isinstance(n_jobs_outer, int):
            raise TypeError("n_jobs_outer must be an integer.")
        if n_jobs_outer < 1:
            if n_jobs_outer != -1:
                raise ValueError(
                    "n_jobs_outer must be greater than zero or equal to -1."
                )

        # n_jobs_inner
        if not isinstance(n_jobs_inner, int):
            raise TypeError("n_jobs_inner must be an integer.")
        if n_jobs_inner < 1:
            if n_jobs_inner != -1:
                raise ValueError(
                    "n_jobs_inner must be greater than zero or equal to -1."
                )

        return (
            name,
            outer_splitter,
            inner_splitters,
            models,
            hyperparameters,
            scorers,
            search_type,
            cv_summary,
            n_iter,
            split_functions,
            n_jobs_outer,
            n_jobs_inner,
        )
