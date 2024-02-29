"""
Class to handle the calculation of quantamental predictions based on adaptive
hyperparameter and model selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from typing import List, Union, Dict, Optional, Callable, Tuple
from tqdm import tqdm

import warnings

from joblib import Parallel, delayed

from macrosynergy.learning.panel_time_series_split import (
    BasePanelSplit,
    ExpandingIncrementPanelSplit,
    RollingKFoldPanelSplit,
)


class SignalOptimizer:
    def __init__(
        self,
        inner_splitter: BasePanelSplit,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        blacklist: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ):
        """
        Class for sequential optimization of raw signals based on quantamental features.
        Optimization is performed through nested cross-validation, with the outer splitter
        an instance of `ExpandingIncrementPanelSplit` reflecting a pipeline through time
        simulating the experience of an investor. In each iteration of the outer splitter,
        a training and test set are created, and a grid search using the specified
        'inner_splitter' is performed to determine an optimal model amongst a set of
        candidate models. Once this is selected, the chosen model is used to make the test
        set forecasts. Lastly, we cast these forecasts back by a frequency period to
        account for the lagged features, creating point-in-time signals.

        The features in the dataframe, X, are expected to be lagged quantamental
        indicators, at a single native frequency unit, with the targets, in y, being the
        cumulative returns at the native frequency. By providing a blacklisting
        dictionary, preferably through macrosynergy.management.make_blacklist, the user
        can specify time periods to ignore.

        :param <BasePanelSplit> inner_splitter: Panel splitter that is used to split
            each training set into smaller (training, test) pairs for cross-validation.
            At present that splitter has to be an instance of `RollingKFoldPanelSplit`,
            `ExpandingKFoldPanelSplit` or `ExpandingIncrementPanelSplit`.
        :param <pd.DataFrame> X: Wide pandas dataframe of features and date-time indexes
            that capture the periods for which the signals are to be calculated.
            Since signals must make time seried predictions, the features in `X` must be
            lagged by one period, i.e., the values used for the current period must be
            those that were originally recorded for the previous period.
            The frequency of features (and targets) determines the frequency at which
            model predictions are made and evaluated. This means that if we have monthly
            data, the learning process uses the performance of monthly predictions.
        :param <Union[pd.DataFrame,pd.Series]> y: Pandas dataframe or series of targets
            corresponding with a time index equal to the features in `X`.
        :param <Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]> blacklist: cross-sections
            with date ranges that should be excluded from the data frame.

        Note:
        Optimization is based on expanding time series panels and maximizes a defined
        criterion over a grid of sklearn pipelines and hyperparameters of the involved
        models. The frequency of the input data sets `X` and `y` determines the frequency
        at which the training set is expanded. The training set itself is split into
        various (training, test) pairs by the `inner_splitter` argument for cross-
        validation. Based on inner cross-validation an optimal model is chosen and used
        for predicting the targets of the next period.
        A prediction for a particular cross-section and time period is made only if all
        required information has been available for that point.
        Optimized signals that are produced by the class are always stored for the
        end of the original data period that precedes the predicted period.
        For example, if the frequency of the input data set is monthly, signals for
        a month are recorded at the end of the previous month. If the frequency is working
        daily, signals for a day are recorded at the end of the previous business day.
        The date adjustment step ensures that the point-in-time principle is followed,
        in the JPMaQS format output of the class.

        # Example use:

        ```python
        # Suppose X_train and y_train comprise monthly-frequency features and targets
        so = SignalOptimizer(
            inner_splitter=RollingKFoldPanelSplit(n_splits=5),
            X=X_train,
            y=y_train,
        )

        # (1) Linear Regression signal with no hyperparameter optimisation
        so.calculate_predictions(
            name="OLS",
            models = {"linreg" : LinearRegression()},
            metric = make_scorer(mean_squared_error, greater_is_better=False),
            hparam_grid = {"linreg" : {}},
        )
        print(so.get_optimized_signals("OLS"))

        # (2) KNN signal with adaptive hyperparameter optimisation
        so.calculate_predictions(
            name="KNN",
            models = {"knn" : KNeighborsRegressor()},
            metric = make_scorer(mean_squared_error, greater_is_better=False),
            hparam_grid = {"knn" : {"n_neighbors" : [1, 2, 5]}},
        )
        print(so.get_optimized_signals("KNN"))

        # (3) Linear regression & KNN mixture signal with adaptive hyperparameter
            optimisation
        so.calculate_predictions(
            name="MIX",
            models = {"linreg" : LinearRegression(), "knn" : KNeighborsRegressor()},
            metric = make_scorer(mean_squared_error, greater_is_better=False),
            hparam_grid = {"linreg" : {}, "knn" : {"n_neighbors" : [1, 2, 5]}},
        )
        print(so.get_optimized_signals("MIX"))

        # (4) Visualise the models chosen by the adaptive signal algorithm for the
        #     nearest neighbors and mixture signals.
        so.models_heatmap(name="KNN")
        so.models_heatmap(name="MIX")
        ```
        """
        # Checks
        if not isinstance(inner_splitter, BasePanelSplit):
            raise TypeError(
                "The inner_splitter argument must be an instance of BasePanelSplit."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The X argument must be a pandas DataFrame.")
        if not isinstance(y, pd.Series) and not isinstance(y, pd.DataFrame):
            raise TypeError("The y argument must be a pandas Series or DataFrame.")
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of y must be datetime.date.")
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't "
                "match."
            )
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

        self.inner_splitter = inner_splitter

        self.X = X
        self.y = y
        self.blacklist = blacklist

        # Create an initial dataframes to store quantamental predictions and model choices
        self.preds = pd.DataFrame(columns=["cid", "real_date", "xcat", "value"])
        self.chosen_models = pd.DataFrame(
            columns=["real_date", "name", "model_type", "hparams"]
        )

    def calculate_predictions(
        self,
        name: str,
        models: Dict[str, Union[BaseEstimator, Pipeline]],
        metric: Callable,
        hparam_grid: Dict[str, Dict[str, List]],
        hparam_type: str = "grid",
        min_cids: int = 4,
        min_periods: int = 12 * 3,
        max_periods: Optional[int] = None,
        n_iter: Optional[int] = 10,
        n_jobs: Optional[int] = -1,
    ) -> None:
        """
        Calculate, store and return sequentially optimized signals for a given process.
        This method implements the nested cross-validation and subsequent signal
        generation. The name of the process, together with models to fit, hyperparameters
        to search over and a metric to optimize, are provided as compulsory arguments.

        :param <str> name: Label of signal optimization process.
        :param <Dict[str, Union[BaseEstimator,Pipeline]]> models: dictionary of sklearn
            predictors or pipelines.
        :param <Callable> metric: A sklearn scorer object that serves as the criterion
            for optimization.
        :param <str> hparam_type: Hyperparameter search type.
            This must be either "grid", "random" or "bayes". Default is "grid".
        :param <Dict[str, Dict[str, List]]> hparam_grid: Nested dictionary defining the
            hyperparameters to consider for each model. The outer dictionary needs keys
            representing the model name and should match the keys in the `models`.
            dictionary. The inner dictionary depends on the hyperparameter search type.
            If hparam_type is "grid", then the inner dictionary should have keys
            corresponding to the hyperparameter names and values equal to a list
            of hyperparameter values to search over. For example:
            hparam_grid = {
                "lasso" : {"alpha" : [1e-1, 1e-2, 1e-3]},
                "knn" : {"n_neighbors" : [1, 2, 5]}
            }.
            If hparam_type is "random", the inner dictionary needs keys corresponding
            to the hyperparameter names and values either equal to a distribution
            from which to sample or a list of them.
            For example:
            hparam_grid = {
                "lasso" : {"alpha" : scipy.stats.expon()},
                "knn" : {"n_neighbors" : scipy.stats.randint(low=1, high=10)}
            }.
            Distributions must provide a rvs method for sampling (such as those from
            scipy.stats.distributions).
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            and https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
            for more details.
        :param <int> min_cids: Minimum number of cross-sections required for the initial
            training set. Default is 4.
        :param <int> min_periods: minimum number of base periods of the input data
            frequency required for the initial training set. Default is 12.
        :param <Optional[int]> max_periods: maximum length of each training set in units
            of the input data frequency. If this maximum is exceeded, the earliest periods
            are cut off. Default is None, which means that the full training history is
            considered in each iteration.
        :param <int> n_iter: Number of iterations to run for random search. Default is 10.
        :param <int> n_jobs: Number of jobs to run in parallel. Default is -1, which uses
            all available cores.

        Note:
        The method produces signals for financial contract positions. They are calculated
        sequentially at the frequency of the input data set. Sequentially here means
        that the training set is expanded by one base period of the frequency.
        Each time the training set itself is split into  various (training, test) pairs by
        the `inner_splitter` argument. Based on inner cross-validation an optimal model
        is chosen and used for predicting the targets of the next period.
        """
        # Checks
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
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
        if not callable(metric):
            raise TypeError("The metric argument must be a callable object.")
        if not isinstance(hparam_type, str):
            raise TypeError("The hparam_type argument must be a string.")
        if hparam_type not in ["grid", "random", "bayes"]:
            raise ValueError(
                "Invalid hyperparameter search type. Must be 'grid', 'random' or 'bayes'."
            )
        if hparam_type == "bayes":
            raise NotImplementedError("Bayesian optimisation not yet implemented.")
        if not isinstance(hparam_grid, dict):
            raise TypeError("The hparam_grid argument must be a dictionary.")
        for pipe_name, pipe_params in hparam_grid.items():
            if not isinstance(pipe_name, str):
                raise TypeError(
                    "The keys of the hparam_grid dictionary must be strings."
                )
            if not isinstance(pipe_params, dict):
                raise TypeError(
                    "The values of the hparam_grid dictionary must be dictionaries."
                )
            if pipe_params != {}:
                for hparam_key, hparam_values in pipe_params.items():
                    if not isinstance(hparam_key, str):
                        raise TypeError(
                            "The keys of the inner hparam_grid dictionaries must be "
                            "strings."
                        )
                    if hparam_type == "grid":
                        if not isinstance(hparam_values, list):
                            raise TypeError(
                                "The values of the inner hparam_grid dictionaries must be "
                                "lists if hparam_type is 'grid'."
                            )
                        if len(hparam_values) == 0:
                            raise ValueError(
                                "The values of the inner hparam_grid dictionaries cannot be "
                                "empty lists."
                            )
                    elif hparam_type == "random":
                        # hparam_values must either be a list or a scipy.stats distribution
                        # create typeerror
                        if isinstance(hparam_values, list):
                            if len(hparam_values) == 0:
                                raise ValueError(
                                    "The values of the inner hparam_grid dictionaries cannot "
                                    "be empty lists."
                                )
                        else:
                            if not hasattr(hparam_values, "rvs"):
                                raise ValueError(
                                    "Invalid random hyperparameter search dictionary element "
                                    f"for hyperparameter {hparam_key}. The dictionary values "
                                    "must be scipy.stats distributions."
                                )
        if sorted(hparam_grid.keys()) != sorted(models.keys()):
            raise ValueError(
                "The keys in the hyperparameter grid must match those in the models "
                "dictionary."
            )
        if not isinstance(min_cids, int):
            raise TypeError("The min_cids argument must be an integer.")
        if min_cids < 1:
            raise ValueError("The min_cids argument must be greater than zero.")
        if not isinstance(min_periods, int):
            raise TypeError("The min_periods argument must be an integer.")
        if min_periods < 1:
            raise ValueError("The min_periods argument must be greater than zero.")
        if max_periods is not None:
            if not isinstance(max_periods, int):
                raise TypeError("The max_periods argument must be an integer.")
            if max_periods < 1:
                raise ValueError("The max_periods argument must be greater than zero.")
        if hparam_type == "random":
            if type(n_iter) != int:
                raise TypeError("The n_iter argument must be an integer.")
            if n_iter < 1:
                raise ValueError("The n_iter argument must be greater than zero.")
        if not isinstance(n_jobs, int):
            raise TypeError("The n_jobs argument must be an integer.")
        if (n_jobs <= 0) and (n_jobs != -1):
            raise ValueError("The n_jobs argument must be greater than zero or -1.")

        # Calculate predictions
        # (1) Create a dataframe to store the signals induced by each model.
        #     The index should be a multi-index with cross-sections equal to those in X
        #     and business-day dates spanning the range of dates in X.
        signal_xs_levels: List[str] = sorted(self.X.index.get_level_values(0).unique())
        original_date_levels: List[pd.Timestamp] = sorted(
            self.X.index.get_level_values(1).unique()
        )
        min_date: pd.Timestamp = min(original_date_levels)
        max_date: pd.Timestamp = max(original_date_levels)
        signal_date_levels: pd.DatetimeIndex = pd.bdate_range(
            start=min_date, end=max_date, freq="B"
        )

        sig_idxs = pd.MultiIndex.from_product(
            [signal_xs_levels, signal_date_levels], names=["cid", "real_date"]
        )

        signal_df: pd.MultiIndex = pd.DataFrame(
            index=sig_idxs, columns=[name], data=np.nan, dtype="float64"
        )
        prediction_data = []
        modelchoice_data = []

        outer_splitter = ExpandingIncrementPanelSplit(
            train_intervals=1,
            test_size=1,
            min_cids=min_cids,
            min_periods=min_periods,
            max_periods=max_periods,
        )

        X = self.X.copy()
        y = self.y.copy()

        train_test_splits = list(outer_splitter.split(X=X, y=y))

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._worker)(
                train_idx=train_idx,
                test_idx=test_idx,
                name=name,
                models=models,
                metric=metric,
                original_date_levels=original_date_levels,
                hparam_type=hparam_type,
                hparam_grid=hparam_grid,
                n_iter=n_iter,
            )
            for train_idx, test_idx in tqdm(
                train_test_splits,
                total=len(train_test_splits),
            )
        )

        prediction_data = []
        modelchoice_data = []

        for pred_data, model_data in results:
            prediction_data.append(pred_data)
            modelchoice_data.append(model_data)

        # Condense the collected data into a single dataframe
        for column_name, xs_levels, date_levels, predictions in prediction_data:
            idx = pd.MultiIndex.from_product([xs_levels, date_levels])
            try:
                signal_df.loc[idx, column_name] = predictions
            except Exception as e:
                warnings.warn(
                    f"Error in signal calculation for {column_name}, date {str(date_levels[0])}. Setting to zero.",
                    RuntimeWarning,
                )
                signal_df.loc[idx, column_name] = 0

        # Now convert signal_df into a quantamental dataframe
        # This will also ffill the last date of each cross-section as this will be an NA.
        signal_df = signal_df.groupby(level=0).ffill()

        # For each blacklisted period, set the signal to NaN
        if self.blacklist is not None:
            for cross_section, periods in self.blacklist.items():
                cross_section_key = cross_section.split("_")[0]
                # Set blacklisted periods to NaN
                if cross_section_key in signal_xs_levels:
                    signal_df.loc[
                        (cross_section_key, slice(periods[0], periods[1])), :
                    ] = np.nan

        signal_df_long: pd.DataFrame = pd.melt(
            frame=signal_df.reset_index(), id_vars=["cid", "real_date"], var_name="xcat"
        )
        self.preds = pd.concat((self.preds, signal_df_long), axis=0).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float64,
            }
        )

        model_df_long = pd.DataFrame(
            columns=self.chosen_models.columns, data=modelchoice_data
        )
        self.chosen_models = pd.concat(
            (
                self.chosen_models,
                model_df_long,
            ),
            axis=0,
        ).astype(
            {
                "real_date": "datetime64[ns]",
                "name": "object",
                "model_type": "object",
                "hparams": "object",
            }
        )

    def _worker(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        name: str,
        models: Dict[str, Union[BaseEstimator, Pipeline]],
        metric: Callable,
        original_date_levels: List[pd.Timestamp],
        hparam_grid: Dict[str, Dict[str, List]],
        n_iter: int = 10,
        hparam_type: str = "grid",
    ):
        """
        Private helper function to run the grid search for a single (train, test) pair
        and a collection of models. It is used to parallelise the pipeline.

        :param <np.ndarray> train_idx: Array of indices corresponding to the training set.
        :param <np.ndarray> test_idx: Array of indices corresponding to the test set.
        :param <str> name: Name of the prediction model.
        :param <Dict[str, Union[BaseEstimator,Pipeline]]> models: dictionary of sklearn
            predictors.
        :param <Callable> metric: Sklearn scorer object.
        :param <List[pd.Timestamp]> original_date_levels: List of dates corresponding to
            the original dataset.
        :param <str> hparam_type: Hyperparameter search type.
            This must be either "grid", "random" or "bayes". Default is "grid".
        :param <Dict[str, Dict[str, List]]> hparam_grid: Nested dictionary denoting the
            hyperparameters to consider for each model.
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            and https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
            for more details.
        :param <int> n_iter: Number of iterations to run for random search. Default is 10.
        :param <str> hparam_type: Hyperparameter search type.
            This must be either "grid", "random" or "bayes". Default is "grid".
        """
        # Set up training and test sets
        X_train_i: pd.DataFrame = self.X.iloc[train_idx]
        y_train_i: pd.Series = self.y.iloc[train_idx]

        X_test_i: pd.DataFrame = self.X.iloc[test_idx]
        # Get correct indices to match with
        test_xs_levels: List[str] = X_test_i.index.get_level_values(0).unique()
        test_date_levels: List[pd.Timestamp] = sorted(
            X_test_i.index.get_level_values(1).unique()
        )
        # Since the features lag behind the targets, the dates need to be adjusted
        # by a single frequency unit
        locs: np.ndarray = (
            np.searchsorted(original_date_levels, test_date_levels, side="left") - 1
        )
        test_date_levels: pd.DatetimeIndex = pd.DatetimeIndex(
            [original_date_levels[i] if i >= 0 else pd.NaT for i in locs]
        )
        optim_name = None
        optim_model = None
        optim_score = -np.inf
        # For each model, run a grid search over the hyperparameters to optimise
        # the provided metric. The best model is then used to make predictions.
        for model_name, model in models.items():
            if hparam_type == "grid":
                search_object = GridSearchCV(
                    estimator=model,
                    param_grid=hparam_grid[model_name],
                    scoring=metric,
                    refit=True,
                    cv=self.inner_splitter,
                    n_jobs=1,
                )
            elif hparam_type == "random":
                search_object = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=hparam_grid[model_name],
                    n_iter=n_iter,
                    scoring=metric,
                    refit=True,
                    cv=self.inner_splitter,
                    n_jobs=1,
                )
            # Run the grid search
            search_object.fit(X_train_i, y_train_i)
            score = search_object.best_score_
            if score > optim_score:
                optim_score = score
                optim_name = model_name
                optim_model = search_object.best_estimator_  # refit = True
                optim_params = search_object.best_params_

        # Store the best estimator predictions
        preds: np.ndarray = optim_model.predict(X_test_i)
        prediction_date = [name, test_xs_levels, test_date_levels, preds]
        # Store information about the chosen model at each time.
        modelchoice_data = [test_date_levels.date[0], name, optim_name, optim_params]

        return prediction_date, modelchoice_data

    def get_optimized_signals(
        self, name: Optional[Union[str, List]] = None
    ) -> pd.DataFrame:
        """
        Returns optimized signals for one or more processes

        :param <Optional[Union[str, List]]> name: Label of signal optimization process.
            Default is all stored in the class instance.

        :return <pd.DataFrame>: Pandas dataframe in JPMaQS format of working daily
            predictions based insequentially optimzed models.
        """
        if name is None:
            return self.preds
        else:
            if isinstance(name, str):
                name = [name]
            elif not isinstance(name, list):
                raise TypeError(
                    "The process name must be a string or a list of strings."
                )

            for n in name:
                if n not in self.preds.xcat.unique():
                    raise ValueError(
                        f"""The process name '{n}' is not in the list of already-run
                        pipelines. Please check the name carefully. If correct, please run 
                        calculate_predictions() first.
                        """
                    )
            return self.preds[self.preds.xcat.isin(name)]

    def get_optimal_models(
        self, name: Optional[Union[str, List]] = None
    ) -> pd.DataFrame:
        """
        Returns the sequences of optimal models for one or more processes

        :param <str> name: Label of signal optimization process. Default is all
            stored in the class instance.

        :return <pd.DataFrame>: Pandas dataframe of the optimal models or hyperparameters
            at the end of the base period in which they were determined (to be applied
            in the subsequent period).
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
        name: str,
        title: Optional[str] = None,
        cap: Optional[int] = 5,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (12, 8),
    ):
        """
        Visualized optimal models used for signal calculation.

        :param <str> name: Name of the prediction model.
        :param <Optional[str]> title: Title of the heatmap. Default is None. This creates
            a figure title of the form "Model Selection Heatmap for {name}".
        :param <Optional[int]> cap: Maximum number of models to display. Default
            (and limit) is 5. The chosen models are the 'cap' most frequently occurring
            in the pipeline.
        :param <Optional[Tuple[Union[int, float], Union[int, float]]]> figsize: Tuple of
            floats or ints denoting the figure size. Default is (12, 8).

        Note:
        This method displays the times at which each model in a learning process
        has been optimal and used for signal generation, as a binary heatmap.
        """
        # Checks
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
        if cap <= 0:
            raise ValueError("The cap must be greater than zero.")
        if cap > 20:
            warnings.warn(
                f"The maximum number of models to display is 20. The cap has been set to "
                "20.",
                RuntimeWarning,
            )
            cap = 20

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

        # Get the chosen models for the specified pipeline to visualise selection.
        chosen_models = self.get_optimal_models()
        chosen_models = chosen_models[chosen_models.name == name].sort_values(
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


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import make_scorer
    from macrosynergy.learning import (
        regression_balanced_accuracy,
    )

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {
        "GBP": (
            pd.Timestamp(year=2009, month=1, day=1),
            pd.Timestamp(year=2012, month=6, day=30),
        ),
        "CAD": (
            pd.Timestamp(year=2015, month=1, day=1),
            pd.Timestamp(year=2100, month=1, day=1),
        ),
    }

    train = msm.categories_df(
        df=dfd, xcats=xcats, cids=cids, val="value", blacklist=black, freq="M", lag=1
    ).dropna()

    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]

    y_long_train = pd.melt(
        frame=y_train.reset_index(), id_vars=["cid", "real_date"], var_name="xcat"
    )

    # (1) Example SignalOptimizer usage.
    #     We get adaptive signals for a linear regression and a KNN regressor, with the
    #     hyperparameters for the latter optimised across regression balanced accuracy.

    models = {
        "OLS": LinearRegression(),
        "KNN": KNeighborsRegressor(),
    }
    metric = make_scorer(regression_balanced_accuracy, greater_is_better=True)

    inner_splitter = RollingKFoldPanelSplit(n_splits=4)

    hparam_grid = {
        "OLS": {},
        "KNN": {"n_neighbors": [1, 2, 5]},
    }

    so = SignalOptimizer(
        inner_splitter=inner_splitter,
        X=X_train,
        y=y_train,
        blacklist=black,
    )

    so.calculate_predictions(
        name="test",
        models=models,
        metric=metric,
        hparam_grid=hparam_grid,
        hparam_type="grid",
    )

    print(so.get_optimized_signals("test"))

    # (2) Example SignalOptimizer usage.
    #     Visualise the model selection heatmap for the two most frequently selected
    #     models.
    so.models_heatmap(name="test", cap=5)

    # (3) Example SignalOptimizer usage.
    #     We get adaptive signals for two KNN regressors.
    #     All chosen models are visualised in a heatmap.
    models2 = {
        "KNN1": KNeighborsRegressor(),
        "KNN2": KNeighborsRegressor(),
    }
    hparam_grid2 = {
        "KNN1": {"n_neighbors": [5, 10]},
        "KNN2": {"n_neighbors": [1, 2]},
    }

    so.calculate_predictions(
        name="test2",
        models=models2,
        metric=metric,
        hparam_grid=hparam_grid2,
        hparam_type="grid",
    )

    so.models_heatmap(name="test2", cap=4)

    # (4) Example SignalOptimizer usage.
    #     Print the predictions and model choices for all pipelines.
    print(so.get_optimized_signals())
    print(so.get_optimal_models())