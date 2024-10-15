"""
Class to handle the calculation of quantamental predictions based on adaptive
hyperparameter and model selection.
"""

import numbers
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from macrosynergy.learning.panel_time_series_split import (
    BasePanelSplit,
    ExpandingIncrementPanelSplit,
    RollingKFoldPanelSplit,
    ExpandingKFoldPanelSplit,
)

from macrosynergy.learning.predictors import LADRegressionSystem
from macrosynergy.management.validation import _validate_Xy_learning
from macrosynergy.compat import JOBLIB_RETURN_AS


class SignalOptimizer:
    def __init__(
        self,
        inner_splitter: BasePanelSplit,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        blacklist: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
        initial_nsplits: Optional[Union[int, np.int_]] = None,
        threshold_ndates: Optional[Union[int, np.int_]] = None,
    ):
        """
        Class for sequential optimization of raw signals based on quantamental features.
        Optimization is performed through nested cross-validation, with the outer splitter
        an instance of `ExpandingIncrementPanelSplit` reflecting a pipeline through time
        simulating the experience of an investor. In each iteration of the outer splitter,
        a training and test set are created, and a hyperparameter search using the specified
        'inner_splitter' is performed to determine an optimal model amongst a set of
        candidate models. Once this is selected, the chosen model is used to make the test
        set forecasts. 
        
        The primary functionality of this class is to forecast the (cumulative) return at
        the NEXT rebalancing date and store the resulting predictions as trading signals 
        within a quantamental dataframe. This means that features in the dataframe, X, are
        expected to be lagged quantamental indicators, at a single native frequency unit,
        with the targets, in y, being the cumulative returns at the native frequency.
        The timestamps in the multi-indexes should refer to those of the unlagged
        returns/targets, as returned by `categories_df` within `macrosynergy.management`.
        Ultimately, these forecasts are cast back by a frequency period, accounting
        for the lagged features, and hence creating point-in-time signals. In other words, 
        a prediction $\\mathbb{E}[r_{t+1}|\\Gamma_{t}]$ is recorded at time $t$, where 
        $r_{t+1}$ refers to the cumulative return at time $t+1$ and $\\Gamma_{t}$ is the
        information set at time $t$.

        By providing a blacklisting dictionary, preferably through
        macrosynergy.management.make_blacklist, the user can specify time periods to
        ignore.

        :param <BasePanelSplit> inner_splitter: Panel splitter that is used to split
            each training set into smaller (training, test) pairs for cross-validation.
            At present that splitter has to be an instance of `RollingKFoldPanelSplit`,
            `ExpandingKFoldPanelSplit` or `ExpandingIncrementPanelSplit`.
        :param <pd.DataFrame> X: Wide pandas dataframe of features, multi-indexed by 
            cross-sections and timestamps. The multi-index must match that of the targets
            in 'y'. By default, it is expected that the features in `X` are
            lagged by one period, i.e. the values used for the period denoted by the
            timestamp must be those that were originally recorded for the previous period.
            This means the timestamps in the multi-indices refer to those of the true unlagged
            returns/targets provided in 'y'. The frequency of features (and targets) determines the
            frequency at which model predictions are made and evaluated.
            This means that if we have monthly data, the learning process concerns
            forecasting returns one month ahead. If 'lagged_features' is False,
            then the assumption of lagged features is dropped. 
        :param <Union[pd.DataFrame,pd.Series]> y: Pandas dataframe or series of targets,
            multi-indexed by cross-sections and timestamps. The multi-index must match 
            that of the features in `X`.
        :param <Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]> blacklist: cross-sections
            with date ranges that should be excluded from the dataframe.
        :param <Optional[Union[int, np.int_]]> initial_nsplits: The number of splits to be
            used in the initial training set for cross-validation. If not None, this parameter
            ovverrides the number of splits in the inner splitter. By setting this value,
            the ensuing signal optimization process uses a changing number of cross-validation
            splits over time. The parameter "threshold_ndates", which defines the dynamics
            of the adaptive number of splits, must be set in this case. Default is None.
        :param <Optional[Union[int, np.int_]]> threshold_ndates: Number of unique dates,
            in units of the native dataset frequency, to be made available for the currently-used
            number of cross-validation splits to increase by one. If not None, the "initial_nsplits"
            parameter must be set. Default is None.

        Note:
        Optimization is based on expanding/rolling time series panels and maximization of a defined
        criterion over a grid of sklearn pipelines and hyperparameters of the involved
        models. The frequency of the input datasets, `X` and `y`, determines the frequency
        at which the training set is expanded/rolled and at which forecasts are made.
        The training set itself is split into various (training, test) pairs by the
        `inner_splitter` argument for cross-validation. Based on inner cross-validation,
        an optimal model is chosen and used for predicting the targets of the next period,
        assuming that 'lagged_features' is True. A prediction for a particular cross-section
        and time period is made only if all required information has been available for that pair.
        Primarily, optimized signals that are produced by the class are always stored for the
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
        self._checks_init_params(
            inner_splitter, X, y, blacklist, initial_nsplits, threshold_ndates
        )

        # Set instance attributes
        self.X = X
        self.y = y
        self.inner_splitter = inner_splitter
        self.blacklist = blacklist
        self.initial_nsplits = initial_nsplits
        self.threshold_ndates = threshold_ndates

        if self.initial_nsplits:
            warnings.warn(
                "The initial_nsplits parameter has been set. Adjusting the number of splits "
                f"of the specified inner splitter to {self.initial_nsplits}",
                RuntimeWarning,
            )
            self.inner_splitter.n_splits = self.initial_nsplits

        # Create initial dataframes to store quantamental predictions and model choices
        self.preds = pd.DataFrame(columns=["cid", "real_date", "xcat", "value"])
        self.chosen_models = pd.DataFrame(
            columns=["real_date", "name", "model_type", "hparams", "n_splits_used"]
        )
        # Create initial dataframe to store model coefficients, if available
        self.ftr_coefficients = pd.DataFrame(
            columns=["real_date", "name"] + list(X.columns)
        )
        self.intercepts = pd.DataFrame(columns=["real_date", "name", "intercepts"])
        # Create initial dataframe to store selected features at each time, assuming
        # some feature selection stage is used in a sklearn pipeline
        self.selected_ftrs = pd.DataFrame(
            columns=["real_date", "name"] + list(X.columns)
        )

        # Create data structure to store correlation matrix of features feeding into the
        # final model and the input features themselves
        self.ftr_corr = pd.DataFrame(columns=["real_date", "name", "predictor_input", "pipeline_input", "pearson"])

    def _checks_init_params(
        self,
        inner_splitter: BasePanelSplit,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        blacklist: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
        initial_nsplits: Optional[Union[int, np.int_]],
        threshold_ndates: Optional[Union[int, np.int_]],
    ):
        """
        Private method to check the initialisation parameters of the class.
        """
        # Check X and y
        _validate_Xy_learning(X, y)

        # Check inner_splitter
        if not isinstance(inner_splitter, BasePanelSplit):
            raise TypeError(
                "The inner_splitter argument must be an instance of BasePanelSplit."
            )

        # Check blacklisting
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

        # Check initial_nsplits
        if initial_nsplits is not None:
            if not isinstance(initial_nsplits, (int, np.int_)):
                raise TypeError("The initial_nsplits argument must be an integer.")
            if threshold_ndates is None:
                raise ValueError(
                    "The threshold_ndates argument must be set if the initial_nsplits "
                    "argument is set."
                )
            if not hasattr(inner_splitter, "n_splits"):
                raise AttributeError(
                    "The inner_splitter object must have an attribute n_splits."
                )
        if threshold_ndates is not None:
            if not isinstance(threshold_ndates, (int, np.int_)):
                raise TypeError("The threshold_ndates argument must be an integer.")
            if initial_nsplits is None:
                raise ValueError(
                    "The initial_nsplits argument must be set if the threshold_ndates "
                    "argument is set."
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
        test_size: int = 1,
        max_periods: Optional[int] = None,
        n_iter: Optional[int] = 10,
        n_jobs: Optional[int] = -1,
        store_correlations: bool = False,
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
        :param <bool> store_correlations: Whether to store the correlations between input
            pipeline features and input predictor features. Default is False.

        Note:
        The method produces signals for financial contract positions. They are calculated
        sequentially at the frequency of the input data set. Sequentially here means
        that the training set is expanded/rolled by one base period of the frequency.
        Each time the training set itself is split into  various (training, test) pairs by
        the `inner_splitter` argument. Based on inner cross-validation an optimal model
        is chosen and used for predicting the targets of the next period.
        """
        # Checks
        self._checks_calcpred_params(
            name=name,
            models=models,
            metric=metric,
            hparam_grid=hparam_grid,
            hparam_type=hparam_type,
            min_cids=min_cids,
            min_periods=min_periods,
            test_size=test_size,
            max_periods=max_periods,
            n_iter=n_iter,
            n_jobs=n_jobs,
            store_correlations=store_correlations,
        )

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
            index=sig_idxs,
            columns=[name],
            data=np.nan,
            dtype="float64",  # TODO: why not float32
        )

        # (2) Set up the splitter, outputs to store the predictions and model choices

        outer_splitter = ExpandingIncrementPanelSplit(
            train_intervals=test_size,
            test_size=test_size,
            min_cids=min_cids,
            min_periods=min_periods,
            max_periods=max_periods,
        )

        X = self.X.copy()
        y = self.y.copy()

        # (3) Run the parallelised worker function to run the signal
        #     optimization algorithm over the trading history.
        train_test_splits = list(outer_splitter.split(X=X, y=y))

        results = tqdm(
            Parallel(n_jobs=n_jobs, **JOBLIB_RETURN_AS)(
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
                    nsplits_add=(
                        np.floor(idx * test_size / self.threshold_ndates)
                        if self.initial_nsplits
                        else None
                    ),
                    store_correlations=store_correlations,
                )
                for idx, (train_idx, test_idx) in enumerate(train_test_splits)
            ),
            total=len(train_test_splits),
        )

        # (4) Collect the results from the parallelised worker function and store them
        #     as dataframes that can be accessed by the user to analyse models chosen
        #     and signals produced over time. If there was trouble with the signal
        #     calculation, the user is warned and the signal is set to zero to indicate
        #     no action was taken.
        prediction_data = []
        modelchoice_data = []
        ftrcoeff_data = []
        intercept_data = []
        ftr_selection_data = []
        ftr_corr_data = []

        for pred_data, model_data, ftr_data, inter_data, ftrselect_data, ftrcorr_data in results:
            prediction_data.append(pred_data)
            modelchoice_data.append(model_data)
            ftrcoeff_data.append(ftr_data)
            intercept_data.append(inter_data)
            ftr_selection_data.append(ftrselect_data)
            ftr_corr_data.extend(ftrcorr_data)


        # Condense the collected data into a single dataframe and forward fill 
        for column_name, idx, predictions in prediction_data:
            signal_df.loc[idx, column_name] = predictions

        signal_df = signal_df.groupby(level=0).ffill()

        # (5) Handle blacklisted periods and ensure the signal dataframe is
        #     a quantamental dataframe.
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

        # (6) Finally, store the model choices made at each time in a dataframe.

        model_df_long = pd.DataFrame(
            columns=self.chosen_models.columns, data=modelchoice_data
        )
        coef_df_long = pd.DataFrame(
            columns=self.ftr_coefficients.columns, data=ftrcoeff_data
        )
        intercept_df_long = pd.DataFrame(
            columns=self.intercepts.columns, data=intercept_data
        )
        ftr_select_df_long = pd.DataFrame(
            columns=self.selected_ftrs.columns, data=ftr_selection_data
        )
        ftr_corr_df_long = pd.DataFrame(
            columns=self.ftr_corr.columns, data=ftr_corr_data
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
                "n_splits_used": "int",
            }
        )

        ftr_coef_types = {col: "float" for col in self.X.columns}
        ftr_coef_types["real_date"] = "datetime64[ns]"
        ftr_coef_types["name"] = "object"

        self.ftr_coefficients = pd.concat(
            (
                self.ftr_coefficients,
                coef_df_long,
            ),
            axis=0,
        ).astype(ftr_coef_types)

        self.intercepts = pd.concat(
            (
                self.intercepts,
                intercept_df_long,
            ),
            axis=0,
        ).astype(
            {
                "real_date": "datetime64[ns]",
                "name": "object",
                "intercepts": "float",
            }
        )

        ftr_selection_types = {col: "int" for col in self.X.columns}
        ftr_selection_types["real_date"] = "datetime64[ns]"
        ftr_selection_types["name"] = "object"

        self.selected_ftrs = pd.concat(
            (
                self.selected_ftrs,
                ftr_select_df_long,
            ),
            axis=0,
        ).astype(ftr_selection_types)

        self.ftr_corr = pd.concat(
            (
                self.ftr_corr,
                ftr_corr_df_long,
            ),
            axis=0,
        ).astype(
            {
                "real_date": "datetime64[ns]",
                "name": "object",
                "predictor_input": "object",
                "pipeline_input": "object",
                "pearson": "float",
            }
        )

    def _checks_calcpred_params(
        self,
        name: str,
        models: Dict[str, Union[BaseEstimator, Pipeline]],
        metric: Callable,
        hparam_grid: Dict[str, Dict[str, List]],
        hparam_type: str,
        min_cids: int,
        test_size: int,
        min_periods: int,
        max_periods: Optional[int],
        n_iter: Optional[int],
        n_jobs: Optional[int],
        store_correlations: bool,
    ):
        """
        Private method to check the calculate_predictions method parameters.
        """
        # Check the pipeline name is a string
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")

        # Check that the models dictionary is correctly formatted
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

        # Check that the metric is callable
        if not callable(metric):
            raise TypeError("The metric argument must be a callable object.")

        # Check the hyperparameter search type is a valid string
        if not isinstance(hparam_type, str):
            raise TypeError("The hparam_type argument must be a string.")
        if hparam_type not in ["grid", "random", "bayes"]:
            raise ValueError(
                "Invalid hyperparameter search type. Must be 'grid', 'random' or 'bayes'."
            )
        if hparam_type == "bayes":
            raise NotImplementedError("Bayesian optimisation not yet implemented.")

        # Check that the hyperparameter grid is correctly formatted
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

        # Check that the keys of the hyperparameter grid match those in the models dict
        if sorted(hparam_grid.keys()) != sorted(models.keys()):
            raise ValueError(
                "The keys in the hyperparameter grid must match those in the models "
                "dictionary."
            )

        # Check the min_cids, min_periods, test_size, max_periods, n_iter and n_jobs arguments
        # are correctly formatted
        if not isinstance(min_cids, int):
            raise TypeError("The min_cids argument must be an integer.")
        if min_cids < 1:
            raise ValueError("The min_cids argument must be greater than zero.")
        if not isinstance(min_periods, int):
            raise TypeError("The min_periods argument must be an integer.")
        if min_periods < 1:
            raise ValueError("The min_periods argument must be greater than zero.")
        if not isinstance(test_size, int):
            raise TypeError("The test_size argument must be an integer.")
        if test_size < 1:
            raise ValueError("The test_size argument must be greater than zero.")
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
        
        if not isinstance(store_correlations, bool):
            raise TypeError("The store_correlations argument must be a boolean.")

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
        nsplits_add: Optional[Union[int, np.int_]] = None,
        store_correlations: bool = False,
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
        :param <Optional[Union[int, np.int_]]> nsplits_add: Additional number of splits
            to add to the number of splits in the inner splitter. Default is None.
        :param <bool> store_correlations: Whether to store the correlations between input
            pipeline features and input predictor features. Default is False.
        """
        # Set up training and test sets
        X_train_i: pd.DataFrame = self.X.iloc[train_idx]
        y_train_i: pd.Series = self.y.iloc[train_idx]
        X_test_i: pd.DataFrame = self.X.iloc[test_idx]

        # Get correct indices to match with
        test_index = X_test_i.index
        test_xs_levels = test_index.get_level_values(0)
        test_date_levels = test_index.get_level_values(1)
        sorted_date_levels = sorted(test_date_levels.unique())

        # Since the features lag behind the targets, the dates need to be adjusted
        # by a single frequency unit
        locs: np.ndarray = (
            np.searchsorted(original_date_levels, sorted_date_levels, side="left") - 1
        )
        adj_test_date_levels: pd.DatetimeIndex = pd.DatetimeIndex(
            [original_date_levels[i] if i >= 0 else pd.NaT for i in locs]
        )

        # Adjust the test index based on adjusted dates
        date_map = dict(zip(test_date_levels, adj_test_date_levels))
        mapped_dates = test_date_levels.map(date_map)
        test_index = pd.MultiIndex.from_arrays(
            [test_xs_levels, mapped_dates], names=["cid", "real_date"]
        )
        test_date_levels = test_index.get_level_values(1)
        sorted_date_levels = sorted(test_date_levels.unique())

        optim_name = None
        optim_model = None
        optim_score = -np.inf

        # If nsplits_add is provided, add it to the number of splits
        if self.initial_nsplits:
            n_splits = self.initial_nsplits + nsplits_add
            self.inner_splitter.n_splits = int(n_splits)
        else:
            n_splits = self.inner_splitter.n_splits

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
            try:
                search_object.fit(X_train_i, y_train_i)
            except Exception as e:
                warnings.warn(
                    f"Error in the grid search for {model_name} at {test_date_levels[0]}: {e}.",
                    RuntimeWarning,
                )
            score = search_object.best_score_
            if score > optim_score:
                optim_score = score
                optim_name = model_name
                optim_model = search_object.best_estimator_
                optim_params = search_object.best_params_

        # Handle case where no model was chosen
        if optim_model is None:
            warnings.warn(
                f"No model was chosen for {name} at {test_date_levels[0]}. Setting to zero.",
                RuntimeWarning,
            )
            preds = np.zeros(X_test_i.shape[0])
            prediction_date = [name, test_index, preds]
            modelchoice_data = [
                test_date_levels.date[0],
                name,
                "None",
                {},
                int(n_splits),
            ]
            coefficients_data = [
                test_date_levels.date[0],
                name,
            ] + [np.nan for _ in range(X_train_i.shape[1])]
            intercept_data = [test_date_levels.date[0], name, np.nan]
            ftr_selection_data = [test_date_levels.date[0], name] + [
                1 for _ in range(X_train_i.shape[1])
            ]
            if store_correlations:
                ftr_corr_data = [
                    [
                        test_date_levels.date[0],
                        name,
                        feature_name,
                        feature_name,
                        1,
                    ]
                    for feature_name in X_train_i.columns
                ]
                return (
                    prediction_date,
                    modelchoice_data,
                    coefficients_data,
                    intercept_data,
                    ftr_selection_data,
                    ftr_corr_data, 
                )
            else:
                return (
                    prediction_date,
                    modelchoice_data,
                    coefficients_data,
                    intercept_data,
                    ftr_selection_data,
                    [],
                )
        # Store the best estimator predictions/signals
        # If optim_model has a create_signal method, use it otherwise use predict
        if hasattr(optim_model, "create_signal"):
            if callable(getattr(optim_model, "create_signal")):
                preds: np.ndarray = optim_model.create_signal(X_test_i)
            else:
                preds: np.ndarray = optim_model.predict(X_test_i)
        else:
            preds: np.ndarray = optim_model.predict(X_test_i)
            
        prediction_data = [name, test_index, preds]

        # See if the best model has coefficients and intercepts
        # First see if the best model is a pipeline object
        ftr_names = np.array(X_train_i.columns)
        if isinstance(optim_model, Pipeline):
            # Check whether a feature selector was used and get the output features if so
            final_estimator = optim_model[-1]
            for _, transformer in reversed(optim_model.steps):
                if isinstance(transformer, SelectorMixin):
                    ftr_names = transformer.get_feature_names_out()
                    break
        else:
            final_estimator = optim_model

        if hasattr(final_estimator, "coef_"):
            if len(final_estimator.coef_.shape) == 1:
                coefs = np.array(final_estimator.coef_)
            elif len(final_estimator.coef_.shape) == 2:
                if final_estimator.coef_.shape[0] != 1:
                    coefs = np.array([np.nan for _ in range(X_train_i.shape[1])])
                else:
                    coefs = np.array(final_estimator.coef_).squeeze()
            else:
                coefs = np.array([np.nan for _ in range(X_train_i.shape[1])])
        else:
            coefs = np.array([np.nan for _ in range(X_train_i.shape[1])])

        coef_ftr_map = {ftr: coef for ftr, coef in zip(ftr_names, coefs)}
        coefs = [
            coef_ftr_map[ftr] if ftr in coef_ftr_map else np.nan
            for ftr in X_train_i.columns
        ]
        if hasattr(final_estimator, "intercept_"):
            if isinstance(final_estimator.intercept_, np.ndarray):
                # Store the intercept if it has length one
                if len(final_estimator.intercept_) == 1:
                    intercepts = final_estimator.intercept_[0]
                else:
                    intercepts = np.nan
            else:
                # The intercept will be a float/integer
                intercepts = final_estimator.intercept_
        else:
            intercepts = np.nan

        # Store information about the chosen model at each time.
        if len(ftr_names) == X_train_i.shape[1]:
            # Then all features were selected
            ftr_selection_data = [test_date_levels.date[0], name] + [
                1 for _ in ftr_names
            ]
        else:
            # Then some features were excluded
            ftr_selection_data = [test_date_levels.date[0], name] + [
                1 if name in ftr_names else 0 for name in np.array(X_train_i.columns)
            ]

        # Store the correlation matrix of the features used in the final model
        if store_correlations:
            if not isinstance(optim_model, Pipeline):
                ftr_corr_data = [
                    [
                        test_date_levels.date[0],
                        name,
                        feature_name,
                        feature_name,
                        1,
                    ]
                    for feature_name in X_train_i.columns
                ] 
            else:
                # Transform the training data to the final feature space
                transformers = Pipeline(steps = optim_model.steps[:-1])
                X_train_transformed = transformers.transform(X_train_i)
                n_features = X_train_transformed.shape[1]
                feature_names = (
                    X_train_transformed.columns
                    if isinstance(X_train_transformed, pd.DataFrame)
                    else [f"Feature {i+1}" for i in range(n_features)]
                )
                # Calculate correlation between each original feature in X_train_i and 
                # the transformed features in X_train_transformed
                if isinstance(X_train_transformed, pd.DataFrame):
                    ftr_corr_data = [
                        [
                            test_date_levels.date[0],
                            name,
                            final_feature_name,
                            input_feature_name,
                            np.corrcoef(X_train_transformed.values[:, idx], X_train_i[input_feature_name])[0, 1]
                        ]
                        for idx, final_feature_name in enumerate(feature_names)
                        for input_feature_name in X_train_i.columns
                    ]
                else:
                    ftr_corr_data = [
                        [
                            test_date_levels.date[0],
                            name,
                            final_feature_name,
                            input_feature_name,
                            np.corrcoef(X_train_transformed[:, idx], X_train_i[input_feature_name])[0, 1]
                        ]
                        for idx, final_feature_name in enumerate(feature_names)
                        for input_feature_name in X_train_i.columns
                    ]

        # Store correlation 
        modelchoice_data = [
            test_date_levels.date[0],
            name,
            optim_name,
            optim_params,
            int(n_splits),
        ]
        coefficients_data = [
            test_date_levels.date[0],
            name,
        ] + coefs
        intercept_data = [test_date_levels.date[0], name, intercepts]

        if store_correlations:
            return (
                prediction_data,
                modelchoice_data,
                coefficients_data,
                intercept_data,
                ftr_selection_data,
                ftr_corr_data,
            )
        else:
            return (
                prediction_data,
                modelchoice_data,
                coefficients_data,
                intercept_data,
                ftr_selection_data,
                [],
            )

    def get_feature_correlations(
        self,
        name: Optional[Union[str, List]] = None,      
    ):
        """
        Returns dataframe of feature correlations for one or more processes

        :param <Optional[Union[str, List]]> name: Label of signal optimization process.
            Default is all stored in the class instance.
        
        :return <pd.DataFrame>: Pandas dataframe of the correlations between the
            features passed into a model pipeline and the post-processed features inputted
            into the final model.
        """
        if name is None:
            return self.ftr_corr
        else:
            if isinstance(name, str):
                name = [name]
            elif not isinstance(name, list):
                raise TypeError(
                    "The process name must be a string or a list of strings."
                )

            for n in name:
                if n not in self.ftr_corr.name.unique():
                    raise ValueError(
                        f"""Either the process name '{n}' is not in the list of already-run
                        pipelines, or no correlations were stored for this pipeline.
                        Please check the name carefully. If correct, please run 
                        calculate_predictions() first.
                        """
                    )
            return self.ftr_corr[self.ftr_corr.name.isin(name)]
        
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

    def get_selected_features(
        self, name: Optional[Union[str, List]] = None
    ) -> pd.DataFrame:
        """
        Returns the selected features over time for one or more processes

        :param <str> name: Label of signal optimization process. Default is all
            stored in the class instance.

        :return <pd.DataFrame>: Pandas dataframe of the selected features over time
            at the end of the base period in which they were determined (to be applied
            in the subsequent period).
        """
        if name is None:
            return self.selected_ftrs
        else:
            if isinstance(name, str):
                name = [name]
            elif not isinstance(name, list):
                raise TypeError(
                    "The process name must be a string or a list of strings."
                )

            for n in name:
                if n not in self.selected_ftrs.name.unique():
                    raise ValueError(
                        f"""The process name '{n}' is not in the list of already-run
                        pipelines. Please check the name carefully. If correct, please run 
                        calculate_predictions() first.
                        """
                    )
            return self.selected_ftrs[self.selected_ftrs.name.isin(name)]

    def feature_selection_heatmap(
        self,
        name: str,
        remove_blanks: bool = True,
        title: Optional[str] = None,
        cap: Optional[int] = None,
        ftrs_renamed: dict = None,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (12, 8),
    ):
        """
        Method to visualise the selected features in a scikit-learn pipeline.

        :param <str> name: Name of the prediction model.
        :param <bool> remove_blanks: Whether to remove features from the heatmap that were
            never selected. Default is True.
        :param <Optional[str]> title: Title of the heatmap. Default is None. This creates
            a figure title of the form "Model Selection Heatmap for {name}".
        :param <int> cap: Maximum number of features to display. Default is None. The chosen
            features are the 'cap' most frequently occurring in the pipeline.
        :param <Optional[dict]> ftrs_renamed: Dictionary to rename the feature names for
            visualisation in the plot axis. Default is None, which uses the original
            feature names.
        :param <Optional[Tuple[Union[int, float], Union[int, float]]]> figsize: Tuple of
            floats or ints denoting the figure size. Default is (12, 8).

        Note:
        This method displays the times at which each feature was used in
        the learning process and used for signal generation, as a binary heatmap.
        """
        # Checks
        self._checks_feature_selection_heatmap(name=name, title=title, ftrs_renamed = ftrs_renamed, figsize=figsize)

        # Get the selected features for the specified pipeline to visualise selection.
        selected_ftrs = self.get_selected_features(name=name)
        selected_ftrs["real_date"] = selected_ftrs["real_date"].dt.date
        selected_ftrs = (
            selected_ftrs.sort_values(by="real_date")
            .drop(columns=["name"])
            .set_index("real_date")
        )

        # Sort dataframe columns in descending order of the number of times they were selected
        ftr_count = selected_ftrs.sum().sort_values(ascending=False)
        if remove_blanks:
            ftr_count = ftr_count[ftr_count > 0]
        if cap is not None:
            ftr_count = ftr_count.head(cap)

        reindexed_columns = ftr_count.index
        selected_ftrs = selected_ftrs[reindexed_columns]
        if ftrs_renamed is not None:
            selected_ftrs.rename(columns=ftrs_renamed, inplace=True)
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        if np.all(selected_ftrs == 1):
            sns.heatmap(selected_ftrs.T, cmap="binary_r", cbar=False)
        else:
            sns.heatmap(selected_ftrs.T, cmap="binary", cbar=False)
        plt.title(title)
        plt.show()

    def _checks_feature_selection_heatmap(
        self,
        name: str,
        title: Optional[str] = None,
        ftrs_renamed: Optional[dict] = None,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (12, 8),
    ):
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.selected_ftrs.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )
        if title is None:
            title = f"Feature Selection Heatmap for {name}"
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
        if ftrs_renamed is not None:
            if not isinstance(ftrs_renamed, dict):
                raise TypeError("The ftrs_renamed argument must be a dictionary.")
            for key, value in ftrs_renamed.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "The keys of the ftrs_renamed dictionary must be strings."
                    )
                if not isinstance(value, str):
                    raise TypeError(
                        "The values of the ftrs_renamed dictionary must be strings."
                    )
                if key not in self.X.columns:
                    raise ValueError(
                        f"""The key {key} in the ftrs_renamed dictionary is not a feature 
                        in the pipeline {name}.
                        """
                    )

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
        self._checks_models_heatmap(name=name, title=title, cap=cap, figsize=figsize)

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

    def _checks_models_heatmap(
        self,
        name: str,
        title: Optional[str] = None,
        cap: Optional[int] = 5,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (12, 8),
    ):
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
            
    def correlations_heatmap(
        self,
        name: str,
        feature_name: str,
        title: Optional[str] = None,
        cap: Optional[int] = None,
        ftrs_renamed: Optional[dict] = None,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (12, 8),
    ):
        """
        Method to visualise correlations between features entering a model, and those that
        entered a preprocessing pipeline. 

        :param <str> name: Name of the prediction model.
        :param <str> feature_name: Name of the feature passed into the final predictor.
        :param <Optional[str]> title: Title of the heatmap. Default is None. This creates
            a figure title of the form "Correlation Heatmap for feature {feature_name}
            and pipeline {name}".
        :param <int> cap: Maximum number of correlations to display. Default is None.
            The chosen features are the 'cap' most highly correlated.
        :param <Optional[dict]> ftrs_renamed: Dictionary to rename the feature names for
            visualisation in the plot axis. Default is None, which uses the original
            feature names.
        :param <Optional[Tuple[Union[int, float], Union[int, float]]> figsize: Tuple of
            floats or ints denoting the figure size. Default is (12, 8).

        Note: 
        This method displays the correlation between a feature that is about to be entered
        into a final predictor and the `cap` most correlated features entered into the 
        original pipeline. 
        """
        # Checks
        self._checks_correlations_heatmap(
            name=name,
            feature_name=feature_name,
            title=title,
            cap=cap,
            ftrs_renamed=ftrs_renamed,
            figsize=figsize
        )

        # Get the correlations
        correlations = self.get_feature_correlations(name=name)
        correlations = correlations[correlations.predictor_input == feature_name]
        correlations = correlations.sort_values(by="real_date").drop(columns=["name"])
        correlations["real_date"] = correlations["real_date"].dt.date

        # Sort this dataframe based on the average correlation with each feature in
        # pipeline_input
        avg_corr = correlations.groupby("pipeline_input")["pearson"].mean()
        avg_corr = avg_corr.sort_values(ascending=False)
        if cap is not None:
            avg_corr = avg_corr.head(cap)
        
        reindexed_columns = avg_corr.index
        correlations = correlations[correlations.pipeline_input.isin(reindexed_columns)]
        if ftrs_renamed is not None:
            # rename items in correlations.pipeline_input based on ftrs_renamed
            # but leave items not in ftrs_renamed as they are
            correlations["pipeline_input"] = correlations["pipeline_input"].map(
                lambda x: ftrs_renamed.get(x, x)
            )
            
        # Create the heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            correlations.pivot(index="pipeline_input", columns="real_date", values="pearson"),
            cmap="coolwarm_r",
            cbar=True,
        )
        if title is None:
            title = f"Correlation Heatmap for feature {feature_name} and pipeline {name}"
        plt.title(title)
        plt.show()

    def _checks_correlations_heatmap(
        self,
        name: str,
        feature_name: str,
        title: Optional[str],
        cap: Optional[int],
        ftrs_renamed: Optional[dict],
        figsize: Tuple[Union[int, float], Union[int, float]]
    ):
        # name
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.ftr_corr.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of pipelines with calculated
                correlation matrices. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first, or make sure `store_correlations` is
                turned on. 
                """
            )
        # feature name
        if not isinstance(feature_name, str):
            raise TypeError("The feature name must be a string.")
        if feature_name not in self.ftr_corr.predictor_input.unique():
            raise ValueError(
                f"""The feature name {feature_name} is not in the list of features that
                were passed into the final predictor. Please check the feature name carefully.
                """
            )
        # title
        if title is not None:
            if not isinstance(title, str):
                raise TypeError("The title must be a string.")
        # cap
        if cap is not None:
            if not isinstance(cap, int):
                raise TypeError("The cap must be an integer.")
            if cap <= 0:
                raise ValueError("The cap must be greater than zero.")
        # ftrs_renamed
        if ftrs_renamed is not None:
            if not isinstance(ftrs_renamed, dict):
                raise TypeError("The ftrs_renamed argument must be a dictionary.")
            for key, value in ftrs_renamed.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "The keys of the ftrs_renamed dictionary must be strings."
                    )
                if not isinstance(value, str):
                    raise TypeError(
                        "The values of the ftrs_renamed dictionary must be strings."
                    )
                if key not in self.X.columns:
                    raise ValueError(
                        f"""The key {key} in the ftrs_renamed dictionary is not a feature 
                        in the pipeline {name}.
                        """
                    )
        # figsize
        if not isinstance(figsize, tuple):
            raise TypeError("The figsize argument must be a tuple.")
        if len(figsize) != 2:
            raise ValueError("The figsize argument must be a tuple of length 2.")
        for element in figsize:
            if not isinstance(element, numbers.Number) and not isinstance(element, bool):
                raise TypeError(
                    "The elements of the figsize tuple must be floats or ints."
                )

    def get_ftr_coefficients(self, name):
        """
        Method to return the feature coefficients for a given pipeline.

        :param <str> name: Name of the pipeline.

        :return <pd.DataFrame>: Pandas dataframe of the changing feature coefficients
            over time for the specified pipeline.
        """
        # Checks
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.ftr_coefficients.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )

        # Return the feature coefficients for the specified pipeline
        ftrcoef_df = self.ftr_coefficients
        return ftrcoef_df[ftrcoef_df.name == name].sort_values(by="real_date")

    def get_intercepts(self, name):
        """
        Method to return the intercepts for a given pipeline.

        :param <str> name: Name of the pipeline.

        :return <pd.DataFrame>: Pandas dataframe of the changing intercepts over time
            for the specified pipeline.
        """
        # Checks
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.intercepts.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )

        # Return the intercepts for the specified pipeline
        intercepts_df = self.intercepts
        return intercepts_df[intercepts_df.name == name].sort_values(by="real_date")

    def get_parameter_stats(self, name, include_intercept=False):
        """
        Function to return the means and standard deviations of linear model feature
        coefficients and intercepts (if available) for a given pipeline.

        :param <str> name: Name of the pipeline.
        :param <Optional[bool]> include_intercept: Whether to include the intercepts in
            the output. Default is False.

        :return Tuple of means and standard deviations of feature coefficients and
            intercepts (if chosen) for the specified pipeline.
        """
        # Checks
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.ftr_coefficients.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )
        if not isinstance(include_intercept, (bool, np.bool_)):
            raise TypeError("The include_intercept argument must be a boolean.")
        if include_intercept:
            if name not in self.intercepts.name.unique():
                raise ValueError(
                    f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
                )

        # Return the means and standard deviations of the feature coefficients and
        # intercepts (if chosen) for the specified pipeline

        ftrcoef_df = self.get_ftr_coefficients(name).iloc[:, 2:]
        if include_intercept:
            intercepts_df = self.get_intercepts(name).iloc[:, 2:]
            return (
                ftrcoef_df.mean(skipna=True),
                ftrcoef_df.std(skipna=True),
                intercepts_df.mean(skipna=True),
                intercepts_df.std(skipna=True),
            )

        return ftrcoef_df.mean(skipna=True), ftrcoef_df.std(skipna=True)

    def coefs_timeplot(
        self,
        name: str,
        ftrs: List[str] = None,
        title: str = None,
        ftrs_renamed: dict = None,
        figsize: Tuple[Union[int, float], Union[int, float]] = (10, 6),
    ):
        """
        Function to plot the time series of feature coefficients for a given pipeline.
        At most, 10 feature coefficient paths can be plotted at once. If more than 10
        features were involved in the learning procedure, the default is to plot the
        first 10 features in the order specified during training. By specifying a `ftrs`
        list (which can be no longer than 10 elements in length), this default behaviour
        can be overridden.

        :param <str> name: Name of the pipeline.
        :param <Optional[List]> ftrs: List of feature names to plot. Default is None.
        :param <Optional[str]> title: Title of the plot. Default is None. This creates
            a figure title of the form "Feature coefficients for pipeline: {name}".
        :param <Optional[dict]> ftrs_renamed: Dictionary to rename the feature names for
            visualisation in the plot legend. Default is None, which uses the original
            feature names.
        :param <Tuple[Union[float, int], Union[float,int]]> figsize: Tuple of floats or
            ints denoting the figure size.

        :return Time series plot of feature coefficients for the given pipeline.
        """
        # Checks
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.ftr_coefficients.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )
        ftrcoef_df = self.get_ftr_coefficients(name)
        if ftrcoef_df.iloc[:, 2:].isna().all().all():
            raise ValueError(
                f"""There are no non-NA coefficients for the pipeline {name}.
                Cannot display a time series plot.
                """
            )
        if ftrs is not None:
            if not isinstance(ftrs, list):
                raise TypeError("The ftrs argument must be a list.")
            if len(ftrs) > 10:
                raise ValueError(
                    "The ftrs list must be no longer than 10 elements in length."
                )
            for ftr in ftrs:
                if not isinstance(ftr, str):
                    raise TypeError("The elements of the ftrs list must be strings.")
                if ftr not in ftrcoef_df.columns:
                    raise ValueError(
                        f"""The feature {ftr} is not in the list of feature coefficients 
                        for the pipeline {name}.
                        """
                    )
        if not isinstance(title, str) and title is not None:
            raise TypeError("The title must be a string.")
        if ftrs_renamed is not None:
            if not isinstance(ftrs_renamed, dict):
                raise TypeError("The ftrs_renamed argument must be a dictionary.")
            for key, value in ftrs_renamed.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "The keys of the ftrs_renamed dictionary must be strings."
                    )
                if not isinstance(value, str):
                    raise TypeError(
                        "The values of the ftrs_renamed dictionary must be strings."
                    )
                if key not in self.X.columns:
                    raise ValueError(
                        f"""The key {key} in the ftrs_renamed dictionary is not a feature 
                        in the pipeline {name}.
                        """
                    )
        if not isinstance(figsize, tuple):
            raise TypeError("The figsize argument must be a tuple.")
        if len(figsize) != 2:
            raise ValueError("The figsize argument must be a tuple of length 2.")
        for element in figsize:
            if not isinstance(element, (int, float, np.int_, np.float_)):
                raise TypeError(
                    "The elements of the figsize tuple must be floats or ints."
                )

        # Set the style
        sns.set_style("darkgrid")

        # Reshape dataframe for plotting
        ftrcoef_df = self.get_ftr_coefficients(name)
        ftrcoef_df = ftrcoef_df.set_index("real_date")
        ftrcoef_df = ftrcoef_df.iloc[:, 1:]

        # Sort dataframe columns in ascending order of the number of Na values in the columns
        na_count = ftrcoef_df.isna().sum().sort_values()
        reindexed_columns = na_count.index
        ftrcoef_df = ftrcoef_df[reindexed_columns]
        
        if ftrs is not None:
            ftrcoef_df = ftrcoef_df[ftrs]
        else:
            if ftrcoef_df.shape[1] > 11:
                ftrcoef_df = pd.concat(
                    (ftrcoef_df.iloc[:, :10], ftrcoef_df.iloc[:, -1]), axis=1
                )

        # Create time series plot
        fig, ax = plt.subplots()
        if ftrs_renamed is not None:
            ftrcoef_df.rename(columns=ftrs_renamed).plot(ax=ax, figsize=figsize)
        else:
            ftrcoef_df.plot(ax=ax, figsize=figsize)

        if title is not None:
            plt.title(title)
        else:
            plt.title(f"Feature coefficients for pipeline: {name}")

        plt.show()

    def intercepts_timeplot(self, name, title=None, figsize=(10, 6)):
        """
        Function to plot the time series of intercepts for a given pipeline.

        :param <str> name: Name of the pipeline.
        :param <Optional[str]> title: Title of the plot. Default is None. This creates
            a figure title of the form "Intercepts for pipeline: {name}".
        :param <Tuple[Union[float, int], Union[float,int]]> figsize: Tuple of floats or
            ints denoting the figure size.

        :return: Time series plot of intercepts for the given pipeline.
        """
        # Checks
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.intercepts.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )
        intercepts_df = self.get_intercepts(name)

        # TODO: the next line will be made redundament once the signal optimiser checks for this
        # and removes any pipelines with all NaN intercepts
        if intercepts_df.iloc[:, 2:].isna().all().all():
            raise ValueError(
                f"""There are no non-NA intercepts for the pipeline {name}.
                Cannot display a time series plot.
                """
            )
        if not isinstance(title, str) and title is not None:
            raise TypeError("The title must be a string.")
        if not isinstance(figsize, tuple):
            raise TypeError("The figsize argument must be a tuple.")
        if len(figsize) != 2:
            raise ValueError("The figsize argument must be a tuple of length 2.")
        for element in figsize:
            if not isinstance(element, (int, float)):
                raise TypeError(
                    "The elements of the figsize tuple must be floats or ints."
                )

        # Set the style
        sns.set_style("darkgrid")

        # Reshape dataframe for plotting
        intercepts_df = intercepts_df.set_index("real_date")
        intercepts_df = intercepts_df.iloc[:, 1]

        # Create time series plot
        fig, ax = plt.subplots()
        intercepts_df.plot(ax=ax, figsize=figsize)
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"Intercepts for pipeline: {name}")

        plt.show()

    def coefs_stackedbarplot(
        self,
        name: str,
        ftrs: List[str] = None,
        title: str = None,
        cap: Optional[int] = None,
        ftrs_renamed: dict = None,
        figsize=(10, 6),
    ):
        """
        Function to create a stacked bar plot of feature coefficients for a given pipeline.
        At most, 10 feature coefficients can be considered in the plot. If more than 10
        features were involved in the learning procedure, the default is to plot the first
        10 features in the order specified during training. By specifying a `ftrs` list
        (which can be no longer than 10 elements in length), this default behaviour can be
        overridden.

        :param <str> name: Name of the pipeline.
        :param <Optional[List]> ftrs: List of feature names to plot. Default is None.
        :param <Optional[str]> title: Title of the plot. Default is None. This creates
            a figure title of the form "Stacked bar plot of model coefficients: {name}".
        :param <int> cap: Maximum number of features to display. Default is None. The chosen
            features are the 'cap' most frequently occurring in the pipeline. This cannot
            exceed 10.
        :param <Optional[dict]> ftrs_renamed: Dictionary to rename the feature names for
            visualisation in the plot legend. Default is None, which uses the original
            feature names.
        :param <Tuple[int, int]> figsize: Tuple of floats or ints denoting the figure size.

        :return: Stacked bar plot of feature coefficients for the given pipeline.
        """
        # Checks
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.ftr_coefficients.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )
        ftrcoef_df = self.get_ftr_coefficients(name)
        if ftrcoef_df.iloc[:, 2:].isna().all().all():
            raise ValueError(
                f"""There are no non-NA coefficients for the pipeline {name}.
                Cannot display a stacked bar plot.
                """
            )
        if ftrs is not None:
            if not isinstance(ftrs, list):
                raise TypeError("The ftrs argument must be a list.")
            if len(ftrs) > 10:
                raise ValueError(
                    "The ftrs list must be no longer than 10 elements in length."
                )
            for ftr in ftrs:
                if not isinstance(ftr, str):
                    raise TypeError("The elements of the ftrs list must be strings.")
                if ftr not in ftrcoef_df.columns:
                    raise ValueError(
                        f"""The feature {ftr} is not in the list of feature coefficients 
                        for the pipeline {name}.
                        """
                    )

        if not isinstance(title, str) and title is not None:
            raise TypeError("The title must be a string.")
        if ftrs_renamed is not None:
            if not isinstance(ftrs_renamed, dict):
                raise TypeError("The ftrs_renamed argument must be a dictionary.")
            for key, value in ftrs_renamed.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "The keys of the ftrs_renamed dictionary must be strings."
                    )
                if not isinstance(value, str):
                    raise TypeError(
                        "The values of the ftrs_renamed dictionary must be strings."
                    )
                if key not in self.X.columns:
                    raise ValueError(
                        f"""The key {key} in the ftrs_renamed dictionary is not a feature 
                        in the pipeline {name}.
                        """
                    )
        if not isinstance(figsize, tuple):
            raise TypeError("The figsize argument must be a tuple.")
        if len(figsize) != 2:
            raise ValueError("The figsize argument must be a tuple of length 2.")
        for element in figsize:
            if not isinstance(element, (int, float, np.int_, np.float_)):
                raise TypeError(
                    "The elements of the figsize tuple must be floats or ints."
                )
        if cap is not None:
            if not isinstance(cap, int):
                raise TypeError("The cap argument must be an integer.")
            if cap <= 0:
                raise ValueError("The cap argument must be greater than zero.")
            if cap > 10:
                raise ValueError("The cap argument must be no greater than 10.")

        # Set the style
        sns.set_style("darkgrid")

        # Reshape dataframe for plotting
        ftrcoef_df = self.get_ftr_coefficients(name)
        years = ftrcoef_df["real_date"].dt.year
        years.name = "year"
        ftrcoef_df.drop(columns=["real_date", "name"], inplace=True)

        # Sort dataframe columns in ascending order of the number of Na values in the columns
        na_count = ftrcoef_df.isna().sum().sort_values()
        reindexed_columns = na_count.index
        ftrcoef_df = ftrcoef_df[reindexed_columns]
        if cap is not None:
            ftrcoef_df = ftrcoef_df.T.head(cap).T
        ftrcoef_df = pd.concat((ftrcoef_df, years), axis=1)

        # Define colour map
        default_cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:10]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "default_cycle", default_cycle_colors
        )

        # Handle case where there are more than 10 features
        if ftrs is not None:
            ftrcoef_df = ftrcoef_df[ftrs + ["year"]]
        else:
            if ftrcoef_df.shape[1] > 11:
                ftrcoef_df = pd.concat(
                    (ftrcoef_df.iloc[:, :10], ftrcoef_df.iloc[:, -1]), axis=1
                )

        # Average the coefficients for each year and separate into positive and negative values
        if ftrs_renamed is not None:
            ftrcoef_df.rename(columns=ftrs_renamed, inplace=True)

        avg_coefs = ftrcoef_df.groupby("year").mean()
        pos_coefs = avg_coefs.clip(lower=0)
        neg_coefs = avg_coefs.clip(upper=0)

        # Create stacked bar plot
        if pos_coefs.sum().any():
            ax = pos_coefs.plot(
                kind="bar", stacked=True, figsize=figsize, colormap=cmap, alpha=0.75
            )
        if neg_coefs.sum().any():
            neg_coefs.plot(
                kind="bar",
                stacked=True,
                figsize=figsize,
                colormap=cmap,
                alpha=0.75,
                ax=ax,
            )

        # Display, title, axis labels
        if title is None:
            plt.title(f"Stacked bar plot of model coefficients: {name}")
        else:
            plt.title(title)

        plt.xlabel("Year")
        plt.ylabel("Average Coefficient Value")
        plt.axhline(0, color="black", linewidth=0.8)  # Adds a line at zero

        # Configure legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            title="Coefficients",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # Display plot
        plt.tight_layout()
        plt.show()

    def nsplits_timeplot(self, name, title=None, figsize=(10, 6)):
        """
        Method to plot the time series for the number of cross-validation splits used
        by the signal optimizer.

        :param <str> name: Name of the pipeline.
        :param <Optional[str]> title: Title of the plot. Default is None. This creates
            a figure title of the form "Number of CV splits for pipeline: {name}".
        :param <Tuple[Union[float, int], Union[float,int]]> figsize: Tuple of floats or
            ints denoting the figure size.

        :return: Time series plot of the number of cross-validation splits for the given
            pipeline.
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
        models_df = self.get_optimal_models(name)

        if not isinstance(title, str) and title is not None:
            raise TypeError("The title must be a string.")
        if not isinstance(figsize, tuple):
            raise TypeError("The figsize argument must be a tuple.")
        if len(figsize) != 2:
            raise ValueError("The figsize argument must be a tuple of length 2.")
        for element in figsize:
            if not isinstance(element, (int, float)):
                raise TypeError(
                    "The elements of the figsize tuple must be floats or ints."
                )

        # Set the style
        sns.set_style("darkgrid")

        # Reshape dataframe for plotting
        models_df = models_df.set_index("real_date").sort_index()
        models_df = models_df.iloc[:, -1]

        # Create time series plot
        # TODO: extend the number of splits line until the first date that the number of splits is incremented
        # This translates into vertical lines at each increment date as opposed to linear interpolation between them.
        fig, ax = plt.subplots()
        models_df.plot(ax=ax, figsize=figsize)
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"Number of CV splits for pipeline: {name}")

        plt.show()


if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import make_scorer
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.decomposition import PCA

    import macrosynergy.management as msm
    from macrosynergy.learning import (
        MapSelector,
        regression_balanced_accuracy,
        PanelStandardScaler,
    )
    from macrosynergy.management.simulate import make_qdf

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
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

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
    train = train[train.index.get_level_values(1) >= pd.Timestamp(year=2005,month=8,day=1)]

    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]

    y_long_train = pd.melt(
        frame=y_train.reset_index(), id_vars=["cid", "real_date"], var_name="xcat"
    )

    # (1) Example PCA usage.
    models = {
        "PLS": Pipeline(
            [
                ("scaler", PanelStandardScaler()),
                ("pca", PCA(n_components = 2)),
                ("model", LinearRegression(fit_intercept=True)),
            ]
        ),
    }
    metric = make_scorer(regression_balanced_accuracy, greater_is_better=True)
    inner_splitter = RollingKFoldPanelSplit(n_splits=4)
    grid = {
        "PLS": {},
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
        hparam_grid=grid,
        hparam_type="grid",
        test_size=3,
        n_jobs=1,
        store_correlations=True,
    )
    so.models_heatmap(name="test")
    so.correlations_heatmap("test", "Feature 1")
    # (1) Example SignalOptimizer usage.
    #     We get adaptive signals for a linear regression with feature selection.
    #     Hyperparameters: whether or not to fit an intercept, usage of positive restriction.
    #     We impose that retraining occurs once every quarter as opposed to every month.

    models = {
        "OLS": Pipeline(
            [
                ("selector", MapSelector(threshold=0.3)),
                ("model", LinearRegression(fit_intercept=True)),
            ]
        ),
    }
    metric = make_scorer(regression_balanced_accuracy, greater_is_better=True)
    inner_splitter = RollingKFoldPanelSplit(n_splits=4)
    grid = {
        "OLS": {"model__fit_intercept": [True, False], "model__positive": [True, False]},
    }
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

    so = SignalOptimizer(
        inner_splitter=inner_splitter,
        X=X_train,
        y=y_train,
        blacklist=black,
        initial_nsplits=5,
        threshold_ndates=24,
    )
    so.calculate_predictions(
        name="test",
        models=models,
        metric=metric,
        hparam_grid=grid,
        hparam_type="grid",
        test_size=3,
        n_jobs=1, # Set to 1 when debugging.
    )
    so.models_heatmap(name="test")
    so.coefs_stackedbarplot("test", ftrs_renamed={"CRY": "carry", "GROWTH": "growth"})
    so.coefs_timeplot("test")
    so.feature_selection_heatmap(
        "test", title="Feature selection heatmap for pipeline: test", ftrs_renamed={"CRY": "carry", "GROWTH": "growth"}
    )
    so.intercepts_timeplot("test")
    so.nsplits_timeplot("test")

    # (2) Example SignalOptimizer usage.
    #     We get adaptive signals for a linear regression.
    #     Hyperparameters: whether or not to fit an intercept, usage of positive restriction.

    models = {
        "OLS": Pipeline(
            [
                ("selector", MapSelector(threshold=0.2)),
                ("model", LADRegressionSystem(fit_intercept=True)),
            ]
        ),
    }

    metric = make_scorer(regression_balanced_accuracy, greater_is_better=True)

    inner_splitter = RollingKFoldPanelSplit(n_splits=4)

    hparam_grid = {
        "OLS": {"model__fit_intercept": [True, False], "model__positive": [True, False]},
    }

    so = SignalOptimizer(
        inner_splitter=inner_splitter,
        X=X_train,
        y=y_train,
        blacklist=black,
        initial_nsplits=5,
        threshold_ndates=24,
    )
    so.calculate_predictions(
        name="test",
        models=models,
        metric=metric,
        hparam_grid=grid,
        hparam_type="grid",
        n_jobs=1,
    )
    so.models_heatmap(name="test")
    so.coefs_stackedbarplot("test", ftrs_renamed={"CRY": "carry", "GROWTH": "growth"})
    so.coefs_timeplot("test")
    so.feature_selection_heatmap(
        "test", title="Feature selection heatmap for pipeline: test"
    )
    so.intercepts_timeplot("test")
    so.nsplits_timeplot("test")
