import numpy as np
import pandas as pd

from macrosynergy.panel import categories_df

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from typing import List, Dict

import warnings

class BasePanelLearner:
    def __init__(self, df, xcats, cids = None, start = None, end = None, blacklist = None, freq = "M", lag = 1, xcat_aggs = ["last", "sum"]):
        """
        Base class for a sequential learning process over a panel. 

        BasePanelLearner contains the logic for a walk-forward pipeline over a panel of 
        paired features and targets. Furthermore, it comprises a range of methods for 
        model selection and hyperparameter tuning at each rebalancing date, in addition to
        methods for storing model selection choices, hyperparameter selection choices and
        cross-validation/validation scores. 

        Parameters
        ----------
        df : pd.DataFrame
            Standardized daily quantamental dataframe with the four columns: "cid", "xcat",
            "real_date" and "value".
        xcats : List[str]
            List of extended categories to be used in the learning process. The last 
            category in the list represents the dependent variable, and all preceding 
            categories will be the explanatory variable(s).
        cids : Optional[List[str]]
            Cross-sections to be included. Default is all in the dataframe. 
        start : Optional[str]
            Start date for considered data in subsequent analysis in ISO 8601 format.
            Default is None i.e. the earliest date in the dataframe. 
        end : Optional[str]
            End date for considered data in subsequent analysis in ISO 8601 format.
            Default is None i.e. the latest date in the dataframe.
        blacklist : Optional[Dict[str, Tuple[str, str]]]
            Blacklisting dictionary specifying date ranges for which cross-sectional 
            information should be excluded. The keys are cross-sections and the values 
            are tuples of start and end dates in ISO 8601 format. Default is None.
        freq : Optional[str]
            Frequency of the data. Default is "M" for monthly data.
        lag : Optional[int]
            Lag of the independent variable(s). Default is 1.
        xcat_aggs : Optional[List[str]]
            List of exactly two aggregation methods for downsampling data to the frequency
            specified in the freq parameter. The first parameter pertains to all
            independent variable downsampling, whilst the second corresponds with the 
            target category. Default is ["last", "sum"]. 

        Attributes
        ----------

        See Also
        --------

        Notes
        -----

        Examples
        --------
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

        # Create long format dataframe
        df_long = categories_df(
            df=self.df,
            xcats=self.xcats,
            cids=self.cids,
            start=self.start,
            end=self.end,
            blacklist=self.blacklist,
            freq=self.freq,
            lag=self.lag,
            xcat_aggs=self.xcat_aggs,
        ).dropna()

        # Create X and y
        self.X = df_long.iloc[:, :-1]
        self.y = df_long.iloc[:, -1]

        # Store necessary information for indexing
        self.index = self.X.index
        self.date_levels = self.X.index.get_level_values(1)
        self.sorted_date_levels = sorted(self.date_levels.unique())
        self.xs_levels = self.X.index.get_level_values(0)
        self.sorted_xs_levels = sorted(self.xs_levels.unique())

    def run(
        self,
        name,
        outer_splitter,
        inner_splitters,
        models,
        hyperparameters,
        scorers, # List or string of scorers, not metrics
        search_type = "grid", # If string, allow "grid", "random", "greedy" or "bayes". Else allow a general sklearn class to be passed here. 
        n_iter = 100, # Number of iterations for random or bayes
        splits_function = None, # Brainstorm this
        use_variance_correction = False, 
        n_jobs_outer = -1,
        n_jobs_inner = 1,
        strategy_eval_periods = None,
    ):
        """
        Run a learning process over the panel. 

        Parameters
        ----------
        name : str
            Category name for the forecasted panel resulting from the learning process. 

        outer_splitter : WalkForwardPanelSplit
            Outer splitter for the learning process. This should be an instance of
            WalkForwardPanelSplit. 

        inner_splitters : Union[BaseCrossValidator, List[BaseCrossValidator]]
            Inner splitters for the learning process. These should be instances of 
            BaseCrossValidator. 

        models : Dict[str, Union[BaseEstimator, List[BaseEstimator]]]
            Dictionary of named models to be used/selected between in the learning
            process. The keys are the names of the models, and the values are scikit-learn
            compatible models, possibly a Pipeline object. 

        hyperparameters : Dict[str, Union[Dict[str, List], Callable]]
            Dictionary of hyperparameters to be used in the learning process. The keys are
            the names of the models, and the values are either dictionaries of hyperparameters
            for grid search or greedy search, or callable objects for Random search and
            Bayesian search.

        scorers : Union[Callable, List[Callable]]
            Scorer(s) to be used for cross-validation. These should be functions that 
            accept an already-fitted estimator, an input matrix `X_test` and a target vector
            `y_test`, and return a scalar score. To convert a `scikit-learn` metric to a scorer,
            please use the `make_scorer` function in `sklearn.metrics`.

        search_type : Union[str, BaseSearchCV]
            Search type for hyperparameter tuning. If a string, allow "grid", "random",
            "greedy" or "bayes". Otherwise, a general `scikit-learn` compatible class
            inheriting from BaseSearchCV can be passed here. 

        n_iter : int
            Number of iterations for random or bayes search. Default is 100.

        splits_function : TODO

        use_variance_correction : Union[bool, List[bool]]
            Boolean indicating whether or not to take into account variance across splits
            in the hyperparameter and model selection process in cross-validation. 
            Default is False.

        n_jobs_outer : int
            Number of jobs to run in parallel for the outer loop in the nested 
            cross-validation. Default is -1 to use all available cores.

        n_jobs_inner : int
            Number of jobs to run in parallel for the inner loop in the nested 
            cross-validation. Default is 1.

        strategy_eval_periods : Optional[int]
            Number of periods to adjust each training set to create an out-of-sample
            hold-out set for value evaluation. If not None, then a thresholded z-score 
            signal is created for each hyperparameter and model choice, and a Sharpe ratio
            is computed and combined with a cross-validation score for model selection. 
        """
        # Checks 
        self._check_run(
            name = name,
            outer_splitter = outer_splitter,
            inner_splitters = inner_splitters,
            models = models,
            hyperparameters = hyperparameters,
            scorers = scorers,
            search_type = search_type,
            n_iter = n_iter,
            splits_function = splits_function,
            use_variance_correction = use_variance_correction,
            n_jobs_outer = n_jobs_outer,
            n_jobs_inner = n_jobs_inner,
        )

        # Determine all outer splits and run the learning process in parallel
        train_test_splits = list(outer_splitter.split(self.X, self.y))

        # Return nested dictionary with results
        optim_results = tqdm(
            Parallel(n_jobs=n_jobs_outer, return_as="generator")(
                delayed(self._worker)(
                    name = name,
                    train_idx = train_idx,
                    test_idx = test_idx,
                    inner_splitters = inner_splitters,
                    models = models,
                    hyperparameters = hyperparameters,
                    scorers = scorers,
                    use_variance_correction = use_variance_correction,
                    search_type = search_type,
                    n_iter = n_iter,
                    splits_function = splits_function,
                    n_jobs_inner = n_jobs_inner,
                    strategy_eval_periods = strategy_eval_periods,
                )
                for idx, (train_idx, test_idx) in enumerate(train_test_splits)
            ),
            total=len(train_test_splits),
        )

        return optim_results

    def _worker(
        self,
        name: str,
        train_idx,
        test_idx,
        inner_splitters: list,
        models: dict,
        hyperparameters: dict,
        scorers: list,
        use_variance_correction: list,
        search_type: str,
        n_iter: int,
        splits_function: callable,
        n_jobs_inner: int,
        strategy_eval_periods: int,
    ):
        """
        Selects an optimal model from a collection of candidates, with optimal hyperparameters.
        This model is fit on a training set, and evaluated on a test set. Predictions, scores
        and hyperparameter/model selection information are stored. 

        Parameters
        ----------
        train_idx : np.ndarray
            Training set indices
        test_idx : np.ndarray
            Test set indices
        inner_splitters : list
            List of inner splitters for the learning process.
        models : dict
            Dictionary of named models to be used/selected between in the learning
            process.
        hyperparameters : dict
            Dictionary of hyperparameters to be used in the learning process, corresponding
            to each model.
        scorers : list
            List of scorers to be used for cross-validation.
        use_variance_correction : list
            List of boolean values indicating whether or not to take into account variance
            across splits in the hyperparameter and model selection process in cross-validation.
            If only a single element is provided, it is broadcasted to all splitters.
        search_type : str
            Search type for hyperparameter tuning.
        n_iter : int
            Number of iterations for random or bayes search.
        splits_function : TODO
        n_jobs_inner : int
            Number of jobs to run in parallel for the inner loop in the nested cross-validation.
        strategy_eval_periods : int
            Number of periods to adjust each training set to create an out-of-sample hold-out
            set for value evaluation.

        Returns
        -------
        TODO
        """
        # Set up training and test sets
        X_train_i: np.ndarray = self.X.values[train_idx, :]
        y_train_i: np.ndarray = self.y.values[train_idx]
        X_test_i: np.ndarray = self.X.values[test_idx,:]

        # Get test set index information to correctly store results
        test_index = self.index[test_idx]
        test_xs_levels = self.xs_levels[test_idx]
        test_date_levels = self.date_levels[test_idx]
        sorted_test_date_levels = np.sort(np.unique(test_date_levels))

        # Features can lag behind the targets, so adjust the test set dates
        # so that the timestamp of predictions correspond to the date at which
        # the prediction is made possible (timestamp of the features)
        locs: np.ndarray = (
            np.searchsorted(self.date_levels, sorted_test_date_levels, side="left") - self.lag
        )
        adj_test_date_levels: pd.DatetimeIndex = pd.DatetimeIndex(
            [self.date_levels[i] if i >= 0 else pd.NaT for i in locs]
        )

        # Create new test set index
        date_map = dict(zip(test_date_levels, adj_test_date_levels))
        mapped_dates = test_date_levels.map(date_map)
        test_index = pd.MultiIndex.from_arrays(
            [test_xs_levels, mapped_dates], names=["cid", "real_date"]
        )
        test_date_levels = test_index.get_level_values(1)
        sorted_date_levels = sorted(test_date_levels.unique())

        # Perform hyperparameter and model selection process
        optim_name, optim_model, optim_params, optim_score = self._hyperparameter_search(
            X_train_i = X_train_i,
            y_train_i = y_train_i,
            inner_splitters = inner_splitters,
            models = models,
            hyperparameters = hyperparameters,
            scorers = scorers,
            use_variance_correction = use_variance_correction,
            search_type = search_type,
            n_iter = n_iter,
            splits_function = splits_function,
            n_jobs_inner = n_jobs_inner,
            strategy_eval_periods = strategy_eval_periods,
        )

        # Handle case where no model was selected
        if optim_name is None:
            warnings.warn(
                f"No model was chosen for {name} at rebalancing date {test_date_levels[0]}. Setting to zero.",
                RuntimeWarning,
            )
            preds = np.zeros(len(test_index))
            prediction_data = [name, test_index, preds]
            modelchoice_data = [
                test_date_levels.date[0],
                name,
                "None",
                {},
                int(n_splits),
            ]
            return prediction_data, modelchoice_data
        
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
        modelchoice_data = [
            test_date_levels.date[0],
            name,
            optim_name,
            optim_params,
            int(n_splits),
        ]

        return prediction_data, modelchoice_data