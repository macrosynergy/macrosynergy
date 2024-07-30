import numpy as np
import pandas as pd

from macrosynergy.panel import categories_df

from joblib import Parallel, delayed
from tqdm.auto import tqdm

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

    def run(
        self,
        name,
        outer_splitter,
        inner_splitters, # List of splitters for hyperparameter tuning
        models,
        hyperparameters,
        scorers, # List or string of scorers, not metrics
        search_type = "grid", # If string, allow "grid", "random", "greedy" or "bayes". Else allow a general sklearn class to be passed here. 
        n_iter = 100, # Number of iterations for random or bayes
        splits_function = None, # Brainstorm this
        use_variance_correction = False, 
        n_jobs_outer = -1,
        n_jobs_inner = 1,
    ):
        """
        Run a learning process over the panel. 

        Parameters
        ----------
        name : str
            Category name of the learning process. 

        outer_splitter : Union[ExpandingIncrementPanelSplit, ExpandingFrequencyPanelSplit]
            Outer splitter for the learning process. This should be an instance of
            either ExpandingIncrementPanelSplit or ExpandingFrequencyPanelSplit.

        inner_splitters : Union[BasePanelSplit, List[BasePanelSplit]]
            Inner splitters for the learning process. These should be instances of 
            BasePanelSplit.

        models : Dict[str, Union[BaseEstimator, List[BaseEstimator]]]
            Dictionary of named models to be used/selected between in the learning
            process. The keys are the names of the models, and the values are scikit-learn
            compatible models, possibly a PipeLine object. 

        hyperparameters : Dict[str, Union[Dict[str, List], Callable]]
            Dictionary of hyperparameters to be used in the learning process. The keys are
            the names of the models, and the values are either dictionaries of hyperparameters
            for grid search or greedy search, or callable objects for Random search and
            Bayesian search.

        scorers : Union[Callable, List[Callable]]
            Scorer(s) to be used in the learning process. These should be functions that 
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

        use_variance_correction : bool
            Boolean indicating whether or not to take into account variance across splits
            in the hyperparameter and model selection process in cross-validation. 
            Default is False.

        n_jobs_outer : int
            Number of jobs to run in parallel for the outer loop in the nested 
            cross-validation. Default is -1 to use all available cores.

        n_jobs_inner : int
            Number of jobs to run in parallel for the inner loop in the nested 
            cross-validation. Default is 1.
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
                )
                for idx, (train_idx, test_idx) in enumerate(train_test_splits)
            ),
            total=len(train_test_splits),
        )

        return optim_results



    