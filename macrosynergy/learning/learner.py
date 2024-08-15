from typing import List
import numpy as np
import pandas as pd

# from macrosynergy.learning.splitters.kfold_splitters import ExpandingKFoldPanelSplit
from macrosynergy.learning.splitters import (
    ExpandingIncrementPanelSplit,
    ExpandingKFoldPanelSplit, 
)
from macrosynergy.management import categories_df

from joblib import Parallel, delayed
from tqdm.auto import tqdm

import warnings


class BasePanelLearner:
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
    ):
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

        # Store necessary index information
        self.index = self.X.index
        self.date_levels = self.index.get_level_values(1)
        self.xs_levels = self.index.get_level_values(0)
        self.unique_date_levels = self.date_levels.unique()
        self.unique_xs_levels = self.xs_levels.unique()

    def run(
        self,
        name,
        outer_splitter,
        inner_splitters,
        models,
        hyperparameters,
        scorers,  # List or string of scorers, not metrics
        search_type="grid",  # If string, allow "grid", "random", "greedy" or "bayes". Else allow a general sklearn class to be passed here.
        n_iter=100,  # Number of iterations for random or bayes
        split_dictionary=None,  # Dictionary of {real_dates}
        use_variance_correction=False,
        n_jobs_outer=-1,
        n_jobs_inner=1,
        strategy_eval_periods=None,
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

        splits_dictionary : Optional[dict]
            Dictionary of changepoints for the number of splits in inner cross-validation
            splitters. Keys are real dates and values are integers. Default is None.

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
            name=name,
            outer_splitter=outer_splitter,
            inner_splitters=inner_splitters,
            models=models,
            hyperparameters=hyperparameters,
            scorers=scorers,
            search_type=search_type,
            n_iter=n_iter,
            split_dictionary=split_dictionary,
            use_variance_correction=use_variance_correction,
            n_jobs_outer=n_jobs_outer,
            n_jobs_inner=n_jobs_inner,
        )

        # Determine all outer splits and run the learning process in parallel
        train_test_splits = list(outer_splitter.split(self.X, self.y))

        # Return nested dictionary with results
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
                    use_variance_correction=use_variance_correction,
                    search_type=search_type,
                    n_iter=n_iter,
                    splits_dictionary=split_dictionary,
                    n_jobs_inner=n_jobs_inner,
                    strategy_eval_periods=strategy_eval_periods,
                )
                for idx, (train_idx, test_idx) in enumerate(train_test_splits)
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
        use_variance_correction,
        search_type,
        n_iter,
        splits_dictionary,
        n_jobs_inner,
    ):
        """
        Worker function for parallel processing of the learning process.
        """
        # Train-test split
        X_train, X_test = self.X.values[train_idx, :], self.X.values[test_idx, :]
        y_train, y_test = self.y.values[train_idx], self.y.values[test_idx]

        # Determine correct timestamps of test set forecasts
        # First get test index information
        test_index = self.index[test_idx]
        test_xs_levels = self.xs_levels[test_idx]
        test_date_levels = self.date_levels[test_idx]
        sorted_date_levels = sorted(test_date_levels.unique())

        # Since the features lag behind the targets, the dates need to be adjusted
        # by the lag applied.
        locs: np.ndarray = (
            np.searchsorted(self.date_levels, sorted_date_levels, side="left")
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

        # Run hyperparameter and model search
        optim_name = None
        optim_model = None
        optim_score = -np.inf

        optim_name, optim_model, optim_score, optim_params, n_splits = (
            self._model_search(
                X_train=X_train,
                y_train=y_train,
                inner_splitters=inner_splitters,
                models=models,
                hyperparameters=hyperparameters,
                scorers=scorers,
                search_type=search_type,
                n_iter=n_iter,
                splits_dictionary=splits_dictionary,
                use_variance_correction=use_variance_correction,
                n_jobs_inner=n_jobs_inner,
            )
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
                int(n_splits),
            ]
            other_data = None

        else:
            # Then a model was selected
            optim_model.fit(X_train, y_train)

            # If optim_model has a create_signal method, use it otherwise use predict
            if hasattr(optim_model, "create_signal"):
                if callable(getattr(optim_model, "create_signal")):
                    preds: np.ndarray = optim_model.create_signal(X_test)
                else:
                    preds: np.ndarray = optim_model.predict(X_test)
            else:
                preds: np.ndarray = optim_model.predict(X_test)

            prediction_data = [name, test_index, preds]

            # Store model choice information
            modelchoice_data = [
                self.date_levels[train_idx],
                name,
                optim_name,
                optim_params,
                int(n_splits),
            ]

            # Store other information - inherited classes can specify this method to store coefficients, intercepts etc if needed
            other_data: List[List] = self._extract_model_info(
                name,
                self.date_levels[train_idx],
                optim_model,
            )

        return (
            prediction_data,
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
        splits_dictionary,
        use_variance_correction,
        n_jobs_inner,
    ):
        """
        Returns optimal model, optimal hyperparameters, optimal validation score and
        number of splits used in non-adaptive splitters.
        """
        for model_name, model in models.items():
            model_scores = {}  # Dictionary {model names: score}
            for splitter_name, splitter in inner_splitters.items():
                splitter_scores = {}  # Dictionary {splitter names: score}
                for scorer_name, scorer in scorers.items():
                    hparam_scores: np.array = self._hyperparameter_search(
                        X_train=X_train,
                        y_train=y_train,
                        model=model,
                        hyperparameters=hyperparameters[model_name],
                        inner_splitter=splitter,
                        scorer=scorer,
                        search_type=search_type,
                        n_iter=n_iter,
                        splits_dictionary=splits_dictionary,
                        use_variance_correction=use_variance_correction,
                        n_jobs_inner=n_jobs_inner,
                    )

    def _check_init(self, *args, **kwargs):
        pass

    def _check_run(self, *args, **kwargs):
        pass


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
    panel_learner = BasePanelLearner(
        df=dfd,
        xcats=["CONTRACT_XR", "BENCH_XR"],
        cids=cids,
    )

    models = {
        "LR": LADRegressionSystem(min_xs_samples=21 * 3),
    }
    hparam_grid = {"LR": {"fit_intercept": [True, False], "positive": [True, False]}}

    scorer = neg_mean_abs_corr

    panel_learner.run(
        name="TEST",
        outer_splitter=ExpandingIncrementPanelSplit(),
        inner_splitters=[ExpandingKFoldPanelSplit(n_splits=5)],
        scorers=[scorer],
        models=models,
        hyperparameters=hparam_grid,
        use_variance_correction=False,
        n_jobs_outer=1,
        n_jobs_inner=1,
    )
