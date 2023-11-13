"""
Collection of functions to convert machine learning model predictions into custom PnLs.
There are two cases that we consider:
1) Simple cross-validation where hyper-parameters are chosen over an initial training set
   and then fixed for all hold-out sets (despite retraining).
2) Nested cross-validation allowing for adaptive hyper-parameter selection at each test
   time.
TODO: write adaptive_preds_to_pnl function
"""

import numpy as np
import pandas as pd

from macrosynergy.learning import PanelTimeSeriesSplit
import macrosynergy.pnl as msn

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from typing import List, Union, Dict, Optional, Callable
from tqdm import tqdm

from joblib import Parallel, delayed


class AdaptiveSignalHandler:
    def __init__(
        self,
        models: Dict[str, Union[BaseEstimator, Pipeline]],
        metrics: Dict[str, Callable],
        outer_splitter: PanelTimeSeriesSplit,
        inner_splitter: PanelTimeSeriesSplit,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Class for the calculation of quantamental predictions based on adaptive 
        hyperparameter and model selection. 

        :param <Dict[str, Union[BaseEstimator,Pipeline]]> models: dictionary of sklearn 
            predictors.
        :param <Dict[str, Callable]> metrics: dictionary of metrics to optimise each model 
            over.
        :param <PanelTimeSeriesSplit> outer_splitter: Panel walk-forward validation 
            splitter that determines the full training sets for the inner splitter.
        :param <PanelTimeSeriesSplit> inner_splitter: Panel splitter that is used to split 
            each training set generated by the outer splitter into  smaller 
            (training, test) pairs.
        :param <pd.DataFrame> X: Wide-format pandas dataframe of features over the time 
            period for which the signal is to be calculated. These should lag behind 
            returns by a single frequency unit.
            The frequency of features (and targets) determines the frequency at which
            model predictions are evaluated. This means that if we have monthly-frequency
            data the learning process uses the performance of monthly predictions.
        :param <pd.Series> y: Pandas series of targets corresponding to the features in X.

        Note: The ultimate objective is to return a dataframe of predictions of a
        machine learning model with adaptive hyperparameter selection at each test time and
        to analyse the subsequent signals.
        """

        self.models = models
        self.metrics = metrics
        self.outer_splitter = outer_splitter
        self.inner_splitter = inner_splitter
        self.X = X
        self.y = y

    def create_qdf(
        self,
        hparam_grid: Union[
            Dict[str, Dict[str, List]],
            Union[Dict[str, Dict[str, Callable]], Dict[str, Dict[str, List[Callable]]]],
        ],
        hparam_type: str = "grid",
        n_iter: int = 10,
    ):
        """
        Method to create a quantamental dataframe from an adaptive machine learning model.

        :param <Dict[str, Dict[str, List]]> hparam_grid: Nested dictionary denoting the
            hyperparameters to consider for each model. The outer dictionary needs keys
            representing the model name and should match with the model keys in models.
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
        :param <str> hparam_type: Hyperparameter search type. 
            This must be either "grid", "random" or "bayes". Default is "grid".
        :param <int> n_iter: Number of iterations to run for random search. Default is 10.
            Only used if hparam_type is "random".

        :return <pd.DataFrame>: Pandas dataframe of working daily signals generated by the
            machine learning model predictions
        """
        if hparam_grid.keys() != self.models.keys():
            raise ValueError(
                "Keys in the hyperparameter grid must match those in models dictionary."
            )
        if hparam_type not in ["grid", "random", "bayes"]:
            raise ValueError(
                "Invalid hyperparameter search type. Must be 'grid', 'random' or 'bayes'."
            )
        if hparam_type == "random":
            for param_dict in hparam_grid.values():
                # Allow for list of dictionaries
                for param in param_dict.values():
                    if isinstance(param, list):
                        for p in param:
                            if not hasattr(p, "rvs"):
                                raise ValueError(
                                    f"""Invalid random hyperparameter search dictionary 
                                    for parameter {param}. The value for the dictionary  
                                    Must be a scipy.stats distribution."""
                                )
                    else:
                        if not hasattr(param, "rvs"):
                            raise ValueError(
                                f"""Invalid random hyperparameter search dictionary for 
                                parameter {param}. The value for the dictionary  
                                Must be a scipy.stats distribution."""
                            )

        elif hparam_type == "bayes":
            raise NotImplementedError("Bayesian optimisation not yet implemented.")

        # (1) Create a dataframe to store the signals induced by each model.
        #     The index should be a multi-index with cross-sections equal to those in X and
        #     business-day dates spanning the range of dates in X.
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
        #    The columns are in the format <MODEL>_<METRIC>
        columns: List[str] = list(
            set(
                [
                    model + "_" + metric if hparam_grid[model] != {} else model
                    for model in self.models.keys()
                    for metric in self.metrics.keys()
                ]
            )
        )
        # TODO: possibly float32 or float16?
        signal_df: pd.MultiIndex = pd.DataFrame(
            index=sig_idxs, columns=columns, data=np.nan, dtype="float64"
        )
        collected_data = []

        for train_idx, test_idx in tqdm(outer_splitter.split(self.X, self.y)):
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
            # Run grid search over the associated training set
            for name, model in models.items():
                # Set up the grid search
                if hparam_grid[name] == {}:
                    # Then there are no hyperparameters to search over
                    # So just fit the model and predict
                    model.fit(X_train_i, y_train_i)
                    preds: np.ndarray = model.predict(X_test_i)
                    collected_data.append(
                        [name, test_xs_levels, test_date_levels, preds]
                    )
                else:
                    # Then there are hyperparameters to search over
                    # There must be some metrics specified
                    if len(metrics) == 0:
                        raise ValueError(
                            "No metrics specified for hyperparameter search."
                        )
                    # Run grid search
                    # TODO: parellise outer loop instead of the grid search since that will
                    #       contain more iterations
                    for metric in metrics.keys():
                        if hparam_type == "grid":
                            search_object = GridSearchCV(
                                estimator=model,
                                param_grid=hparam_grid[name],
                                scoring=metrics[metric],
                                refit=True,
                                cv=inner_splitter,
                                n_jobs=-1,
                            )
                        elif hparam_type == "random":
                            search_object = RandomizedSearchCV(
                                estimator=model,
                                param_distributions=hparam_grid[name],
                                n_iter=n_iter,
                                scoring=metrics[metric],
                                refit=True,
                                cv=inner_splitter,
                                n_jobs=-1,
                            )
                        # Fit the grid search
                        search_object.fit(X_train_i, y_train_i)
                        # Store the best estimator predictions
                        preds: np.ndarray = search_object.predict(X_test_i)
                        collected_data.append(
                            [
                                f"{name}_{metric}",
                                test_xs_levels,
                                test_date_levels,
                                preds,
                            ]
                        )

        # Condense the collected data into a single dataframe
        for column_name, xs_levels, date_levels, predictions in collected_data:
            idx = pd.MultiIndex.from_product([xs_levels, date_levels])
            signal_df.loc[idx, column_name] = predictions

        # Now convert signal_df into a quantamental dataframe
        signal_df = signal_df.groupby(level=0).ffill()
        signal_df_long: pd.DataFrame = pd.melt(
            frame=signal_df.reset_index(), id_vars=["cid", "real_date"], var_name="xcat"
        )

        return signal_df_long


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import make_scorer, mean_absolute_error
    from macrosynergy.learning import (
        regression_balanced_accuracy,
        MapSelectorTransformer,
    )

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example 1: Unbalanced panel """

    df_cids2 = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids2.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats2 = pd.DataFrame(index=xcats, columns=cols)
    df_xcats2.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats2.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats2.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    # dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)
    dfd2 = msm.categories_df(
        df=dfd2, xcats=xcats, cids=cids, val="value", blacklist=black, freq="M", lag=1
    ).dropna()
    X = dfd2.drop(columns=["XR"])
    y = dfd2["XR"]
    y_long = pd.melt(
        frame=y.reset_index(), id_vars=["cid", "real_date"], var_name="xcat"
    )

    # (1) Example usage for adaptive_preds_to_signal()

    models = {
        "ols": LinearRegression(),
        "knn": KNeighborsRegressor(),
    }
    metrics = {
        "bac": make_scorer(regression_balanced_accuracy, greater_is_better=True),
        "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
    }
    outer_splitter = PanelTimeSeriesSplit(
        train_intervals=1, test_size=1, min_cids=4, min_periods=12
    )
    inner_splitter = PanelTimeSeriesSplit(n_split_method="rolling", n_splits=4)

    ash = AdaptiveSignalHandler(
        models=models,
        metrics=metrics,
        outer_splitter=outer_splitter,
        inner_splitter=inner_splitter,
        X=X,
        y=y,
    )
    
    df_sig = ash.create_qdf(
        hparam_grid={"ols": {}, "knn": {"n_neighbors": [1, 2, 5]}}, hparam_type="grid"
    )

    print(df_sig)
