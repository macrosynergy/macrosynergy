"""
Class to handle the calculation of quantamental predictions based on adaptive
hyperparameter and model selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from typing import List, Union, Dict, Optional, Callable, Tuple
from tqdm import tqdm

import logging

from joblib import Parallel, delayed

from macrosynergy.learning.panel_time_series_split import BasePanelSplit, ExpandingIncrementPanelSplit, RollingKFoldPanelSplit


class AdaptiveSignalHandler:
    def __init__(
        self,
        inner_splitter: BasePanelSplit,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Class for the calculation of quantamental predictions based on adaptive
        hyperparameter and model selection. For a given collection of (training, test)
        pairs that expand/roll by a single frequency unit, an optimal model for each
        training set is chosen based on a specified hyperparameter search. The nature of
        the search is determined by splitting X and y according to the defined
        inner_splitter and the search type.
        The optimal model is then used to make test set forecasts.

        :param <BasePanelSplit> inner_splitter: Panel splitter that is used to split
            each training set into smaller (training, test) pairs.
        :param <pd.DataFrame> X: Wide-format pandas dataframe of features over the time
            period for which the signal is to be calculated. These should lag behind
            returns by a single frequency unit. This means that the date refers to the
            date that the returns are realised, with features lagging behind by a frequency unit.
            The frequency of features (and targets) determines the frequency at which
            model predictions are evaluated. This means that if we have monthly-frequency
            data, the learning process uses the performance of monthly predictions.
        :param <pd.Series> y: Pandas series of targets corresponding to the features in X.

        Note: The ultimate objective is to return a dataframe of predictions of a
        machine learning model with adaptive hyperparameter selection at each test time and
        to analyse the subsequent signals.
        """

        self.inner_splitter = inner_splitter
        self.X = X
        self.y = y

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
        hparam_type: str,
        min_cids: int = 4,
        min_periods: int = 12 * 3,
        max_periods: Optional[int] = None,
        n_iter: Optional[int] = 10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method to store & return a dataframe of quantamental predictions for financial returns.
        At each test time, the model that maximises the metric over the respective training
        set is chosen, using the inner splitter specified on class instantiation. The model
        type chosen at each time period is also stored in a separate dataframe.

        :param <str> name: Name of the prediction model.
        :param <Dict[str, Union[BaseEstimator,Pipeline]]> models: dictionary of sklearn
            predictors.
        :param <Callable> metric: Sklearn scorer object.
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
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            and https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
            for more details.
        :param <str> hparam_type: Hyperparameter search type.
            This must be either "grid", "random" or "bayes". Default is "grid".
        :param <int> min_cids: Minimum number of cross-sections required for the initial
            training set. Default is 4.
        :param <int> min_periods: minimum number of time periods required for the initial
            training set. Default is 12.
        :param <int> max_periods: maximum length of each training set.
            If the maximum is exceeded, the earliest periods are cut off.
            Default is None.
        :param <int> n_iter: Number of iterations to run for random search. Default is 10.

        :return <Tuple[pd.DataFrame, pd.DataFrame]>: Pandas dataframe of working daily signals generated by the
            machine learning model predictions, as well as the model choices at each time
            unit given by the native data frequency.
        """
        if hparam_grid.keys() != models.keys():
            raise ValueError(
                "The keys in the hyperparameter grid must match those in the models dictionary."
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
                                    f"Invalid random hyperparameter search dictionary for parameter {param}. The value for the dictionary  Must be a scipy.stats distribution."
                                )
                    else:
                        if not hasattr(param, "rvs"):
                            raise ValueError(
                                f"Invalid random hyperparameter search dictionary for parameter {param}. The value for the dictionary  Must be a scipy.stats distribution."
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
                        n_jobs=-1,
                    )
                elif hparam_type == "random":
                    search_object = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=hparam_grid[model_name],
                        n_iter=n_iter,
                        scoring=metric,
                        refit=True,
                        cv=inner_splitter,
                        n_jobs=-1,
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
            prediction_data.append(
                [
                    name,
                    test_xs_levels,
                    test_date_levels,
                    preds,
                ]
            )

            # Store information about the chosen model at each time.
            modelchoice_data.append(
                [test_date_levels.date[0], name, optim_name, optim_params]
            )

        # Condense the collected data into a single dataframe
        for column_name, xs_levels, date_levels, predictions in prediction_data:
            idx = pd.MultiIndex.from_product([xs_levels, date_levels])
            signal_df.loc[idx, column_name] = predictions

        # Now convert signal_df into a quantamental dataframe
        signal_df = signal_df.groupby(level=0).ffill()
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

        return (
            signal_df_long[signal_df_long.xcat == name],
            model_df_long[model_df_long.name == name],
        )

    def get_all_preds(self) -> pd.DataFrame:
        """
        Return the predictions dataframe for all calculated predictions in the current
        class instantiation.

        :return <pd.DataFrame>: Pandas dataframe of working daily predictions generated by
            the machine learning models at the native dataset frequency.
        """
        return self.preds

    def get_all_models(self) -> pd.DataFrame:
        """
        Return the dataframe comprising the selected models at each time for all
        calculated predictions in the current class instantiation.

        :return <pd.DataFrame>: Pandas dataframe of the chosen models at each time for all
            calculated predictions in the current class instantiation.
        """
        return self.chosen_models

    def models_heatmap(self, name: str, cap: int = 5, figsize: Tuple[int,int] = (12, 8)):
        """
        Method to visualise the times at which each model in an adaptive machine learning
        model pipeline is selected, as a binary heatmap. By default, the number of models
        to be displayed is capped at the 5 most frequently selected.

        :param <str> name: Name of the prediction model.
        :param <int> cap: Maximum number of models to display. Default (and limit) is 5.
            The chosen models are the 'cap' most frequently occurring in the pipeline.
        :param <tuple> figsize: Tuple of integers denoting the figure size. Default is
            (12, 8).
        """
        # Type and value checks
        if type(name) != str:
            raise TypeError("The pipeline name must be a string.")
        if type(cap) != int:
            raise TypeError("The cap must be an integer.")
        if name not in self.chosen_models.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated pipelines.
                Please check the pipeline name carefully. If correct, please run 
                calculate_predictions() first.
                """
            )
        if cap > 5:
            logging.warning(
                f"The maximum number of models to display is 5. The cap has been set to 5."
            )
            cap = 5
        
        # Get the chosen models for the specified pipeline to visualise selection.
        chosen_models = self.get_all_models()
        chosen_models = chosen_models[chosen_models.name == name].sort_values(
            by="real_date"
        )
        chosen_models["model_hparam_id"] = chosen_models.apply(
            lambda row: row["model_type"]
            if row["hparams"] == {}
            else f"{row['model_type']}_"
            + "_".join([f"{key}={value}" for key, value in row["hparams"].items()]),
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
        for _ , row in chosen_models.iterrows():
            model_id = row["model_hparam_id"]
            date = row["real_date"]
            binary_matrix.at[model_id, date] = 1

        # Display the heatmap. 
        plt.figure(figsize=figsize)
        sns.heatmap(binary_matrix, cmap="binary")
        plt.title(f"Model Selection Heatmap for {name}")
        plt.xlabel("Time")
        plt.ylabel("Models")
        plt.show()


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

    # (1) Example AdaptiveSignalHandler usage.
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

    ash = AdaptiveSignalHandler(
        inner_splitter=inner_splitter,
        X=X,
        y=y,
    )

    preds, models = ash.calculate_predictions(
        name="test",
        models=models,
        metric=metric,
        hparam_grid=hparam_grid,
        hparam_type="grid",
    )

    print(preds, models)

    # (2) Example AdaptiveSignalHandler usage.
    #     Visualise the model selection heatmap for the two most frequently selected models.
    ash.models_heatmap(name="test", cap=2)

    # (3) Example AdaptiveSignalHandler usage.
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

    preds2, models2 = ash.calculate_predictions(
        name="test2",
        models=models2,
        metric=metric,
        hparam_grid=hparam_grid2,
        hparam_type="grid",
    )

    print(preds2, models2)
    ash.models_heatmap(name="test2", cap=4)

    # (4) Example AdaptiveSignalHandler usage.
    #     Print the predictions and model choices for all pipelines.
    print(ash.get_all_preds())
    print(ash.get_all_models())
