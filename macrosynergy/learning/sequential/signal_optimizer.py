"""
Class to determine and store sequentially-optimized panel forecasts based on statistical
machine learning. 
"""

import numbers

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectorMixin
from sklearn.pipeline import Pipeline

from macrosynergy.learning import ExpandingIncrementPanelSplit
from macrosynergy.learning.sequential import BasePanelLearner
from macrosynergy.management.utils import concat_categorical, _insert_as_categorical
from macrosynergy.management.types import QuantamentalDataFrame


class SignalOptimizer(BasePanelLearner):
    """
    Class for sequential optimization of return forecasts based on panels of quantamental
    features.

    Parameters
    ----------
    df : pd.DataFrame
        Daily quantamental dataframe in JPMaQS format containing a panel of features, as
        well as a panel of returns.
    xcats : list
        List comprising feature names, with the last element being the response variable
        name. The features and the response variable must be categories in the dataframe.
    cids : list, optional
        List of cross-section identifiers for consideration in the panel. Default is None,
        in which case all cross-sections in `df` are considered.
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
        Frequency of the analysis. Default is "M" for monthly.
    lag : int, optional
        Number of periods to lag the response variable. Default is 1.
    xcat_aggs : list, optional
        List of aggregation functions to apply to the features, used when `freq` is not
        `D`. Default is ["last", "sum"].
    generate_labels : callable, optional
        Function to transform the response variable into either alternative regression
        targets or classification labels. Default is None.

    Notes
    -----
    The `SignalOptimizer` class is used to predict the response variable, usually a panel
    of asset class returns, based on a panel of features that are lagged by a specified
    number of periods. This is done in a sequential manner, by specifying the size of an
    initial training set, choosing an optimal model out of a provided collection
    (with associated hyperparameters), forecasting the return panel, and then expanding
    the training set to include the now-realized returns. The process continues until the
    end of the dataset is reached.

    In addition to storing forecasts, this class also stores useful information for
    analysis such as the models selected at each point in time, the feature coefficients
    and intercepts (where relevant) of selected models, as well as the features
    selected by any feature selection modules.

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
        cids=None,
        start=None,
        end=None,
        blacklist=None,
        freq="M",
        lag=1,
        xcat_aggs=["last", "sum"],
        generate_labels=None,
    ):
        # Run checks and necessary dataframe massaging
        super().__init__(
            df=df,
            xcats=xcats,
            cids=cids,
            start=start,
            end=end,
            blacklist=blacklist,
            freq=freq,
            lag=lag,
            xcat_aggs=xcat_aggs,
            generate_labels=generate_labels,
        )

        # Create forecast dataframe index
        min_date = min(self.unique_date_levels)
        max_date = max(self.unique_date_levels)
        forecast_date_levels = pd.date_range(start=min_date, end=max_date, freq="B")
        self.forecast_idxs = pd.MultiIndex.from_product(
            [self.unique_xs_levels, forecast_date_levels], names=["cid", "real_date"]
        )

        # Create initial dataframes to store relevant quantities from the learning process
        self.preds = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"]).astype(
            {
                "real_date": "datetime64[ns]",
                "cid": "category",
                "xcat": "category",
                "value": "float32",
            }
        )
        self.feature_importances = pd.DataFrame(
            columns=["real_date", "name"] + list(self.X.columns)
        ).astype(
            {
                **{col: "float32" for col in self.X.columns},
                "real_date": "datetime64[ns]",
                "name": "category",
            }
        )
        self.intercepts = pd.DataFrame(
            columns=["real_date", "name", "intercepts"]
        ).astype(
            {
                "real_date": "datetime64[ns]",
                "name": "category",
                "intercepts": "float32",
            }
        )

        self.selected_ftrs = pd.DataFrame(
            columns=["real_date", "name"] + list(self.X.columns)
        ).astype(
            {
                **{col: "int" for col in self.X.columns},
                "real_date": "datetime64[ns]",
                "name": "category",
            }
        )

        self.store_correlations = False
        # Create data structure to store correlation matrix of features feeding into the
        # final model and the input features themselves
        self.ftr_corr = pd.DataFrame(
            columns=[
                "real_date",
                "name",
                "predictor_input",
                "pipeline_input",
                "pearson",
            ]
        ).astype(
            {
                "real_date": "datetime64[ns]",
                "name": "category",
                "predictor_input": "category",
                "pipeline_input": "category",
                "pearson": "float",
            }
        )

    def calculate_predictions(
        self,
        name,
        models,
        hyperparameters,
        scorers,
        inner_splitters,
        search_type="grid",
        normalize_fold_results=False,
        cv_summary="mean",
        min_cids=4,
        min_periods=12 * 3,
        test_size=1,
        max_periods=None,
        split_functions=None,
        n_iter=None,
        n_jobs_outer=-1,
        n_jobs_inner=1,
        store_correlations=False,
    ):
        """
        Determine forecasts and store relevant quantities over time.

        Parameters
        ----------
        name : str
            Name of the signal optimization process.
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
        search_type : str, optional
            Type of hyperparameter optimization to perform. Default is "grid". Options are
            "grid" and "prior".
        normalize_fold_results : bool, optional
            Whether to normalize the scores across folds before combining them. Default is
            False.
        cv_summary : str or callable, optional
            Summary function to use to combine scores across cross-validation folds.
            Default is "mean". Options are "mean", "median" or a callable function.
        min_cids : int, optional
            Minimum number of cross-sections required for the initial
            training set. Default is 4.
        min_periods : int, optional
            Minimum number of periods required for the initial training set, in units of
            the frequency `freq` specified in the constructor. Default is 36.
        test_size : int, optional
            Number of periods to pass before retraining a selected model. Default is 1.
        max_periods : int, optional
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
        store_correlations : bool
            Whether to store the correlations between input pipeline features and input
            predictor features. Default is False.
        """
        if not isinstance(store_correlations, bool):
            raise TypeError("The store_correlations argument must be a boolean.")

        if store_correlations and not all(
            [isinstance(model, Pipeline) for model in models.values()]
        ):
            raise ValueError(
                "The store_correlations argument is only valid when all models are Scikit-learn Pipelines."
            )
        self.store_correlations = store_correlations

        # Set up outer splitter
        outer_splitter = ExpandingIncrementPanelSplit(
            train_intervals=test_size,
            test_size=test_size,
            min_cids=min_cids,
            min_periods=min_periods,
            max_periods=max_periods,
        )

        results = self.run(
            name=name,
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

        self._check_duplicate_results(name)

        # Collect results from the worker
        # quantamental_data, model_data, other_data
        prediction_data = []
        model_choice_data = []
        ftr_coef_data = []
        intercept_data = []
        ftr_selection_data = []
        ftr_corr_data = []

        for split_result in results:
            prediction_data.append(split_result["predictions"])
            model_choice_data.append(split_result["model_choice"])
            ftr_coef_data.append(split_result["feature_importances"])
            intercept_data.append(split_result["intercepts"])
            ftr_selection_data.append(split_result["selected_ftrs"])
            ftr_corr_data.extend(split_result["ftr_corr"])

        # First create pandas dataframes to store the forecasts
        forecasts_df = pd.DataFrame(
            index=self.forecast_idxs, columns=[name], data=np.nan, dtype="float32"
        )
        # Create quantamental dataframe of forecasts
        for idx, forecasts in prediction_data:
            forecasts_df.loc[idx, name] = forecasts

        forecasts_df = forecasts_df.groupby(level=0).ffill().dropna()

        if self.blacklist is not None:
            for cross_section, periods in self.blacklist.items():
                cross_section_key = cross_section.split("_")[0]
                if cross_section_key in self.unique_xs_levels:
                    forecasts_df.loc[
                        (cross_section_key, slice(periods[0], periods[1])), :
                    ] = np.nan

        forecasts_df.columns = forecasts_df.columns.astype("category")
        forecasts_df_long = pd.melt(
            frame=forecasts_df.reset_index(),
            id_vars=["real_date", "cid"],
            var_name="xcat",
        )
        self.preds = concat_categorical(
            df1=self.preds,
            df2=forecasts_df_long,
        )

        # Store model selection data
        model_df_long = pd.DataFrame(
            columns=[col for col in self.chosen_models.columns if col != "name"],
            data=model_choice_data,
        ).astype({"model_type": "category"})
        model_df_long = _insert_as_categorical(model_df_long, "name", name, 1)

        self.chosen_models = concat_categorical(
            df1=self.chosen_models,
            df2=model_df_long,
        )

        # Store feature coefficients
        coef_df_long = pd.DataFrame(
            columns=[col for col in self.feature_importances.columns if col != "name"],
            data=ftr_coef_data,
        )
        coef_df_long = _insert_as_categorical(coef_df_long, "name", name, 1)
        self.feature_importances = concat_categorical(
            self.feature_importances,
            coef_df_long,
        )

        # Store intercept
        intercept_df_long = pd.DataFrame(
            columns=[col for col in self.intercepts.columns if col != "name"],
            data=intercept_data,
        )
        intercept_df_long = _insert_as_categorical(intercept_df_long, "name", name, 1)
        self.intercepts = concat_categorical(
            self.intercepts,
            intercept_df_long,
        )

        # Store selected features
        ftr_select_df_long = pd.DataFrame(
            columns=[col for col in self.selected_ftrs.columns if col != "name"],
            data=ftr_selection_data,
        )
        ftr_select_df_long = _insert_as_categorical(ftr_select_df_long, "name", name, 1)
        self.selected_ftrs = concat_categorical(
            self.selected_ftrs,
            ftr_select_df_long,
        )

        ftr_corr_df_long = pd.DataFrame(
            columns=self.ftr_corr.columns, data=ftr_corr_data
        )

        self.ftr_corr = concat_categorical(
            self.ftr_corr,
            ftr_corr_df_long,
        )

    def _check_duplicate_results(self, name):
        conditions = [
            ("preds", "xcat", name),
            ("feature_importances", "name", name),
            ("intercepts", "name", name),
            ("selected_ftrs", "name", name),
            ("ftr_corr", "name", name),
            ("chosen_models", "name", name),
        ]
        self._remove_results(conditions)

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
        optimal_model : RegressorMixin, ClassifierMixin or Pipeline
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
            Dictionary containing feature importance scores, intercepts, selected features
            and correlations between inputs to pipelines and those entered into a final
            model.
        """
        if optimal_model is not None:
            if hasattr(optimal_model, "create_signal"):
                if callable(getattr(optimal_model, "create_signal")):
                    preds = optimal_model.create_signal(X_test)
            else:
                preds = optimal_model.predict(X_test)
        else:
            preds = np.zeros(X_test.shape[0])

        prediction_data = [adjusted_test_index, preds]

        feature_names = np.array(X_train.columns)
        if isinstance(optimal_model, Pipeline):
            final_estimator = optimal_model[-1]
            for _, transformer in reversed(optimal_model.steps):
                if isinstance(transformer, SelectorMixin):
                    feature_names = transformer.get_feature_names_out()
                    break
        else:
            final_estimator = optimal_model

        coefs = np.full(X_train.shape[1], np.nan)

        if hasattr(final_estimator, "feature_importances_") or (
            hasattr(final_estimator, "coef_")
        ):
            if hasattr(final_estimator, "feature_importances_"):
                coef = final_estimator.feature_importances_
            elif hasattr(final_estimator, "coef_"):
                coef = final_estimator.coef_
            # Reshape coefficients for storage compatibility
            if coef.ndim == 1:
                coefs = coef
            elif coef.ndim == 2:
                if coef.shape[0] == 1:
                    coefs = coef.flatten()

        coef_ftr_map = {ftr: coef for ftr, coef in zip(feature_names, coefs)}
        coefs = [
            coef_ftr_map[ftr] if ftr in coef_ftr_map else np.nan
            for ftr in X_train.columns
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

        # Get feature selection information
        if len(feature_names) == X_train.shape[1]:
            # Then all features were selected
            ftr_selection_data = [timestamp] + [1 for _ in feature_names]
        else:
            # Then some features were excluded
            ftr_selection_data = [timestamp] + [
                1 if name in feature_names else 0 for name in np.array(X_train.columns)
            ]

        ftr_corr_data = self._get_ftr_corr_data(
            pipeline_name, optimal_model, X_train, timestamp
        )

        # Store data
        split_result = {
            "feature_importances": [timestamp] + coefs,
            "intercepts": [timestamp, intercepts],
            "selected_ftrs": ftr_selection_data,
            "predictions": prediction_data,
            "ftr_corr": ftr_corr_data,
        }

        return split_result

    def _get_ftr_corr_data(self, pipeline_name, optimal_model, X_train, timestamp):
        """
        Returns a list of correlations between the input features to a pipeline and the
        features inputted into the final model, at each retraining date.

        Parameters
        ----------
        pipeline_name : str
            Name of the signal optimization process.
        optimal_model : RegressorMixin, ClassifierMixin or Pipeline
            Optimal model selected at each retraining date.
        X_train : pd.DataFrame
            Input feature matrix.
        timestamp : pd.Timestamp
            Timestamp of the retraining date.

        Returns
        -------
        list
            List of correlations between the input features to a pipeline and the
            features inputted into the final model, at each retraining date.
        """
        if self.store_correlations and optimal_model is not None:
            # Transform the training data to the final feature space
            transformers = Pipeline(steps=optimal_model.steps[:-1])
            X_train_transformed = transformers.transform(X_train)
            n_features = X_train_transformed.shape[1]
            feature_names = (
                X_train_transformed.columns
                if isinstance(X_train_transformed, pd.DataFrame)
                else [f"Feature {i+1}" for i in range(n_features)]
            )
            # Calculate correlation between each original feature in X_train and
            # the transformed features in X_train_transformed
            if isinstance(X_train_transformed, pd.DataFrame):
                X_train_transformed = X_train_transformed.values

            ftr_corr_data = [
                [
                    timestamp,
                    pipeline_name,
                    final_feature_name,
                    input_feature_name,
                    np.corrcoef(
                        X_train_transformed[:, idx],
                        X_train[input_feature_name],
                    )[0, 1],
                ]
                for idx, final_feature_name in enumerate(feature_names)
                for input_feature_name in X_train.columns
            ]

        elif self.store_correlations and optimal_model is None:
            ftr_corr_data = [
                [
                    timestamp,
                    pipeline_name,
                    feature_name,
                    feature_name,
                    1,
                ]
                for feature_name in X_train.columns
            ]
        else:
            ftr_corr_data = []

        return ftr_corr_data

    def get_optimized_signals(self, name=None):
        """
        Returns optimized signals for one or more processes

        Parameters
        ----------
        name : str or list, optional
            Label(s) of signal optimization process(es). Default is all stored in the
            class instance.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe in JPMaQS format of working daily predictions.
        """
        if name is None:
            preds = self.preds
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
            preds = self.preds[self.preds.xcat.isin(name)]
        # return self.preds[self.preds.xcat.isin(name)]
        signals_df = QuantamentalDataFrame(
            df=preds,
            categorical=self.df.InitializedAsCategorical,
        ).to_original_dtypes()
        return signals_df

    def get_selected_features(self, name=None):
        """
        Returns the selected features over time for one or more processes.

        Parameters
        ----------
        name: str or list, optional
            Label(s) of signal optimization process(es). Default is all stored in the
            class instance.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of the selected features at each retraining date.
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

    def get_feature_importances(self, name=None):
        """
        Returns feature importances for a given pipeline.

        Parameters
        ----------
        name: str or list, optional
            Label(s) of signal optimization process(es). Default is all stored in the
            class instance.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of the feature importances, if available, learnt at each
            retraining date for a given pipeline.

        Notes
        -----
        Availability of feature importances is subject to the selected model having a
        `feature_importances_` or `coef_` attribute.
        """
        if name is None:
            return self.feature_importances
        else:
            if isinstance(name, str):
                name = [name]
            elif not isinstance(name, list):
                raise TypeError(
                    "The process name must be a string or a list of strings."
                )

            for n in name:
                if n not in self.feature_importances.name.unique():
                    raise ValueError(
                        f"""The process name '{n}' is not in the list of already-run
                        pipelines. Please check the name carefully. If correct, please run 
                        calculate_predictions() first.
                        """
                    )
            return self.feature_importances[
                self.feature_importances.name.isin(name)
            ].sort_values(by="real_date")

    def get_intercepts(self, name=None):
        """
        Returns intercepts for a given pipeline.

        Parameters
        ----------
        name: str or list, optional
            Label(s) of signal optimization process(es). Default is all stored in the
            class instance.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of the intercepts, if available, learnt at each retraining
            date for a given pipeline.
        """
        if name is None:
            return self.intercepts
        else:
            if isinstance(name, str):
                name = [name]
            elif not isinstance(name, list):
                raise TypeError(
                    "The process name must be a string or a list of strings."
                )

            for n in name:
                if n not in self.intercepts.name.unique():
                    raise ValueError(
                        f"""The process name '{n}' is not in the list of already-run
                        pipelines. Please check the name carefully. If correct, please run 
                        calculate_predictions() first.
                        """
                    )
            return self.intercepts[self.intercepts.name.isin(name)].sort_values(
                by="real_date"
            )

    def get_feature_correlations(
        self,
        name=None,
    ):
        """
        Returns dataframe of feature correlations for one or more processes

        Parameters
        ----------
        name: str or list, optional
            Label(s) of signal optimization process(es). Default is all stored in the
            class instance.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of the correlations between the features passed into a model
            pipeline and the post-processed features inputted into the final model.
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

    def feature_selection_heatmap(
        self,
        name,
        remove_blanks=True,
        title=None,
        cap=None,
        ftrs_renamed=None,
        figsize=(12, 8),
    ):
        """
        Visualise the features chosen by the final selector in a scikit-learn pipeline
        over time, for a given signal optimization process that has been run.

        Parameters
        ----------
        name : str
            Name of the previously run signal optimization process.
        remove_blanks : bool, optional
            Whether to remove features from the heatmap that were never selected. Default
            is True.
        title : str, optional
            Title of the heatmap. Default is None. This creates a figure title of the form
            "Model Selection Heatmap for {name}".
        cap : int, optional
            Maximum number of features to display. Default is None. The chosen features
            are the 'cap' most frequently occurring in the pipeline.
        ftrs_renamed : dict, optional
            Dictionary to rename the feature names for visualisation in the plot axis.
            Default is None, which uses the original feature names.
        figsize : tuple of floats or ints, optional
            Tuple of floats or ints denoting the figure size. Default is (12, 8).

        Notes
        -----
        This method displays the features selected by the final selector in a scikit-learn
        pipeline over time, for a given signal optimization process that has been run.
        This information is contained within a binary heatmap.
        This does not take into account inherent feature selection within the predictor.
        """
        # Checks
        self._checks_feature_selection_heatmap(
            name=name, title=title, ftrs_renamed=ftrs_renamed, figsize=figsize
        )

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
        title=None,
        ftrs_renamed=None,
        figsize=(12, 8),
    ):
        """
        Checks for the feature_selection_heatmap method.

        Parameters
        ----------
        name : str
            Name of the previously run signal optimization process.
        title : str, optional
            Title of the heatmap. Default is None. This creates a figure title of the form
            "Feature Selection Heatmap for {name}".
        ftrs_renamed : dict, optional
            Dictionary to rename the feature names for visualisation in the plot axis.
            Default is None, which uses the original feature names.
        figsize : tuple of floats or ints, optional
            Tuple of floats or ints denoting the figure size. Default is (12, 8).
        """
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

    def correlations_heatmap(
        self,
        name: str,
        feature_name: str,
        title: str = None,
        cap: int = None,
        ftrs_renamed: dict = None,
        figsize: tuple = (12, 8),
    ):
        """
        Method to visualise correlations between features entering a model, and those that
        entered a preprocessing pipeline.

        Parameters
        ----------
        name : str
            Name of the signal optimization process.
        feature_name : str
            Name of the feature passed into the final predictor.
        title : str, optional
            Title of the heatmap. Default is None. This creates a figure title of the form
            "Correlation Heatmap for feature {feature_name} and pipeline {name}".
        cap : int, optional
            Maximum number of correlations to display. Default is None. The chosen features
            are the 'cap' most highly correlated.
        ftrs_renamed : dict, optional
            Dictionary to rename the feature names for visualisation in the plot axis.
            Default is None, which uses the original feature names.
        figsize : tuple of floats or ints, optional
            Tuple of floats or ints denoting the figure size. Default is (12, 8).

        Notes
        -----
        This method displays the correlation between a feature that is about to be entered
        into a final predictor and the `cap` most correlated features entered into the
        original pipeline. This information is contained within a heatmap.

        In a given pipeline, the features that enter it can be transformed in any way.
        Sometimes the transformation is non-trivial, resulting in a feature space that is
        not easily interpretable. This method allows the user to see how the original
        features are correlated with the features that enter the final model, providing
        insight into the transformation process.

        As an example, dimensionality reduction techniques such as PCA and LDA rotate the
        feature space, resulting in factors that can be hard to interpret. A neural network
        aims to learn a non-linear transformation of the feature space, which can also be
        hard to interpret. This method allows the user to see how the original features are
        correlated with the transformed features, providing insight into the transformation
        that took place.
        """
        # Checks
        self._checks_correlations_heatmap(
            name=name,
            feature_name=feature_name,
            title=title,
            cap=cap,
            ftrs_renamed=ftrs_renamed,
            figsize=figsize,
        )

        # Get the correlations
        correlations = self.get_feature_correlations(name=name)
        correlations = correlations[correlations.predictor_input == feature_name]
        correlations = correlations.sort_values(by="real_date").drop(columns=["name"])
        correlations["real_date"] = correlations["real_date"].dt.date

        # Sort this dataframe based on the average correlation with each feature in
        # pipeline_input
        avg_corr = correlations.groupby("pipeline_input", observed=True)[
            "pearson"
        ].mean()
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
            correlations.pivot(
                index="pipeline_input", columns="real_date", values="pearson"
            ),
            cmap="coolwarm_r",
            cbar=True,
        )
        if title is None:
            title = (
                f"Correlation Heatmap for feature {feature_name} and pipeline {name}"
            )
        plt.title(title)
        plt.show()

    def _checks_correlations_heatmap(
        self,
        name: str,
        feature_name: str,
        title: str,
        cap: int,
        ftrs_renamed: dict,
        figsize: tuple,
    ):
        """
        Checks for the correlations_heatmap method.

        Parameters
        ----------
        name : str
            Name of the signal optimization process.
        feature_name : str
            Name of the feature passed into the final predictor.
        title : str
            Title of the heatmap. Default is None. This creates a figure title of the form
            "Correlation Heatmap for feature {feature_name} and pipeline {name}".
        cap : int
            Maximum number of correlations to display. Default is None. The chosen features
            are the 'cap' most highly correlated.
        ftrs_renamed : dict
            Dictionary to rename the feature names for visualisation in the plot axis.
            Default is None, which uses the original feature names.
        figsize : tuple of floats or ints
            Tuple of floats or ints denoting the figure size. Default is (12, 8).
        """
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
            if not isinstance(element, numbers.Number) or isinstance(element, bool):
                raise TypeError(
                    "The elements of the figsize tuple must be floats or ints."
                )

    def feature_importance_timeplot(
        self,
        name,
        ftrs=None,
        title=None,
        ftrs_renamed=None,
        figsize=(10, 6),
    ):
        """
        Visualise time series of feature importances for the final predictor in a
        given pipeline, when available.

        Parameters
        ----------
        name : str
            Name of the previously run signal optimization process.
        ftrs : list, optional
            List of feature names to plot. Default is None.
        title : str, optional
            Title of the plot. Default is None. This creates a figure title of the form
            "Feature importances for pipeline: {name}".
        ftrs_renamed : dict, optional
            Dictionary to rename the feature names for visualisation in the plot legend.
            Default is None, which uses the original feature names.
        figsize : tuple of floats or ints, optional
            Tuple of floats or ints denoting the figure size. Default is (10, 6).

        Notes
        -----
        This method displays the time series of feature importances for a given pipeline,
        when available. Availability depends on whether or not the final predictor in the
        pipeline has either a `coefs_` or `feature_importances_` attribute. This
        information is contained within a line plot. The default behaviour is to sort the
        feature importance columns in ascending order of the number of NAs, accounting for
        a possible feature selection module in the pipeline and plot the feature
        importances for the first 10 features in the sorted order. If more than 10
        features were involved in the learning procedure, the default is to plot the
        feature importances for the first 10 sorted features. By specifying a `ftrs` list
        (which can be no longer than 10 elements in length), this default behaviour can be
        overridden.

        By sorting by NAs, the plot displays the model feature importances for either the
        first 10 features in the dataframe or, when a feature selection module was present,
        the 10 most frequently selected features.
        """
        # Checks
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.feature_importances.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )
        ftrcoef_df = self.get_feature_importances(name)
        if ftrcoef_df.iloc[:, 2:].isna().all().all():
            raise ValueError(
                f"""There are no non-NA feature importances for the pipeline {name}.
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
        ftrcoef_df = self.get_feature_importances(name)
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
            plt.title(f"Feature importances for pipeline: {name}")

        plt.show()

    def intercepts_timeplot(self, name, title=None, figsize=(10, 6)):
        """
        Visualise time series of intercepts for a given pipeline, when available.

        Parameters
        ----------
        name : str
            Name of the previously run signal optimization process.
        title : str, optional
            Title of the plot. Default is None. This creates a figure title of the form
            "Intercepts for pipeline: {name}".
        figsize : tuple of floats or ints, optional
            Tuple of floats or ints denoting the figure size. Default is (10, 6).

        Notes
        -----
        This method displays the time series of intercepts for a given pipeline, when
        available. This information is contained within a line plot.
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
        name,
        ftrs=None,
        title=None,
        cap=None,
        ftrs_renamed=None,
        figsize=(10, 6),
    ):
        """
        Visualise feature coefficients for a given pipeline in a stacked bar plot.

        Parameters
        ----------
        name : str
            Name of the previously run signal optimization process.
        ftrs : list, optional
            List of feature names to plot. Default is None.
        title : str, optional
            Title of the plot. Default is None. This creates a figure title of the form
            "Stacked bar plot of model coefficients: {name}".
        cap : int, optional
            Maximum number of features to display. Default is None. The chosen features
            are the 'cap' most frequently occurring in the pipeline. This cannot exceed
            10.
        ftrs_renamed : dict, optional
            Dictionary to rename the feature names for visualisation in the plot legend.
            Default is None, which uses the original feature names.
        figsize : tuple of floats or ints, optional
            Tuple of floats or ints denoting the figure size. Default is (10, 6).

        Notes
        -----
        This method displays the average feature coefficients for a given pipeline in each
        calendar year, when available. This information is contained within a stacked bar
        plot. The default behaviour is to plot the first 10 features in the order specified
        during training. If more than 10 features were involved in the learning procedure,
        the default is to plot the first 10 features. By specifying a `ftrs` list (which
        can be no longer than 10 elements in length), this default behaviour can be
        overridden.
        """
        # Checks
        if not isinstance(name, str):
            raise TypeError("The pipeline name must be a string.")
        if name not in self.feature_importances.name.unique():
            raise ValueError(
                f"""The pipeline name {name} is not in the list of already-calculated 
                pipelines. Please check the pipeline name carefully. If correct, please 
                run calculate_predictions() first.
                """
            )
        ftrcoef_df = self.get_feature_importances(name)
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
        ftrcoef_df = self.get_feature_importances(name)
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

        avg_coefs = ftrcoef_df.groupby("year", observed=True).mean()
        pos_coefs = avg_coefs.clip(lower=0)
        neg_coefs = avg_coefs.clip(upper=0)

        ax = None
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

        Parameters
        ----------
        name : str
            Name of the previously run signal optimization process.
        title : str, optional
            Title of the plot. Default is None. This creates a figure title of the form
            "Stacked bar plot of model coefficients: {name}".
        figsize : tuple of floats or ints, optional
            Tuple of floats or ints denoting the figure size. Default is (10, 6).
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
        models_df = models_df.loc[:, "n_splits_used"]
        models_df_expanded = pd.DataFrame(models_df.tolist(), index=models_df.index)

        # Create time series plot
        # TODO: extend the number of splits line until the first date that the number of splits is incremented
        # This translates into vertical lines at each increment date as opposed to linear interpolation between them.
        fig, ax = plt.subplots()
        models_df_expanded.plot(ax=ax, figsize=figsize)
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"Number of CV splits for pipeline: {name}")

        plt.show()


if __name__ == "__main__":
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
    from macrosynergy.learning import (
        ExpandingKFoldPanelSplit,
        SignWeightedLinearRegression,
        TimeWeightedLinearRegression,
    )
    import scipy.stats as stats
    from macrosynergy.management.simulate import make_qdf
    from macrosynergy.learning.model_evaluation.scorers.scorers import neg_mean_abs_corr
    from macrosynergy.management.types import QuantamentalDataFrame

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2012-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2012-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2012-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2012-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {
        "GBP": (
            pd.Timestamp(year=2009, month=1, day=1),
            pd.Timestamp(year=2012, month=6, day=30),
        ),
        "CAD": (
            pd.Timestamp(year=2015, month=1, day=1),
            pd.Timestamp(year=2016, month=1, day=1),
        ),
    }

    so = SignalOptimizer(
        df=dfd,
        xcats=["CRY", "GROWTH", "INFL", "XR"],
        cids=cids,
        blacklist=black,
    )

    so.calculate_predictions(
        name="LR",
        models={
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "TWLS": TimeWeightedLinearRegression(),
        },
        hyperparameters={
            "Ridge": {
                "fit_intercept": [True, False],
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000],
            },
            "Lasso": {
                "fit_intercept": [True, False],
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000],
            },
            "TWLS": {
                "half_life": [24, 36, 60, 120, 240],
                "fit_intercept": [True, False],
            },
        },
        scorers={
            "r2": make_scorer(r2_score),
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        },
        inner_splitters={
            "ExpandingKFold": ExpandingKFoldPanelSplit(n_splits=5),
            "SecondSplit": ExpandingKFoldPanelSplit(n_splits=10),
        },
        search_type="prior",
        n_iter=6,
        cv_summary="mean-std",
        n_jobs_outer=-1,
        n_jobs_inner=1,
        normalize_fold_results=True,
        split_functions={
            "ExpandingKFold": lambda n: n // 12,
            "SecondSplit": None,
        },
    )

    so.models_heatmap("LR")
    so.feature_importance_timeplot("LR")
    so.coefs_stackedbarplot("LR")
    so.nsplits_timeplot("LR")

    # Test a random forest
    from sklearn.ensemble import RandomForestRegressor

    so.calculate_predictions(
        name="RF",
        models={
            "RF": RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                max_features="sqrt",
                max_samples=0.1,
            ),
        },
        hyperparameters={
            "RF": {},
        },
        scorers={
            "r2": make_scorer(r2_score),
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        },
        inner_splitters={
            "ExpandingKFold": ExpandingKFoldPanelSplit(n_splits=2),
            "SecondSplit": ExpandingKFoldPanelSplit(n_splits=3),
        },
        search_type="grid",
        cv_summary="mean-std",
        n_jobs_outer=-1,
        n_jobs_inner=1,
    )

    so.feature_importance_timeplot("RF")
