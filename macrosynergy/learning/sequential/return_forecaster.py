"""
Class to produce point forecasts of returns given knowledge of an indicator state on a
specific date.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectorMixin
from sklearn.pipeline import Pipeline

from macrosynergy.learning.sequential import BasePanelLearner
from macrosynergy.management.utils import (
    concat_categorical,
    _insert_as_categorical,
    reduce_df,
)
from macrosynergy.management.types import QuantamentalDataFrame


class ReturnForecaster(BasePanelLearner):
    """
    Class to produce return forecasts for a single forward frequency, based on the
    indicator states at a specific date.

    Parameters
    ----------
    df : pd.DataFrame
        Daily quantamental dataframe in JPMaQS format containing a panel of features, as
        well as a panel of returns.
    xcats : list
        List comprising feature names, with the last element being the response variable
        name. The features and the response variable must be categories in the dataframe.
    real_date : str
        Date in ISO 8601 format at which time a forward forecast is made based on the
        information states on that day.
    cids : list, optional
        List of cross-section identifiers for consideration in the panel. Default is None,
        in which case all cross-sections in `df` are considered.
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
    This class is a simple interface to produce a single period forward
    forecast. The `real_date` parameter specifies the date of the information state used
    to generate the forecast. As an example, if the provided date is "2025-03-01", a
    monthly frequency is specified and the lag is 1, the information
    states on this date are set aside, and the previous data is downsampled to monthly
    (with the features lagged by 1 period).  On this dataset, model selection and fitting
    happen - and the forecast is produced for the single out-of-sample period (March 2025).
    """

    def __init__(
        self,
        df,
        xcats,
        real_date,
        cids=None,
        blacklist=None,
        freq="M",
        lag=1,
        xcat_aggs=["last", "sum"],
        generate_labels=None,
    ):
        # Parent checks
        self._check_init(
            df=df,
            xcats=xcats,
            cids=cids,
            start=None,
            end=None,
            blacklist=blacklist,
            freq=freq,
            lag=lag,
            xcat_aggs=xcat_aggs,
            generate_labels=generate_labels,
        )
        # Additional checks to those carried in the parent class
        if not isinstance(real_date, str):
            raise TypeError("The real_date argument must be a string.")
        try:
            pd.to_datetime(real_date)
        except ValueError:
            raise ValueError("'real_date' must be in ISO 8601 format.")
        
        self.real_date = pd.to_datetime(real_date)
        df_adj = reduce_df(df=df, xcats=xcats)
        self._check_factor_availability(df_adj, xcats, self.real_date)

        if not isinstance(lag, int):
            raise TypeError("The lag argument must be an integer.")
        if lag < 1:
            raise ValueError("The lag argument must be at least 1.")

        # Separate in-sample and out-of-sample data
        # NOTE: We include real_date in the training set due to the required
        # lag on the macro factors. In addition, the last available return will always
        # be that from the previous day. This means that no leakage is introduced.
        # This was included for the sake of a unit test.  
        df_train = df[df.real_date <= self.real_date]
        df_test = df[df.real_date == self.real_date]

        # Set up supervised learning training set
        super().__init__(
            df=df_train,
            xcats=xcats,
            cids=cids,
            start=None,
            end=None,
            blacklist=blacklist,
            freq=freq,
            lag=lag,
            xcat_aggs=xcat_aggs,
            generate_labels=generate_labels,
            skip_checks=True # So that the checks aren't run twice
        )

        # Set up out-of-sample dataset for forecasting
        self.X_test = (
            reduce_df(df=df_test, blacklist=blacklist)
            .pivot(index=["cid", "real_date"], columns="xcat", values="value")[
                self.X.columns
            ]
            .dropna()
        )
        self.unique_test_levels = self.X_test.index.get_level_values(0).unique()

        # Set up data structures for analytics
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
        n_iter=None,
        n_jobs_cv=1,
        n_jobs_model=1,
        store_correlations=False,
    ):
        """
        Calculate predictions for the out-of-sample period.

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
        n_iter : int, optional
            Number of iterations to run in random hyperparameter search. Default is None.
        n_jobs_cv : int, optional
            Number of parallel jobs to run the cross-validation process. Default is 1.
        n_jobs_model : int, optional
            Number of parallel jobs to run the model fitting process (if relevant).
            Default is 1.
        store_correlations : bool
            Whether to store the correlations between input pipeline features and input
            predictor features. Default is False.
        """
        # Checks
        self._check_run(
            name=name,
            outer_splitter=None,
            inner_splitters=inner_splitters,
            models=models,
            hyperparameters=hyperparameters,
            scorers=scorers,
            search_type=search_type,
            normalize_fold_results=normalize_fold_results,
            cv_summary=cv_summary,
            n_iter=n_iter,
            split_functions=None,
            n_jobs_outer=n_jobs_cv,
            n_jobs_inner=n_jobs_model,
        )
        if not isinstance(store_correlations, bool):
            raise TypeError("The store_correlations argument must be a boolean.")

        if store_correlations and not all(
            [isinstance(model, Pipeline) for model in models.values()]
        ):
            raise ValueError(
                "The store_correlations argument is only valid when all models are Scikit-learn Pipelines."
            )
        self.store_correlations = store_correlations

        # Get training and test indices
        train_idx = list(range(len(self.X)))
        base_splits = self._get_base_splits(inner_splitters)

        optim_results = self._worker(
            name=name,
            train_idx=train_idx,
            test_idx=[],
            inner_splitters=inner_splitters,
            models=models,
            hyperparameters=hyperparameters,
            scorers=scorers,
            cv_summary=cv_summary,
            search_type=search_type,
            normalize_fold_results=normalize_fold_results,
            n_iter=n_iter,
            n_jobs_inner=n_jobs_cv,
            base_splits=base_splits,
            n_splits_add=None,
            timestamp=self.real_date,
        )

        self._check_duplicate_results(name)

        # Collect results from the worker
        # quantamental_data, model_data, other_data
        model_choice_data = []
        ftr_coef_data = []
        intercept_data = []
        ftr_selection_data = []
        ftr_corr_data = []

        model_choice_data.append(optim_results["model_choice"])
        ftr_coef_data.append(optim_results["feature_importances"])
        intercept_data.append(optim_results["intercepts"])
        ftr_selection_data.append(optim_results["selected_ftrs"])
        ftr_corr_data.extend(optim_results["ftr_corr"])

        # First create pandas dataframes to store the forecasts
        forecasts_df = pd.DataFrame(
            index=self.X_test.index, columns=[name], data=np.nan, dtype="float32"
        )
        # Create quantamental dataframe of forecasts
        model = (
            models[optim_results["model_choice"][1]]
            .set_params(**optim_results["model_choice"][3])
            .fit(self.X, self.y)
        )
        forecasts = model.predict(self.X_test)
        forecasts_df.iloc[:, 0] = forecasts

        if self.blacklist is not None:
            for cross_section, periods in self.blacklist.items():
                cross_section_key = cross_section.split("_")[0]
                if cross_section_key in self.unique_test_levels:
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
        Returns forward forecasts for one or more pipelines.

        Parameters
        ----------
        name : str or list, optional
            Label(s) of forecast(s). Default is all stored in the
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

            invalid_names = [n for n in name if n not in self.preds.xcat.unique()]
            if invalid_names:
                raise ValueError(
                    f"""The following process name(s) are not in the list of already-run
                    pipelines: {invalid_names}. Please check the names carefully. If
                    correct, please run calculate_predictions() first.
                    """
                )
            preds = self.preds[self.preds.xcat.isin(name)]

        signals_df = QuantamentalDataFrame(
            df=preds,
            categorical=self.df.InitializedAsCategorical,
        ).to_original_dtypes()

        return signals_df

    def get_selected_features(self, name=None):
        """
        Returns the selected features for one or more pipelines.

        Parameters
        ----------
        name: str or list, optional
            Label(s) of pipeline(s). Default is all stored in the
            class instance.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of the selected features at each retraining date.
        """
        if name is None:
            return self.selected_ftrs
        
        if isinstance(name, str):
            name = [name]
        elif not isinstance(name, list):
            raise TypeError(
                "The process name must be a string or a list of strings."
            )

        invalid_names = [n for n in name if n not in self.selected_ftrs.name.unique()]
        if invalid_names:
            raise ValueError(
                f"""The following process name(s) are not in the list of already-run
                pipelines: {invalid_names}. Please check the names carefully. If
                correct, please run calculate_predictions() first.
                """
            )
        return self.selected_ftrs[self.selected_ftrs.name.isin(name)]

    def get_feature_importances(self, name=None):
        """
        Returns feature importances for one or more pipelines.

        Parameters
        ----------
        name: str or list, optional
            Label(s) of pipeline(s). Default is all stored in the
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

        if isinstance(name, str):
            name = [name]
        elif not isinstance(name, list):
            raise TypeError(
                "The process name must be a string or a list of strings."
            )

        invalid_names = [n for n in name if n not in self.feature_importances.name.unique()]
        if invalid_names:
            raise ValueError(
                f"""The following process name(s) are not in the list of already-run
                pipelines: {invalid_names}. Please check the names carefully. If
                correct, please run calculate_predictions() first.
                """
            )

        return self.feature_importances[
            self.feature_importances.name.isin(name)
        ].sort_values(by="real_date")

    def get_intercepts(self, name=None):
        """
        Returns intercepts for one or more pipelines.

        Parameters
        ----------
        name: str or list, optional
            Label(s) of pipeline(s). Default is all stored in the
            class instance.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of the intercepts, if available, learnt at each retraining
            date for a given pipeline.
        """
        if name is None:
            return self.intercepts

        if isinstance(name, str):
            name = [name]
        elif not isinstance(name, list):
            raise TypeError(
                "The process name must be a string or a list of strings."
            )
        
        invalid_names = [n for n in name if n not in self.intercepts.name.unique()]
        if invalid_names:
            raise ValueError(
                f"""The following process name(s) are not in the list of already-run
                pipelines: {invalid_names}. Please check the names carefully. If
                correct, please run calculate_predictions() first.
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
        Returns dataframe of feature correlations for one or more pipelines.

        Parameters
        ----------
        name: str or list, optional
            Label(s) of the pipeline(s). Default is all stored in the
            class instance.

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of the correlations between the features passed into a model
            pipeline and the post-processed features inputted into the final model.
        """
        if name is None:
            return self.ftr_corr

        if isinstance(name, str):
            name = [name]
        elif not isinstance(name, list):
            raise TypeError(
                "The process name must be a string or a list of strings."
            )
        
        invalid_names = [n for n in name if n not in self.ftr_corr.name.unique()]
        if invalid_names:
            raise ValueError(
                f"""The following process name(s) are not in the list of already-run
                pipelines: {invalid_names}. Please check the names carefully. If
                correct, please run calculate_predictions() first.
                """
            )

        return self.ftr_corr[self.ftr_corr.name.isin(name)]

    def _check_factor_availability(self, df, xcats, real_date):
        """
        Check the date is in the dataframe and all categories are available on the date.
        """
        # Check that the date is in the span of the dataframe
        min_date = df.real_date.min()
        max_date = df.real_date.max()
        if real_date <= min_date or real_date > max_date:
            raise ValueError(
                f"Real date {real_date} is either not larger than the earliest date in the dataframe"
                " or nor smaller or equal to the latest date in the dataframe."
            )

        # Check that the date is in the dataframe
        if real_date not in df.real_date.unique():
            raise ValueError(f"Real date {real_date} is not in the dataframe.")

        # Check that all categories are available on the date
        num_categories = len(df[df.real_date == real_date].xcat.unique())
        if num_categories != len(xcats):
            raise ValueError(
                f"Not all categories are available on the real date {real_date}."
            )


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    from macrosynergy.management.types import QuantamentalDataFrame

    from macrosynergy.learning import (
        ExpandingKFoldPanelSplit,
        sharpe_ratio,
        MapSelector,
        PanelStandardScaler,
        PanelPCA,
    )

    from sklearn.metrics import make_scorer
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler

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

    # Initialize the return forecaster
    rf = ReturnForecaster(
        df=dfd,
        xcats=["CRY", "GROWTH", "INFL", "XR"],
        real_date="2020-12-31",
        freq="M",
        lag=1,
        xcat_aggs=["last", "sum"],
    )

    rf.calculate_predictions(
        name="ridge1",
        models={"Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])},
        hyperparameters={"Ridge": {}},
        scorers={
            "sharpe": make_scorer(sharpe_ratio, greater_is_better=True),
        },
        inner_splitters={
            "Expanding": ExpandingKFoldPanelSplit(5),
        },
        cv_summary="mean-std",
    )

    rf.calculate_predictions(
        name="ridge100",
        models={
            "Ridge": Pipeline(
                [("scaler", StandardScaler()), ("ridge", Ridge(alpha=100))]
            )
        },
        hyperparameters={"Ridge": {}},
        scorers={
            "sharpe": make_scorer(sharpe_ratio, greater_is_better=True),
        },
        inner_splitters={
            "Expanding": ExpandingKFoldPanelSplit(5),
        },
        cv_summary="mean-std",
    )

    rf.calculate_predictions(
        name="var+lr",
        models={
            "Ridge": Pipeline(
                [
                    ("scaler", PanelStandardScaler()),
                    ("selector", MapSelector(n_factors=2)),
                    ("ridge", LinearRegression()),
                ]
            )
        },
        hyperparameters={"Ridge": {}},
        scorers={
            "sharpe": make_scorer(sharpe_ratio, greater_is_better=True),
        },
        inner_splitters={
            "Expanding": ExpandingKFoldPanelSplit(5),
        },
        cv_summary="mean-std",
    )

    rf.calculate_predictions(
        name="pca+lr",
        models={
            "Ridge": Pipeline(
                [
                    ("scaler", PanelStandardScaler()),
                    ("selector", PanelPCA(n_components=2, adjust_signs=True)),
                    ("ridge", LinearRegression()),
                ]
            )
        },
        hyperparameters={"Ridge": {}},
        scorers={
            "sharpe": make_scorer(sharpe_ratio, greater_is_better=True),
        },
        inner_splitters={
            "Expanding": ExpandingKFoldPanelSplit(5),
        },
        cv_summary="mean-std",
        store_correlations=True,
    )

    print(rf.get_optimized_signals())
    print(rf.get_feature_importances())
    print(rf.get_intercepts())
    print(rf.get_selected_features())
    print(rf.get_feature_correlations())
