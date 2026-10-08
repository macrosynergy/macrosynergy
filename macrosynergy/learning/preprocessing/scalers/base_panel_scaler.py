from abc import ABC, abstractmethod

import pandas as pd
from pandas.core.dtypes.common import is_datetime64_any_dtype
from sklearn.base import BaseEstimator, TransformerMixin

from macrosynergy.compat import OneToOneFeatureMixin


class BasePanelScaler(BaseEstimator, TransformerMixin, OneToOneFeatureMixin, ABC):
    """
    Base class for scaling a panel of features in a learning pipeline.

    Parameters
    ----------
    type : str, default="panel"
        The panel dimension over which the scaling is applied. Options are
        "panel" and "cross_section".
    columns : list of str, optional
        The columns to scale. Remaining columns are passed through unchanged and need
        not be numeric. If None, all columns are scaled. Default is None.

    Notes
    -----
    Learning algorithms can benefit from scaling each feature to a similar range. This
    ensures they consider each feature equally in the model training process. It can
    also encourage faster convergence of an optimization algorithm.
    """

    def __init__(self, type="panel", columns=None):
        # Checks
        if not isinstance(type, str):
            raise TypeError("`type` must be a string.")
        if type not in ["panel", "cross_section"]:
            raise ValueError("`type` must be either 'panel' or 'cross_section'.")
        if columns is not None:
            if not isinstance(columns, (list, tuple)) or not all(
                isinstance(column, str) for column in columns
            ):
                raise TypeError("`columns` must be None or a list of column names.")
            if len(columns) == 0:
                raise ValueError("`columns` must not be empty.")
            if len(set(columns)) != len(columns):
                raise ValueError("`columns` must not contain duplicates.")

        # Attributes
        self.type = type
        self.columns = columns
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Fit method to learn training set quantities for feature scaling.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix.
        y : pd.Series or pd.DataFrame, default=None
            The target vector.

        Returns
        -------
        self
            The fitted scaler.
        """
        # Checks
        self._check_fit_params(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns
        self.columns_ = X.columns if self.columns is None else pd.Index(self.columns)

        # Set up hash table for storing statistics
        unique_cross_sections = X.index.get_level_values(0).unique()
        self.statistics: dict = {
            cross_section: {feature_name: None for feature_name in self.columns_}
            for cross_section in unique_cross_sections
        }
        self.statistics["panel"] = {
            feature_name: None for feature_name in self.columns_
        }

        # Extract statistics for each feature
        for feature in self.columns_:
            if self.type == "cross_section":
                # Get unique training cross-sections
                unique_cross_sections = X.index.get_level_values(0).unique()
                for cross_section in unique_cross_sections:
                    self.statistics[cross_section][feature] = self.extract_statistics(
                        X.loc[cross_section], feature
                    )
            self.statistics["panel"][feature] = self.extract_statistics(X, feature)

        return self

    def transform(self, X):
        """
        Transform method to scale the input data based on extracted training statistics.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.

        Returns
        -------
        X_transformed : pandas.DataFrame
            The feature matrix with scaled features.
        """
        # Checks
        self._check_transform_params(X)

        # Transform the data
        unique_cross_sections = X.index.get_level_values(0).unique()
        scaled_columns = []
        for feature in self.columns_:
            if self.type == "cross_section":
                # Scale each cross-section based on stored cross sectional statistics in abstract method
                # If the cross-section is not in the statistics dictionary, use the panel statistics
                X_transformed = pd.concat(
                    [
                        self.scale(
                            X.loc[cross_section],
                            feature,
                            self.statistics.get(
                                cross_section, self.statistics["panel"]
                            )[feature],
                        )
                        for cross_section in unique_cross_sections
                    ],
                    axis=0,
                )
            else:
                # Scale the panel based on stored panel statistics in abstract method
                X_transformed = self.scale(
                    X, feature, self.statistics["panel"][feature]
                )
            # Add transformed column to list
            scaled_columns.append(X_transformed)

        # Concatenate the transformed columns
        if self.columns is None:
            X_transformed = pd.DataFrame(
                pd.concat(scaled_columns, axis=1).values,
                index=X.index,
                columns=X.columns,
            )
        else:
            # Overwrite only the scaled columns so that passthrough dtypes are kept
            X_transformed = X.copy()
            X_transformed[self.columns_] = pd.concat(scaled_columns, axis=1).values

        return X_transformed

    @abstractmethod
    def extract_statistics(self, X, feature):
        """
        Determine the relevant statistics for feature scaling.
        """
        pass

    @abstractmethod
    def scale(self, X, feature, statistics):
        """
        Scale the input data based on the relevant statistics.
        """
        pass

    def _check_fit_params(self, X, y):
        """
        Checks the input data for the fit method.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.
        """
        # Checks only necessary on X
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the selector must be a pandas dataframe. ",
                "If used as part of an sklearn pipeline, ensure that previous steps ",
                "return a pandas dataframe.",
            )
        columns = X.columns if self.columns is None else pd.Index(self.columns)
        missing_columns = columns.difference(X.columns)
        if len(missing_columns) > 0:
            raise ValueError(
                "The following columns to be scaled are missing from the input feature "
                f"matrix: {list(missing_columns)}."
            )
        if not X[columns].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All scaled columns in the input feature matrix for PanelStandardScaler",
                " must be numeric.",
            )
        if X[columns].isnull().values.any():
            raise ValueError(
                "The scaled columns in the input feature matrix for PanelStandardScaler "
                "must not contain any missing values."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("The input feature matrix for X must be multi-indexed.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not is_datetime64_any_dtype(X.index.get_level_values(1).dtype):
            raise TypeError("The inner index of X must be datetime.")

    def _check_transform_params(self, X):
        """
        Input checks for the transform method.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the scaler must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        columns = X.columns if self.columns is None else pd.Index(self.columns)
        missing_columns = columns.difference(X.columns)
        if len(missing_columns) > 0:
            raise ValueError(
                "The following columns to be scaled are missing from the input feature "
                f"matrix: {list(missing_columns)}."
            )
        if not X[columns].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All scaled columns in the input feature matrix for PanelStandardScaler",
                " must be numeric.",
            )
        if X[columns].isnull().values.any():
            raise ValueError(
                "The scaled columns in the input feature matrix for PanelStandardScaler "
                "must not contain any missing values."
            )
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not is_datetime64_any_dtype(X.index.get_level_values(1).dtype):
            raise TypeError("The inner index of X must be datetime.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The input feature matrix must have the same number of columns as the "
                "training feature matrix."
            )
        if not X.columns.equals(self.feature_names_in_):
            raise ValueError(
                "The input feature matrix must have the same columns as the training "
                "feature matrix."
            )
