import numpy as np
import pandas as pd

import datetime

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

from abc import ABC, abstractmethod

class BasePanelScaler(BaseEstimator, TransformerMixin, OneToOneFeatureMixin, ABC):
    """
    Base class for scaling a panel of features in a learning pipeline. 

    Parameters
    ----------
    type : str, default="panel"
        The panel dimension over which the scaling is applied. Options are 
        "panel" and "cross_section". 
    """
    def __init__(self, type = "panel"):
        # Checks 
        if not isinstance(type, str):
            raise TypeError("`type` must be a string.")
        if type not in ["panel", "cross_section"]:
            raise ValueError("`type` must be either 'panel' or 'cross_section'.")
        
        # Attributes
        self.type = type

    def fit(self, X, y = None):
        """
        Fit method to learn training set quantities for feature scaling. 

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix.
        y : pd.Series or pd.DataFrame, default=None
            The target vector.
        """
        # Checks 
        self._check_fit_params(X, y)

        # Set up hash table for storing statistics
        unique_cross_sections = X.index.get_level_values(0).unique()
        self.statistics: dict = {cross_section : {feature_name : None for feature_name in X.columns} for cross_section in unique_cross_sections}
        self.statistics["panel"] = {feature_name : None for feature_name in X.columns}

        # Extract statistics for each feature
        for feature in X.columns:
            if self.type == "cross_section":
                # Get unique training cross-sections
                unique_cross_sections = X.index.get_level_values(0).unique()
                for cross_section in unique_cross_sections:
                    self.statistics[cross_section][feature] = self.extract_statistics(X.loc[cross_section], feature)
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
        for feature in X.columns:
            if self.type == "cross_section":
                # Scale each cross-section based on stored cross sectional statistics in abstract method
                # If the cross-section is not in the statistics dictionary, use the panel statistics
                X_transformed = pd.concat(
                    [
                        self.scale(X.loc[cross_section], feature, self.statistics.get(cross_section, self.statistics["panel"])[feature])
                        for cross_section in unique_cross_sections
                    ],
                    axis=0,
                )
            else:
                # Scale the panel based on stored panel statistics in abstract method
                X_transformed = self.scale(X, feature, self.statistics["panel"][feature])
            # Add transformed column to list
            scaled_columns.append(X_transformed)

        # Concatenate the transformed columns
        X_transformed = pd.concat(scaled_columns, axis=1)

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
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        
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
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")