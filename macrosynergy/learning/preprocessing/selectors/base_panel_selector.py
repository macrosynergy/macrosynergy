"""
Base class for feature selection on panel data.
"""

import numpy as np
import pandas as pd

import datetime
import warnings

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.exceptions import NotFittedError

from abc import ABC, abstractmethod

class BasePanelSelector(BaseEstimator, SelectorMixin, ABC):
    """
    Base class for statistical feature selection over a panel.
    """
    def fit(self, X, y):
        """
        Learn optimal features based on a training set pair (X, y).

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.
        """
        # Checks
        self._check_fit_params(X, y)

        # Store useful input information
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.feature_names_in_ = X.columns

        # Standardise the features for fair comparison
        X = ((X - X.mean()) / X.std()).copy()

        self.mask  = self.determine_features(X, y)

        return self
    
    @abstractmethod
    def determine_features(self, X, y):
        """
        Determine mask of selected features based on a training set pair (X, y).

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.

        Returns
        -------
        mask : np.ndarray
            Boolean mask of selected features.
        """
        pass
    
    def transform(self, X):
        """
        Transform method to return only the selected features of the dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.

        Returns
        -------
        X_transformed : pandas.DataFrame
            The feature matrix with only the selected features.
        """
        # Checks
        self._check_transform_params(X)

        # Return selected features
        if sum(self.mask) == 0:
            # Then no features were selected
            warnings.warn(
                "No features were selected. At the given time, no trading decisions can be made based on these features.",
                RuntimeWarning,
            )
            return X.iloc[:, :0]
        
        return X.loc[:, self.mask]
    
    def _get_support_mask(self):
        """
        Private method to return a boolean mask of the features selected for the Pandas
        dataframe.
        """
        return self.mask
    
    def get_feature_names_out(self):
        """
        Method to mask feature names according to selected features.
        """
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The selector has not been fitted. Please fit the selector ",
                "before calling get_feature_names_out()."
            )

        return self.feature_names_in_[self.get_support(indices=False)]
    
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
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the selector must be a pandas dataframe. ",
                "If used as part of an sklearn pipeline, ensure that previous steps ",
                "return a pandas dataframe."
            )
        if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            raise TypeError(
                "Target vector for the selector must be a pandas series or dataframe. ",
                "If used as part of an sklearn pipeline, ensure that previous steps ",
                "return a pandas series or dataframe."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "The target dataframe must have only one column. If used as part of ",
                    "an sklearn pipeline, ensure that previous steps return a pandas ",
                    "series or single-column dataframe."
                )
        if not X.index.nlevels == 2:
            raise ValueError("X must be multi-indexed with two levels.")
        if not y.index.nlevels == 2:
            raise ValueError("y must be multi-indexed with two levels.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not y.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of y must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not y.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of y must be datetime.date.")
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't "
                "match."
            )
        
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
                "Input feature matrix for the selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not X.index.nlevels == 2:
            raise ValueError("X must be multi-indexed with two levels.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
    
        if not X.shape[-1] == self.p:
            raise ValueError(
                "The number of columns of the dataframe to be transformed, X, doesn't "
                "match the number of columns of the training dataframe."
            )
    