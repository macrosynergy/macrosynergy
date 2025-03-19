"""
Base class for feature selection on panel data.
"""

import pandas as pd

import warnings

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.exceptions import NotFittedError

from abc import ABC, abstractmethod


class BasePanelSelector(BaseEstimator, SelectorMixin, ABC):
    """
    Base class for statistical feature selection over a panel.
    """

    def __init__(self):
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Learn optimal features based on a training set pair (X, y).

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame, optional
            The target vector.
        """
        # Checks
        self._check_fit_params(X, y)

        # Store useful input information
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns

        # Standardise the features for fair comparison
        X = ((X - X.mean()) / X.std()).copy()

        self.mask = self.determine_features(X, y)

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
                "before calling get_feature_names_out().",
            )

        return self.feature_names_in_[self.get_support(indices=False)]

    def _check_fit_params(self, X, y):
        """
        Checks the input data for the fit method.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame, optional
            The target vector.
        """
        # Check input feature matrix
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the selector must be a pandas dataframe. ",
                "If used as part of an sklearn pipeline, ensure that previous steps ",
                "return a pandas dataframe.",
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("The input dataframe must be multi-indexed.")
        if not X.index.nlevels == 2:
            raise ValueError("X must be multi-indexed with two levels.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All columns in the input feature matrix for a panel selector ",
                "must be numeric.",
            )
        if X.isnull().values.any():
            raise ValueError(
                "The input feature matrix for a panel selector must not contain any "
                "missing values."
            )
        # Check target vector, if provided
        if y is not None:
            if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
                raise TypeError(
                    "Target vector for the selector must be a pandas series or dataframe. ",
                    "If used as part of an sklearn pipeline, ensure that previous steps ",
                    "return a pandas series or dataframe.",
                )
            if isinstance(y, pd.DataFrame):
                if y.shape[1] != 1:
                    raise ValueError(
                        "The target dataframe must have only one column. If used as part of ",
                        "an sklearn pipeline, ensure that previous steps return a pandas ",
                        "series or single-column dataframe.",
                    )
            if not isinstance(y.index, pd.MultiIndex):
                raise ValueError("The target vector must be multi-indexed.")
            if not y.index.nlevels == 2:
                raise ValueError("y must be multi-indexed with two levels.")
            if not y.index.get_level_values(0).dtype == "object":
                raise TypeError("The outer index of y must be strings.")
            if not y.index.get_level_values(1).dtype == "datetime64[ns]":
                raise TypeError("The inner index of y must be datetime.date.")
            if not X.index.equals(y.index):
                raise ValueError(
                    "The indices of the input dataframe X and the output dataframe y don't "
                    "match."
                )
            if y.isnull().values.any():
                raise ValueError(
                    "The target vector for a panel selector must not contain any "
                    "missing values."
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
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("The input dataframe must be multi-indexed.")
        if not X.index.nlevels == 2:
            raise ValueError("X must be multi-indexed with two levels.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All columns in the input feature matrix for a panel selector ",
                "must be numeric.",
            )
        if X.isnull().values.any():
            raise ValueError(
                "The input feature matrix for a panel selector must not contain any "
                "missing values."
            )

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
