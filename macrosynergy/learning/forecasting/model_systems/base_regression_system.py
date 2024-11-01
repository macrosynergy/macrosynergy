import numpy as np
import pandas as pd

import datetime
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin


class BaseRegressionSystem(BaseEstimator, RegressorMixin, ABC):
    def __init__(
        self,
        roll="full",
        min_xs_samples=2,
        data_freq=None,
    ):
        """
        Base class for systems of regressors.

        Parameters
        ----------
        roll : int or str, default = "full"
            The lookback of the rolling window for the regression. If "full",
            the entire cross-sectional history is used for each regression. Otherwise,
            this should be specified in units of the data frequency. If `data_freq` is not
            None or "unadjusted", then an integer value for `roll` should be expressed in
            units of the data frequency provided in the `data_freq` argument.
        min_xs_samples : int, default=2
            The minimum number of samples required in each cross-section training set for
            a regression model to be fitted for that cross-section. If `data_freq` is None
            or "unadjusted", this parameter is specified in terms of the native dataset
            frequency. Otherwise, this parameter should be expressed in units of the
            frequency specified in the `data_freq` argument.
        data_freq : str, optional
            Training set data frequency. This is primarily to be used within the context
            of market beta estimation in the `BetaEstimator` class in
            `macrosynergy.learning`, allowing one to cross-validate the underlying data
            frequency for good beta estimation. Accepted strings are 'unadjusted', 'W' for
            weekly, 'M' for monthly and 'Q' for quarterly. It is recommended to set this
            parameter to "W", "M" or "Q" only when the native dataset frequency is greater.

        Notes
        -----
        Systems of regressors are used to fit a different regression model on each
        cross-section of a panel. This is useful when one believes the within-group
        relationships are sufficiently different to warrant separate models, or when
        Simpson's paradox is a concern.

        A concern with this approach, however, is that the number of samples in each
        cross-section may be too small to fit a model. This is particularly true when
        dealing with low-frequency macro quantamental data.
        """
        # Checks
        if not isinstance(roll, (str, int)):
            raise TypeError("roll must be either a string or integer.")
        if isinstance(roll, str) and roll != "full":
            raise ValueError("roll must equal `full` when a string is specified.")
        if isinstance(roll, int) and roll <= 1:
            raise ValueError(
                "roll must be greater than 1 when an integer is specified."
            )

        if not isinstance(min_xs_samples, int):
            raise TypeError("The min_xs_samples argument must be an integer.")
        if min_xs_samples < 2:
            raise ValueError("The min_xs_samples argument must be at least 2.")

        if data_freq is not None:
            if not isinstance(data_freq, str):
                raise TypeError("The data_freq argument must be a string.")
            if data_freq not in ["unadjusted", "W", "M", "Q"]:
                raise ValueError(
                    "data_freq must be one of 'unadjusted', 'W', 'M' or 'Q'."
                )

        # Set attributes
        self.roll = roll
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples

        self.models_ = None

    def fit(
        self,
        X,
        y,
    ):
        """
        Fit a regression on each cross-section of a panel, subject to availability.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.Series, pd.DataFrame or np.ndarray
            Target variable.

        Returns
        -------
        self : BaseRegressionSystem
            Fitted regression system object.
        """
        # Checks
        y = self._check_fit_params(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns
        self.models_ = {}

        # Downsample data frequency if necessary
        if (self.data_freq is not None) and (self.data_freq != "unadjusted"):
            X = self._downsample_by_data_freq(X)
            y = self._downsample_by_data_freq(y)

        # Iterate over cross-sections and fit a regression model on each
        cross_sections = X.index.unique(level=0)
        for section in cross_sections:
            X_section = X.xs(section, level=0, drop_level=False)
            y_section = y.xs(section, level=0, drop_level=False)

            unique_dates = sorted(X_section.index.unique())
            num_dates = len(unique_dates)
            # Skip cross-sections with insufficient samples
            if not self._check_xs_dates(self.min_xs_samples, num_dates):
                continue
            # Roll the data if necessary
            if self.roll and self.roll != "full":
                if num_dates <= self.roll:
                    continue
                else:
                    X_section, y_section = self.roll_dates(
                        self.roll, X_section, y_section, unique_dates
                    )
            # Fit the model
            self._fit_cross_section(section, X_section, y_section)

        return self

    def _fit_cross_section(self, section, X_section, y_section):
        """
        Fit a regression model on a single cross-section.

        Parameters
        ----------
        section : str
            The identifier of the cross-section.
        X_section : pd.DataFrame
            Input feature matrix for the cross-section.
        y_section : pd.Series
            Target variable for the cross-section.
        """
        model = self.create_model()
        model.fit(pd.DataFrame(X_section), y_section)
        # Store model and coefficients
        self.models_[section] = model
        self.store_model_info(section, model)

    def predict(
        self,
        X,
    ):
        """
        Make predictions over a panel dataset based on trained observation-specific
        models.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        predictions : pd.Series
            Pandas series of predictions, multi-indexed by cross-section and date.
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The X argument must be a pandas DataFrame.")
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not np.all(X.columns == self.feature_names_in_):
            raise ValueError(
                "The input feature matrix must have the same columns as the",
                "training feature matrix.",
            )
        if len(X.columns) != self.n_features_in_:
            raise ValueError(
                "The input feature matrix must have the same number of",
                "columns as the training feature matrix.",
            )
        if X.isnull().values.any():
            raise ValueError(
                "The input feature matrix must not contain any missing values."
            )
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All columns in the input feature matrix for regression systems",
                " must be numeric.",
            )

        predictions = pd.Series(index=X.index, data=np.nan)

        # Store predictions for each test cross-section, if an existing model is available
        cross_sections = predictions.index.get_level_values(0).unique()
        for idx, section in enumerate(cross_sections):
            if section in self.models_.keys():
                # If a model exists, return the estimated OOS contract return.
                predictions[predictions.index.get_level_values(0) == section] = (
                    self.models_[section].predict(X.xs(section, level=0)).flatten()
                )

        return predictions

    def roll_dates(self, roll, X_section, y_section, unique_dates):
        """
        Adjust dataset to be contained within a rolling window.

        Parameters
        ----------
        roll : int
            The lookback of the rolling window.
        X_section : pd.DataFrame
            Input feature matrix for the cross-section.
        y_section : pd.Series
            Target variable for the cross-section.
        unique_dates : list
            List of unique dates in the cross-section.

        Returns
        -------
        X_section : pd.DataFrame
            Input feature matrix for the cross-section, adjusted for the rolling window.
        y_section : pd.Series
            Target variable for the cross-section, adjusted for the rolling window.
        """
        right_dates = unique_dates[-roll:]
        
        common_index = X_section.index.intersection(right_dates)

        X_section = X_section.reindex(common_index)
        y_section = y_section.reindex(common_index)

        return X_section, y_section

    @abstractmethod
    def store_model_info(self, section, model):
        """
        Store necessary model information for explainability.

        Parameters
        ----------
        section : str
            The identifier of the cross-section.
        model : RegressorMixin
            The fitted regression model.

        Notes
        ------
        Must be overridden.
        """
        pass

    @abstractmethod
    def create_model(self):
        """
        Instantiate a regression model for a given cross-section.

        Notes
        -----
        Must be overridden.
        """
        pass

    def _check_xs_dates(self, min_xs_samples, num_dates):
        """
        Cross-sectional availability check.

        Parameters
        ----------
        min_xs_samples : int
            The minimum number of samples required in each cross-section training set for
            a regression model to be fitted.
        num_dates : int
            The number of unique dates in the cross-section.

        Returns
        -------
        bool
            True if the number of samples is sufficient, False otherwise
        """
        if num_dates < min_xs_samples:
            return False

        return True

    def _downsample_by_data_freq(self, df):
        """
        Resample the input dataset to the specified data frequency.

        Parameters
        ----------
        df : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            Resampled feature matrix.
        """
        return (
            df.groupby(
                [
                    pd.Grouper(level="cid"),
                    pd.Grouper(level="real_date", freq=self.data_freq),
                ]
            )
            .sum()
            .copy()
        )

    def _check_fit_params(self, X, y):
        """
        Input checks for the fit method parameters.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.Series, pd.DataFrame or np.ndarray
            Target variable.
        """
        # X
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The X argument must be a pandas DataFrame.")
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All columns in the input feature matrix for regression systems",
                " must be numeric.",
            )
        if X.isnull().values.any():
            raise ValueError(
                "The input feature matrix for regression systems must not contain any "
                "missing values."
            )

        if not isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError(
                "The y argument must be a pandas DataFrame, Series or numpy array."
            )
        if len(X) != len(y):
            raise ValueError("The number of samples in X and y must match.")
        if isinstance(y, np.ndarray):
            # This can happen during sklearn's GridSearch when a voting regressor is used
            if y.ndim != 1 and y.ndim != 2:
                raise ValueError("y must be a 1D or 2D array.")
            if y.ndim == 2 and y.shape[1] != 1:
                raise ValueError("y must have only one column.")
            y = pd.Series(y, index=X.index)
        if not isinstance(y, np.ndarray):
            if not np.issubdtype(y.values.dtype, np.number):
                raise ValueError("The target vector must be numeric.")
            if y.isnull().values.any():
                raise ValueError(
                    "The target vector must not contain any missing values."
                )
        else:
            if not np.issubdtype(y.dtype, np.number):
                raise ValueError("The target vector must be numeric.")
            if np.isnan(y).any():
                raise ValueError(
                    "The target vector must not contain any missing values."
                )

        return y
