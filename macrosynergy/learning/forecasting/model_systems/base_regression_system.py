import numpy as np
import pandas as pd

import datetime
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin
from macrosynergy.management.validation import _validate_Xy_learning


class BaseRegressionSystem(BaseEstimator, RegressorMixin, ABC):
    def __init__(
        self,
        roll=None,
        data_freq="D",
        min_xs_samples=2,
    ):
        """
        Base class for systems of regressors.

        Parameters
        ----------
        roll : int, default=None
            The lookback of the rolling window for the regression. If None,
            the entire cross-sectional history is used for each regression. This should
            be specified in units of the data frequency, possibly adjusted by the
            data_freq attribute.
        data_freq : str, default="D"
            Training set data frequency. This is primarily to be used within the context
            of market beta estimation in the MarketBetaEstimator class in
            `macrosynergy.learning`. Accepted strings are 'D' for daily, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly.
        min_xs_samples : int, default=2
            The minimum number of samples required in each cross-section training set for
            a regression model to be fitted.

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
        if (roll is not None) and (not isinstance(roll, int)):
            raise TypeError("roll must be an integer or None.")
        if (roll is not None) and (roll <= 0):
            raise ValueError("roll must be a positive integer.")

        if not isinstance(data_freq, str):
            raise TypeError("The data_freq argument must be a string.")
        if data_freq not in ["D", "W", "M", "Q"]:
            raise ValueError(
                "Invalid data frequency. Accepted values are 'D', 'W', 'M' and 'Q'."
            )
        if not isinstance(min_xs_samples, int):
            raise TypeError("The min_xs_samples argument must be an integer.")
        if min_xs_samples < 2:
            raise ValueError("The min_xs_samples argument must be at least 2.")

        self.roll = roll
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples

        self.models_ = {}

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
        y : Union[pd.DataFrame, pd.Series, numpy.ndarray]
            Target variable.
        """
        # Checks
        self._check_fit_params(X, y)
        self.n_ = X.shape[1]
        # Adjust the data frequency of the input data if necessary
        min_xs_samples = self.select_data_freq()
        cross_sections = X.index.get_level_values(0).unique()
        X = self._downsample_by_data_freq(X)
        y = self._downsample_by_data_freq(y)

        # Fit a model on each cross-section
        for section in cross_sections:
            X_section = X[X.index.get_level_values(0) == section]
            y_section = y[y.index.get_level_values(0) == section]
            unique_dates = sorted(X_section.index.unique())
            num_dates = len(unique_dates)
            # Check if there are enough samples to fit a model
            if not self._check_xs_dates(min_xs_samples, num_dates):
                continue
            # If a roll is specified, then adjust the dates accordingly
            # Only do this if the number of dates is greater than the roll
            if self.roll:
                if num_dates < self.roll:
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
        if self.n_ != X.shape[1]:
            raise ValueError(
                "The number of features in X does not match the number of "
                "features in the training data."
            )

        predictions = pd.Series(index=X.index, data=np.nan)

        # Check whether each test cross-section has an associated model
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
        mask = X_section.index.isin(right_dates)
        X_section = X_section[mask]
        y_section = y_section[mask]
        return X_section, y_section

    def select_data_freq(self):
        """
        Adjust cross-sectional availability requirement when frequency resampling
        is applied.

        Returns
        -------
        min_xs_samples : int
            The minimum number of samples required in each cross-section training set for
            a regression model to be fitted.
        """
        if self.data_freq == "D":
            min_xs_samples = self.min_xs_samples
        elif self.data_freq == "W":
            min_xs_samples = self.min_xs_samples // 5
        elif self.data_freq == "M":
            min_xs_samples = self.min_xs_samples // 21
        elif self.data_freq == "Q":
            min_xs_samples = self.min_xs_samples // 63
        else:
            raise ValueError(
                "Invalid data frequency. Accepted values are 'D', 'W', 'M' and 'Q'."
            )
        return min_xs_samples

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
        y : Union[pd.DataFrame, pd.Series, numpy.ndarray]
            Target variable.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The X argument must be a pandas DataFrame.")
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError(
                "The y argument must be a pandas DataFrame, Series or " "numpy array."
            )
        if len(X) != len(y):
            raise ValueError("The number of samples in X and y must match.")
        if isinstance(y, np.ndarray):
            # This can happen during sklearn's GridSearch when a voting regressor is used
            if y.ndim != 1 or y.ndim != 2:
                raise ValueError("y must be a 1D or 2D array.")
            if y.ndim == 2 and y.shape[1] != 1:
                raise ValueError("y must have only one column.")
            y = pd.Series(y, index=X.index)
