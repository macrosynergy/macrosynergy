import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from typing import Union
from abc import ABC, abstractmethod

class BaseModifiedRegressor(BaseEstimator, RegressorMixin, ABC):
    def __init__(
        self,
        model: RegressorMixin,
        method: str,
        error_offset: float = 1e-2,
        bootstrap_method: str = "panel",
        bootstrap_iters: int = 1000,
        resample_ratio: Union[float, int] = 1,
    ):
        """
        Base class for linear regressors where coefficients are modified by estimated
        standard errors to account for statistical precision of the estimates.

        :param <RegressorMixin> model: The underlying linear model to be modified. This 
            model must have `coef_` and `intercept_` attributes, in accordance with 
            standard `scikit-learn` conventions.
        :param <str> method: The method used to modify the coefficients. Accepted values
            are "standard" and "bootstrap".
        :param <float> error_offset: A small offset to add to the standard errors to
            prevent division by zero in the case of very small standard errors. Default
            value is 1e-2.
        :param <str> bootstrap_method: The bootstrap method used to modify the coefficients.
            Accepted values are "panel", "period", "cross", "cross_per_period" 
            and "period_per_cross". Default value is "panel".
        :param <int> bootstrap_iters: The number of bootstrap iterations to perform in 
            order to determine the standard errors of the model parameters under the bootstrap 
            approach. Default value is 1000.   
        :param <Union[float, int]> resample_ratio: The ratio of resampling units comprised
            in each bootstrap dataset. This is a fraction of the quantity of the panel
            component to be resampled. Default value is 1.

        :return None
        """
        self.model = model
        self.method = method
        self.error_offset = error_offset
        self.bootstrap_method = bootstrap_method
        self.bootstrap_iters = bootstrap_iters
        self.resample_ratio = resample_ratio

        self.check_init_params(
            model=self.model,
            method=self.method,
            error_offset=self.error_offset,
            bootstrap_method=self.bootstrap_method,
            bootstrap_iters=self.bootstrap_iters,
            resample_ratio=self.resample_ratio,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ):
        """
        Fit method to fit the underlying linear model, as passed into the constructor, 
        and subsequently modify coefficients based on estimated standard errors to produce 
        a trading signal that accounts for the statistical precision of the factor 

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.

        :return <BaseModifiedRegressor>
        """
        # Checks
        self.check_fit_params(X=X, y=y)

        # Fit 
        self.model.fit(X, y)
        
        if not hasattr(self.model, "coef_"):
            raise AttributeError("The underlying model must have a `coef_` attribute.")
        if not hasattr(self.model, "intercept_"):
            raise AttributeError("The underlying model must have an `intercept_` attribute.")
        
        # Modify coefficients
        if self.method == "standard":
            self.intercept_, self.coef_ = self.adjust_analytical_se(
                self.model,
                X,
                y,
            )
        elif self.method == "bootstrap":
            self.intercept_, self.coef_ = self.adjust_bootstrap_se(
                self.model,
                X,
                y,
                self.bootstrap_method,
                self.bootstrap_iters,
                self.resample_ratio
            )

        return self
    
    def predict(
        self,
        X: pd.DataFrame,
    ):
        """
        Predict method to predict the target variable using the underlying linear model.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")

        return np.dot(X, self.coef_) + self.intercept_
    
    def adjust_analytical_se(
        self,
        model: RegressorMixin,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        **kwargs,
    ):
        raise NotImplementedError(
            "Analytical standard error adjustments are not available for most models."
            "This function must be implemented in a subclass of BaseModifiedRegressor "
            "if known standard error expressions are available."
        )
    
    def adjust_bootstrap_se(
        self,
        model: RegressorMixin,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        bootstrap_method: str,
        bootstrap_iters: int,
        resample_ratio: Union[float, int],
    ):
        # Create storage for bootstrap coefficients and intercepts
        bootstrap_coefs = np.zeros((bootstrap_iters, X.shape[1]))
        bootstrap_intercepts = np.zeros(bootstrap_iters)
        
        # Now create each of the bootstrap datasets
        for i in range(bootstrap_iters):
            # If method is panel, sample with replacement from the entire dataset
            if bootstrap_method == "panel":
                bootstrap_idx = np.random.choice(
                    np.arange(X.shape[0]),
                    size=int(X.shape[0] * resample_ratio),
                    replace=True,
                )
                X_resampled = X.values[bootstrap_idx, :]
                y_resampled = y.values[bootstrap_idx]

            elif bootstrap_method == "period":
                # Resample the unique time periods from the panel
                # and select all observations within those periods
                unique_time_periods = X.index.get_level_values(1).unique()
                bootstrap_periods = np.random.choice(
                    unique_time_periods,
                    size=int(len(unique_time_periods) * resample_ratio),
                    replace=True,
                )
                # now get samples from X and y within those periods
                indices = []
                for period in bootstrap_periods:
                    period_indices = X.index[X.index.get_level_values(1) == period]
                    indices.extend(period_indices.tolist())

                bootstrap_idx = pd.Index(indices)

                X_resampled = X.loc[bootstrap_idx]
                y_resampled = y.loc[bootstrap_idx]

            elif bootstrap_method == "cross":
                # Resample the unique cross sections from the panel
                # and select all observations within those cross sections
                unique_cross_sections = X.index.get_level_values(0).unique()
                bootstrap_cross_sections = np.random.choice(
                    unique_cross_sections,
                    size=int(len(unique_cross_sections) * resample_ratio),
                    replace=True,
                )
                # now get samples from X and y within those cross sections
                indices = []
                for cross_section in bootstrap_cross_sections:
                    cross_section_indices = X.index[X.index.get_level_values(0) == cross_section]
                    indices.extend(cross_section_indices.tolist())

                bootstrap_idx = pd.Index(indices)

                X_resampled = X.loc[bootstrap_idx]
                y_resampled = y.loc[bootstrap_idx]

            elif bootstrap_method == "cross_per_period":
                # Resample observations within each unique time period
                unique_time_periods = X.index.get_level_values(1).unique()
                indices = [] 

                for time_period in unique_time_periods:
                    period_indices = X.index[X.index.get_level_values(1) == time_period]
                    bootstrap_idx = np.random.choice(
                        period_indices,
                        size=int(len(period_indices) * resample_ratio),
                        replace=True,
                    )
                    indices.extend(bootstrap_idx)

                bootstrap_idx = pd.Index(indices)

                X_resampled = X.loc[bootstrap_idx]
                y_resampled = y.loc[bootstrap_idx]

            elif bootstrap_method == "period_per_cross":
                # Resample observations within each unique cross section
                unique_cross_sections = X.index.get_level_values(0).unique()
                indices = [] 

                for cross_section in unique_cross_sections:
                    cross_section_indices = X.index[X.index.get_level_values(0) == cross_section]
                    bootstrap_idx = np.random.choice(
                        cross_section_indices,
                        size=int(len(cross_section_indices) * resample_ratio),
                        replace=True,
                    )
                    indices.extend(bootstrap_idx)

                bootstrap_idx = pd.Index(indices)

                X_resampled = X.loc[bootstrap_idx]
                y_resampled = y.loc[bootstrap_idx]

            model.fit(X_resampled, y_resampled)

            # Store the coefficients and intercepts
            bootstrap_coefs[i] = model.coef_
            bootstrap_intercepts[i] = model.intercept_

        # Calculate the standard errors
        coef_se = np.std(bootstrap_coefs, axis=0, ddof=0)
        intercept_se = np.std(bootstrap_intercepts,ddof=0)

        # Adjust the coefficients and intercepts by the standard errors
        coef = model.coef_ / (coef_se + self.error_offset)
        intercept = model.intercept_ / (intercept_se + self.error_offset)

        return intercept, coef
        
    def check_init_params(
        model: RegressorMixin,
        method: str,
        error_offset: float,
        bootstrap_method: str,
        bootstrap_iters: int,
        resample_ratio: Union[float, int],
    ):
        """
        Method to check the validity of the initialization parameters of the class.

        :param <RegressorMixin> model: The underlying linear model to be modified. This
            model must have `coef_` and `intercept_` attributes once fit, in accordance
            with standard `scikit-learn` conventions.
        :param <str> method: The method used to modify the coefficients. Accepted values
            are "standard" and "bootstrap".
        :param <float> error_offset: A small offset to add to the standard errors to
            prevent division by zero in the case of very small standard errors.
        :param <str> bootstrap_method: The bootstrap method used to modify the
        coefficients. Accepted values are "panel", "period", "cross", "cross_per_period" 
            and "period_per_cross".
        :param <int> bootstrap_iters: The number of bootstrap iterations to perform in 
            order to determine the standard errors of the model parameters under the
            bootstrap approach.
        :param <Union[float, int]> resample_ratio: The ratio of resampling units comprised
            in each bootstrap dataset. This is a fraction of the quantity of the panel
            component to be resampled.

        :return None
        """
        # model
        if not isinstance(model, BaseEstimator):
            raise TypeError("model must be a valid `scikit-learn` estimator.")
        if not isinstance(model, RegressorMixin):
            raise TypeError("model must be a valid `scikit-learn` regressor.")
        # method
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        if method not in ["standard", "bootstrap"]:
            raise ValueError("method must be either 'standard' or 'bootstrap'.")
        # error_offset
        if not isinstance(error_offset, (float, int)):
            raise TypeError("error_offset must be a float or an integer.")
        if error_offset <= 0:
            raise ValueError("error_offset must be greater than 0.")
        # bootstrap_method
        if method == "bootstrap":
            if not isinstance(bootstrap_method, str):
                raise TypeError("bootstrap_method must be a string.")
            if bootstrap_method not in [
                "standard",
                "panel",
                "period",
                "cross",
                "cross_per_period",
                "period_per_cross",
            ]:
                raise ValueError(
                    "bootstrap_method must be one of 'standard', 'panel', 'period', "
                    "'cross', 'cross_per_period', or 'period_per_cross'."
                )
        # bootstrap_iters
        if method == "bootstrap":
            if not isinstance(bootstrap_iters, int):
                raise TypeError("bootstrap_iters must be an integer.")
            if bootstrap_iters <= 0:
                raise ValueError("bootstrap_iters must be a positive integer.")
            
        # resample_ratio
        if method == "bootstrap":
            if not isinstance(resample_ratio, (float, int)):
                raise TypeError("resample_ratio must be a float or an integer.")
            if resample_ratio <= 0:
                raise ValueError("resample_ratio must be greater than 0.")
            if resample_ratio > 1:
                raise ValueError("resample_ratio must be less than or equal to 1.")
            
    def check_fit_params(
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ):
        """
        Method to check the validity of the fit parameters of the class.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            raise TypeError(
                "Target vector must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if isinstance(y, pd.DataFrame) and y.shape[1] != 1:
            raise ValueError(
                "The target dataframe must have only one column. If used as part of "
                "an sklearn pipeline, ensure that previous steps return a pandas "
                "series or dataframe."
            )

        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of y must be datetime.date.")
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't "
                "match."
            )
        