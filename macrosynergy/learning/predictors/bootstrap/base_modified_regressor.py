import numpy as np
import pandas as pd
import datetime

from sklearn.base import BaseEstimator, RegressorMixin, clone
from macrosynergy.learning.predictors.bootstrap import BasePanelBootstrap
from abc import ABC, abstractmethod

from typing import Union, Optional

class BaseModifiedRegressor(BaseEstimator, RegressorMixin, BasePanelBootstrap, ABC):
    def __init__(
        self,
        model: RegressorMixin,
        method: str,
        error_offset: float = 1e-5,
        bootstrap_method: str = "panel",
        bootstrap_iters: int = 100,
        resample_ratio: Union[float, int] = 1,
        max_features: Optional[Union[str, int, float]] = None,
        analytic_method: Optional[str] = None,
    ):
        """
        Base class for linear regressors where coefficients are modified by estimated
        standard errors to account for statistical precision of the estimates.

        :param <RegressorMixin> model: The underlying linear model to be modified. This
            model must have `coef_` and `intercept_` attributes, in accordance with
            standard `scikit-learn` conventions.
        :param <str> method: The method used to modify the coefficients. Accepted values
            are "analytic" and "bootstrap".
        :param <float> error_offset: A small offset to add to the standard errors to
            prevent division by zero in the case of very small standard errors. Default
            value is 1e-5.
        :param <str> bootstrap_method: The bootstrap method used to modify the coefficients.
            Accepted values are "panel", "period", "cross", "cross_per_period"
            and "period_per_cross". Default value is "panel".
        :param <int> bootstrap_iters: The number of bootstrap iterations to perform in
            order to determine the standard errors of the model parameters under the bootstrap
            approach. Default value is 100.
        :param <Union[float, int]> resample_ratio: The ratio of resampling units comprised
            in each bootstrap dataset. This is a fraction of the quantity of the panel
            component to be resampled. Default value is 1.
        :param <Optional[Union[str, int, float]]> max_features: The number of features to
            consider in each bootstrap dataset. This can be used to increase the 
            variation of the bootstrap datasets. Default is None and currently not
            implemented.
        :param <Optional[str]> analytic_method: The analytic method used to calculate
            standard errors. Expressions for analytic standard errors are expected to be
            written within the method `adjust_analytical_se` and this parameter can be
            passed into `adjust_analyical_se` for an alternative analytic standard error
            estimate, for instance White's estimator. Default value is None.

        :return None
        """
        # Checks
        super().__init__(
            bootstrap_method=bootstrap_method,
            resample_ratio=resample_ratio,
            max_features=max_features,
        )

        self._check_additional_init_params(
            model=model,
            method=method,
            error_offset=error_offset,
            bootstrap_iters=bootstrap_iters,
            analytic_method=analytic_method,
        )

        # Set attributes
        self.model = model
        self.method = method
        self.error_offset = error_offset
        self.bootstrap_iters = bootstrap_iters
        self.analytic_method = analytic_method

    def _check_additional_init_params(
        self,
        model: RegressorMixin,
        method: str,
        error_offset: float,
        bootstrap_iters: int,
        analytic_method: Optional[str],
    ):
        """
        Method to check the validity of the initialization parameters of the class.

        :param <RegressorMixin> model: The underlying linear model to be modified. This
            model must have `coef_` and `intercept_` attributes once fit, in accordance
            with standard `scikit-learn` conventions.
        :param <str> method: The method used to modify the coefficients. Accepted values
            are "analytic" and "bootstrap".
        :param <float> error_offset: A small offset to add to the standard errors to
            prevent division by zero in the case of very small standard errors.
        :param <int> bootstrap_iters: The number of bootstrap iterations to perform in
            order to determine the standard errors of the model parameters under the
            bootstrap approach.
        :param <Optional[str]> analytic_method: The analytic method used to calculate
            standard errors. Expressions for analytic standard errors are expected to be
            written within the method `adjust_analytical_se` and this parameter can be
            passed into `adjust_analyical_se` for an alternative analytic standard error
            estimate, for instance White's estimator.

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
        if method not in ["analytic", "bootstrap"]:
            raise ValueError("method must be either 'analytic' or 'bootstrap'.")
        
        # error_offset
        if not isinstance(error_offset, (float, int)):
            raise TypeError("error_offset must be a float or an integer.")
        if error_offset <= 0:
            raise ValueError("error_offset must be greater than 0.")
        
        # bootstrap_iters
        if method == "bootstrap":
            if not isinstance(bootstrap_iters, int):
                raise TypeError("bootstrap_iters must be an integer.")
            if bootstrap_iters <= 0:
                raise ValueError("bootstrap_iters must be a positive integer.")
            
        # analytic_method
        if method == "analytic":
            if analytic_method is not None:
                if not isinstance(analytic_method, str):
                    raise TypeError("analytic_method must be a string.")
                
    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ):
        """
        Fit method to fit the underlying linear model, as passed into the constructor,
        and subsequently modify coefficients based on estimated standard errors to produce
        a trading signal that accounts for the statistical precision of the factor weights.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.

        :return <BaseModifiedRegressor>
        """
        # Checks
        self._check_fit_params(X=X, y=y)

        # Fit
        self.model.fit(X, y)

        if not hasattr(self.model, "coef_"):
            raise AttributeError("The underlying model must have a `coef_` attribute.")
        if not hasattr(self.model, "intercept_"):
            raise AttributeError(
                "The underlying model must have an `intercept_` attribute."
            )
        
        # Modify coefficients
        if self.method == "analytic":
            self.intercept_, self.coef_ = self.adjust_analytical_se(
                self.model,
                X,
                y,
                self.analytic_method,
            )
        elif self.method == "bootstrap":
            # clone the model to avoid modifying the original model
            model = clone(self.model)
            self.intercept_, self.coef_ = self.adjust_bootstrap_se(
                model,
                X,
                y,
            )

        return self
    
    def predict(
        self,
        X: pd.DataFrame,
    ):
        """
        Method to predict the target variable using the underlying linear model.

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
        
        return self.model.predict(X)

    def create_signal(
        self,
        X: pd.DataFrame,
    ):
        """
        Method to create a signal based on adjusting the underlying factor model weights
        by their standard errors.

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
    
    def adjust_bootstrap_se(
        self,
        model: RegressorMixin,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ):
        # Create storage for bootstrap coefficients and intercepts
        bootstrap_coefs = np.zeros((self.bootstrap_iters, X.shape[1]))
        bootstrap_intercepts = np.zeros(self.bootstrap_iters)

        # Bootstrap loop
        for i in range(self.bootstrap_iters):
            X_resampled, y_resampled = self.create_bootstrap_dataset(X, y)
            model.fit(X_resampled, y_resampled)
            bootstrap_coefs[i] = model.coef_
            bootstrap_intercepts[i] = model.intercept_

        # Calculate standard errors
        coef_se = np.std(bootstrap_coefs, axis=0, ddof=0)
        intercept_se = np.std(bootstrap_intercepts, ddof=0)

        # Adjust the coefficients and intercepts by the standard errors
        coef = self.model.coef_ / (coef_se + self.error_offset)
        intercept = self.model.intercept_ / (intercept_se + self.error_offset)

        return intercept, coef
    
    def adjust_analytical_se(
        self,
        model: RegressorMixin,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        analytic_method: Optional[str],
    ):
        raise NotImplementedError(
            "Analytical standard error adjustments are not available for most models."
            "This function must be implemented in a subclass of BaseModifiedRegressor "
            "if known standard error expressions are available."
        )
    
    def _check_fit_params(
        self,
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

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    obj = BaseModifiedRegressor(model=LinearRegression(), method="bootstrap")