import pandas as pd

from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin

class BaseModifiedRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model: RegressorMixin,
        method: str,
        bootstrap_method: str = "panel",
        bootstrap_iters: int = 1000,
        sample_size_ratio: Union[float, int] = 1,
    ):
        """
        Base class for linear regressors where coefficients are modified by estimated
        standard errors to account for statistical precision of the estimates.

        :param <RegressorMixin> model: The underlying linear model to be modified. This 
            model must have `coef_` and `intercept_` attributes, in accordance with 
            standard `scikit-learn` conventions.
        :param <str> method: The method used to modify the coefficients. Accepted values
            are "standard" and "bootstrap".
        :param <str> bootstrap_method: The method used to modify the coefficients. Accepted values
            are "standard", "bs_panel", "bs_periods", "bs_cross", "bs_cross_per_period" 
            and "bs_period_per_cross".
        :param <int> bootstrap_iters: The number of bootstrap iterations to perform in 
            order to determine the standard errors of the model parameters. 
        :param <Union[float, int]> sample_size_ratio: The ratio of the sample size to be
            used in the bootstrap iterations.

        :return None
        """
        self.model = model
        self.method = method
        self.bootstrap_method = bootstrap_method
        self.bootstrap_iters = bootstrap_iters
        self.sample_size_ratio = sample_size_ratio

        self.check_init_params(
            model=self.model,
            method=self.method,
            bootstrap_method=self.bootstrap_method,
            bootstrap_iters=self.bootstrap_iters,
            sample_size_ratio=self.sample_size_ratio,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ):
        """
        Fit method to fit the underlying linear model, as passed into the constructor, 
        and subsequently modify coefficients based on estimated standard errors. 

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
                self.sample_size_ratio
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
        pass
    
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
        sample_size_ratio: Union[float, int],
    ):
        # TODO
        pass
        
    def check_init_params(
        model: RegressorMixin,
        method: str,
        bootstrap_method: str,
        bootstrap_iters: int,
        sample_size_ratio: Union[float, int],
    ):
        """
        Method to check the validity of the initialization parameters of the class.

        :param <RegressorMixin> model: The underlying linear model to be modified. This
            model must have `coef_` and `intercept_` attributes once fit, in accordance
            with standard `scikit-learn` conventions.
        :param <str> method: The method used to modify the coefficients. Accepted values
            are "standard" and "bootstrap".
        :param <str> bootstrap_method: The method used to modify the coefficients.
            Accepted values are "standard", "bs_panel", "bs_periods", "bs_cross",
            "bs_cross_per_period" and "bs_period_per_cross".
        :param <int> bootstrap_iters: The number of bootstrap iterations to perform in
            order to determine the standard errors of the model parameters.
        :param <Union[float, int]> sample_size_ratio: The ratio of the sample size to be
            used in the bootstrap iterations.

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
        # bootstrap_method
        if method == "bootstrap":
            if not isinstance(bootstrap_method, str):
                raise TypeError("bootstrap_method must be a string.")
            if bootstrap_method not in [
                "standard",
                "bs_panel",
                "bs_periods",
                "bs_cross",
                "bs_cross_per_period",
                "bs_period_per_cross",
            ]:
                raise ValueError(
                    "bootstrap_method must be one of 'standard', 'bs_panel', 'bs_periods', 'bs_cross', 'bs_cross_per_period' or 'bs_period_per_cross'."
                )
        # bootstrap_iters
        if method == "bootstrap":
            if not isinstance(bootstrap_iters, int):
                raise TypeError("bootstrap_iters must be an integer.")
            if bootstrap_iters <= 0:
                raise ValueError("bootstrap_iters must be a positive integer.")
            
        # sample_size_ratio
        if method == "bootstrap":
            if not isinstance(sample_size_ratio, (float, int)):
                raise TypeError("sample_size_ratio must be a float or an integer.")
            if sample_size_ratio <= 0:
                raise ValueError("sample_size_ratio must be greater than 0.")
            if sample_size_ratio > 1:
                raise ValueError("sample_size_ratio must be less than or equal to 1.")
            
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
        