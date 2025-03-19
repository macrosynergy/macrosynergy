import numpy as np
import pandas as pd
import numbers

from sklearn.base import RegressorMixin, BaseEstimator, clone

from macrosynergy.learning.forecasting.bootstrap import BasePanelBootstrap

from abc import ABC


class BaseModifiedRegressor(BaseEstimator, RegressorMixin, BasePanelBootstrap, ABC):
    def __init__(
        self,
        model,
        method,
        error_offset=1e-5,
        bootstrap_method="panel",
        bootstrap_iters=100,
        resample_ratio=1,
        max_features=None,
        analytic_method=None,
    ):
        """
        Modified linear regression model. Estimated coefficients are divided
        by estimated standard errors to form an auxiliary factor model.

        Parameters
        ----------
        model : RegressorMixin
            Underlying linear regression model to be modified to account
            for statistical precision of parameter estimates. This model must
            have `coef_` and `intercept_` attributes, in accordance with
            `scikit-learn` convention.
        method : str
            Method to modify coefficients. Accepted values are
            "analytic" or "bootstrap".
        error_offset : float, default = 1e-5
            Small offset to add to estimated standard errors in order to prevent
            small denominators during the coefficient adjustment.
        bootstrap_method : str, default = "panel"
            Method used to modify coefficients, when `method = bootstrap`.
            Accepted values are "panel", "period", "cross", "cross_per_period",
            "period_per_cross".
        bootstrap_iters : int, default = 100
            Number of bootstrap iterations to determine standard errors, used
            only when `method = bootstrap`.
        resample_ratio : numbers.Number, default = 1
            Ratio of resampling units in each bootstrap dataset, used only
            when `method = bootstrap`. This is a fraction of the quantity of
            the panel component to be resampled.
        max_features : str or int or float, default = None
            Number of features consider in each bootstrap dataset. This is
            used to increase the amount of variation in bootstrap datasets.
            Accepted values are "sqrt", "log2", an integer number of features and
            a floating point proportion of features. Default behaviour is to raise
            a NotImplementedError.
        analytic_method : str, default = None
            The analytic method used to determine standard errors. This parameter
            is passed into `adjust_analyical_se`, which should be implemented
            by the user if analytical, model-specific, expressions are required.

        Notes
        -----
        Parametric regression models are fit by finding optimal parameters that
        minimize a loss function. In the frequentist statistics framework, "true"
        population-wide values exist for these parameters, which can only be
        estimated from sampled data. Consequently, our parameter estimates can be
        considered to be realizations from a random variable, and hence subject to
        sampling variation. Broadly speaking, the greater the amount of independent data
        sampled, the smaller the variation in parameter estimates. In other words,
        parameter estimates are more unreliable when less data is seen during training.
        By estimating the standard deviation of their sampling distributions - a.k.a.
        their "standard errors" - we can adjust our model coefficients to account for
        lack of statistical precision.

        In our modified parametric regression models, each estimated parameter is
        divided by the estimated standard error (plus an offset). This means that greater
        volatility in a parameter estimate due to lack of data is accounted for by
        reducing the magnitude of this estimate, whilst greater certainty in the precision
        of the estimate is reflected by inflating a regression coefficient.

        Use of this class is only recommended for linear models, since these
        regression models are interpretable and the coefficient adjustment can
        accordingly be interpreted as increasing the relevance of factors whose
        coefficients we are more confident in, and decreasing relevance for factors
        whose coefficients we are less confident in. For a more complex function,
        for instance a neural network, amending model coefficients can be disastrous;
        it would be unclear how such adjustment would affect the downstream performance
        of the model. As a consequence, this class should be used with care
        and we recommend its use for linear models only.
        """
        # Checks
        super().__init__(
            bootstrap_method=bootstrap_method,
            resample_ratio=resample_ratio,
            max_features=max_features,
        )

        self._check_init_params(
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

    def _check_init_params(
        self,
        model,
        method,
        error_offset,
        bootstrap_iters,
        analytic_method,
    ):
        """
        Constructor parameter checks.

        Parameters
        ----------
        model : RegressorMixin
            Underlying linear regression model to be modified to account
            for statistical precision of parameter estimates. This model must
            have `coef_` and `intercept_` attributes, in accordance with
            `scikit-learn` convention.
        method : str
            Method to modify coefficients. Accepted values are
            "analytic" or "bootstrap".
        error_offset : float, default = 1e-5
            Small offset to add to estimated standard errors in order to prevent
            small denominators during the coefficient adjustment.
        bootstrap_iters : int, default = 100
            Number of bootstrap iterations to determine standard errors, used
            only when `method = bootstrap`.
        analytic_method : str, default = None
            The analytic method used to determine standard errors. This parameter
            is passed into `adjust_analyical_se`, which should be implemented
            by the user if analytical, model-specific, expressions are required.
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
        if not isinstance(error_offset, numbers.Number):
            raise TypeError("error_offset must be a float or an integer.")
        if error_offset <= 0:
            raise ValueError("error_offset must be greater than 0.")

        # bootstrap_iters
        if method == "bootstrap":
            if not isinstance(bootstrap_iters, numbers.Integral):
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
        X,
        y,
    ):
        """
        Fit a linear model and modify coefficients based on standard errors.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Target vector associated with each sample in X.

        Returns
        -------
        self
            Fitted estimator.
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
        X,
    ):
        """
        Predict using the unadjusted linear model.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        np.ndarray or pd.Series
            Predicted values.
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
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise TypeError("All columns in X must be numeric.")
        if X.isnull().values.any():
            raise ValueError("X must not contain missing values.")
        return self.model.predict(X)

    def create_signal(
        self,
        X,
    ):
        """
        Predict using the coefficient-adjusted linear model.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        np.ndarray or pd.Series
            Signal from the adjusted factor model based on X.

        Notes
        -----
        We define an additional `create_signal` method instead of using the
        `predict` method in order to not interfere with hyperparameter
        searches with standard metrics. Moreover, outputs from the adjusted
        factor model are not valid predictions, but are valid trading signals.
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
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise TypeError("All columns in X must be numeric.")
        if X.isnull().values.any():
            raise ValueError("X must not contain missing values.")

        return np.dot(X, self.coef_) + self.intercept_

    def adjust_bootstrap_se(
        self,
        model,
        X,
        y,
    ):
        """
        Adjust the coefficients of the linear model by bootstrap standard errors.

        Parameters
        ----------
        model : RegressorMixin
            The underlying linear model to be modified.
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Target vector associated with each sample in X.

        Returns
        -------
        intercept : float
            Adjusted intercept.
        coef : np.ndarray
            Adjusted coefficients.
        """
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
        model,
        X,
        y,
        analytic_method,
    ):
        """
        Adjust the coefficients of the linear model by an analytical
        standard error formula.

        Parameters
        ----------
        model : RegressorMixin
            The underlying linear model to be modified.
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Target vector associated with each sample in X.
        analytic_method : str
            The analytic method used to calculate standard errors.

        Returns
        -------
        intercept : float
            Adjusted intercept.
        coef : np.ndarray
            Adjusted coefficients.

        Notes
        -----
        Analytical standard errors are model-specific, meaning that
        they must be implemented in a subclass of BaseModifiedRegressor.
        """
        raise NotImplementedError(
            "Analytical standard error adjustments are not available for most models."
            "This function must be implemented in a subclass of BaseModifiedRegressor "
            "if known standard error expressions are available."
        )

    def _check_fit_params(
        self,
        X,
        y,
    ):
        """
        Check parameter validity for the fit method.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Target vector associated with each sample in X.
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
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not y.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not y.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't "
                "match."
            )

        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise TypeError("All columns in X must be numeric.")
        if isinstance(y, pd.DataFrame):
            if not pd.api.types.is_numeric_dtype(y.iloc[:, 0]):
                raise TypeError("All columns in y must be numeric.")
        else:
            if not pd.api.types.is_numeric_dtype(y):
                raise TypeError("All columns in y must be numeric.")
        if X.isnull().values.any():
            raise ValueError("X must not contain missing values.")
        if y.isnull().values.any():
            raise ValueError("y must not contain missing values.")
