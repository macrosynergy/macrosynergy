import numbers
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from macrosynergy.learning.forecasting.bootstrap import (
    BaseModifiedRegressor,
)
from macrosynergy.learning.forecasting.linear_model.ls_regressors.weighted_ls_regressors import (
    SignWeightedLinearRegression,
    TimeWeightedLinearRegression,
)

class ModifiedLinearRegression(BaseModifiedRegressor):
    def __init__(
        self,
        method,
        fit_intercept = True,
        positive = False,
        error_offset = 1e-2,
        bootstrap_method = "panel",
        bootstrap_iters = 1000,
        resample_ratio = 1,
        analytic_method = None,
    ):
        """
        Modified OLS linear regression model. Estimated coefficients are divided
        by estimated standard errors to form an auxiliary factor model.
        
        Parameters
        ----------
        method : str
            Method to modify coefficients. Accepted values are
            "analytic" or "bootstrap".
        fit_intercept : bool, default = True
            Whether to fit an intercept term in the model. Default is True.
        positive : bool, default = False
            Whether to constrain the coefficients to be positive. Default is False.
        error_offset : float, default = 1e-2
            Small offset to add to estimated standard errors in order to prevent
            small denominators during the coefficient adjustment. 
        bootstrap_method : str, default = "panel"
            Method used to modify coefficients, when `method = bootstrap`.
            Accepted values are "panel", "period", "cross", "cross_per_period",
            "period_per_cross". 
        bootstrap_iters : int, default = 1000
            Number of bootstrap iterations to determine standard errors, used 
            only when `method = bootstrap`.
        resample_ratio : numbers.Number, default = 1
            Ratio of resampling units in each bootstrap dataset, used only
            when `method = bootstrap`. This is a fraction of the quantity of
            the panel component to be resampled.
        analytic_method : str, default = None
            The analytic method used to determine standard errors. If `method = analytic`,
            the default standard error expressions for an OLS linear regression model 
            are used. If `analytic_method = "White"`, the heteroskedasticity-robust
            White estimator is used to estimate the standard errors. 
        """
        # Checks
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.") 
        self.fit_intercept = fit_intercept
        self.positive = positive

        super().__init__(
            model=LinearRegression(
                fit_intercept=self.fit_intercept, positive=self.positive
            ),
            method=method,
            error_offset=error_offset,
            bootstrap_method=bootstrap_method,
            bootstrap_iters=bootstrap_iters,
            resample_ratio=resample_ratio,
            analytic_method=analytic_method,
        )

    def adjust_analytical_se(
        self,
        model,
        X,
        y,
        analytic_method = None,
    ):
        """
        Adjust the coefficients of the OLS linear regression model
        by an analytical standard error formula.

        Parameters
        ----------
        model : LinearRegression
            The underlying OLS linear regression model to be modified.
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Target vector associated with each sample in X.
        analytic_method : str, default = None
            The analytic method used to calculate standard errors.

        Returns
        -------
        intercept : float
            Adjusted intercept.
        coef : np.ndarray
            Adjusted coefficients.

        Notes
        -----
        By default, the calculated standard errors use the usual standard 
        error expression for OLS linear regression models under the assumption
        of multivariate normality, homoskedasticity and zero mean of the model errors.
        If `analytic_method = "White"`, the HC3 White estimator is used. 

        References
        ----------
        [1] https://online.stat.psu.edu/stat462/node/131/ 
        [2] https://en.wikipedia.org/wiki/Heteroskedasticity-consistent_standard_errors
        """
        # Checks
        if analytic_method is not None:
            if not isinstance(analytic_method, str):
                raise TypeError("analytic_method must be a string.")
            if analytic_method not in ["White"]:
                raise ValueError("analytic_method must be 'White'.")

        if self.fit_intercept:
            X_new = np.column_stack((np.ones(len(X)), X.values))
        else:
            X_new = X.values

        # Calculate the standard errors
        predictions = model.predict(X)
        residuals = (y - predictions).to_numpy()
        XtX_inv = np.linalg.inv(X_new.T @ X_new)
        if analytic_method is None:
            se = np.sqrt(
                np.diag(
                    XtX_inv
                    * np.sum(np.square(residuals))
                    / (X_new.shape[0] - X_new.shape[1])
                )
            )

        elif analytic_method == "White":
            # Implement HC3
            leverages = np.sum((X_new @ XtX_inv) * X_new, axis=1)
            weights = 1 / (1 - leverages) ** 2
            residuals_squared = np.square(residuals)
            weighted_residuals_squared = weights * residuals_squared
            Omega = X_new.T * weighted_residuals_squared @ X_new
            cov_matrix = XtX_inv @ Omega @ XtX_inv
            se = np.sqrt(np.diag(cov_matrix))

        else:
            raise NotImplementedError(
                "Currently, only the standard and HC3 White standard errors are implemented"
            )

        if self.fit_intercept:
            coef_se = se[1:]
            intercept_se = se[0]
        else:
            coef_se = se
            intercept_se = 0

        # Adjust the coefficients and intercepts by the standard errors
        coef = model.coef_ / (coef_se + self.error_offset)
                              
        intercept = model.intercept_ / (intercept_se + self.error_offset)

        return intercept, coef

    def set_params(self, **params):
        """
        Setter method to update the parameters of the ModifiedLinearRegression

        Parameters
        ----------
        **params : dict
            Dictionary of parameters to update.
        """
        super().set_params(**params)
        if "fit_intercept" in params or "positive" in params:
            # Re-initialize the LinearRegression instance with updated parameters
            self.model = LinearRegression(
                fit_intercept=self.fit_intercept, positive=self.positive
            )

        return self
    
class ModifiedSignWeightedLinearRegression(BaseModifiedRegressor):
    def __init__(
        self,
        method,
        fit_intercept = True,
        positive = False,
        error_offset = 1e-2,
        bootstrap_method = "panel",
        bootstrap_iters = 1000,
        resample_ratio = 1,
        analytic_method = None,
    ):
        """
        Modified SWLS linear regression model. Estimated coefficients are divided
        by estimated standard errors to form an auxiliary factor model.

        Parameters
        ----------
        method : str
            Method to modify coefficients. Accepted values are
            "analytic" or "bootstrap".
        fit_intercept : bool, default = True
            Whether to fit an intercept term in the model. Default is True.
        positive : bool, default = False
            Whether to constrain the coefficients to be positive. Default is False.
        error_offset : float, default = 1e-2
            Small offset to add to estimated standard errors in order to prevent
            small denominators during the coefficient adjustment. 
        bootstrap_method : str, default = "panel"
            Method used to modify coefficients, when `method = bootstrap`.
            Accepted values are "panel", "period", "cross", "cross_per_period",
            "period_per_cross". 
        bootstrap_iters : int, default = 1000
            Number of bootstrap iterations to determine standard errors, used 
            only when `method = bootstrap`.
        resample_ratio : numbers.Number, default = 1
            Ratio of resampling units in each bootstrap dataset, used only
            when `method = bootstrap`. This is a fraction of the quantity of
            the panel component to be resampled.
        analytic_method : str, default = None
            The analytic method used to determine standard errors. If `method = analytic`,
            the default standard error expressions for an OLS linear regression model 
            are used. If `analytic_method = "White"`, the heteroskedasticity-robust
            White estimator is used to estimate the standard errors. 
        """
        self.fit_intercept = fit_intercept
        self.positive = positive

        super().__init__(
            model=SignWeightedLinearRegression(
                fit_intercept=self.fit_intercept, positive=self.positive
            ),
            method=method,
            error_offset=error_offset,
            bootstrap_method=bootstrap_method,
            bootstrap_iters=bootstrap_iters,
            resample_ratio=resample_ratio,
            analytic_method=analytic_method,
        )

    def adjust_analytical_se(
        self,
        model,
        X,
        y,
        analytic_method = None,
    ):
        """
        Adjust the coefficients of the SWLS linear regression model
        by an analytical standard error formula.

        Parameters
        ----------
        model : SignWeightedLinearRegression
            The underlying SWLS linear regression model to be modified.
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Target vector associated with each sample in X.
        analytic_method : str, default = None
            The analytic method used to calculate standard errors.

        Returns
        -------
        intercept : float
            Adjusted intercept.
        coef : np.ndarray
            Adjusted coefficients.

        Parameters
        ----------
        The analytical parameter estimates for WLS are:

        .. math::

        \hat{\Beta}^{\text{WLS}} = (X^{\intercal}WX)^{-1}X^{\intercal}y

        where:
            - :math: `X` is the input feature matrix, possibly with a column of
                ones representing the choice of an intercept.
            - :math: `W` is the positive-definite, symmetric weight matrix, a 
                diagonal matrix with sample weights along the main diagonal.
            - :math: `y` is the dependent variable vector. 

        Since `W` is a positive-definite, symmetric matrix, it has a square root
        equal to the diagonal matrix with square roots of the sample weights along
        the diagonal. Hence, the WLS estimator can be rewritten as:

        .. math::

        \hat{\Beta}^{\text{WLS}} = ((W^{\frac{1}{2}}X}^{\intercal}(W^{\frac{1}{2}}X))^{-1}(W^{\frac{1}{2}}X)^{\intercal}(W^{\frac{1}{2}}y)

        This is precisely the OLS estimator for a rescaled matrix :math: `\sim{X} = W^{\frac{1}{2}}X`
        and a rescaled dependent variable :math: `\sim{y} = W^{\frac{1}{2}}y`.
        Hence, the usual standard error estimate and White's estimator can be applied
        based on a rescaling of the design matrix and associated target vector.
        """
        # Checks
        if analytic_method is not None:
            if not isinstance(analytic_method, str):
                raise TypeError("analytic_method must be a string.")
            if analytic_method not in ["White"]:
                raise ValueError("analytic_method must be 'White'.")
            
        if self.fit_intercept:
            X_new = np.column_stack((np.ones(len(X)), X.values))
        else:
            X_new = X.values

        # Get model weights
        weights = model.sample_weights
        # Rescale features and targets by the sign-weighted linear regression sample weights
        X_new = np.sqrt(weights[:, np.newaxis]) * X_new
        y_new = np.sqrt(weights) * y

        # Calculate the standard errors
        predictions = model.predict(X)
        residuals = (y - predictions).to_numpy() * np.sqrt(weights)
        XtX_inv = np.linalg.inv(X_new.T @ X_new)
        if analytic_method is None:
            se = np.sqrt(
                np.diag(
                    XtX_inv
                    * np.sum(np.square(residuals))
                    / (X_new.shape[0] - X_new.shape[1])
                )
            )

        elif analytic_method == "White":
            # Implement HC3
            leverages = np.sum((X_new @ XtX_inv) * X_new, axis=1)
            weights = 1 / (1 - leverages) ** 2
            residuals_squared = np.square(residuals)
            weighted_residuals_squared = weights * residuals_squared
            Omega = X_new.T * weighted_residuals_squared @ X_new
            cov_matrix = XtX_inv @ Omega @ XtX_inv
            se = np.sqrt(np.diag(cov_matrix))

        else:
            raise NotImplementedError(
                "Currently, only the standard and White standard errors are implemented"
            )
        
        if self.fit_intercept:
            coef_se = se[1:]
            intercept_se = se[0]
        else:
            coef_se = se
            intercept_se = 0

        # Adjust the coefficients and intercepts by the standard errors
        coef = model.coef_ / (coef_se + self.error_offset)
        intercept = model.intercept_ / (intercept_se + self.error_offset)

        return intercept, coef

    def set_params(self, **params):
        """
        Setter method to update the parameters of the
        ModifiedSignWeightedLinearRegression.

        Parameters
        ----------
        **params : dict
            Dictionary of parameters to update.
        """
        super().set_params(**params)
        if "fit_intercept" in params or "positive" in params:
            # Re-initialize the SignWeightedLinearRegression instance with updated parameters
            self.model = SignWeightedLinearRegression(
                fit_intercept=self.fit_intercept, positive=self.positive
            )

        return self
    
class ModifiedTimeWeightedLinearRegression(BaseModifiedRegressor):
    def __init__(
        self,
        method,
        fit_intercept = True,
        positive = False,
        half_life = 252,
        error_offset = 1e-2,
        bootstrap_method = "panel",
        bootstrap_iters = 1000,
        resample_ratio = 1,
        analytic_method = None,
    ):
        """
        Modified TWLS linear regression model. Estimated coefficients are divided
        by estimated standard errors to form an auxiliary factor model.

        Parameters
        ----------
        method : str
            Method to modify coefficients. Accepted values are
            "analytic" or "bootstrap".
        fit_intercept : bool, default = True
            Whether to fit an intercept term in the model. Default is True.
        positive : bool, default = False
            Whether to constrain the coefficients to be positive. Default is False.
        half_life : int, default = 252
            The half-life of the exponential weighting function
            used to calculate the sample weights.
        error_offset : float, default = 1e-2
            Small offset to add to estimated standard errors in order to prevent
            small denominators during the coefficient adjustment. 
        bootstrap_method : str, default = "panel"
            Method used to modify coefficients, when `method = bootstrap`.
            Accepted values are "panel", "period", "cross", "cross_per_period",
            "period_per_cross". 
        bootstrap_iters : int, default = 1000
            Number of bootstrap iterations to determine standard errors, used 
            only when `method = bootstrap`.
        resample_ratio : numbers.Number, default = 1
            Ratio of resampling units in each bootstrap dataset, used only
            when `method = bootstrap`. This is a fraction of the quantity of
            the panel component to be resampled.
        analytic_method : str, default = None
            The analytic method used to determine standard errors. If `method = analytic`,
            the default standard error expressions for an OLS linear regression model 
            are used. If `analytic_method = "White"`, the heteroskedasticity-robust
            White estimator is used to estimate the standard errors. 
        """
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.half_life = half_life

        super().__init__(
            model=TimeWeightedLinearRegression(
                fit_intercept=self.fit_intercept, positive=self.positive, half_life=self.half_life
            ),
            method=method,
            error_offset=error_offset,
            bootstrap_method=bootstrap_method,
            bootstrap_iters=bootstrap_iters,
            resample_ratio=resample_ratio,
            analytic_method=analytic_method,
        )

    def adjust_analytical_se(
        self,
        model,
        X,
        y,
        analytic_method,
    ):
        """
        Adjust the coefficients of the TWLS linear regression model
        by an analytical standard error formula.

        Parameters
        ----------
        model : TimeWeightedLinearRegression
            The underlying TWLS linear regression model to be modified.
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Target vector associated with each sample in X.
        analytic_method : str, default = None
            The analytic method used to calculate standard errors.

        Returns
        -------
        intercept : float
            Adjusted intercept.
        coef : np.ndarray
            Adjusted coefficients.

        Parameters
        ----------
        The analytical parameter estimates for WLS are:

        .. math::

        \hat{\Beta}^{\text{WLS}} = (X^{\intercal}WX)^{-1}X^{\intercal}y

        where:
            - :math: `X` is the input feature matrix, possibly with a column of
                ones representing the choice of an intercept.
            - :math: `W` is the positive-definite, symmetric weight matrix, a 
                diagonal matrix with sample weights along the main diagonal.
            - :math: `y` is the dependent variable vector. 

        Since `W` is a positive-definite, symmetric matrix, it has a square root
        equal to the diagonal matrix with square roots of the sample weights along
        the diagonal. Hence, the WLS estimator can be rewritten as:

        .. math::

        \hat{\Beta}^{\text{WLS}} = ((W^{\frac{1}{2}}X}^{\intercal}(W^{\frac{1}{2}}X))^{-1}(W^{\frac{1}{2}}X)^{\intercal}(W^{\frac{1}{2}}y)

        This is precisely the OLS estimator for a rescaled matrix :math: `\sim{X} = W^{\frac{1}{2}}X`
        and a rescaled dependent variable :math: `\sim{y} = W^{\frac{1}{2}}y`.
        Hence, the usual standard error estimate and White's estimator can be applied
        based on a rescaling of the design matrix and associated target vector.
        """
        # Checks
        if analytic_method is not None:
            if not isinstance(analytic_method, str):
                raise TypeError("analytic_method must be a string.")
            if analytic_method not in ["White"]:
                raise ValueError("analytic_method must be 'White'.")
            
        if self.fit_intercept:
            X_new = np.column_stack((np.ones(len(X)), X.values))
        else:
            X_new = X.values

        # Get model weights
        weights = model.sample_weights
        # Rescale features and targets by the sign-weighted linear regression sample weights
        X_new = np.sqrt(weights[:, np.newaxis]) * X_new
        y_new = np.sqrt(weights) * y

        # Calculate the standard errors
        predictions = model.predict(X)
        residuals = (y - predictions).to_numpy() * np.sqrt(weights)
        XtX_inv = np.linalg.inv(X_new.T @ X_new)
        if analytic_method is None:
            se = np.sqrt(
                np.diag(
                    XtX_inv
                    * np.sum(np.square(residuals))
                    / (X_new.shape[0] - X_new.shape[1])
                )
            )

        elif analytic_method == "White":
            # Implement HC3
            leverages = np.sum((X_new @ XtX_inv) * X_new, axis=1)
            weights = 1 / (1 - leverages) ** 2
            residuals_squared = np.square(residuals)
            weighted_residuals_squared = weights * residuals_squared
            Omega = X_new.T * weighted_residuals_squared @ X_new
            cov_matrix = XtX_inv @ Omega @ XtX_inv
            se = np.sqrt(np.diag(cov_matrix))

        else:
            raise NotImplementedError(
                "Currently, only the standard and White standard errors are implemented"
            )
        
        if self.fit_intercept:
            coef_se = se[1:]
            intercept_se = se[0]
        else:
            coef_se = se
            intercept_se = 0

        # Adjust the coefficients and intercepts by the standard errors
        coef = model.coef_ / (coef_se + self.error_offset)
        intercept = model.intercept_ / (intercept_se + self.error_offset)

        return intercept, coef

    def set_params(self, **params):
        """
        Setter method to update the parameters of the
        ModifiedTimeWeightedLinearRegression.

        Parameters
        ----------
        **params : dict
            Dictionary of parameters to update.
        """
        super().set_params(**params)
        if "fit_intercept" in params or "positive" in params or "half_life" in params:
            # Re-initialize the SignWeightedLinearRegression instance with updated parameters
            self.model = TimeWeightedLinearRegression(
                fit_intercept=self.fit_intercept, positive=self.positive, half_life=self.half_life
            )

        return self