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
        fit_intercept=True,
        positive=False,
        error_offset=1e-2,
        bootstrap_method="panel",
        bootstrap_iters=1000,
        resample_ratio=1,
        analytic_method=None,
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

        Notes
        -----
        OLS linear regression models are fit by finding optimal parameters that
        minimize the total squared residuals. In the frequentist statistics framework,
        "true" population-wide values exist for these parameters, which can only be
        estimated from sampled data. Consequently, our parameter estimates can be
        considered to be realizations from a random variable, and hence subject to
        sampling variation. Broadly speaking, the greater the amount of independent data
        sampled, the smaller the variation in parameter estimates. In other words,
        parameter estimates are more unreliable when less data is seen during training.
        By estimating the standard deviation of their sampling distributions - a.k.a.
        their "standard errors" - we can adjust our model coefficients to account for
        lack of statistical precision.

        In our `ModifiedLinearRegression`, each estimated parameter is divided by the
        estimated standard error (plus an offset). This means that greater
        volatility in a parameter estimate due to lack of data is accounted for by
        reducing the magnitude of this estimate, whilst greater certainty in the precision
        of the estimate is reflected by inflating a regression coefficient.

        This procedure works for linear models because they're interpretable, with the
        coefficient adjustment having the interpretation of increasing the relevance of
        factors whose coefficients we are more confident in, and decreasing relevance for
        factors whose coefficients we are less confident in.
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
        analytic_method=None,
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

        Returns
        -------
        self
            The ModifiedLinearRegression instance with updated parameters.
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
        fit_intercept=True,
        positive=False,
        error_offset=1e-2,
        bootstrap_method="panel",
        bootstrap_iters=1000,
        resample_ratio=1,
        analytic_method=None,
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

        Notes
        -----
        SWLS linear regression models are fit by finding optimal parameters that
        minimize the total sign-weighted squared residuals. In the frequentist statistics
        framework, "true" population-wide values exist for these parameters, which can
        only be estimated from sampled data. Consequently, our parameter estimates can be
        considered to be realizations from a random variable, and hence subject to
        sampling variation. Broadly speaking, the greater the amount of independent data
        sampled, the smaller the variation in parameter estimates. In other words,
        parameter estimates are more unreliable when less data is seen during training.
        By estimating the standard deviation of their sampling distributions - a.k.a.
        their "standard errors" - we can adjust our model coefficients to account for
        lack of statistical precision.

        In our `ModifiedSignWeightedLinearRegression`, each estimated parameter is divided
        by the estimated standard error (plus an offset). This means that greater
        volatility in a parameter estimate due to lack of data is accounted for by
        reducing the magnitude of this estimate, whilst greater certainty in the precision
        of the estimate is reflected by inflating a regression coefficient.

        This procedure works for linear models because they're interpretable, with the
        coefficient adjustment having the interpretation of increasing the relevance of
        factors whose coefficients we are more confident in, and decreasing relevance for
        factors whose coefficients we are less confident in.
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
        analytic_method=None,
    ):
        r"""
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

        Notes
        ----------
        The analytical parameter estimates for WLS are:

        .. math::

            \hat{\beta}^{\text{WLS}} = (X^{\intercal}WX)^{-1}X^{\intercal}y

        where:
            - `X` is the input feature matrix, possibly with a column of ones representing the choice of an intercept.
            - `W` is the positive-definite, symmetric weight matrix, a diagonal matrix with sample weights along the main diagonal.
            - `y` is the dependent variable vector.

        Since `W` is a positive-definite, symmetric matrix, it has a square root
        equal to the diagonal matrix with square roots of the sample weights along
        the diagonal. Hence, the WLS estimator can be rewritten as:

        .. math::

            \hat{\beta}^{\text{WLS}} = ((({W^{1/2}X})^{\intercal}(W^{1/2}X))^{-1}(W^{1/2}X)^{\intercal}(W^{1/2}y))

        This is precisely the OLS estimator for a rescaled matrix

        .. math::

            \tilde {X} = W^{1/2}X

        and a rescaled dependent variable

        .. math::

            \tilde {y} = W^{1/2}y

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

        Returns
        -------
        self
            The ModifiedSignWeightedLinearRegression instance with updated parameters.
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
        fit_intercept=True,
        positive=False,
        half_life=252,
        error_offset=1e-2,
        bootstrap_method="panel",
        bootstrap_iters=1000,
        resample_ratio=1,
        analytic_method=None,
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

        Notes
        -----
        TWLS linear regression models are fit by finding optimal parameters that
        minimize the total time-weighted squared residuals. In the frequentist statistics
        framework, "true" population-wide values exist for these parameters, which can
        only be estimated from sampled data. Consequently, our parameter estimates can be
        considered to be realizations from a random variable, and hence subject to
        sampling variation. Broadly speaking, the greater the amount of independent data
        sampled, the smaller the variation in parameter estimates. In other words,
        parameter estimates are more unreliable when less data is seen during training.
        By estimating the standard deviation of their sampling distributions - a.k.a.
        their "standard errors" - we can adjust our model coefficients to account for
        lack of statistical precision.

        In our `ModifiedTimeWeightedLinearRegression`, each estimated parameter is divided
        by the estimated standard error (plus an offset). This means that greater
        volatility in a parameter estimate due to lack of data is accounted for by
        reducing the magnitude of this estimate, whilst greater certainty in the precision
        of the estimate is reflected by inflating a regression coefficient.

        This procedure works for linear models because they're interpretable, with the
        coefficient adjustment having the interpretation of increasing the relevance of
        factors whose coefficients we are more confident in, and decreasing relevance for
        factors whose coefficients we are less confident in.
        """
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.half_life = half_life

        super().__init__(
            model=TimeWeightedLinearRegression(
                fit_intercept=self.fit_intercept,
                positive=self.positive,
                half_life=self.half_life,
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
        r"""
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

        Notes
        ----------
        The analytical parameter estimates for WLS are:

        .. math::

            \hat{\beta}^{\text{WLS}} = (X^{\intercal}WX)^{-1}X^{\intercal}y

        where:
            - `X` is the input feature matrix, possibly with a column of ones representing the choice of an intercept.
            - `W` is the positive-definite, symmetric weight matrix, a diagonal matrix with sample weights along the main diagonal.
            - `y` is the dependent variable vector.

        Since `W` is a positive-definite, symmetric matrix, it has a square root
        equal to the diagonal matrix with square roots of the sample weights along
        the diagonal. Hence, the WLS estimator can be rewritten as:

        .. math::

            \hat{\beta}^{\text{WLS}} = (({W^{1/2}X})^{\intercal}(W^{1/2}X))^{-1}(W^{1/2}X)^{\intercal}(W^{1/2}y))

        This is precisely the OLS estimator for a rescaled matrix

        .. math::
        
            \tilde {X} = W^{1/2}X

        and a rescaled dependent variable

        .. math::
        
            \tilde {y} = W^{1/2}y

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

        Returns
        -------
        self
            The ModifiedTimeWeightedLinearRegression instance with updated parameters.
        """
        super().set_params(**params)
        if "fit_intercept" in params or "positive" in params or "half_life" in params:
            # Re-initialize the SignWeightedLinearRegression instance with updated parameters
            self.model = TimeWeightedLinearRegression(
                fit_intercept=self.fit_intercept,
                positive=self.positive,
                half_life=self.half_life,
            )

        return self


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm
    from macrosynergy.learning import SignalOptimizer, ExpandingKFoldPanelSplit

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, r2_score

    # Randomly generate an unbalanced panel dataset, multi-indexed by cross-section and
    # real_date

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0, 1, 0, 3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 0, 1, 0, 0]
    df_xcats.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 0, 1, -0.9, 0]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", 0, 1, 0.8, 0]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd = msm.reduce_df(df=dfd, cids=cids, xcats=xcats, blacklist=black)

    dfd = dfd.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd.drop(columns=["XR"])
    y = dfd["XR"]

    # First instantiate the BaseModifiedRegressor
    obj = BaseModifiedRegressor(model=LinearRegression(), method="bootstrap")

    # Demonstrate ModifiedLinearRegression usage
    method_pairs = [
        ("analytic", "panel", None),
        ("analytic", "panel", "White"),
        ("bootstrap", "panel", None),
        ("bootstrap", "period", None),
        ("bootstrap", "cross", None),
        ("bootstrap", "cross_per_period", None),
        ("bootstrap", "period_per_cross", None),
    ]
    for method in method_pairs:
        model = ModifiedLinearRegression(
            method=method[0],
            bootstrap_method=method[1],
            bootstrap_iters=100,
            resample_ratio=0.75,
            analytic_method=method[2],
        )
        # Fit the model
        model.fit(X, y)
        print("----")
        print("Modified OLS method:", method)
        print("Modified OLS intercept:", model.intercept_)
        print("Modified OLS coefficients:", model.coef_)
        print("Modified OLS predictions:", model.predict(X))
        print("Modified OLS signal:", model.create_signal(X))
        print("----")

        # Grid search
        cv = GridSearchCV(
            estimator=model,
            param_grid={
                "fit_intercept": [True, False],
                "positive": [True, False],
            },
            cv=3,
            n_jobs=-1,
        )
        cv.fit(X, y)
        print("----")
        print("Modified OLS grid search results:")
        print(pd.DataFrame(cv.cv_results_))
        print("----")
