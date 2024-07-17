import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin, BaseEstimator
from base_modified_regressor import BaseModifiedRegressor

from typing import Union, Optional

class ModifiedLinearRegression(BaseModifiedRegressor):
    def __init__(
        self,
        method: str,
        fit_intercept: bool = True,
        positive: bool = False,
        error_offset: float = 1e-2,
        bootstrap_method: str = "panel",
        bootstrap_iters: int = 1000,
        resample_ratio: Union[float, int] = 1,
        analytic_method: Optional[str] = None,
    ):
        """
        Custom class to train an OLS linear regression model with coefficients modified
        by estimated standard errors to account for statistical precision of the estimates.

        :param <str> method: The method used to modify the coefficients. Accepted values
            are "analytic" and "bootstrap".
        :param <bool> fit_intercept: Whether to fit an intercept term in the model.
            Default is True.
        :param <bool> positive: Whether to constrain the coefficients to be positive.
            Default is False.
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
        :param <Optional[str]> analytic_method: The analytic method used to calculate
            standard errors. Expressions for analytic standard errors are expected to be
            written within the method `adjust_analytical_se` and this parameter can be
            passed into `adjust_analyical_se` for an alternative analytic standard error
            estimate, for instance White's estimator. Default value is None.

        :return None
        """
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
        model: RegressorMixin,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        analytic_method: Optional[str],
    ):
        """
        Method to adjust the coefficients of the linear model by an analytical
        standard error expression. The default is to use the standard error estimate
        obtained through assuming multivariate normality of the model errors as well as
        heteroskedasticity and zero mean. If `analytic_method` is "White", the White
        estimator is used to estimate the standard errors.

        :param <RegressorMixin> model: The underlying linear model to be modified. This
            model must have `coef_` and `intercept_` attributes, in accordance with
            standard `scikit-learn` conventions.
        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        :param <Optional[str]> analytic_method: The analytic method used to calculate
            standard errors. If None, the default method is used. Currently, the only
            alternative we offer is White's estimator, which requires "White" to be
            specified. Default value is None.

        :return <float>, <np.ndarray>: The adjusted intercept and coefficients.
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
                    / (X.shape[0] - X.shape[1])
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
        super().set_params(**params)
        if "fit_intercept" in params or "positive" in params:
            # Re-initialize the LinearRegression instance with updated parameters
            self.model = LinearRegression(
                fit_intercept=self.fit_intercept, positive=self.positive
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
            cv=5,
            n_jobs=-1,
        )
        cv.fit(X, y)
        print("----")
        print("Modified OLS grid search results:")
        print(pd.DataFrame(cv.cv_results_))
        print("----")

        # Try with signal optimizer
        inner_splitter = ExpandingKFoldPanelSplit(n_splits=5)
        so = SignalOptimizer(
            inner_splitter=inner_splitter,
            X=X,
            y=y,
        )
        so.calculate_predictions(
            name=f"ModifiedOLS_{method[0]}_{method[1]}",
            models={
                "mlr": model,
            },
            metric=make_scorer(r2_score, greater_is_better=True),
            hparam_grid={
                "mlr": {
                    "fit_intercept": [True, False],
                    "positive": [True, False],
                },
            },
            hparam_type="grid",
            test_size=21 * 12 * 2,
            n_jobs = -1
        )
        so.models_heatmap(f"ModifiedOLS_{method[0]}_{method[1]}")
        so.coefs_stackedbarplot(f"ModifiedOLS_{method[0]}_{method[1]}")
