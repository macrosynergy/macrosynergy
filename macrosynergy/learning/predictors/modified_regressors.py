import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from typing import Union
from abc import ABC, abstractmethod

import datetime

from collections import Counter, defaultdict


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
            raise AttributeError(
                "The underlying model must have an `intercept_` attribute."
            )

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
                self.resample_ratio,
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

        index_array = np.array(X.index.tolist())
        level_0_values = index_array[:, 0]
        level_1_values = index_array[:, 1]

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
                bootstrap_periods = np.random.choice(
                    level_1_values,
                    size=int(len(level_1_values) * resample_ratio),
                    replace=True,
                )
                period_counts = dict(Counter(bootstrap_periods))
                count_to_periods = defaultdict(list)
                for period, count in period_counts.items():
                    count_to_periods[count].append(period)

                X_resampled = np.empty((0, X.shape[1]))
                y_resampled = np.empty(0)
                for count, periods in count_to_periods.items():
                    X_resampled = np.vstack([
                        X_resampled,
                        np.tile(
                            X[X.index.get_level_values(1).isin(periods)].values,
                            (count, 1)
                        ),
                    ])
                    y_resampled = np.append(
                        y_resampled,
                        np.tile(
                            y[y.index.get_level_values(1).isin(periods)].values,
                            count
                        ),
                    )
                #unique_counts = []
                #indices = []#

                #for count in unique_counts:
                #    periods_with_count_c = [period for period, c in counter.items() if c == count]
                #    for period in periods_with_count_c:
                #        period_indices = np.where(level_1_values == period)[0]
                #        indices.append(np.random.choice(period_indices, count))
                #for period, count in counter.items():
                #    period_indices = np.where(level_1_values == period)[0]
                #    indices.append(np.tile(period_indices, count))


                #X_resampled = X[X.index.get_level_values(1).isin(bootstrap_periods)].values
                #y_resampled = y[y.index.get_level_values(1).isin(bootstrap_periods)].values

                #repeated_indices = np.hstack([
                #    np.where(level_1_values == period)[0] for period in bootstrap_periods
                #])


                #period_counts = Counter(bootstrap_periods)

                #indices = np.hstack([
                #    np.repeat(np.where(level_1_values == period)[0], count)
                #    for period, count in period_counts.items()
                #])

                #X_resampled = X.values[indices, :]
                #y_resampled = y.values[indices]
                # now get samples from X and y within those periods
                #X_resampled = np.empty((0, X.shape[1]))
                #y_resampled = np.empty(0)

                #for period in bootstrap_periods.unique():

                #    period_indices = np.where(level_1_values == period)[0]
                #    X_resampled = np.vstack((X_resampled, X.values[period_indices, :]))
                #    y_resampled = np.append(y_resampled, y.values[period_indices])

            elif bootstrap_method == "cross":
                # Resample the unique cross sections from the panel
                # and select all observations within those cross sections
                bootstrap_cross_sections = np.random.choice(
                    unique_cross_sections,
                    size=int(len(unique_cross_sections) * resample_ratio),
                    replace=True,
                )
                # now get samples from X and y within those cross sections
                indices = []
                for cross_section in bootstrap_cross_sections:
                    cross_section_indices = X.index[
                        X.index.get_level_values(0) == cross_section
                    ]
                    indices.extend(cross_section_indices.tolist())

                bootstrap_idx = pd.Index(indices)

                X_resampled = X.loc[bootstrap_idx]
                y_resampled = y.loc[bootstrap_idx]

            elif bootstrap_method == "cross_per_period":
                # Resample observations within each unique time period
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
                indices = []

                for cross_section in unique_cross_sections:
                    cross_section_indices = X.index[
                        X.index.get_level_values(0) == cross_section
                    ]
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
        intercept_se = np.std(bootstrap_intercepts, ddof=0)

        # Adjust the coefficients and intercepts by the standard errors
        coef = model.coef_ / (coef_se + self.error_offset)
        intercept = model.intercept_ / (intercept_se + self.error_offset)

        return intercept, coef

    def check_init_params(
        self,
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
    ):
        """
        Custom class to train an OLS linear regression model with coefficients modified 
        by estimated standard errors to account for statistical precision of the estimates.
        
        :param <str> method: The method used to modify the coefficients. Accepted values
            are "standard" and "bootstrap".
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

        :return None
        """
        self.fit_intercept = fit_intercept
        self.positive = positive

        super().__init__(
            model=LinearRegression(fit_intercept=self.fit_intercept, positive=self.positive),
            method=method,
            error_offset=error_offset,
            bootstrap_method=bootstrap_method,
            bootstrap_iters=bootstrap_iters,
            resample_ratio=resample_ratio,
        )

    def adjust_analytical_se(
        self,
        model: RegressorMixin,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ):
        """
        Method to adjust the coefficients of the linear model by the analytical
        standard error expression obtain through assuming multivariate normality of the
        model errors.

        :param <RegressorMixin> model: The underlying linear model to be modified. This
            model must have `coef_` and `intercept_` attributes, in accordance with
            standard `scikit-learn` conventions.
        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.

        :return <float>, <np.ndarray>: The adjusted intercept and coefficients.
        """
        # Calculate the standard errors
        coef_se = np.sqrt(
            np.diag(
                np.linalg.inv(np.dot(X.T, X))
                * np.sum(np.square(y - model.predict(X)))
                / (X.shape[0] - X.shape[1])
            )
        )
        intercept_se = np.sqrt(
            np.sum(np.square(y - model.predict(X))) / (X.shape[0] - X.shape[1])
        )

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

    """ Linear regression with analytical standard error adjustment """

    """model = ModifiedLinearRegression(method="standard")
    model.fit(X, y)
    print("Modified OLS intercept:", model.intercept_)
    print("Modified OLS coefficients:", model.coef_)
    print("Modified OLS signal:", model.predict(X))

    # Grid search for a modified linear regression with analytical standard error adjustment
    cv = GridSearchCV(
        estimator=ModifiedLinearRegression(method="standard"),
        param_grid={
            "fit_intercept": [True, False],
            "positive": [True, False],
        },
        cv=5,
        n_jobs=-1,
    )
    cv.fit(X, y)
    print("Modified OLS grid search results:")
    print("-----------------------------")
    print(pd.DataFrame(cv.cv_results_))
    print("-----------------------------")

    # Try with signal optimizer
    inner_splitter = ExpandingKFoldPanelSplit(n_splits = 5)
    so = SignalOptimizer(
        inner_splitter=inner_splitter,
        X = X,
        y = y,
    )
    so.calculate_predictions(
        name="ModifiedOLS_analytic",
        models = {
            "mlr": ModifiedLinearRegression(method="standard"),
        },
        metric = make_scorer(r2_score, greater_is_better=True),
        hparam_grid= {
            "mlr": {
                "fit_intercept": [True, False],
                "positive": [True, False],
            },
        },
        hparam_type="grid",
        test_size=21 * 12 * 2,
    )
    so.models_heatmap("ModifiedOLS_analytic")
    so.coefs_stackedbarplot("ModifiedOLS_analytic")"""

    """ Linear regression with panel bootstrap adjustment """

    model = ModifiedLinearRegression(
        method="bootstrap",
        bootstrap_method="period",
        bootstrap_iters=100,
        resample_ratio=0.5,
    )
    model.fit(X, y)
    print("Modified OLS intercept:", model.intercept_)
    print("Modified OLS coefficients:", model.coef_)
    print("Modified OLS signal:", model.predict(X))

    # Grid search for a modified linear regression with analytical standard error adjustment
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
    print("Modified OLS grid search results:")
    print("-----------------------------")
    print(pd.DataFrame(cv.cv_results_))
    print("-----------------------------")

    # Try with signal optimizer
    inner_splitter = ExpandingKFoldPanelSplit(n_splits = 5)
    so = SignalOptimizer(
        inner_splitter=inner_splitter,
        X = X,
        y = y,
    )
    so.calculate_predictions(
        name="ModifiedOLS_period",
        models = {
            "mlr": model,
        },
        metric = make_scorer(r2_score, greater_is_better=True),
        hparam_grid= {
            "mlr": {
                "fit_intercept": [True, False],
                "positive": [True, False],
            },
        },
        hparam_type="grid",
        test_size=21 * 12 * 2,
    )
    so.models_heatmap("ModifiedOLS_period")
    so.coefs_stackedbarplot("ModifiedOLS_period")
