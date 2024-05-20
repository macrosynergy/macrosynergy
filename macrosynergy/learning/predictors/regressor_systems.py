import datetime
import numbers
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge

from macrosynergy.learning.predictors import LADRegressor


class BaseRegressionSystem(BaseEstimator, RegressorMixin, ABC):

    def __init__(
        self,
        model_partial: Callable[[], BaseEstimator],
        roll: int = None,
        data_freq: str = "D",
        min_xs_samples: int = 2,
    ):
        self.model_partial = model_partial
        self.roll = roll
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples

        self.models = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ):
        """
        Fit method to fit a rolling linear regression on each cross-section, subject to
        cross-sectional data availability.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        """
        # TODO: requirement checks
        # Adjust min_xs_samples based on the frequency of the data
        min_xs_samples, roll = self.select_data_freq()

        cross_sections = X.index.get_level_values(0).unique()

        X = self._downsample_by_data_freq(X)
        y = self._downsample_by_data_freq(y)

        for section in cross_sections:
            X_section = X.xs(section, level=0)
            y_section = y.xs(section, level=0)
            # Check if there are enough samples to fit a model
            unique_dates = sorted(X_section.index.unique())
            num_dates = len(unique_dates)
            if num_dates < min_xs_samples:
                # Skip to the next cross-section
                continue
            # If a roll is specified, then adjust the dates accordingly
            if self.roll:
                X_section, y_section = self.roll_dates(
                    roll, X_section, y_section, unique_dates
                )
            # Fit the model
            model = self.model_partial()
            model.fit(pd.DataFrame(X_section), y_section)
            # Store model and coefficients
            self.models[section] = model
            self.store_model_info(section, model)

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ):
        """
        Predict method to make model predictions over a panel based on the fitted
        seemingly unrelated regression.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <pd.Series>: Pandas series of predictions, multi-indexed by cross-section
            and date.
        """
        # TODO: implement checks
        predictions = pd.Series(index=X.index, data=np.nan)

        # Check whether each test cross-section has an associated model
        cross_sections = predictions.index.get_level_values(0).unique()
        for idx, section in enumerate(cross_sections):
            if section in self.models.keys():
                # If a model exists, return the estimated OOS contract return.
                predictions[predictions.index.get_level_values(0) == section] = (
                    self.models[section].predict(X.xs(section, level=0)).flatten()
                )

        return predictions

    def _downsample_by_data_freq(self, df):
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

    def roll_dates(self, roll, X_section, y_section, unique_dates):
        right_dates = unique_dates[-roll:]
        mask = X_section.index.isin(right_dates)
        X_section = X_section[mask]
        y_section = y_section[mask]
        return X_section, y_section

    def select_data_freq(self):
        roll = None
        if self.data_freq == "D":
            min_xs_samples = self.min_xs_samples
            if self.roll:
                roll = self.roll
        elif self.data_freq == "W":
            min_xs_samples = self.min_xs_samples // 5
            if self.roll:
                roll = self.roll // 5
        elif self.data_freq == "M":
            min_xs_samples = self.min_xs_samples // 21
            if self.roll:
                roll = self.roll // 21
        elif self.data_freq == "Q":
            min_xs_samples = self.min_xs_samples // 63
            if self.roll:
                roll = self.roll // 63
        else:
            raise ValueError("Invalid data frequency. Accepted values are 'D', 'W', 'M' and 'Q'.")
        return min_xs_samples, roll

    @abstractmethod
    def store_model_info(self, section, model):
        pass


class LinearRegressionSystem(BaseRegressionSystem):
    """
    Custom scikit-learn predictor class to create a linear OLS seemingly unrelated
    regression model. This means that separate regressions are estimated for each
    cross-section, but evaluation is performed over the panel. Consequently, the results of
    a hyperparameter search will choose a single set of hyperparameters for all cross-sections,
    but the model parameters themselves may differ across cross-sections.
    """

    def __init__(
        self,
        roll: int = None,
        fit_intercept: bool = True,
        positive: bool = False,
        data_freq: str = "D",
        min_xs_samples: int = 2,
    ):
        """
        Initializes a rolling seemingly unrelated OLS linear regression model. Since
        separate models are estimated for each cross-section, a minimum constraint on the
        number of samples per cross-section, called min_xs_samples, is required for
        sensible inference.

        :param <int> roll: The lookback of the rolling window for the regression. If None,
            the entire cross-sectional history is used for each regression.
        :param <bool> fit_intercept: Boolean indicating whether or not to fit intercepts
            for each regression.
        :param <bool> positive: Boolean indicating whether or not to enforce positive
            coefficients for each regression.
        :param <str> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            MarketBetaEstimator class in `macrosynergy.learning`. Accpeted strings
            are 'D' for daily, 'W' for weekly, 'M' for monthly and 'Q' for quarterly.
            Default is 'D'.
        :param <int> min_xs_samples: The minimum number of samples required in each
            cross-section training set for a regression model to be fitted.
        """
        # TODO: requirement checks
        self.roll = roll
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples

        # Create data structures to store model information for each cross-section
        self.coefs_ = {}
        self.intercepts_ = {}

        model = partial(
            LinearRegression, fit_intercept=self.fit_intercept, positive=self.positive
        )
        LinearRegression(fit_intercept=self.fit_intercept, positive=self.positive)
        super().__init__(
            model_partial=model,
            roll=roll,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
        )

    def store_model_info(self, section, model):
        self.coefs_[section] = model.coef_[0]
        self.intercepts_[section] = model.intercept_


class LADRegressionSystem(BaseRegressionSystem):
    """
    Custom scikit-learn predictor class to create a linear LAD seemingly unrelated
    regression model. This means that separate regressions are estimated for each
    cross-section, but evaluation is performed over the panel. Consequently, the results of
    a hyperparameter search will choose a single set of hyperparameters for all cross-sections,
    but the model parameters themselves may differ across cross-sections.
    """

    def __init__(
        self,
        roll: int = None,
        fit_intercept: bool = True,
        positive: bool = False,
        data_freq: str = "D",
        min_xs_samples: int = 2,
    ):
        """
        Initializes a rolling seemingly unrelated LAD linear regression model. Since
        separate models are estimated for each cross-section, a minimum constraint on the
        number of samples per cross-section, called min_xs_samples, is required for
        sensible inference.

        :param <int> roll: The lookback of the rolling window for the regression. If None,
            the entire cross-sectional history is used for each regression.
        :param <bool> fit_intercept: Boolean indicating whether or not to fit intercepts
            for each regression.
        :param <bool> positive: Boolean indicating whether or not to enforce positive
            coefficients for each regression.
        :param <str> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            MarketBetaEstimator class in `macrosynergy.learning`. Accpeted strings
            are 'D' for daily, 'W' for weekly, 'M' for monthly and 'Q' for quarterly.
            Default is 'D'.
        :param <int> min_xs_samples: The minimum number of samples required in each
            cross-section training set for a regression model to be fitted.
        """
        # TODO: requirement checks
        self.roll = roll
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples

        # Create data structures to store model information for each cross-section
        self.coefs_ = {}
        self.intercepts_ = {}

        model = partial(
            LADRegressor, fit_intercept=self.fit_intercept, positive=self.positive
        )
        super().__init__(
            model_partial=model,
            roll=roll,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
        )

    def store_model_info(self, section, model):
        self.coefs_[section] = model.coef_[0]
        self.intercepts_[section] = model.intercept_


class RidgeRegressionSystem(BaseRegressionSystem):
    """
    Custom scikit-learn predictor class to create a seemingly unrelated ridge
    regression model. This means that separate regressions, possibly rolling, are estimated for each
    cross-section, but evaluation is performed over the panel. Consequently, the results of
    a hyperparameter search will choose a single set of hyperparameters for all cross-sections,
    but the model parameters themselves may differ across cross-sections.
    """

    def __init__(
        self,
        roll: int = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        positive: bool = False,
        data_freq: str = "D",
        min_xs_samples: int = 2,
        tol: float = 1e-4,
        solver: str = "cholesky",
    ):
        """
        Initializes a rolling seemingly unrelated ridge regression model. Since
        separate models are estimated for each cross-section, a minimum constraint on the
        number of samples per cross-section, called min_xs_samples, is required for
        sensible inference.

        :param <int> roll: The lookback of the rolling window for the regression. If None,
            the entire cross-sectional history is used for each regression.
        :param <float> alpha: Regularization hyperparameter. Greater values specify stronger
            regularization. This must be a value in $[0, np.inf]$. Default is 1.0.
        :param <bool> fit_intercept: Boolean indicating whether or not to fit intercepts
            for each regression.
        :param <bool> positive: Boolean indicating whether or not to enforce positive
            coefficients for each regression.
        :param <str> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            MarketBetaEstimator class in `macrosynergy.learning`. Accpeted strings
            are 'D' for daily, 'W' for weekly, 'M' for monthly and 'Q' for quarterly.
            Default is 'D'.
        :param <int> min_xs_samples: The minimum number of samples required in each
            cross-section training set for a regression model to be fitted.
        :param <float> tol: The tolerance for termination. Default is 1e-4.
        :param <str> solver: Solver to use in the computational routines. Options are
            'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga' and 'lbfgs'.
            Default is 'cholesky'.
        """
        # TODO: requirement checks
        self.roll = roll
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples
        self.tol = tol
        self.solver = solver

        # Create data structures to store model information for each cross-section
        self.coefs_ = {}
        self.intercepts_ = {}

        model = partial(
            Ridge,
            fit_intercept=self.fit_intercept,
            positive=self.positive,
            alpha=self.alpha,
            tol=self.tol,
            solver=self.solver,
        )

        super().__init__(
            model_partial=model,
            roll=roll,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
        )

    def store_model_info(self, section, model):
        self.coefs_[section] = model.coef_[0]
        self.intercepts_[section] = model.intercept_

    class LinearRegressionSystem(BaseRegressionSystem):
        """
        Custom scikit-learn predictor class to create a linear OLS seemingly unrelated
        regression model. This means that separate regressions are estimated for each
        cross-section, but evaluation is performed over the panel. Consequently, the results of
        a hyperparameter search will choose a single set of hyperparameters for all cross-sections,
        but the model parameters themselves may differ across cross-sections.
        """

        def __init__(
            self,
            roll: int = None,
            fit_intercept: bool = True,
            positive: bool = False,
            data_freq: str = "D",
            min_xs_samples: int = 2,
        ):
            """
            Initializes a rolling seemingly unrelated OLS linear regression model. Since
            separate models are estimated for each cross-section, a minimum constraint on the
            number of samples per cross-section, called min_xs_samples, is required for
            sensible inference.

            :param <int> roll: The lookback of the rolling window for the regression. If None,
                the entire cross-sectional history is used for each regression.
            :param <bool> fit_intercept: Boolean indicating whether or not to fit intercepts
                for each regression.
            :param <bool> positive: Boolean indicating whether or not to enforce positive
                coefficients for each regression.
            :param <str> data_freq: Training set data frequency. This is primarily
                to be used within the context of market beta estimation in the
                MarketBetaEstimator class in `macrosynergy.learning`. Accpeted strings
                are 'D' for daily, 'W' for weekly, 'M' for monthly and 'Q' for quarterly.
                Default is 'D'.
            :param <int> min_xs_samples: The minimum number of samples required in each
                cross-section training set for a regression model to be fitted.
            """
            # TODO: requirement checks
            self.roll = roll
            self.fit_intercept = fit_intercept
            self.positive = positive
            self.data_freq = data_freq
            self.min_xs_samples = min_xs_samples

            # Create data structures to store model information for each cross-section
            self.coefs_ = {}
            self.intercepts_ = {}

            model = partial(
                LinearRegression,
                fit_intercept=self.fit_intercept,
                positive=self.positive,
            )
            LinearRegression(fit_intercept=self.fit_intercept, positive=self.positive)
            super().__init__(
                model_partial=model,
                roll=roll,
                data_freq=data_freq,
                min_xs_samples=min_xs_samples,
            )

        def store_model_info(self, section, model):
            self.coefs_[section] = model.coef_[0]
            self.intercepts_[section] = model.intercept_


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    import macrosynergy.management as msm
    from macrosynergy.management import make_qdf

    np.random.seed(1)

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2013-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2010-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2010-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2010-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2010-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2010-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd = msm.reduce_df(df=dfd, cids=cids, xcats=xcats, blacklist=black)

    dfd = dfd.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd.drop(columns=["XR"])
    y = dfd["XR"]
    
    LinearRegressionSystem(data_freq="W").fit(X, y)
