from abc import ABC, abstractmethod
from typing import Union, Optional
import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge

from macrosynergy.learning.predictors import LADRegressor

from macrosynergy.management.validation import _validate_Xy_learning


class BaseRegressionSystem(BaseEstimator, RegressorMixin, ABC):

    def __init__(
        self,
        roll: Union[int, str] = "full",
        min_xs_samples: int = 2,
        data_freq: str = "unadjusted",
    ):
        """
        Base class for cross-sectional systems of regressors.

        :param <Union[int,str]> roll: The lookback of the rolling window for the regression.
            If "full", the entire cross-sectional history is used for each regression.
            Otherwise, this parameter should be an integer specified in units of the native
            data frequency, possibly adjusted by the data_freq attribute. Default is "full".
        :param <int> min_xs_samples: The minimum number of samples required in each
            cross-section training set for a regression model to be fitted.
        :param <str> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. If not 'unadjusted', it is assumed
            the native dataset frequency is daily before downsampling by summation.
            Default is 'unadjusted'.
        """
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
        Fit method to fit a regression on each cross-section, subject to
        cross-sectional data availability.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        """
        # Fit checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The X argument must be a pandas DataFrame.")
        if isinstance(y, np.ndarray):
            # This can happen during sklearn's GridSearch when a voting regressor is used
            y = pd.Series(y, index=X.index)

        # Create data structures to store model information for each cross-section
        self.coefs_ = {}
        self.intercepts_ = {}

        _validate_Xy_learning(X, y)

        cross_sections = X.index.unique(level=0)

        if self.data_freq != "unadjusted":
            # Downsample data frequency and adjust min_xs_samples correspondingly
            min_xs_samples = self.select_data_freq()
            X = self._downsample_by_data_freq(X)
            y = self._downsample_by_data_freq(y)
        else:
            min_xs_samples = self.min_xs_samples
        for section in cross_sections:
            X_section = X.xs(section, level=0, drop_level=False)
            y_section = y.xs(section, level=0, drop_level=False)
            unique_dates = X_section.index.unique()
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
        """
        model = self.create_model()
        model.fit(pd.DataFrame(X_section), y_section)
        # Store model and coefficients
        self.models[section] = model
        self.store_model_info(section, model)

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
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The X argument must be a pandas DataFrame.")
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")

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

    def roll_dates(self, roll, X_section, y_section, unique_dates):
        right_dates = unique_dates[-roll:]
        mask = X_section.index.isin(right_dates)
        X_section = X_section[mask]
        y_section = y_section[mask]
        return X_section, y_section

    def select_data_freq(self):
        if self.data_freq == "W":
            min_xs_samples = self.min_xs_samples / 5
        elif self.data_freq == "M":
            min_xs_samples = self.min_xs_samples / 21
        elif self.data_freq == "Q":
            min_xs_samples = self.min_xs_samples / 63
        else:
            raise ValueError(
                "Invalid data frequency. Accepted values are 'W', 'M' and 'Q'."
            )
        return min_xs_samples

    @abstractmethod
    def store_model_info(self, section, model):
        pass

    @abstractmethod
    def create_model(self):
        """
        Method use to instantiate a regression model for a given cross-section.

        Must be overridden.
        """
        pass

    def _check_xs_dates(self, min_xs_samples, num_dates):
        if num_dates < min_xs_samples:
            return False
        return True

    def _downsample_by_data_freq(self, df):
        # Look into whether it is fine to remove the copy
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


class LinearRegressionSystem(BaseRegressionSystem):
    """
    Custom scikit-learn predictor class to create a system of OLS linear regression models
    for each cross-section. Evaluation is performed over the panel, meaning the results of
    a hyperparameter search will choose a single set of hyperparameters for all cross-sections,
    but the model parameters themselves may differ across cross-sections.

    .. note::

      This estimator is still **experimental**: the predictions
      and the API might change without any deprecation cycle.
    """

    def __init__(
        self,
        roll: int = None,
        fit_intercept: bool = True,
        positive: bool = False,
        data_freq: str = "unadjusted",
        min_xs_samples: int = 2,
    ):
        """
        Initializes a (optional) rolling system of OLS linear regression models for each 
        cross-section. Since separate models are estimated for each cross-section,
        a minimum constraint on the number of samples per cross-section,
        called min_xs_samples, is required for sensible inference.

        :param <int> roll: The lookback of the rolling window for the regression. If None,
            the entire cross-sectional history is used. This should
            be specified in units of the data frequency, possibly adjusted by the
            data_freq attribute.
        :param <bool> fit_intercept: Boolean indicating whether or not to fit intercepts
            for each regression.
        :param <bool> positive: Boolean indicating whether or not to enforce positive
            coefficients for each regression.
        :param <str> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. If not 'unadjusted', it is assumed
            the native dataset frequency is daily before downsampling by summation.
            Default is 'unadjusted'.
        :param <int> min_xs_samples: The minimum number of samples required in each
            cross-section training set for a regression model to be fitted.
        """
        # Checks
        self._check_init_params(
            roll, fit_intercept, positive, data_freq, min_xs_samples
        )

        self.roll = roll
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples

        super().__init__(
            roll=self.roll,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
        )

    def create_model(self):
        return LinearRegression(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
        )

    def store_model_info(self, section, model):
        self.coefs_[section] = model.coef_[0]
        self.intercepts_[section] = model.intercept_

    def _check_init_params(
        self, roll, fit_intercept, positive, data_freq, min_xs_samples
    ):
        if (roll is not None) and (not isinstance(roll, int)):
            raise TypeError("roll must be an integer or None.")
        if (roll is not None) and (roll <= 0):
            raise ValueError("roll must be a positive integer.")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")
        if not isinstance(data_freq, str):
            raise TypeError("data_freq must be a string.")
        if data_freq not in ["unadjusted", "W", "M", "Q"]:
            raise ValueError("data_freq must be one of 'unadjusted', 'W', 'M' or 'Q'.")
        if not isinstance(min_xs_samples, int):
            raise TypeError("min_xs_samples must be an integer.")
        if min_xs_samples <= 0:
            raise ValueError("min_xs_samples must be a positive integer.")


class LADRegressionSystem(BaseRegressionSystem):
    """
    Custom scikit-learn predictor class to create a system of linear LAD regressions
    for each cross section. Evaluation is performed over the panel, meaning the results of
    a hyperparameter search will choose a single set of hyperparameters for all cross-sections,
    but the model parameters themselves may differ across cross-sections.

    .. note::

      This estimator is still **experimental**: the predictions
      and the API might change without any deprecation cycle.
    """

    def __init__(
        self,
        roll: int = None,
        fit_intercept: bool = True,
        positive: bool = False,
        data_freq: str = "unadjusted",
        min_xs_samples: int = 2,
    ):
        """
        Initializes a (optional) rolling system of LAD linear regression models for each
        cross-section. Since separate models are estimated for each cross-section,
        a minimum constraint on the number of samples per cross-section, called
        min_xs_samples, is required for sensible inference.

        :param <int> roll: The lookback of the rolling window for the regression. If None,
            the entire cross-sectional history is used for each regression.
        :param <bool> fit_intercept: Boolean indicating whether or not to fit intercepts
            for each regression.
        :param <bool> positive: Boolean indicating whether or not to enforce positive
            coefficients for each regression.
        :param <str> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. If not 'unadjusted', it is assumed
            the native dataset frequency is daily before downsampling by summation.
            Default is 'unadjusted'.
        :param <int> min_xs_samples: The minimum number of samples required in each
            cross-section training set for a regression model to be fitted.
        """
        # Checks
        self._check_init_params(
            roll, fit_intercept, positive, data_freq, min_xs_samples
        )

        self.roll = roll
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples

        super().__init__(
            roll=roll,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
        )

    def create_model(self):
        return LADRegressor(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
        )

    def store_model_info(self, section, model):
        self.coefs_[section] = model.coef_[0]
        self.intercepts_[section] = model.intercept_

    def _check_init_params(
        self, roll, fit_intercept, positive, data_freq, min_xs_samples
    ):
        if not isinstance(roll, int) and roll is not None:
            raise TypeError("roll must be an integer or None.")
        if (roll is not None) and (roll <= 0):
            raise ValueError("roll must be a positive integer.")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")
        if not isinstance(data_freq, str):
            raise TypeError("data_freq must be a string.")
        if data_freq not in ["unadjusted", "W", "M", "Q"]:
            raise ValueError("data_freq must be one of 'unadjusted', 'W', 'M' or 'Q'.")
        if not isinstance(min_xs_samples, int):
            raise TypeError("min_xs_samples must be an integer.")
        if min_xs_samples <= 0:
            raise ValueError("min_xs_samples must be a positive integer.")

class RidgeRegressionSystem(BaseRegressionSystem):
    """
    Custom scikit-learn predictor class to create a system of ridge
    regression models for each cross-section. Evaluation is performed over the panel, meaning the results of
    a hyperparameter search will choose a single set of hyperparameters for all cross-sections,
    but the model parameters themselves may differ across cross-sections.

    .. note::

      This estimator is still **experimental**: the predictions
      and the API might change without any deprecation cycle.
    """

    def __init__(
        self,
        roll: int = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        positive: bool = False,
        data_freq: str = "unadjusted",
        min_xs_samples: int = 2,
        tol: float = 1e-4,
        solver: str = "lsqr",
    ):
        """
        Initializes a (optional) rolling system of ridge regression models for each 
        cross-section. Since separate models are estimated for each cross-section,
        a minimum constraint on the number of samples per cross-section, called
        min_xs_samples, is required for sensible inference.

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
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. If not 'unadjusted', it is assumed
            the native dataset frequency is daily before downsampling by summation.
            Default is 'unadjusted'.
        :param <int> min_xs_samples: The minimum number of samples required in each
            cross-section training set for a regression model to be fitted.
        :param <float> tol: The tolerance for termination. Default is 1e-4.
        :param <str> solver: Solver to use in the computational routines. Options are
            'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga' and 'lbfgs'.
            Default is 'lsqr'.
        """
        # Checks
        self._check_init_params(
            roll, alpha, fit_intercept, positive, data_freq, min_xs_samples, tol, solver
        )

        self.roll = roll
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples
        self.tol = tol
        self.solver = solver

        super().__init__(
            roll=roll,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
        )

    def create_model(self):
        return Ridge(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
            alpha=self.alpha,
            tol=self.tol,
            solver=self.solver,
        )

    def store_model_info(self, section, model):
        self.coefs_[section] = model.coef_[0]
        self.intercepts_[section] = model.intercept_

    def _check_init_params(
        self,
        roll,
        alpha,
        fit_intercept,
        positive,
        data_freq,
        min_xs_samples,
        tol,
        solver,
    ):
        if not isinstance(roll, int) and roll is not None:
            raise TypeError("roll must be an integer or None.")
        if roll is not None and roll <= 0:
            raise ValueError("roll must be a positive integer.")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be either an integer or a float.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")
        if not isinstance(data_freq, str):
            raise TypeError("data_freq must be a string.")
        if data_freq not in ["unadjusted", "W", "M", "Q"]:
            raise ValueError("data_freq must be one of 'unadjusted', 'W', 'M' or 'Q'.")
        if not isinstance(min_xs_samples, int):
            raise TypeError("min_xs_samples must be an integer.")
        if min_xs_samples <= 0:
            raise ValueError("min_xs_samples must be a positive integer.")
        if not isinstance(tol, (int, float)):
            raise TypeError("tol must be either an integer or a float.")
        if tol <= 0:
            raise ValueError("tol must be a positive number.")
        if solver not in [
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
            "lbfgs",
        ]:
            raise ValueError(
                "solver must be one of 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga' or 'lbfgs'."
            )


class CorrelationVolatilitySystem(BaseRegressionSystem):
    """
    Custom scikit-learn predictor class written specifically to estimate betas for
    financial contracts with respect to a benchmark return series. Since an estimated beta
    can be decomposed into correlation and volatility components, this class aims to estimate
    these separately, allowing for different lookbacks and weighting schemes for both
    components.

    .. note::

      This estimator is still **experimental**: the predictions
      and the API might change without any deprecation cycle.
    """

    def __init__(
        self,
        correlation_lookback: Optional[int] = None,
        correlation_type: str = "pearson",
        volatility_lookback: int = 21,
        volatility_window_type: str = "rolling",
        data_freq: str = "unadjusted",
        min_xs_samples: int = 2,
    ):
        """
        Initialize CorrelationVolatilitySystem class.

        :param <Optional[int]> correlation_lookback: The lookback period for the correlation
            calculation. This should be in units of the dataset frequency, possibly
            relating to data_freq. Default is None (use all available history).
        :param <str> correlation_type: The type of correlation to be calculated.
            Accepted values are 'pearson', 'kendall' and 'spearman'. Default is 'pearson'.
        :param <int> volatility_lookback: The lookback period for the volatility
            calculation. This should be in units of the dataset frequency,
            possibly relating to data_freq. Default is 21.
        :param <str> volatility_window_type: The type of window to use for the volatility
            calculation. Accepted values are 'rolling' and 'exponential'. Default is 'rolling'.
        :param <str> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. If not 'unadjusted', it is assumed
            the native dataset frequency is daily before downsampling by summation.
            Default is 'unadjusted'.
        :param <int> min_xs_samples: The minimum number of samples required in each
            cross-section training set for a regression model to be fitted.
        """
        # Checks
        self._check_init_params(
            correlation_lookback,
            correlation_type,
            volatility_lookback,
            volatility_window_type,
            data_freq,
            min_xs_samples,
        )

        self.correlation_lookback = correlation_lookback
        self.correlation_type = correlation_type
        self.volatility_lookback = volatility_lookback
        self.volatility_window_type = volatility_window_type
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples

        super().__init__(
            roll=None,
            data_freq=data_freq,
            min_xs_samples=min_xs_samples,
        )

    def _fit_cross_section(self, section, X_section, y_section):
        # Estimate local standard deviations of the benchmark and contract return
        if self.volatility_window_type == "rolling":
            X_section_std = X_section.values[-self.volatility_lookback :, 0].std(ddof=1)
            y_section_std = y_section.values[-self.volatility_lookback:].std(ddof=1)
        elif self.volatility_window_type == "exponential":
            alpha = 2 / (self.volatility_lookback + 1)
            weights = np.array([(1 - alpha) ** i for i in range(len(X_section))][::-1])
            X_section_std = np.sqrt(np.cov(X_section.values.flatten(), aweights=weights))
            y_section_std = np.sqrt(np.cov(y_section.values, aweights=weights))

        # Estimate local correlation between the benchmark and contract return
        if self.correlation_lookback is not None:
            X_section_corr = X_section.values[-self.correlation_lookback:][:,0]
            y_section_corr = y_section.values[-self.correlation_lookback:]
            if self.correlation_type == "pearson":
                corr = np.corrcoef(X_section_corr,y_section_corr)[0,1]
            elif self.correlation_type == "spearman":
                X_section_ranks = np.argsort(np.argsort(X_section_corr))
                y_section_ranks = np.argsort(np.argsort(y_section_corr))
                corr = np.corrcoef(X_section_ranks,y_section_ranks)[0,1]
            elif self.correlation_type == "kendall":
                corr = stats.kendalltau(X_section_corr,y_section_corr).statistic
        else:
            if self.correlation_type == "pearson":
                corr = np.corrcoef(X_section.values[:,0],y_section.values)[0,1]
            elif self.correlation_type == "spearman":
                X_section_ranks = np.argsort(np.argsort(X_section.values[:,0]))
                y_section_ranks = np.argsort(np.argsort(y_section.values))
                corr = np.corrcoef(X_section_ranks,y_section_ranks)[0,1]
            elif self.correlation_type == "kendall":
                corr = stats.kendalltau(X_section.values[:,0],y_section.values).statistic

        # Get beta estimate and store it
        beta = corr * (y_section_std / X_section_std)
        self.store_model_info(section, beta)

    def predict(
        self,
        X: pd.DataFrame,
    ):
        """
        Predict method to make a naive zero prediction for each cross-section. This is
        because the only use of this class is to estimate betas, which were computed
        during the fit method, whose quality can be assessed by a custom scikit-learn metric
        that directly access the betas without the need for a predict method.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <pd.Series>: Pandas series of zero predictions, multi-indexed by cross-section
            and date.
        """
        predictions = pd.Series(index=X.index, data=0)

        return predictions

    def store_model_info(self, section, beta):
        self.coefs_[section] = beta

    def create_model(self):
        raise NotImplementedError("This method is not implemented for this class.")

    def _check_xs_dates(self, min_xs_samples, num_dates):
        if num_dates < min_xs_samples:
            return False
        # If the correlation lookback is greater than the number of available dates, skip
        # to the next cross-section
        if (
            self.correlation_lookback is not None
            and num_dates < self.correlation_lookback
        ):
            return False
        # If the volatility lookback is greater than the number of available dates, skip
        # to the next cross-section
        if num_dates < self.volatility_lookback:
            return False
        return True

    def _check_init_params(
        self,
        correlation_lookback,
        correlation_type,
        volatility_lookback,
        volatility_window_type,
        data_freq,
        min_xs_samples,
    ):
        if correlation_lookback is not None:
            if not isinstance(correlation_lookback, int):
                raise TypeError("correlation_lookback must be an integer.")
            if correlation_lookback <= 0:
                raise ValueError("correlation_lookback must be a positive integer.")
        if not isinstance(correlation_type, str):
            raise TypeError("correlation_type must be a string.")
        if correlation_type not in ["pearson", "kendall", "spearman"]:
            raise ValueError(
                "correlation_type must be one of 'pearson', 'kendall' or 'spearman'."
            )
        if not isinstance(volatility_lookback, int):
            raise TypeError("volatility_lookback must be an integer.")
        if volatility_lookback <= 0:
            raise ValueError("volatility_lookback must be a positive integer.")
        if not isinstance(volatility_window_type, str):
            raise TypeError("volatility_window_type must be a string.")
        if volatility_window_type not in ["rolling", "exponential"]:
            raise ValueError(
                "volatility_window_type must be one of 'rolling' or 'exponential'."
            )
        if not isinstance(data_freq, str):
            raise TypeError("data_freq must be a string.")
        if data_freq not in ["unadjusted", "W", "M", "Q"]:
            raise ValueError("data_freq must be one of 'unadjusted', 'W', 'M' or 'Q.")
        if not isinstance(min_xs_samples, int):
            raise TypeError("min_xs_samples must be an integer.")
        if min_xs_samples <= 0:
            raise ValueError("min_xs_samples must be a positive integer.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    import macrosynergy.management as msm
    from macrosynergy.management import make_qdf
    from pyinstrument import Profiler
    np.random.seed(1)

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "BENCH_XR", "CRY", "GROWTH", "INFL"]
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
    df_xcats.loc["BENCH_XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2010-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2010-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2010-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd = msm.reduce_df(df=dfd, cids=cids, xcats=xcats, blacklist=black)

    dfd = dfd.pivot(index=["cid", "real_date"], columns="xcat", values="value")

    # Demonstration of LADRegressionSystem usage
    #X1 = dfd.drop(columns=["XR", "BENCH_XR"])
    #y1 = dfd["XR"]
    #def profile_model_fitting(model):
    #    model.fit(X1, y1)

    #profiler = Profiler()
    #profiler.start()
    #profile_model_fitting(LADRegressionSystem(data_freq="W"))
    #profiler.stop()
    #with open('profile_report.html', 'w') as f:
    #    f.write(profiler.output_html())
    #print(profiler.output_text(unicode=True, color=True))

    # Demonstration of CorrelationVolatilitySystem usage

    X2 = pd.DataFrame(dfd["BENCH_XR"])
    y2 = dfd["XR"]
    profiler = Profiler()
    profiler.start()
    cv = CorrelationVolatilitySystem(volatility_window_type="exponential",correlation_lookback=21).fit(X2, y2)
    profiler.stop()
    with open('corrvol_report.html', 'w') as f:
        f.write(profiler.output_html())
    print(cv.coefs_)

    # Demonstration of LinearRegressionSystem usage
    X1 = dfd.drop(columns=["XR", "BENCH_XR"])
    y1 = dfd["XR"]
    lr = LinearRegressionSystem(data_freq="W").fit(X1, y1)
    print(lr.coefs_)
