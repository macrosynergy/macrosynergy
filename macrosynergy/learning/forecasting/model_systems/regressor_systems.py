from abc import ABC, abstractmethod
from typing import Union, Optional
import datetime

import numpy as np
import pandas as pd
from macrosynergy.learning.forecasting.model_systems import BaseRegressionSystem
from sklearn.linear_model import LinearRegression, Ridge

from macrosynergy.learning.forecasting import LADRegressor

from macrosynergy.management.validation import _validate_Xy_learning


class LinearRegressionSystem(BaseRegressionSystem):
    """
    Cross-sectional system of linear regression models. 

    Parameters
    ----------
    roll : int or None, default=None
        The lookback of the rolling window for the regression.
    fit_intercept : bool, default=True
        Whether to fit an intercept for each regression.
    positive : bool, default=False
        Whether to enforce positive coefficients for each regression.
    data_freq : str, default='D'
        Training set data frequency for resampling. 
        Accepted values are 'D' for daily, 'W' for weekly, 'M' for monthly and
        'Q' for quarterly.
    min_xs_samples : int, default=2
        The minimum number of samples required in each cross-section training set
        for a regression model to be fitted.

    Notes
    -----
    Separate regression models are fit for each cross-section, but evaluation is performed
    over the panel. Consequently, the results of a hyperparameter search will choose
    a single set of hyperparameters for all cross-sections, but the model parameters
    themselves may differ across cross-sections.

    This estimator is primarily intended for use within the context of market beta
    estimation, but can be plausibly used for return forecasting or other downstream tasks.
    The `data_freq` parameter is particularly intended for cross-validating market beta
    estimation models, since choosing the underlying data frequency is of interest in
    quant analysis.
    """
    def __init__(
        self,
        roll = None,
        fit_intercept = True,
        positive = False,
        data_freq = "D",
        min_xs_samples = 2,
    ):
        # Call the parent class constructor
        super().__init__(roll=roll, data_freq=data_freq, min_xs_samples=min_xs_samples)

        # Additional checks
        self._check_init_params(
            fit_intercept, positive,
        )

        # Additional attributes
        self.fit_intercept = fit_intercept
        self.positive = positive

        # Create data structures to store model information for each cross-section
        self.coefs_ = {}
        self.intercepts_ = {}

    def create_model(self):
        """
        Instantiate a linear regression model.

        Returns
        -------
        LinearRegression
            A linear regression model with the specified hyperparameters.
        """
        return LinearRegression(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
        )

    def store_model_info(self, section, model):
        """
        Store the coefficients and intercepts of a fitted linear regression model.

        Parameters
        ----------
        section : str
            The cross-section identifier.
        model : LinearRegression
            The fitted linear regression model.
        """
        self.coefs_[section] = model.coef_[0]
        self.intercepts_[section] = model.intercept_

    def _check_init_params(
        self, fit_intercept, positive,
    ):
        """
        Parameter checks for the LinearRegressionSystem constructor.

        Parameters
        ----------
        fit_intercept : bool
            Whether to fit an intercept for each regression.
        positive : bool
            Whether to enforce positive coefficients for each regression.
        """
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")


class LADRegressionSystem(BaseRegressionSystem):
    """
    Cross-sectional system of LAD regression models. 

    Parameters
    ----------
    roll : int or None, default=None
        The lookback of the rolling window for the regression.
    fit_intercept : bool, default=True
        Whether to fit an intercept for each regression.
    positive : bool, default=False
        Whether to enforce positive coefficients for each regression.
    data_freq : str, default='D'
        Training set data frequency for resampling. 
        Accepted values are 'D' for daily, 'W' for weekly, 'M' for monthly and
        'Q' for quarterly.
    min_xs_samples : int, default=2
        The minimum number of samples required in each cross-section training set
        for a regression model to be fitted.

    Notes
    -----
    Separate regression models are fit for each cross-section, but evaluation is performed
    over the panel. Consequently, the results of a hyperparameter search will choose
    a single set of hyperparameters for all cross-sections, but the model parameters
    themselves may differ across cross-sections.

    This estimator is primarily intended for use within the context of market beta
    estimation, but can be plausibly used for return forecasting or other downstream tasks.
    The `data_freq` parameter is particularly intended for cross-validating market beta
    estimation models, since choosing the underlying data frequency is of interest in
    quant analysis.
    """

    def __init__(
        self,
        roll = None,
        fit_intercept = True,
        positive = False,
        data_freq = "D",
        min_xs_samples = 2,
    ):
        # Call the parent class constructor
        super().__init__(roll=roll, data_freq=data_freq, min_xs_samples=min_xs_samples)

        # Additional checks
        self._check_init_params(
            fit_intercept, positive,
        )

        # Additional attributes
        self.fit_intercept = fit_intercept
        self.positive = positive

        # Create data structures to store model information for each cross-section
        self.coefs_ = {}
        self.intercepts_ = {}

    def create_model(self):
        """
        Instantiate a LAD regression model.

        Returns
        -------
        LADRegressor
            A LAD regression model with the specified hyperparameters.
        """
        return LADRegressor(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
        )

    def store_model_info(self, section, model):
        """
        Store the coefficients and intercepts of a fitted LAD regression model.

        Parameters
        ----------
        section : str
            The cross-section identifier.
        model : LADRegressor
            The fitted linear regression model.
        """
        self.coefs_[section] = model.coef_[0]
        self.intercepts_[section] = model.intercept_

    def _check_init_params(
        self, fit_intercept, positive,
    ):
        """
        Parameter checks for the LADRegressionSystem constructor.

        Parameters
        ----------
        fit_intercept : bool
            Whether to fit an intercept for each regression.
        positive : bool
            Whether to enforce positive coefficients for each regression.
        """
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")


class RidgeRegressionSystem(BaseRegressionSystem):
    """
    Cross-sectional system of ridge regression models. 

    Parameters
    ----------
    roll : int or None, default=None
        The lookback of the rolling window for the regression.
    alpha : float, default=1.0
        L2 regularization hyperparameter. Greater values specify stronger regularization.
    fit_intercept : bool, default=True
        Whether to fit an intercept for each regression.
    positive : bool, default=False
        Whether to enforce positive coefficients for each regression.
    data_freq : str, default='D'
        Training set data frequency for resampling. 
        Accepted values are 'D' for daily, 'W' for weekly, 'M' for monthly and
        'Q' for quarterly.
    min_xs_samples : int, default=2
        The minimum number of samples required in each cross-section training set
        for a regression model to be fitted.
    tol : float, default=1e-4
        The tolerance for termination.
    solver : str, default='lsqr'
        Solver to use in the computational routines. Options are 'auto', 'svd', 'cholesky',
        'lsqr', 'sparse_cg', 'sag', 'saga' and 'lbfgs'.

    Notes
    -----
    Separate regression models are fit for each cross-section, but evaluation is performed
    over the panel. Consequently, the results of a hyperparameter search will choose
    a single set of hyperparameters for all cross-sections, but the model parameters
    themselves may differ across cross-sections.

    This estimator is primarily intended for use within the context of market beta
    estimation, but can be plausibly used for return forecasting or other downstream tasks.
    The `data_freq` parameter is particularly intended for cross-validating market beta
    estimation models, since choosing the underlying data frequency is of interest in
    quant analysis.
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
        solver: str = "lsqr",
    ):
        # Call the parent class constructor
        super().__init__(roll=roll, data_freq=data_freq, min_xs_samples=min_xs_samples)

        # Checks
        self._check_init_params(
            alpha, fit_intercept, positive, tol, solver,
        )

        # Additional attributes
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.tol = tol
        self.solver = solver

        # Create data structures to store model information for each cross-section
        self.coefs_ = {}
        self.intercepts_ = {}

    def create_model(self):
        """
        Instantiate a ridge regression model.

        Returns
        -------
        Ridge
            A ridge regression model with the specified hyperparameters.
        """
        return Ridge(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
            alpha=self.alpha,
            tol=self.tol,
            solver=self.solver,
        )

    def store_model_info(self, section, model):
        """
        Store the coefficients and intercepts of a fitted ridge regression model.

        Parameters
        ----------
        section : str
            The cross-section identifier.
        model : Ridge
            The fitted ridge regression model.
        """
        self.coefs_[section] = model.coef_[0]
        self.intercepts_[section] = model.intercept_

    def _check_init_params(
        self,
        alpha,
        fit_intercept,
        positive,
        tol,
        solver,
    ):
        """
        Parameter checks for the RidgeRegressionSystem constructor.

        Parameters
        ----------
        alpha : float
            L2 regularization hyperparameter. Greater values specify stronger
            regularization.
        fit_intercept : bool
            Whether to fit an intercept for each regression.
        positive : bool
            Whether to enforce positive coefficients for each regression.
        tol : float
            The tolerance for termination.
        solver : str
            Solver to use in the computational routines.
        """
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be either an integer or a float.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")
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
                "solver must be one of 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', "
                "'sag', 'saga' or 'lbfgs'."
            )


class CorrelationVolatilitySystem(BaseRegressionSystem):
    """
    Custom scikit-learn predictor class written specifically to estimate betas for
    financial contracts with respect to a benchmark return series. Since an estimated beta
    can be decomposed into correlation and volatility components, this class aims to estimate
    these separately, allowing for different lookbacks and weighting schemes for both
    components.

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle.
    """

    def __init__(
        self,
        correlation_lookback: Optional[int] = None,
        correlation_type: str = "pearson",
        volatility_lookback: int = 21,
        volatility_window_type: str = "rolling",
        data_freq: str = "D",
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
        :param <str> data_freq: Training set data frequency for downsampling. Default is 'D'.
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

        # Create data structures to store the estimated betas for each cross-section
        self.coefs_ = {}

    def _fit_cross_section(self, section, X_section, y_section):
        # Estimate local standard deviations of the benchmark and contract return
        if self.volatility_window_type == "rolling":
            X_section_std = X_section[-self.volatility_lookback :].std().iloc[-1]
            y_section_std = y_section[-self.volatility_lookback :].std()
        elif self.volatility_window_type == "exponential":
            X_section_std = (
                X_section.ewm(span=self.volatility_lookback).std().iloc[-1, 0]
            )
            y_section_std = y_section.ewm(span=self.volatility_lookback).std().iloc[-1]

        # Estimate local correlation between the benchmark and contract return
        if self.correlation_lookback is not None:
            X_section_corr = X_section.tail(self.correlation_lookback)
            y_section_corr = y_section.tail(self.correlation_lookback)
            corr = X_section_corr.corrwith(
                y_section_corr, method=self.correlation_type
            ).iloc[-1]
        else:
            corr = X_section.corrwith(y_section, method=self.correlation_type).iloc[-1]

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
        if data_freq not in ["D", "W", "M", "Q"]:
            raise ValueError("data_freq must be one of 'D', 'W', 'M' or 'Q.")
        if not isinstance(min_xs_samples, int):
            raise TypeError("min_xs_samples must be an integer.")
        if min_xs_samples <= 0:
            raise ValueError("min_xs_samples must be a positive integer.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    import macrosynergy.management as msm
    from macrosynergy.management import make_qdf

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

    # Demonstration of CorrelationVolatilitySystem usage

    X2 = pd.DataFrame(dfd["BENCH_XR"])
    y2 = dfd["XR"]
    cv = CorrelationVolatilitySystem().fit(X2, y2)
    print(cv.coefs_)

    # Demonstration of LinearRegressionSystem usage
    X1 = dfd.drop(columns=["XR", "BENCH_XR"])
    y1 = dfd["XR"]
    lr = LinearRegressionSystem(data_freq="W").fit(X1, y1)
    print(lr.coefs_)
