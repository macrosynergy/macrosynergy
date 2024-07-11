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
        data_freq: Optional[str] = None,
    ):
        """
        Base class for cross-sectional systems of regressors.

        :param <Union[int,str]> roll: The lookback of the rolling window for the regression.
            If "full", the entire cross-sectional history is used for each regression.
            Otherwise, this parameter should be an integer specified in units of the native
            data frequency. If `data_freq` is not None, then an integer value for `roll`
            should be expressed in units of the frequency specified in `data_freq`.
            Default is "full".
        :param <int> min_xs_samples: The minimum number of samples required in a given
            cross-section for a regression model to be fit for that cross-section.
            If `data_freq` is None, this parameter is specified in units of the underlying
            dataset frequency. Otherwise, this parameter should be expressed in units of
            the frequency specified in `data_freq`. Default is 2.
        :param <Optional[str]> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. It is recommended to set this parameter
            to "W", "M" or "Q" only when the native dataset frequency is greater.
            Default is None.
        """
        # Checks
        if not isinstance(roll, (int, str)):
            raise TypeError("roll must be an integer or string.")
        if (isinstance(roll, int)) and (roll <= 1):
            raise ValueError("roll must be greater than 1 when an integer is specified.")
        if (isinstance(roll, str)) and (roll != "full"):
            raise ValueError("roll must equal `full` when a string is specified.")
        if not isinstance(min_xs_samples, int):
            raise TypeError("min_xs_samples must be an integer.")
        if min_xs_samples <= 1:
            raise ValueError("min_xs_samples must be a positive integer greater than one.")
        if data_freq is not None:
            if not isinstance(data_freq, str):
                raise TypeError("data_freq must be a string.")
            if data_freq not in ["unadjusted", "W", "M", "Q"]:
                raise ValueError("data_freq must be one of 'unadjusted', 'W', 'M' or 'Q'.")

        
        # Assignments
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

        :return <BaseRegressionSystem>: Fitted regression system object.
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The X argument must be a pandas DataFrame.")
        if isinstance(y, np.ndarray):
            # This can happen during sklearn's GridSearch when a voting regressor is used
            y = pd.Series(y, index=X.index)
        _validate_Xy_learning(X, y)

        # Create data structures to store model information for each cross-section
        self.coefs_ = {}
        self.intercepts_ = {}

        # Downsample data frequency if necessary
        if (self.data_freq is not None) and (self.data_freq != "unadjusted"):
            # Downsample data frequency
            X = self._downsample_by_data_freq(X)
            y = self._downsample_by_data_freq(y)

        # Iterate over cross-sections and fit a regression model on each
        cross_sections = X.index.unique(level=0)
        for section in cross_sections:
            X_section = X.xs(section, level=0, drop_level=False)
            y_section = y.xs(section, level=0, drop_level=False)
            unique_dates = X_section.index.unique() # TODO: sort?
            num_dates = len(unique_dates)
            # Skip cross-section if it has insufficient samples
            if not self._check_xs_dates(self.min_xs_samples, num_dates):
                continue
            # Roll the data if necessary
            if self.roll and self.roll != "full":
                if num_dates <= self.roll:
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

        :param <str> section: Cross-section identifier.
        :param <pd.DataFrame> X_section: Pandas dataframe of input features for the
            given cross-section.
        :param <pd.Series> y_section: Pandas series of target values for the given
            cross-section.

        :return: None
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
        regression system. If a model does not exist for a given cross-section, the
        prediction will be NaN. This is a necessary feature of this style of regression 
        model but it has the implication that custom metrics may need to be used, in 
        accordance with the `scikit-learn` API, to evaluate the model's performance.

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

        # Create a series to store predictions
        predictions = pd.Series(index=X.index, data=np.nan)

        # Store predictions for each test cross-section, if an existing model is available
        cross_sections = predictions.index.get_level_values(0).unique()
        for idx, section in enumerate(cross_sections):
            if section in self.models.keys():
                predictions[predictions.index.get_level_values(0) == section] = (
                    self.models[section].predict(X.xs(section, level=0)).flatten()
                )

        return predictions

    def roll_dates(self, roll, X_section, y_section, unique_dates):
        """
        Helper method to truncate data history to the last `roll` dates.

        :param <int> roll: The lookback of the rolling window for the regression.
        :param <pd.DataFrame> X_section: Pandas dataframe of input features for the
            given cross-section.
        :param <pd.Series> y_section: Pandas series of target values for the given
            cross-section.
        :param <pd.DatetimeIndex> unique_dates: Pandas datetime index of unique dates
            for the given cross-section.

        :return <Tuple[pd.DataFrame, pd.Series]>: Tuple of truncated input features and
            target values.
        """
        right_dates = unique_dates[-roll:]
        mask = X_section.index.isin(right_dates)
        X_section = X_section[mask]
        y_section = y_section[mask]
        return X_section, y_section

    @abstractmethod
    def store_model_info(self, section, model):
        """
        Abstract method to store model information for a given cross-section.

        Must be overridden.

        :param <str> section: Cross-section identifier.
        :param <RegressorMixin> model: Fitted regression model for the given cross-section.

        :return: None
        """
        pass

    @abstractmethod
    def create_model(self):
        """
        Method use to instantiate a regression model for a given cross-section.

        Must be overridden.

        :return: None
        """
        pass

    def _check_xs_dates(self, min_xs_samples, num_dates):
        """
        Private method to check whether or not a given cross-section comprises enough
        training samples to fit a regression model.

        :param <int> min_xs_samples: The minimum number of samples required in a given
            cross-section for a regression model to be fit for that cross-section.
        :param <int> num_dates: The number of unique training dates in the cross-section.

        :return <bool>: Boolean indicating whether or not the cross-section has enough
            samples to fit a regression model.
        """
        if num_dates < min_xs_samples:
            return False
        return True

    def _downsample_by_data_freq(self, df):
        """
        Private method to downsample a dataframe by the data frequency specified in the
        `data_freq` parameter.

        :param <pd.DataFrame> df: Pandas dataframe to downsample.

        :return <pd.DataFrame>: Downsampled pandas dataframe.
        """
        return (
            df.groupby(
                [
                    pd.Grouper(level="cid"),
                    pd.Grouper(level="real_date", freq=self.data_freq),
                ]
            )
            .sum()
            .copy() # TODO: is copying necessary?
        )

class LinearRegressionSystem(BaseRegressionSystem):
    """
    Cross-sectional system of OLS linear regression models for panel data.

    Hyperparameters are shared across cross-sections, but the model parameters
    are allowed to differ. In this sense, the system equations are "seemingly unrelated".

    :param <Union[int, str]> roll: The lookback of the rolling window for the regression.
        If "full", the entire cross-sectional history is used for each regression.
        Otherwise, this parameter should be an integer specified in units of the native
        data frequency. If `data_freq` is not None, then an integer value for `roll`
        should be expressed in units of the frequency specified in `data_freq`.
        Default is "full".
    :param <bool> fit_intercept: Boolean indicating whether or not to fit intercepts
        for each regression. Default is True.
    :param <bool> positive: Boolean indicating whether or not to enforce positive
        coefficients for each regression. Default is False.
    :param <int> min_xs_samples: The minimum number of samples required in a given 
        cross-section for a regression model to be fit for that cross-section.
        If `data_freq` is None, this parameter is specified in units of the underlying
        dataset frequency. Otherwise, this parameter should be expressed in units of
        the frequency specified in `data_freq`. Default is 2.
    :param <Optional[str]> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. It is recommended to set this parameter
            to "W", "M" or "Q" only when the native dataset frequency is greater.
            Default is None.

    Notes
    -----
    From an implementation perspective, this class just fits (possibly) rolling
    OLS or NNLS (when `positive` is True) linear regression models for each cross-section,
    providing enough samples are available in that cross-section. Model coefficients and
    intercepts are stored in the `coefs_` and `intercepts_` dictionaries respectively,
    for which the keys represent cross-sections and the values are the model parameters.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        positive: bool = False,
        roll: Union[int, str] = "full",
        min_xs_samples: int = 2,
        data_freq: Optional[str] = None,
    ):
        # Checks
        self._check_init_params(
            fit_intercept, positive,
        )

        self.fit_intercept = fit_intercept
        self.positive = positive
        self.roll = roll
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
        """
        Method to store model information for a given cross-section. The coefs_ and 
        intercepts_ dictionaries are updated with the extracted model coefficients and
        intercepts for each cross-section for which sufficient data is available.
        """
        self.coefs_[section] = model.coef_
        self.intercepts_[section] = model.intercept_

    def _check_init_params(
        self, fit_intercept, positive
    ):
        # fit_intercept
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        
        # positive
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")


class LADRegressionSystem(BaseRegressionSystem):
    """
    Cross-sectional system of LAD linear regression models for panel data.

    Hyperparameters are shared across cross-sections, but the model parameters
    are allowed to differ. In this sense, the system equations are "seemingly unrelated".

    :param <Union[int, str]> roll: The lookback of the rolling window for the regression.
        If "full", the entire cross-sectional history is used for each regression.
        Otherwise, this parameter should be an integer specified in units of the native
        data frequency. If `data_freq` is not None, then an integer value for `roll`
        should be expressed in units of the frequency specified in `data_freq`.
        Default is "full".
    :param <bool> fit_intercept: Boolean indicating whether or not to fit intercepts
        for each regression. Default is True.
    :param <bool> positive: Boolean indicating whether or not to enforce positive
        coefficients for each regression. Default is False.
    :param <int> min_xs_samples: The minimum number of samples required in a given 
        cross-section for a regression model to be fit for that cross-section.
        If `data_freq` is None, this parameter is specified in units of the underlying
        dataset frequency. Otherwise, this parameter should be expressed in units of
        the frequency specified in `data_freq`. Default is 2.
    :param <Optional[str]> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. It is recommended to set this parameter
            to "W", "M" or "Q" only when the native dataset frequency is greater.
            Default is None.

    Notes
    -----
    From an implementation perspective, this class just fits (possibly) rolling
    LAD or NNLAD (when `positive` is True) linear regression models for each cross-section,
    providing enough samples are available in that cross-section. Model coefficients and
    intercepts are stored in the `coefs_` and `intercepts_` dictionaries respectively,
    for which the keys represent cross-sections and the values are the model parameters.

    LAD regression is a robust regression technique that minimizes the sum of the absolute
    residuals, as opposed to the sum of the squared residuals in OLS regression i.e. L1
    loss instead of L2 loss.
    """

    def __init__(
        self,
        roll: Union[int, str] = "full",
        fit_intercept: bool = True,
        positive: bool = False,
        min_xs_samples: int = 2,
        data_freq: Optional[str] = None,
    ):
        # Checks
        self._check_init_params(
            fit_intercept, positive,
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
        self.coefs_[section] = model.coef_
        self.intercepts_[section] = model.intercept_

    def _check_init_params(
        self, fit_intercept, positive,
    ):
        # fit_intercept
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        
        # positive
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")

class RidgeRegressionSystem(BaseRegressionSystem):
    """
    Cross-sectional system of Ridge regression models for panel data.

    Hyperparameters are shared across cross-sections, but the model parameters
    are allowed to differ. In this sense, the system equations are "seemingly unrelated".

    :param <Union[int, str]> roll: The lookback of the rolling window for the regression.
        If "full", the entire cross-sectional history is used for each regression.
        Otherwise, this parameter should be an integer specified in units of the native
        data frequency. If `data_freq` is not None, then an integer value for `roll`
        should be expressed in units of the frequency specified in `data_freq`.
        Default is "full".
    :param <float> alpha: Regularization hyperparameter. Greater values specify stronger
        regularization. This must be a finite, non-negative value. Default is 1.0.
    :param <bool> fit_intercept: Boolean indicating whether or not to fit intercepts
        for each regression. Default is True.
    :param <bool> positive: Boolean indicating whether or not to enforce positive
        coefficients for each regression. Default is False.
    :param <float> tol: The tolerance for termination. Default is 1e-4.
    :param <str> solver: Solver to use in the computational routines. Options are
        'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga' and 'lbfgs'.
        Default is 'lsqr'. If `positive` is True, the solver must be 'lbfgs' or 'auto'.
    :param <int> min_xs_samples: The minimum number of samples required in a given 
        cross-section for a regression model to be fit for that cross-section.
        If `data_freq` is None, this parameter is specified in units of the underlying
        dataset frequency. Otherwise, this parameter should be expressed in units of
        the frequency specified in `data_freq`. Default is 2.
    :param <Optional[str]> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. It is recommended to set this parameter
            to "W", "M" or "Q" only when the native dataset frequency is greater.
            Default is None.

    Notes
    -----
    From an implementation perspective, this class just fits (possibly) rolling
    Ridge regression models for each cross-section, provided enough samples are available
    in that cross-section. When `positive` is True, the optimization is performed so that
    model coefficients are non-negative. Model coefficients and intercepts are stored in
    the `coefs_` and `intercepts_` dictionaries respectively, for which the keys represent
    cross-sections and the values are the model parameters.

    Ridge regression is a so-called regularized regression technique that minimizes the
    sum of the squared residuals subject to a restriction on the L2 norm of the coefficients.
    This restriction is controlled by the regularization hyperparameter `alpha`. The loss 
    function is called L2 loss with L2 regularization.
    """

    def __init__(
        self,
        roll: Union[int, str] = "full",
        alpha: float = 1.0,
        fit_intercept: bool = True,
        positive: bool = False,
        tol: float = 1e-4,
        solver: str = "lsqr",
        min_xs_samples: int = 2,
        data_freq: Optional[str] = None,
    ):
        # Checks
        self._check_init_params(
            alpha, fit_intercept, positive, tol, solver
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
        self.coefs_[section] = model.coef_
        self.intercepts_[section] = model.intercept_

    def _check_init_params(
        self,
        alpha,
        fit_intercept,
        positive,
        tol,
        solver,
    ):
        # alpha
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be either an integer or a float.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        # fit_intercept
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        # positive
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")
        # tol
        if not isinstance(tol, (int, float)):
            raise TypeError("tol must be either an integer or a float.")
        if tol <= 0:
            raise ValueError("tol must be a positive number.")
        # solver
        if not isinstance(solver, str):
            raise TypeError("solver must be a string.")
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
        if positive and solver not in ["lbfgs", "auto"]:
            raise ValueError(
                "solver must be one of 'lbfgs' or 'auto' when positive=True."
            )


class CorrelationVolatilitySystem(BaseRegressionSystem):
    """
    Cross-sectional system of moving average models to estimate correlation and volatility
    components of a macro beta separately over a panel of financial contracts.

    :param <Union[int, str]> correlation_lookback: The lookback period for the correlation
        calculation. If "full", the entire cross-sectional history is used. Otherwise,
        this parameter should be an integer specified in units of the native dataset
        frequency. If `data_freq` is not None, then an integer value for
        `correlation_lookback` should be expressed in units of the frequency specified
        in `data_freq`. Default is "full".
    :param <Union[int, str]> volatility_lookback: The lookback period for the volatility
        calculation. If "full", the entire cross-sectional history is used. Otherwise, 
        this parameter should be an integer specified in the native dataset frequency.
        If `data_freq` is not None, then an integer value for `volatility_lookback` should
        be expressed in units of the frequency specified in `data_freq`. Default is 21.
    :param <str> correlation_type: The type of correlation to be calculated.
        Accepted values are 'pearson', 'kendall' and 'spearman'. Default is 'pearson'.
    :param <str> volatility_window_type: The type of window to use for the volatility
        calculation. Accepted values are 'rolling' and 'exponential'. Default is 'rolling'.
    :param <int> min_xs_samples: The minimum number of samples required in a given 
        cross-section for a regression model to be fit for that cross-section.
        If `data_freq` is None, this parameter is specified in units of the underlying
        dataset frequency. Otherwise, this parameter should be expressed in units of
        the frequency specified in `data_freq`. Default is 2.
    :param <Optional[str]> data_freq: Training set data frequency. This is primarily
            to be used within the context of market beta estimation in the
            BetaEstimator class in `macrosynergy.learning`, allowing for cross-validation
            of the underlying dataset frequency for good beta estimation. Accepted strings
            are 'unadjusted' to use the native data set frequency, 'W' for weekly,
            'M' for monthly and 'Q' for quarterly. It is recommended to set this parameter
            to "W", "M" or "Q" only when the native dataset frequency is greater.
            Default is None.

    Notes
    -----
    From an implementation perspective, this class just estimates the local correlation 
    between time-varying independent and dependent variables, as well as the local
    standard deviations. Since a simple linear regression beta can be decomposed into the
    product of the correlation and the ratio of the standard deviations, separate estimation 
    of these quantities results in an estimator of the true beta. 
    """

    def __init__(
        self,
        correlation_lookback: Union[int, str] = "full",
        volatility_lookback: Union[int, str] = "full",
        correlation_type: str = "pearson",
        volatility_window_type: str = "rolling",
        min_xs_samples: int = 2,
        data_freq: Optional[str] = None,
    ):
        # Checks
        self._check_init_params(
            correlation_lookback,
            correlation_type,
            volatility_lookback,
            volatility_window_type,
        )

        self.correlation_lookback = correlation_lookback
        self.correlation_type = correlation_type
        self.volatility_lookback = volatility_lookback
        self.volatility_window_type = volatility_window_type
        self.data_freq = data_freq
        self.min_xs_samples = min_xs_samples

        super().__init__(
            roll="full",
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
        self.coefs_[section] = [beta]

    def create_model(self):
        raise NotImplementedError("This method is not implemented for this class.")

    def _check_xs_dates(self, min_xs_samples, num_dates):
        if num_dates < min_xs_samples:
            return False
        # If the correlation lookback is greater than the number of available dates, skip
        # to the next cross-section
        if (
            self.correlation_lookback is not None
            and self.correlation_lookback != "full"
            and num_dates < self.correlation_lookback
        ):
            return False
        # If the volatility lookback is greater than the number of available dates, skip
        # to the next cross-section
        if (
            self.volatility_lookback is not None
            and self.volatility_lookback != "full"
            and num_dates < self.volatility_lookback
        ):
            return False
        return True

    def _check_init_params(
        self,
        correlation_lookback,
        correlation_type,
        volatility_lookback,
        volatility_window_type,
    ):
        # correlation_lookback
        if not isinstance(correlation_lookback, (int, str)):
            raise TypeError("correlation_lookback must be an integer or string.")
        if (isinstance(correlation_lookback, int)) and (correlation_lookback <= 1):
            raise ValueError("correlation_lookback must be greater than 1 when an integer is specified.")
        if (not isinstance(correlation_lookback, str)) and (correlation_lookback != "full"):
            raise ValueError("correlation_lookback must equal `full` when a string is specified.")
        # volatility_lookback
        if not isinstance(volatility_lookback, (int, str)):
            raise TypeError("volatility_lookback must be an integer or string.")
        if (isinstance(volatility_lookback, int)) and (volatility_lookback <= 1):
            raise ValueError("volatility_lookback must be greater than 1 when an integer is specified.")
        if (not isinstance(volatility_lookback, str)) and (volatility_lookback != "full"):
            raise ValueError("volatility_lookback must equal `full` when a string is specified.")
        # correlation_type
        if not isinstance(correlation_type, str):
            raise TypeError("correlation_type must be a string.")
        if correlation_type not in ["pearson", "kendall", "spearman"]:
            raise ValueError(
                "correlation_type must be one of 'pearson', 'kendall' or 'spearman'."
            )
        # volatility_window_type
        if not isinstance(volatility_window_type, str):
            raise TypeError("volatility_window_type must be a string.")
        if volatility_window_type not in ["rolling", "exponential"]:
            raise ValueError(
                "volatility_window_type must be one of 'rolling' or 'exponential'."
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    import macrosynergy.management as msm
    from macrosynergy.management import make_qdf
    np.random.seed(1)

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "BENCH_XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

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

    X1 = dfd.drop(columns=["XR", "BENCH_XR"])
    y1 = dfd["XR"]

    # Demonstration of LADRegressionSystem usage
    model = LinearRegressionSystem()
    model.fit(X1, y1)
    print(f"OLS system coefficients: {model.coefs_}")
    print(f"OLS system intercepts: {model.intercepts_}")

    # Demonstration of LADRegressionSystem usage
    model = LADRegressionSystem()
    model.fit(X1, y1)
    print(f"LAD system coefficients: {model.coefs_}")
    print(f"LAD system intercepts: {model.intercepts_}")

    # Demonstration of RidgeRegressionSystem usage
    model = RidgeRegressionSystem()
    model.fit(X1, y1)
    print(f"Ridge system coefficients: {model.coefs_}")
    print(f"Ridge system intercepts: {model.intercepts_}")

    # Demonstration of CorrelationVolatilitySystem usage
    model = CorrelationVolatilitySystem()
    model.fit(X1, y1)
    print(model.coefs_)