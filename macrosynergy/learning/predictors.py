import numpy as np
import pandas as pd

from scipy.optimize import minimize

from functools import partial
import warnings

import datetime

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from typing import Union
from abc import abstractmethod, ABC

import numbers


class NaivePredictor(BaseEstimator, RegressorMixin):
    """
    Naive predictor class to output a column of features as the predictions themselves.
    It expects only one column in the input matrix for which predictions are to be made.
    This class would typically be used in the final stage of a `scikit-learn` pipeline,
    following either `FeatureAverager` or `ZnScoreAverager` stages.
    """

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.array(X)


class BaseWeightedRegressor(BaseEstimator, RegressorMixin, ABC):
    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        """
        Fit method to fit the underlying model with weighted samples, as passed
        into the constructor.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.

        :return <BaseWeightedRegressor>
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

        # Fit
        self.sample_weights = self._calculate_sample_weights(y)
        self.model.fit(X, y, sample_weight=self.sample_weights)
        if hasattr(self.model, "coef_"):
            self.coef_ = self.model.coef_
        if hasattr(self.model, "intercept_"):
            self.intercept_ = self.model.intercept_

        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict method to make model predictions on the input feature matrix X based on
        the previously fit model.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <np.ndarray>: Numpy array of predictions.
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the SignWeightedLinearRegression must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")

        # Predict
        return self.model.predict(X)

    @abstractmethod
    def _calculate_sample_weights(self, y: Union[pd.DataFrame, pd.Series]):
        """
        Abstract method for calculating sample weights based on the target variable `y`.
        """
        pass


class SignWeightedRegressor(BaseWeightedRegressor):

    def __init__(self, model: BaseEstimator):
        """
        Custom class to create a weighted regressor, with the sample weights
        chosen by inverse frequency of the label's sign in the training set.

        :param <BaseEstimator> model: The underlying model to be trained with weighted samples.

        NOTE: By weighting the contribution of different training samples based on the
        sign of the label, the model is encouraged to learn equally from both positive and negative return samples,
        irrespective of class imbalance. If there are more positive targets than negative
        targets in the training set, then the negative target samples are given a higher
        weight in the model training process. The opposite is true if there are more
        negative targets than positive targets.
        """
        self.model = model

    def _calculate_sample_weights(self, targets: Union[pd.DataFrame, pd.Series]):
        """
        Private helper method to calculate the sample weights, chosen by inverse frequency
        of the label's sign in the training set.

        :param <Union[pd.DataFrame, pd.Series]> targets: Pandas series or dataframe of targets.

        :return <tuple[np.ndarray, float, float]>: Tuple of sample weights, positive weight and negative weight.
        """
        pos_sum = np.sum(targets >= 0)
        neg_sum = np.sum(targets < 0)

        pos_weight = len(targets) / (2 * pos_sum) if pos_sum > 0 else 0
        neg_weight = len(targets) / (2 * neg_sum) if neg_sum > 0 else 0

        sample_weights = np.where(targets >= 0, pos_weight, neg_weight)
        return sample_weights


class TimeWeightedRegressor(BaseWeightedRegressor):
    def __init__(self, half_life: Union[float, int], model: BaseEstimator):
        """
        Custom class to create a weighted regressor, where the training sample
        weights exponentially decay by sample recency, given a prescribed half_life.

        :param <Union[float, int]> half_life: The number of time periods in units of the native data frequency for the weight attributed to the most recent sample (one) to decay by half.
        :param <BaseEstimator> model: The underlying model to be trained with weighted samples.

        NOTE: By weighting the contribution of different training samples based on the
        observation date, the model is encouraged to prioritise newer information.
        The half-life denotes the number of time periods in units of the native data
        frequency for the weight attributed to the most recent sample (one) to decay by half.
        """
        if not isinstance(half_life, (float, int)):
            raise TypeError("half_life must be a float or an integer.")
        if half_life <= 1:
            raise ValueError("The half-life must be greater than 1.")
        self.half_life = half_life
        self.model = model

    def _calculate_sample_weights(self, targets: Union[pd.DataFrame, pd.Series]):
        """
        Private helper method to calculate the sample weights, chosen by an exponentially
        decaying function of the sample recency.

        :param <Union[pd.DataFrame, pd.Series]> targets: Pandas series or dataframe of targets.

        :return <np.ndarray>: Numpy array of sample weights.
        """

        dates = sorted(targets.index.get_level_values(1).unique(), reverse=True)
        num_dates = len(dates)
        weights = np.power(2, -np.arange(num_dates) / self.half_life)

        weight_map = dict(zip(dates, weights))
        self.sample_weights = (
            targets.index.get_level_values(1).map(weight_map).to_numpy()
        )

        return self.sample_weights


class SignWeightedLinearRegression(SignWeightedRegressor):
    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: int = None,
        positive: bool = False,
    ):
        """
        Custom class to create a WLS linear regression model, with the sample weights
        chosen by inverse frequency of the label's sign in the training set.

        :param <bool> fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param <bool> copy_X: If True, X will be copied; else, it may be overwritten.
        :param <int> n_jobs: The number of jobs to use for the computation.
        :param <bool> positive: When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.

        NOTE: By weighting the contribution of different training samples based on the
        sign of the label, the model is encouraged to learn equally from both positive and negative return samples,
        irrespective of class imbalance. If there are more positive targets than negative
        targets in the training set, then the negative target samples are given a higher
        weight in the model training process. The opposite is true if there are more
        negative targets than positive targets.
        """

        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(copy_X, bool):
            raise TypeError("copy_X must be a boolean.")
        if not isinstance(n_jobs, int) and n_jobs != None:
            raise TypeError("n_jobs must be an integer or None.")
        if n_jobs is not None:
            if n_jobs < -1 or n_jobs == 0:
                raise ValueError("n_jobs must be a positive integer or -1.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")

        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        model = LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            positive=self.positive,
        )
        self.coef_ = None
        self.intercept_ = None
        super().__init__(model)

    def set_params(self, **params):
        super().set_params(**params)
        if "fit_intercept" in params or "positive" in params:
            # Re-initialize the LinearRegression instance with updated parameters
            self.model = LinearRegression(
                fit_intercept=self.fit_intercept, positive=self.positive
            )

        return self


class TimeWeightedLinearRegression(TimeWeightedRegressor):
    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: int = None,
        positive: bool = False,
        half_life: Union[float, int] = 21 * 12,
    ):
        """
        Custom class to create a WLS linear regression model, where the training sample
        weights exponentially decay by sample recency, given a prescribed half_life.

        :param <bool> fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param <bool> copy_X: If True, X will be copied; else, it may be overwritten.
        :param <int> n_jobs: The number of jobs to use for the computation.
        :param <bool> positive: When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.
        :param <Union[float, int]> half_life: The number of time periods in units of the native data frequency for the weight attributed to the most recent sample (one) to decay by half.
        """
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(copy_X, bool):
            raise TypeError("copy_X must be a boolean.")
        if not isinstance(n_jobs, int) and n_jobs != None:
            raise TypeError("n_jobs must be an integer or None.")
        if n_jobs is not None:
            if n_jobs < -1 or n_jobs == 0:
                raise ValueError("n_jobs must be a positive integer or -1.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")

        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        model = LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            positive=self.positive,
        )
        self.coef_ = None
        self.intercept_ = None
        super().__init__(half_life=half_life, model=model)

    def set_params(self, **params):
        super().set_params(**params)
        if "fit_intercept" in params or "positive" in params:
            # Re-initialize the LinearRegression instance with updated parameters
            self.model = LinearRegression(
                fit_intercept=self.fit_intercept, positive=self.positive
            )

        return self


class SignWeightedLADRegressor(SignWeightedRegressor):
    def __init__(
        self,
        fit_intercept: bool = True,
        positive: bool = False,
    ):
        """
        Custom class to create a weighted LAD linear regression model, with the sample weights
        chosen by inverse frequency of the label's sign in the training set.

        :param <bool> fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculation.
        :param <bool> positive: When set to True, forces the coefficients to be positive.

        NOTE: By weighting the contribution of different training samples based on the
        sign of the label, the model is encouraged to learn equally from both positive and negative return samples,
        irrespective of class imbalance. If there are more positive targets than negative
        targets in the training set, then the negative target samples are given a higher
        weight in the model training process. The opposite is true if there are more
        negative targets than positive targets.
        """

        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")

        self.fit_intercept = fit_intercept
        self.positive = positive
        model = LADRegressor(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
        )
        self.coef_ = None
        self.intercept_ = None
        super().__init__(model)

    def set_params(self, **params):
        super().set_params(**params)
        if "fit_intercept" in params or "positive" in params:
            # Re-initialize the LinearRegression instance with updated parameters
            self.model = LADRegressor(
                fit_intercept=self.fit_intercept, positive=self.positive
            )

        return self


class TimeWeightedLADRegressor(TimeWeightedRegressor):
    def __init__(
        self,
        fit_intercept: bool = True,
        positive: bool = False,
        half_life: Union[float, int] = 21 * 12,
    ):
        """
        Custom class to create a weighted LAD linear regression model, where the training sample
        weights exponentially decay by sample recency, given a prescribed half_life.

        :param <bool> fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
        :param <bool> positive: When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.
        :param <Union[float, int]> half_life: The number of time periods in units of the native data frequency for the weight attributed to the most recent sample (one) to decay by half.
        """
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")

        self.fit_intercept = fit_intercept
        self.positive = positive
        model = LADRegressor(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
        )
        self.coef_ = None
        self.intercept_ = None
        super().__init__(half_life=half_life, model=model)

    def set_params(self, **params):
        super().set_params(**params)
        if "fit_intercept" in params or "positive" in params:
            # Re-initialize the LinearRegression instance with updated parameters
            self.model = LADRegressor(
                fit_intercept=self.fit_intercept, positive=self.positive
            )

        return self


class LADRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        fit_intercept=True,
        positive=False,
        tol=None,
    ):
        """
        Custom class to create a linear regression model with model fit determined
        by minimising L1 (absolute) loss.

        :param <bool> fit_intercept: Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations.
        :param <bool> positive: If True, the optimisation is restricted so that model
            weights (other than the intercept) are greater or equal to zero.
            This is a way of incorporating prior knowledge about the relationship between
            model features and corresponding targets into the model fit.
        :param <float> tol: The tolerance for termination. Default is None.

        NOTE: Standard OLS linear regression models are fit by minimising the residual
            sum of squares. Our implemented (L)east (A)bsolute (D)eviation regression
            model fits a linear model by minimising the residual absolute deviations.
            Consequently, the model fit is more robust to outliers than an OLS fit.
        """
        # Checks
        if not isinstance(fit_intercept, bool):
            raise TypeError("'fit_intercept' must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("'positive' must be a boolean.")
        if tol is not None and not isinstance(tol, (int, float)):
            raise TypeError("'tol' must be a float or int.")
        if tol is not None and tol <= 0:
            raise ValueError("'tol' must be a positive number.")

        # Initialise
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        sample_weight: np.ndarray = None,
    ):
        """
        Fit method to fit a LAD linear regression model.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        :param <np.ndarray> sample_weight: Numpy array of sample weights to create a
            weighted LAD regression model. Default is None for equal sample weights.

        :return <LADRegressor>
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the LADRegressor must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            raise TypeError(
                "Target vector for the LADRegressor must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
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

        if sample_weight is not None:
            if not isinstance(sample_weight, np.ndarray):
                try:
                    sample_weight = sample_weight.to_numpy().flatten()
                except Exception as e:
                    try:
                        sample_weight = np.array(sample_weight).flatten()
                    except Exception as e:
                        raise TypeError(
                            "The sample weights must be contained within a numpy array."
                        )
            if sample_weight.ndim != 1:
                raise ValueError("The sample weights must be a 1D numpy array.")
            for w in sample_weight:
                if not isinstance(w, numbers.Number):
                    raise TypeError(
                        "All elements of the sample weights must be numeric."
                    )
            if len(sample_weight) != X.shape[0]:
                raise ValueError(
                    "The number of sample weights must match the number of samples in the input feature matrix."
                )

        # Fit
        X = X.copy()
        y = y.copy()

        if self.fit_intercept:
            X.insert(0, "intercept", 1)

        n_cols = X.shape[1]

        if self.positive:
            if self.fit_intercept:
                bounds = [(None, None)] + [(0, None) for _ in range(n_cols - 1)]
            else:
                bounds = [(0, None) for _ in range(n_cols)]
        else:
            bounds = [(None, None) for _ in range(n_cols)]

        # optimisation
        init_weights = np.zeros(n_cols)
        optim_results = minimize(
            fun=partial(
                self._l1_loss,
                X=X,
                y=y,
                sample_weight=sample_weight,
            ),
            x0=init_weights,
            method="SLSQP",  # TODO: make this an option in the constructor.
            bounds=bounds,
            tol=self.tol,
        )

        if not optim_results.success:
            self.intercept_ = None
            self.coef_ = None
            warnings.warn(
                "LADRegressor optimisation failed. Setting intercept and coefficients to None.",
                RuntimeWarning,
            )
            return self

        if self.fit_intercept:
            # Then store the intercept and feature weights
            self.intercept_ = optim_results.x[0]
            self.coef_ = optim_results.x[1:]
        else:
            self.intercept_ = None
            self.coef_ = optim_results.x

        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict method to make model predictions on the input feature matrix X based on
        the previously fit model.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <np.ndarray>: Numpy array of predictions.
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the LADRegressor must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")

        # Predict
        if self.intercept_ is None and self.coef_ is None:
            warnings.warn(
                "LADRegressor model fit failed. Predicting a zero signal.",
                RuntimeWarning,
            )
            return pd.Series(0, index=X.index)

        X = X.copy()
        if self.fit_intercept:
            return (X.dot(self.coef_) + self.intercept_).values
        else:
            return X.dot(self.coef_).values

    def _l1_loss(
        self,
        weights: np.ndarray,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        sample_weight: np.ndarray = None,
    ):
        """
        Private helper method to determine the mean training L1 loss induced by 'weights'.

        :param <np.ndarray> weights: LADRegressor model coefficients to be optimised.
        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        :param <np.ndarray> sample_weight: Numpy array of sample weights to create a
            weighted LAD regression model. Default is None for equal sample weights.

        :return <float>: Training loss induced by 'weights'.
        """
        # Checks
        if not isinstance(weights, np.ndarray):
            raise TypeError("The weights must be contained within a numpy array.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the LADRegressor must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            raise TypeError(
                "Target vector for the LADRegressor must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
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

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if not isinstance(sample_weight, np.ndarray):
            raise TypeError(
                "The sample weights must be contained within a numpy array."
            )
        if sample_weight.ndim != 1:
            raise ValueError("The sample weights must be a 1D numpy array.")
        if len(sample_weight) != X.shape[0]:
            raise ValueError(
                "The number of sample weights must match the number of samples in the input feature matrix."
            )

        if isinstance(y, pd.DataFrame):
            raw_residuals = y.iloc[:, 0] - X.dot(weights)
        else:  # y is a series
            raw_residuals = y - X.dot(weights)
        abs_residuals = np.abs(raw_residuals)
        weighted_abs_residuals = abs_residuals * sample_weight

        return np.mean(weighted_abs_residuals)


if __name__ == "__main__":
    from macrosynergy.management import make_qdf
    import macrosynergy.management as msm
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LinearRegression

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

    # OLS Linear Regression
    lm = LinearRegression()
    lm.fit(X, y)
    lm_preds = lm.predict(X)
    # Sign-Weighted Linear Regression
    swlm = SignWeightedLinearRegression()
    swlm.fit(X, y)
    swlm_preds = swlm.predict(X)
    # Time-Weighted Linear Regression
    twlm = TimeWeightedLinearRegression(half_life=12 * 21)  # One year half life
    twlm.fit(X, y)
    twlm_preds = twlm.predict(X)

    # Plot results
    plt.title(
        "OLS vs Sign-Weighted vs Time-Weighted Linear Regression on mock random data"
    )
    plt.plot(lm_preds, label="OLS", color="tab:blue", alpha=0.5)
    plt.plot(swlm_preds, label="Sign-Weighted", color="tab:orange", alpha=0.5)
    plt.plot(twlm_preds, label="Time-Weighted", color="tab:green", alpha=0.5)

    plt.legend()
    plt.show()

    # Plot histogram of predictions
    fig, ax = plt.subplots(ncols=3)
    ax[0].hist(lm_preds, bins=200, color="tab:blue")
    ax[0].set_title("OLS")
    ax[1].hist(swlm_preds, bins=200, color="tab:orange")
    ax[1].set_title("Sign-Weighted")
    ax[2].hist(twlm_preds, bins=200, color="tab:green")
    ax[2].set_title("Time-Weighted")

    plt.suptitle("Histograms of predictions")
    plt.tight_layout()
    plt.show()
