import datetime
import numbers
import warnings
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin


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
            raw_residuals = y.values[:,0] - X.dot(weights).values
        else:  # y is a series
            raw_residuals = y.values - np.dot(X.values,weights)
        abs_residuals = np.abs(raw_residuals)
        weighted_abs_residuals = abs_residuals * sample_weight

        return np.mean(weighted_abs_residuals)
        
if __name__ == "__main__":
    from sklearn.metrics import make_scorer

    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf
    import timeit

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {
        "GBP": (
            pd.Timestamp(year=2009, month=1, day=1),
            pd.Timestamp(year=2012, month=6, day=30),
        ),
        "CAD": (
            pd.Timestamp(year=2015, month=1, day=1),
            pd.Timestamp(year=2100, month=1, day=1),
        ),
    }

    train = msm.categories_df(
        df=dfd, xcats=xcats, cids=cids, val="value", blacklist=black, freq="M", lag=1
    ).dropna()

    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]

    # Fit LAD regression 1 
    lad = LADRegressor()
    start_time = timeit.default_timer()
    lad.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time for LAD regressor V1: ", elapsed)
    print("LAD regressor V1 coefficients: ", lad.coef_)


    # Profiling
    from pyinstrument import Profiler
    from macrosynergy.learning import LADRegressionSystem

    def profile_model_fitting(X, y):
        model = LADRegressionSystem(fit_intercept=True, positive=False)
        model.fit(X, y)

    profiler = Profiler()
    profiler.start()
    profile_model_fitting(X_train, y_train)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
