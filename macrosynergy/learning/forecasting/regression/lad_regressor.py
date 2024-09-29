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
        fit_intercept = True,
        positive = False,
        alpha = 0,
        shrinkage_type = "l1",
        tol = None,
    ):
        """
        Linear regression with L1 loss.
        
        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether or not to add an intercept to the model.
        positive: bool, default=False
            Whether or not to enforce non-negativity of model weights,
            with exception to a model intercept.
        alpha: float, default=0
            Shrinkage hyperparameter.
        shrinkage_type: str, default="l1"
            Type of shrinkage regularization to perform. 
        tol: float, default=None
            Tolerance for convergence of the learning algorithm (SLSQP).
            This is passed into the 'tol' parameter of the scipy.optimize.minimize
            function.

        Notes
        -----
        A dependent variable is modelled as a linear combination of the input features.
        The weights associated with each feature (and the intercept) are determined by
        finding the weights that minimise the average absolute model residuals. 

        If `alpha` is positive, then shrinkage-based regularization is applied to the
        non-intercept model coefficients. The type of shrinkage is determined by the
        `shrinkage_type` parameter. If `shrinkage_type` is "l1", then L1 regularization
        is applied. If `shrinkage_type` is "l2", then L2 regularization is applied.

        Mathematically, the following optimization problem is solved:

        .. math::

        \underset{w, b}{\text{argmin}} \frac{1}{N} \sum_{i=1}^{N} |y_{i} - x_{i}^{T}w - b| + 2N \times \alpha \times L_{p}(w)^{p}

        where:
            - N is the number of samples in the training data.
            - :math:`y_{i}` is the response for the i-th sample.
            - :math:`x_{i}` is the feature vector for the i-th sample.
            - :math:`w` is the vector of model coefficients.
            - :math:`b` is the model intercept.
            - :math:`L_{p}(w)` is the :math:`L_{p}` norm of the model coefficients, where :math:`p` is 1 or 2.
            - :math:`alpha` is the regularization hyperparameter.
        """
        # Checks
        self._check_init_params(fit_intercept, positive, alpha, shrinkage_type, tol)

        # Initialise
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.alpha = alpha
        self.shrinkage_type = shrinkage_type
        self.tol = tol

        # Set values of quantities to learn to None
        self.coef_ = None
        self.intercept_ = None

    def fit(
        self,
        X,
        y,
        sample_weight: np.ndarray = None,
    ):
        """
        Learn LAD regression model parameters. 

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input feature matrix.
        y : pd.Series or pd.DataFrame or np.ndarray
            Target vector associated with each sample in X.
        sample_weight : np.ndarray, default=None
            Numpy array of sample weights to create a weighted LAD regression model.
        """
        # Checks
        self._check_fit_params(X, y, sample_weight)

        # Fit
        X = X.copy()
        y = y.copy()

        if self.fit_intercept:
            X.insert(0, "intercept", 1)

        n_cols = X.shape[1]

        # Optimization bounds
        if self.positive:
            if self.fit_intercept:
                bounds = [(None, None)] + [(0, None)] * (n_cols-1) 
            else:
                bounds = [(0, None)] * n_cols
        else:
            bounds = [(None, None)] * n_cols 

        # Optimisation
        init_weights = np.zeros(n_cols)
        optim_results = minimize(
            fun=partial(
                self._l1_loss,
                X=X,
                y=y,
                sample_weight=sample_weight,
                alpha=self.alpha,
                shrinkage_type=self.shrinkage_type,
            ),
            x0=init_weights,
            method="SLSQP",
            bounds=bounds,
            tol=self.tol,
        )

        if not optim_results.success:
            raise RuntimeError(
                "LADRegressor optimization failed to converge. "
            )

        if self.fit_intercept:
            # Then store the intercept and feature weights
            self.intercept_ = optim_results.x[0]
            self.coef_ = optim_results.x[1:]
        else:
            self.intercept_ = None
            self.coef_ = optim_results.x

        return self

    def predict(self, X):
        """
        Predict dependent variable using the fitted LAD regression model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray
            Numpy array of predictions.
        """
        # Checks
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for the LADRegressor must be either a pandas dataframe "
                "or numpy array. If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )

        # Predict
        if self.coef_ is None:
            return np.zeros(X.shape[0])

        if self.fit_intercept:
            return (X.dot(self.coef_) + self.intercept_).values
        else:
            return X.dot(self.coef_).values

    def _l1_loss(
        self,
        weights,
        X,
        y,
        sample_weight = None,
        alpha: float = 0,
        shrinkage_type: str = "l1",
    ):
        """
        Determine L1 loss induced by 'weights'.
        
        Parameters
        ----------
        weights : np.ndarray
            LADRegressor model coefficients to be optimised.
        X : pd.DataFrame or np.ndarray
            Input features.
        y : pd.DataFrame or pd.Series or np.ndarray
            Targets associated with each sample in X.
        sample_weight : np.ndarray, default=None
            Sample weights to create a weighted LAD regression model.
        alpha : float, default=0
            Shrinkage hyperparameter.
        shrinkage_type : str, default="l1"
            Type of shrinkage regularization to perform.

        Returns
        -------
        loss : float
            L1 loss induced by 'weights'.
        """
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        if isinstance(y, pd.DataFrame):
            raw_residuals = y.iloc[:, 0] - X.dot(weights)
        else:  # y is a series
            raw_residuals = y - X.dot(weights)
        abs_residuals = np.abs(raw_residuals)
        weighted_abs_residuals = abs_residuals * sample_weight

        if alpha > 0:
            if shrinkage_type == "l1":
                if self.fit_intercept:
                    l1_norm = np.sum(np.abs(weights[1:]))
                else:
                    l1_norm = np.sum(np.abs(weights))
                return np.mean(weighted_abs_residuals) + 2 * X.shape[0] * alpha * l1_norm
            elif shrinkage_type == "l2":
                if self.fit_intercept:
                    l2_norm = np.sum(weights[1:] ** 2)
                else:
                    l2_norm = np.sum(weights ** 2)
                return np.mean(weighted_abs_residuals) + 2 * X.shape[0] * alpha * l2_norm

        return np.mean(weighted_abs_residuals)
    
    def _check_init_params(self, fit_intercept, positive, alpha, shrinkage_type, tol):
        """
        Checks for constructor parameters.
        """
        # fit_intercept
        if not isinstance(fit_intercept, bool):
            raise TypeError("The fit_intercept parameter must be a boolean.")
        # positive
        if not isinstance(positive, bool):
            raise TypeError("The positive parameter must be a boolean.")
        # alpha
        if not isinstance(alpha, numbers.Number):
            raise TypeError("The alpha parameter must be numeric.")
        if alpha < 0:
            raise ValueError("The alpha parameter must be non-negative.")
        # shrinkage_type
        if not isinstance(shrinkage_type, str):
            raise TypeError("The shrinkage_type parameter must be a string.")
        if shrinkage_type not in ["l1", "l2"]:
            raise ValueError(
                "The shrinkage_type parameter must be either 'l1' or 'l2'."
            )
        # tol
        if tol is not None:
            if not isinstance(tol, numbers.Number):
                raise TypeError("The tol parameter must be numeric.")
            if tol < 0:
                raise ValueError("The tol parameter must be non-negative.")

    def _check_fit_params(self, X, y, sample_weight):
        """
        Checks for fit method parameters.
        """
        # X
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for the LADRegressor must be a pandas dataframe or numpy array."
            )
        # y
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Dependent variable for the LADRegressor must be a pandas series, dataframe or numpy array."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "The dependent variable dataframe must have only one column. If used as part of "
                    "an sklearn pipeline, ensure that previous steps return a pandas "
                    "series or dataframe."
                )
        elif isinstance(y, np.ndarray):
            if y.ndim != 1:
                raise ValueError(
                    "The dependent variable numpy array must be 1D. If the dependent variable is 2D, "
                    "please either flatten the array or double check the contents of `y`."
                )
        if len(X) != len(y):
            raise ValueError(
                "The number of samples in the input feature matrix must match the number of samples in the dependent variable."
            )

        # sample_weight
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
