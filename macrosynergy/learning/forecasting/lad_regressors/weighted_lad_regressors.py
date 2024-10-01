import numpy as np
import pandas as pd

from macrosynergy.learning import LADRegressor
from macrosynergy.learning.forecasting import SignWeightedRegressor, TimeWeightedRegressor
from sklearn.base import BaseEstimator, RegressorMixin

import numbers

class SignWeightedLADRegressor(SignWeightedRegressor):
    def __init__(
        self,
        fit_intercept: bool = True,
        positive = False,
        alpha = 0,
        shrinkage_type = "l1",
        tol = None,
        maxiter = None,
    ):
        """
        LAD regressor with sign-weighted loss.

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
        maxiter: int, default=None
            Maximum number of iterations for the learning algorithm (SLSQP).
            This is passed into the 'maxiter' key of the options dictionary in the
            scipy.optimize.minimize function.


        Notes
        -----
        A dependent variable is modelled as a linear combination of the input features.
        The weights associated with each feature (and the intercept) are determined by
        finding the weights that minimise the weighted average absolute model residuals,
        where the weighted average is based on inverse frequency of the sign of the
        dependent variable.


        By weighting the contribution of different training samples based on the
        sign of the label, the model is encouraged to learn equally from both positive and
        negative return samples, irrespective of class imbalance. If there are more
        positive targets than negative targets in the training set, then the negative
        target samples are given a higher weight in the model training process.
        The opposite is true if there are more negative targets than positive targets.
        """
        super().__init__(
            model = LADRegressor(
                fit_intercept=fit_intercept,
                positive=positive,
                alpha=alpha,
                shrinkage_type=shrinkage_type,
                tol=tol,
                maxiter=maxiter,
            ),
        )

    def set_params(self, **params):
        super().set_params(**params)
        
        relevant_params = {"fit_intercept", "positive", "alpha", "shrinkage_type"}
        
        if relevant_params.intersection(params):
            self.model = LADRegressor(
                fit_intercept=self.fit_intercept,
                positive=self.positive,
                alpha=self.alpha,
                shrinkage_type=self.shrinkage_type
            )
        
        return self


class TimeWeightedLADRegressor(TimeWeightedRegressor):
    def __init__(
        self,
        fit_intercept = True,
        positive = False,
        half_life = 21 * 12,
        alpha = 0,
        shrinkage_type = "l1",
        tol = None,
        maxiter = None,
    ):
        """
        LAD regressor with time-weighted loss.

        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether or not to add an intercept to the model.
        positive: bool, default=False
            Whether or not to enforce non-negativity of model weights,
            with exception to a model intercept.
        half_life: int, default=21 * 12
            Half-life of the exponential decay function used to weight the
            contribution of different training samples based on the time of
            the sample.
        alpha: float, default=0
            Shrinkage hyperparameter.
        shrinkage_type: str, default="l1"
            Type of shrinkage regularization to perform.
        tol: float, default=None
            Tolerance for convergence of the learning algorithm (SLSQP).
            This is passed into the 'tol' parameter of the scipy.optimize.minimize
            function.
        maxiter: int, default=None
            Maximum number of iterations for the learning algorithm (SLSQP).
            This is passed into the 'maxiter' key of the options dictionary in the
            scipy.optimize.minimize function.

        Notes
        -----
        A dependent variable is modelled as a linear combination of the input features.
        The weights associated with each feature (and the intercept) are determined by
        finding the weights that minimise the weighted average absolute model residuals,
        where the weighted average is based on an exponential decay specified by the
        half-life parameter, where more recent samples are given higher weight.


        By weighting the contribution of different training samples based on the
        timestamp, the model is encouraged to prioritise more recent samples in the
        model training process. The half-life denotes the number of time periods in units
        of the native dataset frequency for the weight attributed to the most recent sample
        (one) to decay by half.
        """
        super().__init__(
            model = LADRegressor(
                fit_intercept=fit_intercept,
                positive=positive,
                alpha=alpha,
                shrinkage_type=shrinkage_type,
                tol=tol,
                maxiter=maxiter,
            ),
        )

    def set_params(self, **params):
        super().set_params(**params)
        
        relevant_params = {"fit_intercept", "positive", "alpha", "shrinkage_type"}
        
        if relevant_params.intersection(params):
            self.model = LADRegressor(
                fit_intercept=self.fit_intercept,
                positive=self.positive,
                alpha=self.alpha,
                shrinkage_type=self.shrinkage_type
            )
        
        return self
