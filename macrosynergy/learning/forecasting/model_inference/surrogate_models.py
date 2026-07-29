import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lars

# TODO: write a general surrogate model in the package from which this inherits
class LarsSurrogateModel(BaseEstimator, RegressorMixin):
    """
    Surrogate model to infer the behaviour of a model through a LARS regression on the model's
    predictions based on the input features. 

    Parameters
    ----------
    estimator : sklearn regressor
        The model to be explained by the surrogate model.
    n_factors : int, default=3
        The number of non-zero coefficients to select in the LARS regression.
    eps : float, default=1e-6
        Machine precision for LARS regression.
    Notes
    -----
    Surrogate models are used to explain the behaviour of complex models by regressing 
    a collection of features on the model's predictions. A key property is that the model
    is explainable. 

    Future versions of this class will allow for a separate factor set to be used 
    for inspection.
    """
    def __init__(self, estimator, n_factors = 3, eps = 1e-6):
        self.estimator = estimator
        self.n_factors = n_factors
        self.eps = eps

    def fit(self, X, y):
        # TODO: clone the estimator
        # TODO: get this to work for single output models too
        self.estimator.fit(X, y)
        signals = self.estimator.predict(X)
        
        # Get the return induced by each signal as well as the portfolio return
        returns = signals * y
        portfolio_returns = returns.sum(axis = 1)
        
        # A LARS regression on the portfolio returns pick up the factors that affect the global portfolio returns
        # TODO: HMMMM think on this more
        global_lars = Lars(n_nonzero_coefs = self.n_factors, fit_intercept = True, eps = self.eps).fit(X, portfolio_returns)
        # LARS regressions on the sector predictions informs on why the predictions were made
        local_lars = {
            target: Lars(n_nonzero_coefs = self.n_factors, fit_intercept = True, eps = self.eps).fit(X, signals[target])
            for target in signals.columns
        }
        self.feature_importances_ = global_lars.coef_ 
        self.features_selected_ = X.columns[
            np.abs(self.feature_importances_) != 0
        ].tolist()
        self.feature_importances_per_asset_ = {
            target: local_lars[target].coef_
            for target in signals.columns
        }
        self.features_selected_per_asset_ = {
            target: X.columns[
                np.abs(local_lars[target].coef_) != 0
            ].tolist()
            for target in signals.columns
        }
        
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def __getattr__(self, name):
        # Needed so that signal optimizer can store the local feature importances correctly
        if "__" not in name:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute {name!r}"
            )

        parts = name.split("__")

        # First part, e.g. "estimator"
        first = parts[0]

        # Prefer fitted attribute estimator_ over constructor attribute estimator
        fitted_first = first + "_"

        if fitted_first in self.__dict__:
            obj = self.__dict__[fitted_first]
        elif first in self.__dict__:
            obj = self.__dict__[first]
        else:
            raise AttributeError(
                f"{self.__class__.__name__!r} object has no attribute "
                f"{first!r} or {fitted_first!r}"
            )

        # Remaining parts, e.g. ["weights_"]
        for part in parts[1:]:
            obj = getattr(obj, part)

        return obj