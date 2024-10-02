import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class NaiveRegressor(BaseEstimator, RegressorMixin):
    """
    Equally weighted unbiased factor model.

    Notes
    -----
    Given a collection of factors that are theoretically positively correlated with a 
    dependent variable, a plausible signal is a simple average of those factors. This is 
    effectively a linear regression model with zero intercept and equal weights for all
    factors. 

    This is a useful benchmark model which works well when the factors are as 
    uncorrelated as possible with one another, because it offers a layer of
    diversification on the underlying return drivers. When the user has strong priors,
    this is often a competitive model that is difficult to beat.
    
    However, it is vital for the features to have been preprocessed to have a positive
    theoretical correlation with the target variable.
    """

    def fit(self, X, y=None):
        """
        Fit method.

        Parameters
        ----------
        X : pandas DataFrame, pandas Series or numpy array
            The input feature matrix.
        y : pandas Series or numpy array
            The target variable.
        Notes
        -----
        This method involves fully trusting one's priors and thus requires no learning 
        element. As a consequence, no training set information is needed.
        """
        # Checks
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError("X must be a pandas DataFrame, pandas Series or numpy array")
        elif isinstance(X, np.ndarray) and ((X.ndim > 2) or (X.ndim < 1)):
            raise ValueError(
                "When X is a numpy array, it must have either 1 or 2 dimensions."
            )
        
        # No learning needed since priors are fully trusted
        return self

    def predict(self, X):
        """
        Predict method.

        Notes
        -----
        The predictions are simply the average of the features across columns of the input
        feature matrix.
        """
        # Checks
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError("X must be a pandas DataFrame, pandas Series or numpy array")
        elif isinstance(X, np.ndarray) and ((X.ndim > 2) or (X.ndim < 1)):
            raise ValueError(
                "When X is a numpy array, it must have either 1 or 2 dimensions."
            )
        
        # Return the naive signal
        if isinstance(X, pd.DataFrame):
            return np.mean(X.values, axis=1)
        elif isinstance(X, pd.Series):
            return X.values
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            return X 
        else:
            return np.mean(X, axis=1)