"""
Scikit-learn compatible feature selection classes.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoLars, ElasticNet, lars_path

from macrosynergy.learning.transformers.selectors import BasePanelSelector
from macrosynergy.learning.transformers.transformers import PanelStandardScaler
from abc import ABC, abstractmethod


class LassoSelector(BasePanelSelector):
    """
    Feature selection using the LASSO. 
    """
    def __init__(self, fit_intercept = True, n_factors = 10, positive = False):
        self.n_factors = n_factors
        self.fit_intercept = fit_intercept
        self.positive = positive

        super().__init__()

    def _select_features(self, X, y):
        # First scale the features for fair feature comparison
        X = PanelStandardScaler().fit_transform(X).copy()
        
        # Compute the LARS path for Lasso
        # coefs_path is a matrix of shape (n_features, n_alphas)
        _, _, coefs_path = lars_path(X.to_numpy(), y.to_numpy(), method='lasso')

        column_indices = np.where(coefs_path[:,self.n_factors] != 0)[0]
        
        return column_indices
