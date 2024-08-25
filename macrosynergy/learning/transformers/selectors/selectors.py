"""
Scikit-learn compatible feature selection classes.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, SelectorMixin

from abc import ABC, abstractmethod

class BasePanelSelector(SelectorMixin, BaseEstimator, ABC):
    """
    Base class for statistical feature selection over a panel. 
    """
    def __init__(self):
        self.feature_names_in_ = None

    def fit(self, X, y = None):
        # Checks
        self._check_fit_params(X, y)

        self.feature_names_in_ = np.array(X.columns)
        self.n = X.shape[0]
        self.p = X.shape[1]

        self.selected_ftr_idxs = self._select_features(X, y)

    @abstractmethod
    def _select_features(self, X, y):
        """
        Determines column indices of X to be selected.
        """
        pass

    def _get_support_mask(self):
        """
        Return a boolean mask of the features selected for the Pandas dataframe.
        """
        mask = np.zeros(self.p, dtype=bool)
        mask[self.selected_ftr_idxs] = True
        return mask
    
    def get_support(self, indices=False):
        """
        Return a mask, or integer index, of the features selected for the Pandas dataframe.
        """
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The Elastic Net selector has not been fitted. Please fit the selector before calling get_support()."
            )
        if not isinstance(indices, (bool, np.bool_)):
            raise ValueError("The 'indices' parameter must be a boolean.")
        if indices:
            return self.selected_ftr_idxs
        else:
            mask = self._get_support_mask()
            return mask

    def transform(self, X):
        pass
