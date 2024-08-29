"""
Base class for a panel feature selection module.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.exceptions import NotFittedError

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
        Returns a boolean mask of the features selected.
        """
        mask = np.zeros(self.p, dtype=bool)
        mask[self.selected_ftr_idxs] = True
        return mask
    
    def get_support(self, indices=False):
        """
        Returns either a boolean mask or integer index
        of the features selected.
        """
        # Checks
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The selector has not been fit.",
                "Please fit the selector before calling get_support()."
            )
        if not isinstance(indices, (bool, np.bool_)):
            raise ValueError("The 'indices' parameter must be a boolean.")
        
        # Return mask or index
        if indices:
            return self.selected_ftr_idxs
        else:
            mask = self._get_support_mask()
            return mask
        
    def get_feature_names_out(self):
        """
        Masks feature names according to selected features.
        """
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The selector has not been fitted.",
                "Please fit the selector before calling get_feature_names_out()."
            )

        return self.feature_names_in_[self.get_support(indices=False)]

    def transform(self, X: pd.DataFrame):
        """
        Transforms a dataframe to only utilise selected features. 
        """
        # Checks
        self._check_transform_params(X)

        if len(self.selected_ftr_idxs) == 0:
            return X.iloc[:, :0]

        return X.iloc[:, self.selected_ftr_idxs]
    
    def _check_fit_params(X, y):
        pass

    def _check_transform_params(X):
        pass