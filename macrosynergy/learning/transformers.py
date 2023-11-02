"""
Collection of scikit-learn transformer classes.

**NOTE: This module is under development, and is not yet ready for production use.**
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, TransformerMixin

class LassoSelectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.lasso = Lasso(alpha=self.alpha).fit(X, y)
        self.selected_ftr_idxs = [i for i in range(len(self.lasso.coef_)) if self.lasso.coef_[i] != 0]

    def transform(self, X):
        
        return X.iloc[:,self.selected_ftr_idxs]