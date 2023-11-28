import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class NaivePredictor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        return self
 
    def predict(self, X):
        return np.array(X)