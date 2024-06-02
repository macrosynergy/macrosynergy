import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class NaivePredictor(BaseEstimator, RegressorMixin):
    """
    Naive predictor class to output a column of features as the predictions themselves.
    It expects only one column in the input matrix for which predictions are to be made.
    This class would typically be used in the final stage of a `scikit-learn` pipeline,
    following either `FeatureAverager` or `ZnScoreAverager` stages.
    """

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.array(X)
