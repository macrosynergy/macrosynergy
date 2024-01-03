import inspect

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

class SignWeightedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, **init_kwargs):
        """
        Custom predictor class to create a sign-weighted regression model. This class is
        specifically tailored for compatibility with any scikit-learn regressor that supports
        a 'sample_weight' parameter in its 'fit' method. Each training sample receives a weight 
        inversely proportional to the frequency of its label's sign in the training set. 
        Consequently, during the model training process, samples with a less common label sign
        are emphasized due to the higher weight they receive. By weighting based on inverse
        frequency, the model is encouraged to  equally learn from both positive and negative
        return samples. 
        """
        if not callable(regressor) or not inspect.isclass(regressor):
            raise TypeError("Regressor must be a class, such as 'LinearRegression'. Please check it isn't an instantiation of a class, such as 'LinearRegression()'.")

        fit_params = inspect.signature(regressor.fit).parameters
        if 'sample_weight' not in fit_params:
            raise ValueError(f"The underlying scikit-learn regressor {regressor} must have a 'sample_weight' parameter in its 'fit' method.")
        
        self.model = regressor(**init_kwargs)

    def __calculate_sample_weights(self, targets):
        pos_prop = np.mean(targets >= 0)
        neg_prop = np.mean(targets < 0)

        pos_weight = 1 / pos_prop if pos_prop > 0 else 0
        neg_weight = 1 / neg_prop if neg_prop > 0 else 0

        sample_weights = np.where(targets >= 0, pos_weight, neg_weight)
        return sample_weights 
    
    def fit(self, X, y):
        sample_weights = self.__calculate_sample_weights(y)
        self.model.fit(X, y, sample_weight=sample_weights)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
class TimeWeightedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, half_life, **init_kwargs):
        """
        Custom predictor class to create a time-weighted regression model. This class is
        specifically tailored for compatibility with any scikit-learn regressor that supports
        a 'sample_weight' parameter in its 'fit' method. Each training sample receives a weight 
        based on the date attributed to the sample. The weights are calculated as an exponentially
        decaying function of the date, with the half-life of the decay specified by the user.
        Consequently, during the model training process, more recent samples
        are emphasized due to the higher weight they receive. By weighting based on  
        recency, the model is encouraged to prioritise newer information. 
        """
        self.model = regressor(**init_kwargs)
        self.half_life = half_life

    def __calculate_sample_weights(self, targets):
        dates = sorted(targets.index.get_level_values(1).unique(), reverse=True)
        num_dates = len(dates)
        weights = np.exp(-np.log(2) * np.arange(num_dates)/self.half_life)
        weights = weights / np.sum()

        weight_map = dict(zip(dates, weights))
        sample_weights = targets.index.get_level_values(1).map(weight_map)

        return sample_weights
        
    def fit(self, X, y):
        sample_weights = self.__calculate_sample_weights(y)
        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X):
        return self.model.predict(X)