import inspect

import numpy as np
import pandas as pd

import datetime

from sklearn.base import BaseEstimator, RegressorMixin

from typing import Union


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
    def __init__(self, regressor: RegressorMixin, **init_kwargs):
        """
        Custom predictor class to create a regression model where different training samples
        are weighted differently depending on the sign of their associated targets. If there
        are more positive targets than negative targets in the training set, then the 
        negative target samples are given a higher weight in the model training process.
        The opposite is true if there are more negative targets than positive targets.
        
        NOTE: This class is specifically tailored for compatibility with any scikit-learn
        regressor that supports a 'sample_weight' parameter in its 'fit' method. Each
        training sample receives a weight inversely proportional to the frequency of its
        label's sign in the training set. Thus, the model is encouraged to learn equally
        from both positive and negative return samples. 
        """
        if not callable(regressor) or not inspect.isclass(regressor):
            raise TypeError("Regressor must be a class, such as 'LinearRegression'. Please check it isn't an instantiation of a class, such as 'LinearRegression()'.")

        if not issubclass(regressor, RegressorMixin) or not issubclass(regressor, BaseEstimator):
            raise TypeError("Regressor must be a subclass of both BaseEstimator and RegressorMixin.")
        
        if not hasattr(regressor, 'fit') or not callable(getattr(regressor, 'fit')):
            raise TypeError("Regressor must have a callable 'fit' method.")
        
        if not hasattr(regressor, 'predict') or not callable(getattr(regressor, 'predict')):
            raise TypeError("Regressor must have a callable 'predict' method.")

        fit_params = inspect.signature(regressor.fit).parameters
        if 'sample_weight' not in fit_params:
            raise ValueError(f"The underlying scikit-learn regressor {regressor} must have a 'sample_weight' parameter in its 'fit' method.")
        
        self.model = regressor(**init_kwargs)

    def __calculate_sample_weights(self, targets: Union[pd.DataFrame, pd.Series]):
        """
        Private helper method to calculate the sample weights, chosen by inverse frequency
        of the label's sign in the training set.

        :param <Union[pd.DataFrame, pd.Series]> targets: Pandas series or dataframe of targets.

        :return <tuple[np.ndarray, float, float]>: Tuple of sample weights, positive weight and negative weight.
        """
        pos_sum = np.sum(targets >= 0)
        neg_sum = np.sum(targets < 0)

        pos_weight = len(targets)/(2*pos_sum) if pos_sum > 0 else 0
        neg_weight = len(targets)/(2*neg_sum) if neg_sum > 0 else 0

        sample_weights = np.where(targets >= 0, pos_weight, neg_weight)
        return sample_weights, pos_weight, neg_weight 
    
    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        """
        Fit method to fit the underlying model with sign weighted samples, as passed
        into the constructor. 

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.

        :return <SignWeightedRegressor>
        """
        # Checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the SignWeightedRegressor must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if (type(y) != pd.Series) and (type(y) != pd.DataFrame):
            raise TypeError(
                "Target vector for the SignWeightedRegressor must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if type(y) == pd.DataFrame:
            if y.shape[1] != 1:
                raise ValueError(
                    "The target dataframe must have only one column. If used as part of "
                    "an sklearn pipeline, ensure that previous steps return a pandas "
                    "series or dataframe."
                )
            
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of y must be datetime.date.")
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't "
                "match."
            )
        
        # Fit
        self.sample_weights, _, _ = self.__calculate_sample_weights(y)
        self.model.fit(X, y, sample_weight=self.sample_weights)
        return self
    
    def predict(self, X: pd.DataFrame):
        """
        Predict method to make model predictions on the input feature matrix X based on 
        the previously fit model. 

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <np.ndarray>: Numpy array of predictions.
        """
        # Checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the SignWeightedRegressor must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        
        # Logic 
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