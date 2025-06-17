import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin

class DataFrameTransformer(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    """
    Meta estimator to reconvert a transformed numpy array back to a pandas DataFrame.
    This maintains the multi-indexed panel structure.

    Parameters
    ----------
    transformer : TransformerMixin
        A scikit-learn transformer with a fit and transform method.
        
    Notes
    -----
    Many scikit-learn compatible transformers convert pandas DataFrames to numpy arrays.
    This can be problematic when working with panel models that require knowledge of the
    panel structure. This class wraps around such transformers to ensure that the output
    is a pandas DataFrame, preserving the original index.

    When no column names are provided, default names of the form "Factor_0", "Factor_1", etc.
    are used for the transformed DataFrame. If column names are provided, they will be used
    instead.
    """
    def __init__(self, transformer, column_names=None):
        self.transformer = transformer
        self.column_names = column_names

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of input features.
        y : pd.Series or pd.DataFrame or np.ndarray
            Pandas series, dataframe or numpy array of targets associated with each sample
            in X.
        """
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        """
        Transform the input data based on the underlying transformer, but return a 
        pandas DataFrame instead of a numpy array.

        Parameters
        ----------
        X : pd.DataFrame or numpy array
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            Transformed data as a pandas DataFrame, preserving the original index and
            using either provided column names or default names.
        """
        transformation = self.transformer.transform(X)
        if isinstance(transformation, pd.DataFrame):
            return transformation
        else:
            # scikit-learn returns a numpy array, convert it back to DataFrame
            if self.column_names is None:
                columns = [f"Factor_{i}" for i in range(transformation.shape[1])]
            else:
                columns = self.column_names
            return pd.DataFrame(
                data = transformation,
                columns = columns,
                index = X.index,
            )
        
    def __getattr__(self, name):
        # Avoid recursion if 'transformer' is not yet set
        if name == "transformer":
            raise AttributeError()
        return getattr(self.transformer, name)
