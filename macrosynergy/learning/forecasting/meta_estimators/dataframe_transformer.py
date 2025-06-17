import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin

class DataFrameTransformer(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    """
    Meta estimator to reconvert a transformed numpy array back to a pandas DataFrame.
    This maintains the multi-indexed panel structure. 
    """
    def __init__(self, transformer, column_names=None):
        self.transformer = transformer
        self.column_names = column_names

    def transform(self, X):
        """
        Transform the input data 
        """
        transformation = super().transform(X)
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
