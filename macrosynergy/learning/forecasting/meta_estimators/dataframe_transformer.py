import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin

class DataFrameTransformer(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    """
    Meta estimator to reconvert a transformed numpy array back to a multiindexed 
    pandas DataFrame. This maintains the multi-indexed panel structure.

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
        # Checks
        if not isinstance(transformer, TransformerMixin):
            raise TypeError("transformer must be a scikit-learn transformer.")
        if column_names is not None and not isinstance(column_names, list):
            raise TypeError("column_names must be a list of strings or None.")
        if column_names is not None and len(column_names) == 0:
            raise ValueError("column_names cannot be an empty list.")
        if column_names is not None and not all(isinstance(name, str) for name in column_names):
            raise ValueError("All column names must be strings.")
        if column_names is not None and len(set(column_names)) != len(column_names):
            raise ValueError("All column names must be unique.")
        
        # Attributes
        self.transformer = transformer
        self.column_names = column_names

    def fit(self, X, y=None):
        """
        Fit the underlying transformer.

        Parameters
        ----------
        X : pd.DataFrame
            Pandas dataframe of input features.
        y : pd.Series or pd.DataFrame or np.ndarray
            Pandas series, dataframe or numpy array of targets associated with each sample
            in X.
        """
        # Checks
        self._check_fit_params(X, y)

        # Fit estimator
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
        # Checks
        self._check_predict_params(X)

        # Transform the data
        transformation = self.transformer.transform(X)

        # Check the number of column names provided is more than or equal to the number of features
        if self.column_names is not None and len(self.column_names) < transformation.shape[1]:
            raise ValueError(
                "The number of column names provided must be greater than or equal to the "
                "number of features in the transformed data."
            )

        # Return the data in the correct format
        if isinstance(transformation, pd.DataFrame):
            return transformation
        else:
            # scikit-learn returns a numpy array, convert it back to DataFrame
            if self.column_names is None:
                columns = [f"Factor_{i}" for i in range(transformation.shape[1])]
            else:
                columns = self.column_names[:transformation.shape[1]]

            return pd.DataFrame(
                data = transformation,
                columns = columns,
                index = X.index,
            )
        
    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying transformer.

        Parameters
        ----------
        name : str
            The name of the attribute to access.
        """
        # Precent infinite recursion
        if name == "transformer":
            raise AttributeError()
        return getattr(self.transformer, name)
    
    def _check_fit_params(self, X, y):
        """
        Checks for fit method parameters.
        """
        # X
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the dataframe transformer must be a pandas "
                "dataframe."
            )
        # y
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Target vector for the dataframe transformer must be either a pandas series, "
                "dataframe or numpy array."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "The dependent variable dataframe must have only one column. If used "
                    "as part of an sklearn pipeline, ensure that previous steps return "
                    "a pandas series or dataframe."
                )
        if isinstance(y, np.ndarray):
            if y.ndim != 1:
                raise ValueError(
                    "When the target vector for the probability estimator is a numpy "
                    "array, it must have one dimension."
                )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of samples in the input feature matrix must match the number "
                "of samples in the target vector."
            )
        # Check NaN values
        if X.isnull().values.any():
            raise ValueError(
                "The input feature matrix contains NaN values. Please handle missing "
                "values before fitting the transformer."
            )
        if isinstance(y, (pd.DataFrame, pd.Series)) and y.isnull().values.any():
            raise ValueError(
                "The target vector contains NaN values. Please handle missing values before "
                "fitting the transformer."
            )
        elif isinstance(y, np.ndarray) and np.isnan(y).any():
            raise ValueError(
                "The target vector contains NaN values. Please handle missing values before "
                "fitting the transformer."
            )
        
        # Check that the number of column names provided is equal to the number of features
        if self.column_names is not None and len(self.column_names) != X.shape[1]:
            raise ValueError(
                "The number of column names provided must match the number of features."
            )

    def _check_predict_params(self, X):
        """
        Checks for predict method parameters.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the probability estimator must be a pandas "
                "dataframe."
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in the input feature matrix must match the number "
                "seen in training."
            )
        
        # Check that the number of column names provided equal to the number of features
        if self.column_names is not None and len(self.column_names) != X.shape[1]:
            raise ValueError(
                "The number of column names provided must be equal to the number of features."
            )
        
if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf
    import pandas as pd
    import numpy as np

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline


    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {
        "GBP": (
            pd.Timestamp(year=2009, month=1, day=1),
            pd.Timestamp(year=2012, month=6, day=30),
        ),
        "CAD": (
            pd.Timestamp(year=2015, month=1, day=1),
            pd.Timestamp(year=2100, month=1, day=1),
        ),
    }

    train = msm.categories_df(
        df=dfd, xcats=xcats, cids=cids, val="value", blacklist=black, freq="M", lag=1
    ).dropna()

    # Training set
    X_train = train.drop(columns=["XR"])
    y_train = np.sign(train["XR"])

    # Model fit
    print(DataFrameTransformer(PCA(n_components = 2)).fit_transform(X_train, y_train))
    print(DataFrameTransformer(PCA(n_components = 2), column_names = [f"PCA_{i}" for i in range(1, 4)]).fit_transform(X_train, y_train))
