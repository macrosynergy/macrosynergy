import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression


class PLSTransformer(BaseEstimator, TransformerMixin):
    """
    Extract PLS components from scikit-learn's PLSRegression.

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS components to extract.
    """
    def __init__(self, n_components=2):
        if not isinstance(n_components, int):
            raise TypeError("n_components must be an integer.")
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        
        self.n_components = n_components
        self.model = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        """
        Fit the PLS model to the data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            The input feature matrix.
        y : pd.DataFrame, pd.Series or np.ndarray
            The target variable.

        Returns
        -------
        self
            The fitted model.
        """
        # Checks
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError(
                "X must be a pandas DataFrame, pandas Series or numpy array"
            )
        elif isinstance(X, np.ndarray) and ((X.ndim != 2)):
            raise ValueError(
                "When X is a numpy array, it must have exactly 2 dimensions."
            )
        
        # Fit the PLS model
        self.model.fit(X, y)

        return self

    def transform(self, X):
        """
        Transform the input data to the latent PLS space.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            The input feature matrix to be transformed.
        """
        # Checks
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError(
                "X must be a pandas DataFrame, pandas Series or numpy array"
            )
        elif isinstance(X, np.ndarray) and ((X.ndim != 2)):
            raise ValueError(
                "When X is a numpy array, it must have exactly 2 dimensions."
            )
        
        # Transform the data using the fitted PLS model
        return self.model.transform(X)
    
if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2012-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2012-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2012-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2012-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    Xy = msm.categories_df(
        df=dfd, xcats=xcats, cids=cids, freq="M", lag=1, xcat_aggs=["last", "sum"]
    ).dropna()
    X = Xy.iloc[:, :-1]
    y = Xy.iloc[:, -1]

    pls = PLSTransformer()
    pls.fit(X, y)
    print(pls.transform(X))
