import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class NaiveRegressor(BaseEstimator, RegressorMixin):
    """
    Equally weighted unbiased factor model.

    Notes
    -----
    Given a collection of factors that are theoretically positively correlated with a
    dependent variable, a plausible signal is a simple average of those factors. This is
    effectively a linear regression model with zero intercept and equal weights for all
    factors.

    This is a useful benchmark model which works well when the factors are as
    uncorrelated as possible with one another, because it offers a layer of
    diversification on the underlying return drivers. When the user has strong priors,
    this is often a competitive model that is difficult to beat.

    However, it is vital for the features to have been preprocessed to have a positive
    theoretical correlation with the target variable.
    """

    def fit(self, X, y=None):
        """
        Fit method.

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
            
        Notes
        -----
        This method involves fully trusting one's priors and thus requires no learning
        element. As a consequence, no training set information is needed.
        """
        # Checks
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError(
                "X must be a pandas DataFrame, pandas Series or numpy array"
            )
        elif isinstance(X, np.ndarray) and ((X.ndim > 2) or (X.ndim < 1)):
            raise ValueError(
                "When X is a numpy array, it must have either 1 or 2 dimensions."
            )

        # No learning needed since priors are fully trusted
        self.n = len(X)
        if isinstance(X, pd.Series):
            self.p = 1
        elif isinstance(X, np.ndarray):
            self.p = X.shape[1] if X.ndim == 2 else 1
        else:
            self.p = X.shape[1]

        self.X_type = type(X)

        return self

    def predict(self, X):
        """
        Predict method.

        Notes
        -----
        The predictions are simply the average of the features across columns of the input
        feature matrix.
        """
        # Checks
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError(
                "X must be a pandas DataFrame, pandas Series or numpy array"
            )
        elif isinstance(X, np.ndarray) and ((X.ndim > 2) or (X.ndim < 1)):
            raise ValueError(
                "When X is a numpy array, it must have either 1 or 2 dimensions."
            )
        if not isinstance(X, self.X_type):
            raise ValueError(
                "X must be of the same type as the input matrix to the fit() method."
            )

        if isinstance(X, np.ndarray) and X.ndim == 1:
            p = 1
        elif isinstance(X, pd.Series):
            p = 1
        else:
            p = X.shape[1]
        if p != self.p:
            raise ValueError(
                "X must have the same number of columns as the input matrix to the "
                "fit() method."
            )

        # Return the naive signal
        if isinstance(X, pd.DataFrame):
            return np.mean(X.values, axis=1)
        elif isinstance(X, pd.Series):
            return X.values
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            return X
        else:
            return np.mean(X, axis=1)


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

    naive = NaiveRegressor()
    naive.fit(X, y)
    print(naive.predict(X))
