import numpy as np
import pandas as pd 

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.pipeline import Pipeline

class TimeWeightedWrapper(BaseEstimator, RegressorMixin):
    """
    Meta-estimator that applies time-based weighting to samples during model fitting.

    Parameters
    ----------
    model : BaseEstimator
        An instance of a scikit-learn compatible regression model.
    half_life : float
        The half-life parameter for the exponential decay weighting.
    """
    def __init__(self, model, half_life):
        # Checks
        self._check_init_params(model, half_life)
        
        # Attributes
        self.model = model
        self.half_life = half_life

    def fit(self, X, y):
        """
        Fit the underlying model with time weights applied.

        Parameters
        ----------
        X : pandas.DataFrame or np.ndarray
            The feature matrix.
        y : pandas.Series or np.ndarray
            The target vector.
        """
        # Checks
        self._check_fit_params(X, y)

        # Derive time weights
        dates = sorted(y.index.get_level_values(1).unique(), reverse=True)
        num_dates = len(dates)
        weights = np.power(2, -np.arange(num_dates) / self.half_life)

        weight_map = dict(zip(dates, weights))
        sample_weights = y.index.get_level_values(1).map(weight_map).to_numpy()

        # Fit model with these sample weights
        self.model.fit(X, y, sample_weight=sample_weights)

        return self

    def predict(self, X):
        """
        Predict using the underlying model.

        Parameters
        ----------
        X : pandas.DataFrame or np.ndarray
            The feature matrix.

        Returns
        -------
        predictions : np.ndarray
            The predicted values.
        """
        # Checks
        self._check_predict_params(X)

        # Predict
        return self.model.predict(X)
    
    def _check_init_params(self, model, half_life):
        if not isinstance(model, BaseEstimator):
            raise TypeError("The 'model' parameter must be a scikit-learn compatible estimator.")
        if not isinstance(model, (RegressorMixin, ClassifierMixin, Pipeline)):
            raise TypeError("The 'model' parameter must be a regressor, classifier or sklearn pipeline.")
        if half_life <= 0:
            raise ValueError("The 'half_life' parameter must be a positive number.")
        
    def _check_fit_params(self, X, y):
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be a numpy array or pandas DataFrame.")
        if not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError("y must be a numpy array, pandas Series, or pandas DataFrame.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be the same.")
        
    def _check_predict_params(self, X):
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be a numpy array or pandas DataFrame.")
        
if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf
    import pandas as pd
    import numpy as np

    from sklearn.linear_model import LinearRegression


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

    # Define and fit the time-weighted model
    model = TimeWeightedWrapper(
        model = LinearRegression(),
        half_life = 12
    ).fit(X_train, y_train)
    print(model.predict(X_train.head()))