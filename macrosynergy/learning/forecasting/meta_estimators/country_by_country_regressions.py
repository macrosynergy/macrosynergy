import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, clone

class CountryByCountryRegression(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    """
    MetaEstimator to fit a `scikit-learn`-compatible regressor on each country's data slice
    in a panel. If a country has fewer samples than `min_xs_samples`, a global model is
    used for the sake of prediction.

    Parameters
    ----------
    estimator : object
        A scikit-learn compatible regressor that will be cloned for each country.
    min_xs_samples : int, default=32
        Minimum number of samples required for fitting a country-specific model. If a country
        has fewer samples, the global model will be used for predictions.

    Notes
    -----
    Country by country regressions model a panel through a "bottoms-up" approach, treating
    each country as a separate regression problem. This is useful when a panel is
    particularly heterogeneous or each time series in the panel is long. Short time series
    results in a low-bias, high-variance model that tends to underperform a global
    forecasting model. Regularization on each country-specific model can help improve
    performance.
    """
    def __init__(self, estimator, min_xs_samples = 32):
        self.estimator = estimator
        self.min_xs_samples = min_xs_samples
        
    def fit(self, X, y):
        # First fit a global model to handle bad country-specific data availability
        self.global_model_ = self.estimator.fit(X, y)
        self.models_ = {}

        # Loop through each country, fit and store a model
        coefs = []
        for xs in X.index.get_level_values(0).unique():
            x_slice = X.loc[xs]
            y_slice = y.loc[xs]

            if len(x_slice) >= self.min_xs_samples:
                model = clone(self.estimator)
                model.fit(x_slice, y_slice)
                self.models_[xs] = model
            else:
                self.models_[xs] = None
                
        # Use the average importance across countries as a measure of global importance.
        if hasattr(self.estimator, "feature_importances_"):
            self.feature_importances_ = np.mean(
                [estimator.feature_importances_ for country, estimator in self.models_.items() if estimator is not None],
                axis=0
            )
        elif hasattr(self.estimator, "coef_"):
            self.feature_importances_ = np.mean(
                [estimator.coef_ for country, estimator in self.models_.items() if estimator is not None],
                axis=0
            )
            self.intercept_ = np.mean(
                [estimator.intercept_ for country, estimator in self.models_.items() if estimator is not None],
                axis=0
            )

        return self
    
    def predict(self, X):
        """
        Predict the target values for the given input data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        preds = []
        for xs in X.index.get_level_values(0).unique():
            x_slice = X.loc[xs]
            model = self.models_.get(xs)
            if model is not None:
                p = pd.Series(model.predict(x_slice), index=x_slice.index)
                preds.append(p)
            else:
                p = pd.Series(self.global_model_.predict(x_slice), index=x_slice.index)
                #p = pd.Series(np.nan, index=x_slice.index)
                preds.append(p)
        return np.array(pd.concat(preds).sort_index())
    
if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf

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

    # Regressor
    X_train = train.drop(columns=["XR"])
    y_train = np.sign(train["XR"])

    cr = CountryByCountryRegression(estimator=LinearRegression()).fit(X_train, y_train)
    print(cr.predict(X_train))