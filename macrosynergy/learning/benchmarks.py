"""
Class to provide benchmark signals based on z-scores.

**NOTE: This module is under development, and is not yet ready for production use.**
"""

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline

class BenchmarkTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Transformer to create a benchmark signal based on z-scores.
        """
        self.current_mean = None
        self.current_sum_squares = None 
        self.current_n = 0


    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.current_mean is None:
            self.current_mean = np.zeros(X.shape[1])
            self.current_sum_squares = np.zeros(X.shape[1])

        n_update = len(X)
        new_n = self.current_n + n_update
        
        # determine updated mean
        new_mean = (self.current_n * self.current_mean + n_update*np.mean(X,axis=0))/(new_n)

        # determine updated standard deviation
        new_sum_squares = self.current_sum_squares + np.sum(np.square(X),axis=0)
        comp1 = (new_sum_squares)/(new_n - 1)
        comp2 = 2 * np.square(new_mean) * (new_n)/(new_n - 1)
        comp3 = (new_n)*np.square(new_mean)/(new_n - 1)
        new_std = np.sqrt(comp1 - comp2 + comp3)

        # update parameters
        self.current_mean = new_mean
        self.current_sum_squares = new_sum_squares
        self.current_n = new_n

        # get z-scores
        normalised_X = (X - self.current_mean)/new_std

        # return the benchmark signal as the sum of the z-scores
        return np.sum(normalised_X, axis = 1)
    
class BenchmarkEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        """
        Basic estimator to use the determined signal from BenchmarkTransformer as the predictions.
        """
        pass
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return X

if __name__ == "__main__":
    from macrosynergy.management.simulate_quantamental_data import make_qdf
    import macrosynergy.management as msm
    from macrosynergy.learning.panel_timeseries_split import PanelTimeSeriesSplit
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example 1: Unbalanced panel """

    df_cids2 = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids2.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats2 = pd.DataFrame(index=xcats, columns=cols)
    df_xcats2.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats2.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats2.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)

    dfd2 = dfd2.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X2 = dfd2.drop(columns=["XR"])
    y2 = dfd2["XR"]

    # Demonstration of BenchmarkTransformer
    pipe = Pipeline([("signal", BenchmarkTransformer()),("identity", BenchmarkEstimator())])
    splitter = PanelTimeSeriesSplit(n_splits=4, n_split_method="expanding")
    # Convert the regression problem into a classification problem
    acc_metric = make_scorer(lambda y_true, y_pred: np.mean(np.sign(y_true) == np.sign(y_pred)))
    print(cross_val_score(pipe, X2, y2, cv=splitter, scoring=acc_metric))
