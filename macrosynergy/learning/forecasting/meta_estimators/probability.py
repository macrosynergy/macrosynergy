from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin

class ProbabilityEstimator(BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    """
    Meta-estimator to return binary classifier probabilities as signals to be processed by 
    the `SignalOptimizer` class in the `macrosynergy.learning subpackage`.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        self.classes_ = [-1,1]
        self.feature_importances_ = None

    def fit(self, X, y):
        self.classifier.fit(X, y)
        if hasattr(self.classifier, "feature_importances_"):
            self.feature_importances_ = self.classifier.feature_importances_ 

        return self
    
    def predict(self, X):
        return self.classifier.predict(X)

    def create_signal(self, X):
        return self.classifier.predict_proba(X)[:,1] - 0.5
    
if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf
    import pandas as pd
    import numpy as np

    from sklearn.linear_model import LogisticRegression

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

    pe = ProbabilityEstimator(LogisticRegression()).fit(X_train, y_train)
    print(pe.create_signal(X_train))