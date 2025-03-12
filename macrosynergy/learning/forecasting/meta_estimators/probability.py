import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin

class ProbabilityEstimator(BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    """
    Meta estimator to create trading signals based on the probability of going long.

    Parameters
    ----------
    classifier : ClassifierMixin
        A scikit-learn classifier.

    Notes
    -----
    This class stores feature importances as the feature importances of the base estimator
    as well as defining a create_signal method that returns the probability of going long
    in excess of 0.5. This is taken into account when used in the SignalOptimizer class
    in this package.
    """
    def __init__(self, classifier):
        if not isinstance(classifier, ClassifierMixin):
            raise TypeError("classifier must be a scikit-learn classifier.")
        
        self.classifier = classifier
        self.classes_ = [-1,1]
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Fit the underlying classifier.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Pandas dataframe or numpy array of input features.
        y : pd.Series or pd.DataFrame or np.ndarray
            Pandas series, dataframe or numpy array of targets associated with each sample
            in X.
        """
        # Checks
        self._check_fit_params(X, y)

        # Model fitting
        self.classifier.fit(X, y)

        # Store feature importances
        if hasattr(self.classifier, "feature_importances_"):
            self.feature_importances_ = self.classifier.feature_importances_
        elif hasattr(self.classifier, "coef_"):
            self.feature_importances_ = np.abs(self.classifier.coef_) / np.sum(np.abs(self.classifier.coef_))

        return self
    
    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : pd.DataFrame or numpy array
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray
            Numpy array of predictions.
        """
        # Checks
        self._check_predict_params(X)

        # Predict
        return self.classifier.predict(X)

    def create_signal(self, X):
        """
        Create a trading signal based on the probability of going long.

        Parameters
        ----------
        X : pd.DataFrame or numpy array
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray
            Numpy array of signals.
        """
        # Checks
        self._check_predict_params(X)

        # Create signal
        return self.classifier.predict_proba(X)[:,1] - 0.5
    
    def _check_fit_params(
        self,
        X,
        y,
    ):
        """
        Checks for fit method parameters.
        """
        # X
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for the probability estimator must be either a pandas "
                "dataframe or numpy array."
            )
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    "When the input feature matrix for the probability estimator is a "
                    "numpy array, it must have two dimensions."
                )
        # y
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Target vector for the probability estimator must be either a pandas series, "
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
        
    def _check_predict_params(self, X):
        """
        Checks for predict method parameters.
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for the probability estimator must be either a pandas "
                "dataframe or numpy array. If used as part of an sklearn pipeline, ensure "
                "that previous steps return a pandas dataframe or numpy array."
            )
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    "When the input feature matrix for the probability estimator is a "
                    "numpy array, it must have two dimensions. If used as part of an "
                    "sklearn pipeline, ensure that previous steps return a two-dimensional "
                    "data structure."
                )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features in the input feature matrix must match the number "
                "seen in training."
            )
        
    def __getattr__(self, attr):
        """
        Get attributes from the underlying classifier.
        """
        if hasattr(self.classifier, attr):
            return getattr(self.classifier, attr)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    
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