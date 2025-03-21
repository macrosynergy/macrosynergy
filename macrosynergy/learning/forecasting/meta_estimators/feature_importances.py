import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, ClassifierMixin

class FIExtractor(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    def __init__(self, estimator):
        """
        Meta estimator to store feature importances of a scikit-learn regressor or
        classifier. Note that this class does not compute permutation importances, but
        rather extracts in-built feature importances and normalizes these to sum to 1.

        Parameters
        ----------
        estimator : RegressorMixin or ClassifierMixin
            A scikit-learn regressor or classifier comprising either a coef_ or
            feature_importances_ attribute.

        Notes
        -----
        It is generally recommended to calculate permutation importances for a machine
        learning pipeline to gauge the importance of each feature. This, however, takes
        time. This class is useful for a quick indication of feature importance.
        """
        if not isinstance(estimator, (ClassifierMixin, RegressorMixin)):
            raise TypeError("estimator must be a scikit-learn predictor.")
        self.estimator = estimator

    def fit(self, X, y):
        """
        Fit the underlying estimator and store normalized feature importances.

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

        # Fit estimator
        self.estimator.fit(X, y)

        if hasattr(self.estimator, "coef_"):
            self.feature_importances_ = np.abs(self.estimator.coef_.flatten()) / np.sum(
                np.abs(self.estimator.coef_)
            )
        elif hasattr(self.estimator, "feature_importances_"):
            self.feature_importances_ = self.estimator.feature_importances_.flatten() / np.sum(
                self.estimator.feature_importances_
            )

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
        return self.estimator.predict(X)

    def __getattr__(self, attr):
        """
        Get attributes from the underlying model.
        """
        if hasattr(self.estimator, attr):
            return getattr(self.estimator, attr)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

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
        
if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf
    import pandas as pd
    import numpy as np

    from sklearn.linear_model import LogisticRegression, LinearRegression

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

    fi = FIExtractor(LinearRegression()).fit(X_train, y_train)
    print(fi.feature_importances_)
