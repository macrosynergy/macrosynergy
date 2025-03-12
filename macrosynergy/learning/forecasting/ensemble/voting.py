import numpy as np
import pandas as pd
import sklearn.ensemble as skl

class VotingRegressor(skl.VotingRegressor):
    """
    Regression model that averages the predictions of many regression models.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples that are used to fit the model.
    weights : array-like of shape (n_estimators,), default=None
        Sequence of weights to assign to models. If None, models are weighted
        equally.
    n_jobs : int, default=None
        The number of jobs to run in parallel for `fit`. `None` means 1 unless
        in a `joblib.parallel_backend` context. `-1` means using all
        processors.
    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as model
        trains.

    Notes
    -----
    This class calculates feature importances as the average of the feature
    importances of the base estimators.
    """
    def __init__(self, estimators, weights=None, n_jobs=None, verbose=False):
        super().__init__(estimators=estimators, weights=weights, n_jobs=n_jobs, verbose=verbose)
        self.feature_importances_ = None
        
    def fit(self, X, y, sample_weight=None, **fit_params):
        """
        Fit the estimators.

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

        super().fit(X, y, sample_weight, **fit_params)
        
        # Calculate feature importances
        importances = []
        for estimator in self.estimators_:
            if hasattr(estimator, "coef_") or hasattr(estimator, "feature_importances_"):
                # Normalize feature importances to sum to 1
                imp = (
                    np.abs(estimator.coef_) / np.sum(np.abs(estimator.coef_))
                    if hasattr(estimator, "coef_")
                    else estimator.feature_importances_
                    / np.sum(estimator.feature_importances_)
                )
                importances.append(imp)
        if len(importances) > 0:
            self.feature_importances_ = np.mean(importances, axis=0)
        
        # Renormalize feature importances to sum to 1
        if self.feature_importances_ is not None:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        return self
    
    def _check_fit_params(self, X, y):
        """
        Checks for fit method parameters
        """
        # X
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for the voting regressor must be either a pandas "
                "dataframe or numpy array."
            )
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    "When the input feature matrix for the voting regressor is a "
                    "numpy array, it must have two dimensions."
                )
        # y
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Target vector for the voting regressor must be either a pandas series, "
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
                    "When the target vector for the voting regressor is a numpy "
                    "array, it must have one dimension."
                )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of samples in the input feature matrix must match the number "
                "of samples in the target vector."
            )
    
class VotingClassifier(skl.VotingClassifier):
    """
    Classification model that votes on the predictions of many classifiers.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples that are used to fit the model.
    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        If 'soft', predicts the class label based on the argmax of the sums of
        the predicted probabilities, which is recommended for an ensemble of
        well-calibrated classifiers.
    weights : array-like of shape (n_estimators,), default=None
        Sequence of weights to assign to models. If None, models are weighted
        equally.
    n_jobs : int, default=None
        The number of jobs to run in parallel for `fit`. `None` means 1 unless
        in a `joblib.parallel_backend` context. `-1` means using all
        processors.
    flatten_transform : bool, default=True
        Affects shape of transform output only when voting='soft'. If True,
        the transform method returns a matrix with shape (n_samples, n_classes*n_classifiers).
        If False, the shape is (n_classifiers, n_samples, n_classes).
    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as model
        trains.

    Notes
    -----
    This class calculates feature importances as the average of the feature
    importances of the base estimators.
    """
    def __init__(self, estimators, voting="hard", weights=None, n_jobs=None, flatten_transform=True, verbose=False):
        super().__init__(estimators=estimators, voting=voting, weights=weights, n_jobs=n_jobs, flatten_transform=flatten_transform, verbose=verbose)
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None, **fit_params):
        """
        Fit the estimators.

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

        # Fit classifiers
        super().fit(X, y, sample_weight, **fit_params)

        # Calculate feature importances
        importances = []
        for estimator in self.estimators_:
            if hasattr(estimator, "coef_") or hasattr(
                estimator, "feature_importances_"
            ):
                # Normalize feature importances to sum to 1
                imp = (
                    np.squeeze(np.abs(estimator.coef_) / np.sum(np.abs(estimator.coef_)))
                    if hasattr(estimator, "coef_")
                    else estimator.feature_importances_
                    / np.sum(estimator.feature_importances_)
                )
                importances.append(imp)

        if len(importances) > 0:
            self.feature_importances_ = np.mean(importances, axis=0)
        
        # Renormalize feature importances to sum to 1
        if self.feature_importances_ is not None:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        return self
    
    def _check_fit_params(self, X, y):
        """
        Checks for fit method parameters
        """
        # X
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for the voting regressor must be either a pandas "
                "dataframe or numpy array."
            )
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    "When the input feature matrix for the voting regressor is a "
                    "numpy array, it must have two dimensions."
                )
        # y
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Target vector for the voting regressor must be either a pandas series, "
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
                    "When the target vector for the voting regressor is a numpy "
                    "array, it must have one dimension."
                )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of samples in the input feature matrix must match the number "
                "of samples in the target vector."
            )


if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf
    import pandas as pd

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

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
    y_train = train["XR"]

    vr = VotingRegressor(
        estimators = [
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor())
        ]
    ).fit(X_train, y_train)

    print(f"Voting regressor feature importances: {vr.feature_importances_}")

    # Classifier
    X_train = train.drop(columns=["XR"])
    y_train = np.sign(train["XR"])

    vr = VotingClassifier(
        estimators = [
            ("lr", LogisticRegression()),
            ("rf", RandomForestClassifier())
        ]
    ).fit(X_train, y_train)

    print(f"Voting classifier feature importances: {vr.feature_importances_}")