import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import check_classification_targets
class KNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors="sqrt", weights="uniform"):
        """
        Nearest neighbors classifier.

        Parameters
        ----------
        n_neighbors : int, float, or str
            Number of neighbors to use. If int, the number of neighbors to use.
            If float, the fraction of the number of samples to use. If "sqrt",
            the square root of the number of samples is used.
        weights : str
            Weight function used to aggregate neighbors. Possible values are "uniform"
            and "distance".

        Notes
        -----
        The class is a wrapper around the KNeighborsClassifier from scikit-learn. It has
        been implemented to allow for the use of a fraction of the number of samples as
        the number of neighbors to use. In addition, the square root of the number of
        samples is a common rule of thumb for the number of neighbors to use - we wanted
        to allow for this option in cross-validation.
        """
        self._check_init_params(n_neighbors, weights)

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn_ = None
        self.classes_ = [-1, 1]

    def fit(self, X, y):
        """
        Fit method.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input feature matrix.
        y : pd.Series or np.ndarray
            The target variable.

        Returns
        -------
        self
            The fitted model.
        """
        self._check_fit_params(X, y)

        if self.n_neighbors == "sqrt":
            n = int(np.sqrt(len(X)))
        elif isinstance(self.n_neighbors, float):
            n = int(self.n_neighbors * len(X))
        else:
            n = self.n_neighbors
        self.knn_ = KNeighborsClassifier(n_neighbors=n, weights=self.weights).fit(X, y)

        return self

    def predict(self, X):
        """
        Predict method.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input feature matrix.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        self._check_predict_params(X)

        return self.knn_.predict(X)

    def predict_proba(self, X):
        """
        Predict probability method.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input feature matrix.

        Returns
        -------
        np.ndarray
            The predicted probabilities.
        """
        self._check_predict_params(X)

        return self.knn_.predict_proba(X)
    
    def __getattr__(self, attr):
        """
        Return the class attributes.

        Parameters
        ----------
        attr : str
            The attribute to return.

        Returns
        -------
        Any
            The attribute.
        """
        try:
            return getattr(self.knn_, attr)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )
    
    def _check_init_params(self, n_neighbors, weights):
        """
        Check the parameters passed to the __init__ method.
        """
        # n_neighbors
        if not isinstance(n_neighbors, (int, float, str)):
            raise TypeError("n_neighbors must be an int, float, or str")
        if isinstance(n_neighbors, str) and n_neighbors != "sqrt":
            raise ValueError('n_neighbors must be "sqrt" if it is a str')
        if isinstance(n_neighbors, float) and ((n_neighbors <= 0) or (n_neighbors >= 1)):
            raise ValueError("n_neighbors must be between 0 and 1 if it is a float")
        if isinstance(n_neighbors, int) and n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than 0 if it is an int")
        if not isinstance(weights, str):
            raise TypeError("weights must be a str")
        if weights not in ["uniform", "distance"]:
            raise ValueError('weights must be "uniform" or "distance')
        
    def _check_fit_params(self, X, y):
        """
        Check the parameters passed to the fit method.
        """
        # Type checks
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pd.DataFrame or np.ndarray")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pd.Series or np.ndarray")
        # Value checks 
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if isinstance(X, (pd.DataFrame, pd.Series)) and isinstance(y, pd.Series):
            if not X.index.equals(y.index):
                raise ValueError("X and y must have the same index")
        # Value checks for X
        if isinstance(X, pd.DataFrame):
            if not isinstance(X.index, pd.MultiIndex):
                raise ValueError("X must have a multi-index")
            if len(X.index.levels) != 2:
                raise ValueError("X must have a multi-index with two levels")
            if not X.index.get_level_values(0).dtype == "object":
                raise TypeError("The outer index of X must be strings.")
            if not X.index.get_level_values(1).dtype == "datetime64[ns]":
                raise TypeError("The inner index of X must be datetime.date.")
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    "When the input feature matrix for nearest neighbor forecasts is a "
                    "numpy array, it must have two dimensions."
                )
        # Value checks for y
        if isinstance(y, pd.Series):
            if not isinstance(y.index, pd.MultiIndex):
                raise ValueError("y must have a multi-index")
            if len(y.index.levels) != 2:
                raise ValueError("y must have a multi-index with two levels")
            if not y.index.get_level_values(0).dtype == "object":
                raise TypeError("The outer index of y must be strings.")
            if not y.index.get_level_values(1).dtype == "datetime64[ns]":
                raise TypeError("The inner index of y must be datetime.date.")
        if isinstance(y, np.ndarray):
            if y.ndim != 1:
                raise ValueError(
                    "When the target variable for nearest neighbor forecasts is a numpy "
                    "array, it must have one dimension."
                )
        check_classification_targets(y)

    def _check_predict_params(self, X):
        """
        Check the parameters passed to the predict method.
        """
        # Type checks
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pd.DataFrame or np.ndarray")
        # Value checks for X
        if isinstance(X, pd.DataFrame):
            if not isinstance(X.index, pd.MultiIndex):
                raise ValueError("X must have a multi-index")
            if len(X.index.levels) != 2:
                raise ValueError("X must have a multi-index with two levels")
            if not X.index.get_level_values(0).dtype == "object":
                raise TypeError("The outer index of X must be strings.")
            if not X.index.get_level_values(1).dtype == "datetime64[ns]":
                raise TypeError("The inner index of X must be datetime.date.")
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    "When the input feature matrix for nearest neighbor forecasts is a "
                    "numpy array, it must have two dimensions."
                )
        
    
if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm
    from macrosynergy.learning import SignalOptimizer, ExpandingKFoldPanelSplit

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, r2_score

    # Randomly generate an unbalanced panel dataset, multi-indexed by cross-section and
    # real_date

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0, 1, 0, 3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 0, 1, 0, 0]
    df_xcats.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 0, 1, -0.9, 0]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", 0, 1, 0.8, 0]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd = msm.reduce_df(df=dfd, cids=cids, xcats=xcats, blacklist=black)

    dfd = dfd.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd.drop(columns=["XR"])
    y = np.sign(dfd["XR"])

    # Fit nearest neighbors classifier
    knn = KNNClassifier(n_neighbors=0.1)
    knn.fit(X, y)
    print(knn.predict(X))
