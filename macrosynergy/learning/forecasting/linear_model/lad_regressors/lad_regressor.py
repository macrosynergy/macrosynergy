import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize

import numbers
import warnings
from functools import partial
from sklearn.exceptions import ConvergenceWarning


class LADRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        fit_intercept=True,
        positive=False,
        alpha=0,
        shrinkage_type="l1",
        tol=None,
        maxiter=None,
    ):
        r"""
        Linear regression with L1 loss.

        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether or not to add an intercept to the model.
        positive: bool, default=False
            Whether or not to enforce non-negativity of model weights,
            with exception to a model intercept.
        alpha: float, default=0
            Shrinkage hyperparameter.
        shrinkage_type: str, default="l1"
            Type of shrinkage regularization to perform.
        tol: float, default=None
            Tolerance for convergence of the learning algorithm (SLSQP).
            This is passed into the 'tol' parameter of the scipy.optimize.minimize
            function.
        maxiter: int, default=None
            Maximum number of iterations for the learning algorithm (SLSQP).
            This is passed into the 'maxiter' key of the options dictionary in the
            scipy.optimize.minimize function.

        Notes
        -----
        A dependent variable is modelled as a linear combination of the input features.
        The weights associated with each feature (and the intercept) are determined by
        finding the weights that minimise the average absolute model residuals.

        If `alpha` is positive, then shrinkage-based regularization is applied to the
        non-intercept model coefficients. The type of shrinkage is determined by the
        `shrinkage_type` parameter. If `shrinkage_type` is "l1", then L1 regularization
        is applied. If `shrinkage_type` is "l2", then L2 regularization is applied.

        Mathematically, the following optimization problem is solved:

        .. math::

        \underset{w, b}{\text{argmin}} \frac{1}{N} \sum_{i=1}^{N} |y_{i} - x_{i}^{T}w - b|
        + 2N \times \alpha \times L_{p}(w)^{p}

        where:
            - N is the number of samples in the training data.
            - :math:`y_{i}` is the response for the i-th sample.
            - :math:`x_{i}` is the feature vector for the i-th sample.
            - :math:`w` is the vector of model coefficients.
            - :math:`b` is the model intercept.
            - :math:`L_{p}(w)` is the :math:`L_{p}` norm of the model coefficients, where
                :math:`p` is 1 or 2.
            - :math:`alpha` is the regularization hyperparameter.
        """
        # Checks
        self._check_init_params(
            fit_intercept, positive, alpha, shrinkage_type, tol, maxiter
        )

        # Initialise
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.alpha = alpha
        self.shrinkage_type = shrinkage_type
        self.tol = tol
        self.maxiter = maxiter

        # Set values of quantities to learn to None
        self.coef_ = None
        self.intercept_ = None

    def fit(
        self,
        X,
        y,
        sample_weight: np.ndarray = None,
    ):
        """
        Learn LAD regression model parameters.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input feature matrix.
        y : pd.Series or pd.DataFrame or np.ndarray
            Target vector associated with each sample in X.
        sample_weight : np.ndarray, default=None
            Numpy array of sample weights to create a weighted LAD regression model.
        """
        # Checks
        self._check_fit_params(X, y, sample_weight)

        # Fit
        X = X.copy()
        y = y.copy()

        if self.fit_intercept:
            if isinstance(X, pd.DataFrame):
                X.insert(0, "intercept", 1)
            else:
                X = np.insert(X, 0, 1, axis=1)

        n_cols = X.shape[1]

        # Optimization bounds
        if self.positive:
            if self.fit_intercept:
                bounds = [(None, None)] + [(0, None)] * (n_cols - 1)
            else:
                bounds = [(0, None)] * n_cols
        else:
            bounds = [(None, None)] * n_cols

        # Set initial weights
        init_weights = (
            np.zeros(n_cols) if not self.fit_intercept else np.zeros(n_cols - 1)
        )
        if self.fit_intercept:
            init_intercept = np.mean(y)
            init_weights = np.concatenate(([init_intercept], init_weights))

        X = X if isinstance(X, np.ndarray) else X.values
        y = y.squeeze() if isinstance(y, np.ndarray) else y.values.squeeze()

        optim_results = minimize(
            fun=partial(
                self._l1_loss,
                X=X,
                y=y,
                sample_weight=sample_weight,
                alpha=self.alpha,
                shrinkage_type=self.shrinkage_type,
            ),
            x0=init_weights,
            method="SLSQP",
            bounds=bounds,
            tol=self.tol,
            options={"maxiter": self.maxiter} if self.maxiter else None,
        )

        # Handle optimization results
        if not optim_results.success:
            warnings.warn(
                "LAD regression failed to converge. Try increasing the number of "
                "iterations or decreasing the tolerance level. The "
                "scipy.optimize.minimize message is: {}".format(optim_results.message),
                ConvergenceWarning,
            )

            return self

        if self.fit_intercept:
            self.intercept_ = optim_results.x[0]
            self.coef_ = optim_results.x[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = optim_results.x

        return self

    def predict(self, X):
        """
        Predict dependent variable using the fitted LAD regression model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray
            Numpy array of predictions.

        Notes
        -----
        If the model learning algorithm failed to converge, the predict method will return
        an array of zeros. This has the interpretation of no buy/sell signal being
        triggered based on this model.
        """
        # Checks
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for the LADRegressor must be either a pandas "
                "dataframe or numpy array. If used as part of an sklearn pipeline, ensure "
                "that previous steps return a pandas dataframe or numpy array."
            )
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    "When the input feature matrix for LADRegressor forecasts is a numpy "
                    "array, it must have two dimensions. If used as part of an sklearn "
                    "pipeline, ensure that previous steps return a two-dimensional data "
                    "structure."
                )

        if X.shape[1] != len(self.coef_):
            raise ValueError(
                "The number of features in the input feature matrix must match the number "
                "of model coefficients."
            )

        if isinstance(X, pd.DataFrame):
            if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
                raise ValueError(
                    "All columns in the input feature matrix must be numeric."
                )
            if X.isnull().values.any():
                raise ValueError(
                    "The input feature matrix for the LADRegressor must not contain any "
                    "missing values."
                )
        else:
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError(
                    "All elements in the input feature matrix must be numeric."
                )
            if np.isnan(X).any():
                raise ValueError(
                    "The input feature matrix for the LADRegressor must not contain any "
                    "missing values."
                )

        # Predict
        if self.coef_ is None:
            return np.zeros(X.shape[0])

        if self.fit_intercept:
            return (X.dot(self.coef_) + self.intercept_).values
        else:
            return X.dot(self.coef_).values

    def _l1_loss(
        self,
        weights,
        X,
        y,
        sample_weight=None,
        alpha: float = 0,
        shrinkage_type: str = "l1",
    ):
        """
        Determine L1 loss induced by 'weights'.

        Parameters
        ----------
        weights : np.ndarray
            LADRegressor model coefficients to be optimised.
        X : np.ndarray
            Input features.
        y : np.ndarray
            Targets associated with each sample in X.
        sample_weight : np.ndarray, default=None
            Sample weights to create a weighted LAD regression model.
        alpha : float, default=0
            Shrinkage hyperparameter.
        shrinkage_type : str, default="l1"
            Type of shrinkage regularization to perform.

        Returns
        -------
        loss : float
            L1 loss induced by 'weights'.
        """
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        raw_residuals = y - X @ weights
        abs_residuals = np.abs(raw_residuals)
        weighted_abs_residuals = abs_residuals * sample_weight

        if alpha > 0:
            if shrinkage_type == "l1":
                if self.fit_intercept:
                    l1_norm = np.sum(np.abs(weights[1:]))
                else:
                    l1_norm = np.sum(np.abs(weights))
                return (
                    np.mean(weighted_abs_residuals) + 2 * X.shape[0] * alpha * l1_norm
                )
            elif shrinkage_type == "l2":
                if self.fit_intercept:
                    l2_norm = np.sum(weights[1:] ** 2)
                else:
                    l2_norm = np.sum(weights**2)
                return (
                    np.mean(weighted_abs_residuals) + 2 * X.shape[0] * alpha * l2_norm
                )

        return np.mean(weighted_abs_residuals)

    def _check_init_params(
        self, fit_intercept, positive, alpha, shrinkage_type, tol, maxiter
    ):
        """
        Checks for constructor parameters.
        """
        # fit_intercept
        if not isinstance(fit_intercept, bool):
            raise TypeError("The fit_intercept parameter must be a boolean.")
        # positive
        if not isinstance(positive, bool):
            raise TypeError("The positive parameter must be a boolean.")
        # alpha
        if (not isinstance(alpha, numbers.Real)) or isinstance(alpha, bool):
            raise TypeError("The alpha parameter must be numeric.")
        if alpha < 0:
            raise ValueError("The alpha parameter must be non-negative.")
        # shrinkage_type
        if not isinstance(shrinkage_type, str):
            raise TypeError("The shrinkage_type parameter must be a string.")
        if shrinkage_type not in ["l1", "l2"]:
            raise ValueError(
                "The shrinkage_type parameter must be either 'l1' or 'l2'."
            )
        # tol
        if tol is not None:
            if (not isinstance(tol, numbers.Real)) or isinstance(tol, bool):
                raise TypeError("The tol parameter must be numeric.")
            if tol < 0:
                raise ValueError("The tol parameter must be non-negative.")
        # maxiter
        if maxiter is not None:
            if (not isinstance(maxiter, numbers.Real)) or isinstance(maxiter, bool):
                raise TypeError("The maxiter parameter must be an integer.")
            if maxiter <= 0:
                raise ValueError("The maxiter parameter must be positive.")

    def _check_fit_params(self, X, y, sample_weight):
        """
        Checks for fit method parameters.
        """
        # Check type of X
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for the LADRegressor must be a pandas dataframe or "
                "numpy array."
            )
        # Check structure of X: no missing values and all numeric
        if isinstance(X, pd.DataFrame):
            if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
                raise ValueError(
                    "All columns in the input feature matrix must be numeric."
                )
            if X.isnull().values.any():
                raise ValueError(
                    "The input feature matrix for the LADRegressor must not contain any "
                    "missing values."
                )
        else:
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError(
                    "All elements in the input feature matrix must be numeric."
                )
            if np.isnan(X).any():
                raise ValueError(
                    "The input feature matrix for the LADRegressor must not contain any "
                    "missing values."
                )

        # Check type of y
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Dependent variable for the LADRegressor must be a pandas series, "
                "dataframe or numpy array."
            )
        # Check structure of y: single numeric column, no missing values
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "The dependent variable dataframe must have only one column. If used "
                    "as part of an sklearn pipeline, ensure that previous steps return "
                    "a pandas series or dataframe."
                )
            if not pd.api.types.is_numeric_dtype(y.iloc[:, 0]):
                raise ValueError(
                    "The dependent variable dataframe must contain only numeric values."
                )
            if y.isnull().values.any():
                raise ValueError(
                    "The dependent variable dataframe must not contain any missing values."
                )
        elif isinstance(y, pd.Series):
            if not pd.api.types.is_numeric_dtype(y):
                raise ValueError(
                    "The dependent variable series must contain only numeric values."
                )
            if y.isnull().values.any():
                raise ValueError(
                    "The dependent variable series must not contain any missing values."
                )
        else:
            if y.ndim != 1:
                raise ValueError(
                    "The dependent variable numpy array must be 1D. If the dependent "
                    "variable is 2D, please either flatten the array or double check the "
                    "contents of `y`."
                )
            if not np.issubdtype(y.dtype, np.number):
                raise ValueError(
                    "All elements in the dependent variable vector must be numeric."
                )
            if np.isnan(y).any():
                raise ValueError(
                    "The dependent variable for the LADRegressor must not contain any "
                    "missing values."
                )

        if len(X) != len(y):
            raise ValueError(
                "The number of samples in the input feature matrix must match the number "
                "of samples in the dependent variable."
            )

        # sample_weight
        if sample_weight is not None:
            if not isinstance(sample_weight, np.ndarray):
                try:
                    sample_weight = sample_weight.to_numpy().flatten()
                except Exception as e:
                    try:
                        sample_weight = np.array(sample_weight).flatten()
                    except Exception as e:
                        raise TypeError(
                            "The sample weights must be contained within a numpy array."
                        )
            if sample_weight.ndim != 1:
                raise ValueError("The sample weights must be a 1D numpy array.")
            for w in sample_weight:
                if not isinstance(w, numbers.Number):
                    raise ValueError(
                        "All elements of the sample weights must be numeric."
                    )
            if len(sample_weight) != X.shape[0]:
                raise ValueError(
                    "The number of sample weights must match the number of samples in the "
                    "input feature matrix."
                )


if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf

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

    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]

    # Fit model
    model = LADRegressor(
        fit_intercept=True, positive=False, alpha=1, shrinkage_type="l1"
    )
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")
