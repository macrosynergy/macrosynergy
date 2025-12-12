import numpy as np
import pandas as pd
import numbers 

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SelectorMixin

from scipy.optimize import minimize

class LinearMultiTargetRegression(BaseEstimator, RegressorMixin):
    """
    Linear regression model with multiple targets, supporting seemingly unrelated
    regression (SUR) via feasible generalized least squares (FGLS).

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term in the regression.
    seemingly_unrelated : bool, default=False
        Whether to make the regression seemingly unrelated.
    covariance_estimator : Union[str, BaseEstimator], default="ewm"
        Choice of covariance estimator. Options are "ml" for maximum likelihood, 
        "ewm" for exponentially weighted moving covariance, or a custom `scikit-learn`
        compatible covariance estimator.
    span : int, default=60
        Span parameter for exponentially weighted covariance estimation of residuals.
    feature_selection : object, default=None
        A feature selection object inheriting from scikit-learn's `SelectorMixin` base
        class in `sklearn.feature_selection`.
        If provided, feature selection is applied per target before fitting. 
    """
    def __init__(
        self,
        fit_intercept=True,
        seemingly_unrelated=False,
        covariance_estimator = "ewm",
        span=60,
        feature_selection=None,
    ):
        # Checks
        if not isinstance(fit_intercept, bool):
            raise TypeError("The 'fit_intercept' parameter must be a boolean.")
        if not isinstance(seemingly_unrelated, bool):
            raise TypeError("The 'seemingly_unrelated' parameter must be a boolean.")
            
        if not isinstance(covariance_estimator, (str, BaseEstimator)):
            raise TypeError("The 'covariance_estimator' parameter must be a string or a BaseEstimator instance.")
        if isinstance(covariance_estimator, str) and covariance_estimator not in ["ml", "ewm"]:
            raise ValueError("If `covariance_estimator` is a string, it must be either 'ml' or 'ewm'.")
        # TODO: add checks for custom covariance estimator when inheriting from BaseEstimator
        if covariance_estimator == "ewm":
            if not isinstance(span, numbers.Integral):
                raise TypeError("The 'span' parameter must be an integer.")
            if span <= 0:
                raise ValueError("The 'span' parameter must be positive.")
        if feature_selection is not None and not isinstance(
            feature_selection, SelectorMixin
        ):
            raise TypeError(
                "The 'feature_selection' parameter must be a SelectorMixin instance."
            )
        
        # Attributes
        self.fit_intercept = fit_intercept
        self.seemingly_unrelated = seemingly_unrelated
        self.covariance_estimator = covariance_estimator
        self.span = span
        self.feature_selection = feature_selection

    def fit(self, X, y, sample_weight=None):
        """
        Fit the linear multi-target regression model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of shape (n_samples, n_features). Should be multi-indexed by 
            asset and real date.
        y : pd.DataFrame
            Target matrix of shape (n_samples, n_assets). Should be multi-indexed by 
            asset and real date.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
        """
        # Checks
        self._check_fit_params(X, y, sample_weight)
        
        # For now, only support pandas dataframes with no missing values
        if isinstance(y, pd.Series):
            y = y.to_frame()
        assert isinstance(y, pd.DataFrame)
        assert y.isna().sum().sum() == 0

        # Store data and metadata
        X = X.copy()
        y = y.copy()

        if self.fit_intercept:
            X.insert(0, "intercept", 1)

        self.assets = list(y.columns)
        self.n_assets = len(self.assets)
        self.n_samples = y.shape[
            0
        ] 
        self.features_ = list(X.columns)
        self.n_features = X.shape[1]

        # Store "initial" coefficients for FGLS
        # When seemingly_unrelated = False, these are final OLS coefficients
        # Separate logic when feature selection is applied
        self.X_features = {}
        self.initial_coefs = {}

        if self.feature_selection is not None:
            col_index = {col: i for i, col in enumerate(X.columns)} # includes an intercept if fit_intercept=True
            for asset in self.assets:
                # Store selected features indices per asset
                if self.fit_intercept:
                    selector = self.feature_selection.fit(X.iloc[:, 1:], y[asset])
                    feats = selector.get_feature_names_out()
                    cols = [0] + [col_index[f] for f in feats]
                else:
                    selector = self.feature_selection.fit(X, y[asset])
                    feats = selector.get_feature_names_out()
                    cols = [col_index[f] for f in feats]
                self.X_features[asset] = cols

                # Fit OLS on selected features and store them
                lr = LinearRegression(fit_intercept=False)
                lr.fit(X.iloc[:, cols], y[asset])
                self.initial_coefs[asset] = lr.coef_
        else:
            # Simply use all features for all assets
            cols = list(range(self.n_features))
            self.X_features = {asset: cols for asset in self.assets}

            # Fit OLS jointly and store coefficients
            lr = LinearRegression(fit_intercept=False).fit(X, y)
            W = lr.coef_.T  # shape (n_features × n_assets)
            self.initial_coefs = {
                asset: W[:, idx] for idx, asset in enumerate(self.assets)
            }

        # If not sur, job is done here
        # Just store OLS coefficients
        if not self.seemingly_unrelated:
            self.coefs_ = {}
            self.intercepts_ = {}

            for asset in self.assets:
                asset_coefs = self.initial_coefs[asset]
                if self.fit_intercept:
                    self.intercepts_[asset] = asset_coefs[0]
                    self.coefs_[asset] = asset_coefs[1:]
                else:
                    self.intercepts_[asset] = 0
                    self.coefs_[asset] = asset_coefs

            return self

        # If sur, calculate covariance of residuals
        resids = pd.DataFrame(
            data=np.column_stack(
                [
                    y[asset].to_numpy()
                    - X.iloc[:, self.X_features[asset]].to_numpy()
                    @ self.initial_coefs[asset]
                    for asset in self.assets
                ]
            ),
            index=y.index,
            columns=y.columns,
        )

        # Estimate covariance matrix
        if self.covariance_estimator == "ewm":
            weights = np.array(
                [(1 - 2 / (self.span + 1)) ** i for i in range(len(y))][::-1]
            )
            cov = np.cov(resids.values.T, aweights=weights)
        elif self.covariance_estimator == "ml":
            cov = np.cov(resids.values.T)
        else:
            self.covariance_estimator.fit(resids)
            cov = self.covariance_estimator.covariance_
        
        # Invert matrix 
        if isinstance(self.covariance_estimator, BaseEstimator) and hasattr(
            self.covariance_estimator, "precision_"
        ):
            invcov = self.covariance_estimator.precision_
        else:
            invcov = np.linalg.inv(cov)

        # FGLS optimization
        # Due to feature selection, create a full matrix but pack/unpack the matrix 
        # only to update trainable coefficients. 
        W_full = np.zeros((self.n_features, self.n_assets))
        for idx, asset in enumerate(self.assets):
            W_full[self.X_features[asset], idx] = self.initial_coefs[asset] # Initialize with OLS coefficients

        # Mask
        mask = np.zeros_like(W_full, dtype=bool) # matrix of Falses
        for idx, asset in enumerate(self.assets):
            mask[self.X_features[asset], idx] = True

        # Flatten W for scipy optimize
        x0 = W_full[mask]

        # Minimise SUR_SELECT loss over the panel
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).reshape(-1)

        self.coefs_, self.intercepts_ = self.__minimise_sur_loss(
            X=X.to_numpy(),
            y=y.to_numpy(),
            invcov=invcov,
            x0=x0,
            mask=mask,
            sample_weight=sample_weight,
        )

        return self

    def __minimise_sur_loss(self, X, y, invcov, x0, mask, sample_weight):
        """
        Fit the SUR model via minimization of the SUR loss function. Coefficients
        and intercepts are extracted after optimization.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target matrix of shape (n_samples, n_assets).
        invcov : np.ndarray
            Inverse of the covariance matrix of residuals.
        x0 : np.ndarray
            Initial coefficients for optimization.
        mask : np.ndarray
            Boolean mask of shape (n_features, n_assets) indicating which coefficients
            are trainable.
        sample_weight : np.ndarray or None
            Optional sample weights for the loss function.
        """
        # Run LBFGS
        res = minimize(
            fun=lambda params: self.__sur_loss_and_grad(
                params, X, y, invcov, mask, sample_weight, return_grad=False
            ),
            x0=x0,
            jac=lambda params: self.__sur_loss_and_grad(
                params, X, y, invcov, mask, sample_weight, return_grad=True
            ),
            method="L-BFGS-B",
        )

        # Reshape into full matrix
        W = np.zeros((self.n_features, self.n_assets))
        W[mask] = res.x

        # Extract intercepts and coefs
        # Convert to asset specific intercepts and coefficients
        if self.fit_intercept:
            coefs_ = {
                asset: W[self.X_features[asset][1:], idx]
                for idx, asset in enumerate(self.assets)
            }
            intercepts_ = {asset: W[0, idx] for idx, asset in enumerate(self.assets)}
        else:
            coefs_ = {
                asset: W[self.X_features[asset], idx]
                for idx, asset in enumerate(self.assets)
            }
            intercepts_ = {asset: 0 for asset in self.assets}

        return coefs_, intercepts_

    def __sur_loss_and_grad(
        self, params, X, y, invcov, mask, sample_weight, return_grad
    ):
        """
        SUR loss and derivative evaluation.

        Parameters
        ----------
        params : np.ndarray
            Flattened array of trainable coefficients.
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target matrix of shape (n_samples, n_assets).
        invcov : np.ndarray
            Inverse of the covariance matrix of residuals.
        mask : np.ndarray
            Boolean mask of shape (n_features, n_assets) indicating which coefficients
            are trainable.
        sample_weight : np.ndarray or None
            Optional sample weights for the loss function.
        return_grad : bool
            Whether to return the gradient instead of the loss.
        """
        # First unpack coefficients
        W = np.zeros((self.n_features, self.n_assets))
        W[mask] = params

        # Residuals
        resids = y - X @ W

        if not return_grad:
            if sample_weight is None:
                loss = np.einsum("ti,ij,tj->", resids, invcov, resids) / self.n_samples
            else:
                loss = (
                    np.einsum("ti,ij,tj,t->", resids, invcov, resids, sample_weight)
                    / self.n_samples
                )

            return loss
        else:
            wresids = resids @ invcov
            if sample_weight is not None:
                wresids = wresids * sample_weight[:, None]

            grad_full = -2 * (X.T @ wresids) / self.n_samples
            grad = grad_full[mask]

            return grad

    def predict(self, X):
        """
        Predict method to return predictions for each asset.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of shape (n_samples, n_features). Should be multi-indexed by 
            asset and real date.
        """
        # Checks
        self._check_predict_params(X)

        # Add intercept and set up predictions dataframe
        X = X.copy()
        if self.fit_intercept:
            X.insert(0, "intercept", 1)

        preds = pd.DataFrame(index=X.index, columns=self.assets)

        # Evaluate for each asset with asset specific coefficients.
        for asset in self.assets:
            coefs = self.coefs_[asset]
            if self.fit_intercept:
                intercept = self.intercepts_[asset]
                preds[asset] = (
                    X.iloc[:, self.X_features[asset][1:]].to_numpy() @ coefs + intercept
                )
            else:
                preds[asset] = X.iloc[:, self.X_features[asset]].to_numpy() @ coefs

        return preds.values
    
    def _check_fit_params(self, X, y, sample_weight):
        # Type checks for X and y
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, (pd.DataFrame, pd.Series)):
            raise TypeError("y must be a pandas DataFrame or Series.")
        # The dataframe must be multi indexed by asset and real date
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed by asset and real date.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed by asset and real date.")
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same multi-index.")
        # This model can't handle NAs.
        if X.isna().sum().sum() > 0:
            raise ValueError("X must not contain missing values.")
        if y.isna().sum().sum() > 0:
            raise ValueError("y must not contain missing values.")
        # Ensure shapes align
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        # If sample weight is set, make sure its shape and type are correct
        if sample_weight is not None:
            if not isinstance(sample_weight, (np.ndarray, list)):
                raise TypeError("sample_weight must be a numpy array or list.")
            if len(sample_weight) != X.shape[0]:
                raise ValueError("sample_weight must have the same number of samples as X and y.")
            for weight in sample_weight:
                if not isinstance(weight, numbers.Number):
                    raise TypeError("All entries in sample_weight must be numeric.")
                if weight < 0:
                    raise ValueError("All entries in sample_weight must be non-negative.")

    def _check_predict_params(self, X):
        # Type checks for X
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        # The dataframe must be multi indexed by asset and real date
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed by asset and real date.")
        # This model can't handle NAs.
        if X.isna().sum().sum() > 0:
            raise ValueError("X must not contain missing values.")
        # Ensure the features align with those seen in training
        if self.fit_intercept:
            if X.shape[1] != self.n_features - 1:
                raise ValueError("X must have the same number of features as during training.")
            if list(X.columns) != list(self.features_[1:]):
                raise ValueError("X must have the same feature columns as during training.")
        else:
            if X.shape[1] != self.n_features:
                raise ValueError("X must have the same number of features as during training.")
            if list(X.columns) != list(self.features_):
                raise ValueError("X must have the same feature columns as during training.")

if __name__ == "__main__":
    from macrosynergy.learning import (
        SignalOptimizer,
        LarsSelector,
    )
    from macrosynergy.management.simulate import make_qdf

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR1", "CRY", "GROWTH", "RATES", "XR2"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2012-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR1"] = ["2012-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2012-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["RATES"] = ["2010-01-01", "2020-12-31", 0, 1, 0.5, 0.5]
    df_xcats.loc["XR2"] = ["2015-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {
        "GBP": (
            pd.Timestamp(year=2009, month=1, day=1),
            pd.Timestamp(year=2012, month=6, day=30),
        ),
        "CAD": (
            pd.Timestamp(year=2015, month=1, day=1),
            pd.Timestamp(year=2016, month=1, day=1),
        ),
    }

    so = SignalOptimizer(
        df=dfd,
        xcats=["CRY", "GROWTH", "RATES", "XR1", "XR2"],
        cids=cids,
        blacklist=black,
        drop_nas=True,
        n_targets=2,
    )
    X = so.X.copy(deep=True)
    y = so.y.copy(deep=True)

    model = LinearMultiTargetRegression(
        seemingly_unrelated=True,
        fit_intercept=False,
        feature_selection=LarsSelector(n_factors=2),
    )
    model.fit(X, y)

    print(model.predict(X))
