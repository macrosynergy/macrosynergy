import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin

from scipy.optimize import minimize

class LinearMultiTargetRegression(BaseEstimator, RegressorMixin):
    def __init__(self, feature_selection=None, fit_intercept=True, seemingly_unrelated=False):
        self.feature_selection = feature_selection
        self.fit_intercept = fit_intercept
        self.seemingly_unrelated = seemingly_unrelated

    def fit(self, X, y):
        # For now, only support pandas dataframes with no missing values
        assert isinstance(y, pd.DataFrame)
        assert y.isna().sum().sum() == 0

        # Store data and metadata
        X = X.copy()
        y = y.copy()

        if self.fit_intercept:
            X.insert(0, "intercept", 1)

        self.assets = list(y.columns)
        self.n_assets = len(self.assets)
        self.n_periods = y.shape[0] # Think about this more carefully for the instance where we operate on panel data
        self.features_ = list(X.columns)
        self.n_features = X.shape[1]

        # Store "initial" coefficients for FGLS
        # When seemingly_unrelated = False, these are final OLS coefficients
        # Separate logic when feature selection is applied 
        self.X_features = {}
        self.initial_coefs = {}

        if self.feature_selection is not None:
            for asset in self.assets:
                # Store selected features indices per asset
                if self.fit_intercept:
                    selector = self.feature_selection.fit(X.iloc[:,1:], y[asset])
                    feats = selector.get_feature_names_out()
                    cols = [0] + [X.columns.get_loc(f) for f in feats]
                else:
                    selector = self.feature_selection.fit(X, y[asset])
                    feats = selector.get_feature_names_out()
                    cols = [X.columns.get_loc(f) for f in feats]
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
            W = lr.coef_.T     # shape (n_features × n_assets)
            self.initial_coefs = {asset: W[:, idx] for idx, asset in enumerate(self.assets)}

        # If not SUR, job is done here
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

        # If SUR, first calculate covariance of residuals
        resids = np.column_stack([
            y[asset].to_numpy() - X.iloc[:, self.X_features[asset]].to_numpy() @ self.initial_coefs[asset]
            for asset in self.assets
        ])
        # TODO: replace with graphical lasso 
        cov = np.cov(resids, rowvar=False)
        invcov = np.linalg.inv(cov)

        """
        Set up FGLS optimization. 
        This is tricky with feature selection, as each asset has different features. 
        My logic is to create a full coefficient matrix W of shape (n_features × n_assets),
        but pack and unpack the matrix only for the selected coefficients.
        We don't want the optimizer to update unselected coefficients, so I create a mask
        to indicate which coefficients are real parameters.
        """
        # Initial full W matrix
        W_full = np.zeros((self.n_features, self.n_assets))
        for idx, asset in enumerate(self.assets):
            W_full[self.X_features[asset], idx] = self.initial_coefs[asset]

        # Mask
        mask = np.zeros_like(W_full, dtype=bool)
        for idx, asset in enumerate(self.assets):
            mask[self.X_features[asset], idx] = True

        # Flatten W for scipy optimize
        x0 = W_full[mask]

        #
        self.coefs_, self.intercepts_ = self.__minimise_sur_loss(
            X = X.to_numpy(),
            y = y.to_numpy(),
            invcov = invcov,
            x0 = x0,
            mask = mask
        )

        return self

    def __minimise_sur_loss(self, X, y, invcov, x0, mask):
        # Run LBFGS 
        res = minimize(
            fun = lambda params: self.__sur_loss_and_grad(params, X, y, invcov, mask, return_grad = False),
            x0 = x0,
            jac = lambda params: self.__sur_loss_and_grad(params, X, y, invcov, mask, return_grad = True),
            method="L-BFGS-B",
        )

        # Reshape into full matrix 
        W = np.zeros((self.n_features, self.n_assets))
        W[mask] = res.x

        # Extract intercepts and coefs
        if self.fit_intercept:
            # Convert to asset specific intercepts and coefficients
            coefs_ = {
                asset : W[self.X_features[asset][1:], idx]
                for idx, asset in enumerate(self.assets)
            }
            intercepts_ = {
                asset : W[0, idx]
                for idx, asset in enumerate(self.assets)
            }
        else:
            # Convert to asset specific intercepts and coefficients
            coefs_ = {
                asset : W[self.X_features[asset], idx]
                for idx, asset in enumerate(self.assets)
            }
            intercepts_ = {
                asset : 0
                for asset in self.assets
            }

        return coefs_, intercepts_
    
    def __sur_loss_and_grad(self, params, X, y, invcov, mask, return_grad):
        # First unpack coefficients
        W = np.zeros((self.n_features, self.n_assets))
        W[mask] = params
        
        # Residuals
        resids = y - X @ W 

        if not return_grad:
            loss = np.einsum("ti,ij,tj->", resids, invcov, resids) / self.n_periods
            return loss
        else:
            grad_full = -2 * X.T @ (resids @ invcov) / self.n_periods
            grad = grad_full[mask]
            return grad

    def predict(self, X):
        X = X.copy()
        if self.fit_intercept:
            X.insert(0, "intercept", 1)

        preds = pd.DataFrame(index=X.index, columns=self.assets)

        for asset in self.assets:
            coefs = self.coefs_[asset]
            if self.fit_intercept:
                intercept = self.intercepts_[asset]
                preds[asset] = X.iloc[:, self.X_features[asset][1:]].to_numpy() @ coefs + intercept
            else:
                preds[asset] = X.iloc[:, self.X_features[asset]].to_numpy() @ coefs

        return preds.values
    
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
    from macrosynergy.learning import (
        ExpandingKFoldPanelSplit, SignalOptimizer
    )
    from macrosynergy.management.simulate import make_qdf
    from macrosynergy.management.types import QuantamentalDataFrame

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR1", "CRY", "GROWTH", "XR2"]
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
        xcats=["CRY", "GROWTH", "XR1", "XR2"],
        cids=cids,
        blacklist=black,
        drop_nas = True,
        n_targets=2,
    )
    X = so.X.copy(deep=True)
    y = so.y.copy(deep=True)

    model = LinearMultiTargetRegression(seemingly_unrelated=True)
    model.fit(X, y)
    model.predict(X)