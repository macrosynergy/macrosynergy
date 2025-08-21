import numpy as np 
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize

class GlobalLocalRegression(BaseEstimator, RegressorMixin):
    r"""
    Linear panel model with hierarchical shrinkage of country-specific (local) coefficients 
    towards unknown global coefficients. Learning means that both country-specific and 
    global coefficients are estimated from data.

    Parameters
    ----------
    local_lambda : float, default=1
        Regularization strength to pull local coefficients towards global coefficients.
    global_lambda : float, default=1
        Regularization strength to pull global coefficients towards zero. 
    positive : bool, default=False
        Whether to constrain all coefficients to be positive. Default is False.
    fit_intercept : bool, default=True
        Whether to fit an intercept term. Default is True.
    min_xs_samples : int, default=36
        Minimum number of samples required in each group for the group to be considered 
        a contribution to the mean squared error component of the loss function.

    Notes
    -----
    A panel can be modelled from a global perspective, where time series of all countries
    are "pooled" or stacked together, meaning that samples from different countries are 
    treated as independent. This is called a pooled regression. With one model fit on all
    countries' data, this is a high-bias, low-variance model. 

    Alternatively, country-by-country regressions can be fit, with a separate model for
    each country. This is low-bias but high-variance, since each model sees less data.

    This implies that a balance can be found between these two extremes by balancing 
    this bias-variance trade-off. Introduction of bias to the country-by-country models
    can lead to a potentially substantial reduction in variance. Mathematically, this fit 
    is found by minimizing the sum of squared residuals for each country, with a term 
    that penalizes deviation of country-specific coefficients from a global coefficient. 
    The global coefficient is also penalized to prevent it from growing too large.

    The loss function is as follows:

    .. math::

            L(\{\beta_i\}_{i=1}^{C}, \beta) = \frac{1}{C} \sum_{i = 1}^{C} \left [ \frac{1}{n_{i}}  \sum_{t=1}^{n_{i}} (y_{it} - x_{it}^{\intercal} \beta_{i})^2 \right ] + \lambda_{\text{local}} \sum_{i=1}^{C} ||\beta_i - \beta||_{2}^{2} + \lambda_{\text{global}} ||\beta||_{2}^{2}
    """
    def __init__(self, local_lambda = 1, global_lambda = 1, positive = False, fit_intercept = True, min_xs_samples = 36):
        # Attributes
        self.local_lambda = local_lambda
        self.global_lambda = global_lambda
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.min_xs_samples = min_xs_samples
    
    def fit(self, X, y, sample_weight = None):
        """
        Fit the global-local model.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix, multi-indexed by `cid` and `real_date`.
        y : pd.DataFrame or pd.Series
            Target vector associated with each sample in X, multi-indexed by `cid` and
            `real_date`.
        sample_weight : np.ndarray, optional
            Sample weights for each sample in X. If provided, it should be a 1D
            array with the same length as the number of samples in X. If None, all samples
            are treated equally.

        Returns
        -------
        self
            Fitted estimator.
        """
        # Fit
        if self.fit_intercept:
            X = X.copy()
            X.insert(0, "intercept", 1)
            
        self.cids_ = sorted(X.index.get_level_values(0).unique())
        self.n_cids_ = len(self.cids_)
        self.n_features_ = X.shape[1]
        self.X_cid_ = {}
        self.y_cid_ = {}
        self.Xy_cid_weights_ = None if sample_weight is None else {}
        
        for cid in self.cids_:
            self.X_cid_[cid] = X.loc[cid].values
            self.y_cid_[cid] = y.loc[cid].values
            if sample_weight is not None:
                self.Xy_cid_weights_[cid] = sample_weight[y.index.get_level_values(0)==cid]

        # Initialise with zeros
        x0 = np.zeros((self.n_cids_ + 1) * self.n_features_)

        # Optional bounds
        if self.fit_intercept:
            if self.positive:
                # Place positive restrictions on the feature weights not intercepts
                bounds = []
                for _ in range(self.n_cids_ + 1):
                    bounds.append((None, None)) 
                    bounds.extend([(0, None)] * (self.n_features_ - 1)) 
            else:
                bounds = [(None, None)] * len(x0)
        else:
            if self.positive:
                bounds = [(0, None)] * len(x0)
            else:
                bounds = [(None, None)] * len(x0)

        result = minimize(
            fun=self.loss,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            jac=self.loss_derivative,
        )

        coefs = result.x.reshape(self.n_cids_ + 1, self.n_features_)
        self.cid_coefs_ = {cid: coefs[i] for i, cid in enumerate(self.cids_)}
        
        if self.fit_intercept:
            self.coef_ = coefs[-1, 1:]
            self.intercept_ = coefs[-1, 0]
        else:
            self.coef_ = coefs[-1]
            self.intercept_ = None

        return self
    
    def loss(self, weights):
        """
        Loss function for the global-local regression model.

        Parameters
        ----------
        weights : np.ndarray
            Flattened array of weights, where the last `n_features_` elements correspond
            with the global coefficients and the rest correspond to the local coefficients
            for each country.
        """
        weights = weights.reshape(self.n_cids_ + 1, self.n_features_)
        total_loss = 0.0
        global_beta = weights[-1, :]
        
        # Likelihood
        for i, g in enumerate(self.cids_):
            X_g = self.X_cid_[g]
            y_g = self.y_cid_[g]
            preds = X_g @ weights[i]
            residuals = y_g - preds

            if self.Xy_cid_weights_ is not None:
                w_g = self.Xy_cid_weights_[g]
                total_loss += np.average(residuals**2, weights=w_g)
            else:
                total_loss += np.mean(residuals**2)
        
        # Local-to-global regularization term
        if self.local_lambda > 0:
            reg = np.sum((weights[:-1] - global_beta)**2)
            total_loss += self.local_lambda * reg
            
        # Global regularization term
        if self.global_lambda > 0:
            reg = np.sum(global_beta**2)
            total_loss += self.global_lambda * reg
            
        return total_loss
    
    def loss_derivative(self, weights):
        """
        Derivative of the loss function with respect to the weights.

        Parameters
        ----------
        weights : np.ndarray
            Flattened array of weights, where the last `n_features_` elements correspond
            with the global coefficients and the rest correspond to the local coefficients
            for each country. 
        """
        weights = weights.reshape(self.n_cids_ + 1, self.n_features_)
        global_beta = weights[-1]
        
        grads = np.zeros_like(weights)

        for i, cid in enumerate(self.cids_):
            X_i = self.X_cid_[cid]
            y_i = self.y_cid_[cid]
            beta_i = weights[i]
            preds = X_i @ beta_i
            residual = preds - y_i
            
            if self.Xy_cid_weights_ is not None:
                w_i = self.Xy_cid_weights_[cid]
                grad_i = 2 * (X_i.T @ (residual * w_i)) / np.sum(w_i)
            else:
                grad_i = 2 * X_i.T @ residual / len(residual)

            grad_i += 2 * self.local_lambda * (beta_i - global_beta)
            grads[i] = grad_i

        # Gradient for global beta
        grad_g = 2 * self.local_lambda * np.sum(global_beta - weights[:-1], axis=0) + 2 * self.global_lambda * global_beta
        grads[-1] = grad_g

        return grads.flatten()
    
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
        if self.fit_intercept:
            X = X.copy()
            X.insert(0, "intercept", 1)
            
        group_indices = X.index.get_level_values(0)
        X_values = X.values
        preds = np.zeros(len(X))

        for group in self.cids_:
            mask = group_indices == group
            if mask.any():
                beta = self.cid_coefs_.get(group, np.zeros(self.n_features_))
                preds[mask] = X_values[mask] @ beta

        return preds
    
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

    # Regressor
    X_train = train.drop(columns=["XR"])
    y_train = np.sign(train["XR"])

    glr = GlobalLocalRegression().fit(X_train, y_train)
    print(glr.predict(X_train))