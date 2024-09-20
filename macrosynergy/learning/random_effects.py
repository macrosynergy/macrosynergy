from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from linearmodels.panel import RandomEffects as lm_RandomEffects, PanelOLS, PooledOLS
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from statsmodels.tools.tools import add_constant

import macrosynergy.management as msm
from macrosynergy.management import make_qdf


class RandomEffects(BaseEstimator):
    """
    A custom sklearn estimator that fits a random effects model using linearmodels.
    """

    def __init__(self, group_col, add_constant=True):
        self.add_constant = add_constant
        self.group_col = group_col

    def fit(self, X, y):
        """
        Fit the random effects model.

        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            Training data, including the group column.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y, accept_pd_dataframe=True)

        if isinstance(y, pd.Series):
            y = y.to_frame()
        if isinstance(X, pd.Series):
            X = X.to_frame()

        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        # Check if group_col exists in X
        if self.group_col not in X.index.names:
            raise ValueError(f"Group column '{self.group_col}' not found in X's index.")

        if self.add_constant:
            X = add_constant(X)

        # Fit the random effects model
        self._fit(X, y)

        return self

    def _fit(self, df_x, df_y):

        y_demeaned = self.demean(df_y)
        x_demeaned = self.demean(df_x)

        # Fixed Effect Estimation
        params, ssr, _, _ = np.linalg.lstsq(x_demeaned, y_demeaned, rcond=None)
        eps = y_demeaned - x_demeaned @ params

        # Between Estimation
        xbar = df_x.groupby(level=self.group_col).mean()
        ybar = df_y.groupby(level=self.group_col).mean()
        params, ssr, _, _ = np.linalg.lstsq(xbar, ybar, rcond=None)
        u = np.asarray(ybar.values) - np.asarray(xbar.values) @ params

        # Estimate variances
        nobs = df_y.shape[0]
        neffects = u.shape[0]
        nvars = df_x.shape[1]

        # Idiosyncratic variance
        sigma2_e = float(np.squeeze(eps.T @ eps)) / (nobs - nvars - neffects + 1)

        obs_counts = np.asarray(df_y.groupby(level=self.group_col).count())
        obs_harmonic_mean = neffects / ((1.0 / obs_counts)).sum()

        # Random effect variance
        sigma2_a = max(0.0, (ssr / (neffects - nvars)) - sigma2_e / obs_harmonic_mean)

        # Theta
        theta = 1.0 - np.sqrt(sigma2_e / (obs_counts * sigma2_a + sigma2_e))

        index = df_y.index
        reindex = index.levels[1][index.codes[1]]

        # Random effects estimation
        ybar = (theta * ybar).loc[reindex]
        xbar = (theta * xbar).loc[reindex]

        y = np.asarray(df_y)
        x = np.asarray(df_x)

        y = y - ybar.values
        x = x - xbar.values

        params, ssr, _, _ = np.linalg.lstsq(x, y, rcond=None)
        eps = y - x @ params

        # Covariance estimation
        cov_matrix = self._cov(x, eps, nobs)

        index = df_y.index
        fitted = pd.DataFrame(df_x.values @ params, index, ["fitted_values"])
        effects = pd.DataFrame(
            np.asarray(df_y) - np.asarray(fitted) - eps,
            index,
            ["estimated_effects"],
        )
        idiosyncratic = pd.DataFrame(eps, index, ["idiosyncratic"])
        residual_ss = float(np.squeeze(eps.T @ eps))

        if self.add_constant:
            y = y - y.mean(0)

        total_ss = float(np.squeeze(y.T @ y))
        r2 = 1 - residual_ss / total_ss

        self.set_params(
            coef_=params,
            fitted=fitted,
            effects=effects,
            idiosyncratic=idiosyncratic,
            cov=cov_matrix,
            r2=r2,
            sigma2_residuals=sigma2_e,
            sigma2_effects=sigma2_a,
        )

    def get_params(self):
        """Get parameters for this estimator."""
        return {"group_col": self.group_col}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def demean(self, df):
        mu = df.groupby(level=self.group_col).transform("mean")
        return (df - mu + df.mean(0)).to_numpy()

    def _s2(self, eps, _nobs, _scale=1.0):
        return _scale * float(np.squeeze(eps.T @ eps)) / _nobs

    def _cov(self, x, eps, _nobs):
        s2 = self._s2(eps, _nobs)
        cov = s2 * np.linalg.inv(x.T @ x)
        return cov


if __name__ == "__main__":

    np.random.seed(1)

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids2 = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids2.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["CAD"] = ["2013-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["GBP"] = ["2010-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["USD"] = ["2010-01-01", "2020-12-31", 0, 1]

    df_xcats2 = pd.DataFrame(index=xcats, columns=cols)
    df_xcats2.loc["XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats2.loc["CRY"] = ["2010-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats2.loc["GROWTH"] = ["2010-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2010-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)

    dfd2 = dfd2.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    XX = dfd2.drop(columns=["XR"])
    yy = dfd2["XR"]

    X, y = XX.copy(), yy.copy()

    ftrs = []
    feature_names_in_ = np.array(X.columns)

    # Convert cross-sections to numeric codes for compatibility with RandomEffects
    unique_xss = sorted(X.index.get_level_values(0).unique())
    xs_codes = dict(zip(unique_xss, range(1, len(unique_xss) + 1)))

    X = X.rename(xs_codes, level=0, inplace=False).copy()
    y = y.rename(xs_codes, level=0, inplace=False).copy()

    # For each column, obtain Wald test p-value
    # Keep significant features
    for col in feature_names_in_[:1]:
        ftr = X[col].copy()
        # ftr = add_constant(ftr)
        # # Swap levels so that random effects are placed on each time period,
        # # as opposed to the cross-section
        # print(ftr)

        # rem = RandomEffects(group_col="real_date", add_constant=False)
        # rem.fit(ftr, y)

        # ftr = add_constant(ftr)
        re = lm_RandomEffects(y.swaplevel(), ftr.swaplevel()).fit()
        print(ftr)
        # est = re.params[col]
        # zstat = est / re.std_errors[col]
        # pval = 2 * (1 - stats.norm.cdf(zstat))
        # print(ftr)
        # # ftr.groupby(level=1).transform("mean")
        # y, ybar = random_effects(y, ftr)
        # break
