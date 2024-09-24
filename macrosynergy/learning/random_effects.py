from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from linearmodels.panel import RandomEffects as lm_RandomEffects, PanelOLS, PooledOLS
from sklearn.base import BaseEstimator
from statsmodels.tools.tools import add_constant

import macrosynergy.management as msm
from macrosynergy.management import make_qdf


class RandomEffects(BaseEstimator):
    """
    A custom sklearn estimator that fits a random effects model using linearmodels.
    """

    def __init__(self, group_col: str ="real_date", fit_intercept: bool = True):
        """
        Initialize the RandomEffects estimator.

        :param group_col: The column of a Pandas MultiIndexed DataFrame to group by.
        :param fit_intercept: Whether to fit an intercept term.
        """
        if not isinstance(group_col, str):
            raise ValueError("group_col must be a string.")
        if not isinstance(fit_intercept, bool):
            raise ValueError("fit_intercept must be a boolean.")
        
        self.fit_intercept = fit_intercept
        self.group_col = group_col

        self.params = None
        self.coef_ = None
        self.intercept_ = None
        self.fitted = None
        self.effects = None
        self.idiosyncratic = None
        self.cov = None
        self.r2 = None
        self.sigma2_residuals = None
        self.sigma2_effects = None
        self.theta = None
        self.residuals = None
        self.residual_ss = None
        self.total_ss = None
        self.nobs = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Fit the random effects model.

        :param X: Pandas DataFrame of features with a multiIndex
            containing the group column.
        :param y: Pandas DataFrame of target values with the same index
            as X.

        :return: The fitted estimator.
        """

        X, y = self.check_X_y(X, y)
        
        if self.fit_intercept:
            X = add_constant(X)

        self._fit(X, y)

        return self

    def _fit(self, df_x, df_y):
        """
        Fit the random effects model.

        :param df_x: Pandas DataFrame of features.
        :param df_y: Pandas DataFrame of target values.
        """

        y_demeaned = self._demean(df_y)
        x_demeaned = self._demean(df_x)

        # Fixed Effect Estimation
        params, _, _, _ = np.linalg.lstsq(x_demeaned, y_demeaned, rcond=None)
        eps = y_demeaned.to_numpy() - x_demeaned.to_numpy() @ params

        # Between Estimation
        xbar = df_x.groupby(level=self.group_col).mean()
        ybar = df_y.groupby(level=self.group_col).mean()
        params, ssr, _, _ = np.linalg.lstsq(xbar, ybar, rcond=None)
        alpha = np.asarray(ybar.values) - np.asarray(xbar.values) @ params

        # Estimate variances
        nobs = df_y.shape[0]
        neffects = alpha.shape[0]
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
        cov_matrix = self._cov(x, ssr, nobs)

        index = df_y.index
        fitted = pd.DataFrame(df_x.values @ params, index, ["fitted_values"])
        effects = pd.DataFrame(
            np.asarray(df_y) - np.asarray(fitted) - eps,
            index,
            ["estimated_effects"],
        )
        idiosyncratic = pd.DataFrame(eps, index, ["idiosyncratic"])
        residual_ss = float(np.squeeze(eps.T @ eps))

        if self.fit_intercept:
            y = y - y.mean(0)

        total_ss = float(np.squeeze(y.T @ y))
        r2 = 1 - residual_ss / total_ss
        
        self.params = pd.Series(params.reshape(-1), index=df_x.columns)

        # For sklearn compatibility
        if self.fit_intercept:
            self.intercept_ = self.params["const"]
            self.coef_ = self.params.drop("const").values.reshape(-1)

        self.fitted = fitted
        self.effects = effects
        self.idiosyncratic = idiosyncratic
        self.cov = cov_matrix
        self.r2 = r2
        self.sigma2_residuals = sigma2_e
        self.sigma2_effects = sigma2_a
        self.theta = theta
        self.residuals = eps
        self.residual_ss = residual_ss
        self.total_ss = total_ss
        self.nobs = nobs

    def _demean(self, df: pd.DataFrame):
        """
        Demean the groups as specified by self.group_col.

        Note: If the model is fit with an intercept, the grand mean
            is added back to the demeaned DataFrame.

        :param df: Pandas DataFrame with `self.group_col` as a level in
            its multiIndex.
        """
        mu = df.groupby(level=self.group_col).transform("mean")
        df_demeaned = df - mu
        if self.fit_intercept:
            return (df_demeaned + df.mean(0))
        else:
            return df_demeaned

    def _cov(self, x: np.ndarray, ssr: np.ndarray, nobs: int):
        """
        Compute the covariance matrix of the parameter estimates.

        :param x: The design matrix.
        :param ssr: The sum of squared residuals.
        :param nobs: The number of observations.
        """
        s2 = float(ssr) / nobs
        cov = s2 * np.linalg.inv(x.T @ x)
        return cov
    
    def check_X_y(self, X, y):

        X = self._df_checks(X)
        y = self._df_checks(y)

        if not (X.index == y.index).all():
            raise ValueError("Index of X and y must be the same.")
        
        return X, y

    def _df_checks(self, df: pd.DataFrame):
        """
        Perform checks on the DataFrame input.
        
        :param df: The DataFrame to check.
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame must have a MultiIndex.")

        if self.group_col not in df.index.names:
            raise ValueError(f"Group column '{self.group_col}' not found in index.")
        
        return df
        

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

    # X = X.rename(xs_codes, level=0, inplace=False).copy()
    # y = y.rename(xs_codes, level=0, inplace=False).copy()

    # For each column, obtain Wald test p-value
    # Keep significant features
    for col in feature_names_in_[:1]:
        ftr = X[col].copy()
        ftr = add_constant(ftr)

        rem = RandomEffects(group_col="real_date", fit_intercept=False)
        rem.fit(ftr, y)

        ftr = ftr.rename(xs_codes, level=0, inplace=False).copy()
        y2 = y.rename(xs_codes, level=0, inplace=False).copy()
        ftr = add_constant(ftr)
        re = lm_RandomEffects(y2.swaplevel(), ftr.swaplevel()).fit()
        print(ftr)
        # est = re.params[col]
        # zstat = est / re.std_errors[col]
        # pval = 2 * (1 - stats.norm.cdf(zstat))
        # print(ftr)
        # # ftr.groupby(level=1).transform("mean")
        # y, ybar = random_effects(y, ftr)
        # break
