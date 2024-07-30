import numpy as np
import pandas as pd

import datetime

from scipy.sparse._csr import csr_matrix
import scipy.stats as stats

from sklearn.linear_model import Lasso, ElasticNet, Lars
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.feature_selection import SelectorMixin, SelectFromModel
from sklearn.exceptions import NotFittedError

from statsmodels.tools.tools import add_constant

from linearmodels.panel import RandomEffects

from typing import Union, Any, Optional

import warnings

from macrosynergy.management import make_qdf
import macrosynergy.management as msm

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


def demean(df):
    mu = df.groupby(level=1).transform("mean")
    return df - mu


def random_effects(df_y, df_x):
    if isinstance(df_y, pd.Series):
        df_y = df_y.to_frame()

    y_demeaned = demean(df_y)
    x_demeaned = demean(df_x)
    y_demeaned += df_y.mean(0)
    x_demeaned += df_x.mean(0)
    y_demeaned = y_demeaned.to_numpy()
    x_demeaned = x_demeaned.to_numpy()

    params, ssr, _, _ = np.linalg.lstsq(x_demeaned, y_demeaned, rcond=None)
    eps = y_demeaned - x_demeaned @ params

    # Between Estimation
    xbar = df_x.groupby(level=1).mean()
    ybar = df_y.groupby(level=1).mean()
    params, ssr, _, _ = np.linalg.lstsq(xbar, ybar, rcond=None)
    u = np.asarray(ybar.values) - np.asarray(xbar.values) @ params

    # Estimate variances
    nobs = df_y.shape[0]
    neffects = u.shape[0]
    nvars = df_x.shape[1]

    # Idiosyncratic variance
    sigma2_e = float(np.squeeze(eps.T @ eps)) / (nobs - nvars - neffects + 1)
    rss = float(np.squeeze(u.T @ u))
    cid_count = np.asarray(df_y.groupby(level=1).count())

    cid_bar = neffects / ((1.0 / cid_count)).sum()

    sigma2_a = max(0.0, (ssr / (neffects - nvars)) - sigma2_e / cid_bar)

    # Theta
    theta = 1.0 - np.sqrt(sigma2_e / (cid_count * sigma2_a + sigma2_e))

    print(theta.mean())
    index = df_y.index
    reindex = index.levels[1][index.codes[1]]
    ybar = (theta * ybar).loc[reindex]
    xbar = (theta * xbar).loc[reindex]
    print(ybar.sum())
    y = np.asarray(df_y)
    x = np.asarray(df_x)

    y = y - ybar.values
    x = x - xbar.values
    print(y.shape, x.shape)
    params, ssr, _, _ = np.linalg.lstsq(x, y, rcond=None)
    eps = y - x @ params

    index = df_y.index
    fitted = pd.DataFrame(df_x.values @ params, index, ["fitted_values"])
    effects = pd.DataFrame(
        np.asarray(df_y) - np.asarray(fitted) - eps,
        index,
        ["estimated_effects"],
    )
    idiosyncratic = pd.DataFrame(eps, index, ["idiosyncratic"])

    residual_ss = float(np.squeeze(eps.T @ eps))

    y_demeaned = y - y.mean(0)
    total_ss = float(np.squeeze(y_demeaned.T @ y_demeaned))
    r2 = 1 - residual_ss / total_ss
    print(x)
    print(y)
    print(x.sum(), y.sum())
    print(params)
    # print(nobs, neffects, nvars)
    # print(cid_count.max())
    # print((rss / (nobs - nvars)))
    # print(cid_count * sigma2_a + sigma2_e)
    # print('theta', theta, theta.max(), theta.sum())

    return x_demeaned
    # return res


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
    ftr = add_constant(ftr)
    # Swap levels so that random effects are placed on each time period,
    # as opposed to the cross-section
    re = RandomEffects(y.swaplevel(), ftr.swaplevel()).fit()
    est = re.params[col]
    zstat = est / re.std_errors[col]
    pval = 2 * (1 - stats.norm.cdf(zstat))

    # ftr.groupby(level=1).transform("mean")
    y, ybar = random_effects(y, ftr)
    break
