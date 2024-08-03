"""
Tools to produce, visualise and use walk-forward validation splits across panels.
"""

import datetime
from typing import Optional, List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    cross_validate,
)
from sklearn.linear_model import Lasso, LinearRegression


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    # """Example 1: Unbalanced panel """

    df_cids2 = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids2.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats2 = pd.DataFrame(index=xcats, columns=cols)
    df_xcats2.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats2.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats2.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)

    dfd2 = dfd2.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X2 = dfd2.drop(columns=["XR"])
    y2 = dfd2["XR"]

    # 1) Demonstration of basic functionality

    # a) n_splits = 4, n_split_method = expanding
    """splitter = ExpandingKFoldPanelSplit(n_splits=4)
    splitter.split(X2, y2)
    cv_results = cross_validate(
        LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
    )
    splitter.visualise_splits(X2, y2)

    # b) n_splits = 4, n_split_method = expanding, Australia only visualisation
    splitter.visualise_splits(X2[X2.index.get_level_values(0)=="AUD"], y2[y2.index.get_level_values(0)=="AUD"])

    # c) n_splits = 4, n_split_method = rolling
    splitter = RollingKFoldPanelSplit(n_splits=4)
    splitter.split(X2, y2)
    cv_results = cross_validate(
        LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
    )
    splitter.visualise_splits(X2, y2)

    # d) n_splits = 4, n_split_method = rolling, Canada only visualisation
    splitter.visualise_splits(X2[X2.index.get_level_values(0)=="CAD"], y2[y2.index.get_level_values(0)=="CAD"]) 

    # e) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12, test_size=1, min_periods=21 * 12, min_cids=4
    )
    splitter.split(X2, y2)
    cv_results = cross_validate(
        LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
    )
    splitter.visualise_splits(X2, y2)"""

    splitter = ExpandingFrequencyPanelSplit(
        expansion_freq="W", test_freq="Y", min_cids=4, min_periods=21 * 12
    )
    splitter.visualise_splits(X2, y2)

    # f) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4, Britain only visualisation
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12, test_size=1, min_periods=21 * 12, min_cids=1
    )
    splitter.visualise_splits(X2, y2)

    # g) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4, max_periods=12*21
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12,
        test_size=21 * 12,
        min_periods=21 * 12,
        min_cids=4,
        max_periods=12 * 21,
    )
    splitter.split(X2, y2)
    cv_results = cross_validate(
        LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
    )
    splitter.visualise_splits(X2, y2)

    # h) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4, max_periods=12*21, USD only visualisation
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12,
        test_size=21 * 12,
        min_periods=21 * 12,
        min_cids=1,
        max_periods=12 * 21,
    )
    splitter.visualise_splits(X2[X2.index.get_level_values(0)=="USD"], y2[y2.index.get_level_values(0)=="USD"])

    # 2) Grid search capabilities
    lasso = Lasso()
    parameters = {"alpha": [0.1, 1, 10]}
    splitter = ExpandingIncrementPanelSplit(
        train_intervals=21 * 12, test_size=1, min_periods=21, min_cids=4
    )
    gs = GridSearchCV(
        lasso,
        parameters,
        cv=splitter,
        scoring="neg_root_mean_squared_error",
        refit=False,
        verbose=3,
    )
    gs.fit(X2, y2)
    print(gs.best_params_)
