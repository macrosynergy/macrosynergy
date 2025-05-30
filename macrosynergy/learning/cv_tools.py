"""
A set of tools for cross-validation of panel data.
"""

import numpy as np
import pandas as pd
import datetime

from macrosynergy.learning import ExpandingKFoldPanelSplit
from macrosynergy.learning.splitters.base_splitters import BasePanelSplit

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer


def panel_cv_scores(
    X,
    y,
    splitter,
    estimators,
    scoring,
    show_longbias = True,
    show_std = False,
    verbose = 1,
    n_jobs = -1,
):
    """
    Returns a dataframe of cross-validation scores for a collection of models, with
    respect to a cross-validation splitter and a set of scorers.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    y : pd.DataFrame or pd.Series
        Target variable.
    splitter : BasePanelSplit
        Panel cross-validation splitter.
    estimators : dict
        Dictionary of models.
    scoring : dict
        Dictionary of scorers.
    show_longbias : bool, optional, default=True
        Whether to show the proportion of times a model predicts a positive return.
    show_std : bool, optional, default=False
        Whether to show the standard deviation of the cross-validation scores over folds.
    verbose : int, optional, default=1
        Verbosity level.
    n_jobs : int, optional, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    pd.DataFrame
        Dataframe of cross-validation scores.

    Notes
    -----
    This function returns a dataframe that is multi-indexed with the outer index
    representing a metric and the inner index representing the mean & (optionally) a
    standard deviation over validation splits. The columns are the estimators.
    """

    # check input types
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas dataframe.")
    if not isinstance(y, (pd.DataFrame, pd.Series)):
        raise TypeError("y must be a pandas dataframe or series.")
    if not isinstance(X.index, pd.MultiIndex):
        raise TypeError("X must be multi-indexed.")
    if not isinstance(y.index, pd.MultiIndex):
        raise TypeError("y must be multi-indexed.")
    if not isinstance(splitter, BasePanelSplit):
        raise TypeError("splitter must be an inherit from BasePanelSplit.")
    if not isinstance(estimators, dict):
        raise TypeError("estimators must be a dictionary.")
    if estimators == {}:
        raise ValueError("estimators must not be an empty dictionary.")
    if np.any([not isinstance(est_name, str) for est_name in estimators.keys()]):
        raise TypeError("estimator names must all be strings.")
    if np.any([not isinstance(est, object) for est in estimators.values()]):
        raise TypeError("estimators must all be objects.")
    if not isinstance(scoring, dict):
        raise TypeError("scoring must be a dictionary.")
    if scoring == {}:
        raise ValueError("scoring must not be an empty dictionary.")
    if np.any([not isinstance(metric_name, str) for metric_name in scoring.keys()]):
        raise TypeError("scorer names must all be strings.")

    if not isinstance(show_longbias, bool):
        raise TypeError("show_longbias must be a boolean.")
    if not isinstance(show_std, bool):
        raise TypeError("show_std must be a boolean.")
    if not isinstance(verbose, int):
        raise TypeError("verbose must be an integer.")
    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an integer.")

    # check the dataframes are in the right format
    if not isinstance(X.index.get_level_values(1)[0], datetime.date):
        raise TypeError("The inner index of X must be datetime.date.")
    if not isinstance(y.index.get_level_values(1)[0], datetime.date):
        raise TypeError("The inner index of y must be datetime.date.")
    if not X.index.equals(y.index):
        raise ValueError(
            "The indices of the input dataframe X and the output dataframe y don't match."
        )

    # check that there is at least one estimator and at least one scoring metric
    if len(estimators) <= 0:
        raise ValueError("There must be at least one estimator provided.")
    if len(scoring) <= 0:
        raise ValueError("There must be at least one scoring metric provided.")
    if verbose < 0:
        raise ValueError("verbose must be a non-negative integer.")
    if (n_jobs < 1) & (n_jobs != -1):
        raise ValueError("n_jobs must either be a positive integer or equal to -1.")

    # construct the dataframe to return
    if show_longbias:
        scoring = scoring.copy()
        scoring["Positive prediction ratio"] = make_scorer(
            lambda y_true, y_pred: np.sum(y_pred > 0) / len(y_pred)
        )
        scoring["Positive test-target ratio"] = make_scorer(
            lambda y_true, y_pred: np.sum(y_true > 0) / len(y_true)
        )

    results: dict = {
        estimator_name: {} for estimator_name in estimators
    }

    for estimator_name, estimator in estimators.items():
        if verbose:
            print(f"Calculating walk-forward validation metrics for {estimator_name}.")
        cv_results: dict = cross_validate(
            estimator,
            X,
            y,
            cv=splitter,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        for metric_name in scoring:
            score: np.ndarray = cv_results[f"test_{metric_name}"]
            results[estimator_name][metric_name] = np.mean(score)
            if show_std:
                results[estimator_name][f"{metric_name}_std"] = np.std(score)

    metrics_df = pd.DataFrame(results)
    if show_std:
        multi_index: pd.MultiIndex = pd.MultiIndex.from_tuples(
            [
                (key.split("_")[0], "std" if "std" in key else "mean")
                for key in metrics_df.index
            ]
        )
        metrics_df.index = multi_index

    return metrics_df


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm
    import macrosynergy.learning as msl

    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.metrics import (
        root_mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
    )

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example 1: Unbalanced panel """

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

    # 1) Demonstration of panel_cv_scores
    splitex = ExpandingKFoldPanelSplit(n_splits=100)
    models = {"OLS": LinearRegression(), "Lasso": Lasso(alpha=0.05)}
    metrics = {
        "neg_rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
        "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "neg_mape": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        "acc": make_scorer(msl.regression_accuracy),
        "bac": make_scorer(msl.regression_balanced_accuracy),
        "map": make_scorer(msl.panel_significance_probability),
        "sharpe": make_scorer(msl.sharpe_ratio),
        "sortino": make_scorer(msl.sortino_ratio),
    }
    df_ev = panel_cv_scores(
        X2,
        y2,
        splitter=splitex,
        estimators=models,
        scoring=metrics,
        show_longbias=True,
        show_std=False,
        n_jobs=-1,
        verbose=1,
    )
    print(df_ev)
