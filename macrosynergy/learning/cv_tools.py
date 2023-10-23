"""
A set of tools for cross-validation of panel data.

**NOTE: This module is under development, and is not yet ready for production use.**
"""

import numpy as np
import pandas as pd 
import datetime
from typing import Union, Optional

from macrosynergy.learning import PanelTimeSeriesSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

def panel_cv_scores(
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    splitter: PanelTimeSeriesSplit,
    estimators: dict,
    scoring: dict,
    verbose: Optional[int] = 0,
    show_longbias: Optional[bool] = True,
    show_std: Optional[bool] = False,
):
    """
    Returns a dataframe of cross-validation scores

    :param <pd.DataFrame> X: Dataframe of features multi-indexed by (cross-section, date).
        The dataframe must be in wide format: each feature is a column.  The dates must
        be in datetime format.
    :param <pd.DataFrame> y: Dataframe of the target variable, multi-indexed by
        (cross-section, date). The dates must be in datetime format.
    :param <PanelTimeSeriesSplit> splitter: splitter object instantiated from
        PanelTimeSeriesSplit.
    :param <dict> estimators: dictionary of estimators, where the keys are the estimator
        names and the values are the sklearn estimator objects.
    :param <dict> scoring: dictionary of scoring metrics, where the keys are the metric
        names and the values are callables
    :param <int> verbose: integer specifying verbosity of the cross-validation process.
        Default is 0.
    :param <bool> show_longbias: boolean specifying whether or not to display the
        proportion of positive returns. Default is True.
    :param <bool> show_std: boolean specifying whether or not to show the standard
        deviation of the cross-validation scores. Default is False.
        
    :return <pd.DataFrame> metrics_df: dataframe comprising means & standard deviations of
        cross-validation metrics for each sklearn estimator, over the walk-forward history.

    N.B.: The performance metrics dataframe returned is multi-indexed with the outer index
    representing a metric and the inner index representing the mean & standard deviation
    of the metric over the walk-forward validation splits. The columns are the estimators.
    """

    # check input types
    assert isinstance(X, pd.DataFrame), "X must be a pandas dataframe."

    assert isinstance(y, (pd.DataFrame, pd.Series)), "y must be a pandas dataframe or series."
    assert isinstance(X.index, pd.MultiIndex), "X must be multi-indexed."
    assert isinstance(y.index, pd.MultiIndex), "y must be multi-indexed."
    assert isinstance(
        splitter, PanelTimeSeriesSplit
    ), "splitter must be an instance of PanelTimeSeriesSplit."
    assert isinstance(estimators, dict), "estimators must be a dictionary."
    assert isinstance(scoring, dict), "scoring must be a dictionary."
    assert isinstance(verbose, int), "verbose must be an integer."

    # check the dataframes are in the right format
    assert isinstance(
        X.index.get_level_values(1)[0], datetime.date
    ), "The inner index of X must be datetime.date."
    assert isinstance(
        y.index.get_level_values(1)[0], datetime.date
    ), "The inner index of y must be datetime.date."
    assert X.index.equals(
        y.index
    ), "The indices of the input dataframe X and the output dataframe y don't match."

    # check that there is at least one estimator and at least one scoring metric
    assert len(estimators) > 0, "There must be at least one estimator provided."
    assert len(scoring) > 0, "There must be at least one scoring metric provided."
    assert verbose >= 0, "verbose must be a non-negative integer."

    # construct the dataframe to return
    if show_longbias:
        scoring["Positive prediction ratio"] = make_scorer(lambda y_true, y_pred: np.sum(y_pred > 0)/len(y_pred))
        scoring["Positive test-target ratio"] = make_scorer(lambda y_true, y_pred: np.sum(y_true > 0)/len(y_true))

    estimator_names = list(estimators.keys())
    metric_names = list(scoring.keys())

    metrics_df = pd.DataFrame(
        columns=estimator_names,
        index=pd.MultiIndex.from_product([metric_names, ["mean", "std"]]),
    ) if show_std else pd.DataFrame(
        columns=estimator_names, index=metric_names)

    for estimator_name, estimator in estimators.items():
        if verbose != 0:
            print(f"Calculating walk-forward validation metrics for {estimator_name}.")
        cv_results = cross_validate(
            estimator, X, y, cv=splitter, scoring=scoring, verbose=verbose
        )
        for metric_name in metric_names:
            if show_std:
                metrics_df.loc[(metric_name, "mean"), estimator_name] = np.mean(
                    cv_results[f"test_{metric_name}"]
                )
                metrics_df.loc[(metric_name, "std"), estimator_name] = np.std(
                    cv_results[f"test_{metric_name}"]
                )
            else:
                metrics_df.loc[metric_name, estimator_name] = np.mean(
                    cv_results[f"test_{metric_name}"]
                )

    return metrics_df

if __name__ == "__main__":
    from macrosynergy.management.simulate_quantamental_data import make_qdf
    import macrosynergy.management as msm

    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.metrics import mean_squared_error

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

    splitex = PanelTimeSeriesSplit(n_splits=4, n_split_method="expanding")
    df_ev = panel_cv_scores(
        X2,
        y2,
        splitter=splitex,
        estimators={"OLS": LinearRegression(), "Lasso": Lasso()},
        scoring={"rmse": make_scorer(mean_squared_error)},
        show_longbias=True,
        show_std=True
    )
    print(df_ev)