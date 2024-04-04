"""
Collection of non-standard scikit-learn performance metrics for evaluation of
machine learning model predictions.
"""

import numpy as np
import pandas as pd
import datetime

from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.tools import add_constant

from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from macrosynergy.learning.panel_time_series_split import ExpandingKFoldPanelSplit

from typing import Union


def panel_significance_probability(
    y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """
    Function to create a linear mixed effects model between the ground truth returns and
    the predicted returns, returning the significance of the model slope.
    Period-specific random effects are included in the model to account for
    return cross-sectional correlations.
    This can be passed into sklearn's make_scorer function to be used as a scorer in a
    grid search or cross validation procedure.

    :param <pd.Series> y_true: Pandas series of ground truth labels. These must be
        multi-indexed by cross-section and date. The dates must be in datetime format.
    :param <Union[pd.Series,np.ndarray]> y_pred: Either a pandas series or numpy array
        of predicted targets. This must have the same length as y_true.

    :return <float> significance_prob: 1 - p-value of the regression slope parameter,
        given by the linear mixed effects model.
    """

    # checks
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas series")

    if not isinstance(y_true.index, pd.MultiIndex):
        raise ValueError("y_true must be multi-indexed.")

    if not isinstance(y_true.index.get_level_values(1)[0], datetime.date):
        raise TypeError("The inner index of y must be datetime.date.")

    if not (len(y_true) == len(y_pred)):
        raise ValueError("y_true and y_pred must have the same length.")

    if np.all(y_true == 0):
        # Sklearn averages each metric over the CV splits.
        # If all the ground truth labels are zero, the regression is invalid due to a
        # singular matrix. Hence, we return zero in this case.
        significance_prob = 0
        return significance_prob

    # regress ground truth against predictions
    X = add_constant(y_pred)
    groups = y_true.index.get_level_values(1)

    # fit model
    re = MixedLM(y_true, X, groups=groups).fit(reml=False)
    pval = re.pvalues.iloc[1]

    return 1 - pval


def regression_accuracy(
    y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """
    Function to return the accuracy between the signs of the predictions and targets.

    :param <pd.Series> y_true: Pandas series of ground truth labels. These must be
        multi-indexed by cross-section and date. The dates must be in datetime format.
    :param <Union[pd.Series,np.ndarray]> y_pred: Either a pandas series or numpy array
        of predicted targets. This must have the same length as y_true.

    :return <float>: Accuracy between the signs of the predictions and targets.
    """

    # checks
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas series")

    if not isinstance(y_true.index, pd.MultiIndex):
        raise ValueError("y_true must be multi-indexed.")

    if not isinstance(y_true.index.get_level_values(1)[0], datetime.date):
        raise TypeError("The inner index of y must be datetime.date.")

    if not (len(y_true) == len(y_pred)):
        raise ValueError("y_true and y_pred must have the same length.")

    return accuracy_score(y_true < 0, y_pred < 0)


def regression_balanced_accuracy(
    y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray]
) -> float:
    """
    Function to return the balanced accuracy between the signs
    of the predictions and targets.

    :param <pd.Series> y_true: Pandas series of ground truth labels. These must be
        multi-indexed by cross-section and date. The dates must be in datetime format.
    :param <Union[pd.Series,np.ndarray]> y_pred: Either a pandas series or numpy array
        of predicted targets. This must have the same length as y_true.

    :return <float>: Balanced accuracy between the signs of the predictions and targets.
    """

    # checks
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas series")

    if not isinstance(y_true.index, pd.MultiIndex):
        raise ValueError("y_true must be multi-indexed.")

    if not isinstance(y_true.index.get_level_values(1)[0], datetime.date):
        raise TypeError("The inner index of y must be datetime.date.")

    if not (len(y_true) == len(y_pred)):
        raise ValueError("y_true and y_pred must have the same length.")

    return balanced_accuracy_score(y_true < 0, y_pred < 0)


def sharpe_ratio(y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Function to return a Sharpe ratio for a strategy where we go long if the predictions
    are positive and short if the predictions are negative.

    :param <pd.Series> y_true: Pandas series of ground truth labels. These must be
        multi-indexed by cross-section and date. The dates must be in datetime format.
    :param <Union[pd.Series,np.ndarray]> y_pred: Either a pandas series or numpy array
        of predicted targets. This must have the same length as y_true.

    :return <float>: Sharpe ratio for the binary strategy.
    """

    # checks
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas series")

    if not isinstance(y_true.index, pd.MultiIndex):
        raise ValueError("y_true must be multi-indexed.")

    if not isinstance(y_true.index.get_level_values(1)[0], datetime.date):
        raise TypeError("The inner index of y must be datetime.date.")

    if not (len(y_true) == len(y_pred)):
        raise ValueError("y_true and y_pred must have the same length.")

    portfolio_returns = np.where(y_pred > 0, y_true, -y_true)
    average_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)

    if std_return == 0:
        # Sklearn averages each metric over the CV splits.
        # If the standard deviation is zero, the portfolio returns are constant.
        # So we return zero to ignore this split in the average.
        sharpe_ratio = 0
    else:
        sharpe_ratio = average_return / std_return

    return sharpe_ratio


def sortino_ratio(y_true: pd.Series, y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Function to return a Sortino ratio for a strategy where we go long if the predictions
    are positive and short if the predictions are negative.

    :param <pd.Series> y_true: Pandas series of ground truth labels. These must be
        multi-indexed by cross-section and date. The dates must be in datetime format.
    :param <Union[pd.Series,np.ndarray]> y_pred: Either a pandas series or numpy array
        of predicted targets. This must have the same length as y_true.

    :return <float>: Sortino ratio for the binary strategy.
    """

    # checks
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas series")

    if not isinstance(y_true.index, pd.MultiIndex):
        raise ValueError("y_true must be multi-indexed.")

    if not isinstance(y_true.index.get_level_values(1)[0], datetime.date):
        raise TypeError("The inner index of y must be datetime.date.")

    if not (len(y_true) == len(y_pred)):
        raise ValueError("y_true and y_pred must have the same length.")

    portfolio_returns = np.where(y_pred > 0, y_true, -y_true)
    negative_returns = portfolio_returns[portfolio_returns < 0]
    average_return = np.mean(portfolio_returns)
    denominator = np.sqrt(np.mean(negative_returns**2))

    if denominator == 0:
        # Sklearn averages each metric over the CV splits.
        # If the denominator is zero, the sortino ratio over this period is invalid.
        # So we return zero to ignore this split in the average.
        sortino_ratio = 0
    else:
        sortino_ratio = average_return / denominator

    return sortino_ratio


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CPI", "GROWTH", "RIR"]

    df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31"]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31"]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31"]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31"]

    tuples = []

    for cid in cids:
        # get list of all elidgible dates
        sdate = df_cids.loc[cid]["earliest"]
        edate = df_cids.loc[cid]["latest"]
        all_days = pd.date_range(sdate, edate)
        work_days = all_days[all_days.weekday < 5]
        for work_day in work_days:
            tuples.append((cid, work_day))

    n_samples = len(tuples)
    ftrs = np.random.normal(loc=0, scale=1, size=(n_samples, 3))
    labels = np.random.normal(loc=0, scale=1, size=n_samples)
    df = pd.DataFrame(
        data=np.concatenate((np.reshape(labels, (-1, 1)), ftrs), axis=1),
        index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
        columns=xcats,
        dtype=np.float32,
    )

    X = df.drop(columns="XR")
    y = df["XR"]

    splitter = ExpandingKFoldPanelSplit(n_splits=4)
    scorer1 = make_scorer(panel_significance_probability, greater_is_better=True)
    scorer2 = make_scorer(sharpe_ratio, greater_is_better=True)
    scorer3 = make_scorer(sortino_ratio, greater_is_better=True)
    scorer4 = make_scorer(regression_accuracy, greater_is_better=True)
    scorer5 = make_scorer(regression_balanced_accuracy, greater_is_better=True)
    cv_results1 = cross_val_score(
        LinearRegression(), X, y, cv=splitter, scoring=scorer1
    )
    cv_results2 = cross_val_score(
        LinearRegression(), X, y, cv=splitter, scoring=scorer2
    )
    cv_results3 = cross_val_score(
        LinearRegression(), X, y, cv=splitter, scoring=scorer3
    )
    cv_results4 = cross_val_score(
        LinearRegression(), X, y, cv=splitter, scoring=scorer4
    )
    cv_results5 = cross_val_score(
        LinearRegression(), X, y, cv=splitter, scoring=scorer5
    )
    print("Probabilities of significances, per split:", cv_results1)
    print("Sharpe ratios, per split:", cv_results2)
    print("Sortino ratios, per split:", cv_results3)
    print("Regression accuracies, per split:", cv_results4)
    print("Regression balanced accuracies, per split:", cv_results5)
    print("Done")
