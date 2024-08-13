"""
Scikit-learn compatible performance metrics for model evaluation.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import datetime

from linearmodels.panel import RandomEffects

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from typing import Union

# Regression accuracy 

def regression_accuracy(
    y_true,
    y_pred,
):
    """
    Accuracy of signs between regression labels and predictions. 

    Parameters
    ----------

    y_true : array-like of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.

    Returns
    -------

    accuracy : float
        The accuracy betweens signs of prediction-target pairs. 
    """

    return accuracy_score(y_true < 0, y_pred < 0)

# Regression balanced accuracy

def regression_balanced_accuracy(
    y_true,
    y_pred,        
):
    """
    Balanced accuracy of signs between regression labels and predictions.

    Parameters
    ----------

    y_true : array-like of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.

    Returns
    -------

    balanced_accuracy : float
        The balanced accuracy betweens signs of prediction-target pairs.
    """
    
    return balanced_accuracy_score(y_true < 0, y_pred < 0)

# Macrosynergy panel test significance

def panel_significance_probability(
    y_true,
    y_pred,
):
    """
    :math:`1 - pval` using the Macrosynergy panel (MAP) test for significance of
    correlation accounting for cross-sectional correlations. 

    Parameters
    ----------

    y_true : pd.Series of shape (n_samples,)
        True regression labels.

    y_pred : pd.Series of shape (n_samples,)
        Predicted regression labels.

    Returns
    -------

    prob_significance : float
        The probability of significance of the relation between predictions and targets. 

    Notes
    -----

    The (Ma)crosynergy (p)anel (MAP) test is a hypothesis test for the significance of 
    a relation between two variables accounting for cross-sectional correlations. A 
    period-specific random effects model is estimated, with a Wald test performed on the
    concerned coefficient. Since the test requires a panel structure, the inputs are
    required to be pd.Series, multi-indexed by cross-section and real date. 
    """
    # Convert cross-section labels to integer codes 
    unique_cross_sections = sorted(y_true.index.get_level_values(0).unique())
    cross_section_codes = dict(zip(unique_cross_sections, range(1, len(unique_cross_sections) + 1)))
    y_true = y_true.rename(cross_section_codes, level=0, inplace=False).copy()
    y_pred = y_pred.rename(cross_section_codes, level=0, inplace=False).copy()

    # Create random effects model
    re = RandomEffects(y_true.swaplevel(), y_pred.swaplevel()).fit()
    coef = re.params[y_pred.name]
    zstat = coef / re.std_errors[y_pred.name]
    pval = 2 * (1 - stats.norm.cdf(zstat))
    
    return 1 - pval

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