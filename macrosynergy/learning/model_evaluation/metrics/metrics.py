"""
Scikit-learn compatible performance metrics for model evaluation.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

from linearmodels.panel import RandomEffects

from sklearn.metrics import accuracy_score, balanced_accuracy_score

def regression_accuracy(
    y_true,
    y_pred,
    type = "panel",
):
    """
    Accuracy of signs between regression labels and predictions. 

    Parameters
    ----------

    y_true : pd.Series of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.
    type : str, default="panel"
        The panel dimension over which to compute the accuracy. Options are "panel", 
        "cross_section" and "time_periods".

    Returns
    -------

    accuracy : float
        The accuracy betweens signs of prediction-target pairs. 

    Notes
    -----
    Accuracy can be calculated over the whole panel, considering all samples irrespective
    of cross-section or time period. It can be beneficial, however, to compute accuracies
    for each cross-section and average them, or equivalently for each time period.
    
    When type = "cross_section", the reported accuracy is the mean accuray across
    cross-sections, reflecting the ability of a model to be effective across all
    cross-sections in the panel.
    
    When type = "time_periods", the reported accuracy is the mean accuracy across time
    periods, reflecting the ability of a model to be effective across all
    time periods/rebalancing dates in the panel.
    """
    if type == "panel":
        return accuracy_score(y_true < 0, y_pred < 0)
    elif type == "cross_section":
        accuracies = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = (y_true.index.get_level_values(0) == cross_section)
            accuracies.append(
                accuracy_score(
                    y_true.values[cross_section_mask] < 0,
                    y_pred[cross_section_mask] < 0
                )
            )
        return np.mean(accuracies)
    elif type == "time_periods":
        accuracies = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = (y_true.index.get_level_values(1) == time_period)
            accuracies.append(
                accuracy_score(
                    y_true.values[time_period_mask] < 0,
                    y_pred[time_period_mask] < 0
                )
            )
        return np.mean(accuracies)



def regression_balanced_accuracy(
    y_true,
    y_pred,
    type = "panel",
):
    """
    Balanced accuracy of signs between regression labels and predictions.

    Parameters
    ----------

    y_true : pd.Series of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.
    type : str, default="panel"
        The panel dimension over which to compute the balanced accuracy. Options are
        "panel", "cross_section" and "time_periods".

    Returns
    -------

    balanced_accuracy : float
        The balanced accuracy betweens signs of prediction-target pairs.

    Notes
    -----
    Balanced accuracy can be calculated over the whole panel, considering all samples
    irrespective of cross-section or time period. It can be beneficial, however, to
    compute balanced accuracies for each cross-section and average them, or equivalently
    for each time period.
    
    When type = "cross_section", the returned balanced accuracy score is
    the mean balanced accuracy across cross-sections, an empirical estimate of the expected
    out-of-sample balanced accuracy for a cross-section.
    
    When type = "time_periods", the returned balanced accuracy score is the mean balanced
    accuracy across time periods, an empirical estimate of the expected out-of-sample 
    balanced accuracy for a time period/rebalancing date.
    """
    if type == "panel":
        return balanced_accuracy_score(y_true < 0, y_pred < 0)
    elif type == "cross-section":
        balanced_accuracies = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = (y_true.index.get_level_values(0) == cross_section)
            balanced_accuracies.append(
                balanced_accuracy_score(
                    y_true.values[cross_section_mask] < 0,
                    y_pred[cross_section_mask] < 0
                )
            )
        return np.mean(balanced_accuracies)
    elif type == "time_periods":
        balanced_accuracies = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = (y_true.index.get_level_values(1) == time_period)
            balanced_accuracies.append(
                balanced_accuracy_score(
                    y_true.values[time_period_mask] < 0,
                    y_pred[time_period_mask] < 0
                )
            )
        return np.mean(balanced_accuracies)
    else:
        raise NotImplementedError("Invalid type. Options are 'panel', 'cross_section' and 'time_periods'.")
    
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

def sharpe_ratio(
    y_true,
    y_pred,
    binary = True,
    type = "panel",
):
    """
    Sharpe ratio of a strategy where the trader goes long when the predictions are positive
    and short when the predictions are negative. 

    Parameters
    ----------
    y_true : pd.Series of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.
    binary : bool, default=True
        Whether to consider only directional returns.
    type : str, default="panel"
        The panel dimension over which to compute the Sharpe ratio. Options are "panel",
        "cross_section" and "time_periods".

    Returns
    -------
    sharpe_ratio : float
        The Sharpe ratio of the strategy.

    Notes
    -----
    A Sharpe ratio can be calculated over the whole panel, considering the mean and standard
    deviation of the returns irrespective of cross-section or time period. It can be 
    beneficial, however to estimate the expected Sharpe for a cross-section or time period
    instead. 

    When type = "cross_section", the returned Sharpe ratio is the mean Sharpe ratio across
    cross-sections, an empirical estimate of the expected Sharpe ratio for a cross-section
    of interest. 

    When type = "time_periods", the returned Sharpe ratio is the mean Sharpe ratio across
    time periods, an empirical estimate of the expected Sharpe ratio for a time period of
    interest.
    """
    if binary:
        portfolio_returns = np.where(y_pred > 0, y_true, -y_true)
    else:
        raise NotImplementedError("Non-binary Sharpe ratio not yet implemented.")
    
    if type == "panel":
        average_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)

        if std_return == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = average_return / std_return

        return sharpe_ratio
    
    elif type == "cross_section":
        sharpe_ratios = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = (y_true.index.get_level_values(0) == cross_section)
            cross_section_returns = portfolio_returns[cross_section_mask]
            average_return = np.mean(cross_section_returns)
            std_return = np.std(cross_section_returns)

            if std_return == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = average_return / std_return

            sharpe_ratios.append(sharpe_ratio)
        
        return np.mean(sharpe_ratios)
    
    elif type == "time_periods":
        sharpe_ratios = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = (y_true.index.get_level_values(1) == time_period)
            time_period_returns = portfolio_returns[time_period_mask]
            average_return = np.mean(time_period_returns)
            std_return = np.std(time_period_returns)

            if std_return == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = average_return / std_return

            sharpe_ratios.append(sharpe_ratio)
        
        return np.mean(sharpe_ratios)
    
    else:
        raise NotImplementedError("Invalid type. Options are 'panel', 'cross_section' and 'time_periods'.")

def sortino_ratio(
    y_true,
    y_pred,
    binary = True,
    type = "panel",
):
    """
    Sortino ratio of a strategy where the trader goes long when the predictions are positive
    and short when the predictions are negative. 

    Parameters
    ----------
    y_true : pd.Series of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.
    binary : bool, default=True
        Whether to consider only directional returns.
    type : str, default="panel"
        The panel dimension over which to compute the Sortino ratio. Options are "panel",
        "cross_section" and "time_periods".

    Returns
    -------
    sortino_ratio : float
        The Sortino ratio of the strategy.

    Notes
    -----
    A Sortino ratio can be calculated over the whole panel, considering the mean and
    downside standard deviation of the returns irrespective of cross-section or time
    period. It can be beneficial, however, to estimate the expected Sortino for a
    cross-section or time period instead. 

    When type = "cross_section", the returned Sortino ratio is the mean Sortino ratio
    across cross-sections, an empirical estimate of the expected Sortino ratio for a
    cross-section of interest. 

    When type = "time_periods", the returned Sortino ratio is the mean Sortino ratio
    across time periods, an empirical estimate of the expected Sortino ratio for a time
    period of interest.
    """
    if binary:
        portfolio_returns = np.where(y_pred > 0, y_true, -y_true)
    else:
        raise NotImplementedError("Non-binary Sortino ratio not yet implemented.")

    if type == "panel":
        negative_returns = portfolio_returns[portfolio_returns < 0]
        average_return = np.mean(portfolio_returns)
        denominator = np.sqrt(np.mean(negative_returns**2))

        if denominator == 0:
            sortino_ratio = 0
        else:
            sortino_ratio = average_return / denominator

        return sortino_ratio
    elif type == "cross_section":
        sortino_ratios = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = (y_true.index.get_level_values(0) == cross_section)
            cross_section_returns = portfolio_returns[cross_section_mask]
            negative_returns = cross_section_returns[cross_section_returns < 0]
            average_return = np.mean(cross_section_returns)
            denominator = np.sqrt(np.mean(negative_returns**2))

            if denominator == 0:
                sortino_ratio = 0
            else:
                sortino_ratio = average_return / denominator

            sortino_ratios.append(sortino_ratio)
        
        return np.mean(sortino_ratios)
    elif type == "time_periods":
        sortino_ratios = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = (y_true.index.get_level_values(1) == time_period)
            time_period_returns = portfolio_returns[time_period_mask]
            negative_returns = time_period_returns[time_period_returns < 0]
            average_return = np.mean(time_period_returns)
            denominator = np.sqrt(np.mean(negative_returns**2))

            if denominator == 0:
                sortino_ratio = 0
            else:
                sortino_ratio = average_return / denominator

            sortino_ratios.append(sortino_ratio)
        
        return np.mean(sortino_ratios)
    else:
        raise NotImplementedError("Invalid type. Options are 'panel', 'cross_section' and 'time_periods'.")

def correlation_coefficient(
    y_true, 
    y_pred,
    correlation_type = "pearson",
    type = "panel",     
):
    """
    Correlation coefficient between true and predicted regression labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.
    correlation_type : str, default="pearson"
        The type of correlation coefficient to compute. Options are "pearson", "spearman"
        and "kendall".

    Returns
    -------
    correlation : float
        The correlation coefficient between true and predicted regression labels.

    Notes
    -----
    A correlation coefficient can be calculated over the whole panel, considering all
    samples irrespective of cross-section or time period. It can be beneficial, however,
    to estimate the expected correlation coefficient for a cross-section or time period 
    instead. 

    When type = "cross_section", the returned correlation coefficient is the mean
    correlation coefficient across cross-sections, an empirical estimate of the expected
    correlation coefficient for a cross-section of interest. 

    When type = "time_periods", the returned correlation coefficient is the mean correlation
    coefficient across time periods, an empirical estimate of the expected correlation
    coefficient for a time period of interest.
    """
    if type == "panel":
        if correlation_type == "pearson":
            correlation = stats.pearsonr(y_true, y_pred)[0]
        elif correlation_type == "spearman":
            correlation = stats.spearmanr(y_true, y_pred)[0]
        elif correlation_type == "kendall":
            correlation = stats.kendalltau(y_true, y_pred)[0]
        else:
            raise ValueError("Invalid correlation type. Options are 'pearson', 'spearman' and 'kendall'.")
        
        return correlation
    elif type == "cross_section":
        correlations = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = (y_true.index.get_level_values(0) == cross_section)
            if correlation_type == "pearson":
                correlation = stats.pearsonr(y_true.values[cross_section_mask], y_pred[cross_section_mask])[0]
            elif correlation_type == "spearman":
                correlation = stats.spearmanr(y_true.values[cross_section_mask], y_pred[cross_section_mask])[0]
            elif correlation_type == "kendall":
                correlation = stats.kendalltau(y_true.values[cross_section_mask], y_pred[cross_section_mask])[0]
            else:
                raise ValueError("Invalid correlation type. Options are 'pearson', 'spearman' and 'kendall'.")
            correlations.append(correlation)
        
        return np.mean(correlations)
    elif type == "time_periods":
        correlations = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = (y_true.index.get_level_values(1) == time_period)
            if correlation_type == "pearson":
                correlation = stats.pearsonr(y_true.values[time_period_mask], y_pred[time_period_mask])[0]
            elif correlation_type == "spearman":
                correlation = stats.spearmanr(y_true.values[time_period_mask], y_pred[time_period_mask])[0]
            elif correlation_type == "kendall":
                correlation = stats.kendalltau(y_true.values[time_period_mask], y_pred[time_period_mask])[0]
            else:
                raise ValueError("Invalid correlation type. Options are 'pearson', 'spearman' and 'kendall'.")
            correlations.append(correlation)
        return np.mean(correlations)