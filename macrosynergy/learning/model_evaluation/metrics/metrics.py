"""
Scikit-learn compatible performance metrics for model evaluation.
"""

import inspect
import numpy as np
import pandas as pd
import scipy.stats as stats

from macrosynergy.learning.random_effects import RandomEffects

from sklearn.metrics import accuracy_score, balanced_accuracy_score


def create_panel_metric(
    y_true,
    y_pred,
    sklearn_metric,
    type="panel",
):
    """
    Evaluation with a scikit-learn metric, respecting the panel structure.

    Parameters
    ----------
    y_true : pd.Series of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.
    sklearn_metric : callable
        A scikit-learn metric function. This function must accept two arguments, `y_true`
        and `y_pred`, which have the same meaning as the arguments passed to this function.
    type : str, default="panel"
        The panel dimension over which to compute the metric. Options are "panel",
        "cross_section" and "time_periods".

    Returns
    -------
    metric : float
        The computed metric.

    Notes
    -----
    This function is a wrapper around a scikit-learn metric, allowing it to be
    evaluated over different panel axes. For instance, the :math:`R^2` metric can be
    evaluated over the whole panel, across cross-sections or across time periods. Instead
    of re-implementing every scikit-learn metric so that evaluation over panel axes is
    possible, the `create_panel_metric` function allows any scikit-learn metric to be
    evaluated over different panel axes.
    """
    # Checks
    _check_metric_params(y_true, y_pred, type)

    if not callable(sklearn_metric):
        raise TypeError("sklearn_metric must be a callable")
    # check sklearn_metric has y_true and y_pred as arguments
    metric_args = inspect.signature(sklearn_metric).parameters
    if not all(arg in metric_args for arg in ["y_true", "y_pred"]):
        raise ValueError("sklearn_metric must accept y_true and y_pred as arguments")

    if type == "panel":
        return sklearn_metric(y_true, y_pred)
    elif type == "cross_section":
        metrics = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = y_true.index.get_level_values(0) == cross_section
            metrics.append(
                sklearn_metric(
                    y_true.values[cross_section_mask], y_pred[cross_section_mask]
                )
            )
        return np.mean(metrics)
    elif type == "time_periods":
        metrics = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = y_true.index.get_level_values(1) == time_period
            metrics.append(
                sklearn_metric(
                    y_true.values[time_period_mask], y_pred[time_period_mask]
                )
            )
        return np.mean(metrics)


def regression_accuracy(
    y_true,
    y_pred,
    type="panel",
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
    of cross-section or time period. It can be beneficial, however, to estimate the
    expected accuracy for a cross-section or time period instead.

    When type = "cross_section", the returned accuracy is the mean accuracy across
    cross-sections, an empirical estimate of the expected accuracy for a cross-section
    of interest.

    When type = "time_periods", the returned accuracy is the mean accuracy across time
    periods, an empirical estimate of the expected accuracy for a time period of interest.
    """
    # Checks
    _check_metric_params(y_true, y_pred, type)

    # Compute accuracy
    if type == "panel":
        return accuracy_score(y_true < 0, y_pred < 0)

    elif type == "cross_section":
        accuracies = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = y_true.index.get_level_values(0) == cross_section
            accuracies.append(
                accuracy_score(
                    y_true.values[cross_section_mask] < 0,
                    y_pred[cross_section_mask] < 0,
                )
            )
        return np.mean(accuracies)

    elif type == "time_periods":
        accuracies = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = y_true.index.get_level_values(1) == time_period
            accuracies.append(
                accuracy_score(
                    y_true.values[time_period_mask] < 0, y_pred[time_period_mask] < 0
                )
            )
        return np.mean(accuracies)


def regression_balanced_accuracy(
    y_true,
    y_pred,
    type="panel",
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
    estimate expected balanced accuracy for a cross-section or time period instead.

    When type = "cross_section", the returned balanced accuracy score is
    the mean balanced accuracy across cross-sections, an empirical estimate of the expected
    balanced accuracy for a cross-section of interest.

    When type = "time_periods", the returned balanced accuracy score is the mean balanced
    accuracy across time periods, an empirical estimate of the expected
    balanced accuracy for a time period of interest.
    """
    # Checks
    _check_metric_params(y_true, y_pred, type)

    # Compute balanced accuracy
    if type == "panel":
        return balanced_accuracy_score(y_true < 0, y_pred < 0)

    elif type == "cross_section":
        balanced_accuracies = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = y_true.index.get_level_values(0) == cross_section
            balanced_accuracies.append(
                balanced_accuracy_score(
                    y_true.values[cross_section_mask] < 0,
                    y_pred[cross_section_mask] < 0,
                )
            )
        return np.mean(balanced_accuracies)

    elif type == "time_periods":
        balanced_accuracies = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = y_true.index.get_level_values(1) == time_period
            balanced_accuracies.append(
                balanced_accuracy_score(
                    y_true.values[time_period_mask] < 0, y_pred[time_period_mask] < 0
                )
            )
        return np.mean(balanced_accuracies)


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
    # Checks
    # y_true
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pd.Series")
    if not y_true.index.nlevels == 2:
        raise ValueError("y_true must be multi-indexed with two levels")
    if not y_true.index.get_level_values(0).dtype == "object":
        raise ValueError("y_true outer index must be strings")
    if not y_true.index.get_level_values(1).dtype == "datetime64[ns]":
        raise ValueError("y_true inner index must be datetime")

    # y_pred
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("y_pred must be either pd.Series or np.ndarray")
    if isinstance(y_pred, np.ndarray):
        if y_pred.ndim != 1:
            raise ValueError(
                "y_pred must be 1-dimensional for 'panel_significance_probability'"
            )
        y_pred = pd.Series(y_pred, index=y_true.index)
    if not y_pred.index.nlevels == 2:
        raise ValueError("y_pred must be multi-indexed with two levels")
    if not y_pred.index.get_level_values(0).dtype == "object":
        raise ValueError("y_pred outer index must be strings")
    if not y_pred.index.get_level_values(1).dtype == "datetime64[ns]":
        raise ValueError("y_pred inner index must be datetime")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred must have the same index")
    if y_pred.isnull().values.any():
        return 0

    re = RandomEffects(fit_intercept=False).fit(y_true, y_pred)
    pval = re.pvals.to_numpy().item()

    return 1 - pval


def sharpe_ratio(
    y_true,
    y_pred,
    binary=True,
    thresh=None,
    type="panel",
):
    """
    Sharpe ratio of a strategy where the trader goes long by a single unit when the
    predictions are positive and short by a single unit when the predictions are negative.

    Parameters
    ----------
    y_true : pd.Series of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.
    binary : bool, default=True
        Whether to consider only directional returns. If True, the portfolio returns only
        consider the sign of the predictions. If False, naive portfolio weights are
        determined. See Notes for more information on their calculation.
    thresh : float, default=None
        The threshold for portfolio weights in the case where binary = False.
    type : str, default="panel"
        The panel dimension over which to compute the Sharpe ratio. Options are "panel",
        "cross_section" and "time_periods".

    Returns
    -------
    sharpe_ratio : float
        The Sharpe ratio of the strategy.

    Notes
    -----
    A Sharpe ratio can be calculated over the whole panel, considering the mean and
    standard deviation of the returns irrespective of cross-section or time period. It can
    be beneficial, however, to estimate the expected Sharpe for a cross-section or time
    period instead.

    When type = "cross_section", the returned Sharpe ratio is the mean Sharpe ratio across
    cross-sections, an empirical estimate of the expected Sharpe ratio for a cross-section
    of interest.

    When type = "time_periods", the returned Sharpe ratio is the mean Sharpe ratio across
    time periods, an empirical estimate of the expected Sharpe ratio for a time period of
    interest.

    This metric can calculate the Sharpe ratio of either binary or non-binary strategies.
    When binary = False, predictions are normalized by their standard deviation in each
    time period. If thresh is not None, the resulting weights are clipped to the range
    [-thresh, thresh]. The resulting portfolio returns are the product of these derived
    weights and the true returns.
    """
    # Checks
    _check_metric_params(y_true, y_pred, type)

    if not isinstance(binary, bool):
        raise TypeError("binary must be a boolean")

    if not isinstance(thresh, (int, float)) and thresh is not None:
        raise TypeError("thresh must be an integer or float")
    if thresh is not None and thresh <= 0:
        raise ValueError("thresh must be positive")

    # Compute Sharpe ratio
    if binary:
        portfolio_returns = np.where(y_pred > 0, y_true, -y_true)
    else:
        # Massage y_pred into portfolio weights
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred, index=y_true.index)
        if thresh is not None:
            portfolio_weights = (
                y_pred / y_pred.groupby(level=1).transform("std")
            ).clip(lower=-thresh, upper=thresh)
        else:
            portfolio_weights = y_pred / y_pred.groupby(level=1).transform("std")
        portfolio_returns = portfolio_weights * y_true

    if type == "panel":
        average_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)

        if std_return == 0:
            std_return = 1

        sharpe_ratio = average_return / std_return

        return sharpe_ratio

    elif type == "cross_section":
        sharpe_ratios = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = y_true.index.get_level_values(0) == cross_section
            cross_section_returns = portfolio_returns[cross_section_mask]
            average_return = np.mean(cross_section_returns)
            std_return = np.std(cross_section_returns)

            if std_return == 0:
                std_return = 1

            sharpe_ratio = average_return / std_return

            sharpe_ratios.append(sharpe_ratio)

        return np.mean(sharpe_ratios)

    elif type == "time_periods":
        sharpe_ratios = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = y_true.index.get_level_values(1) == time_period
            time_period_returns = portfolio_returns[time_period_mask]
            average_return = np.mean(time_period_returns)
            std_return = np.std(time_period_returns)

            if std_return == 0:
                std_return = 1

            sharpe_ratio = average_return / std_return

            sharpe_ratios.append(sharpe_ratio)

        return np.mean(sharpe_ratios)


def sortino_ratio(
    y_true,
    y_pred,
    binary=True,
    thresh=None,
    type="panel",
):
    """
    Sortino ratio of a strategy where the trader goes long when the predictions are
    positive by a single unit and short by a single unit when the predictions are negative.

    Parameters
    ----------
    y_true : pd.Series of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.
    binary : bool, default=True
        Whether to consider only directional returns. If True, the portfolio returns only
        consider the sign of the predictions. If False, naive portfolio weights are
        determined. See Notes for more information on their calculation.
    thresh : float, default=None
        The threshold for portfolio weights in the case where binary = False.
    type : str, default="panel"
        The panel dimension over which to compute the Sharpe ratio. Options are "panel",
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

    This metric can calculate the Sortino ratio of either binary or non-binary strategies.
    When binary = False, predictions are normalized by their standard deviation in each
    time period. If thresh is not None, the resulting weights are clipped to the range
    [-thresh, thresh]. The resulting portfolio returns are the product of these derived
    weights and the true returns.
    """
    # Checks
    _check_metric_params(y_true, y_pred, type)

    if not isinstance(binary, bool):
        raise TypeError("binary must be a boolean")

    if not isinstance(thresh, (int, float)) and thresh is not None:
        raise TypeError("thresh must be an integer or float")
    if thresh is not None and thresh <= 0:
        raise ValueError("thresh must be positive")

    # Compute Sortino ratio
    if binary:
        portfolio_returns = np.where(y_pred > 0, y_true, -y_true)
    else:
        # Massage y_pred into portfolio weights
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred, index=y_true.index)
        if thresh is not None:
            portfolio_weights = (
                y_pred / y_pred.groupby(level=1).transform("std")
            ).clip(lower=-thresh, upper=thresh)
        else:
            portfolio_weights = y_pred / y_pred.groupby(level=1).transform("std")
        portfolio_returns = portfolio_weights * y_true

    if type == "panel":
        negative_returns = portfolio_returns[portfolio_returns < 0]
        average_return = np.mean(portfolio_returns)
        denominator = np.std(negative_returns)

        if denominator == 0 or np.isnan(denominator):
            denominator = 1

        sortino_ratio = average_return / denominator

        return sortino_ratio

    elif type == "cross_section":
        sortino_ratios = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = y_true.index.get_level_values(0) == cross_section
            cross_section_returns = portfolio_returns[cross_section_mask]
            negative_returns = cross_section_returns[cross_section_returns < 0]
            average_return = np.mean(cross_section_returns)
            denominator = np.std(negative_returns)

            if denominator == 0 or np.isnan(denominator):
                denominator = 1

            sortino_ratio = average_return / denominator

            sortino_ratios.append(sortino_ratio)

        return np.mean(sortino_ratios)

    elif type == "time_periods":
        sortino_ratios = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = y_true.index.get_level_values(1) == time_period
            time_period_returns = portfolio_returns[time_period_mask]
            negative_returns = time_period_returns[time_period_returns < 0]
            average_return = np.mean(time_period_returns)
            denominator = np.std(negative_returns)

            if denominator == 0 or np.isnan(denominator):
                denominator = 1

            sortino_ratio = average_return / denominator

            sortino_ratios.append(sortino_ratio)

        return np.mean(sortino_ratios)


def correlation_coefficient(
    y_true,
    y_pred,
    correlation_type="pearson",
    type="panel",
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
    # Checks
    _check_metric_params(y_true, y_pred, type)

    if not isinstance(correlation_type, str):
        raise TypeError("correlation_type must be a string")
    if correlation_type not in ["pearson", "spearman", "kendall"]:
        raise ValueError(
            "Invalid correlation type. Options are 'pearson', 'spearman' and 'kendall'."
        )

    # Compute correlation coefficient
    if type == "panel":
        if correlation_type == "pearson":
            correlation = stats.pearsonr(y_true, y_pred)[0]
        elif correlation_type == "spearman":
            correlation = stats.spearmanr(y_true, y_pred)[0]
        elif correlation_type == "kendall":
            correlation = stats.kendalltau(y_true, y_pred)[0]

        return correlation

    elif type == "cross_section":
        correlations = []
        unique_cross_sections = y_true.index.get_level_values(0).unique()
        for cross_section in unique_cross_sections:
            cross_section_mask = y_true.index.get_level_values(0) == cross_section
            if correlation_type == "pearson":
                correlation = stats.pearsonr(
                    y_true.values[cross_section_mask], y_pred[cross_section_mask]
                )[0]
            elif correlation_type == "spearman":
                correlation = stats.spearmanr(
                    y_true.values[cross_section_mask], y_pred[cross_section_mask]
                )[0]
            elif correlation_type == "kendall":
                correlation = stats.kendalltau(
                    y_true.values[cross_section_mask], y_pred[cross_section_mask]
                )[0]

            correlations.append(correlation)

        return np.mean(correlations)

    elif type == "time_periods":
        correlations = []
        unique_time_periods = y_true.index.get_level_values(1).unique()
        for time_period in unique_time_periods:
            time_period_mask = y_true.index.get_level_values(1) == time_period
            if correlation_type == "pearson":
                correlation = stats.pearsonr(
                    y_true.values[time_period_mask], y_pred[time_period_mask]
                )[0]
            elif correlation_type == "spearman":
                correlation = stats.spearmanr(
                    y_true.values[time_period_mask], y_pred[time_period_mask]
                )[0]
            elif correlation_type == "kendall":
                correlation = stats.kendalltau(
                    y_true.values[time_period_mask], y_pred[time_period_mask]
                )[0]

            correlations.append(correlation)

        return np.mean(correlations)


def _check_metric_params(
    y_true,
    y_pred,
    type,
):
    """
    Generic input validation for model evaluation metrics.

    Parameters
    ----------
    y_true : pd.Series of shape (n_samples,)
        True regression labels.
    y_pred : array-like of shape (n_samples,)
        Predicted regression labels.
    type : str
        The panel dimension over which to compute the metric. Options are "panel",
        "cross_section" and "time_periods".
    """
    # y_true
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pd.Series")
    if not y_true.index.nlevels == 2:
        raise ValueError("y_true must be multi-indexed with two levels")
    if not y_true.index.get_level_values(0).dtype == "object":
        raise ValueError("y_true outer index must be strings")
    if not y_true.index.get_level_values(1).dtype == "datetime64[ns]":
        raise ValueError("y_true inner index must be datetime")

    # y_pred
    if not isinstance(y_pred, (np.ndarray, pd.Series)):
        raise TypeError("y_pred must be either a numpy array or pandas series")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # type
    if not isinstance(type, str):
        raise TypeError("type must be a string")
    if type not in ["panel", "cross_section", "time_periods"]:
        raise ValueError(
            "Invalid type. Options are 'panel', 'cross_section' and 'time_periods'"
        )


if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf
    from sklearn.linear_model import LinearRegression

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "BMXR"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2012-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2012-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["BMXR"] = ["2012-01-01", "2020-12-31", 1, 2, 0.95, 1]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    Xy = msm.categories_df(
        df=dfd, xcats=xcats, cids=cids, freq="M", lag=1, xcat_aggs=["last", "sum"]
    ).dropna()
    X = Xy.iloc[:, :-1]
    y = Xy.iloc[:, -1]

    lr = LinearRegression()
    lr.fit(X, y)

    # Accuracies
    print(
        "\nAccuracy over panel: "
        f"{regression_accuracy(y, lr.predict(X), type='panel')}"
    )
    print(
        "Cross-sectional accuracy: "
        f"{regression_accuracy(y, lr.predict(X), type='cross_section')}"
    )
    print(
        "Periodic accuracy: "
        f"{regression_accuracy(y, lr.predict(X), type='time_periods')}"
    )

    # Un-annualized Sharpe ratios
    print(
        "\nUn-annualized Sharpe over panel: "
        f"{sharpe_ratio(y, lr.predict(X), type='panel')}"
    )
    print(
        "Cross-sectional un-annualized Sharpe: "
        f"{sharpe_ratio(y, lr.predict(X), type='cross_section')}"
    )
    print(
        "Periodic un-annualized Sharpe: "
        f"{sharpe_ratio(y, lr.predict(X), type='time_periods')}"
    )

    # Kendall correlation
    print(
        "\nKendall correlation over panel: "
        f"{correlation_coefficient(y, lr.predict(X), correlation_type='kendall', type='panel')}"
    )
    print(
        "Cross-sectional Kendall correlation: "
        f"{correlation_coefficient(y, lr.predict(X), correlation_type='kendall', type='cross_section')}"
    )
    print(
        "Periodic Kendall correlation: "
        f"{correlation_coefficient(y, lr.predict(X), correlation_type='kendall', type='time_periods')}"
    )
